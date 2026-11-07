#!/usr/bin/env python3
"""Audit the saved 256/1k comparison without retraining either recommender."""

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import duckdb
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from compare_sasrec import item_mapping, paired_intervals
from run_sasrec import model_args, write
from sasrec import SASRec, examples, metrics, most_popular, ranks


def aligned_examples(data, reference, split, length):
    """Keep reference targets/users before constructing histories in another vocabulary."""
    manifest = json.loads((data / "manifest.json").read_text())
    names = [p["split"] for p in manifest["periods"]]
    files = [str(data / "events" / f"{name}.parquet") for name in names[:names.index(split) + 1]]
    with duckdb.connect(config={"threads": 2, "memory_limit": "1GB"}) as db:
        db.read_parquet(files).create_view("events")
        db.read_parquet(str(reference / "users.parquet")).create_view("users")
        db.read_parquet(str(reference / "events" / f"{split}.parquet")).filter("is_target").create_view("targets")
        values = db.execute("""SELECT u.user_idx AS owner, e.item_idx, e.history_size,
                e.event_id, t.event_id IS NOT NULL AS chosen
            FROM events e JOIN users u USING (uid) LEFT JOIN targets t USING (event_id)
            ORDER BY owner, e.timestamp, e.event_id""").fetchnumpy()
    starts = np.flatnonzero(np.r_[True, values["owner"][1:] != values["owner"][:-1]])
    contexts, indices = [], []
    for start, end in zip(starts, np.r_[starts[1:], len(values["owner"])]):
        chosen = np.flatnonzero(values["chosen"][start:end]) + start
        padded = np.pad(values["item_idx"][start:end], (length, 0))
        windows = np.lib.stride_tricks.sliding_window_view(padded, length)
        contexts.append(windows[values["history_size"][chosen]])
        indices.append(chosen)
    indices = np.concatenate(indices)
    return (np.concatenate(contexts).astype(np.int32), values["item_idx"][indices].astype(np.int64),
            values["owner"][indices], values["event_id"][indices])


def cohort_audit(small, large):
    with duckdb.connect(config={"threads": 2, "memory_limit": "1GB"}) as db:
        db.read_parquet(str(small / "users.parquet")).create_view("users")
        db.read_parquet([str(p) for p in sorted((small / "events").glob("*.parquet"))], filename=True).create_view("old")
        db.read_parquet([str(p) for p in sorted((large / "events").glob("*.parquet"))], filename=True).create_view("all_new")
        db.sql("SELECT e.* FROM all_new e JOIN users USING (uid)").create_view("new")
        row = db.execute("""SELECT count(*) FILTER (WHERE n.event_id IS NULL),
            count(*) FILTER (WHERE o.uid != n.uid OR o.item_id != n.item_id OR o.timestamp != n.timestamp
                OR o.played_ratio_pct != n.played_ratio_pct OR o.is_organic != n.is_organic
                OR o.is_positive != n.is_positive),
            count(*) FILTER (WHERE o.is_target AND NOT n.is_target)
            FROM old o LEFT JOIN new n USING (event_id)""").fetchone()
        extra = db.execute("""SELECT regexp_extract(filename, '([^/]+)\\.parquet$', 1) AS split, count(*)
            FROM new ANTI JOIN old USING (event_id) GROUP BY split ORDER BY split""").fetchall()
    if any(row):
        raise ValueError(f"Reference events changed or disappeared: {row}")
    return dict(zip(("missing_reference_events", "changed_raw_fields_or_labels", "lost_reference_targets"), row),
                extra_reference_user_events=dict(extra))


def repeat_distribution(data, reference):
    with duckdb.connect(config={"threads": 2, "memory_limit": "1GB"}) as db:
        db.read_parquet([str(p) for p in sorted((data / "events").glob("*.parquet"))], filename=True).create_view("events")
        db.read_parquet(str(reference / "users.parquet")).create_view("reference_users")
        db.execute("""CREATE TABLE groups AS SELECT uid,timestamp,arg_max(item_idx,event_id) AS last_item
            FROM events GROUP BY uid,timestamp""")
        rows = db.execute("""WITH marked AS (
            SELECT e.*,g.last_item FROM events e ASOF LEFT JOIN groups g
            ON e.uid=g.uid AND e.timestamp>g.timestamp)
            SELECT regexp_extract(filename,'([^/]+)\\.parquet$',1) AS split,
                CASE WHEN uid IN (SELECT uid FROM reference_users) THEN 'original_256' ELSE 'added_users' END AS cohort,
                count(*),count(*) FILTER (WHERE item_idx=last_item)
            FROM marked WHERE is_target GROUP BY split,cohort ORDER BY split,cohort""").fetchall()
    return [{"split": s, "cohort": c, "targets": n, "last_token_targets": r, "last_token_share": r / n}
            for s, c, n, r in rows if s in ("train", "model_valid", "test")]


def segment(rank_old, rank_large, rank_natural, mask):
    count = int(mask.sum())
    if not count:
        return {"events": 0}
    left = np.where(rank_old <= 10, 1 / np.log2(rank_old + 1), 0)
    right = np.where(rank_large <= 10, 1 / np.log2(rank_large + 1), 0)
    return {"events": count, "share": float(mask.mean()), "256": metrics(rank_old[mask]),
            "1k_old_inputs": metrics(rank_large[mask]), "1k_natural_inputs": metrics(rank_natural[mask]),
            "ndcg_delta_contribution": float(((right - left) * mask).mean()),
            "hit1_256": float((rank_old[mask] == 1).mean()),
            "hit1_1k": float((rank_large[mask] == 1).mean())}


def audit(device, output):
    small, large = (ROOT / f"data/processed/{name}" for name in ("yambda50m-pilot-audio", "yambda50m-1k-audio"))
    old_run, new_run = (ROOT / f"data/runs/{name}" for name in ("sasrec-control-early-stop", "sasrec-1k-early-stop"))
    old_meta, new_meta = (json.loads((p / "train-mps/training.json").read_text()) for p in (old_run, new_run))
    for folder, meta, data in ((old_run, old_meta, small), (new_run, new_meta, large)):
        if hashlib.sha256((folder / "train-mps/best.pt").read_bytes()).hexdigest() != meta["checkpoint_sha256"]:
            raise ValueError("Checkpoint hash mismatch")
        if hashlib.sha256((data / "manifest.json").read_bytes()).hexdigest() != meta["data_manifest_sha256"]:
            raise ValueError("Dataset manifest mismatch")
    output.mkdir(parents=True, exist_ok=True)
    write(output / "cohort-data.json", cohort_audit(small, large))
    write(output / "repeat-distribution.json", repeat_distribution(large, small))
    mapping = item_mapping(small, large)
    x, y, users = examples(small, "test", 50)
    pop = most_popular(small, np.r_[y, x[:, -1]])
    target_pop, last_pop = np.split(pop, 2)
    last_pop_rank = np.where(x[:, -1] == y, 1, target_pop + 1 - (last_pop < target_pop))
    write(output / "simple-baselines.json", {
        "description": "Last reference token first, then train MostPop without duplicates; same targets/candidates",
        "mostpop": metrics(target_pop), "last_plus_mostpop": metrics(last_pop_rank),
        "last_token_only_ndcg_lower_bound": float((x[:, -1] == y).mean())})
    rx, ry, ru, event_ids = aligned_examples(small, small, "test", 50)
    if not (np.array_equal(x, rx) and np.array_equal(y, ry) and np.array_equal(users, ru)):
        raise ValueError("Independent reference prefixes do not match examples()")
    natural, ny, nu, natural_ids = aligned_examples(large, small, "test", 50)
    if not (np.array_equal(ny, mapping[y]) and np.array_equal(nu, users) and np.array_equal(natural_ids, event_ids)):
        raise ValueError("Natural-history predictions are not aligned with reference targets")
    saved = torch.load(new_run / "train-mps/best.pt", map_location="cpu", weights_only=True)
    torch.set_num_threads(saved["config"]["threads"])
    model = SASRec(saved["items"], **model_args(saved["config"]))
    model.load_state_dict(saved["weights"])
    model.to(device).eval()
    natural_rank = ranks(model, natural, ny, device, 256, mapping[2:])
    old_predictions = np.load(old_run / "evaluation-mps.npz")
    large_predictions = np.load(new_run / "comparison-256.npz")
    rank_old, rank_large = old_predictions["ranks"], large_predictions["ranks"]
    changed = np.any(mapping[x] != natural, axis=1)
    last = x[:, -1] == y
    seen = np.any(x == y[:, None], axis=1)
    masks = {"all": np.ones(len(y), bool), "last_token": last, "earlier_in_window": seen & ~last,
             "absent_from_window": ~seen, "same_inputs": ~changed, "changed_inputs": changed,
             "last_token_same_inputs": last & ~changed, "last_token_changed_inputs": last & changed}
    result = {"examples": len(y), "changed_context_share": float(changed.mean()),
              "mean_changed_positions": float((mapping[x] != natural).sum(1).mean()),
              "changed_last_token_share": float((mapping[x][:, -1] != natural[:, -1]).mean()),
              "reference_prefixes_independently_verified": True, "same_raw_events_and_targets": True,
              "256_metrics": metrics(rank_old), "1k_old_inputs_metrics": metrics(rank_large),
              "1k_natural_inputs_metrics": metrics(natural_rank),
              "natural_minus_old_inputs": paired_intervals(natural_rank, rank_large, users, 42),
              "natural_minus_256": paired_intervals(natural_rank, rank_old, users, 42),
              "segments": {name: segment(rank_old, rank_large, natural_rank, mask) for name, mask in masks.items()}}
    write(output / "history-ablation.json", result)
    local = ROOT / "data/diagnostics/sasrec-1k-audit"
    local.mkdir(parents=True, exist_ok=True)
    np.savez(local / "natural-history.npz", ranks=natural_rank, users=users, targets=y, event_ids=event_ids,
             changed=changed, last=last, seen=seen)
    training = {}
    for name, folder, meta in (("256", old_run, old_meta), ("1k", new_run, new_meta)):
        state = torch.load(folder / "train-mps/best.pt", map_location="cpu", weights_only=True)["weights"]
        items = state["item.weight"][2:].norm(dim=1).numpy()
        positions = state["position.weight"].norm(dim=1).numpy()
        training[name] = {"selected_epoch": meta["best_epoch"],
            "optimizer_steps_at_selected_epoch": math.ceil(meta["train_examples"] / meta["config"]["batch_size"]) * meta["best_epoch"],
            "positive_presentations": meta["train_examples"] * meta["best_epoch"],
            "initial_validation": meta["initial_validation"],
            "selected_validation": meta["epochs"][meta["best_epoch"] - 1]["validation"],
            "all_weights_finite": all(torch.isfinite(v).all().item() for v in state.values()),
            "item_norm_quantiles": np.quantile(items, [.1, .5, .9, .99]).tolist(),
            "position_norm_quantiles": np.quantile(positions, [.1, .5, .9, .99]).tolist(),
            "last_position_norm": float(positions[-1])}
    write(output / "training-audit.json", training)
    print(json.dumps({k: v for k, v in result.items() if k != "segments"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "mps"], default="mps")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/experiments/sasrec-1k-audit")
    args = parser.parse_args()
    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS unavailable")
    audit(args.device, args.output)
