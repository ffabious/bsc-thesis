#!/usr/bin/env python3
"""Inspect last-track repeats in saved joint-BCE pilots without training."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import duckdb
import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts"), str(Path(__file__).resolve().parent)]
from audit_pilot_scale import aligned_examples
from audit_pilot_training import load, states, vector_ranks
from compare_sasrec import item_mapping, paired_intervals
from run_sasrec import write
from sasrec import examples, metrics, most_popular, ranks


def distribution(values):
    return {"mean": float(np.mean(values)),
            "p10_median_p90": np.quantile(values, [.1, .5, .9]).tolist()}


def rank_summary(rank):
    return {**metrics(rank), "hit@1": float((rank == 1).mean()),
            "hit@10": float((rank <= 10).mean()), **distribution(rank)}


@torch.inference_mode()
def traced_states(model, contexts, device):
    captured = {f"block_{i + 1}": [] for i in range(len(model.blocks))}
    handles = [block.register_forward_hook(
        lambda module, inputs, output, key=key: captured[key].append(output[:, -1].clone()))
        for key, block in zip(captured, model.blocks)]
    try:
        query = states(model, contexts, device)
    finally:
        for handle in handles:
            handle.remove()
    last = model.item.weight[torch.as_tensor(contexts[:, -1], device=device)]
    initial = model.input_projection(last) * model.position.embedding_dim ** .5 + model.position.weight[-1]
    parts = {"input": initial, **{key: torch.cat(value) for key, value in captured.items()}, "output": query}
    return query, {key: F.cosine_similarity(value, last).cpu().numpy() for key, value in parts.items()}


@torch.inference_mode()
def geometry(query, vectors, columns):
    """Score only selected events; keep per-event arrays local."""
    values = {name: [] for name in ("target_score", "top_score", "margin", "target_cosine",
                                   "top_cosine", "target_norm", "top_norm", "query_norm", "top_column")}
    for start in range(0, len(query), 128):
        q = query[start:start + 128]
        target = torch.as_tensor(columns[start:start + len(q)], device=q.device)
        scores = q @ vectors.T
        top = scores.argmax(1)
        positive = scores.gather(1, target[:, None]).squeeze(1)
        best = scores.gather(1, top[:, None]).squeeze(1)
        target_vector, top_vector = vectors[target], vectors[top]
        row = dict(target_score=positive, top_score=best, margin=positive - best,
                   target_cosine=F.cosine_similarity(q, target_vector),
                   top_cosine=F.cosine_similarity(q, top_vector),
                   target_norm=target_vector.norm(dim=1), top_norm=top_vector.norm(dim=1),
                   query_norm=q.norm(dim=1), top_column=top)
        for name, value in row.items():
            values[name].append(value.cpu().numpy())
    return {name: np.concatenate(value) for name, value in values.items()}


def audit(device, output, local):
    torch.set_num_threads(4)
    small, large = (ROOT / f"data/processed/{name}" for name in
                    ("yambda50m-pilot-audio", "yambda50m-1k-audio"))
    folders = {"256": ROOT / "data/runs/sasrec-256-joint-bce",
               "1k": ROOT / "data/runs/sasrec-1k-joint-bce"}
    sources = {}
    for name, folder in folders.items():
        meta = json.loads((folder / "train-mps/training.json").read_text())
        checksum = hashlib.sha256((folder / "train-mps/best.pt").read_bytes()).hexdigest()
        assert checksum == meta["checkpoint_sha256"]
        assert meta["config"]["loss_reduction"] == "joint_mean"
        data = ROOT / meta["config"]["data"]
        assert hashlib.sha256((data / "manifest.json").read_bytes()).hexdigest() == meta["data_manifest_sha256"]
        sources[name] = {"checkpoint_sha256": checksum, "selected_epoch": meta["best_epoch"]}
    mapping = item_mapping(small, large)
    output.mkdir(parents=True, exist_ok=True)
    local.mkdir(parents=True, exist_ok=True)
    result = {"sources": sources, "exploratory_not_model_selection": True, "splits": {}}
    for split in ("model_valid", "test"):
        x, y, users = examples(small, split, 50)
        chosen = np.flatnonzero(x[:, -1] == y)
        natural, ny, nu, event_ids = aligned_examples(large, small, split, 50)
        assert np.array_equal(ny, mapping[y]) and np.array_equal(nu, users)
        natural, x, y, users = natural[chosen], x[chosen], y[chosen], users[chosen]
        changed = np.any(natural != mapping[x], axis=1)
        same_last = natural[:, -1] == mapping[y]
        entry = {"events": len(y), "users": int(len(np.unique(users))),
                 "unique_tracks": int(len(np.unique(y))), "changed_history_share": float(changed.mean()),
                 "natural_last_still_target_share": float(same_last.mean()), "models": {}}
        arrays = dict(users=users, targets=y, reference_indices=chosen, event_ids=event_ids[chosen],
                      changed_history=changed, natural_last_still_target=same_last)
        for name, folder in folders.items():
            model = load(folder, device)
            contexts = x if name == "256" else mapping[x]
            candidates = np.arange(2, len(mapping)) if name == "256" else mapping[2:]
            query, trace = traced_states(model, contexts, device)
            with torch.inference_mode():
                vectors = model.item.weight[torch.as_tensor(candidates, device=device)]
                last_vectors = model.item.weight[torch.as_tensor(contexts[:, -1], device=device)]
                raw = vector_ranks(query, vectors, y - 2)
                unit = vector_ranks(query, F.normalize(vectors, dim=1), y - 2)
                lookup = vector_ranks(last_vectors, vectors, y - 2)
            last_only = np.zeros_like(contexts)
            last_only[:, -1] = contexts[:, -1]
            only_query, only_trace = traced_states(model, last_only, device)
            only = vector_ranks(only_query, vectors, y - 2)
            details = geometry(query, vectors, y - 2)
            variants = {"raw": raw, "unit_candidate_norm": unit,
                        "last_embedding_query": lookup, "last_only_context": only}
            for keep in (2, 10):
                recent = contexts.copy()
                recent[:, :-keep] = 0
                variants[f"recent_{keep}_context"] = vector_ranks(states(model, recent, device), vectors, y - 2)
            if name == "1k":
                variants["natural_history"] = vector_ranks(states(model, natural, device), vectors, y - 2)
            if split == "test":
                path = folder / ("evaluation-mps.npz" if name == "256" else "comparison-256.npz")
                saved = np.load(path)
                assert np.array_equal(saved["users"][chosen], users)
                assert np.array_equal(saved["targets"][chosen], y)
                assert np.array_equal(raw, saved["ranks"][chosen])
            entry["models"][name] = {
                "variants": {key: rank_summary(value) for key, value in variants.items()},
                "geometry": {key: distribution(value) for key, value in details.items() if key != "top_column"},
                "stage_last_item_cosine": {key: distribution(value) for key, value in trace.items()},
                "last_only_stage_cosine": {key: distribution(value) for key, value in only_trace.items()},
                "same_history": rank_summary(raw[~changed]),
                "changed_history": rank_summary(raw[changed])}
            prediction = details["top_column"] + 2
            wrong = prediction != y
            seen = np.any(x == prediction[:, None], axis=1)
            popular = most_popular(small, prediction)
            entry["models"][name]["top1_competitors"] = {
                "top1_equals_last": int((~wrong).sum()),
                "top1_other_history_item": int((wrong & seen).sum()),
                "top1_outside_window": int((wrong & ~seen).sum()),
                "wrong_top1_in_reference_train_pop_top100": int((wrong & (popular <= 100)).sum()),
                "wrong_top1_events": int(wrong.sum()), "top1_pop_rank_median": float(np.median(popular))}
            for key, value in {**variants, **details}.items():
                arrays[f"{name}_{key}"] = value
            for key, value in trace.items():
                arrays[f"{name}_cosine_{key}"] = value
            del model, query, vectors, last_vectors
            print(f"{split}: {name}, {len(y)} repeat events scored", flush=True)
        left, right = arrays["256_raw"], arrays["1k_raw"]
        buckets = lambda rank: np.where(rank == 1, 0, np.where(rank <= 10, 1, np.where(rank <= 100, 2, 3)))
        entry["rank_transition_256_to_1k"] = np.bincount(buckets(left) * 4 + buckets(right), minlength=16).reshape(4, 4).tolist()
        entry["transition_labels"] = ["1", "2..10", "11..100", ">100"]
        entry["paired_delta_1k_minus_256"] = paired_intervals(right, left, users, 42)
        result["splits"][split] = entry
        np.savez(local / f"{split}.npz", **arrays)
        write(output / "repeat-scoring.json", result)
    return result


def full_validation(device, output, local):
    small, large = (ROOT / f"data/processed/{name}" for name in
                    ("yambda50m-pilot-audio", "yambda50m-1k-audio"))
    folder = ROOT / "data/runs/sasrec-1k-joint-bce"
    meta = json.loads((folder / "train-mps/training.json").read_text())
    x, y, users = examples(large, "model_valid", 50)
    with duckdb.connect() as db:
        old_users = db.execute("""SELECT n.user_idx FROM read_parquet(?) n
            JOIN read_parquet(?) old USING (uid)""",
            [str(large / "users.parquet"), str(small / "users.parquet")]).fetchnumpy()["user_idx"]
    original, last = np.isin(users, old_users), x[:, -1] == y
    model = load(folder, device)
    raw = ranks(model, x, y, device, 256)
    assert metrics(raw) == meta["epochs"][meta["best_epoch"] - 1]["validation"]
    x[:, :-2] = 0
    recent = ranks(model, x, y, device, 256)
    result = {"same_full_1k_validation_and_candidates": True, "selected_checkpoint_validation_reproduced": True,
              "checkpoint_sha256": meta["checkpoint_sha256"], "groups": {}}
    for name, mask in {"all": np.ones(len(y), bool), "original_256": original,
                       "added_users": ~original, "last_token": last, "other_targets": ~last,
                       "original_last_token": original & last, "added_last_token": ~original & last}.items():
        result["groups"][name] = {"events": int(mask.sum()), "raw": rank_summary(raw[mask]),
                                  "recent_2": rank_summary(recent[mask])}
    result["paired_delta_recent_minus_raw"] = paired_intervals(recent, raw, users, 42)
    np.savez(local / "full-validation.npz", raw=raw, recent_2=recent, users=users, targets=y, last=last, original=original)
    write(output / "validation-context.json", result)
    print("Full 1k validation: full history and recent 2 evaluated", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "mps"], default="mps")
    parser.add_argument("--full-validation", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/experiments/sasrec-joint-repeat-audit")
    args = parser.parse_args()
    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS unavailable")
    local = ROOT / "data/diagnostics/sasrec-joint-repeat-audit"
    audit(args.device, args.output, local)
    if args.full_validation:
        full_validation(args.device, args.output, local)
