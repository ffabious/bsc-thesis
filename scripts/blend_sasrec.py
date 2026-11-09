#!/usr/bin/env python3
"""Mix query-normalized scores: select on validation, evaluate frozen weights on test."""

import argparse
import hashlib
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch

from compare_sasrec import item_mapping, paired_intervals
from run_sasrec import ROOT, intervals, memory, model_args, write
from sasrec import SASRec, examples, mask_context, metrics, sync


def read(path):
    return json.loads(path.read_text())


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_grid(alphas):
    values = np.asarray(alphas, dtype=float)
    if (values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all()
            or values[0] != 0 or values[-1] != 1 or np.any(np.diff(values) <= 0)):
        raise ValueError("Alpha grid must increase from 0 to 1, including both endpoints")


def select_alpha(curve):
    return max(curve, key=lambda row: (row["metrics"]["ndcg@10"], -row["alpha"]))["alpha"]


def blend_scores(long, recent, alpha):
    # Preserve exact endpoint arithmetic for comparison with saved single-model ranks.
    if alpha == 0:
        return long
    if alpha == 1:
        return recent
    return (1 - alpha) * long + alpha * recent


def normalize_scores(scores, epsilon=1e-8):
    """Center and scale candidate logits per query in place, without labels."""
    std, mean = torch.std_mean(scores, dim=1, unbiased=False, keepdim=True)
    return scores.sub_(mean).div_(std.clamp_min(epsilon))


def score_ranks(scores, ids, target):
    actual = scores.gather(1, torch.searchsorted(ids, target)[:, None])
    return (1 + (scores > actual).sum(1, dtype=torch.int32)
            + ((scores == actual) & (ids[None] < target[:, None])).sum(1, dtype=torch.int32))


def load_pair(protocol, pilot, device):
    models, configs, sources = {}, {}, {}
    for role, name in protocol["runs"][pilot].items():
        run = ROOT / name
        trained = read(run / "train-mps/training.json")
        path = run / "train-mps/best.pt"
        digest = checksum(path)
        if digest != trained["checkpoint_sha256"]:
            raise ValueError(f"Checkpoint changed: {path}")
        saved = torch.load(path, map_location="cpu", weights_only=True)
        config = saved["config"]
        if config != trained["config"] or checksum(ROOT / config["data"] / "manifest.json") != trained["data_manifest_sha256"]:
            raise ValueError("Checkpoint configuration or dataset manifest changed")
        model = SASRec(saved["items"], **model_args(config))
        model.load_state_dict(saved["weights"])
        models[role] = model.to(device).eval().requires_grad_(False)
        configs[role] = config
        sources[role] = {"run": name, "checkpoint_sha256": digest, "epoch": saved["epoch"],
                         "data_manifest_sha256": trained["data_manifest_sha256"], "items": saved["items"]}
    if (configs["long"]["data"] != configs["recent"]["data"]
            or sources["long"]["items"] != sources["recent"]["items"]
            or configs["long"].get("context_keep", configs["long"]["length"]) != 50
            or configs["recent"].get("context_keep", configs["recent"]["length"]) != 2):
        raise ValueError("Expected models with histories 50 and 2 in the same vocabulary")
    return models, configs, sources


@torch.inference_mode()
def blended_ranks(models, configs, x, y, device, batch_size, alphas, candidates=None, epsilon=1e-8):
    count = models["long"].item.num_embeddings
    candidates = np.arange(2, count) if candidates is None else np.asarray(candidates, dtype=np.int64)
    if (candidates.ndim != 1 or not len(candidates) or candidates[0] < 2
            or candidates[-1] >= count or np.any(np.diff(candidates) <= 0) or not np.isin(y, candidates).all()):
        raise ValueError("Candidates must be sorted known IDs and contain every target")
    ids = torch.as_tensor(candidates, device=device)
    vectors = {role: model.item(ids) for role, model in models.items()}
    result = np.empty((len(alphas), len(y)), dtype=np.int32)
    for start in range(0, len(y), batch_size):
        end = min(start + batch_size, len(y))
        scores = {}
        for role, model in models.items():
            view = mask_context(x[start:end, -model.length:].copy(), configs[role].get("context_keep"))
            query = model(torch.as_tensor(view, dtype=torch.long, device=device))
            scores[role] = query @ vectors[role].T
        target = torch.as_tensor(y[start:end], device=device)
        # Rank endpoints before normalization to preserve exact single-model arithmetic.
        for index, alpha in enumerate(alphas):
            if alpha in (0, 1):
                result[index, start:end] = score_ranks(scores["long" if alpha == 0 else "recent"], ids, target).cpu().numpy()
        if any(0 < alpha < 1 for alpha in alphas):
            for value in scores.values():
                normalize_scores(value, epsilon)
        # Keep only per-event ranks; full catalog scores never accumulate across batches.
        for index, alpha in enumerate(alphas):
            if alpha in (0, 1):
                continue
            mixed = blend_scores(scores["long"], scores["recent"], alpha)
            result[index, start:end] = score_ranks(mixed, ids, target).cpu().numpy()
        if (start // batch_size + 1) % 32 == 0:
            print(f"scored {end}/{len(y)}, {len(alphas)} weights", flush=True)
    return result


def rank_metrics(rank):
    return {**metrics(rank), "hit@1": float((rank == 1).mean())}


def select(protocol, pilot, device, output):
    destination = output / "selection.json"
    if destination.exists():
        raise FileExistsError(f"Selection is already fixed: {destination}")
    models, configs, sources = load_pair(protocol, pilot, device)
    data = ROOT / configs["long"]["data"]
    x, y, users = examples(data, "model_valid", configs["long"]["length"])
    started = time.perf_counter()
    rank = blended_ranks(models, configs, x, y, device, protocol["batch_size"], protocol["alphas"], epsilon=protocol["epsilon"])
    sync(device)
    curve = [{"alpha": alpha, "metrics": rank_metrics(value)} for alpha, value in zip(protocol["alphas"], rank)]
    for index, role in ((0, "long"), (-1, "recent")):
        trained = read(ROOT / sources[role]["run"] / "train-mps/training.json")
        if metrics(rank[index]) != trained["epochs"][trained["best_epoch"] - 1]["validation"]:
            raise ValueError(f"Validation endpoint differs from training log: {role}")
    result = {"pilot": pilot, "alpha": select_alpha(curve), "selected_on": "model_valid",
              "protocol": protocol, "sources": sources, "curve": curve,
              "examples": len(y), "users": int(len(np.unique(users))), "catalog_items": sources["long"]["items"],
              "endpoint_validation_reproduced": True, "scoring_seconds": time.perf_counter() - started,
              "device": device, "torch": torch.__version__, "numpy": np.__version__,
              "python": platform.python_version(), **memory(device)}
    output.mkdir(parents=True, exist_ok=True)
    np.savez(output / "validation-ranks.npz", ranks=rank, alphas=protocol["alphas"], targets=y, users=users)
    write(destination, result)
    print(f"Selected alpha={result['alpha']} for {pilot} using model_valid only", flush=True)


def evaluate(protocol, pilot, device, output):
    selection_path = output / "selection.json"
    selection = read(selection_path)
    models, configs, sources = load_pair(protocol, pilot, device)
    if selection["pilot"] != pilot or selection["protocol"] != protocol or selection["sources"] != sources:
        raise ValueError("Selection protocol or model sources changed")
    alpha = selection["alpha"]
    if alpha != select_alpha(selection["curve"]):
        raise ValueError("Selected coefficient differs from the validation maximum")
    selection_hash = checksum(selection_path)
    alphas = sorted(set([0., alpha, 1.]))
    data = ROOT / configs["long"]["data"]
    scopes = {"native": data}
    if pilot in ("1k", "1k-large"):
        scopes["shared"] = ROOT / protocol["shared_reference"]
    for scope, reference in scopes.items():
        destination = output / f"{scope}.json"
        if destination.exists():
            previous = read(destination)
            if pilot in ("256-large", "1k-large") and previous["selection_sha256"] == selection_hash and previous["sources"] == sources:
                continue
            raise FileExistsError(f"Evaluation already exists: {destination}")
        x, y, users = examples(reference, "test", configs["long"]["length"])
        last = x[:, -1] == y
        candidates = None
        model_x, model_y = x, y
        if scope != "native":
            mapping = item_mapping(reference, data)
            candidates, model_x, model_y = mapping[2:], mapping[x], mapping[y]
        started = time.perf_counter()
        rank = blended_ranks(models, configs, model_x, model_y, device, protocol["batch_size"], alphas,
                             candidates, protocol["epsilon"])
        sync(device)
        elapsed = time.perf_counter() - started
        endpoints = {}
        for value, role in ((0., "long"), (1., "recent")):
            comparison = "comparison-256" if scope == "shared" else f"comparison-{scope}"
            source = ROOT / sources[role]["run"] / ("evaluation-mps" if scope == "native" else comparison)
            saved = np.load(source.with_suffix(".npz"))
            baseline = read(source.with_suffix(".json"))
            actual = rank[alphas.index(value)]
            delta = np.abs(actual - saved["ranks"])
            if (baseline["checkpoint_sha256"] != sources[role]["checkpoint_sha256"]
                    or metrics(saved["ranks"]) != baseline["metrics"]
                    or not np.array_equal(y, saved["targets"]) or not np.array_equal(users, saved["users"])
                    or delta.max() > 1 or metrics(actual) != metrics(saved["ranks"])):
                raise ValueError(f"Endpoint differs from saved evaluation: {source}")
            endpoints[role] = {"different_ranks": int((delta != 0).sum()), "max_rank_difference": int(delta.max()),
                               "same_metrics": True}
        chosen = {"long_50": rank[alphas.index(0.)], "recent_2": rank[alphas.index(1.)], "blend": rank[alphas.index(alpha)]}
        result = {"pilot": pilot, "scope": scope, "alpha": alpha, "selection_sha256": selection_hash,
                  "sources": sources, "examples": len(y), "users": int(len(np.unique(users))),
                  "catalog_items": sources["long"]["items"] if candidates is None else len(candidates),
                  "metrics": {name: rank_metrics(value) for name, value in chosen.items()},
                  "cluster_bootstrap_95ci": {name: intervals(value, users, protocol["seed"]) for name, value in chosen.items()},
                  "paired_delta_blend_minus": {name: paired_intervals(chosen["blend"], value, users, protocol["seed"])
                                               for name, value in chosen.items() if name != "blend"},
                  "segments": {name: {"events": int(mask.sum()), "metrics": {key: rank_metrics(value[mask]) for key, value in chosen.items()}}
                               for name, mask in {"last_token": last, "other_targets": ~last}.items()},
                  "endpoint_checks": endpoints, "diagnostic_scoring_seconds": elapsed, "device": device, **memory(device)}
        np.savez(output / f"{scope}-ranks.npz", ranks=rank, alphas=alphas, targets=y, users=users)
        write(destination, result)
        if checksum(selection_path) != selection_hash:
            raise ValueError("Selection changed while evaluating test")
        print(f"{pilot} {scope}: alpha={alpha}, {result['metrics']['blend']}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["select", "evaluate"])
    parser.add_argument("--pilot", choices=["256", "1k", "256-large", "1k-large"], required=True)
    parser.add_argument("--device", choices=["cpu", "mps"], default="mps")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/experiments/sasrec-score-blend.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS unavailable")
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") != "0":
        parser.error("Disable MPS CPU fallback")
    protocol = read(args.config)
    validate_grid(protocol["alphas"])
    if protocol["selection_metric"] != "ndcg@10" or protocol["tie_break"] != "smaller_alpha":
        parser.error("This experiment selects validation NDCG@10, then smaller alpha")
    if protocol["score_transform"] != "per_query_zscore" or not protocol["epsilon"] > 0:
        parser.error("This experiment uses per-query z-scores with positive epsilon")
    torch.set_num_threads(protocol["threads"])
    (select if args.mode == "select" else evaluate)(protocol, args.pilot, args.device, args.output)
