#!/usr/bin/env python3
"""Compare a larger-user model on the saved pilot's exact inputs and catalog."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import duckdb
import numpy as np
import torch

from run_sasrec import ROOT, intervals, model_args, write
from sasrec import SASRec, examples, metrics, ranks, sync


def item_mapping(reference, data):
    with duckdb.connect(config={"threads": 2, "memory_limit": "512MB"}) as db:
        rows = db.execute("""SELECT old.item_idx, new.item_idx FROM read_parquet(?) old
            LEFT JOIN read_parquet(?) new USING (item_id) ORDER BY old.item_idx""",
            [str(reference / "items.parquet"), str(data / "items.parquet")]).fetchall()
    if any(new is None for _, new in rows):
        raise ValueError("Training vocabulary does not cover the reference catalog")
    if [old for old, _ in rows] != list(range(2, len(rows) + 2)):
        raise ValueError("Reference item indices must be contiguous")
    return np.array([0, 1] + [new for _, new in rows], dtype=np.int64)


def paired_intervals(rank, baseline, users, seed):
    ids, inverse = np.unique(users, return_inverse=True)
    counts = np.bincount(inverse)
    draws = np.random.default_rng(seed).integers(0, len(ids), (2000, len(ids)))
    result = {}
    for name, values in {"ndcg@10": lambda r: np.where(r <= 10, 1 / np.log2(r + 1), 0),
                         "recall@100": lambda r: (r <= 100).astype(float)}.items():
        difference = values(rank) - values(baseline)
        sums = np.bincount(inverse, weights=difference)
        ci = np.quantile(sums[draws].sum(1) / counts[draws].sum(1), [.025, .975])
        result[name] = {"difference": float(difference.mean()), "cluster_bootstrap_95ci": ci.tolist()}
    return result


def compare(checkpoint, reference_path, device, output):
    if output.exists() or output.with_suffix(".npz").exists():
        raise FileExistsError(f"Choose a new comparison output: {output}")
    trained = json.loads(checkpoint.with_name("training.json").read_text())
    checksum = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if checksum != trained["checkpoint_sha256"]:
        raise ValueError("Checkpoint differs from the completed training run")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    reference = json.loads(reference_path.read_text())
    data, small = ROOT / saved["config"]["data"], ROOT / reference["config"]["data"]
    for meta, folder in ((trained, data), (reference, small)):
        if meta["data_manifest_sha256"] != hashlib.sha256((folder / "manifest.json").read_bytes()).hexdigest():
            raise ValueError("Dataset manifest changed after training/evaluation")
    if model_args(saved["config"]) != model_args(reference["config"]):
        raise ValueError("Comparison requires the same architecture")
    torch.set_num_threads(saved["config"]["threads"])
    model = SASRec(saved["items"], **model_args(saved["config"]))
    model.load_state_dict(saved["weights"])
    model.to(device).eval()
    x, y, users = examples(small, "test", model.length)
    baseline = np.load(reference_path.with_suffix(".npz"))
    if (not np.array_equal(y, baseline["targets"]) or not np.array_equal(users, baseline["users"])
            or metrics(baseline["ranks"]) != reference["metrics"]):
        raise ValueError("Saved reference predictions do not match the reference dataset/metrics")
    mapping = item_mapping(small, data)
    candidates = mapping[2:]
    context_sha256 = hashlib.sha256(x.tobytes()).hexdigest()
    x, mapped_y = mapping[x], mapping[y]
    batch_size = saved["config"]["eval_batch_size"]
    ranks(model, x[:batch_size], mapped_y[:batch_size], device, batch_size, candidates)
    sync(device)
    started = time.perf_counter()
    rank = ranks(model, x, mapped_y, device, batch_size, candidates)
    sync(device)
    elapsed = time.perf_counter() - started
    result = {"checkpoint_sha256": checksum, "checkpoint_epoch": saved["epoch"],
              "reference_checkpoint_sha256": reference["checkpoint_sha256"],
              "training_data": str(data.relative_to(ROOT)), "reference_data": str(small.relative_to(ROOT)),
              "training_data_manifest_sha256": trained["data_manifest_sha256"],
              "reference_data_manifest_sha256": reference["data_manifest_sha256"],
              "reference_context_sha256": context_sha256, "same_inputs_targets_and_candidates": True,
              "examples": len(y), "users": int(len(np.unique(users))),
              "training_catalog_items": saved["items"], "comparison_catalog_items": len(candidates),
              "reference_metrics": reference["metrics"], "metrics": metrics(rank),
              "cluster_bootstrap_95ci": intervals(rank, users, saved["config"]["seed"]),
              "paired_delta": paired_intervals(rank, baseline["ranks"], users, saved["config"]["seed"]),
              "evaluation_seconds": elapsed, "device": device, "torch": torch.__version__}
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output.with_suffix(".npz"), ranks=rank, users=users, targets=y)
    write(output, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference-evaluation", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "mps"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS unavailable")
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") != "0":
        parser.error("Disable CPU fallback")
    print(json.dumps(compare(args.checkpoint, args.reference_evaluation, args.device, args.output), indent=2))
