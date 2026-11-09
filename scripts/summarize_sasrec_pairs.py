#!/usr/bin/env python3
"""Save aggregate results and paired comparisons without selecting anything on test."""

import argparse
import hashlib
import json
import numpy as np

from compare_sasrec import paired_intervals
from run_sasrec import ROOT, write
from sasrec import metrics


def read(path):
    return json.loads(path.read_text())


def blend(path):
    meta = read(path.with_suffix(".json"))
    arrays = np.load(path.with_name(path.name + "-ranks.npz"))
    chosen = arrays["alphas"].tolist().index(meta["alpha"])
    rank = arrays["ranks"][chosen]
    if metrics(rank) != {key: meta["metrics"]["blend"][key] for key in ("ndcg@10", "recall@100")}:
        raise ValueError(f"Stored blend metrics differ: {path}")
    return meta, rank, arrays["targets"], arrays["users"]


def paired(left, right):
    if not all(np.array_equal(left[i], right[i]) for i in (2, 3)):
        raise ValueError("Paired evaluation requires identical targets, event ordering and user groups")
    return paired_intervals(left[1], right[1], left[3], 42)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilots", nargs="+", choices=["256", "1k"], default=["256"])
    args = parser.parse_args()
    output = ROOT / "reports/experiments/sasrec-large"
    results, native, shared, comparisons = {}, {}, {}, {}
    old_root, new_root = ROOT / "data/runs/sasrec-score-blend", ROOT / "data/runs/sasrec-large-blend"
    for pilot in args.pilots:
        old = blend(old_root / pilot / "native")
        new = blend(new_root / f"{pilot}-large/native")
        native[pilot] = {"old": old[0], "large": new[0]}
        comparisons[f"{pilot}_native_large_minus_old_blend"] = paired(new, old)
        for name in ("selection", "native"):
            source = new_root / f"{pilot}-large/{name}.json"
            (output / f"{pilot}-{name}.json").write_bytes(source.read_bytes())
        for role, suffix in (("long", "joint-bce"), ("recent", "recent-2")):
            folder = ROOT / f"data/runs/sasrec-{pilot}-large-{role}"
            training = read(folder / "train-mps/training.json")
            if training["stop_reason"] != "validation_patience":
                raise ValueError("Training must finish by validation, not an epoch cap")
            checkpoint = folder / "train-mps/best.pt"
            if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != training["checkpoint_sha256"]:
                raise ValueError("Checkpoint changed")
            (output / f"{pilot}-{role}-training.json").write_bytes((folder / "train-mps/training.json").read_bytes())
            (output / f"{pilot}-{role}-evaluation.json").write_bytes((folder / "evaluation-mps.json").read_bytes())
            benchmark = read(folder / "benchmark-mps.json")
            results[f"{pilot}_{role}"] = {
                "best_epoch": training["best_epoch"], "epochs": len(training["epochs"]),
                "elapsed_seconds": training["elapsed_seconds"], "parameters": training["parameters"],
                "encoder_parameters": training["parameters"] - (read(ROOT / training["config"]["data"] / "manifest.json")["candidate_items"] + 2) * 64,
                "checkpoint_sha256": training["checkpoint_sha256"], "benchmark": benchmark,
                "best_validation": training["epochs"][training["best_epoch"] - 1]["validation"]}
            old_file = ROOT / f"data/runs/sasrec-{pilot}-{suffix}/evaluation-mps.npz"
            old_values, new_values = np.load(old_file), np.load(folder / "evaluation-mps.npz")
            if not all(np.array_equal(old_values[key], new_values[key]) for key in ("targets", "users")):
                raise ValueError("Single-model events changed")
            comparisons[f"{pilot}_{role}_large_minus_old"] = paired_intervals(new_values["ranks"], old_values["ranks"], new_values["users"], 42)

    shared = {"old_256": blend(old_root / "256/native"), "large_256": blend(new_root / "256-large/native")}
    if "1k" in args.pilots:
        shared.update(old_1k=blend(old_root / "1k/shared"), large_1k=blend(new_root / "1k-large/shared"))
        comparisons["shared_1k_large_minus_old_blend"] = paired(shared["large_1k"], shared["old_1k"])
        comparisons["shared_large_1k_minus_large_256"] = paired(shared["large_1k"], shared["large_256"])
        comparisons["shared_large_1k_minus_old_256"] = paired(shared["large_1k"], shared["old_256"])
        (output / "1k-shared.json").write_bytes((new_root / "1k-large/shared.json").read_bytes())
    summary = {"training": results, "native": native, "shared": {key: value[0] for key, value in shared.items()},
               "paired_comparisons": comparisons,
               "limitations": ["one training seed", "test previously used in pilot research",
                               "native metrics across different catalogs are not directly comparable",
                               "short input uses native length 2 instead of masked length 50",
                               "z-score normalization does not calibrate interaction probabilities"]}
    write(output / "summary.json", summary)
    print(json.dumps({key: value["metrics"]["blend"] for key, value in summary["shared"].items()}, indent=2))


if __name__ == "__main__":
    main()
