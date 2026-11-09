#!/usr/bin/env python3
"""Compare saved ranks on identical events, including user and history segments."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from compare_sasrec import paired_intervals
from run_sasrec import ROOT, write
from sasrec import examples, metrics


def read(path):
    meta = json.loads(path.read_text())
    arrays = np.load(path.with_suffix(".npz"))
    if metrics(arrays["ranks"]) != meta["metrics"]:
        raise ValueError(f"Metrics differ from saved ranks: {path}")
    return meta, arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--evaluations", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reference, baseline = read(args.reference)
    split, config = reference.get("split", "test"), reference["config"]
    data = ROOT / config["data"]
    checksum = hashlib.sha256((data / "manifest.json").read_bytes()).hexdigest()
    x, y, users = examples(data, split, config["length"])
    _, inverse = np.unique(users, return_inverse=True)
    counts = np.bincount(inverse)
    last, seen = x[:, -1] == y, (x == y[:, None]).any(1)
    masks = {"last_item": last, "other_in_window": seen & ~last, "outside_window": ~seen}
    rows = {}
    for path in dict.fromkeys([args.reference, *args.evaluations]):
        meta, arrays = read(path)
        current = meta["config"]
        if (meta.get("split", "test") != split or meta["data_manifest_sha256"] != checksum
                or current["data"] != config["data"] or current["length"] != config["length"]
                or current.get("context_keep", config["length"]) != config["length"]
                or meta["catalog_items"] != reference["catalog_items"]
                or not np.array_equal(arrays["targets"], y) or not np.array_equal(arrays["users"], users)):
            raise ValueError(f"Inputs, targets, split or candidates differ: {path}")
        rank = arrays["ranks"]
        quality = lambda r: {**metrics(r), "hit@1": float((r == 1).mean())}
        values = {"ndcg@10": np.where(rank <= 10, 1 / np.log2(rank + 1), 0),
                  "recall@100": (rank <= 100).astype(float)}
        rows[str(path)] = {"checkpoint_sha256": meta["checkpoint_sha256"], "config": current,
            "metrics": quality(rank),
            "user_macro": {name: float((np.bincount(inverse, weights=value) / counts).mean())
                           for name, value in values.items()},
            "paired_delta": paired_intervals(rank, baseline["ranks"], users, 42),
            "segments": {name: {"examples": int(mask.sum()), **quality(rank[mask])}
                         for name, mask in masks.items()}}
    write(args.output, {"split": split, "examples": len(y), "users": len(counts),
                        "catalog_items": reference["catalog_items"], "reference": str(args.reference),
                        "context_sha256": hashlib.sha256(x.tobytes()).hexdigest(), "runs": rows})


if __name__ == "__main__":
    main()
