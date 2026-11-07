#!/usr/bin/env python3
"""Measure validation composition and scoring mechanisms in the saved pilot models."""

import argparse
import json
import sys
from pathlib import Path

import duckdb
import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from compare_sasrec import item_mapping, paired_intervals
from run_sasrec import model_args, write
from sasrec import SASRec, batch, examples, metrics, negatives, ranks


@torch.inference_mode()
def states(model, x, device):
    model.eval()
    return torch.cat([model(torch.as_tensor(x[start:start + 256], dtype=torch.long, device=device))
                      for start in range(0, len(x), 256)])


@torch.inference_mode()
def vector_ranks(query, vectors, columns, batch_size=256):
    """The columns follow reference item order, including its deterministic tie break."""
    ids = torch.arange(len(vectors), device=query.device)
    result = []
    for start in range(0, len(query), batch_size):
        target = torch.as_tensor(columns[start:start + batch_size], device=query.device)
        scores = query[start:start + batch_size] @ vectors.T
        actual = scores.gather(1, target[:, None])
        rank = 1 + (scores > actual).sum(1, dtype=torch.int32)
        rank += ((scores == actual) & (ids[None] < target[:, None])).sum(1, dtype=torch.int32)
        result.append(rank.cpu().numpy())
    return np.concatenate(result)


def load(folder, device, initial=False):
    saved = torch.load(folder / "train-mps/best.pt", map_location="cpu", weights_only=True)
    if initial:
        torch.manual_seed(saved["config"]["seed"])
        if device == "mps":
            torch.mps.manual_seed(saved["config"]["seed"])
    model = SASRec(saved["items"], **model_args(saved["config"]))
    if not initial:
        model.load_state_dict(saved["weights"])
    return model.to(device).eval()


def audit(device, output):
    torch.set_num_threads(4)
    small, large = (ROOT / f"data/processed/{name}" for name in ("yambda50m-pilot-audio", "yambda50m-1k-audio"))
    old_run, new_run = (ROOT / f"data/runs/{name}" for name in ("sasrec-control-early-stop", "sasrec-1k-early-stop"))
    meta = json.loads((new_run / "train-mps/training.json").read_text())
    xv, yv, uv = examples(large, "model_valid", 50)
    with duckdb.connect() as db:
        reference_users = db.execute("""SELECT n.user_idx FROM read_parquet(?) n
            JOIN read_parquet(?) o USING (uid)""", [str(large / "users.parquet"), str(small / "users.parquet")]).fetchnumpy()["user_idx"]
    reference = np.isin(uv, reference_users)
    last = xv[:, -1] == yv
    model = load(new_run, device)
    trained_rank = ranks(model, xv, yv, device, 256)
    print("Selected checkpoint validation recomputed", flush=True)
    initial_model = load(new_run, device, initial=True)
    initial_rank = ranks(initial_model, xv, yv, device, 256)
    print("Initial model validation recomputed", flush=True)
    if metrics(trained_rank) != meta["epochs"][meta["best_epoch"] - 1]["validation"]:
        raise ValueError("Selected checkpoint validation differs from the training log")
    if metrics(initial_rank) != meta["initial_validation"]:
        raise ValueError("Seed reconstruction differs from the logged initial model")
    cohorts = {}
    for name, mask in {"all": np.ones(len(yv), bool), "original_256": reference, "added_users": ~reference,
                       "last_token": last, "other_targets": ~last}.items():
        cohorts[name] = {"events": int(mask.sum()), "share": float(mask.mean()),
                         "last_token_share": float(last[mask].mean()),
                         "initial": metrics(initial_rank[mask]), "trained": metrics(trained_rank[mask])}
    write(output / "validation-composition.json", {"reproduced_initial_and_selected_validation": True, "cohorts": cohorts})
    del xv, yv, uv, initial_rank, trained_rank, model, initial_model

    x, y, users = examples(small, "test", 50)
    mapping = item_mapping(small, large)
    repeat = x[:, -1] == y
    results = {}
    for name, folder, contexts, candidates in (
            ("256", old_run, x, np.arange(2, len(mapping))),
            ("1k", new_run, mapping[x], mapping[2:])):
        model = load(folder, device)
        initial = load(folder, device, initial=True)
        initial_rank = ranks(initial, contexts, candidates[y - 2], device, 256, candidates)
        del initial
        query = states(model, contexts, device)
        with torch.inference_mode():
            vectors = model.item.weight[torch.as_tensor(candidates, device=device)]
            last_vectors = model.item.weight[torch.as_tensor(contexts[:, -1], device=device)]
            unit_rank = vector_ranks(query, F.normalize(vectors, dim=1), y - 2)
            copy_rank = vector_ranks(last_vectors, vectors, y - 2)
            unit_query = F.normalize(query, dim=1)
            cosine_last = (unit_query * F.normalize(last_vectors, dim=1)).sum(1).cpu().numpy()
            raw_check = vector_ranks(query[:256], vectors, y[:256] - 2)
        baseline_path = folder / ("evaluation-mps.npz" if name == "256" else "comparison-256.npz")
        baseline = np.load(baseline_path)["ranks"]
        if not np.array_equal(raw_check, baseline[:256]):
            raise ValueError("Cached-state scoring differs from saved ranks")
        variants = {"trained": baseline, "initial": initial_rank, "unit_candidate_norm": unit_rank,
                    "last_token_lookup_query": copy_rank}
        results[name] = {"cached_state_ranks_verified_on_256_events": True,
            "mean_query_unit_vector_norm": float(unit_query.mean(0).norm().item()),
            "mean_query_last_item_cosine": float(cosine_last.mean()),
            "repeat_query_last_item_cosine": float(cosine_last[repeat].mean()),
            "variants": {label:{"all":metrics(rank), "last_token":metrics(rank[repeat]),
                                  "other_targets":metrics(rank[~repeat])} for label,rank in variants.items()}}
        print(f"Scoring mechanisms evaluated: {name}", flush=True)
        del model, query, vectors, last_vectors, unit_query
    write(output / "scoring-ablation.json", {"exploratory_not_model_selection": True, "models": results})
    print(json.dumps(results, indent=2), flush=True)


def reference_validation(device, output):
    small, large = (ROOT / f"data/processed/{name}" for name in ("yambda50m-pilot-audio", "yambda50m-1k-audio"))
    old_run, new_run = (ROOT / f"data/runs/{name}" for name in ("sasrec-control-early-stop", "sasrec-1k-early-stop"))
    torch.set_num_threads(4)
    x, y, users = examples(small, "model_valid", 50)
    mapping = item_mapping(small, large)
    old = ranks(load(old_run, device), x, y, device, 256)
    new = ranks(load(new_run, device), mapping[x], mapping[y], device, 256, mapping[2:])
    meta = json.loads((old_run / "train-mps/training.json").read_text())
    if metrics(old) != meta["epochs"][meta["best_epoch"] - 1]["validation"]:
        raise ValueError("Reference validation differs from the training log")
    result = {"same_inputs_and_candidates": True, "examples": len(y), "256": metrics(old), "1k": metrics(new),
              "paired_delta": paired_intervals(new, old, users, 42)}
    write(output / "reference-validation.json", result)
    print(json.dumps(result, indent=2), flush=True)


def loss_gradient_probe(device, output):
    small, large = (ROOT / f"data/processed/{name}" for name in ("yambda50m-pilot-audio", "yambda50m-1k-audio"))
    model = load(ROOT / "data/runs/sasrec-1k-early-stop", device)
    x, y, _ = examples(small, "model_valid", 50)
    mapping = item_mapping(small, large)
    sequence, target = batch(mapping[x[:256]], mapping[y[:256]], slice(None), device)
    sampled = negatives(mapping[y[:256]], 16, model.item.num_embeddings - 2, np.random.default_rng(42))
    with torch.no_grad():
        state = model(sequence)
        positive = (state * model.item(target)).sum(-1).cpu().double().requires_grad_()
        negative = (state[:, None] * model.item(torch.as_tensor(sampled, device=device))).sum(-1).cpu().double().requires_grad_()
    current = F.softplus(-positive).mean() + F.softplus(negative).mean()
    joint = (F.softplus(-positive).sum() + F.softplus(negative).sum()) / (positive.numel() + negative.numel())
    left = torch.autograd.grad(current, (positive, negative))
    right = torch.autograd.grad(joint, (positive, negative))
    torch.testing.assert_close(right[0], left[0] / 17)
    torch.testing.assert_close(right[1], left[1] * 16 / 17)
    result = {"split": "model_valid", "examples": len(positive), "negatives_per_positive": 16,
              "fixed_logits_and_negatives": True, "optimizer_step_performed": False,
              "joint_to_current_positive_gradient_factor": 1 / 17,
              "joint_to_current_each_negative_gradient_factor": 16 / 17,
              "relative_negative_weight_increase": 16,
              "current_positive_gradient_l1": float(left[0].abs().sum()),
              "current_negative_gradient_l1": float(left[1].abs().sum()),
              "joint_positive_gradient_l1": float(right[0].abs().sum()),
              "joint_negative_gradient_l1": float(right[1].abs().sum()),
              "note": "Reduction check on detached logits, CPU float64; not a quality or training-improvement experiment."}
    write(output / "loss-gradient-probe.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "mps"], default="mps")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/experiments/sasrec-1k-audit")
    args = parser.parse_args()
    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS unavailable")
    args.output.mkdir(parents=True, exist_ok=True)
    audit(args.device, args.output)
    reference_validation(args.device, args.output)
    loss_gradient_probe(args.device, args.output)
