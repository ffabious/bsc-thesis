#!/usr/bin/env python3
"""Replay on-request refresh policies on every observed timestamp group, without test."""

import argparse
import hashlib
import json
import os
import platform
import time
from pathlib import Path

import duckdb
import numpy as np
import torch

from compare_sasrec import paired_intervals
from replay_sasrec import distribution, encode_queries, quality, rank_vectors
from run_sasrec import ROOT, memory, model_args, write
from sasrec import SASRec, examples, metrics, ranks, sync


def policies(config):
    for key in ("event_thresholds", "time_thresholds_seconds"):
        values = config[key]
        if not values or values != sorted(set(values)) or any(type(v) is not int or v <= 0 for v in values):
            raise ValueError(f"{key} must be sorted distinct positive integers")
    result = [{"name": name, "family": name, "events": 0, "seconds": 0} for name in ("always", "never")]
    result += [{"name": f"events_{n}", "family": "events", "events": n, "seconds": 0}
               for n in config["event_thresholds"]]
    result += [{"name": f"time_{t}", "family": "time", "events": 0, "seconds": t}
               for t in config["time_thresholds_seconds"]]
    result += [{"name": f"hybrid_{n}_{t}", "family": "hybrid", "events": n, "seconds": t}
               for n in config["event_thresholds"] for t in config["time_thresholds_seconds"]]
    return result


def load_requests(data, splits, length):
    if splits != ["controller_train", "policy_valid"]:
        raise ValueError("Policy replay requires controller_train then policy_valid; test is forbidden")
    manifest = json.loads((data / "manifest.json").read_text())
    names = [p["split"] for p in manifest["periods"]]
    files = [str(data / "events" / f"{s}.parquet") for s in names[:names.index(splits[-1]) + 1]]
    selected = [str(data / "events" / f"{s}.parquet") for s in splits]
    case = "CASE " + " ".join(f"WHEN filename = ? THEN {i+1}" for i in range(len(splits))) + " ELSE 0 END"
    with duckdb.connect(config={"threads": 4, "memory_limit": "2GB"}) as db:
        db.read_parquet(files, filename=True).create_view("events")
        v = db.execute(f"""SELECT user_idx,item_idx,timestamp,event_id,history_size,is_target,
            {case} AS split FROM events ORDER BY user_idx,timestamp,event_id""", selected).fetchnumpy()
    parts = {k: [] for k in ("queries", "users", "timestamp", "history", "split_idx", "group_size",
                             "last_input_timestamp", "target_request", "targets", "target_event_ids", "source_row")}
    counts, offset = np.zeros(len(splits), dtype=np.int64), 0
    starts = np.flatnonzero(np.r_[True, v["user_idx"][1:] != v["user_idx"][:-1]])
    for start, end in zip(starts, np.r_[starts[1:], len(v["user_idx"])]):
        times, items = v["timestamp"][start:end], v["item_idx"][start:end]
        group = np.flatnonzero(np.r_[True, times[1:] != times[:-1]])
        requests = group[v["split"][start + group] > 0]
        if not len(requests):
            continue
        if (requests == 0).any():
            raise ValueError("A request lacks a nonempty strictly past history")
        # The absolute within-user group start is the strict-past history count.
        if not np.array_equal(v["history_size"][start + requests], requests):
            raise ValueError("history_size includes the current timestamp group")
        windows = np.lib.stride_tricks.sliding_window_view(np.pad(items, (length, 0)), length)
        parts["queries"].append(windows[requests].astype(np.int32))
        parts["history"].append(requests)
        parts["last_input_timestamp"].append(times[requests - 1])
        size = np.diff(np.r_[group, len(times)])
        parts["group_size"].append(size[np.searchsorted(group, requests)])
        for key, array in (("users", v["user_idx"]), ("timestamp", v["timestamp"]), ("split_idx", v["split"])):
            parts[key].append(array[start + requests])
        target = np.flatnonzero(v["is_target"][start:end] & (v["split"][start:end] > 0))
        target_group = np.searchsorted(requests, target, side="right") - 1
        parts["target_request"].append(target_group + offset)
        parts["targets"].append(items[target])
        parts["target_event_ids"].append(v["event_id"][start + target])
        source_row = np.empty(len(target), dtype=np.int64)
        for i in range(len(splits)):
            mask = v["split"][start + target] == i + 1
            source_row[mask] = counts[i] + np.arange(mask.sum())
            counts[i] += mask.sum()
        parts["source_row"].append(source_row)
        offset += len(requests)
    result = {key: np.concatenate(value) for key, value in parts.items()}
    order = np.lexsort((result["users"], result["timestamp"], result["split_idx"]))
    inverse = np.empty(len(order), dtype=np.int64)
    inverse[order] = np.arange(len(order))
    for key in ("queries", "users", "timestamp", "history", "split_idx", "group_size", "last_input_timestamp"):
        result[key] = result[key][order]
    result["target_request"] = inverse[result["target_request"]]
    if not np.all(result["last_input_timestamp"] < result["timestamp"]):
        raise ValueError("A request observes the current or future timestamp group")
    return result


def schedule(stream, policy, user_count):
    """Decisions see only user, current clock and strictly past observed-event count."""
    refs = np.empty(len(stream["users"]), dtype=np.int32)
    refresh = np.empty(len(refs), dtype=bool)
    cached = np.full(user_count, -1, dtype=np.int32)
    current_split = -1
    users, times, history = stream["users"], stream["timestamp"], stream["history"]
    for i, (split, user, timestamp, count) in enumerate(zip(stream["split_idx"], users, times, history)):
        if split != current_split:
            cached.fill(-1)
            current_split = split
        previous = cached[user]
        update = (previous < 0 or policy["family"] == "always" or
                  (policy["events"] > 0 and count - history[previous] >= policy["events"]) or
                  (policy["seconds"] > 0 and timestamp - times[previous] >= policy["seconds"]))
        if update:
            cached[user] = i
        refs[i], refresh[i] = cached[user], update
    return refs, refresh


@torch.inference_mode()
def timed_branches(model, stream, vectors, request_ids, snapshot_ids, updates, config, device):
    """Measure serial inline refresh/write/serve or hit/serve; pre-existing cache is setup."""
    cache = vectors[torch.as_tensor(snapshot_ids, device=device)].cpu().numpy().copy()
    table = model.item.weight[2:].T

    def operation(i):
        if updates[i]:
            x = torch.as_tensor(stream["queries"][request_ids[i:i+1]], dtype=torch.long, device=device)
            query = model(x)
            cache[i:i+1] = query.cpu().numpy()
        else:
            query = torch.as_tensor(cache[i:i+1], device=device)
        return (query @ table).topk(min(100, table.shape[1]), dim=1).indices.cpu()

    for i in range(config["benchmark_warmup"]):
        operation(i % len(request_ids))
        sync(device)
    steps, trials = config["benchmark_steps"], []
    for repeat in range(config["benchmark_repeats"]):
        started = time.perf_counter()
        for i in range(repeat * steps, (repeat + 1) * steps):
            operation(i)
            sync(device)
        trials.append((time.perf_counter() - started) * 1000 / steps)
    return {"repeat_mean_ms": trials, "median_mean_ms": float(np.median(trials)),
            "sample_requests": len(request_ids), "sample_refreshes": int(updates.sum())}


def branch_costs(model, stream, vectors, split_idx, config, device):
    pool = np.flatnonzero(stream["split_idx"] == split_idx)
    n = config["benchmark_steps"] * config["benchmark_repeats"]
    take = np.random.default_rng(config["seed"] + split_idx).choice(pool, n, replace=True)
    return {name: timed_branches(model, stream, vectors, take, take, np.full(n, update), config, device)
            for name, update in (("refresh_write_serve", True), ("cache_hit_serve", False))}


def policy_cost(refreshes, requests, decision_ms, branches):
    refresh = branches["refresh_write_serve"]["median_mean_ms"]
    hit = branches["cache_hit_serve"]["median_mean_ms"]
    serving = refreshes * refresh + (requests - refreshes) * hit
    repeats = [(refreshes * r + (requests - refreshes) * h) / requests + decision_ms
               for r, h in zip(branches["refresh_write_serve"].get("repeat_mean_ms", [refresh]),
                               branches["cache_hit_serve"].get("repeat_mean_ms", [hit]))]
    return {"estimated_total_ms": serving + decision_ms * requests,
            "estimated_ms_per_request": serving / requests + decision_ms,
            "cost_repeat_estimates_ms_per_request": repeats,
            "decision_ms_per_request": decision_ms,
            "refresh_branch_total_ms": refreshes * refresh,
            "hit_branch_total_ms": (requests - refreshes) * hit}


def frontier(rows, metric):
    """Two-objective Pareto points, retaining all exactly equivalent settings."""
    return [r["name"] for r in rows if not any(
        q["cost"]["estimated_ms_per_request"] <= r["cost"]["estimated_ms_per_request"] and
        q["metrics"][metric] >= r["metrics"][metric] and
        (q["cost"]["estimated_ms_per_request"] < r["cost"]["estimated_ms_per_request"] or
         q["metrics"][metric] > r["metrics"][metric]) for q in rows)]


def select_budgets(rows, budgets):
    always = next(r for r in rows if r["name"] == "always")
    result = []
    for budget in budgets:
        eligible = [r for r in rows if r["cost"]["estimated_ms_per_request"] <= budget * always["cost"]["estimated_ms_per_request"]]
        chosen = max(eligible, key=lambda r: (r["metrics"]["ndcg@10"], r["metrics"]["recall@100"],
                                             -r["cost"]["estimated_ms_per_request"])) if eligible else None
        result.append({"relative_budget": budget, "policy": chosen["name"] if chosen else None})
    return result


def summarize_split(stream, rank, refs, refresh, choices, split_idx, branches, decision_ms, seed):
    requests = np.flatnonzero(stream["split_idx"] == split_idx)
    targets = np.flatnonzero(stream["split_idx"][stream["target_request"]] == split_idx)
    target_requests = stream["target_request"][targets]
    users = stream["users"][target_requests]
    fresh_queries, y = stream["queries"][target_requests], stream["targets"][targets]
    last = fresh_queries[:, -1] == y
    seen = (fresh_queries == y[:, None]).any(1)
    initialization = len(np.unique(stream["users"][requests]))
    rows = []
    for j, choice in enumerate(choices):
        states, updates = refs[j, requests], int(refresh[j, requests].sum())
        age_events = stream["history"][requests] - stream["history"][states]
        age_seconds = stream["timestamp"][requests] - stream["timestamp"][states]
        rows.append({**choice, "metrics": quality(rank[j]),
                     "policy_minus_always": paired_intervals(rank[j], rank[0], users, seed),
                     "requests": len(requests), "refreshes": updates, "initializations": initialization,
                     "subsequent_refreshes": updates - initialization, "hits": len(requests) - updates,
                     "refresh_fraction": updates / len(requests),
                     "cost": policy_cost(updates, len(requests), decision_ms[j][split_idx-1], branches),
                     "request_age_events": distribution(age_events),
                     "request_age_seconds": distribution(age_seconds),
                     "target_fresh_fraction": float((refs[j, target_requests] == target_requests).mean()),
                     "strata": {key: {"examples": int(mask.sum()), "metrics": quality(rank[j, mask])}
                                for key, mask in (("last_item", last), ("other_in_window", seen & ~last), ("outside_window", ~seen)) if mask.any()}})
    return {"requests": len(requests), "observed_listens": int(stream["group_size"][requests].sum()),
            "request_users": initialization, "targets": len(targets), "target_users": len(np.unique(users)),
            "policies": rows, "frontiers": {metric: frontier(rows, metric) for metric in ("ndcg@10", "recall@100")}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "mps"], required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output directory")
    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS unavailable")
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") != "0":
        parser.error("Disable CPU fallback")
    config = json.loads(args.config.read_text())
    choices = policies(config)
    torch.set_num_threads(config["threads"])
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    data, checkpoint = ROOT / config["data"], ROOT / config["checkpoint"]
    manifest = json.loads((data / "manifest.json").read_text())
    trained = json.loads(checkpoint.with_name("training.json").read_text())
    if digest(checkpoint) != config["checkpoint_sha256"] or trained["checkpoint_sha256"] != config["checkpoint_sha256"]:
        raise ValueError("Checkpoint changed")
    if digest(data / "manifest.json") != trained["data_manifest_sha256"]:
        raise ValueError("Dataset manifest differs from training")
    names = [p["split"] for p in manifest["periods"]]
    used = {f"events/{s}.parquet" for s in names[:names.index(config["splits"][-1]) + 1]}
    used.update(("items.parquet", "users.parquet"))
    hashes = {r["path"]: r["sha256"] for r in manifest["files"] if r["path"] in used}
    for name, checksum in hashes.items():
        if digest(data / name) != checksum:
            raise ValueError(f"Dataset changed: {name}")
    sources = {str(p.relative_to(ROOT)): digest(p) for p in (
        ROOT / "scripts/replay_sasrec_policies.py", ROOT / "scripts/replay_sasrec.py", ROOT / "scripts/sasrec.py",
        ROOT / "scripts/run_sasrec.py", ROOT / "scripts/compare_sasrec.py", args.config.resolve())}
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if saved["items"] != manifest["candidate_items"] or saved["config"]["data"] != config["data"]:
        raise ValueError("Checkpoint vocabulary/data differs")
    model = SASRec(saved["items"], **model_args(saved["config"]))
    model.load_state_dict(saved["weights"])
    model.to(args.device).eval()

    def state_digest():
        value = hashlib.sha256()
        for name, tensor in model.state_dict().items():
            value.update(name.encode())
            value.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        return value.hexdigest()

    before = state_digest()
    started = time.perf_counter()
    stream = load_requests(data, config["splits"], model.length)
    prepare_seconds = time.perf_counter() - started
    refs, refresh, decision_ms = [], [], []
    for policy in choices:
        trials = []
        for _ in range(config["benchmark_repeats"]):
            trial = []
            for i in (1, 2):
                take = stream["split_idx"] == i
                part = {k: stream[k][take] for k in ("users", "timestamp", "history", "split_idx")}
                started = time.perf_counter()
                schedule(part, policy, manifest["selected_users"])
                trial.append((time.perf_counter() - started) * 1000 / take.sum())
            trials.append(trial)
        r, f = schedule(stream, policy, manifest["selected_users"])
        refs.append(r)
        refresh.append(f)
        decision_ms.append(np.median(trials, axis=0).tolist())
    refs, refresh = np.array(refs), np.array(refresh)
    if not np.array_equal(refs[0], np.arange(len(stream["queries"]))):
        raise ValueError("Always is not fresh on all requests")
    if not np.all(stream["users"][refs] == stream["users"][None]):
        raise ValueError("Cache crossed users")
    if not np.all(stream["split_idx"][refs] == stream["split_idx"][None]):
        raise ValueError("Cache was not reset at split boundary")
    if not np.all(stream["last_input_timestamp"][refs] < stream["timestamp"][None]):
        raise ValueError("Future event leaked into a cache")
    print(json.dumps({"requests": len(stream["queries"]), "targets": len(stream["targets"]), "policies": len(choices)}), flush=True)
    vectors, encode_seconds = encode_queries(model, stream["queries"], args.device, config["batch_size"])
    results, costs, checks, ranked, timings = {}, {}, {}, [], {}
    args.output.mkdir(parents=True)
    for index, split in enumerate(config["splits"], 1):
        take = stream["split_idx"][stream["target_request"]] == index
        targets = stream["targets"][take]
        requests = stream["target_request"][take]
        rank, timings[split] = rank_vectors(model, vectors, refs[:, requests], targets, args.device, config["batch_size"])
        x, y, u = examples(data, split, model.length)
        rows = stream["source_row"][take]
        if not (np.array_equal(y[rows], targets) and np.array_equal(u[rows], stream["users"][requests]) and
                np.array_equal(x[rows], stream["queries"][requests])):
            raise ValueError("Fresh inputs differ from the original evaluator")
        oracle = ranks(model, x, y, args.device, config["batch_size"])[rows]
        if metrics(oracle) != metrics(rank[0]):
            raise ValueError("Fresh metrics differ from the original evaluator")
        checks[split] = {"same_fresh_inputs_and_metrics": True, "targets": len(y),
                         "exact_rank_matches": int((oracle == rank[0]).sum()),
                         "maximum_rank_difference": int(np.abs(oracle - rank[0]).max())}
        costs[split] = branch_costs(model, stream, vectors, index, config, args.device)
        results[split] = summarize_split(stream, rank, refs, refresh, choices, index, costs[split], decision_ms, config["seed"])
        ranked.append((take, rank))
        print(json.dumps({"split": split, "fresh": quality(rank[0]), "frontiers": results[split]["frontiers"]}), flush=True)
        if index == 1:
            selection = {"source_split": split, "criterion": "max NDCG, ties Recall then lower estimated cost",
                         "budgets": select_budgets(results[split]["policies"], config["cost_budgets_relative_to_always"])}
            write(args.output / "selection.json", selection)  # Frozen before evaluating policy_valid.
    rank = np.empty((len(choices), len(stream["targets"])), dtype=np.int32)
    for take, values in ranked:
        rank[:, take] = values
    cpu = SASRec(saved["items"], **model_args(saved["config"]))
    cpu.load_state_dict(saved["weights"])
    p, t = np.unravel_index(np.random.default_rng(config["seed"]).choice(rank.size, 256, replace=False), rank.shape)
    oracle = ranks(cpu, stream["queries"][refs[p, stream["target_request"][t]]], stream["targets"][t], "cpu", config["batch_size"])
    if metrics(oracle) != metrics(rank[p,t]):
        raise ValueError("CPU/MPS metrics differ")
    checks["cpu"] = {"pairs": len(t), "same_metrics": True, "exact_rank_matches": int((oracle == rank[p,t]).sum()),
                     "maximum_rank_difference": int(np.abs(oracle - rank[p,t]).max())}
    mixed = {}
    checked = ["always", "never", *dict.fromkeys(r["policy"] for r in selection["budgets"] if r["policy"])]
    for index, split in enumerate(config["splits"], 1):
        pool = np.flatnonzero(stream["split_idx"] == index)
        n = config["benchmark_steps"] * config["benchmark_repeats"]
        take = np.random.default_rng(config["seed"] + index + 100).choice(pool, n, replace=True)
        mixed[split] = {}
        for name in checked:
            j = next(i for i, policy in enumerate(choices) if policy["name"] == name)
            block = timed_branches(model, stream, vectors, take, refs[j,take], refresh[j,take], config, args.device)
            predicted = policy_cost(int(refresh[j,take].sum()), n, 0, costs[split])["estimated_ms_per_request"]
            mixed[split][name] = {**block, "predicted_sample_ms": predicted,
                                  "measured_minus_predicted_percent": 100 * (block["median_mean_ms"] / predicted - 1)}
    if state_digest() != before or digest(checkpoint) != config["checkpoint_sha256"]:
        raise ValueError("Frozen weights changed")
    for name, checksum in hashes.items():
        if digest(data / name) != checksum:
            raise ValueError(f"Dataset changed during replay: {name}")
    for name, checksum in sources.items():
        if digest(ROOT / name) != checksum:
            raise ValueError(f"Source changed during replay: {name}")
    np.savez(args.output / "replay.npz", **stream, refs=refs, refresh=refresh, ranks=rank)
    write(args.output / "results.json", {"config": config, "policies": choices, "results": results,
          "branch_costs": costs, "mixed_workload_checks": mixed, "selection": selection, "verification": checks,
          "cost_scope": "Count-weighted measured serial B=1 inline refresh/write/serve and hit/serve branches plus CPU decisions; no network/history storage",
          "offline_execution": {"prepare_seconds": prepare_seconds, "encode_unique_seconds": encode_seconds,
                                "unique_snapshots": len(vectors), "scoring": timings},
          "checkpoint_sha256": digest(checkpoint), "data_manifest_sha256": digest(data / "manifest.json"),
          "frozen_state_sha256": before, "used_data_files_sha256": hashes, "source_sha256": sources,
          "device": args.device, "torch": torch.__version__, "numpy": np.__version__, "duckdb": duckdb.__version__,
          "python": platform.python_version(), "macos": platform.mac_ver()[0], **memory(args.device)})
    print(json.dumps({"selection": selection, "mixed_workload_checks": mixed}), flush=True)


if __name__ == "__main__":
    main()
