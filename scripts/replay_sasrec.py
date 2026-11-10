#!/usr/bin/env python3
"""Paired snapshot-age replay and local serving-cost measurements for frozen SASRec."""

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
from run_sasrec import ROOT, memory, model_args, write
from sasrec import SASRec, examples, metrics, ranks, sync


def settings(config):
    result = [{"name": "fresh", "unit": "fresh", "lag": 0}]
    for key, unit in (("event_lags", "events"), ("time_lags_seconds", "seconds")):
        values = config[key]
        if not values or values[0] != 0 or values != sorted(set(values)) or any(type(v) is not int or v < 0 for v in values):
            raise ValueError(f"{key} must contain sorted distinct nonnegative integers starting at zero")
        result.extend({"name": f"{unit}_{v}", "unit": unit, "lag": v} for v in values if v)
    return result


def cutoffs(times, history, timestamps, conditions):
    if not np.array_equal(history, np.searchsorted(times, timestamps, side="left")):
        raise ValueError("history_size must exclude the entire current timestamp group")
    boundaries = np.r_[0, np.flatnonzero(times[1:] != times[:-1]) + 1, len(times)]
    cuts, seconds = [], []
    for condition in conditions:
        lag, unit = condition["lag"], condition["unit"]
        if unit == "seconds":
            cut = np.searchsorted(times, timestamps - lag, side="left")
            age = np.full(len(history), lag, dtype=np.int64)
        else:
            # A cache snapshot cannot observe only part of an unordered timestamp group.
            wanted = np.maximum(history - lag, 0)
            cut = boundaries[np.searchsorted(boundaries, wanted, side="right") - 1]
            age = timestamps - times[cut]
        cuts.append(cut)
        seconds.append(age)
    return np.array(cuts), np.array(seconds)


def load_replay(data, splits, length, conditions):
    if not splits or len(set(splits)) != len(splits) or not set(splits) <= {"controller_train", "policy_valid"}:
        raise ValueError("Replay is restricted to controller_train and policy_valid")
    manifest = json.loads((data / "manifest.json").read_text())
    names = [p["split"] for p in manifest["periods"]]
    files = [str(data / "events" / f"{name}.parquet") for name in names[:max(names.index(s) for s in splits) + 1]]
    selected = [str(data / "events" / f"{s}.parquet") for s in splits]
    case = "CASE " + " ".join(f"WHEN is_target AND filename = ? THEN {i+1}" for i in range(len(splits))) + " ELSE 0 END"
    with duckdb.connect(config={"threads": 4, "memory_limit": "2GB"}) as db:
        db.read_parquet(files, filename=True).create_view("events")
        v = db.execute(f"""SELECT user_idx, item_idx, timestamp, event_id, history_size,
            {case} AS target_split FROM events ORDER BY user_idx, timestamp, event_id""", selected).fetchnumpy()
    parts = {key: [] for key in ("queries", "query_users", "query_last_timestamp", "refs", "targets", "users",
             "timestamp", "event_ids", "split_idx", "source_row", "lag_events", "lag_seconds", "activity_24h")}
    coverage = {s: {"all_targets": 0, "included": 0, "excluded": 0, "excluded_users": set(),
                    "missing_history_by_lag": {c["name"]: 0 for c in conditions}} for s in splits}
    starts = np.flatnonzero(np.r_[True, v["user_idx"][1:] != v["user_idx"][:-1]])
    offset = 0
    for start, end in zip(starts, np.r_[starts[1:], len(v["user_idx"])]):
        chosen = np.flatnonzero(v["target_split"][start:end]) + start
        if not len(chosen):
            continue
        times, items = v["timestamp"][start:end], v["item_idx"][start:end]
        history, timestamps = v["history_size"][chosen], v["timestamp"][chosen]
        cuts, seconds = cutoffs(times, history, timestamps, conditions)
        keep = (cuts > 0).all(0)  # One fixed paired cohort for every lag and both lag families.
        owner, split = int(v["user_idx"][start]), v["target_split"][chosen]
        source_row = np.empty(len(chosen), dtype=np.int64)
        for i, name in enumerate(splits, 1):
            mask, row = split == i, coverage[name]
            source_row[mask] = row["all_targets"] + np.arange(mask.sum())
            row["all_targets"] += int(mask.sum())
            row["included"] += int((mask & keep).sum())
            row["excluded"] += int((mask & ~keep).sum())
            if np.any(mask & ~keep):
                row["excluded_users"].add(owner)
            for condition, count in zip(conditions, ((cuts == 0) & mask[None]).sum(1)):
                row["missing_history_by_lag"][condition["name"]] += int(count)
        if not keep.any():
            continue
        unique, inverse = np.unique(cuts[:, keep], return_inverse=True)
        windows = np.lib.stride_tricks.sliding_window_view(np.pad(items, (length, 0)), length)
        parts["queries"].append(windows[unique].astype(np.int32))
        parts["query_users"].append(np.full(len(unique), owner, dtype=np.int32))
        parts["query_last_timestamp"].append(times[unique - 1])
        parts["refs"].append(inverse.reshape(cuts[:, keep].shape) + offset)
        offset += len(unique)
        for key, array in {"targets": v["item_idx"][chosen], "timestamp": timestamps,
                           "event_ids": v["event_id"][chosen], "split_idx": split, "source_row": source_row,
                           "users": np.full(len(chosen), owner, dtype=np.int32),
                           "activity_24h": history - np.searchsorted(times, timestamps - 86400, side="left")}.items():
            parts[key].append(array[keep])
        parts["lag_events"].append(history[None, keep] - cuts[:, keep])
        parts["lag_seconds"].append(seconds[:, keep])
    if not parts["queries"]:
        raise ValueError("No targets have a nonempty history at every lag")
    result = {key: np.concatenate(value, axis=1 if key in ("refs", "lag_events", "lag_seconds") else 0)
              for key, value in parts.items()}
    if not np.all(result["query_users"][result["refs"]] == result["users"][None]):
        raise ValueError("Snapshot crossed user boundaries")
    if not np.all(result["query_last_timestamp"][result["refs"]] < result["timestamp"][None] - result["lag_seconds"]):
        raise ValueError("Snapshot contains events unavailable at its refresh time")
    for row in coverage.values():
        row["excluded_users"] = len(row["excluded_users"])
    return result, coverage


@torch.inference_mode()
def encode_queries(model, queries, device, batch_size):
    result = torch.empty((len(queries), model.item.embedding_dim), device=device)
    sync(device)
    started = time.perf_counter()
    for start in range(0, len(queries), batch_size):
        seq = torch.as_tensor(queries[start:start + batch_size], dtype=torch.long, device=device)
        result[start:start + len(seq)] = model(seq)
    sync(device)
    return result, time.perf_counter() - started


@torch.inference_mode()
def rank_vectors(model, vectors, refs, targets, device, batch_size):
    width = model.item.num_embeddings
    keys, inverse = np.unique(refs * width + targets[None], return_inverse=True)
    states, labels = keys // width, keys % width
    output = np.empty(len(keys), dtype=np.int32)
    ids = torch.arange(2, width, device=device)
    sync(device)
    started = time.perf_counter()
    for start in range(0, len(keys), batch_size):
        chosen, rows = np.unique(states[start:start + batch_size], return_inverse=True)
        query = vectors[torch.as_tensor(chosen, device=device)]
        scores = query @ model.item.weight[2:].T
        scores = scores[torch.as_tensor(rows, device=device)]
        target = torch.as_tensor(labels[start:start + batch_size], device=device)
        actual = scores.gather(1, (target - 2)[:, None])
        rank = 1 + (scores > actual).sum(1, dtype=torch.int32)
        rank += ((scores == actual) & (ids[None] < target[:, None])).sum(1, dtype=torch.int32)
        output[start:start + len(target)] = rank.cpu().numpy()
    sync(device)
    return output[inverse].reshape(refs.shape), {"unique_state_target_pairs": len(keys),
                                               "scoring_ranking_seconds": time.perf_counter() - started}


def quality(rank):
    return {**metrics(rank), "hit@1": float((rank == 1).mean())}


def distribution(values):
    return dict(zip(("min", "median", "p90", "max"), np.quantile(values, [0, .5, .9, 1]).tolist()))


def summarize(replay, rank, conditions, splits, seed):
    results = {}
    for index, name in enumerate(splits, 1):
        mask = replay["split_idx"] == index
        y, users = replay["targets"][mask], replay["users"][mask]
        fresh_x = replay["queries"][replay["refs"][0, mask]]
        last, seen = fresh_x[:, -1] == y, (fresh_x == y[:, None]).any(1)
        activity = replay["activity_24h"][mask]
        groups = {"last_item": last, "other_in_window": seen & ~last, "outside_window": ~seen,
                  "past_day_0_9": activity < 10, "past_day_10_49": (activity >= 10) & (activity < 50),
                  "past_day_50_plus": activity >= 50}
        fresh = rank[0, mask]
        rows = []
        for j, condition in enumerate(conditions):
            current, refs = rank[j, mask], replay["refs"][j, mask]
            strata = {key: {"examples": int(take.sum()), "metrics": quality(current[take]) if take.any() else None,
                            "fresh_metrics": quality(fresh[take]) if take.any() else None} for key, take in groups.items()}
            rows.append({**condition, "metrics": quality(current),
                         "stale_minus_fresh": paired_intervals(current, fresh, users, seed), "strata": strata,
                         "unique_snapshots": int(len(np.unique(refs))),
                         "unchanged_context_fraction": float((refs == replay["refs"][0, mask]).mean()),
                         "age_events": distribution(replay["lag_events"][j, mask]),
                         "age_seconds": distribution(replay["lag_seconds"][j, mask])})
        results[name] = {"examples": len(y), "users": len(np.unique(users)), "conditions": rows}
    return results


@torch.inference_mode()
def benchmark_cost(model, replay, vectors, config, device, user_count):
    chosen = np.flatnonzero(replay["split_idx"] == 1)
    _, first = np.unique(replay["users"][chosen], return_index=True)
    chosen = chosen[first]
    pool = replay["queries"][replay["refs"][0, chosen]]
    users = replay["users"][chosen]
    gpu_pool = torch.as_tensor(pool, dtype=torch.long, device=device)
    gpu_query = vectors[torch.as_tensor(replay["refs"][0, chosen], device=device)]
    cache = np.zeros((user_count, model.item.embedding_dim), dtype=np.float32)
    cache[users] = gpu_query.cpu().numpy()
    table = model.item.weight[2:].T
    serve = lambda query: (query @ table).topk(min(100, table.shape[1]), dim=1).indices.cpu()
    results = {"cache_bytes": cache.nbytes, "cache_location": "CPU NumPy; model/item table on selected device",
               "scope": "pre-tokenized histories, full catalog top-100, CPU result; synchronized local batches; no network",
               "amortization": "cache_hit_e2e + refresh_encode_write / reuse_count; not the cost of a fixed-lag policy",
               "batches": []}
    for batch_size, steps in zip(config["benchmark_batches"], config["benchmark_steps"]):
        draws = np.random.default_rng(config["seed"]).integers(len(pool), size=(steps * config["benchmark_repeats"] + config["benchmark_warmup"], batch_size))
        gpu_draws = torch.as_tensor(draws, device=device)
        row = {"batch_size": batch_size, "steps_per_repeat": steps, "components": {}}
        for kind in ("encode_only", "score_top100", "cache_lookup_transfer", "refresh_encode_write", "fresh_e2e", "cache_hit_e2e"):
            cursor = 0

            def operation():
                nonlocal cursor
                take, gpu_take = draws[cursor], gpu_draws[cursor]
                cursor += 1
                if kind == "encode_only":
                    return model(gpu_pool[gpu_take])
                if kind == "score_top100":
                    return serve(gpu_query[gpu_take])
                if kind in ("cache_lookup_transfer", "cache_hit_e2e"):
                    query = torch.as_tensor(cache[users[take]], device=device)
                    return serve(query) if kind == "cache_hit_e2e" else query
                sequence = torch.as_tensor(pool[take], dtype=torch.long, device=device)
                query = model(sequence)
                if kind == "fresh_e2e":
                    return serve(query)
                cache[users[take]] = query.cpu().numpy()

            for _ in range(config["benchmark_warmup"]):
                operation()
                sync(device)
            trials = []
            for _ in range(config["benchmark_repeats"]):
                started = time.perf_counter()
                for _ in range(steps):
                    operation()
                    sync(device)
                trials.append((time.perf_counter() - started) * 1000 / steps)
            row["components"][kind] = {"batch_ms": trials, "median_batch_ms": float(np.median(trials)),
                                       "median_per_request_ms": float(np.median(trials)) / batch_size}
        cost = lambda key: row["components"][key]["median_per_request_ms"]
        row["amortized"] = [{"predictions_per_refresh": n, "estimated_per_request_ms": cost("cache_hit_e2e") + cost("refresh_encode_write") / n}
                            for n in config["reuse_counts"]]
        row["cache_hit_speedup_without_refresh"] = cost("fresh_e2e") / cost("cache_hit_e2e")
        results["batches"].append(row)
    return results


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
    conditions = settings(config)
    torch.set_num_threads(config["threads"])
    data, checkpoint = ROOT / config["data"], ROOT / config["checkpoint"]
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = json.loads((data / "manifest.json").read_text())
    trained = json.loads(checkpoint.with_name("training.json").read_text())
    if digest(checkpoint) != config["checkpoint_sha256"] or trained["checkpoint_sha256"] != config["checkpoint_sha256"]:
        raise ValueError("Checkpoint changed")
    if digest(data / "manifest.json") != trained["data_manifest_sha256"]:
        raise ValueError("Dataset manifest differs from training")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if saved["items"] != manifest["candidate_items"] or saved["config"]["data"] != config["data"]:
        raise ValueError("Checkpoint vocabulary or data path differs from the replay")
    model = SASRec(saved["items"], **model_args(saved["config"]))
    model.load_state_dict(saved["weights"])
    model.to(args.device).eval()
    def state_digest():
        value = hashlib.sha256()
        for name, tensor in model.state_dict().items():
            value.update(name.encode())
            value.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        return value.hexdigest()

    state_before = state_digest()
    started = time.perf_counter()
    replay, coverage = load_replay(data, config["splits"], model.length, conditions)
    prepare_seconds = time.perf_counter() - started
    periods = [p["split"] for p in manifest["periods"]]
    used = {f"events/{name}.parquet" for name in periods[:max(periods.index(s) for s in config["splits"]) + 1]}
    used.update(("items.parquet", "users.parquet"))
    hashes = {row["path"]: row["sha256"] for row in manifest["files"] if row["path"] in used}
    for name, checksum in hashes.items():
        if digest(data / name) != checksum:
            raise ValueError(f"Dataset file changed: {name}")
    sources = {str(p.relative_to(ROOT)): digest(p) for p in
               (ROOT / "scripts/replay_sasrec.py", ROOT / "scripts/sasrec.py", ROOT / "scripts/run_sasrec.py",
                ROOT / "scripts/compare_sasrec.py", args.config.resolve())}
    print(json.dumps({"coverage": coverage, "unique_snapshots": len(replay["queries"])}, indent=2), flush=True)
    vectors, encode_seconds = encode_queries(model, replay["queries"], args.device, config["batch_size"])
    rank, timing = rank_vectors(model, vectors, replay["refs"], replay["targets"], args.device, config["batch_size"])
    fresh_checks = {}
    for i, name in enumerate(config["splits"], 1):
        x, y, users = examples(data, name, model.length)
        mask = replay["split_idx"] == i
        take = replay["source_row"][mask]
        if (not np.array_equal(x[take], replay["queries"][replay["refs"][0, mask]]) or
                not np.array_equal(y[take], replay["targets"][mask]) or not np.array_equal(users[take], replay["users"][mask])):
            raise ValueError("Fresh inputs differ from the existing evaluation path")
        oracle = ranks(model, x, y, args.device, config["batch_size"])
        if metrics(oracle[take]) != metrics(rank[0, mask]):
            raise ValueError("Fresh replay metrics differ from the existing ranker")
        fresh_checks[name] = {"all_events": len(y), "all_fresh_metrics": quality(oracle),
                             "covered_fresh_metrics": quality(oracle[take]), "same_metrics": True,
                             "exact_rank_matches": int((oracle[take] == rank[0, mask]).sum()),
                             "maximum_rank_difference": int(np.abs(oracle[take] - rank[0, mask]).max())}
    results = summarize(replay, rank, conditions, config["splits"], config["seed"])
    cpu = SASRec(saved["items"], **model_args(saved["config"]))
    cpu.load_state_dict(saved["weights"])
    lag, take = np.unravel_index(np.random.default_rng(config["seed"]).choice(rank.size, min(256, rank.size), replace=False), rank.shape)
    oracle = ranks(cpu, replay["queries"][replay["refs"][lag, take]], replay["targets"][take], "cpu", config["batch_size"])
    if metrics(oracle) != metrics(rank[lag, take]):
        raise ValueError("CPU oracle metrics differ from the snapshot replay")
    cpu_check = {"pairs": len(take), "same_metrics": True, "exact_rank_matches": int((oracle == rank[lag, take]).sum()),
                 "maximum_rank_difference": int(np.abs(oracle - rank[lag, take]).max())}
    cost = benchmark_cost(model, replay, vectors, config, args.device, manifest["selected_users"])
    if state_digest() != state_before or digest(checkpoint) != config["checkpoint_sha256"]:
        raise ValueError("Frozen model changed during replay")
    for name, checksum in hashes.items():
        if digest(data / name) != checksum:
            raise ValueError(f"Dataset changed during replay: {name}")
    for name, checksum in sources.items():
        if digest(ROOT / name) != checksum:
            raise ValueError(f"Source changed during replay: {name}")
    args.output.mkdir(parents=True)
    np.savez(args.output / "replay.npz", **replay, ranks=rank)
    write(args.output / "results.json", {"config": config, "conditions": conditions, "coverage": coverage,
          "results": results, "fresh_checks": fresh_checks, "cpu_check": cpu_check, "cost": cost,
          "offline_execution": {"prepare_seconds": prepare_seconds, "unique_snapshots": len(replay["queries"]),
                                "encode_unique_seconds": encode_seconds, **timing},
          "checkpoint_sha256": digest(checkpoint), "data_manifest_sha256": digest(data / "manifest.json"),
          "frozen_state_sha256": state_before, "used_data_files_sha256": hashes, "source_sha256": sources,
          "device": args.device, "torch": torch.__version__, "numpy": np.__version__, "duckdb": duckdb.__version__,
          "python": platform.python_version(), "macos": platform.mac_ver()[0], **memory(args.device)})
    print(json.dumps({name: r["conditions"][0]["metrics"] for name, r in results.items()}, indent=2))


if __name__ == "__main__":
    main()
