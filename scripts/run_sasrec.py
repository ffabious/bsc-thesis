#!/usr/bin/env python3
"""Train, benchmark or evaluate a local SASRec pilot on CPU/MPS."""

import argparse
import hashlib
import json
import os
import platform
import resource
import time
from pathlib import Path

import numpy as np
import torch

from sasrec import SASRec, batch, examples, loss, metrics, most_popular, negatives, ranks, sync

ROOT = Path(__file__).resolve().parents[1]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def memory(device):
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result = {"process_peak_rss_mib": rss / (1024 ** 2 if platform.system() == "Darwin" else 1024)}
    if device == "mps":
        result.update(mps_tensor_mib=torch.mps.current_allocated_memory() / 1024 ** 2,
                      mps_driver_mib=torch.mps.driver_allocated_memory() / 1024 ** 2)
    return result


def model_args(config):
    return {key: config[key] for key in ("length", "dim", "heads", "blocks", "dropout",
                                       "item_dim", "freeze_items") if key in config}


def item_digest(model):
    return hashlib.sha256(model.item.weight.detach().cpu().numpy().tobytes()).hexdigest()


def make_model(config, data, items):
    model = SASRec(items, **model_args(config))
    if config.get("freeze_items"):
        if bool(config.get("item_checkpoint")) == bool(config.get("item_features")):
            raise ValueError("Choose exactly one frozen item source: checkpoint or features")
        if config.get("item_checkpoint"):
            path = ROOT / config["item_checkpoint"]
            trained = json.loads(path.with_name("training.json").read_text())
            checksum = hashlib.sha256(path.read_bytes()).hexdigest()
            if checksum != config["item_checkpoint_sha256"] or checksum != trained["checkpoint_sha256"]:
                raise ValueError("Item source checkpoint checksum mismatch")
            if trained["data_manifest_sha256"] != hashlib.sha256((data / "manifest.json").read_bytes()).hexdigest():
                raise ValueError("Item source checkpoint uses a different dataset/vocabulary")
            saved = torch.load(path, map_location="cpu", weights_only=True)
            if saved["items"] != items:
                raise ValueError("Item source checkpoint has a different catalog size")
            # Transfer only the item table; the ranker retains its fresh initialization.
            values = saved["weights"]["item.weight"]
        else:
            if not json.loads((data / "manifest.json").read_text()).get("closed_audio_catalog"):
                raise ValueError("Frozen audio training requires rebuilt, fully covered splits")
            folder = ROOT / config["item_features"]
            manifest = json.loads((folder / "manifest.json").read_text())
            if manifest["missing_items"] != 0:
                raise ValueError("Frozen audio experiment requires complete coverage without fallbacks")
            if manifest["items_sha256"] != hashlib.sha256((data / "items.parquet").read_bytes()).hexdigest():
                raise ValueError("Audio vectors use a different item vocabulary")
            path = folder / "vectors.npy"
            if manifest["vectors_sha256"] != hashlib.sha256(path.read_bytes()).hexdigest():
                raise ValueError("Audio vector checksum mismatch")
            values = torch.from_numpy(np.load(path))
        if values.shape != model.item.weight.shape or not torch.isfinite(values).all():
            raise ValueError("Item vector shape/values do not match the model")
        with torch.no_grad():
            model.item.weight.copy_(values)
    return model


def optimizer_for(model, config):
    return torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                            lr=config["learning_rate"], betas=(0.9, 0.98))


def step(model, optimizer, sequence, target, negative):
    optimizer.zero_grad(set_to_none=True)
    value = loss(model, sequence, target, negative)
    value.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    optimizer.step()
    return value.detach()


def benchmark(config, data, device):
    x, y, _ = examples(data, "train", config["length"])
    count = json.loads((data / "manifest.json").read_text())["candidate_items"]
    model = make_model(config, data, count).to(device).train()
    optimizer = optimizer_for(model, config)
    rng = np.random.default_rng(config["seed"])
    take = rng.choice(len(y), config["batch_size"], replace=False)
    sequence, target = batch(x, y, take, device)
    negative = torch.as_tensor(negatives(y[take], config["negatives"], count, rng), device=device)
    for _ in range(config["benchmark_warmup"]):
        step(model, optimizer, sequence, target, negative)
    sync(device)
    timings = []
    for _ in range(config["benchmark_repeats"]):
        started = time.perf_counter()
        for _ in range(config["benchmark_steps"]):
            value = step(model, optimizer, sequence, target, negative)
        sync(device)
        timings.append((time.perf_counter() - started) / config["benchmark_steps"])
    if not torch.isfinite(value).item():
        raise ValueError("Non-finite benchmark loss")
    result = {"train_step_ms": (np.array(timings) * 1000).tolist(),
              "train_examples_per_second": config["batch_size"] / float(np.median(timings)),
              "parameters": sum(p.numel() for p in model.parameters()),
              "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
              **memory(device)}
    # Memory snapshots are separate from timed trials; the allocator may retain cached blocks.
    optimizer.zero_grad(set_to_none=True)
    probe = loss(model, sequence, target, negative)
    sync(device)
    result["memory_after_forward"] = memory(device)
    probe.backward()
    sync(device)
    result["memory_after_backward"] = memory(device)
    optimizer.step()
    sync(device)
    result["memory_after_update"] = memory(device)
    return result


def train(config, data, device, output):
    if output.exists():
        raise FileExistsError(f"Choose a new training output directory: {output}")
    output.mkdir(parents=True)
    started = time.perf_counter()
    x, y, _ = examples(data, "train", config["length"])
    vx, vy, _ = examples(data, "model_valid", config["length"])
    count = json.loads((data / "manifest.json").read_text())["candidate_items"]
    model = make_model(config, data, count).to(device)
    optimizer = optimizer_for(model, config)
    frozen_before = item_digest(model) if config.get("freeze_items") else None
    rng = np.random.default_rng(config["seed"])
    initial = metrics(ranks(model, vx, vy, device, config["eval_batch_size"]))
    result = {"train_examples": len(y), "validation_examples": len(vy), "initial_validation": initial,
              "mostpop_validation": metrics(most_popular(data, vy)), "epochs": [],
              "parameters": sum(p.numel() for p in model.parameters()),
              "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)}
    best, waiting = -1.0, 0
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        order = rng.permutation(len(y))
        total = torch.zeros((), device=device)
        sync(device)
        epoch_start = time.perf_counter()
        for start in range(0, len(y), config["batch_size"]):
            indices = order[start:start + config["batch_size"]]
            sequence, target = batch(x, y, indices, device)
            negative = torch.as_tensor(negatives(y[indices], config["negatives"], count, rng), device=device)
            total += step(model, optimizer, sequence, target, negative) * len(indices)
            if (start // config["batch_size"] + 1) % 500 == 0:
                print(f"epoch {epoch}, {start + len(indices)}/{len(y)}, "
                      f"loss={total.item() / (start + len(indices)):.4f}", flush=True)
        sync(device)
        train_seconds = time.perf_counter() - epoch_start
        mean_loss = total.item() / len(y)
        if not np.isfinite(mean_loss):
            raise ValueError("Non-finite training loss")
        validation_start = time.perf_counter()
        quality = metrics(ranks(model, vx, vy, device, config["eval_batch_size"]))
        sync(device)
        row = {"epoch": epoch, "loss": mean_loss, "train_seconds": train_seconds,
               "train_examples_per_second": len(y) / train_seconds,
               "validation_seconds": time.perf_counter() - validation_start,
               "validation": quality, **memory(device)}
        result["epochs"].append(row)
        print(json.dumps(row), flush=True)
        if quality["ndcg@10"] > best:
            best, waiting = quality["ndcg@10"], 0
            torch.save({"weights": {k: v.cpu() for k, v in model.state_dict().items()},
                        "config": config, "items": count, "epoch": epoch}, output / "best.pt")
            result["best_epoch"] = epoch
        else:
            waiting += 1
        write(output / "training.json", result)
        if waiting >= config["patience"]:
            break
    result.update(elapsed_seconds=time.perf_counter() - started,
                  checkpoint_sha256=hashlib.sha256((output / "best.pt").read_bytes()).hexdigest())
    if frozen_before:
        result.update(item_sha256_before=frozen_before, item_sha256_after=item_digest(model))
        if result["item_sha256_before"] != result["item_sha256_after"]:
            raise ValueError("Frozen item vectors changed during training")
    return result


def intervals(rank, users, seed):
    ids, inverse = np.unique(users, return_inverse=True)
    counts = np.bincount(inverse)
    draws = np.random.default_rng(seed).integers(0, len(ids), (2000, len(ids)))
    result = {}
    for name, values in {"ndcg@10": np.where(rank <= 10, 1 / np.log2(rank + 1), 0),
                         "recall@100": (rank <= 100).astype(float)}.items():
        sums = np.bincount(inverse, weights=values)
        result[name] = np.quantile(sums[draws].sum(1) / counts[draws].sum(1), [0.025, 0.975]).tolist()
    return result


def evaluate(config, data, device, checkpoint, output):
    trained = json.loads(checkpoint.with_name("training.json").read_text())
    if trained["data_manifest_sha256"] != hashlib.sha256((data / "manifest.json").read_bytes()).hexdigest():
        raise ValueError("Evaluation data differs from the training manifest")
    if trained["checkpoint_sha256"] != hashlib.sha256(checkpoint.read_bytes()).hexdigest():
        raise ValueError("Checkpoint checksum differs from the completed training run")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = SASRec(saved["items"], **model_args(saved["config"]))
    model.load_state_dict(saved["weights"])
    model.to(device).eval()
    x, y, users = examples(data, "test", model.length)
    # Warm-up is timed separately and uses no labels for optimization.
    warm = min(config["eval_batch_size"], len(y))
    ranks(model, x[:warm], y[:warm], device, config["eval_batch_size"])
    sync(device)
    started = time.perf_counter()
    rank = ranks(model, x, y, device, config["eval_batch_size"])
    sync(device)
    elapsed = time.perf_counter() - started
    np.savez(output.with_suffix(".npz"), ranks=rank, users=users, targets=y)
    pop = most_popular(data, y)
    return {"checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "checkpoint_epoch": saved["epoch"], "examples": len(y), "users": int(len(np.unique(users))),
            "catalog_items": saved["items"], "metrics": metrics(rank),
            "cluster_bootstrap_95ci": intervals(rank, users, config["seed"]),
            "mostpop_metrics": metrics(pop), "mostpop_95ci": intervals(pop, users, config["seed"]),
            "evaluation_seconds": elapsed, "examples_per_second": len(y) / elapsed,
            **memory(device)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["benchmark", "train", "evaluate"])
    parser.add_argument("--device", choices=["cpu", "mps"], required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/experiments/sasrec-pilot.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS is unavailable; CPU fallback is not a valid MPS measurement")
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") != "0":
        parser.error("Disable PYTORCH_ENABLE_MPS_FALLBACK before benchmarking")
    config = json.loads(args.config.read_text())
    data = ROOT / config["data"]
    torch.set_num_threads(config["threads"])
    torch.manual_seed(config["seed"])
    if args.device == "mps":
        torch.mps.manual_seed(config["seed"])
    if args.mode == "benchmark":
        result = benchmark(config, data, args.device)
    elif args.mode == "train":
        result = train(config, data, args.device, args.output)
    else:
        if args.checkpoint is None:
            parser.error("evaluate requires --checkpoint")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result = evaluate(config, data, args.device, args.checkpoint, args.output)
    result.update(config=config, device=args.device, torch=torch.__version__, numpy=np.__version__,
                  python=platform.python_version(), macos=platform.mac_ver()[0],
                  mps_available=torch.backends.mps.is_available(),
                  data_manifest_sha256=hashlib.sha256((data / "manifest.json").read_bytes()).hexdigest())
    destination = args.output / "training.json" if args.mode == "train" else args.output
    write(destination, result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
