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

from sasrec import SASRec, batch, examples, loss, mask_context, metrics, most_popular, negatives, ranks, sync

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
                                       "item_dim", "freeze_items", "ff_dim") if key in config}


def input_examples(data, split, config):
    x, y, users = examples(data, split, config["length"])
    return mask_context(x, config.get("context_keep")), y, users


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


def step(model, optimizer, sequence, target, negative, reduction="separate_mean", objective="bce"):
    optimizer.zero_grad(set_to_none=True)
    value = loss(model, sequence, target, negative, reduction, objective)
    value.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    optimizer.step()
    return value.detach()


def benchmark(config, data, device):
    x, y, _ = input_examples(data, "train", config)
    count = json.loads((data / "manifest.json").read_text())["candidate_items"]
    model = make_model(config, data, count).to(device).train()
    optimizer = optimizer_for(model, config)
    reduction = config.get("loss_reduction", "separate_mean")
    rng = np.random.default_rng(config["seed"])
    def measured_step():
        take = rng.integers(len(y), size=config["batch_size"])
        sequence, target = batch(x, y, take, device)
        negative = torch.as_tensor(negatives(y[take], config["negatives"], count, rng), device=device)
        return step(model, optimizer, sequence, target, negative, reduction, config.get("objective", "bce"))
    for _ in range(config["benchmark_warmup"]):
        measured_step()
    sync(device)
    timings = []
    for _ in range(config["benchmark_repeats"]):
        started = time.perf_counter()
        for _ in range(config["benchmark_steps"]):
            value = measured_step()
        sync(device)
        timings.append((time.perf_counter() - started) / config["benchmark_steps"])
    if not torch.isfinite(value).item():
        raise ValueError("Non-finite benchmark loss")
    result = {"train_step_ms": (np.array(timings) * 1000).tolist(),
              "train_examples_per_second": config["batch_size"] / float(np.median(timings)),
              "parameters": sum(p.numel() for p in model.parameters()),
              "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
              **memory(device)}
    result.update(train_examples=len(y), batch_size=config["batch_size"],
                  timing_includes="random prefix gathering, negatives, forward, backward, clipping, Adam",
                  estimated_train_epoch_seconds=len(y) / result["train_examples_per_second"])
    # Memory snapshots are separate from timed trials; the allocator may retain cached blocks.
    optimizer.zero_grad(set_to_none=True)
    take = rng.integers(len(y), size=config["batch_size"])
    sequence, target = batch(x, y, take, device)
    negative = torch.as_tensor(negatives(y[take], config["negatives"], count, rng), device=device)
    probe = loss(model, sequence, target, negative, reduction, config.get("objective", "bce"))
    sync(device)
    result["memory_after_forward"] = memory(device)
    probe.backward()
    sync(device)
    result["memory_after_backward"] = memory(device)
    optimizer.step()
    sync(device)
    result["memory_after_update"] = memory(device)
    probe_count = min(len(y), config["eval_batch_size"] * 4)
    take = rng.integers(len(y), size=probe_count)
    ranks(model, x[take[:config["eval_batch_size"]]], y[take[:config["eval_batch_size"]]],
          device, config["eval_batch_size"])
    sync(device)
    started = time.perf_counter()
    ranks(model, x[take], y[take], device, config["eval_batch_size"])
    sync(device)
    result.update(full_catalog_scoring_examples=probe_count,
                  full_catalog_scoring_examples_per_second=probe_count / (time.perf_counter() - started),
                  memory_after_scoring=memory(device))
    return result


def save_checkpoint(path, value):
    temporary = path.with_suffix(".part")
    torch.save(value, temporary)
    temporary.replace(path)


def train(config, data, device, output, resume=False):
    if output.exists() and not resume:
        raise FileExistsError(f"Choose a new training output directory: {output}")
    output.mkdir(parents=True, exist_ok=resume)
    started = time.perf_counter()
    manifest_hash = hashlib.sha256((data / "manifest.json").read_bytes()).hexdigest()
    x, y, _ = input_examples(data, "train", config)
    vx, vy, _ = input_examples(data, "model_valid", config)
    count = json.loads((data / "manifest.json").read_text())["candidate_items"]
    model = make_model(config, data, count).to(device)
    optimizer = optimizer_for(model, config)
    reduction = config.get("loss_reduction", "separate_mean")
    frozen_before = item_digest(model) if config.get("freeze_items") else None
    rng = np.random.default_rng(config["seed"])
    elapsed_before, partial_seconds, cursor, order = 0., 0., 0, None
    best, waiting, epoch = -1.0, 0, 0
    total = torch.zeros((), device=device)
    if resume:
        saved = torch.load(output / "last.pt", map_location="cpu", weights_only=True)
        if (saved["config"] != config or saved["data_manifest_sha256"] != manifest_hash
                or saved["device"] != device):
            raise ValueError("Resume configuration, device or dataset manifest changed")
        model.load_state_dict(saved["weights"])
        optimizer.load_state_dict(saved["optimizer"])
        rng.bit_generator.state = saved["numpy_rng"]
        torch.set_rng_state(saved["torch_rng"])
        if device == "mps":
            torch.mps.set_rng_state(saved["mps_rng"])
        result = saved["result"]
        best, waiting, epoch = saved["best"], saved["waiting"], saved["epoch"]
        elapsed_before, partial_seconds, cursor = saved["elapsed_seconds"], saved["partial_seconds"], saved["cursor"]
        order = saved["order"].numpy() if saved["order"] is not None else None
        total.fill_(saved["total_loss"])
        del saved
        print(f"Resumed epoch {epoch}, cursor {cursor}, best {best:.6f}", flush=True)
    else:
        initial = metrics(ranks(model, vx, vy, device, config["eval_batch_size"]))
        result = {"train_examples": len(y), "validation_examples": len(vy), "initial_validation": initial,
                  "mostpop_validation": metrics(most_popular(data, vy)), "epochs": [],
                  "parameters": sum(p.numel() for p in model.parameters()),
                  "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)}

    def save_progress(seconds):
        sync(device)
        save_checkpoint(output / "last.pt", {
            "config": config, "data_manifest_sha256": manifest_hash, "device": device, "result": result,
            "weights": model.state_dict(), "optimizer": optimizer.state_dict(),
            "numpy_rng": rng.bit_generator.state, "torch_rng": torch.get_rng_state(),
            "mps_rng": torch.mps.get_rng_state() if device == "mps" else None,
            "best": best, "waiting": waiting, "epoch": epoch, "cursor": cursor,
            "order": torch.from_numpy(order) if order is not None else None,
            "total_loss": total.item(), "partial_seconds": seconds,
            "elapsed_seconds": elapsed_before + time.perf_counter() - started})

    while "stop_reason" not in result and (config["epochs"] is None or epoch < config["epochs"] or order is not None):
        if order is None:
            epoch += 1
            order = rng.permutation(len(y))
            cursor, partial_seconds = 0, 0.
            total.zero_()
        model.train()
        sync(device)
        epoch_start = time.perf_counter()
        for start in range(cursor, len(y), config["batch_size"]):
            indices = order[start:start + config["batch_size"]]
            sequence, target = batch(x, y, indices, device)
            negative = torch.as_tensor(negatives(y[indices], config["negatives"], count, rng), device=device)
            total += step(model, optimizer, sequence, target, negative, reduction,
                          config.get("objective", "bce")) * len(indices)
            cursor = start + len(indices)
            if (start // config["batch_size"] + 1) % 500 == 0:
                print(f"epoch {epoch}, {start + len(indices)}/{len(y)}, "
                      f"loss={total.item() / (start + len(indices)):.4f}", flush=True)
            every = config.get("checkpoint_steps", 0)
            if every and cursor < len(y) and (start // config["batch_size"] + 1) % every == 0:
                save_progress(partial_seconds + time.perf_counter() - epoch_start)
        sync(device)
        train_seconds = partial_seconds + time.perf_counter() - epoch_start
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
            save_checkpoint(output / "best.pt", {"weights": {k: v.cpu() for k, v in model.state_dict().items()},
                            "config": config, "items": count, "epoch": epoch})
            result["best_epoch"] = epoch
        else:
            waiting += 1
        order, cursor = None, 0
        if waiting >= config["patience"]:
            result["stop_reason"] = "validation_patience"
        elif config["epochs"] is not None and epoch >= config["epochs"]:
            result["stop_reason"] = "epoch_limit"
        write(output / "training.json", result)
        save_progress(0.)
    result.setdefault("stop_reason", "epoch_limit")
    result.update(elapsed_seconds=elapsed_before + time.perf_counter() - started,
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


def evaluate(config, data, device, checkpoint, output, split="test"):
    if split not in ("model_valid", "test"):
        raise ValueError("Evaluation split must be model_valid or test")
    trained = json.loads(checkpoint.with_name("training.json").read_text())
    if trained["data_manifest_sha256"] != hashlib.sha256((data / "manifest.json").read_bytes()).hexdigest():
        raise ValueError("Evaluation data differs from the training manifest")
    if trained["checkpoint_sha256"] != hashlib.sha256(checkpoint.read_bytes()).hexdigest():
        raise ValueError("Checkpoint checksum differs from the completed training run")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = SASRec(saved["items"], **model_args(saved["config"]))
    model.load_state_dict(saved["weights"])
    model.to(device).eval()
    if config.get("context_keep") != saved["config"].get("context_keep"):
        raise ValueError("Evaluation context_keep differs from the checkpoint configuration")
    x, y, users = input_examples(data, split, saved["config"])
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
    return {"split": split, "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
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
    parser.add_argument("--split", choices=["model_valid", "test"], default="test",
                        help="Evaluation only; model_valid keeps recipe selection separate from test")
    parser.add_argument("--resume", action="store_true", help="Resume last.pt, including optimizer and RNG states")
    args = parser.parse_args()
    if args.mode != "evaluate" and args.split != "test":
        parser.error("--split is only valid for evaluation")
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
        result = train(config, data, args.device, args.output, args.resume)
    else:
        if args.checkpoint is None:
            parser.error("evaluate requires --checkpoint")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result = evaluate(config, data, args.device, args.checkpoint, args.output, args.split)
    result.update(config=config, device=args.device, torch=torch.__version__, numpy=np.__version__,
                  python=platform.python_version(), macos=platform.mac_ver()[0],
                  mps_available=torch.backends.mps.is_available(),
                  data_manifest_sha256=hashlib.sha256((data / "manifest.json").read_bytes()).hexdigest())
    destination = args.output / "training.json" if args.mode == "train" else args.output
    write(destination, result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
