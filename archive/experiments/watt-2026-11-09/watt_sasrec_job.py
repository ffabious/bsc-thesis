#!/usr/bin/env python3
"""Single-GPU Watt entrypoint; persist checkpoints across preemptions."""

import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import torch
import yt.wrapper as yt

import run_sasrec

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/experiments/sasrec-1k-large-recent-cuda.json"


def main():
    sys.stdout = sys.stderr  # YT retains job stderr for monitoring.
    token = os.environ.pop("YT_SECURE_VAULT_YT_TOKEN")
    client = yt.YtClient(proxy="watt", token=token)
    remote = os.environ["MODEL_TRAINING_OUTPUT"]
    output = ROOT / "data/runs/sasrec-1k-large-recent/train-cuda"
    output.parent.mkdir(parents=True, exist_ok=True)
    config = json.loads(CONFIG.read_text())
    data = ROOT / config["data"]
    manifest = json.loads((data / "manifest.json").read_text())
    for entry in manifest["files"]:
        if entry["path"] in ("events/train.parquet", "events/model_valid.parquet"):
            with (data / entry["path"]).open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != entry["sha256"]:
                    raise ValueError("Input parquet checksum mismatch")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one visible CUDA GPU is required")
    print(json.dumps({"title": "Model Training", "user": client.get_user_name(),
                      "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
                      "cuda": torch.version.cuda, "precision": "FP32, TF32 disabled"}), flush=True)

    def upload(path):
        with path.open("rb") as stream:
            client.write_file(remote + "/" + path.name, stream)

    benchmark = output.parent / "benchmark-cuda.json"
    sys.argv = ["run_sasrec.py", "benchmark", "--device", "cuda", "--config", str(CONFIG), "--output", str(benchmark)]
    run_sasrec.main()
    upload(benchmark)
    resume = client.exists(remote + "/last.pt")
    if resume:
        output.mkdir(exist_ok=True)
        with (output / "last.pt").open("wb") as stream:
            shutil.copyfileobj(client.read_file(remote + "/last.pt"), stream, length=4 * 1024**2)
        if client.exists(remote + "/best.pt"):
            with (output / "best.pt").open("wb") as stream:
                shutil.copyfileobj(client.read_file(remote + "/best.pt"), stream, length=4 * 1024**2)
    original_save = run_sasrec.save_checkpoint

    def save_and_upload(path, value):
        original_save(path, value)
        upload(path)
        if (output / "training.json").exists():
            upload(output / "training.json")

    run_sasrec.save_checkpoint = save_and_upload
    sys.argv = ["run_sasrec.py", "train", "--device", "cuda", "--config", str(CONFIG), "--output", str(output)]
    if resume:
        sys.argv.append("--resume")
    run_sasrec.main()
    upload(output / "training.json")
    print("Model Training completed", flush=True)


if __name__ == "__main__":
    main()
