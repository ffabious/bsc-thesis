#!/usr/bin/env python3
"""Submit one minimal Model Training job as the authenticated Watt user."""

import hashlib
import json
import os
import tarfile
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yt.wrapper as yt

ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT / "data/runs/sasrec-watt"
IMAGE = "registry.yandex.net/pydl/nvidia-pytorch:25.05.0"


def main():
    LOCAL.mkdir(parents=True, exist_ok=True)
    if (LOCAL / "operation.json").exists():
        raise FileExistsError("Operation already submitted; inspect operation.json instead of duplicating it")
    client = yt.YtClient(proxy="watt", token=os.environ["YT_TOKEN"])
    user = client.get_user_name()
    print(f"Authenticated as {user}", flush=True)
    if user != "k-greshnov":
        raise RuntimeError(f"Unexpected authenticated user: {user}")
    config_path = Path("configs/experiments/sasrec-1k-large-recent-cuda.json")
    config = json.loads((ROOT / config_path).read_text())
    data = Path(config["data"])
    members = [config_path, Path("scripts/sasrec.py"), Path("scripts/run_sasrec.py"),
               Path("scripts/watt_sasrec_job.py"), data / "manifest.json",
               data / "events/train.parquet", data / "events/model_valid.parquet"]
    bundle = LOCAL / "model-training.tar.gz"
    with tarfile.open(bundle, "w:gz", compresslevel=1) as archive:
        for path in members:
            archive.add(ROOT / path, arcname=str(Path("thesis") / path))
    now = datetime.now(timezone.utc)
    remote = f"//tmp/model-training-{user}-{now:%Y%m%d}-{uuid.uuid4().hex[:8]}"
    (LOCAL / "payload.json").write_text(json.dumps({"remote": remote, "bundle": str(bundle)}, indent=2) + "\n")
    print(f"Creating {remote}", flush=True)
    client.create("map_node", remote, recursive=True, attributes={
        "title": "Model Training", "expiration_time": (now + timedelta(days=7)).strftime("%Y-%m-%dT%H:%M:%SZ")})
    remote_bundle = remote + "/model-training.tar.gz"
    print("Uploading train/validation bundle", flush=True)
    with bundle.open("rb") as stream:
        client.write_file(remote_bundle, stream)
    command = ("set -eu; tar -xzf model-training.tar.gz; "
               "python3 -m pip install --no-cache-dir --disable-pip-version-check "
               "--index-url https://pypi.yandex-team.ru/simple/ "
               "duckdb==2.0.0 numpy==2.5.4 yandex-yt==0.15.13 >&2; "
               "exec python3 -u thesis/scripts/watt_sasrec_job.py")
    task = {"command": command, "job_count": 1, "gpu_limit": 1, "cpu_limit": 1,
            "set_container_cpu_limit": True, "memory_limit": 2 * 1024**3,
            "disk_space_limit": 2 * 1024**3, "docker_image": IMAGE,
            "file_paths": [remote_bundle], "environment": {
                "MODEL_TRAINING_OUTPUT": remote, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1", "PYTORCH_ENABLE_MPS_FALLBACK": "0"}}
    builder = (yt.spec_builders.VanillaSpecBuilder().title("Model Training")
               .pool("research_gpu").pool_trees(["gpu_hainan_80g"])
               .max_failed_job_count(1).max_stderr_count(3)
               .secure_vault({"YT_TOKEN": os.environ["YT_TOKEN"]})
               .tasks({"Model Training": task}))
    print("Submitting Model Training", flush=True)
    operation = client.run_operation(builder, sync=False)
    result = {"title": "Model Training", "operation_id": operation.id,
              "url": f"https://yt.yandex-team.ru/watt/operations/{operation.id}/details",
              "user": user, "cluster": "watt", "pool": "research_gpu", "tree": "gpu_hainan_80g",
              "gpu": 1, "cpu": 1, "ram_gib": 2, "disk_gib": 2, "image": IMAGE,
              "remote": remote, "created_utc": now.isoformat(), "config": config,
              "bundle_bytes": bundle.stat().st_size,
              "bundle_sha256": hashlib.sha256(bundle.read_bytes()).hexdigest()}
    (LOCAL / "operation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    try:
        main()
    except yt.YtError as error:
        # Request parameters may contain secure_vault; never dump the full exception.
        def messages(value):
            text = str(value.get("message", ""))
            for name in ("YT_TOKEN", "DOCS_TOKEN", "INTRASEARCH_TOKEN"):
                if os.environ.get(name):
                    text = text.replace(os.environ[name], "[redacted]")
            return [{"code": value.get("code"), "message": text[:2000]}] + [
                item for child in value.get("inner_errors", []) for item in messages(child)]
        print(json.dumps(messages(error.simplify()), ensure_ascii=False))
        raise SystemExit(1)
