#!/usr/bin/env python3
"""Train the agreed large pilots locally, then select weights before test scoring."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/experiments/sasrec-large-blend.json"


def run(script, *arguments):
    command = [sys.executable, str(ROOT / "scripts" / script), *map(str, arguments)]
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True, env={**os.environ, "PYTORCH_ENABLE_MPS_FALLBACK": "0"})


def main():
    protocol = json.loads(PROTOCOL.read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilots", nargs="+", choices=list(protocol["runs"]), default=["256-large"])
    args = parser.parse_args()
    runs = {pilot: protocol["runs"][pilot] for pilot in args.pilots}
    # Sequential jobs avoid GPU contention. Completed runs retain their original checkpoints.
    for sources in runs.values():
        for role in ("recent", "long"):
            folder = ROOT / sources[role] / "train-mps"
            config = ROOT / "configs/experiments" / (folder.parent.name + ".json")
            completed = folder / "training.json"
            if completed.exists():
                metadata = json.loads(completed.read_text())
                if metadata.get("stop_reason") == "validation_patience" and metadata.get("checkpoint_sha256"):
                    if metadata["config"] != json.loads(config.read_text()):
                        raise ValueError("Completed training uses a different configuration")
                    continue
            resume = ["--resume"] if (folder / "last.pt").exists() else []
            run("run_sasrec.py", "train", "--device", "mps", "--config", config, "--output", folder, *resume)

    # Persist all requested coefficients before any new test scoring.
    for pilot in runs:
        output = ROOT / f"data/runs/sasrec-large-blend/{pilot}"
        if not (output / "selection.json").exists():
            run("blend_sasrec.py", "select", "--pilot", pilot, "--config", PROTOCOL, "--output", output)

    for pilot, sources in runs.items():
        for role, name in sources.items():
            folder = ROOT / name
            config = ROOT / "configs/experiments" / (folder.name + ".json")
            if not (folder / "evaluation-mps.json").exists():
                run("run_sasrec.py", "evaluate", "--device", "mps", "--config", config,
                    "--checkpoint", folder / "train-mps/best.pt", "--output", folder / "evaluation-mps.json")
            if pilot == "1k-large" and not (folder / "comparison-256.json").exists():
                suffix = "joint-bce" if role == "long" else "recent-2"
                run("compare_sasrec.py", "--device", "mps", "--allow-different-architecture",
                    "--checkpoint", folder / "train-mps/best.pt", "--reference-evaluation",
                    f"data/runs/sasrec-256-{suffix}/evaluation-mps.json", "--output", folder / "comparison-256.json")
        run("blend_sasrec.py", "evaluate", "--pilot", pilot, "--config", PROTOCOL,
            "--output", ROOT / f"data/runs/sasrec-large-blend/{pilot}")


if __name__ == "__main__":
    main()
