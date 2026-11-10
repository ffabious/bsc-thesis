#!/usr/bin/env python3
"""Export replay metrics and publication-friendly PNG/SVG figures."""

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "data/.matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / "data/.plot-cache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save(fig, output, name):
    for suffix in ("png", "svg"):
        path = output / f"{name}.{suffix}"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        if suffix == "svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    plt.close(fig)


def quality_curves(result, output, unit):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for color, (split, block) in zip(("#2864a0", "#c36b22"), result["results"].items()):
        rows = [r for r in block["conditions"] if r["unit"] in ("fresh", unit)]
        x = np.array([r["lag"] for r in rows])
        label = f"{split} (n={block['examples']:,})"
        for col, metric in enumerate(("ndcg@10", "recall@100")):
            values = np.array([r["metrics"][metric] for r in rows])
            delta = np.array([-r["stale_minus_fresh"][metric]["difference"] for r in rows])
            ci = np.array([r["stale_minus_fresh"][metric]["cluster_bootstrap_95ci"] for r in rows])
            axes[0,col].plot(x, values, "o-", color=color, label=label, linewidth=1.7, markersize=4)
            axes[0,col].axhline(values[0], color=color, linestyle=":", linewidth=1, alpha=.65)
            axes[1,col].plot(x, delta, "o-", color=color, linewidth=1.7, markersize=4)
            axes[1,col].fill_between(x, -ci[:,1], -ci[:,0], color=color, alpha=.15)
    labels = ["0", "1", "5", "10", "25", "50"] if unit == "events" else ["0", "5m", "30m", "2h", "1d", "7d"]
    for col, name in enumerate(("NDCG@10", "Recall@100")):
        axes[0,col].set_title(name)
        axes[0,col].set_ylabel("Quality; dotted line = fresh")
        axes[0,col].set_ylim(bottom=0)
        axes[1,col].set_ylabel(f"Fresh − stale {name}")
        axes[1,col].axhline(0, color="#666666", linewidth=.8, linestyle="--")
        for row in range(2):
            threshold = 1 if unit == "events" else 300
            axes[row,col].set_xscale("symlog", linthresh=threshold)
            axes[row,col].set_xlim(-.07 * threshold, x[-1] * 1.15)
            axes[row,col].set_xticks(x, labels)
            axes[row,col].minorticks_off()
        axes[1,col].set_xlabel("Minimum withheld events" if unit == "events" else "Snapshot time lag")
    fig.suptitle(f"Frozen SASRec 256 CE — {'event' if unit == 'events' else 'time'} lag", fontsize=14, y=1.01)
    fig.legend(*axes[0,0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(.5,.98), ncol=2, frameon=False)
    fig.text(.5, -.01, "Same paired events at every lag. Shading: pointwise 95% user-bootstrap CI of the paired loss.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0,0,1,.92))
    save(fig, output, f"{unit}-lags")


def cost_curves(result, output):
    fig, axes = plt.subplots(1, len(result["cost"]["batches"]), figsize=(10.5, 4), squeeze=False)
    for ax, block in zip(axes[0], result["cost"]["batches"]):
        x = [r["predictions_per_refresh"] for r in block["amortized"]]
        y = [r["estimated_per_request_ms"] for r in block["amortized"]]
        component = lambda name: block["components"][name]["median_per_request_ms"]
        ax.plot(x, y, "o-", color="#6550a0", label="Hit + refresh / R")
        ax.axhline(component("fresh_e2e"), color="#c36b22", linestyle="--", label="Fresh every request")
        ax.axhline(component("cache_hit_e2e"), color="#2864a0", linestyle=":", label="Hit only; no refresh cost")
        ax.set_xscale("log")
        ax.set_xticks(x, [str(v) for v in x])
        ax.minorticks_off()
        ax.set_title(f"Batch {block['batch_size']}")
        ax.set_xlabel("Predictions per refresh (R)")
        ax.set_ylabel("Milliseconds per request")
        ax.set_ylim(bottom=0)
    fig.suptitle("Amortized local inference cost — illustration, not a policy frontier", fontsize=12, y=1.02)
    fig.legend(*axes[0,0].get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(.5,-.08), ncol=3, frameon=False)
    fig.tight_layout()
    save(fig, output, "amortized-cost")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = json.loads(args.input.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": .2})
    rows = []
    for split, block in result["results"].items():
        for r in block["conditions"]:
            row = {"split": split, "condition": r["name"], "unit": r["unit"], "lag": r["lag"],
                   "examples": block["examples"], "users": block["users"], **r["metrics"],
                   "unique_snapshots": r["unique_snapshots"], "unchanged_context_fraction": r["unchanged_context_fraction"]}
            for metric in ("ndcg@10", "recall@100"):
                delta = r["stale_minus_fresh"][metric]
                row.update({f"{metric}_delta": delta["difference"], f"{metric}_delta_ci_low": delta["cluster_bootstrap_95ci"][0],
                            f"{metric}_delta_ci_high": delta["cluster_bootstrap_95ci"][1]})
            for unit in ("events", "seconds"):
                row.update({f"age_{unit}_{key}": value for key, value in r[f"age_{unit}"].items()})
            rows.append(row)
    with (args.output / "quality.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    with (args.output / "cost.csv").open("w", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(("batch_size", "component", "median_batch_ms", "median_per_request_ms", "repeat_batch_ms"))
        for block in result["cost"]["batches"]:
            for name, row in block["components"].items():
                writer.writerow((block["batch_size"], name, row["median_batch_ms"], row["median_per_request_ms"], json.dumps(row["batch_ms"])))
    for unit in ("events", "seconds"):
        quality_curves(result, args.output, unit)
    cost_curves(result, args.output)


if __name__ == "__main__":
    main()
