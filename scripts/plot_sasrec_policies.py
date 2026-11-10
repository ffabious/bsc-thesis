#!/usr/bin/env python3
"""Export policy quality/cost tables and Pareto plots from a completed local replay."""

import argparse
import csv
import json
from pathlib import Path

from plot_sasrec_replay import plt, save


COLORS = {"always": "#b84238", "never": "#555555", "events": "#2864a0", "time": "#c37b18", "hybrid": "#26906b"}
MARKERS = {"always": "*", "never": "X", "events": "o", "time": "s", "hybrid": "^"}


def plot_frontiers(result, output, cost_axis):
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8), sharex="col")
    selected = {r["policy"] for r in result["selection"]["budgets"] if r["policy"]}
    for row, (split, block) in enumerate(result["results"].items()):
        rows = block["policies"]
        always = next(r for r in rows if r["name"] == "always")
        x = lambda r: r["cost"]["estimated_ms_per_request"] / always["cost"]["estimated_ms_per_request"] if cost_axis == "total" else r["refresh_fraction"]
        for col, metric in enumerate(("ndcg@10", "recall@100")):
            ax = axes[row,col]
            for family, color in COLORS.items():
                chosen = [r for r in rows if r["family"] == family]
                ax.scatter([x(r) for r in chosen], [r["metrics"][metric] for r in chosen],
                           c=color, marker=MARKERS[family], s=85 if family in ("always", "never") else 32,
                           label=family, alpha=.85, zorder=3)
            if cost_axis == "total":
                points = sorted([r for r in rows if r["name"] in block["frontiers"][metric]], key=x)
            else:
                points = sorted([r for r in rows if not any(q["refresh_fraction"] <= r["refresh_fraction"] and
                    q["metrics"][metric] >= r["metrics"][metric] and
                    (q["refresh_fraction"] < r["refresh_fraction"] or q["metrics"][metric] > r["metrics"][metric]) for q in rows)], key=x)
            ax.plot([x(r) for r in points], [r["metrics"][metric] for r in points], "--", color="#555555", alpha=.6, zorder=2)
            for r in rows:
                if r["name"] in selected:
                    ax.scatter([x(r)], [r["metrics"][metric]], s=100, facecolors="none", edgecolors="black", linewidths=1, zorder=4)
                    if col == 0:
                        first = x(r) == min(x(p) for p in rows if p["name"] in selected)
                        ax.annotate(r["name"], (x(r),r["metrics"][metric]), xytext=(6 if first else -6,7),
                                    textcoords="offset points", fontsize=8, ha="left" if first else "right")
            ax.axhline(always["metrics"][metric], color=COLORS["always"], linewidth=.8, linestyle=":")
            ax.set_title(f"{split} — {'NDCG@10' if col == 0 else 'Recall@100'}", fontsize=11)
            ax.set_ylim(bottom=0)
            ax.set_ylabel("Quality")
            if row == 1:
                ax.set_xlabel("Estimated local ms/request / always" if cost_axis == "total" else "Refreshes / requests (initialization included)")
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5,.98), ncol=5, frameon=False)
    fig.suptitle("Frozen SASRec 256 CE — actual on-request refresh policies", fontsize=14, y=1.01)
    text = "Dashed: point-estimate Pareto frontier. Rings: CT-selected policies, carried unchanged to PV."
    if cost_axis == "total":
        text += "\nCost = measured B=1 refresh/write/serve and hit/serve branches × exact replay counts + CPU decisions."
    fig.text(.5,-.01,text,ha="center",fontsize=9)
    fig.tight_layout(rect=(0,0,1,.95))
    save(fig,output,f"frontier-{cost_axis}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = json.loads(args.input.read_text())
    args.output.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False,"axes.grid":True,"grid.alpha":.2})
    rows = []
    for split, block in result["results"].items():
        always = next(r for r in block["policies"] if r["name"] == "always")
        for r in block["policies"]:
            cost = r["cost"]
            row = {"split":split, "policy":r["name"], "family":r["family"], "event_threshold":r["events"],
                   "time_threshold_seconds":r["seconds"], "targets":block["targets"], "requests":block["requests"],
                   "refreshes":r["refreshes"], "initializations":r["initializations"], "hits":r["hits"],
                   "refresh_fraction":r["refresh_fraction"], **r["metrics"], **cost,
                   "relative_cost_to_always":cost["estimated_ms_per_request"] / always["cost"]["estimated_ms_per_request"],
                   "target_fresh_fraction":r["target_fresh_fraction"]}
            for metric in ("ndcg@10","recall@100"):
                delta = r["policy_minus_always"][metric]
                row.update({f"{metric}_delta":delta["difference"], f"{metric}_delta_ci_low":delta["cluster_bootstrap_95ci"][0],
                            f"{metric}_delta_ci_high":delta["cluster_bootstrap_95ci"][1],
                            f"{metric}_pareto":r["name"] in block["frontiers"][metric]})
            for unit in ("events","seconds"):
                row.update({f"age_{unit}_{k}":v for k,v in r[f"request_age_{unit}"].items()})
            rows.append(row)
    with (args.output / "policies.csv").open("w",newline="") as file:
        writer = csv.DictWriter(file,fieldnames=list(rows[0]),lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    with (args.output / "cost.csv").open("w",newline="") as file:
        writer = csv.writer(file,lineterminator="\n")
        writer.writerow(("split","branch","median_mean_ms","repeat_mean_ms"))
        for split, branches in result["branch_costs"].items():
            for name,r in branches.items():
                writer.writerow((split,name,r["median_mean_ms"],json.dumps(r["repeat_mean_ms"])))
    for axis in ("total","refresh"):
        plot_frontiers(result,args.output,axis)


if __name__ == "__main__":
    main()
