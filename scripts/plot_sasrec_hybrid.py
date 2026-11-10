#!/usr/bin/env python3
"""Export fresh/stale hybrid quality curves and per-policy metrics."""

import argparse
import csv
import json
from pathlib import Path

from plot_sasrec_replay import plt,save


def curves(result,output,family):
    fig,axes = plt.subplots(2,2,figsize=(11.5,7.5),sharex=True)
    recent_label = result["config"].get("recent_label","BCE")
    for row,(split,block) in enumerate(result["results"].items()):
        policies = [p for p in block["policies"] if p["family"] == family]
        x = [p["seconds"] if family == "time" else p["events"] for p in policies]
        for col,metric in enumerate(("ndcg@10","recall@100")):
            ax = axes[row,col]
            ax.plot(x,[p["long_metrics"][metric] for p in policies],"o-",color="#c36b22",label="Cached long CE",markersize=4)
            ax.plot(x,[p["hybrid_metrics"][metric] for p in policies],"o-",color="#26906b",label="Cached long + fresh recent",markersize=4)
            ax.axhline(block["recent_metrics"][metric],color="#2864a0",linestyle="--",label=f"Fresh recent-only {recent_label}")
            ax.axhline(block["policies"][0]["long_metrics"][metric],color="#666666",linestyle=":",label="Fresh long CE")
            ax.axhline(block["policies"][0]["hybrid_metrics"][metric],color="#26906b",linestyle=":",label="Fresh hybrid")
            labels = [f"{v//3600}h" if v%3600==0 else f"{v//60}m" for v in x] if family=="time" else [str(v) for v in x]
            ax.set_xscale("log")
            ax.set_xticks(x,labels,rotation=45,ha="right",fontsize=8)
            ax.minorticks_off()
            ax.set_ylim(bottom=0)
            ax.set_title(f"{split} — {'NDCG@10' if col==0 else 'Recall@100'}",fontsize=11)
            ax.set_ylabel("Quality")
            if row==1:
                ax.set_xlabel("Long-cache on-request TTL" if family=="time" else "Observed listens per long refresh")
    fig.suptitle(f"SASRec 256 — cached long CE + fresh recent {recent_label} (2 events)",fontsize=14,y=1.01)
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc="upper center",bbox_to_anchor=(.5,.98),ncol=3,frameon=False,fontsize=9)
    fig.text(.5,-.01,"Per-policy alpha selected on CT and frozen on PV. CT quality is fitted; PV is the evaluation period.",ha="center",fontsize=9)
    fig.tight_layout(rect=(0,0,1,.91))
    save(fig,output,f"hybrid-{family}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",type=Path,required=True)
    parser.add_argument("--selection",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    result,selection = json.loads(args.input.read_text()),json.loads(args.selection.read_text())
    args.output.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False,"axes.grid":True,"grid.alpha":.2})
    rows = []
    for split,b in result["results"].items():
        for p in b["policies"]:
            row = {"split":split,"policy":p["name"],"event_threshold":p["events"],"time_threshold_seconds":p["seconds"],
                   "alpha_recent":p["alpha"],"long_refresh_fraction":p["long_refresh_fraction"],"targets":b["targets"],"users":b["users"]}
            for role,value in (("long",p["long_metrics"]),("recent",b["recent_metrics"]),("hybrid",p["hybrid_metrics"])):
                row.update({f"{role}_{m}":v for m,v in value.items()})
            for base,values in p["hybrid_minus"].items():
                for metric,d in values.items():
                    row.update({f"minus_{base}_{metric}":d["difference"],f"minus_{base}_{metric}_ci_low":d["cluster_bootstrap_95ci"][0],
                                f"minus_{base}_{metric}_ci_high":d["cluster_bootstrap_95ci"][1]})
            rows.append(row)
    with (args.output/"metrics.csv").open("w",newline="") as f:
        writer = csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n")
        writer.writeheader();writer.writerows(rows)
    with (args.output/"alpha-grid.csv").open("w",newline="") as f:
        writer = csv.writer(f,lineterminator="\n")
        writer.writerow(("policy","alpha_recent","ct_ndcg@10","ct_recall@100","selected"))
        for p in selection["policies"]:
            for q in p["curve"]:
                writer.writerow((p["name"],q["alpha"],q["metrics"]["ndcg@10"],q["metrics"]["recall@100"],q["alpha"]==p["alpha"]))
    for family in ("time","events"):
        curves(result,args.output,family)


if __name__ == "__main__":
    main()
