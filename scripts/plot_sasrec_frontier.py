#!/usr/bin/env python3
"""Plot full-serving quality/cost frontiers and export resource measurements."""

import argparse
import csv
import json
from pathlib import Path

from plot_sasrec_replay import plt,save
from blend_sasrec import checksum
from profile_sasrec_hybrid import budgets
from replay_sasrec_policies import frontier


def merge_bce(current,previous):
    if current["device"]!=previous["device"] or current["sources"]["long"]!=previous["sources"]["long"]:
        raise ValueError("Comparison requires the same device and long checkpoint/data")
    label = "CE50+CE2 / CE50+BCE2"
    if current["config"].get("recent_input_length") is not None:
        label += f", recent input length {current['config']['recent_input_length']}"
    result = {**current,"model_label":label,"results":{},"calibration":{}}
    for split,b in current["results"].items():
        prior = previous["results"][split]
        same = {p["name"]:p for p in b["points"] if p["family"]=="long"}
        for p in prior["points"]:
            if p["family"]=="long" and (p["metrics"]!=same[p["name"]]["metrics"] or p["requests"]!=same[p["name"]]["requests"]):
                raise ValueError("Comparison quality cohorts differ")
        # Keep one canonical long-only calibration; append only the distinct BCE branches.
        extra = [{**p,"family":p["family"]+"_bce","name":p["family"]+"_bce:"+p["policy"]}
                 for p in prior["points"] if p["family"]!="long"]
        points = b["points"]+extra
        result["results"][split] = {"points":points,"frontiers":{m:frontier(points,m) for m in ("ndcg@10","recall@100")}}
    for batch,cases in current["calibration"].items():
        extra = {"bce__"+name:p for name,p in previous["calibration"][batch].items()
                 if name not in ("long_direct","long_refresh","long_hit","encode_long","score_long")}
        result["calibration"][batch] = {**cases,**extra}
    result["ct_budget_selection"] = budgets(result["results"]["controller_train"]["points"],
        current["config"]["cost_budgets_relative_to_fresh_long"])
    return result


def plot_frontier(result,output):
    fig,axes = plt.subplots(2,2,figsize=(11.5,8),sharex=True)
    picked = {r["point"] for r in result["ct_budget_selection"] if r["point"]}
    colors = {"long":"#c36b22","recent":"#2864a0","hybrid":"#26906b",
              "hybrid_bce":"#8653a6","recent_bce":"#777777"}
    labels = {"long":"Long CE","recent":"Recent CE","hybrid":"CE50 + CE2",
              "hybrid_bce":"CE50 + BCE2","recent_bce":"Recent BCE"}
    for row,(split,b) in enumerate(result["results"].items()):
        for col,metric in enumerate(("ndcg@10","recall@100")):
            ax = axes[row,col]
            for family,color in colors.items():
                points = [p for p in b["points"] if p["family"]==family]
                if not points:
                    continue
                ax.scatter([p["cost"]["estimated_ms_per_request"] for p in points],
                           [p["metrics"][metric] for p in points],s=28 if family!="recent" else 85,
                           marker="D" if family.startswith("recent") else "o",c=color,label=labels[family],alpha=.8,zorder=3)
            front = sorted([p for p in b["points"] if p["name"] in b["frontiers"][metric]],
                           key=lambda p:p["cost"]["estimated_ms_per_request"])
            ax.plot([p["cost"]["estimated_ms_per_request"] for p in front],[p["metrics"][metric] for p in front],
                    "--",c="#666666",alpha=.55,zorder=2)
            for p in b["points"]:
                controls = {"long:always":"Fresh long","hybrid:always":"Fresh CE+CE","recent:direct":"Recent CE",
                            "hybrid_bce:always":"Fresh CE+BCE","recent_bce:direct":"Recent BCE"}
                if p["name"] in picked or p["name"] in controls:
                    c = p["cost"];x,y = c["estimated_ms_per_request"],p["metrics"][metric]
                    ax.errorbar([x],[y],xerr=[[x-c["min_repeat_ms_per_request"]],[c["max_repeat_ms_per_request"]-x]],
                                fmt="none",ecolor="#999999",elinewidth=.6,alpha=.65,zorder=1)
                    if p["name"] in picked:
                        ax.scatter([x],[y],s=85,facecolors="none",edgecolors="black",linewidths=.8,zorder=4)
                    if col==0 and p["name"] in controls:
                        ax.annotate(controls[p["name"]],
                                    (x,y),xytext=(5,7),textcoords="offset points",fontsize=8)
            ax.set_title(f"{split} — {'NDCG@10' if col==0 else 'Recall@100'}",fontsize=11)
            ax.set_ylabel("Quality");ax.set_ylim(bottom=0)
            ax.margins(x=.14)
            if row==1:
                ax.set_xlabel("Estimated full local serving cost (ms/request, B=1)")
    fig.suptitle(f"SASRec 256 — {result.get('model_label','CE50 + CE2')} — {result['device'].upper()} frontier",fontsize=14,y=1.01)
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc="upper center",bbox_to_anchor=(.5,.98),ncol=3,frameon=False)
    fig.text(.5,-.01,"Rings: CT NDCG-based budget selections. Bars: repeat cost range, not confidence intervals.\n"
             "Includes active branches, refresh/write/lookup, full-catalog scoring, normalization/mix and final top-100 CPU result.",
             ha="center",fontsize=9)
    fig.tight_layout(rect=(0,0,1,.91))
    save(fig,output,f"frontier-{result['device']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",type=Path,nargs="+",required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--bce-reference",type=Path,help="Add independently measured CE50+BCE2 points on the matching device")
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    results = [json.loads(p.read_text()) for p in args.input]
    for result in results:
        if result["config"].get("recent_input_length") is not None:
            result["model_label"] = f"CE50 + CE2, recent input length {result['config']['recent_input_length']}"
    if args.bce_reference:
        previous = json.loads(args.bce_reference.read_text())
        matches = [i for i,r in enumerate(results) if r["device"]==previous["device"]]
        if len(matches)!=1:
            raise ValueError("Exactly one matching device is required")
        i = matches[0]
        results[i] = merge_bce(results[i],previous)
        results[i]["merged_profile_sha256"] = {str(args.input[i]):checksum(args.input[i]),
                                               str(args.bce_reference):checksum(args.bce_reference)}
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False,"axes.grid":True,"grid.alpha":.2})
    resources,points = [],[]
    for result in results:
        for batch,cases in result["calibration"].items():
            for name,c in cases.items():
                resources.append({"device":result["device"],"batch_size":int(batch),"path":name,
                    "median_ms_per_request":c["median_ms_per_request"],"min_repeat":min(c["repeat_mean_ms_per_request"]),
                    "max_repeat":max(c["repeat_mean_ms_per_request"]),"repeat_mean_ms_per_request":c["repeat_mean_ms_per_request"]})
        for split,b in result["results"].items():
            for p in b["points"]:
                row = {"device":result["device"],"split":split,**{k:v for k,v in p.items() if k not in ("metrics","cost")},
                       **p["metrics"],**p["cost"]}
                row.update({f"pareto_{m}":p["name"] in names for m,names in b["frontiers"].items()})
                points.append(row)
        plot_frontier(result,args.output)
        (args.output/f"frontier-{result['device']}-data.json").write_text(json.dumps(result,indent=2)+"\n")
    for filename,rows in (("resources.csv",resources),("frontier.csv",points)):
        with (args.output/filename).open("w",newline="") as file:
            writer = csv.DictWriter(file,fieldnames=list(rows[0]),lineterminator="\n")
            writer.writeheader();writer.writerows(rows)


if __name__=="__main__":
    main()
