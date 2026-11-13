#!/usr/bin/env python3
"""Compare the frozen policy grid, CE42 and CT-selected refresh controllers."""

import csv
import json
import shutil
from pathlib import Path

import numpy as np

from blend_sasrec import checksum
from compare_sasrec import paired_intervals
from plot_sasrec_replay import plt,save
from replay_encoder_pairs import paired_effect,terms
from replay_sasrec_policies import frontier
from run_sasrec import ROOT,write


def read(path):return json.loads(path.read_text())


def csv_file(path,rows):
    with path.open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n");writer.writeheader();writer.writerows(rows)


def cost_matched_mixture(rows,cap):
    """Best CT expected NDCG of two fixed user-level policies at a cost ceiling."""
    best=None
    for left in rows:
        a=left["cost"]["estimated_ms_per_request"]
        if a>cap:continue
        for right in rows:
            b=right["cost"]["estimated_ms_per_request"]
            if b<cap or b<=a:continue
            p=(cap-a)/(b-a);score=(1-p)*left["metrics"]["ndcg@10"]+p*right["metrics"]["ndcg@10"]
            if best is None or score>best["ct_ndcg"]:
                best={"left":left["name"],"right":right["name"],"right_probability":p,"ct_ndcg":score,"ct_cost_ms":cap}
    return best


def main():
    root=ROOT/"data/runs/student43-controller-calibrated";student=ROOT/"data/runs/student43-policies";ce=ROOT/"data/runs/sasrec-256-policies-wide"
    out=ROOT/"reports/experiments/student43-maintenance";out.mkdir(parents=True,exist_ok=True)
    c=read(root/"results.json");s=read(root/"policy-results.json");b=read(ce/"results.json")
    v=dict(np.load(student/"replay.npz"));old=dict(np.load(ce/"replay.npz"));new=dict(np.load(root/"replay.npz"))
    for key in old:
        if key!="ranks":assert np.array_equal(old[key],v[key]),key
    for key in ["targets","target_request","target_event_ids"]:assert np.array_equal(new[key],v[key]),key
    assert s["policies"]==b["policies"] and c["checkpoint_sha256"]==s["checkpoint_sha256"]
    assert checksum(root/"selection.json")==c["selection_sha256"]
    result={"ce42_comparisons":{},"controller_comparisons":{},"combined_frontiers":{},"mixture_reference":{},"no_test_events_read":True}
    ce_rows=[];controller_rows=[];mixture_rows=[]
    for i,split in enumerate(["controller_train","policy_valid"],1):
        take=v["split_idx"][v["target_request"]]==i;users=v["users"][v["target_request"]][take]
        result["ce42_comparisons"][split]={}
        for j,row in enumerate(s["results"][split]["policies"]):
            previous=b["results"][split]["policies"][j];assert previous["name"]==row["name"]
            effect=paired_intervals(v["ranks"][j,take],old["ranks"][j,take],users,42)
            result["ce42_comparisons"][split][row["name"]]=effect
            entry={"split":split,"policy":row["name"],"refresh_fraction":row["refresh_fraction"]}
            for metric in ["ndcg@10","recall@100","hit@1"]:
                entry[f"ce42_{metric}"]=previous["metrics"][metric];entry[f"student43_{metric}"]=row["metrics"][metric]
                if metric in effect:
                    e=effect[metric];entry.update({f"delta_{metric}":e["difference"],f"ci_low_{metric}":e["cluster_bootstrap_95ci"][0],f"ci_high_{metric}":e["cluster_bootstrap_95ci"][1]})
            ce_rows.append(entry)
        always=s["results"][split]["policies"][0];result["controller_comparisons"][split]=[]
        for selected in c["selection"]["budgets"]:
            if not selected["controller"]:continue
            controller=next(p for p in c["results"][split] if p["name"]==selected["controller"])
            baseline=next(p for p in s["results"][split]["policies"] if p["name"]==selected["matched_simple"])
            k=list(new["names"]).index(controller["name"]);j=next(j for j,p in enumerate(s["policies"]) if p["name"]==baseline["name"])
            paired=paired_intervals(new["ranks"][k,take],v["ranks"][j,take],users,42)
            dominating={metric:[p["name"] for p in s["results"][split]["policies"] if
                p["cost"]["estimated_ms_per_request"]<=controller["cost"]["estimated_ms_per_request"] and p["metrics"][metric]>=controller["metrics"][metric]]
                for metric in ["ndcg@10","recall@100"]}
            row={"budget":selected["relative_budget"],"controller":controller,"baseline":baseline,"paired":paired,"dominating_simple_policies":dominating}
            result["controller_comparisons"][split].append(row)
            entry={"split":split,"ct_budget":selected["relative_budget"],"controller":controller["name"],"baseline":baseline["name"]}
            for kind,p in [("controller",controller),("baseline",baseline)]:
                entry.update({f"{kind}_cost_ms":p["cost"]["estimated_ms_per_request"],f"{kind}_relative_cost":p["cost"]["estimated_ms_per_request"]/always["cost"]["estimated_ms_per_request"],
                    f"{kind}_refresh_fraction":p["refresh_fraction"],**{f"{kind}_{key}":value for key,value in p["metrics"].items()}})
            entry["controller_more_cost_percent"]=100*(entry["controller_cost_ms"]/entry["baseline_cost_ms"]-1)
            for key,e in paired.items():entry.update({f"delta_{key}":e["difference"],f"ci_low_{key}":e["cluster_bootstrap_95ci"][0],f"ci_high_{key}":e["cluster_bootstrap_95ci"][1]})
            controller_rows.append(entry)
            mix=cost_matched_mixture(c["selection"]["tuning_simple_rows"],selected["tuning_controller"]["cost"]["estimated_ms_per_request"])
            assert mix is not None
            left=next(j for j,p in enumerate(s["policies"]) if p["name"]==mix["left"]);right=next(j for j,p in enumerate(s["policies"]) if p["name"]==mix["right"])
            p=mix["right_probability"];left_row=s["results"][split]["policies"][left];right_row=s["results"][split]["policies"][right]
            expected={key:(1-p)*left_row["metrics"][key]+p*right_row["metrics"][key] for key in ["ndcg@10","recall@100","hit@1"]}
            effect={key:paired_effect((values-(1-p)*terms(v["ranks"][left,take])[key]-p*terms(v["ranks"][right,take])[key])[None],users,42)[0]
                    for key,values in terms(new["ranks"][k,take]).items()}
            cost=(1-p)*left_row["cost"]["estimated_ms_per_request"]+p*right_row["cost"]["estimated_ms_per_request"]
            row["expected_mixture"]={**mix,"metrics":expected,"cost_ms":cost,"paired_controller_minus_expectation":effect,
                "scope":"CT-chosen independent per-user randomized policy in expectation; assignment overhead and realized assignment variance not measured"}
            flat={"split":split,"ct_budget":selected["relative_budget"],"controller":controller["name"],**mix,"mixture_cost_ms":cost,
                "controller_cost_ms":controller["cost"]["estimated_ms_per_request"]}
            for key,e in effect.items():flat.update({f"mixture_{key}":expected[key],f"controller_{key}":controller["metrics"][key],
                f"delta_{key}":e["difference"],f"ci_low_{key}":e["cluster_bootstrap_95ci"][0],f"ci_high_{key}":e["cluster_bootstrap_95ci"][1]})
            mixture_rows.append(flat)
        combined=s["results"][split]["policies"]+c["results"][split]
        result["combined_frontiers"][split]={metric:frontier(combined,metric) for metric in ["ndcg@10","recall@100"]}
    csv_file(out/"compare-ce42.csv",ce_rows);csv_file(out/"controller-comparison.csv",controller_rows);csv_file(out/"mixture-reference.csv",mixture_rows)
    result["sources_sha256"]={str(p.relative_to(ROOT)):checksum(p) for p in [Path(__file__).resolve(),root/"results.json",root/"policy-results.json",root/"replay.npz",root/"selection.json",student/"replay.npz",ce/"replay.npz",ce/"results.json"]}
    write(out/"comparison.json",result)
    for name,path in [("policy-results.json",root/"policy-results.json"),("controller-results.json",root/"results.json"),
        ("model.json",root/"model.json"),("controller-selection.json",root/"selection.json"),("policy-selection.json",root/"policy-selection.json"),
        ("policy-config.json",ROOT/"configs/experiments/student43-policies.json"),("controller-config.json",ROOT/"configs/experiments/student43-controller.json")]:shutil.copyfile(path,out/name)
    calibration=[];mixed=[]
    for split,block in c["calibration"].items():
        for name,branches in {"plain":block["plain"],**block["controllers"]}.items():
            for branch,row in branches.items():calibration.append({"split":split,"path":name,"branch":branch,"median_mean_ms":row["median_mean_ms"],"repeat_mean_ms":json.dumps(row["repeat_mean_ms"])})
        for name,row in c["mixed_workload_checks"][split]["cases"].items():mixed.append({"split":split,"path":name,"measured_ms":row["median_mean_ms"],"predicted_ms":row["predicted_sample_ms"],"error_percent":row["measured_minus_predicted_percent"]})
    csv_file(out/"calibration.csv",calibration);csv_file(out/"mixed-validation.csv",mixed)
    plt.rcParams.update({"axes.spines.top":False,"axes.spines.right":False,"axes.grid":True,"grid.alpha":.2})
    fig,axes=plt.subplots(2,2,figsize=(11,7))
    for i,split in enumerate(["controller_train","policy_valid"]):
        plain=s["results"][split]["policies"];control=c["results"][split];always=plain[0]["cost"]["estimated_ms_per_request"]
        for j,metric in enumerate(["ndcg@10","recall@100"]):
            ax=axes[i,j];cost=lambda p:p["cost"]["estimated_ms_per_request"]/always
            ax.scatter([cost(p) for p in plain],[p["metrics"][metric] for p in plain],s=20,color="#838383",alpha=.6,label="100 simple policies")
            joined=plain+control;on=set(result["combined_frontiers"][split][metric]);points=sorted([p for p in joined if p["name"] in on],key=cost)
            ax.plot([cost(p) for p in points],[p["metrics"][metric] for p in points],"--",color="#777777",linewidth=1)
            for row in result["controller_comparisons"][split]:
                p=row["controller"];b=row["baseline"]
                ax.scatter([cost(p)],[p["metrics"][metric]],marker="*",s=140,color="#168469",label="Controller (CT-selected)" if row is result["controller_comparisons"][split][0] else None,zorder=5)
                ax.scatter([cost(b)],[b["metrics"][metric]],s=70,facecolors="none",edgecolors="#bd6a23",linewidths=1.2,label="CT-matched simple comparator" if row is result["controller_comparisons"][split][0] else None)
                ax.annotate(f"{row['budget']:.0%}",(cost(p),p["metrics"][metric]),xytext=(4,6),textcoords="offset points",fontsize=8)
            ax.set(title=f"{split} — {metric}",xlabel="Full local cost / student43 always",ylabel="Quality",ylim=(0,None))
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc="upper center",ncol=3,frameon=False)
    fig.text(.5,.005,"Labels: CT budget ceilings. Actual costs can differ. Cost includes complete inline controller CPU + MPS branches.",ha="center",fontsize=9)
    fig.tight_layout(rect=(0,.035,1,.94));save(fig,out,"controller-frontier")
    print(json.dumps({"pv_controller_comparison":result["controller_comparisons"]["policy_valid"]},indent=2))


if __name__=="__main__":main()
