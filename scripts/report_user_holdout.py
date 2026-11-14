#!/usr/bin/env python3
"""Report the frozen final cohort with its prespecified primary comparison family."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from blend_sasrec import checksum
from compare_sasrec import paired_intervals
from final_holdout import verified_protocol
from plot_sasrec_replay import plt,save
from report_controller_models import terms
from report_student_maintenance import csv_file
from run_sasrec import ROOT,write


def read(path):return json.loads(path.read_text())


def main():
    protocol,commit=verified_protocol();config=protocol["config"];root=ROOT/config["output"];source=root/"evaluation";out=ROOT/config["report"]
    run=read(source/"results.json");quality=read(source/"quality.json");v=dict(np.load(source/"replay.npz"));data=read(root/"dataset/manifest.json")
    assert run["freeze_commit"]==commit and run["protocol_sha256"]==checksum(out/"protocol.json")
    for metadata in [run,data]:
        for path,h in metadata["source_sha256"].items():assert checksum(ROOT/path)==h,path
    assert checksum(source/"replay.npz")==run["replay_sha256"] and run["no_training_or_selection_on_holdout"]
    protected={source/name:checksum(source/name) for name in ["replay.npz","results.json","quality.json"]}
    guard=subprocess.run([sys.executable,str(ROOT/"scripts/replay_user_holdout.py")],cwd=ROOT,capture_output=True,text=True,timeout=20)
    assert guard.returncode==1 and "Final evaluation already exists" in guard.stderr
    assert all(checksum(path)==h for path,h in protected.items())
    users=v["users"][v["target_request"]];ids,inverse=np.unique(users,return_inverse=True);counts=np.bincount(inverse)
    draws=np.random.default_rng(config["bootstrap_seed"]).integers(0,len(ids),(config["bootstrap_repeats"],len(ids)))
    weights=np.array([np.bincount(row,minlength=len(ids)) for row in draws]);den=weights@counts
    checked=0
    def effect(delta,level):
        nonlocal checked
        sums=np.bincount(inverse,weights=delta);replicates=(weights@sums)/den
        quantiles=[(1-level)/2,(1+level)/2];ci=np.quantile(replicates,quantiles)
        independent=sums[draws].sum(1)/counts[draws].sum(1)
        np.testing.assert_allclose(ci,np.quantile(independent,quantiles),rtol=0,atol=1e-14);checked+=1
        return {"difference":float(delta.mean()),"confidence_level":level,"cluster_bootstrap_ci":ci.tolist()}
    result={"freeze_commit":commit,"cohort":protocol["cohort"],"coverage":data["coverage"]["test"],"requests":quality["requests"],"targets":len(users),
        "active_users":len(np.unique(v["users"])),"positive_users":len(ids),"primary":[],"secondary_mixtures":[],"no_refit_or_selection":True}
    fresh=terms(v["ranks"][0]);rows=[]
    for j,name in enumerate(v["names"]):
        metric=terms(v["ranks"][j]);loss={key:effect(metric[key]-fresh[key],.95) for key in metric}
        row={"name":str(name),"metrics":quality["policies"][j]["metrics"],"refresh_fraction":float(v["refresh"][j].mean()),"paired_minus_fresh":loss}
        rows.append(row)
    result["policies"]=rows
    for pick in protocol["controllers"]:
        name=f"controller/{pick['name']}";j=list(v["names"]).index(name);b=list(v["names"]).index(pick["matched_simple"])
        values=terms(v["ranks"][j]);baseline=terms(v["ranks"][b]);effects={}
        for metric in values:effects[metric]=effect(values[metric]-baseline[metric],.975 if metric=="ndcg@10" else .95)
        # Existing rank-bootstrap implementation independently checks the secondary95% intervals.
        previous=paired_intervals(v["ranks"][j],v["ranks"][b],users,config["bootstrap_seed"])
        for metric,old in previous.items():np.testing.assert_allclose(effect(values[metric]-baseline[metric],.95)["cluster_bootstrap_ci"],old["cluster_bootstrap_95ci"],rtol=0,atol=1e-14)
        result["primary"].append({"ct_budget":pick["ct_budget"],"controller":rows[j],"simple":rows[b],"paired":effects,
            "primary_family":"Two NDCG comparisons;97.5% per interval, Bonferroni family2. Recall secondary95%."})
        mix=pick["expected_mixture"]
        left=list(v["names"]).index(mix["left"]);right=list(v["names"]).index(mix["right"]);p=mix["right_probability"]
        expected={metric:(1-p)*value+p*terms(v["ranks"][right])[metric] for metric,value in terms(v["ranks"][left]).items()}
        result["secondary_mixtures"].append({"ct_budget":pick["ct_budget"],**mix,"paired":{metric:effect(value-expected[metric],.95) for metric,value in values.items()},
            "scope":"Fixed development-CT expected user assignment; actual new-cohort cost can differ; assignment variance/cost not measured"})
    resources=[];primary=[]
    for regime,block in run["results"].items():
        always=next(row for row in block["policies"] if row["name"]=="always")["cost"]["estimated_ms_per_request"]
        for row in block["policies"]:
            sample=run["measurements"]["cases"][f"{regime}/{row['name']}"];paired=run["measurements"]["cases"][f"vector/{row['name']}"]
            for key in ["sample_requests","sample_refreshes","unique_sample_requests"]:assert sample[key]==paired[key]
            np.testing.assert_allclose(row["cost"]["estimated_ms_per_request"],np.mean(sample["repeat_mean_ms"]),rtol=0,atol=1e-14)
            assert abs(row["refresh_fraction"]-row["sample_refresh_fraction"])<=.5/sample["sample_requests"]+1e-12
            resources.append({"regime":regime,"policy":row["name"],"refresh_fraction":row["refresh_fraction"],"sample_refresh_fraction":row["sample_refresh_fraction"],
                "cost_ms":row["cost"]["estimated_ms_per_request"],"relative_cost":row["cost"]["estimated_ms_per_request"]/always,**row["metrics"]})
        for comparison in result["primary"]:
            c=next(row for row in resources if row["regime"]==regime and row["policy"]==comparison["controller"]["name"])
            b=next(row for row in resources if row["regime"]==regime and row["policy"]==comparison["simple"]["name"])
            flat={"regime":regime,"ct_budget":comparison["ct_budget"],"controller":c["policy"],"simple":b["policy"],
                "controller_cost_ms":c["cost_ms"],"simple_cost_ms":b["cost_ms"],"controller_relative_cost":c["relative_cost"],"simple_relative_cost":b["relative_cost"],
                "controller_more_cost_percent":100*(c["cost_ms"]/b["cost_ms"]-1)}
            for metric,e in comparison["paired"].items():flat.update({f"controller_{metric}":c[metric],f"simple_{metric}":b[metric],
                f"delta_{metric}":e["difference"],f"ci_level_{metric}":e["confidence_level"],f"ci_low_{metric}":e["cluster_bootstrap_ci"][0],f"ci_high_{metric}":e["cluster_bootstrap_ci"][1]})
            primary.append(flat)
    assert np.all(v["users"][v["refs"]]==v["users"][None]) and np.all(v["last_input_timestamp"][v["refs"]]<v["timestamp"][None])
    result["verification"]={"independent_bootstrap_intervals":checked,"primary_family_frozen":True,"paired_direct_serving_cases":len(resources),"source_data_checkpoint_hashes":True,
        "frozen_protocol_committed_before_event_preparation":data["freeze_commit"]==commit,"same_events_and_catalog_per_comparison":True,
        "causal_cache_refs":True,"cpu_oracle":run["cpu_oracle"],"encoder_frozen":run["frozen_state_unchanged"],"repeat_evaluation_guard_preserves_outputs":True}
    result["source_sha256"]={str(path.relative_to(ROOT)):checksum(path) for path in [Path(__file__).resolve(),source/"results.json",source/"quality.json",source/"replay.npz",out/"protocol.json",root/"dataset/manifest.json"]}
    result["execution"]={"data_seconds":data["elapsed_seconds"],"evaluation_seconds":run["total_seconds"],"memory":run["memory"]}
    write(out/"comparison.json",result);csv_file(out/"primary-comparison.csv",primary);csv_file(out/"policies.csv",resources)
    csv_file(out/"quality.csv",[{"policy":row["name"],"refresh_fraction":row["refresh_fraction"],**row["metrics"]} for row in rows])
    shutil.copyfile(source/"results.json",out/"serving-results.json");shutil.copyfile(source/"quality.json",out/"quality-results.json")
    fig,axes=plt.subplots(1,2,figsize=(11,4.8))
    denominator=next(row["cost_ms"] for row in resources if row["regime"]=="vector" and row["policy"]=="always")
    for ax,metric in zip(axes,["ndcg@10","recall@100"]):
        for regime,color in [("vector","#888888"),("list","#168469")]:
            block=run["results"][regime];policies=block["policies"]
            ax.scatter([p["cost"]["estimated_ms_per_request"]/denominator for p in policies],[p["metrics"][metric] for p in policies],s=18,c=color,alpha=.5,label=regime)
            points=sorted([p for p in policies if p["name"] in block["frontiers"][metric]],key=lambda p:p["cost"]["estimated_ms_per_request"])
            ax.plot([p["cost"]["estimated_ms_per_request"]/denominator for p in points],[p["metrics"][metric] for p in points],c=color,linewidth=1)
            for p in [p for p in policies if p["name"].startswith("controller/")]:ax.scatter(p["cost"]["estimated_ms_per_request"]/denominator,p["metrics"][metric],c=color,marker="*",s=130,zorder=5)
        ax.set(xlabel="Direct complete cost / vector always",ylabel=metric,title=f"User-disjoint test — {metric}",ylim=(0,None));ax.grid(alpha=.2)
    fig.legend(*axes[0].get_legend_handles_labels(),loc="upper center",ncol=2,frameon=False)
    fig.text(.5,.005,"Frozen106 rules +2 controller thresholds. Stars: CT-selected controllers. Descriptive test frontier; no retuning.",ha="center",fontsize=9)
    fig.tight_layout(rect=(0,.035,1,.91));save(fig,out,"frontier")
    print(json.dumps({"primary":result["primary"],"mixtures":result["secondary_mixtures"],"verification":result["verification"],"execution":result["execution"]},indent=2))


if __name__=="__main__":main()
