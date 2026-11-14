#!/usr/bin/env python3
"""Audit fixed-horizon labels and compare the resulting closed-loop controller."""

import json
import shutil
from pathlib import Path

import numpy as np
import torch

from blend_sasrec import checksum
from compare_sasrec import paired_intervals
from controller_horizon import horizon_design,horizon_gain
from plot_sasrec_replay import plt,save
from refresh_controller import training_examples
from report_controller_models import independent_ci,read,terms
from report_student_maintenance import cost_matched_mixture,csv_file
from run_sasrec import ROOT,write


def main():
    root=ROOT/"data/runs/controller-mlp-horizon5";label_root=ROOT/"data/runs/controller-horizon-labels"
    run=read(root/"results.json");current=dict(np.load(root/"replay.npz"));label_meta=read(label_root/"results.json")
    labels=dict(np.load(label_root/"labels.npz"));pairs=dict(np.load(label_root/"pairs.npz"))
    base=ROOT/run["config"]["policy_replay"];stream=dict(np.load(base/"replay.npz"));baseline=read(root/"policy-results.json")
    old_roots={name:ROOT/f"data/runs/controller-{name}" for name in ["mlp42","ridge8"]}
    old={name:read(path/"results.json") for name,path in old_roots.items()}
    arrays={name:dict(np.load(path/"replay.npz")) for name,path in old_roots.items()}
    for metadata in [run,label_meta]:
        for path,h in metadata["source_sha256"].items():assert checksum(ROOT/path)==h,path
    assert checksum(label_root/"labels.npz")==label_meta["label_sha256"]==run["config"]["training_labels_sha256"]
    assert checksum(root/"replay.npz")==run["replay_sha256"] and checksum(root/"selection.json")==run["selection_sha256"]
    for name,path in old_roots.items():assert checksum(path/"replay.npz")==old[name]["replay_sha256"]
    ct=stream["split_idx"]==1;fit_end=run["fit"]["fit_cutoff"];early_end=run["fit"]["cutoff"]
    regions={"fit":ct&(stream["timestamp"]<fit_end),"early":ct&(stream["timestamp"]>=fit_end)&(stream["timestamp"]<early_end)}
    width=torch.load(ROOT/baseline["config"]["checkpoint"],map_location="cpu",weights_only=True)["items"]+2
    manual_checks=0;horizon_one_delta=[]
    for prefix,region in regions.items():
        rows=training_examples(stream,stream["ranks"],stream["refs"],region)
        for field in ["requests","previous"]:np.testing.assert_array_equal(rows[field],labels[f"{prefix}_{field}"])
        one=horizon_gain(horizon_design(stream,rows,region,1,width),pairs["keys"],pairs["ranks"])
        np.testing.assert_allclose(one,rows["gain"],rtol=0,atol=1e-14);horizon_one_delta.append(float(np.abs(one-rows["gain"]).max()))
        # Deliberately scalar implementation, independent of vectorized label construction.
        for j in np.random.default_rng(42).choice(len(rows["requests"]),64,replace=False):
            request=rows["requests"][j];previous=rows["previous"][j]
            future=np.flatnonzero(region&(stream["users"]==stream["users"][request])&(stream["timestamp"]>=stream["timestamp"][request]))[:5]
            total=0.
            for future_request in future:
                target_ids=np.flatnonzero(stream["target_request"]==future_request)
                for target in stream["targets"][target_ids]:
                    keys=np.array([request,previous],dtype=np.int64)*width+target
                    positions=np.searchsorted(pairs["keys"],keys);np.testing.assert_array_equal(pairs["keys"][positions],keys)
                    ranks=pairs["ranks"][positions]
                    values=np.where(ranks<=10,1/np.log2(ranks+1),0);total+=values[0]-values[1]
            np.testing.assert_allclose(total/len(future),labels[f"{prefix}_gain"][j],rtol=0,atol=1e-14);manual_checks+=1
    fit=dict(np.load(root/"training-examples.npz"));reference=dict(np.load(old_roots["mlp42"]/"training-examples.npz"))
    for field in ["requests","previous","weights"]:np.testing.assert_array_equal(fit[field],reference[field])
    out=ROOT/"reports/experiments/controller-horizon";out.mkdir(parents=True,exist_ok=True)
    result={"comparisons":[],"labels":label_meta,"no_test_events_read":True};table=[];checked=0
    for idx,split in enumerate(["controller_train","policy_valid"],1):
        take=stream["split_idx"][stream["target_request"]]==idx;users=stream["users"][stream["target_request"]][take]
        simple=baseline["results"][split]["policies"];always=simple[0]["cost"]["estimated_ms_per_request"]
        for selected in run["selection"]["budgets"]:
            controller=next(p for p in run["results"][split] if p["name"]==selected["controller"])
            k=list(current["names"]).index(controller["name"]);rank=current["ranks"][k,take]
            j=next(j for j,p in enumerate(baseline["policies"]) if p["name"]==selected["matched_simple"])
            references={"simple":stream["ranks"][j,take]};old_rows={}
            for name,metadata in old.items():
                pick=next(p for p in metadata["selection"]["budgets"] if p["relative_budget"]==selected["relative_budget"])
                position=list(arrays[name]["names"]).index(pick["controller"])
                references[name]=arrays[name]["ranks"][position,take]
                old_rows[name]=next(p for p in metadata["results"][split] if p["name"]==pick["controller"])
            effects={name:paired_intervals(rank,reference,users,42) for name,reference in references.items()}
            for name,block in effects.items():
                for metric,effect in block.items():
                    np.testing.assert_allclose(independent_ci(terms(rank)[metric]-terms(references[name])[metric],users),effect["cluster_bootstrap_95ci"],rtol=0,atol=1e-14);checked+=1
            row={"split":split,"budget":selected["relative_budget"],"controller":controller,"simple":simple[j],"other_controllers":old_rows,"paired":effects}
            mix=cost_matched_mixture(run["selection"]["tuning_simple_rows"],selected["tuning_controller"]["cost"]["estimated_ms_per_request"])
            if mix:
                left=next(i for i,p in enumerate(simple) if p["name"]==mix["left"]);right=next(i for i,p in enumerate(simple) if p["name"]==mix["right"]);p=mix["right_probability"]
                expected={metric:(1-p)*values+p*terms(stream["ranks"][right,take])[metric] for metric,values in terms(stream["ranks"][left,take]).items()}
                row["mixture"]={**mix,"paired":{metric:{"difference":float((values-expected[metric]).mean()),"cluster_bootstrap_95ci":independent_ci(values-expected[metric],users)} for metric,values in terms(rank).items()}}
            result["comparisons"].append(row)
            table.append({"split":split,"budget":selected["relative_budget"],"policy":controller["name"],"simple":simple[j]["name"],
                "relative_cost":controller["cost"]["estimated_ms_per_request"]/always,"simple_relative_cost":simple[j]["cost"]["estimated_ms_per_request"]/always,
                "refresh_fraction":controller["refresh_fraction"],**controller["metrics"]})
    result["verification"]={"manual_horizon_examples":manual_checks,"horizon_one_matches_immediate_max_delta":max(horizon_one_delta),
        "same_fit_requests_states_weights":True,"independent_paired_intervals":checked,"no_region_boundary_crossing":True}
    result["source_sha256"]={str(path.relative_to(ROOT)):checksum(path) for path in [Path(__file__).resolve(),root/"results.json",root/"replay.npz",label_root/"results.json",label_root/"pairs.npz"]}
    write(out/"comparison.json",result);csv_file(out/"comparison.csv",table)
    for filename in ["results.json","model.json","model-training.json","selection.json","policy-selection.json"]:shutil.copyfile(root/filename,out/filename)
    shutil.copyfile(label_root/"results.json",out/"label-preparation.json")
    fig,axes=plt.subplots(1,2,figsize=(10,4.5))
    for ax,metric in zip(axes,["ndcg@10","recall@100"]):
        rows=[r for r in table if r["split"]=="policy_valid"]
        ax.plot([r["relative_cost"] for r in rows],[r[metric] for r in rows],"o--",c="#168469",label="Horizon5")
        for name in old:
            priced=read(old_roots[name]/"policy-results.json");always=priced["results"]["policy_valid"]["policies"][0]["cost"]["estimated_ms_per_request"]
            rows=old[name]["results"]["policy_valid"]
            ax.plot([r["cost"]["estimated_ms_per_request"]/always for r in rows],[r["metrics"][metric] for r in rows],"o--",label=name)
        ax.set(xlabel="Full cost / own-run always",ylabel=metric,title=f"policy_valid — {metric}");ax.grid(alpha=.2)
    fig.legend(*axes[0].get_legend_handles_labels(),loc="upper center",ncol=3,frameon=False);fig.tight_layout(rect=(0,0,1,.9));save(fig,out,"frontier")
    print(json.dumps({"pv":[row for row in result["comparisons"] if row["split"]=="policy_valid"],"verification":result["verification"]},indent=2))


if __name__=="__main__":main()
