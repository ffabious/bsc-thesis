#!/usr/bin/env python3
"""Test a small preregistered extension of the refresh grid using shared costs."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from blend_sasrec import checksum
from compare_sasrec import paired_intervals
from distill_sasrec import load_student
from plot_sasrec_replay import plt,save
from replay_sasrec import encode_queries,quality
from replay_sasrec_policies import frontier,rank_reusing,schedule,summarize_split
from report_student_maintenance import csv_file
from run_sasrec import ROOT,state_digest,write
from sasrec import ranks
from train_refresh_controller import region_rows


def read(path):return json.loads(path.read_text())


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=ROOT/"configs/experiments/student43-dense-policies.json")
    parser.add_argument("--device",choices=["cpu","mps"],default="mps")
    args=parser.parse_args();config=read(args.config);torch.set_num_threads(config["threads"])
    root=ROOT/config["output"];out=ROOT/config["report"]
    if root.exists() or out.exists():parser.error("Choose new outputs")
    source=ROOT/config["policy_replay"];controller_root=ROOT/config["controller"]
    base=read(controller_root/"policy-results.json");control=read(controller_root/"results.json")
    v=dict(np.load(source/"replay.npz"));c=dict(np.load(controller_root/"replay.npz"))
    paths=[args.config.resolve(),Path(__file__).resolve(),source/"replay.npz",source/"results.json",
        controller_root/"policy-results.json",controller_root/"results.json",controller_root/"replay.npz"]
    paths += [ROOT/"scripts"/p for p in ["replay_sasrec_policies.py","replay_sasrec.py","sasrec.py","run_sasrec.py","train_refresh_controller.py"]]
    model_path=ROOT/base["config"]["checkpoint"];assert checksum(model_path)==base["checkpoint_sha256"]==config["checkpoint_sha256"]
    paths += [model_path];hashes={str(p.relative_to(ROOT)):checksum(p) for p in paths}
    ct=v["split_idx"]==1;valid=ct&(v["timestamp"]>=control["fit"]["cutoff"])
    choices=config["policies"];refs=[];updates=[];decision=[];started=time.perf_counter()
    for policy in choices:
        r,f=schedule(v,policy,256);refs.append(r);updates.append(f);trial=[]
        for _ in range(3):
            row=[]
            for idx in [1,2]:
                take=v["split_idx"]==idx;part={key:v[key][take] for key in ["users","timestamp","history","split_idx"]}
                begin=time.perf_counter();schedule(part,policy,256);row.append(1000*(time.perf_counter()-begin)/take.sum())
            trial.append(row)
        decision.append(np.median(trial,axis=0).tolist())
    refs,updates=np.array(refs),np.array(updates)
    assert np.all(v["users"][refs]==v["users"][None]) and np.all(v["split_idx"][refs]==v["split_idx"][None])
    assert np.all(v["last_input_timestamp"][refs]<v["timestamp"][None])
    model,_=load_student(model_path.parent,args.device);state=state_digest(model)
    vectors,encoding=encode_queries(model,v["queries"],args.device,256)
    new_rank,scoring=rank_reusing(model,vectors,refs[:,v["target_request"]],v["targets"],args.device,256,
        (v["refs"][:,v["target_request"]],v["ranks"]))
    full_rank=np.r_[v["ranks"],new_rank];full_refs=np.r_[v["refs"],refs];full_refresh=np.r_[v["refresh"],updates]
    all_choices=base["policies"]+choices;all_decision=[[p["cost"]["decision_ms_per_request"] for p in base["results"][split]["policies"]] for split in config["splits"]]
    all_decision=np.r_[np.array(all_decision).T,np.array(decision)]
    ct_targets=np.flatnonzero(ct[v["target_request"]])
    tuning=region_rows([p["name"] for p in all_choices],full_rank[:,ct_targets],ct_targets,full_refresh,valid,v,all_decision[:,0],base["branch_costs"]["controller_train"])
    selected=[]
    for row in control["selection"]["budgets"]:
        cap=row["tuning_controller"]["cost"]["estimated_ms_per_request"]
        eligible=[p for p in tuning if p["cost"]["estimated_ms_per_request"]<=cap]
        best=max(eligible,key=lambda p:(p["metrics"]["ndcg@10"],p["metrics"]["recall@100"],-p["cost"]["estimated_ms_per_request"]))
        selected.append({"budget":row["relative_budget"],"controller":row["controller"],"simple":best["name"],"ct_tuning":best})
    root.mkdir(parents=True);write(root/"selection.json",{"source":"CT last four days","selected":selected,"policies":choices})
    selection_hash=checksum(root/"selection.json");result={"config":config,"results":{},"comparisons":{},"selection":selected,"selection_sha256":selection_hash}
    policy_rows=[];comparisons=[]
    for idx,split in enumerate(config["splits"],1):
        take=v["split_idx"][v["target_request"]]==idx;users=v["users"][v["target_request"]][take]
        combined=summarize_split(v,full_rank[:,take],full_refs,full_refresh,all_choices,idx,base["branch_costs"][split],all_decision,42)
        result["results"][split]=combined;result["comparisons"][split]=[];always=combined["policies"][0]["cost"]["estimated_ms_per_request"]
        for p in combined["policies"]:
            policy_rows.append({"split":split,"policy":p["name"],"new":p["name"] in [p["name"] for p in choices],"relative_cost":p["cost"]["estimated_ms_per_request"]/always,"refresh_fraction":p["refresh_fraction"],**p["metrics"]})
        for pick in selected:
            k=list(c["names"]).index(pick["controller"]);j=next(j for j,p in enumerate(all_choices) if p["name"]==pick["simple"])
            effect=paired_intervals(c["ranks"][k,take],full_rank[j,take],users,42)
            simple=combined["policies"][j];ctrl=next(p for p in control["results"][split] if p["name"]==pick["controller"])
            row={**pick,"controller_row":ctrl,"simple_row":simple,"paired":effect};result["comparisons"][split].append(row)
            flat={"split":split,"budget":pick["budget"],"controller":pick["controller"],"simple":pick["simple"],"controller_cost":ctrl["cost"]["estimated_ms_per_request"]/always,"simple_cost":simple["cost"]["estimated_ms_per_request"]/always}
            for metric,e in effect.items():flat.update({f"controller_{metric}":ctrl["metrics"][metric],f"simple_{metric}":simple["metrics"][metric],f"delta_{metric}":e["difference"],f"ci_low_{metric}":e["cluster_bootstrap_95ci"][0],f"ci_high_{metric}":e["cluster_bootstrap_95ci"][1]})
            comparisons.append(flat)
    cpu,_=load_student(model_path.parent,"cpu");p,t=np.unravel_index(np.random.default_rng(42).choice(new_rank.size,256,replace=False),new_rank.shape)
    expected=ranks(cpu,v["queries"][refs[p,v["target_request"][t]]],v["targets"][t],"cpu",256)
    assert quality(expected)==quality(new_rank[p,t]) and state_digest(model)==state
    for path,h in hashes.items():assert checksum(ROOT/path)==h,path
    assert checksum(root/"selection.json")==selection_hash
    np.savez_compressed(root/"replay.npz",refs=refs,refresh=updates,ranks=new_rank)
    result.update(source_sha256=hashes,cpu_oracle={"same_metrics":True,"exact_ranks":int((expected==new_rank[p,t]).sum()),"max_rank_delta":int(np.abs(expected-new_rank[p,t]).max())},
        execution={"encoding":encoding,"scoring":scoring,"total_seconds":time.perf_counter()-started},no_test_events_read=True,
        cost_scope="Fixed shared coupled calibration from student43-maintenance; new CPU schedule times; no new serving benchmark")
    write(root/"results.json",result);out.mkdir(parents=True);write(out/"results.json",result);write(out/"config.json",config)
    csv_file(out/"policies.csv",policy_rows);csv_file(out/"controller-comparison.csv",comparisons)
    fig,axes=plt.subplots(2,2,figsize=(11,7))
    for i,split in enumerate(config["splits"]):
        block=result["results"][split];always=block["policies"][0]["cost"]["estimated_ms_per_request"]
        for j,key in enumerate(["ndcg@10","recall@100"]):
            ax=axes[i,j]
            for take,color,label in [(block["policies"][:100],"#aaa","Original 100"),(block["policies"][100:],"#c87220","Six new rules"),(control["results"][split],"#168469","Fixed ridge controller")]:
                ax.scatter([p["cost"]["estimated_ms_per_request"]/always for p in take],[p["metrics"][key] for p in take],color=color,label=label,s=28 if color=="#aaa" else 65,marker="*" if color=="#168469" else "o")
            ax.set(title=f"{split} — {key}",xlabel="Shared full cost / always",ylabel="Quality",ylim=(0,None));ax.grid(alpha=.2)
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc="upper center",ncol=3,frameon=False);fig.tight_layout(rect=(0,0,1,.94));save(fig,out,"frontier")
    print(json.dumps({"selection":selected,"pv":result["comparisons"]["policy_valid"],"execution":result["execution"]},indent=2))


if __name__=="__main__":main()
