#!/usr/bin/env python3
"""Compare a frozen student with CE50 and a saved reference student."""

import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from blend_sasrec import checksum
from compare_sasrec import paired_intervals
from distill_sasrec import load_student,validation
from plot_sasrec_replay import plt,save
from replay_sasrec import quality
from run_sasrec import ROOT,write
from sasrec import ranks

MODELS=["ce50","reference_student","student"]


def read(path):
    return json.loads(path.read_text())


def describe(values,users,mask,total,intervals=True):
    selected={k:v[mask] for k,v in values.items()}
    measures={k:quality(v) for k,v in selected.items()}
    result={"targets":int(mask.sum()),"users":int(len(np.unique(users[mask]))),"metrics":measures,"new_minus":{}}
    for base in MODELS[:2]:
        delta={k:measures["student"][k]-measures[base][k] for k in ["ndcg@10","recall@100","hit@1"]}
        result["new_minus"][base]={"delta":delta,"ndcg_contribution":delta["ndcg@10"]*int(mask.sum())/total}
        if intervals:result["new_minus"][base]["paired"]=paired_intervals(selected["student"],selected[base],users[mask],42)
    return result


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--input",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--device",choices=["cpu","mps"],default="mps")
    args=parser.parse_args();root,out=args.input,args.output
    if out.exists():parser.error("Comparison output already exists")
    protocol=read(args.config);selection=read(root/"selection.json");evaluation=read(root/"evaluation.json")
    assert selection["locked_before_holdout_read"] and evaluation["selection_sha256"]==checksum(root/"selection.json")
    folders={"ce50":ROOT/protocol["control_run"],"reference_student":ROOT/protocol["comparison_run"],
             "student":ROOT/selection["student"]["run"]}
    runs={k:read(p/"training.json") for k,p in folders.items()}
    for name,folder in folders.items():assert checksum(folder/"best.pt")==runs[name]["checkpoint_sha256"]
    assert runs["reference_student"]["checkpoint_sha256"]==protocol["comparison_checkpoint_sha256"]
    assert runs["student"]["stop_reason"]=="validation_patience"
    assert runs["student"]["config"]["teacher_alpha"]==0 and runs["reference_student"]["config"]["teacher_alpha"]==protocol.get("comparison_alpha",.4)
    assert runs["student"]["config"]["distillation_temperature"]==protocol["temperature"]
    assert runs["reference_student"]["config"]["distillation_temperature"]==protocol.get("comparison_temperature",2)
    assert runs["student"]["config"]["distillation_weight"]==runs["reference_student"]["config"]["distillation_weight"]==1
    for key in runs["ce50"]["config"]:
        assert runs["student"]["config"][key]==runs["reference_student"]["config"][key]==runs["ce50"]["config"][key],key
    for key in ["initial_state_sha256","initial_rng_sha256","initial_mps_rng_sha256"]:
        assert runs["student"][key]==runs["reference_student"][key],key
    torch.set_num_threads(protocol["teacher"]["threads"])
    cache=ROOT/protocol["comparison_events"]
    assert checksum(cache)==read(ROOT/"reports/experiments/sasrec-256-distillation-audit/additional-slices.json")["source_events_sha256"]
    v=dict(np.load(cache));mv=v["split_idx"]==0;later=~mv;fresh=dict(np.load(root/"evaluation-ranks.npz"))
    for key in ["targets","users","split_idx"]:assert np.array_equal(v[key][later],fresh[key]),key
    assert np.array_equal(v["control"][later],fresh["control"]) and np.array_equal(fresh["teacher"],fresh["control"])
    reference_rank=v["student"];reference_paths=[]
    if protocol.get("comparison_ranks"):
        path=ROOT/protocol["comparison_ranks"];reference=dict(np.load(path))
        for key in ["targets","users","event_ids","split_idx"]:assert np.array_equal(reference[key],v[key]),key
        previous_selection=read(path.parent/"selection.json")
        assert previous_selection["student"]["checkpoint_sha256"]==protocol["comparison_checkpoint_sha256"]
        previous_eval=read(path.parent/"evaluation.json")
        assert previous_eval["selection_sha256"]==checksum(path.parent/"selection.json")
        reference_rank=reference[protocol["comparison_rank_key"]]
        previous_raw=dict(np.load(path.parent/"evaluation-ranks.npz"))
        assert np.array_equal(reference_rank[later],previous_raw["student"])
        reference_paths=[path,path.parent/"selection.json",path.parent/"evaluation.json",path.parent/"evaluation-ranks.npz"]
    model,_=load_student(folders["student"],args.device)
    new=np.empty(len(mv),dtype=np.int32);new[mv]=ranks(model,v["queries"][mv],v["targets"][mv],args.device,256)
    new[later]=fresh["student"]
    values={"ce50":v["control"],"reference_student":reference_rank,"student":new}
    for name,r in runs.items():
        actual=quality(values[name][mv]);assert all(actual[k]==val for k,val in validation(r).items()),name
    labels=["CE50 (reused)",protocol.get("reference_label","Mixture-teacher student (reused)"),protocol.get("student_label","CE50-teacher student")]
    result={"splits":{},"model_labels":dict(zip(MODELS,labels)),"fixed_teacher_alpha":protocol["fixed_teacher_alpha"],
        "temperature":protocol["temperature"],"reference_temperature":runs["reference_student"]["config"]["distillation_temperature"],
        "weight":selection["student"]["weight"],"selection_sha256":checksum(root/"selection.json"),
        "no_test_events_read":True,"no_new_baseline_training":True,"seed":42,"ci_scope":"2000 whole-user, event-weighted, pointwise, exploratory; no training variance"}
    data=ROOT/runs["ce50"]["config"]["data"];periods=read(data/"manifest.json")["periods"]
    segment_rows=[];metric_rows=[];user_rows=[];day_rows=[]
    for idx,split in enumerate(["model_valid","controller_train","policy_valid"]):
        take=v["split_idx"]==idx;n=int(take.sum());groups={
            "history":{key:v["stratum"]==i for i,key in enumerate(["last_item","second_last_only","earlier_in_window","outside_window"])},
            "popularity":{"top_1000":v["pop_rank"]<=1000,"1001_to_10000":(v["pop_rank"]>1000)&(v["pop_rank"]<=10000),"over_10000":v["pop_rank"]>10000},
            "intersection":{"last_item_over_10000":(v["stratum"]==0)&(v["pop_rank"]>10000)}}
        block={"all":describe(values,v["users"],take,n),"groups":{}}
        for family,masks in groups.items():
            block["groups"][family]={}
            for name,m in masks.items():
                entry=describe(values,v["users"],take&m,n);block["groups"][family][name]=entry
                segment_rows.append({"split":split,"family":family,"segment":name,"targets":entry["targets"],
                    **{f"{model}_{metric}":entry["metrics"][model][metric] for model in MODELS for metric in ["ndcg@10","recall@100","hit@1"]},
                    **{f"new_minus_{base}":entry["new_minus"][base]["delta"]["ndcg@10"] for base in MODELS[:2]}})
        for base in MODELS[:2]:
            assert np.isclose(sum(r["new_minus"][base]["ndcg_contribution"] for r in block["groups"]["history"].values()),block["all"]["new_minus"][base]["delta"]["ndcg@10"])
        bucket=lambda r:np.select([r==1,r<=10,r<=100],[0,1,2],default=3)
        block["transition_labels"]=["1","2..10","11..100",">100"]
        block["rank_transitions"]={base:np.bincount(bucket(values[base][take])*4+bucket(new[take]),minlength=16).reshape(4,4).tolist() for base in MODELS[:2]}
        start=next(p["start"] for p in periods if p["split"]==split);days=(v["timestamp"]-start)//86400
        block["user_macro_delta"]={}
        for field,array,destination in [("user_idx",v["users"],user_rows),("day",days,day_rows)]:
            current=[]
            for key in np.unique(array[take]):
                entry=describe(values,v["users"],take&(array==key),n,False)
                row={"split":split,field:int(key),"targets":entry["targets"],
                    **{f"{model}_{metric}":entry["metrics"][model][metric] for model in MODELS for metric in ["ndcg@10","recall@100","hit@1"]},
                    **{f"new_minus_{base}":entry["new_minus"][base]["delta"]["ndcg@10"] for base in MODELS[:2]}}
                destination.append(row);current.append(row)
            if field=="user_idx":block["user_macro_delta"]={base:float(np.mean([r[f"new_minus_{base}"] for r in current])) for base in MODELS[:2]}
        result["splits"][split]=block
        metric_rows.extend({"split":split,"model":name,**block["all"]["metrics"][name]} for name in MODELS)
    result["training"]={name:{"reused":name!="student","epochs":len(r["epochs"]),"best_epoch":r["best_epoch"],
        "validation":validation(r),"elapsed_seconds":r["elapsed_seconds"],"parameters":r["parameters"],
        "peak_rss_mib":max(e["process_peak_rss_mib"] for e in r["epochs"]),
        "max_saved_mps_driver_mib":max(e["mps_driver_mib"] for e in r["epochs"])} for name,r in runs.items()}
    result["source_sha256"]={str(p.resolve().relative_to(ROOT)):checksum(p) for p in [Path(__file__),args.config,cache,
        root/"selection.json",root/"evaluation-ranks.npz"]+reference_paths+[folder/"best.pt" for folder in folders.values()]}
    out.mkdir(parents=True);write(out/"comparison.json",result)
    np.savez_compressed(root/"comparison-ranks.npz",**values,targets=v["targets"],users=v["users"],event_ids=v["event_ids"],split_idx=v["split_idx"])
    curves=[]
    for name,r in runs.items():
        for e in [{"epoch":0,"validation":r["initial_validation"]}]+r["epochs"]:
            curves.append({"model":name,"epoch":e["epoch"],"best":e["epoch"]==r["best_epoch"],**e["validation"],
                "loss":e.get("loss"),"ce":e.get("loss_components",{}).get("ce",e.get("loss")),
                "kl_t2":e.get("loss_components",{}).get("kl_t2"),"train_seconds":e.get("train_seconds")})
    for name,rows in [("metrics.csv",metric_rows),("segments.csv",segment_rows),("users.csv",user_rows),("days.csv",day_rows),("training.csv",curves)]:
        with (out/name).open("w",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n");writer.writeheader();writer.writerows(rows)
    for name,source in [("protocol.json",args.config),("initialization.json",root/"initialization.json"),
            ("teacher-check.json",root/"teacher-check.json"),("matched-setup.json",root/"matched-setup.json"),("fixed-teacher.json",root/"fixed-teacher.json"),
            ("selection.json",root/"selection.json"),("evaluation.json",root/"evaluation.json"),("training.json",folders["student"]/"training.json"),
            ("tests.txt",root/"tests.txt")]:shutil.copyfile(source,out/name)
    plt.rcParams.update({"axes.spines.top":False,"axes.spines.right":False,"axes.grid":True,"grid.alpha":.2})
    fig,axes=plt.subplots(1,2,figsize=(10,3.7))
    for ax,metric in zip(axes,["ndcg@10","recall@100"]):
        for (name,r),label,color in zip(runs.items(),labels,["#c36b22","#8653a6","#26906b"]):
            vals=[r["initial_validation"]]+[e["validation"] for e in r["epochs"]]
            ax.plot(range(len(vals)),[x[metric] for x in vals],"o-",label=label,color=color,markersize=4)
            ax.scatter([r["best_epoch"]],[validation(r)[metric]],s=85,facecolors="none",edgecolors=color)
        ax.set(xlabel="Epoch (0 = same random initialization)",ylabel=metric,title="model_valid")
    axes[0].legend(fontsize=8);fig.tight_layout();save(fig,out,"validation")
    print(json.dumps({"training":result["training"],"policy_valid":result["splits"]["policy_valid"]["all"]},indent=2))


if __name__=="__main__":main()
