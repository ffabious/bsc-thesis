#!/usr/bin/env python3
"""Compare CE50/student pairs across fixed seeds on identical events and items."""

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

MODELS=["ce42","student42","ce43","student43"]
WITHIN={"student42_minus_ce42":("student42","ce42"),"student43_minus_ce43":("student43","ce43")}
CONTRASTS={**WITHIN,"ce43_minus_ce42":("ce43","ce42"),"student43_minus_student42":("student43","student42")}


def read(path):
    return json.loads(path.read_text())


def describe(values,users,mask,total,contrasts):
    metrics={name:quality(rank[mask]) for name,rank in values.items()}
    differences={}
    for name,(left,right) in contrasts.items():
        delta={key:metrics[left][key]-metrics[right][key] for key in ["ndcg@10","recall@100","hit@1"]}
        differences[name]={"delta":delta,"ndcg_contribution":delta["ndcg@10"]*int(mask.sum())/total,
            "paired":paired_intervals(values[left][mask],values[right][mask],users[mask],42)}
    return {"targets":int(mask.sum()),"users":int(len(np.unique(users[mask]))),"metrics":metrics,"differences":differences}


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device",choices=["cpu","mps"],default="mps")
    args=parser.parse_args();torch.set_num_threads(4)
    root=ROOT/"data/runs/paired-seed-repeat";student=root/"student";out=ROOT/"reports/experiments/paired-seed-repeat"
    if out.exists():parser.error("Comparison output already exists")
    protocol=read(ROOT/"configs/experiments/paired-repeat-student.json")
    selected=read(student/"selection.json");evaluation=read(student/"evaluation.json")
    assert selected["locked_before_holdout_read"] and checksum(student/"selection.json")==evaluation["selection_sha256"]
    folders={"ce42":ROOT/"data/runs/sasrec-256-full-ce/train-mps",
        "student42":ROOT/"data/runs/student-temperature-control/kd-1/train-mps",
        "ce43":root/"ce50/train-mps","student43":student/"kd-1/train-mps"}
    runs={name:read(path/"training.json") for name,path in folders.items()}
    for name,path in folders.items():assert checksum(path/"best.pt")==runs[name]["checkpoint_sha256"]
    for key,value in runs["ce42"]["config"].items():
        if key=="seed":continue
        assert all(r["config"][key]==value for r in runs.values()),key
    for seed in [42,43]:
        ce,kd=runs[f"ce{seed}"],runs[f"student{seed}"]
        assert ce["config"]["seed"]==kd["config"]["seed"]==seed
        assert kd["config"]["teacher_alpha"]==0 and kd["config"]["distillation_temperature"]==kd["config"]["distillation_weight"]==1
        assert kd["teacher_sources"]["long"]["checkpoint_sha256"]==ce["checkpoint_sha256"]
    for key in ["initial_state_sha256","initial_rng_sha256","initial_mps_rng_sha256"]:
        assert runs["ce43"][key]==runs["student43"][key]
    features=ROOT/"data/diagnostics/sasrec-256-distillation-audit/events.npz";v=dict(np.load(features))
    assert checksum(features)==read(ROOT/"reports/experiments/sasrec-256-distillation-audit/additional-slices.json")["source_events_sha256"]
    previous=ROOT/"data/runs/student-temperature-control/comparison-ranks.npz";old=dict(np.load(previous))
    fresh=dict(np.load(student/"evaluation-ranks.npz"));teacher=dict(np.load(student/"teacher/model-valid.npz"))
    mv=v["split_idx"]==0;later=~mv
    replay_path=ROOT/protocol["reference_replay"]/"replay.npz";replay=dict(np.load(replay_path));q=replay["target_request"]
    for key,expected in [("queries",replay["queries"][q]),("event_ids",replay["target_event_ids"]),
        ("timestamp",replay["timestamp"][q])]:assert np.array_equal(v[key][later],expected),key
    for key in ["targets","users","event_ids","split_idx"]:assert np.array_equal(old[key],v[key]),key
    for key in ["targets","users","split_idx"]:assert np.array_equal(fresh[key],v[key][later]),key
    assert np.array_equal(fresh["original"],old["ce50"][later]) and np.array_equal(fresh["teacher"],fresh["control"])
    assert not evaluation["reused_control_ranks"]
    assert np.array_equal(teacher["targets"],v["targets"][mv]) and np.array_equal(teacher["users"],v["users"][mv])
    values={"ce42":old["ce50"],"student42":old["student"]}
    values["ce43"]=np.empty(len(mv),dtype=np.int32);values["ce43"][mv]=teacher["ranks"];values["ce43"][later]=fresh["control"]
    model,_=load_student(folders["student43"],args.device)
    values["student43"]=np.empty(len(mv),dtype=np.int32)
    values["student43"][mv]=ranks(model,v["queries"][mv],v["targets"][mv],args.device,256)
    values["student43"][later]=fresh["student"]
    for name,r in runs.items():assert all(quality(values[name][mv])[k]==val for k,val in validation(r).items()),name
    result={"splits":{},"within_seed_contrasts":WITHIN,"source_seed":43,"reference_seed":42,"bootstrap_seed":42,
        "no_test_events_read":True,"new_training_runs":2,"ci_scope":"2000 whole-user, event-weighted, fixed checkpoints; two seeds do not estimate training variance reliably"}
    data=ROOT/runs["ce43"]["config"]["data"];periods=read(data/"manifest.json")["periods"]
    metrics_rows=[];segments=[];effects=[];users=[];days=[]
    for idx,split in enumerate(["model_valid","controller_train","policy_valid"]):
        take=v["split_idx"]==idx;n=int(take.sum());block={"all":describe(values,v["users"],take,n,CONTRASTS),"groups":{}}
        masks={"history":{name:v["stratum"]==i for i,name in enumerate(["last_item","second_last_only","earlier_in_window","outside_window"])},
            "popularity":{"top_1000":v["pop_rank"]<=1000,"1001_to_10000":(v["pop_rank"]>1000)&(v["pop_rank"]<=10000),"over_10000":v["pop_rank"]>10000},
            "intersection":{"last_item_over_10000":(v["stratum"]==0)&(v["pop_rank"]>10000)}}
        for family,groups in masks.items():
            block["groups"][family]={}
            for name,mask in groups.items():
                entry=describe(values,v["users"],take&mask,n,WITHIN);block["groups"][family][name]=entry
                segments.append({"split":split,"family":family,"segment":name,"targets":entry["targets"],
                    **{f"{model}_{k}":entry["metrics"][model][k] for model in MODELS for k in ["ndcg@10","recall@100","hit@1"]}})
        for name in WITHIN:assert np.isclose(sum(g["differences"][name]["ndcg_contribution"] for g in block["groups"]["history"].values()),block["all"]["differences"][name]["delta"]["ndcg@10"])
        bucket=lambda r:np.select([r==1,r<=10,r<=100],[0,1,2],default=3)
        block["transition_labels"]=["1","2..10","11..100",">100"]
        block["rank_transitions"]={name:np.bincount(bucket(values[right][take])*4+bucket(values[left][take]),minlength=16).reshape(4,4).tolist() for name,(left,right) in WITHIN.items()}
        for name,(left,right) in WITHIN.items():
            delta=block["all"]["differences"][name]["delta"]
            effects.append({"split":split,"seed":runs[left]["config"]["seed"],**delta,
                **{f"relative_{k}_pct":100*delta[k]/block["all"]["metrics"][right][k] for k in ["ndcg@10","recall@100"]}})
        block["two_seed_mean_delta"]={key:float(np.mean([block["all"]["differences"][name]["delta"][key] for name in WITHIN])) for key in ["ndcg@10","recall@100","hit@1"]}
        start=next(p["start"] for p in periods if p["split"]==split);day=(v["timestamp"]-start)//86400
        for field,array,destination in [("user_idx",v["users"],users),("day",day,days)]:
            for key in np.unique(array[take]):
                mask=take&(array==key);q={name:quality(rank[mask]) for name,rank in values.items()}
                destination.append({"split":split,field:int(key),"targets":int(mask.sum()),
                    **{f"{name}_{k}":q[name][k] for name in MODELS for k in ["ndcg@10","recall@100","hit@1"]},
                    **{name:q[left]["ndcg@10"]-q[right]["ndcg@10"] for name,(left,right) in WITHIN.items()}})
        result["splits"][split]=block
        metrics_rows.extend({"split":split,"model":name,**q} for name,q in block["all"]["metrics"].items())
    result["training"]={name:{"reused":name.endswith("42"),"epochs":len(r["epochs"]),"best_epoch":r["best_epoch"],
        "selected_updates":r["best_epoch"]*((r["train_examples"]+r["config"]["batch_size"]-1)//r["config"]["batch_size"]),
        "validation":validation(r),"elapsed_seconds":r["elapsed_seconds"],"parameters":r["parameters"],
        "peak_rss_mib":max(e["process_peak_rss_mib"] for e in r["epochs"]),
        "max_saved_mps_driver_mib":max(e["mps_driver_mib"] for e in r["epochs"])} for name,r in runs.items()}
    source=[Path(__file__),features,previous,replay_path,student/"selection.json",student/"evaluation-ranks.npz",student/"teacher/model-valid.npz"]
    source += [p/"best.pt" for p in folders.values()]
    result["source_sha256"]={str(p.resolve().relative_to(ROOT)):checksum(p) for p in source}
    out.mkdir(parents=True);write(out/"comparison.json",result)
    np.savez_compressed(root/"comparison-ranks.npz",**values,targets=v["targets"],users=v["users"],event_ids=v["event_ids"],split_idx=v["split_idx"])
    curves=[]
    for name,r in runs.items():
        for row in [{"epoch":0,"validation":r["initial_validation"]}]+r["epochs"]:
            curves.append({"model":name,"epoch":row["epoch"],"best":row["epoch"]==r["best_epoch"],**row["validation"],
                "loss":row.get("loss"),"ce":row.get("loss_components",{}).get("ce",row.get("loss")),
                "kl_t2":row.get("loss_components",{}).get("kl_t2"),"train_seconds":row.get("train_seconds")})
    for name,rows in [("metrics.csv",metrics_rows),("effects.csv",effects),("segments.csv",segments),("users.csv",users),("days.csv",days),("training.csv",curves)]:
        with (out/name).open("w",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n");writer.writeheader();writer.writerows(rows)
    for name,source in [("config-ce43.json",ROOT/"configs/experiments/paired-repeat-ce50.json"),("config-student43.json",ROOT/"configs/experiments/paired-repeat-student.json"),
        ("setup.json",root/"setup.json"),("pair-initialization.json",root/"pair-initialization.json"),("initialization.json",student/"initialization.json"),
        ("teacher-check.json",student/"teacher-check.json"),("teacher-selection.json",student/"fixed-teacher.json"),("selection.json",student/"selection.json"),
        ("evaluation.json",student/"evaluation.json"),("training-ce43.json",folders["ce43"]/"training.json"),("training-student43.json",folders["student43"]/"training.json"),
        ("resume.json",root/"student-resume.json"),("tests.txt",root/"tests.txt")]:shutil.copyfile(source,out/name)
    plt.rcParams.update({"axes.spines.top":False,"axes.spines.right":False,"axes.grid":True,"grid.alpha":.2})
    fig,axes=plt.subplots(1,2,figsize=(10,3.7))
    for ax,key in zip(axes,["ndcg@10","recall@100"]):
        for (name,r),color in zip(runs.items(),["#888888","#8653a6","#c36b22","#26906b"]):
            vals=[r["initial_validation"]]+[e["validation"] for e in r["epochs"]]
            ax.plot(range(len(vals)),[q[key] for q in vals],"o-",label=name,color=color,markersize=4)
            ax.scatter([r["best_epoch"]],[validation(r)[key]],s=85,facecolors="none",edgecolors=color)
        ax.set(xlabel="Epoch (same initial weights within each seed)",ylabel=key,title="model_valid")
    axes[0].legend(fontsize=8);fig.tight_layout();save(fig,out,"validation")
    print(json.dumps({"training":result["training"],"policy_valid":result["splits"]["policy_valid"]["all"]},indent=2))


if __name__=="__main__":main()
