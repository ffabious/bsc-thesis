#!/usr/bin/env python3
"""Compare frozen CE/student pairs using the original fixed-lag replay inputs."""

import argparse
import csv
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from blend_sasrec import checksum
from distill_sasrec import load_student
from plot_sasrec_replay import plt,save
from replay_sasrec import encode_queries,quality,rank_vectors,settings,summarize
from run_sasrec import ROOT,memory,model_args,state_digest,write
from sasrec import ranks


def read(path):
    return json.loads(path.read_text())


def terms(rank):
    return {"ndcg@10":np.where(rank<=10,1/np.log2(rank+1),0),"recall@100":(rank<=100).astype(float)}


def paired_effect(delta,users,seed=42):
    """Event-weighted differences, resampling whole users with replacement."""
    ids,inverse=np.unique(users,return_inverse=True);counts=np.bincount(inverse)
    draws=np.random.default_rng(seed).integers(0,len(ids),(2000,len(ids)))
    result=[]
    for values in delta:
        sums=np.bincount(inverse,weights=values,minlength=len(ids))
        ci=np.quantile(sums[draws].sum(1)/counts[draws].sum(1),[.025,.975])
        result.append({"difference":float(values.mean()),"cluster_bootstrap_95ci":ci.tolist()})
    return result


def compare_pair(student,control,users,seed=42):
    result={};control_terms=terms(control)
    for key,s in terms(student).items():
        delta=s-control_terms[key]
        # Positive: the student loses less quality to staleness than its own CE.
        result[key]={"student_minus_ce":paired_effect(delta,users,seed),
            "loss_reduction_vs_ce":paired_effect(delta-delta[[0]],users,seed)}
    return result


def plot(result,out,unit):
    conditions=[(j,c) for j,c in enumerate(result["conditions"]) if c["unit"] in ["fresh",unit]]
    x=[c["lag"] for _,c in conditions];threshold=1 if unit=="events" else 300
    labels=["0","1","5","10","25","50"] if unit=="events" else ["0","5m","30m","2h","1d","7d"]
    colors=dict(zip(result["models"],["#858585","#8851aa","#ce7020","#168469"]))
    fig,axes=plt.subplots(4,2,figsize=(11.5,11.5),sharex=True)
    for col,split in enumerate(result["config"]["splits"]):
        axes[0,col].set_title(split)
        for name,model in result["models"].items():
            rows=model["results"][split]["conditions"]
            for row,key in [(0,"ndcg@10"),(2,"recall@100")]:
                selected=[rows[j] for j,_ in conditions]
                axes[row,col].plot(x,[r["metrics"][key] for r in selected],"o-",color=colors[name],label=name,markersize=4)
                effect=[r["stale_minus_fresh"][key] for r in selected]
                axes[row+1,col].plot(x,[-r["difference"] for r in effect],"o-",color=colors[name],markersize=4)
                ci=np.array([r["cluster_bootstrap_95ci"] for r in effect])
                axes[row+1,col].fill_between(x,-ci[:,1],-ci[:,0],color=colors[name],alpha=.09)
        for row,label in enumerate(["NDCG@10","Fresh − stale NDCG@10","Recall@100","Fresh − stale Recall@100"]):
            ax=axes[row,col];ax.set_ylabel(label);ax.set_xscale("symlog",linthresh=threshold)
            ax.set_xticks(x,labels);ax.minorticks_off();ax.set_xlim(-.07*threshold,x[-1]*1.1)
            if row%2:ax.axhline(0,color="#555555",linestyle="--",linewidth=.7)
            else:ax.set_ylim(bottom=0)
        axes[-1,col].set_xlabel("Minimum withheld events" if unit=="events" else "Snapshot time lag")
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc="upper center",ncol=4,frameon=False)
    fig.text(.5,.005,"Identical events and snapshots. Shading: pointwise 95% whole-user CI; fixed checkpoints.",ha="center",fontsize=9)
    fig.tight_layout(rect=(0,.025,1,.96));save(fig,out,f"{unit}-lags")


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=ROOT/"configs/experiments/paired-staleness-replay.json")
    parser.add_argument("--output",type=Path,default=ROOT/"data/runs/paired-staleness-replay")
    parser.add_argument("--report",type=Path,default=ROOT/"reports/experiments/paired-staleness-replay")
    parser.add_argument("--device",choices=["cpu","mps"],default="mps")
    args=parser.parse_args();config=read(args.config);torch.set_num_threads(config["threads"])
    if args.output.exists() or args.report.exists():parser.error("Choose unused output directories")
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK","0")!="0":parser.error("Disable CPU fallback")
    reference=ROOT/config["reference_replay"];old=read(reference/"results.json");raw=reference/"replay.npz"
    old_verification=read(ROOT/"reports/experiments/sasrec-256-replay/verification.json")
    assert checksum(raw)==config["reference_arrays_sha256"]==old_verification["raw_arrays_sha256"]
    conditions=settings(config);assert conditions==old["conditions"] and config["splits"]==old["config"]["splits"]
    data=ROOT/old["config"]["data"];assert checksum(data/"manifest.json")==old["data_manifest_sha256"]
    hashes={str((data/p).relative_to(ROOT)):h for p,h in old["used_data_files_sha256"].items()}
    sources=[args.config,Path(__file__)]+[ROOT/"scripts"/p for p in ["replay_sasrec.py","sasrec.py","run_sasrec.py","compare_sasrec.py","distill_sasrec.py","blend_sasrec.py","plot_sasrec_replay.py"]]
    hashes.update({str(p.resolve().relative_to(ROOT)):checksum(p) for p in sources})
    hashes.update({str((reference/p).relative_to(ROOT)):checksum(reference/p) for p in ["results.json","replay.npz"]})
    prior=ROOT/config["fresh_comparison"];features=ROOT/config["fresh_features"]
    assert checksum(prior)==config["fresh_arrays_sha256"]
    known=dict(np.load(prior));v=dict(np.load(features));replay=dict(np.load(raw))
    proof=read(ROOT/"reports/experiments/paired-seed-repeat/comparison.json")
    assert checksum(features)==proof["source_sha256"][str(features.relative_to(ROOT))]
    order=np.argsort(known["event_ids"]);take=order[np.searchsorted(known["event_ids"][order],replay["event_ids"])]
    for key in ["event_ids","targets","users","split_idx"]:assert np.array_equal(known[key][take],replay[key]),key
    assert np.array_equal(v["event_ids"],known["event_ids"])
    assert np.array_equal(v["queries"][take],replay["queries"][replay["refs"][0]])
    assert np.all(replay["query_users"][replay["refs"]]==replay["users"][None])
    assert np.all(replay["query_last_timestamp"][replay["refs"]]<replay["timestamp"][None]-replay["lag_seconds"])
    hashes.update({str(p.relative_to(ROOT)):checksum(p) for p in [prior,features]})
    for p,h in hashes.items():assert checksum(ROOT/p)==h,p
    args.output.mkdir(parents=True);started=time.perf_counter();values={};models={}
    sample=np.unravel_index(np.random.default_rng(config["seed"]).choice(replay["ranks"].size,256,replace=False),replay["ranks"].shape)
    for name,source in config["models"].items():
        folder=ROOT/source["run"];trained=read(folder/"training.json")
        assert trained["checkpoint_sha256"]==checksum(folder/"best.pt")==source["checkpoint_sha256"]
        assert source["checkpoint_sha256"]==proof["source_sha256"][str((folder/"best.pt").relative_to(ROOT))]
        assert trained["data_manifest_sha256"]==old["data_manifest_sha256"] and trained["config"]["length"]==50
        assert trained["config"]["seed"]==int(name[-2:])
        if name.startswith("student"):
            assert trained["config"]["teacher_alpha"]==0 and trained["config"]["distillation_temperature"]==trained["config"]["distillation_weight"]==1
        hashes.update({str((folder/p).relative_to(ROOT)):checksum(folder/p) for p in ["training.json","best.pt"]})
        model,base=load_student(folder,args.device);state=state_digest(model)
        assert model_args(base)==model_args(read(ROOT/config["models"]["ce42"]["run"]/"training.json")["config"])
        if name=="ce42":
            assert source["checkpoint_sha256"]==old["checkpoint_sha256"] and state==old["frozen_state_sha256"]
            rank=replay["ranks"];timing={"reused_reference_ranks":True};cpu_check=old["cpu_check"]
        else:
            vectors,seconds=encode_queries(model,replay["queries"],args.device,config["batch_size"])
            rank,timing=rank_vectors(model,vectors,replay["refs"],replay["targets"],args.device,config["batch_size"])
            timing.update(reused_reference_ranks=False,encode_unique_seconds=seconds);del vectors
            cpu,_=load_student(folder,"cpu");lag,idx=sample
            expected=ranks(cpu,replay["queries"][replay["refs"][lag,idx]],replay["targets"][idx],"cpu",config["batch_size"])
            assert quality(expected)==quality(rank[lag,idx])
            cpu_check={"pairs":len(idx),"same_metrics":True,"exact_rank_matches":int((expected==rank[lag,idx]).sum()),
                "maximum_rank_difference":int(np.abs(expected-rank[lag,idx]).max())};del cpu
        fresh_check={}
        for i,split in enumerate(config["splits"],1):
            mask=replay["split_idx"]==i;expected=known[name][take][mask];actual=rank[0,mask]
            assert quality(actual)==quality(expected)
            fresh_check[split]={"metrics_equal":True,"exact_ranks":int((actual==expected).sum()),"max_rank_delta":int(np.abs(actual-expected).max())}
        assert state_digest(model)==state
        models[name]={**source,"best_epoch":trained["best_epoch"],"frozen_state_sha256":state,"frozen_state_unchanged":True,
            "results":summarize(replay,rank,conditions,config["splits"],config["seed"]),
            "fresh_check":fresh_check,"cpu_check":cpu_check,"execution":timing};values[name]=rank
        print(json.dumps({"model":name,"execution":timing,"fresh_check":fresh_check}),flush=True);del model
    result={"config":config,"conditions":conditions,"coverage":old["coverage"],"models":models,"comparisons":{},
        "unique_snapshots":len(replay["queries"]),"targets":len(replay["targets"]),"catalog_items":114125,
        "no_test_events_read":True,"new_training_runs":0,"device":args.device,"torch":torch.__version__,"numpy":np.__version__,
        "evaluation_seconds":time.perf_counter()-started,**memory(args.device)}
    comparison_rows=[];quality_rows=[]
    for i,split in enumerate(config["splits"],1):
        mask=replay["split_idx"]==i;result["comparisons"][split]={}
        for seed in [42,43]:
            effect=compare_pair(values[f"student{seed}"][:,mask],values[f"ce{seed}"][:,mask],replay["users"][mask],config["seed"])
            result["comparisons"][split][str(seed)]=effect
            for metric,stats in effect.items():
                for j,c in enumerate(conditions):
                    entry={"split":split,"seed":seed,"metric":metric,**c}
                    for key,rows in stats.items():
                        entry[key]=rows[j]["difference"];lo,hi=rows[j]["cluster_bootstrap_95ci"]
                        entry.update({f"{key}_ci_low":lo,f"{key}_ci_high":hi})
                    comparison_rows.append(entry)
        for name,model in models.items():
            for c in model["results"][split]["conditions"]:
                entry={"split":split,"model":name,**{k:c[k] for k in ["name","unit","lag","unchanged_context_fraction"]},**c["metrics"]}
                for key,stat in c["stale_minus_fresh"].items():
                    entry.update({f"{key}_stale_minus_fresh":stat["difference"],f"{key}_retained_fraction":c["metrics"][key]/model["results"][split]["conditions"][0]["metrics"][key]})
                quality_rows.append(entry)
    for path,h in hashes.items():assert checksum(ROOT/path)==h,path
    result["source_sha256"]=hashes
    np.savez_compressed(args.output/"ranks.npz",**values,targets=replay["targets"],users=replay["users"],event_ids=replay["event_ids"],split_idx=replay["split_idx"])
    result["ranks_sha256"]=checksum(args.output/"ranks.npz")
    args.report.mkdir(parents=True);write(args.report/"config.json",config)
    for name,rows in [("quality.csv",quality_rows),("comparisons.csv",comparison_rows)]:
        with (args.report/name).open("w",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n");writer.writeheader();writer.writerows(rows)
    plt.rcParams.update({"axes.spines.top":False,"axes.spines.right":False,"axes.grid":True,"grid.alpha":.2})
    for unit in ["events","seconds"]:plot(result,args.report,unit)
    result["elapsed_seconds"]=time.perf_counter()-started
    write(args.output/"results.json",result);write(args.report/"results.json",result)
    print(json.dumps({"completed":True,"elapsed_seconds":result["elapsed_seconds"],"report":str(args.report)}),flush=True)


if __name__=="__main__":main()
