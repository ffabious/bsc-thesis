#!/usr/bin/env python3
"""Locate the frozen student's validation/PV gap without retraining or test reads."""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import duckdb
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"scripts"))
from blend_sasrec import checksum,load_pair,score_ranks
from compare_sasrec import paired_intervals
from distill_sasrec import Teacher,load_student,validation
from replay_sasrec import quality
from run_sasrec import write
from sasrec import examples,metrics,ranks

NAMES=["model_valid","controller_train","policy_valid"]
MODELS=["control","student","teacher"]
STRATA=["last_item","second_last_only","earlier_in_window","outside_window"]


def ndcg(rank):
    return np.where(rank<=10,1/np.log2(rank+1),0)


def inputs(data):
    files=[str(data/"events"/f"{name}.parquet") for name in ["train"]+NAMES]
    case="CASE filename "+" ".join(f"WHEN ? THEN {i}" for i in range(3))+" ELSE -1 END"
    with duckdb.connect(config={"threads":4,"memory_limit":"1GB"}) as db:
        db.read_parquet(files,filename=True).create_view("events")
        v=db.execute(f"""SELECT user_idx,item_idx,timestamp,event_id,history_size,is_target,
            is_positive,is_organic,{case} AS split_idx FROM events
            ORDER BY user_idx,timestamp,event_id""",files[1:]).fetchnumpy()
    chosen=np.flatnonzero(v["is_target"]&(v["split_idx"]>=0))
    result={key:v[source][chosen] for key,source in [("users","user_idx"),("targets","item_idx"),
        ("event_ids","event_id"),("timestamp","timestamp"),("split_idx","split_idx"),("organic","is_organic")]}
    contexts=[];gaps=[];activity=[]
    starts=np.flatnonzero(np.r_[True,v["user_idx"][1:]!=v["user_idx"][:-1]])
    for start,end in zip(starts,np.r_[starts[1:],len(v["user_idx"])]):
        take=chosen[(chosen>=start)&(chosen<end)]
        if not len(take):continue
        times=v["timestamp"][start:end];h=v["history_size"][take];now=v["timestamp"][take]
        assert np.array_equal(h,np.searchsorted(times,now,side="left")) and (h>0).all()
        window=np.lib.stride_tricks.sliding_window_view(np.pad(v["item_idx"][start:end],(50,0)),50)
        contexts.append(window[h]);gaps.append(now-times[h-1])
        activity.append(h-np.searchsorted(times,now-86400,side="left"))
    x=np.concatenate(contexts).astype(np.int32);y=result["targets"]
    matches=x==y[:,None];last=matches[:,-1];second=matches[:,-2]&~last;seen=matches.any(1)
    result.update(queries=x,gap_seconds=np.concatenate(gaps),activity_24h=np.concatenate(activity),
        stratum=np.select([last,second,seen],[0,1,2],default=3))
    count=json.loads((data/"manifest.json").read_text())["candidate_items"]
    pop=np.bincount(v["item_idx"][(v["split_idx"]<0)&v["is_positive"]],minlength=count+2)
    ordered=np.lexsort((np.arange(2,count+2),-pop[2:]))+2
    pop_rank=np.zeros(count+2,dtype=np.int32);pop_rank[ordered]=np.arange(1,count+1)
    result["pop_rank"]=pop_rank[y]
    assert (result["gap_seconds"]>0).all() and (y>=2).all()
    return result


def segment(values,mask):
    count=int(mask.sum())
    if not count:return {"targets":0}
    ds=ndcg(values["student"])-ndcg(values["control"])
    dr=(values["student"]<=100).astype(int)-(values["control"]<=100)
    return {"targets":count,"share":float(mask.mean()),"users":int(len(np.unique(values["users"][mask]))),
        "metrics":{name:{**metrics(values[name][mask]),"hit@1":float((values[name][mask]==1).mean())} for name in MODELS},
        "ndcg_delta":float(ds[mask].mean()),"ndcg_contribution":float(ds[mask].sum()/len(mask)),
        "recall_delta":float(dr[mask].mean()),"recall_contribution":float(dr[mask].sum()/len(mask)),
        "ndcg_gross_loss":float(-ds[mask&(ds<0)].sum()/len(mask)),
        "ndcg_gross_gain":float(ds[mask&(ds>0)].sum()/len(mask))}


def analyze(v):
    all_events=np.ones(len(v["targets"]),bool)
    groups={"history":{name:v["stratum"]==i for i,name in enumerate(STRATA)},
        "gap":dict(zip(["up_to_5min","5_to_30min","over_30min"],
            [v["gap_seconds"]<=300,(v["gap_seconds"]>300)&(v["gap_seconds"]<=1800),v["gap_seconds"]>1800])),
        "activity":dict(zip(["up_to_20","21_to_100","over_100"],
            [v["activity_24h"]<=20,(v["activity_24h"]>20)&(v["activity_24h"]<=100),v["activity_24h"]>100])),
        "popularity":dict(zip(["top_1000","1001_to_10000","over_10000"],
            [v["pop_rank"]<=1000,(v["pop_rank"]>1000)&(v["pop_rank"]<=10000),v["pop_rank"]>10000])),
        "organic":{str(i):v["organic"]==i for i in np.unique(v["organic"])}}
    result={"all":segment(v,all_events),"paired_delta":paired_intervals(v["student"],v["control"],v["users"],42),
        "segments":{family:{name:segment(v,m) for name,m in masks.items()} for family,masks in groups.items()}}
    for family,rows in result["segments"].items():
        assert sum(row["targets"] for row in rows.values())==len(all_events),family
        assert np.isclose(sum(row.get("ndcg_contribution",0) for row in rows.values()),result["all"]["ndcg_delta"]),family
    ds=ndcg(v["student"])-ndcg(v["control"]);lost=ds<0;won=ds>0
    result["event_changes"]={"ndcg_wins":int(won.sum()),"ndcg_losses":int(lost.sum()),
        "ndcg_equal":int((ds==0).sum()),"lost_top1":int(((v["control"]==1)&(v["student"]>1)).sum()),
        "gained_top1":int(((v["student"]==1)&(v["control"]>1)).sum()),
        "teacher_not_worse_on_loss_events":float((ndcg(v["teacher"])[lost]>=ndcg(v["control"])[lost]).mean()),
        "teacher_strictly_better_on_loss_events":float((ndcg(v["teacher"])[lost]>ndcg(v["control"])[lost]).mean())}
    bucket=lambda rank:np.select([rank==1,rank<=10,rank<=100],[0,1,2],default=3)
    result["transition_labels"]=["1","2..10","11..100",">100"]
    result["rank_transitions"]=np.bincount(bucket(v["control"])*4+bucket(v["student"]),minlength=16).reshape(4,4).tolist()
    user_rows=[]
    for user in np.unique(v["users"]):
        m=v["users"]==user;row=segment(v,m);loo=(ds.sum()-ds[m].sum())/(len(ds)-m.sum())
        user_rows.append({"user_idx":int(user),"targets":row["targets"],"ndcg_delta":row["ndcg_delta"],
            "ndcg_contribution":row["ndcg_contribution"],"gross_loss_contribution":row["ndcg_gross_loss"],
            "leave_one_out_delta":float(loo),"repeat_share":float((v["stratum"][m]==0).mean())})
    user_rows.sort(key=lambda row:row["ndcg_contribution"])
    result["users"]={"macro_ndcg_delta":float(np.mean([r["ndcg_delta"] for r in user_rows])),
        "winners":sum(r["ndcg_delta"]>0 for r in user_rows),"losers":sum(r["ndcg_delta"]<0 for r in user_rows),
        "leave_one_out_range":[min(r["leave_one_out_delta"] for r in user_rows),max(r["leave_one_out_delta"] for r in user_rows)],
        "largest_losses":user_rows[:10],"largest_gains":user_rows[-10:][::-1]}
    result["days"]=[]
    for day in np.unique(v["day"]):
        result["days"].append({"day":int(day),**segment(v,v["day"]==day)})
    return result,user_rows


def decomposition(left,right):
    a=left["segments"]["history"];b=right["segments"]["history"]
    # Symmetric composition/within-stratum accounting; descriptive, not causal.
    composition=sum((b[k]["share"]-a[k]["share"])*(b[k]["ndcg_delta"]+a[k]["ndcg_delta"])/2 for k in STRATA)
    within=sum((b[k]["ndcg_delta"]-a[k]["ndcg_delta"])*(b[k]["share"]+a[k]["share"])/2 for k in STRATA)
    change=right["all"]["ndcg_delta"]-left["all"]["ndcg_delta"]
    assert np.isclose(composition+within,change)
    return {"ndcg_delta_change":change,"composition":composition,"within_stratum":within,
        "mv_delta_with_pv_mix":sum(b[k]["share"]*a[k]["ndcg_delta"] for k in STRATA),
        "pv_delta_with_mv_mix":sum(a[k]["share"]*b[k]["ndcg_delta"] for k in STRATA)}


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device",choices=["cpu","mps"],default="mps")
    parser.add_argument("--output",type=Path,default=ROOT/"reports/experiments/sasrec-256-distillation-audit")
    args=parser.parse_args();out=args.output;local=ROOT/"data/diagnostics/sasrec-256-distillation-audit"
    if out.exists() or local.exists():parser.error("Choose a new output; existing audit is immutable")
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK","0")!="0":parser.error("Disable CPU fallback")
    torch.set_num_threads(4)
    cold=ROOT/"data/runs/sasrec-256-distillation-cold";protocol=json.loads((cold/"teacher/selection.json").read_text())
    selected=json.loads((cold/"selection.json").read_text());data=ROOT/"data/processed/yambda50m-pilot-audio"
    v=inputs(data);mv=v["split_idx"]==0;later=~mv
    stream=dict(np.load(ROOT/"data/runs/sasrec-256-policies-wide/replay.npz"))
    cached=dict(np.load(cold/"evaluation-ranks.npz"));q=stream["target_request"]
    for key,expected in [("queries",stream["queries"][q]),("event_ids",stream["target_event_ids"]),
            ("targets",cached["targets"]),("users",cached["users"]),("split_idx",cached["split_idx"])]:
        assert np.array_equal(v[key][later],expected),key
    x,y,u=examples(data,"model_valid",50)
    assert np.array_equal(x,v["queries"][mv]) and np.array_equal(y,v["targets"][mv]) and np.array_equal(u,v["users"][mv])
    for name in ["control","student"]:
        folder=ROOT/selected[name]["run"];model,_=load_student(folder,args.device)
        rank=ranks(model,x,y,args.device,256);trained=json.loads((folder/"training.json").read_text())
        assert metrics(rank)==validation(trained)
        v[name]=np.empty(len(mv),dtype=np.int32);v[name][mv]=rank;v[name][later]=cached[name]
        del model
        print(f"{name}: model_valid ranks reproduced",flush=True)
    models,_,sources=load_pair(protocol["protocol"],"256",args.device)
    assert sources==protocol["sources"]
    teacher=Teacher(models,protocol["alpha"],2);ids=torch.arange(2,models["long"].item.num_embeddings,device=args.device)
    parts=[]
    for start in range(0,len(y),256):
        seq=torch.as_tensor(x[start:start+256],dtype=torch.long,device=args.device)
        target=torch.as_tensor(y[start:start+256],dtype=torch.long,device=args.device)
        parts.append(score_ranks(teacher.logits(seq),ids,target).cpu().numpy())
    v["teacher"]=np.empty(len(mv),dtype=np.int32);v["teacher"][mv]=np.concatenate(parts);v["teacher"][later]=cached["teacher"]
    assert quality(v["teacher"][mv])==next(p["metrics"] for p in protocol["curve"] if p["alpha"]==protocol["alpha"])
    periods=json.loads((data/"manifest.json").read_text())["periods"]
    out.mkdir(parents=True);local.mkdir(parents=True);result={"splits":{},"no_training":True,"no_test_events_read":True}
    rows=[];users=[]
    for i,name in enumerate(NAMES):
        take=v["split_idx"]==i;current={key:value[take] for key,value in v.items()}
        start=next(p["start"] for p in periods if p["split"]==name);current["day"]=(current["timestamp"]-start)//86400
        result["splits"][name],user_rows=analyze(current);users.extend({"split":name,**r} for r in user_rows)
        for family,groups in result["splits"][name]["segments"].items():
            for group,r in groups.items():
                if r["targets"]:rows.append({"split":name,"family":family,"segment":group,
                    **{k:r[k] for k in ["targets","share","ndcg_delta","ndcg_contribution","recall_delta","recall_contribution"]},
                    **{f"{m}_{metric}":r["metrics"][m][metric] for m in MODELS for metric in ["ndcg@10","recall@100"]}})
        print(f"{name}: {result['splits'][name]['all']['ndcg_delta']:+.6f} NDCG",flush=True)
    result["history_composition_decomposition"]=decomposition(result["splits"][NAMES[0]],result["splits"][NAMES[2]])
    common=np.intersect1d(v["users"][mv],v["users"][v["split_idx"]==2]);result["common_users"]={"count":len(common),"splits":{}}
    for i in [0,2]:
        take=(v["split_idx"]==i)&np.isin(v["users"],common);current={k:a[take] for k,a in v.items()}
        result["common_users"]["splits"][NAMES[i]]=segment(current,np.ones(take.sum(),bool))
    for name,values in [("segments.csv",rows),("users.csv",users)]:
        with (out/name).open("w",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=list(values[0]),lineterminator="\n");writer.writeheader();writer.writerows(values)
    np.savez_compressed(local/"events.npz",**v)
    paths=[cold/"selection.json",cold/"evaluation-ranks.npz",data/"manifest.json",Path(__file__).resolve()]
    paths += [ROOT/selected[name]["run"]/"best.pt" for name in ["control","student"]]
    result["source_sha256"]={str(p.relative_to(ROOT)):checksum(p) for p in paths}
    result["device"]=args.device;result["torch"]=torch.__version__
    write(out/"audit.json",result)


if __name__=="__main__":main()
