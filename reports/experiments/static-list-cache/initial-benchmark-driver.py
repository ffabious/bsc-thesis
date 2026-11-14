#!/usr/bin/env python3
"""Compare complete vector-cache and static-list-cache paths on identical requests."""

import argparse
import json
import time
from pathlib import Path

import duckdb
import numpy as np
import torch

from blend_sasrec import checksum
from distill_sasrec import load_student
from list_cache import canonical_topk,previous_states,refresh_decision
from refresh_controller import HistoryFeatures,make_predictor
from replay_sasrec_policies import frontier,policy_cost,select_budgets
from run_sasrec import ROOT,memory,state_digest,write
from sasrec import sync


def read(path):return json.loads(path.read_text())


@torch.inference_mode()
def measure(model,vectors,top100,stream,features,controller,cases,config):
    rng=np.random.default_rng(config["seed"]);n=config["benchmark_steps"]*config["benchmark_repeats"]
    operations=[];metadata={};trials={};predict=make_predictor(controller)
    for case in cases:
        take=rng.choice(case["pool"],n,replace=True)
        for regime in ["vector","list"]:
            key=f"{regime}/{case['name']}";trials[key]=[]
            cache=vectors[case["refs"][take]].copy();lists=top100[case["refs"][take]].copy()
            def operation(k,case=case,take=take,cache=cache,lists=lists,regime=regime):
                i=take[k];update=refresh_decision(stream,features,predict,case,i)
                assert update==case["refresh"][i]
                if update:
                    query=model(torch.as_tensor(stream["queries"][i:i+1],dtype=torch.long,device="mps"))
                    cache[k:k+1]=query.cpu().numpy()
                elif regime=="list":return lists[k].copy()
                else:query=torch.as_tensor(cache[k:k+1],device="mps")
                output=canonical_topk(query@model.item.weight[2:].T)[0];sync("mps")
                if regime=="list":lists[k]=output;return lists[k].copy()
                return output
            operations.append((key,operation))
            metadata[key]={"sample_requests":n,"sample_refreshes":int(case["refresh"][take].sum()),"unique_sample_requests":len(np.unique(take))}
    for _,operation in operations:
        for k in range(config["benchmark_warmup"]):operation(k%n)
    sync("mps");orders=[]
    for repeat in range(config["benchmark_repeats"]):
        order=rng.permutation(len(operations));orders.append([operations[j][0] for j in order])
        for j in order:
            key,operation=operations[j];started=time.perf_counter()
            for k in range(repeat*config["benchmark_steps"],(repeat+1)*config["benchmark_steps"]):operation(k)
            trials[key].append(1000*(time.perf_counter()-started)/config["benchmark_steps"])
    return {"cases":{key:{**metadata[key],"median_mean_ms":float(np.median(values)),"repeat_mean_ms":values} for key,values in trials.items()},"interleaved_order":orders}


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--config",type=Path,required=True)
    args=parser.parse_args();config=read(args.config);torch.set_num_threads(config["threads"])
    root=ROOT/config["output"];equiv=read(root/"equivalence.json");base=ROOT/config["policy_replay"];control=ROOT/config["controller"]
    if (root/"benchmark.json").exists():parser.error("Benchmark already exists")
    for filename,h in equiv["artifacts_sha256"].items():assert checksum(root/filename)==h,filename
    for path,h in equiv["source_sha256"].items():assert checksum(ROOT/path)==h,path
    stream=dict(np.load(base/"replay.npz"));c=dict(np.load(control/"replay.npz"));meta=read(base/"results.json");learned=read(control/"results.json")
    vectors=np.load(root/"vectors.npy");top100=np.load(root/"top100.npy");predictions=dict(np.load(root/"ranks.npz"))
    data=ROOT/meta["config"]["data"];start=next(p["start"] for p in read(data/"manifest.json")["periods"] if p["split"]=="controller_train")
    with duckdb.connect(config={"threads":4,"memory_limit":"512MB"}) as db:
        prefix=db.execute("SELECT user_idx,timestamp FROM read_parquet(?) WHERE timestamp>=? AND timestamp<?",
            [str(data/"events/model_valid.parquet"),start-86400,start]).fetchnumpy()
    features=HistoryFeatures(stream,prefix["user_idx"],prefix["timestamp"])
    choices=[]
    for j,policy in enumerate(meta["policies"]):
        choices.append({"name":policy["name"],"family":policy["family"],"policy":policy,"refs":stream["refs"][j],"refresh":stream["refresh"][j]})
    for j,name in enumerate(c["names"]):
        threshold=next(p["threshold"] for p in learned["selection"]["candidates"] if p["name"]==name)
        choices.append({"name":f"mlp42/{name}","family":f"mlp42/{name}","threshold":threshold,"refs":c["refs"][j],"refresh":c["refresh"][j],"before":c["before"][j]})
    representatives=["always","never","events_3","time_180","hybrid_5_1800",*[f"mlp42/{name}" for name in c["names"]]]
    by_name={choice["name"]:choice for choice in choices}
    for name in representatives:
        if "before" not in by_name[name]:by_name[name]["before"]=previous_states(stream,by_name[name]["refs"])
    checkpoint=ROOT/meta["config"]["checkpoint"];model,_=load_student(checkpoint.parent,"mps");state=state_digest(model)
    paths=[args.config.resolve(),Path(__file__).resolve(),root/"equivalence.json",checkpoint,control/"model.json"]
    paths += [root/name for name in equiv["artifacts_sha256"]]+[base/"replay.npz",base/"results.json",control/"replay.npz",control/"results.json"]
    paths += [ROOT/"scripts"/name for name in ["list_cache.py","refresh_controller.py","sasrec.py","replay_sasrec_policies.py"]]
    hashes={str(path.relative_to(ROOT)):checksum(path) for path in paths};started=time.perf_counter()
    result={"config":config,"calibration":{},"results":{},"selection":{},"mixed":{},"source_sha256":hashes,"no_test_events_read":True,
        "serving_scope":"Complete B=1 policy/CPU prediction, vector read/write, scoring, canonical top100 CPU output; list mode also writes/reads/copies top100; history loading/network excluded",
        "controller_scope":"The three thresholds fixed in H2 are reused, without retuning for the new serving regime",
        "family_representatives":representatives}
    for idx,split in enumerate(["controller_train","policy_valid"],1):
        pool=np.flatnonzero(stream["split_idx"]==idx);cases=[]
        for name in representatives:
            choice=by_name[name]
            for branch,update in [("refresh_write_serve",True),("cache_hit_serve",False)]:
                subset=pool[choice["refresh"][pool]==update]
                if len(subset):cases.append({**choice,"name":f"{choice['family']}/{branch}","pool":subset})
        calibration=measure(model,vectors,top100,stream,features,learned["model"],cases,config);result["calibration"][split]=calibration
        result["results"][split]={};result["selection"][split]={};checks=set(["always","never",*representatives])
        for regime in ["vector","list"]:
            rows=[]
            for choice in choices:
                branches={branch:calibration["cases"].get(f"{regime}/{choice['family']}/{branch}",calibration["cases"][f"{regime}/events/cache_hit_serve"])
                    for branch in ["refresh_write_serve","cache_hit_serve"]}
                quality=next(p["cached_top100"] for p in equiv["comparisons"] if p["split"]==split and p["name"]==choice["name"])
                updates=int(choice["refresh"][pool].sum())
                rows.append({"name":choice["name"],"metrics":quality,"refreshes":updates,"requests":len(pool),"refresh_fraction":updates/len(pool),
                    "cost":policy_cost(updates,len(pool),0,branches)})
            selection=select_budgets(rows,config["budgets"]) if split=="controller_train" else result["selection"]["controller_train"][regime]
            checks.update(p["policy"] for p in selection if p["policy"])
            result["results"][split][regime]={"policies":rows,"frontiers":{metric:frontier(rows,metric) for metric in ["ndcg@10","recall@100"]}}
            result["selection"][split][regime]=selection
        # CT chooses all published budget points; the complete PV frontier is descriptive.
        if split=="policy_valid":checks.update(p["policy"] for regime in result["selection"]["controller_train"].values() for p in regime if p["policy"])
        for name in checks:
            if "before" not in by_name[name]:by_name[name]["before"]=previous_states(stream,by_name[name]["refs"])
        mixed=measure(model,vectors,top100,stream,features,learned["model"],[{**by_name[name],"pool":pool} for name in sorted(checks)],config)
        for name,row in mixed["cases"].items():
            regime,policy=name.split("/",1);choice=by_name[policy]
            branches={branch:calibration["cases"].get(f"{regime}/{choice['family']}/{branch}",calibration["cases"][f"{regime}/events/cache_hit_serve"]) for branch in ["refresh_write_serve","cache_hit_serve"]}
            predicted=policy_cost(row["sample_refreshes"],row["sample_requests"],0,branches)["estimated_ms_per_request"]
            row.update(predicted_ms=predicted,error_percent=100*(row["median_mean_ms"]/predicted-1))
        result["mixed"][split]=mixed
        print(json.dumps({"split":split,"selected":result["selection"][split],"max_mixed_error":max(abs(p["error_percent"]) for p in mixed["cases"].values())}),flush=True)
    assert state_digest(model)==state
    for path,h in hashes.items():assert checksum(ROOT/path)==h,path
    result.update(total_seconds=time.perf_counter()-started,memory=memory("mps"),frozen_state_unchanged=True)
    write(root/"benchmark.json",result)


if __name__=="__main__":main()
