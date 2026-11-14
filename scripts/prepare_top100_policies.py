#!/usr/bin/env python3
"""Prepare fixed-grid CT/PV quality references with exact top100, not full ranks."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from blend_sasrec import checksum
from distill_sasrec import load_student
from list_cache import canonical_topk,list_ranks
from replay_sasrec import encode_queries,quality
from replay_sasrec_policies import branch_costs,load_requests,schedule,select_budgets,summarize_split
from run_sasrec import ROOT,memory,state_digest,write
from sasrec import ranks


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--config",type=Path,required=True)
    args=parser.parse_args();config=json.loads(args.config.read_text());torch.set_num_threads(config["threads"])
    output=ROOT/config["output"]
    if output.exists():parser.error("Choose a new quality-reference output")
    grid=ROOT/config["grid_reference"]["path"];assert checksum(grid)==config["grid_reference"]["sha256"]
    choices=json.loads(grid.read_text())["policies"];assert len(choices)==106 and choices[0]["family"]=="always"
    data=ROOT/config["data"];manifest=json.loads((data/"manifest.json").read_text());checkpoint=ROOT/config["checkpoint"]
    training=json.loads(checkpoint.with_name("training.json").read_text())
    assert checksum(checkpoint)==config["checkpoint_sha256"]==training["checkpoint_sha256"]
    assert checksum(data/"manifest.json")==training["data_manifest_sha256"]
    used={f"events/{name}.parquet" for name in ["train","model_valid","controller_train","policy_valid"]}|{"items.parquet","users.parquet"}
    data_hashes={row["path"]:row["sha256"] for row in manifest["files"] if row["path"] in used}
    assert len(data_hashes)==len(used)
    for path,h in data_hashes.items():assert checksum(data/path)==h,path
    paths=[args.config.resolve(),Path(__file__).resolve(),grid,checkpoint,data/"manifest.json"]
    paths += [ROOT/"scripts"/name for name in ["list_cache.py","replay_sasrec_policies.py","replay_sasrec.py","sasrec.py","run_sasrec.py","distill_sasrec.py"]]
    hashes={str(path.relative_to(ROOT)):checksum(path) for path in paths};started=time.perf_counter()
    model,_=load_student(checkpoint.parent,"mps");state=state_digest(model)
    assert model.length==50 and model.item.num_embeddings==manifest["candidate_items"]+2
    stream=load_requests(data,config["splits"],model.length);refs=[];updates=[];decision=[]
    print(json.dumps({"requests":len(stream["queries"]),"targets":len(stream["targets"]),"catalog":manifest["candidate_items"],"policies":len(choices)}),flush=True)
    for policy in choices:
        r,f=schedule(stream,policy,manifest["selected_users"]);refs.append(r);updates.append(f);trials=[]
        for _ in range(3):
            row=[]
            for idx in [1,2]:
                take=stream["split_idx"]==idx;part={key:stream[key][take] for key in ["users","timestamp","history","split_idx"]}
                begin=time.perf_counter();schedule(part,policy,manifest["selected_users"]);row.append(1000*(time.perf_counter()-begin)/take.sum())
            trials.append(row)
        decision.append(np.median(trials,axis=0).tolist())
    refs,updates=np.array(refs),np.array(updates);assert np.array_equal(refs[0],np.arange(len(stream["queries"])))
    assert np.all(stream["users"][refs]==stream["users"][None]) and np.all(stream["split_idx"][refs]==stream["split_idx"][None])
    assert np.all(stream["last_input_timestamp"][refs]<stream["timestamp"][None])
    vectors,encoding=encode_queries(model,stream["queries"],"mps",256)
    top100=np.empty((len(vectors),100),dtype=np.int64);scoring=time.perf_counter()
    for begin in range(0,len(vectors),256):
        scores=vectors[begin:begin+256]@model.item.weight[2:].T;top100[begin:begin+len(scores)]=canonical_topk(scores)
        if begin%51200==0:print(json.dumps({"snapshot_lists":begin+len(scores),"total":len(vectors)}),flush=True)
    scoring=time.perf_counter()-scoring;rank=list_ranks(top100,refs[:,stream["target_request"]],stream["targets"])
    result={"config":config,"policies":choices,"results":{},"branch_costs":{},"checkpoint_sha256":config["checkpoint_sha256"],
        "data_manifest_sha256":training["data_manifest_sha256"],"used_data_files_sha256":data_hashes,"source_sha256":hashes,
        "frozen_state_sha256":state,"device":"mps","rank_scope":"Exact top100 positions; all other ranks are the outside-top100 sentinel 1000000; not valid for MRR or full-rank analysis"}
    for idx,split in enumerate(config["splits"],1):
        branches=branch_costs(model,stream,vectors,idx,config,"mps");result["branch_costs"][split]=branches
        take=stream["split_idx"][stream["target_request"]]==idx
        result["results"][split]=summarize_split(stream,rank[:,take],refs,updates,choices,idx,branches,decision,config["seed"])
    result["selection"]={"criterion":"max CT NDCG, ties Recall then lower vector-cache full cost",
        "budgets":select_budgets(result["results"]["controller_train"]["policies"],config["cost_budgets_relative_to_always"])}
    p,t=np.unravel_index(np.random.default_rng(42).choice(rank.size,256,replace=False),rank.shape)
    cpu,_=load_student(checkpoint.parent,"cpu")
    expected=ranks(cpu,stream["queries"][refs[p,stream["target_request"][t]]],stream["targets"][t],"cpu",256)
    assert quality(expected)==quality(rank[p,t]) and np.array_equal(expected<=100,rank[p,t]<=100)
    assert state_digest(model)==state
    for path,h in hashes.items():assert checksum(ROOT/path)==h,path
    for path,h in data_hashes.items():assert checksum(data/path)==h,path
    output.mkdir(parents=True);np.save(output/"top100.npy",top100)
    np.savez_compressed(output/"replay.npz",**stream,refs=refs,refresh=updates,ranks=rank)
    inside=expected<=100;result.update(top100_cache_sha256=checksum(output/"top100.npy"),replay_sha256=checksum(output/"replay.npz"),
        cpu_oracle={"pairs":256,"same_metrics":True,"same_top100_membership":True,"max_top100_rank_delta":int(np.abs(expected[inside]-rank[p,t][inside]).max(initial=0))},
        execution={"encoding_seconds":encoding,"scoring_seconds":scoring,"total_seconds":time.perf_counter()-started},memory=memory("mps"),no_test_events_read=True)
    write(output/"results.json",result);print(json.dumps({"completed":True,"execution":result["execution"],"cpu_oracle":result["cpu_oracle"]}),flush=True)


if __name__=="__main__":main()
