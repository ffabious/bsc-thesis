#!/usr/bin/env python3
"""Evaluate the committed user holdout once, without fitting or selecting policies."""

import json
import time
from pathlib import Path

import duckdb
import numpy as np
import torch

from benchmark_list_cache import measure
from blend_sasrec import checksum
from distill_sasrec import load_student
from final_holdout import verified_protocol
from list_cache import canonical_topk,list_ranks,previous_states
from refresh_controller import HistoryFeatures,rollout
from replay_sasrec import encode_queries,quality
from replay_sasrec_policies import frontier,request_stream,schedule
from run_sasrec import ROOT,memory,state_digest,write
from sasrec import ranks


@torch.inference_mode()
def main():
    protocol,commit=verified_protocol();config=protocol["config"];root=ROOT/config["output"];data=root/"dataset";out=root/"evaluation"
    if out.exists():raise FileExistsError("Final evaluation already exists; do not tune on this holdout")
    manifest=json.loads((data/"manifest.json").read_text());assert manifest["freeze_commit"]==commit
    for row in manifest["files"]:assert checksum(data/row["path"])==row["sha256"],row["path"]
    assert checksum(data/"items.parquet")==protocol["vocabulary_sha256"]
    torch.set_num_threads(config["threads"]);checkpoint=ROOT/config["checkpoint"];model,_=load_student(checkpoint.parent,"mps");state=state_digest(model)
    controller_path=ROOT/config["controller"]/"model.json";controller=json.loads(controller_path.read_text());assert checksum(controller_path)==protocol["controller_model_sha256"]
    paths=[Path(__file__).resolve(),root/"protocol.json",data/"manifest.json",checkpoint,controller_path]
    paths += [ROOT/"scripts"/name for name in ["final_holdout.py","list_cache.py","benchmark_list_cache.py","refresh_controller.py","replay_sasrec_policies.py","replay_sasrec.py","sasrec.py"]]
    hashes={str(path.relative_to(ROOT)):checksum(path) for path in paths};started=time.perf_counter()
    stream=request_stream(data,["test"],model.length);start=next(p["start"] for p in protocol["periods"] if p["split"]=="test")
    assert np.all(stream["timestamp"]>=start) and np.all(stream["split_idx"]==1)
    with duckdb.connect(config={"threads":4,"memory_limit":"512MB"}) as db:
        prefix=db.execute("SELECT user_idx,timestamp FROM read_parquet(?) WHERE timestamp>=? AND timestamp<?",
            [str(data/"events/policy_valid.parquet"),start-86400,start]).fetchnumpy()
    features=HistoryFeatures(stream,prefix["user_idx"],prefix["timestamp"]);choices=[]
    for policy in protocol["policies"]:
        refs,updates=schedule(stream,policy,config["users"])
        choices.append({"name":policy["name"],"policy":policy,"refs":refs,"refresh":updates,"before":previous_states(stream,refs)})
    for policy in protocol["controllers"]:
        refs,updates,before=rollout(stream,features,controller,policy["threshold"])
        choices.append({"name":f"controller/{policy['name']}","threshold":policy["threshold"],"refs":refs,"refresh":updates,"before":before})
    refs=np.array([choice["refs"] for choice in choices]);updates=np.array([choice["refresh"] for choice in choices])
    assert np.all(stream["users"][refs]==stream["users"][None]) and np.all(stream["last_input_timestamp"][refs]<stream["timestamp"][None])
    print(json.dumps({"requests":len(stream["queries"]),"targets":len(stream["targets"]),"policies":len(choices),"freeze_commit":commit}),flush=True)
    vectors,encoding=encode_queries(model,stream["queries"],"mps",256);top100=np.empty((len(vectors),100),dtype=np.int64);scoring=time.perf_counter()
    for begin in range(0,len(vectors),256):
        scores=vectors[begin:begin+256]@model.item.weight[2:].T;top100[begin:begin+len(scores)]=canonical_topk(scores)
        if begin%25600==0:print(json.dumps({"snapshot_lists":begin+len(scores),"total":len(vectors)}),flush=True)
    scoring=time.perf_counter()-scoring;rank=list_ranks(top100,refs[:,stream["target_request"]],stream["targets"])
    out.mkdir();np.save(out/"top100.npy",top100)
    np.savez_compressed(out/"replay.npz",**stream,refs=refs,refresh=updates,ranks=rank,names=np.array([choice["name"] for choice in choices]))
    write(out/"quality.json",{"policies":[{"name":choice["name"],"metrics":quality(rank[j]),"refreshes":int(updates[j].sum()),
        "refresh_fraction":float(updates[j].mean())} for j,choice in enumerate(choices)],"requests":len(stream["queries"]),"targets":len(stream["targets"]),
        "encoding_seconds":encoding,"scoring_seconds":scoring,"protocol_sha256":checksum(root/"protocol.json"),"freeze_commit":commit,
        "rank_scope":"Exact top100, outside sentinel1000000","source_sha256":hashes,"no_refit_or_selection":True})
    pool=np.arange(len(stream["queries"]));cpu_vectors=vectors.cpu().numpy()
    print(json.dumps({"quality_complete":True,"direct_serving_cases":2*len(choices)}),flush=True)
    benchmark_config={**config,"seed":config["bootstrap_seed"]}
    measured=measure(model,cpu_vectors,top100,stream,features,controller,[{**choice,"pool":pool} for choice in choices],benchmark_config,stratified=True)
    summary={}
    for regime in ["vector","list"]:
        rows=[]
        for j,choice in enumerate(choices):
            sample=measured["cases"][f"{regime}/{choice['name']}"]
            rows.append({"name":choice["name"],"metrics":quality(rank[j]),"refresh_fraction":float(updates[j].mean()),
                "sample_refresh_fraction":sample["sample_refreshes"]/sample["sample_requests"],"cost":{"estimated_ms_per_request":sample["mean_ms"]}})
        summary[regime]={"policies":rows,"frontiers":{metric:frontier(rows,metric) for metric in ["ndcg@10","recall@100"]}}
    p,t=np.unravel_index(np.random.default_rng(42).choice(rank.size,256,replace=False),rank.shape);cpu,_=load_student(checkpoint.parent,"cpu")
    expected=ranks(cpu,stream["queries"][refs[p,stream["target_request"][t]]],stream["targets"][t],"cpu",256)
    assert quality(expected)==quality(rank[p,t]) and np.array_equal(expected<=100,rank[p,t]<=100)
    assert state_digest(model)==state
    for path,h in hashes.items():assert checksum(ROOT/path)==h,path
    inside=expected<=100
    result={"results":summary,"measurements":measured,"source_sha256":hashes,"freeze_commit":commit,"protocol_sha256":checksum(root/"protocol.json"),
        "replay_sha256":checksum(out/"replay.npz"),"cpu_oracle":{"pairs":256,"same_metrics":True,"same_top100_membership":True,
        "max_top100_rank_delta":int(np.abs(expected[inside]-rank[p,t][inside]).max(initial=0))},"frozen_state_unchanged":True,"no_training_or_selection_on_holdout":True,
        "memory":memory("mps"),"total_seconds":time.perf_counter()-started,
        "cost_scope":"Direct paired complete B=1 policy/controller + vector/list serving, stratified384-request samples, whole-sample mean; full test frontier descriptive, choices frozen on development CT"}
    write(out/"results.json",result);print(json.dumps({"completed":True,"seconds":result["total_seconds"],"cpu_oracle":result["cpu_oracle"]}),flush=True)


if __name__=="__main__":main()
