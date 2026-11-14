#!/usr/bin/env python3
"""Verify static top100 cache predictions against existing vector-cache replays."""

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
from run_sasrec import ROOT,memory,state_digest,write


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--config",type=Path,required=True)
    args=parser.parse_args();config=json.loads(args.config.read_text());torch.set_num_threads(config["threads"])
    base=ROOT/config["policy_replay"];control=ROOT/config["controller"];out=ROOT/config["output"]
    if out.exists():parser.error("Choose a new cache output")
    meta=json.loads((base/"results.json").read_text());v=dict(np.load(base/"replay.npz"));c=dict(np.load(control/"replay.npz"))
    for key in ["targets","target_request","target_event_ids"]:np.testing.assert_array_equal(v[key],c[key])
    for path,key in [(base/"replay.npz","policy_arrays_sha256"),(base/"results.json","policy_results_sha256"),
                     (control/"replay.npz","controller_arrays_sha256"),(control/"results.json","controller_results_sha256")]:assert checksum(path)==config[key]
    model_path=ROOT/meta["config"]["checkpoint"];assert checksum(model_path)==meta["checkpoint_sha256"]
    paths=[args.config.resolve(),Path(__file__).resolve(),base/"replay.npz",base/"results.json",control/"replay.npz",control/"results.json",model_path]
    paths += [ROOT/"scripts"/name for name in ["list_cache.py","replay_sasrec.py","sasrec.py","distill_sasrec.py"]]
    hashes={str(path.relative_to(ROOT)):checksum(path) for path in paths};started=time.perf_counter()
    model,_=load_student(model_path.parent,"mps");state=state_digest(model)
    vectors,encoding=encode_queries(model,v["queries"],"mps",256);top100=np.empty((len(vectors),100),dtype=np.int64)
    audit=0;scoring=time.perf_counter()
    for start in range(0,len(vectors),256):
        scores=vectors[start:start+256]@model.item.weight[2:].T;lists=canonical_topk(scores)
        top100[start:start+len(lists)]=lists
        if start==0:
            full=scores.cpu().numpy();ids=np.arange(full.shape[1])
            for row in range(len(lists)):
                for position in [0,9,99]:
                    target=lists[row,position];rank=1+np.count_nonzero((full[row]>full[row,target])|((full[row]==full[row,target])&(ids<target)))
                    assert rank==position+1;audit+=1
        if start%25600==0:print(json.dumps({"snapshot_lists":start+len(lists),"total":len(vectors)}),flush=True)
    scoring=time.perf_counter()-scoring
    all_refs=np.r_[v["refs"],c["refs"]];old_rank=np.r_[v["ranks"],c["ranks"]]
    assert np.all(v["users"][all_refs]==v["users"][None]) and np.all(v["split_idx"][all_refs]==v["split_idx"][None])
    assert np.all(v["last_input_timestamp"][all_refs]<v["timestamp"][None])
    new_rank=list_ranks(top100,all_refs[:,v["target_request"]],v["targets"])
    names=[p["name"] for p in meta["policies"]]+[f"mlp42/{name}" for name in c["names"]]
    comparisons=[]
    for idx,split in enumerate(["controller_train","policy_valid"],1):
        take=v["split_idx"][v["target_request"]]==idx
        for j,name in enumerate(names):
            old=old_rank[j,take];new=new_rank[j,take]
            comparisons.append({"split":split,"name":name,"old":quality(old),"cached_top100":quality(new),
                "top100_membership_differences":int(((old<=100)!=(new<=100)).sum()),
                "rank_differences_inside_top100":int(((old!=new)&(old<=100)&(new<=100)).sum())})
    assert state_digest(model)==state
    for path,h in hashes.items():assert checksum(ROOT/path)==h,path
    out.mkdir(parents=True);np.save(out/"top100.npy",top100);np.save(out/"vectors.npy",vectors.cpu().numpy())
    np.savez_compressed(out/"ranks.npz",ranks=new_rank,names=np.array(names))
    result={"config":config,"comparisons":comparisons,"canonical_same_score_rank_checks":audit,"source_sha256":hashes,
        "snapshot_count":len(top100),"checkpoint_sha256":meta["checkpoint_sha256"],"frozen_state_unchanged":True,"no_test_events_read":True,
        "execution":{"encoding_seconds":encoding,"scoring_seconds":scoring,"total_seconds":time.perf_counter()-started},
        "memory":memory("mps"),"cache_bytes_per_user":{"vector":vectors.shape[1]*4,"top100":100*8,"both":vectors.shape[1]*4+800},
        "artifacts_sha256":{name:checksum(out/name) for name in ["top100.npy","vectors.npy","ranks.npz"]},
        "equivalence_scope":"Frozen user vector/item table/full candidate set; canonical score-desc/item-id-asc top100. Old batched ranks may differ numerically across score kernels."}
    write(out/"equivalence.json",result)
    print(json.dumps({"execution":result["execution"],"audit":audit,"membership_differences":sum(p["top100_membership_differences"] for p in comparisons),
        "rank_differences":sum(p["rank_differences_inside_top100"] for p in comparisons),"max_metric_delta":max(abs(p["old"][metric]-p["cached_top100"][metric]) for p in comparisons for metric in p["old"])}),flush=True)


if __name__=="__main__":main()
