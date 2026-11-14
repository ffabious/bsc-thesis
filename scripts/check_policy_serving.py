#!/usr/bin/env python3
"""Direct paired serving checks for CT-chosen transfer policies in both regimes."""

import argparse
import json
import time
from pathlib import Path

import duckdb
import numpy as np
import torch

from benchmark_list_cache import measure
from blend_sasrec import checksum
from distill_sasrec import load_student
from list_cache import previous_states
from refresh_controller import HistoryFeatures
from replay_sasrec import encode_queries
from run_sasrec import ROOT,state_digest,write


def read(path):return json.loads(path.read_text())


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--run",type=Path,required=True)
    args=parser.parse_args();root=args.run.resolve();result=read(root/"results.json");config=result["config"];torch.set_num_threads(config["threads"])
    output=root/"direct-serving.json"
    if output.exists():parser.error("Serving check already exists")
    base=ROOT/config["policy_replay"];meta=read(base/"results.json");stream=dict(np.load(base/"replay.npz"));c=dict(np.load(root/"replay.npz"))
    top_file=ROOT/meta["config"]["top100_cache"];assert checksum(top_file)==meta["top100_cache_sha256"]
    top100=np.load(top_file);priced=read(root/"policy-results.json");data=ROOT/meta["config"]["data"]
    start=next(p["start"] for p in read(data/"manifest.json")["periods"] if p["split"]=="controller_train")
    with duckdb.connect(config={"threads":4,"memory_limit":"512MB"}) as db:
        prefix=db.execute("SELECT user_idx,timestamp FROM read_parquet(?) WHERE timestamp>=? AND timestamp<?",
            [str(data/"events/model_valid.parquet"),start-86400,start]).fetchnumpy()
    features=HistoryFeatures(stream,prefix["user_idx"],prefix["timestamp"])
    names={"always","never","events_3","time_180","time_1800","time_86400"}
    names.update(p["matched_simple"] for p in result["selection"]["budgets"] if p["matched_simple"])
    names.update(p["policy"] for p in priced["selection"]["budgets"] if p["policy"])
    choices=[]
    for name in sorted(names):
        j=next(j for j,p in enumerate(meta["policies"]) if p["name"]==name)
        choices.append({"name":name,"policy":meta["policies"][j],"refs":stream["refs"][j],"refresh":stream["refresh"][j],"before":previous_states(stream,stream["refs"][j])})
    for j,name in enumerate(c["names"]):
        threshold=next(p["threshold"] for p in result["selection"]["candidates"] if p["name"]==name)
        choices.append({"name":f"controller/{name}","threshold":threshold,"refs":c["refs"][j],"refresh":c["refresh"][j],"before":c["before"][j]})
    checkpoint=ROOT/meta["config"]["checkpoint"];model,_=load_student(checkpoint.parent,"mps");state=state_digest(model)
    paths=[Path(__file__).resolve(),root/"results.json",root/"replay.npz",root/"policy-results.json",base/"results.json",base/"replay.npz",top_file,checkpoint]
    paths += [ROOT/"scripts"/name for name in ["benchmark_list_cache.py","list_cache.py","refresh_controller.py","replay_sasrec.py","sasrec.py"]]
    hashes={str(path.relative_to(ROOT)):checksum(path) for path in paths};started=time.perf_counter()
    vectors,encoding=encode_queries(model,stream["queries"],"mps",256);vectors=vectors.cpu().numpy();blocks={};summary={}
    for idx,split in enumerate(["controller_train","policy_valid"],1):
        pool=np.flatnonzero(stream["split_idx"]==idx)
        blocks[split]=measure(model,vectors,top100,stream,features,result["model"],[{**choice,"pool":pool} for choice in choices],config,stratified=True)
        summary[split]={}
        for regime in ["vector","list"]:
            rows=[]
            for choice in choices:
                sample=blocks[split]["cases"][f"{regime}/{choice['name']}"];name=choice["name"]
                source=result["results"][split] if name.startswith("controller/") else priced["results"][split]["policies"]
                quality=next(p["metrics"] for p in source if p["name"]==name.split("/",1)[-1])
                rows.append({"name":name,"metrics":quality,"refresh_fraction":float(choice["refresh"][pool].mean()),
                    "sample_refresh_fraction":sample["sample_refreshes"]/sample["sample_requests"],"cost_ms":sample["mean_ms"]})
            summary[split][regime]=rows
        print(json.dumps({"split":split,"cases":len(choices),"completed":True}),flush=True)
    assert state_digest(model)==state
    for path,h in hashes.items():assert checksum(ROOT/path)==h,path
    write(output,{"results":summary,"measurements":blocks,"source_sha256":hashes,"encoding_seconds":encoding,
        "total_seconds":time.perf_counter()-started,"no_test_events_read":True,"frozen_state_unchanged":True,
        "scope":"Paired complete B=1 vector/list paths; fixed CT-selected controllers/comparators and preregistered resource controls; stratified samples384 per policy/period, whole-sample mean"})


if __name__=="__main__":main()
