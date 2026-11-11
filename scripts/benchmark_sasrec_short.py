#!/usr/bin/env python3
"""Paired native/padded recent timing, using the same frozen models and CT inputs."""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from blend_sasrec import checksum,load_pair
from profile_sasrec_hybrid import Serving
from run_sasrec import ROOT,memory,write
from sasrec import sync


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--device",choices=["cpu","mps"],required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    if args.output.exists() or os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK","0")!="0":
        parser.error("Choose a new output and disable CPU fallback")
    config = json.loads(args.config.read_text());torch.set_num_threads(config["threads"])
    run,folder = ROOT/config["hybrid_run"],ROOT/config["reference_replay"]
    hybrid = json.loads((run/"results.json").read_text())
    selection = json.loads((run/"selection.json").read_text())
    stream = dict(np.load(folder/"replay.npz"));pool = np.flatnonzero(stream["split_idx"]==1)
    models,configs,sources = load_pair(hybrid["config"],"256",args.device)
    if sources!=hybrid["sources"] or checksum(run/"selection.json")!=hybrid["selection_sha256"]:
        raise ValueError("Frozen models or alpha changed")
    services = {"short":Serving(models,configs,args.device,2),"padded":Serving(models,configs,args.device)}
    paths = ["encoder","recent","hybrid"];alpha = selection["policies"][0]["alpha"]
    records,probes = {},{}
    for batch,steps,repeats,warmup in zip(config["batch_sizes"],config["steps"],config["repeats"],config["warmup"]):
        draws = np.random.default_rng(config["seed"]+batch).choice(pool,size=(steps*repeats,batch),replace=True)
        cache = np.zeros((batch,models["long"].item.embedding_dim),dtype=np.float32)
        names = [f"{kind}_{path}" for kind in services for path in paths]
        values = {name:[] for name in names}

        def operation(name,i):
            kind,path = name.split("_");service = services[kind];x = stream["queries"][draws[i]]
            if path=="encoder":
                return service.encode("recent",x)
            return service.serve(x,cache,True,1 if path=="recent" else alpha,True)

        for name in names:
            for i in range(warmup):
                operation(name,i%len(draws));sync(args.device)
        rng = np.random.default_rng(config["seed"])
        for rep in range(repeats):
            for name in rng.permutation(names):
                sync(args.device);started = time.perf_counter()
                for i in range(rep*steps,(rep+1)*steps):
                    operation(name,i);sync(args.device)
                values[name].append((time.perf_counter()-started)*1000/steps/batch)
            print(f"paired {args.device} B={batch}, repeat {rep+1}/{repeats}",flush=True)
        records[str(batch)] = {name:{"repeat_mean_ms_per_request":v,"median_ms_per_request":float(np.median(v))} for name,v in values.items()}
        probes[str(batch)] = {}
        for kind,service in services.items():
            trace = [];service.serve(stream["queries"][draws[0]],cache,True,alpha,True,trace)
            probes[str(batch)][kind] = trace
    data = ROOT/configs["long"]["data"]
    if any(checksum(data/p)!=h for p,h in hybrid["data_files_sha256"].items()):
        raise ValueError("Inputs changed")
    write(args.output,{"config":config,"device":args.device,"sources":sources,"measurements":records,
        "memory_probes":probes,"alpha":alpha,"no_test_events_read":True,
        "source_sha256":{str(p.relative_to(ROOT)):checksum(p) for p in [ROOT/"scripts/benchmark_sasrec_short.py",
            ROOT/"scripts/profile_sasrec_hybrid.py",ROOT/"scripts/sasrec.py",folder/"replay.npz",args.config.resolve()]},**memory(args.device)})


if __name__=="__main__":
    main()
