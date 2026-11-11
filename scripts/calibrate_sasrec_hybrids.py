#!/usr/bin/env python3
"""Interleave CE/BCE serving calibration when separate profiles drift in time."""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from blend_sasrec import checksum,load_pair
from profile_sasrec_hybrid import Serving,budgets,cost_rows,estimated_cost
from replay_sasrec import encode_queries
from replay_sasrec_policies import frontier
from run_sasrec import ROOT,memory,write
from sasrec import sync


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ce-profile",type=Path,required=True)
    parser.add_argument("--bce-profile",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    if args.output.exists() or os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK","0")!="0":
        parser.error("Choose a new output directory and disable CPU fallback")
    inputs = {"ce":args.ce_profile.resolve(),"bce":args.bce_profile.resolve()}
    prior = {k:json.loads(p.read_text()) for k,p in inputs.items()}
    cfg = prior["ce"]["config"];torch.set_num_threads(cfg["threads"])
    if any(p["device"]!="mps" or p["input_widths"]!={"long":50,"recent":2} for p in prior.values()):
        raise ValueError("Expected two native2 MPS profiles")
    for key in ("reference_replay","batch_sizes","steps","repeats","warmup","seed","threads"):
        if cfg[key]!=prior["bce"]["config"][key]:
            raise ValueError(f"Calibration protocols differ: {key}")
    services,hybrids,selections = {},{},{}
    for kind,p in prior.items():
        run = ROOT/p["config"]["hybrid_run"]
        hybrids[kind] = json.loads((run/"results.json").read_text())
        selections[kind] = json.loads((run/"selection.json").read_text())
        models,configs,sources = load_pair(hybrids[kind]["config"],"256","mps")
        if sources!=p["sources"]:
            raise ValueError("Models changed")
        if kind=="bce":
            if sources["long"]!=prior["ce"]["sources"]["long"]:
                raise ValueError("Long checkpoint differs")
            models["long"] = services["ce"].models["long"]
        services[kind] = Serving(models,configs,"mps",2)
    folder = ROOT/cfg["reference_replay"]
    stream = dict(np.load(folder/"replay.npz"));reference = json.loads((folder/"results.json").read_text())
    paths = [*inputs.values(),folder/"replay.npz",folder/"results.json",Path(__file__).resolve(),
             ROOT/"scripts/profile_sasrec_hybrid.py",ROOT/"scripts/sasrec.py"]
    hashes = {str(p.relative_to(ROOT)):checksum(p) for p in paths}
    pool = np.flatnonzero(stream["split_idx"]==1)
    calibration = {kind:{} for kind in services}
    for batch,steps,repeats,warmup in zip(cfg["batch_sizes"],cfg["steps"],cfg["repeats"],cfg["warmup"]):
        draws = np.random.default_rng(cfg["seed"]+batch).choice(pool,(steps*repeats,batch),replace=True)
        chosen,inverse = np.unique(draws,return_inverse=True)
        vectors = {kind:{role:encode_queries(m,s.inputs(role,stream["queries"][chosen]),"mps",256)[0]
                         for role,m in s.models.items()} for kind,s in services.items()}
        shared = {"long_direct","long_refresh","long_hit","encode_long","score_long"}
        tags = [(kind,name) for kind in services for name in prior[kind]["calibration"][str(batch)]
                if kind=="ce" or name not in shared]
        cache = vectors["ce"]["long"][torch.as_tensor(inverse.reshape(-1),device="mps")].cpu().numpy().reshape(-1,batch,64)

        def operation(tag,i):
            kind,name = tag;s = services[kind];q = stream["queries"][draws[i]]
            if name.startswith("encode_"):
                return s.encode(name.split("_")[1],q)
            if name.startswith("score_"):
                role = name.split("_")[1]
                v = vectors[kind][role][torch.as_tensor(inverse.reshape(draws.shape)[i],device="mps")]
                return (v@s.models[role].item.weight[2:].T).topk(100,dim=1).indices.cpu()
            if name.startswith("long_"):
                alpha,refresh,direct = 0,name!="long_hit",name=="long_direct"
            elif name=="recent_direct":
                alpha,refresh,direct = 1,True,True
            elif name=="hybrid_direct":
                alpha,refresh,direct = selections[kind]["policies"][0]["alpha"],True,True
            else:
                alpha,refresh,direct = float(name.split("_")[-1]),"refresh" in name,False
            return s.serve(q,cache[i],refresh,alpha,direct)

        values = {tag:[] for tag in tags}
        for tag in tags:
            for i in range(warmup):operation(tag,i%len(draws));sync("mps")
        rng = np.random.default_rng(cfg["seed"])
        for rep in range(repeats):
            for index in rng.permutation(len(tags)):
                tag = tags[index];sync("mps");started = time.perf_counter()
                for i in range(rep*steps,(rep+1)*steps):operation(tag,i);sync("mps")
                values[tag].append((time.perf_counter()-started)*1000/steps/batch)
            print(f"interleaved MPS B={batch}, repeat {rep+1}/{repeats}",flush=True)
        for kind in services:
            calibration[kind][str(batch)] = {}
            for name in prior[kind]["calibration"][str(batch)]:
                v = values[("ce" if name in shared else kind,name)]
                calibration[kind][str(batch)][name] = {"repeat_mean_ms_per_request":v,"median_ms_per_request":float(np.median(v))}
        del vectors,cache
    output = {}
    for kind,p in prior.items():
        results = {split:{"points":cost_rows(hybrids[kind],reference,split,calibration[kind]["1"],p["decisions_ms"][split])}
                   for split in hybrids[kind]["config"]["splits"]}
        for block in results.values():
            block["frontiers"] = {m:frontier(block["points"],m) for m in ("ndcg@10","recall@100")}
        picked = budgets(results["controller_train"]["points"],cfg["cost_budgets_relative_to_fresh_long"])
        output[kind] = {**p,"calibration":calibration[kind],"results":results,"ct_budget_selection":picked,"mixed_checks":{},
            "source_sha256":{**p["source_sha256"],**hashes},
            "interleaved_calibration":{"shared_long_controls":True,"source_profiles":{k:str(v.relative_to(ROOT)) for k,v in inputs.items()},
                "memory_scope":"RSS and stage probes below remain from the original separate two-model process; joint process is reported separately"}}
    # Validate complete mixed paths in the same interleaved process, not against old timings.
    names = {kind:list(dict.fromkeys(["long:always","hybrid:always","hybrid:never","hybrid:time_86400"]+
             [c["point"] for c in d["ct_budget_selection"] if c["point"] and not c["point"].startswith("recent:")]))
             for kind,d in output.items()}
    policy_index = {p["name"]:i for i,p in enumerate(reference["policies"])}
    for idx,split in enumerate(["controller_train","policy_valid"],1):
        n = cfg["mixed_steps"]*cfg["mixed_repeats"]
        take = np.random.default_rng(cfg["seed"]+idx+100).choice(np.flatnonzero(stream["split_idx"]==idx),n,replace=True)
        tasks = [(k,name) for k,v in names.items() for name in v];caches = {};alphas = {};flags = {}
        for kind,name in tasks:
            family,policy = name.split(":");j = policy_index[policy]
            v,_ = encode_queries(services["ce"].models["long"],stream["queries"][stream["refs"][j,take]],"mps",256)
            caches[kind,name] = v.cpu().numpy();flags[kind,name] = stream["refresh"][j,take]
            alphas[kind,name] = 0 if family=="long" else selections[kind]["policies"][j]["alpha"]
        def serve(task,i):
            kind,name = task
            return services[kind].serve(stream["queries"][take[i:i+1]],caches[task][i:i+1],bool(flags[task][i]),alphas[task],name.endswith(":always"))
        timings = {task:[] for task in tasks}
        for task in tasks:
            for i in range(16):serve(task,i);sync("mps")
        rng = np.random.default_rng(cfg["seed"])
        for rep in range(cfg["mixed_repeats"]):
            for index in rng.permutation(len(tasks)):
                task = tasks[index];sync("mps");started = time.perf_counter()
                for i in range(rep*cfg["mixed_steps"],(rep+1)*cfg["mixed_steps"]):serve(task,i);sync("mps")
                timings[task].append((time.perf_counter()-started)*1000/cfg["mixed_steps"])
        for kind in output:
            output[kind]["mixed_checks"][split] = {}
            for name in names[kind]:
                task = (kind,name);family,policy = name.split(":");v = timings[task]
                expected = estimated_cost(family,policy,alphas[task],float(flags[task].mean()),0,calibration[kind]["1"])["estimated_ms_per_request"]
                output[kind]["mixed_checks"][split][name] = {"alpha":alphas[task],"requests":n,"sample_refreshes":int(flags[task].sum()),
                    "repeat_mean_ms_per_request":v,"median_ms_per_request":float(np.median(v)),"predicted_sample_ms":expected,
                    "measured_minus_predicted_percent":100*(float(np.median(v))/expected-1)}
    if any(checksum(ROOT/p)!=h for p,h in hashes.items()):raise ValueError("Source changed")
    for kind,d in output.items():
        for src in d["sources"].values():
            if checksum(ROOT/src["run"]/"train-mps/best.pt")!=src["checkpoint_sha256"]:raise ValueError("Checkpoint changed")
        d["interleaved_calibration"]["joint_process_memory"] = memory("mps")
        write(args.output/f"{kind}.json",d)


if __name__=="__main__":
    main()
