#!/usr/bin/env python3
"""Measure complete local serving paths, then cost the frozen CT/PV schedules."""

import argparse
import hashlib
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch

from blend_sasrec import checksum,load_pair,normalize_scores,score_ranks
from replay_hybrid_sasrec import naive_ranks
from replay_sasrec import encode_queries
from replay_sasrec_policies import frontier,schedule
from run_sasrec import ROOT,memory,write
from sasrec import mask_context,metrics,sync


class Serving:
    def __init__(self,models,configs,device,recent_input_length=None):
        self.models,self.configs,self.device = models,configs,device
        if recent_input_length is not None and (type(recent_input_length) is not int or
                recent_input_length!=configs["recent"].get("context_keep")):
            raise ValueError("Short recent input must match the checkpoint context_keep")
        self.recent_input_length = recent_input_length

    def inputs(self,role,queries):
        if role=="recent" and self.recent_input_length is not None:
            queries = queries[:, -self.recent_input_length:]
        return mask_context(queries.copy(),self.configs[role].get("context_keep"))

    def mark(self,trace,name):
        if trace is not None:
            sync(self.device)
            trace.append({"stage":name,**memory(self.device)})

    @torch.inference_mode()
    def encode(self,role,queries):
        x = self.inputs(role,queries)
        return self.models[role](torch.as_tensor(x,dtype=torch.long,device=self.device))

    @torch.inference_mode()
    def scores(self,queries,cache,refresh,alpha,direct=False,trace=None):
        scores = {}
        if alpha<1:
            if refresh:
                query = self.encode("long",queries)
                if not direct:
                    cache[:] = query.cpu().numpy()
            else:
                query = torch.as_tensor(cache,device=self.device)
            self.mark(trace,"long_encode_or_lookup")
            scores["long"] = query @ self.models["long"].item.weight[2:].T
            self.mark(trace,"long_scores")
        if alpha>0:
            query = self.encode("recent",queries)
            self.mark(trace,"recent_encode")
            scores["recent"] = query @ self.models["recent"].item.weight[2:].T
            self.mark(trace,"recent_scores")
        if alpha==0:
            return scores["long"]
        if alpha==1:
            return scores["recent"]
        for s in scores.values():
            normalize_scores(s)
        mixed = (1-alpha)*scores["long"] + alpha*scores["recent"]
        self.mark(trace,"normalized_and_mixed")
        return mixed

    @torch.inference_mode()
    def serve(self,queries,cache,refresh,alpha,direct=False,trace=None):
        scores = self.scores(queries,cache,refresh,alpha,direct,trace)
        result = scores.topk(min(100,scores.shape[1]),dim=1).indices.cpu()
        self.mark(trace,"top100_cpu_result")
        return result


def calibrate(service,stream,draws,vectors,lookup,steps,repeats,warmup,alphas,fresh_alpha,seed):
    batch_size = draws.shape[1]
    full = [("long_direct",0,True,True),("long_refresh",0,True,False),("long_hit",0,False,False),
            ("recent_direct",1,True,True),("hybrid_direct",fresh_alpha,True,True)]
    use = alphas if batch_size==1 else [fresh_alpha] if 0<fresh_alpha<1 else []
    full += [(f"hybrid_{'refresh' if refresh else 'hit'}_{a:.1f}",a,refresh,False) for a in use for refresh in (False,True)]
    cases = {name:(alpha,refresh,direct) for name,alpha,refresh,direct in full}
    names = [*cases,"encode_long","encode_recent","score_long","score_recent"]
    flat = draws.ravel()
    caches = {name:vectors["long"][torch.as_tensor(lookup[flat],device=service.device)].cpu().numpy().reshape(-1,batch_size,64).copy()
              for name in cases}

    def operation(name,i):
        take = draws[i]
        if name in cases:
            alpha,refresh,direct = cases[name]
            return service.serve(stream["queries"][take],caches[name][i],refresh,alpha,direct)
        kind,role = name.split("_")
        if kind=="encode":
            return service.encode(role,stream["queries"][take])
        query = vectors[role][torch.as_tensor(lookup[take],device=service.device)]
        scores = query @ service.models[role].item.weight[2:].T
        return scores.topk(min(100,scores.shape[1]),dim=1).indices.cpu()

    for name in names:
        for i in range(warmup):
            operation(name,i%len(draws));sync(service.device)
    results = {name:[] for name in names}
    rng = np.random.default_rng(seed)
    for repeat in range(repeats):
        # Each repeat visits every path in shuffled order on the same request block.
        for name in rng.permutation(names):
            sync(service.device)
            started = time.perf_counter()
            for i in range(repeat*steps,(repeat+1)*steps):
                operation(name,i);sync(service.device)
            results[name].append((time.perf_counter()-started)*1000/steps/batch_size)
        print(f"serving {service.device}, batch {batch_size}, repeat {repeat+1}/{repeats}",flush=True)
    return {name:{"repeat_mean_ms_per_request":v,"median_ms_per_request":float(np.median(v))} for name,v in results.items()}


def estimated_cost(family,policy,alpha,refresh_fraction,decisions,paths):
    def series(name):
        return np.array(paths[name]["repeat_mean_ms_per_request"])
    if family=="recent" or alpha==1:
        values = series("recent_direct")
    elif policy=="always":
        values = series("long_direct" if alpha==0 else "hybrid_direct")
    else:
        prefix = "long" if alpha==0 else "hybrid"
        suffix = "" if alpha==0 else f"_{alpha:.1f}"
        values = (refresh_fraction*series(f"{prefix}_refresh{suffix}") +
                  (1-refresh_fraction)*series(f"{prefix}_hit{suffix}") + decisions)
    return {"estimated_ms_per_request":float(np.median(values)),"repeat_estimates_ms_per_request":values.tolist(),
            "min_repeat_ms_per_request":float(values.min()),"max_repeat_ms_per_request":float(values.max())}


def cost_rows(hybrid,reference,split,paths,decisions):
    block = hybrid["results"][split]
    rows = []
    for p,ref in zip(block["policies"],reference["results"][split]["policies"]):
        if p["name"]!=ref["name"]:
            raise ValueError("Policy order differs")
        for family,alpha,key in (("long",0,"long_metrics"),("hybrid",p["alpha"],"hybrid_metrics")):
            rate = ref["refresh_fraction"]
            active = alpha<1
            rows.append({"name":f"{family}:{p['name']}","family":family,"policy":p["name"],"alpha_recent":alpha,
                         "metrics":p[key],"requests":ref["requests"],"long_refreshes":ref["refreshes"] if active else 0,
                         "long_initializations":ref["initializations"] if active and p["name"]!="always" else 0,
                         "long_refresh_fraction":rate if active else 0,
                         "long_cache_bytes":256*64*4 if active and p["name"]!="always" else 0,
                         "cost":estimated_cost(family,p["name"],alpha,rate,decisions[p["name"]],paths)})
    rows.append({"name":"recent:direct","family":"recent","policy":"direct","alpha_recent":1,
                 "metrics":block["recent_metrics"],"requests":reference["results"][split]["requests"],
                 "long_refreshes":0,"long_initializations":0,"long_refresh_fraction":0,
                 "long_cache_bytes":0,
                 "cost":estimated_cost("recent","direct",1,0,0,paths)})
    return rows


def budgets(rows,limits):
    baseline = next(r for r in rows if r["name"]=="long:always")["cost"]["estimated_ms_per_request"]
    choices = []
    for limit in limits:
        eligible = [r for r in rows if r["cost"]["estimated_ms_per_request"]<=baseline*limit]
        best = max(eligible,key=lambda r:(r["metrics"]["ndcg@10"],r["metrics"]["recall@100"],
                   -r["cost"]["estimated_ms_per_request"],r["family"]!="hybrid")) if eligible else None
        choices.append({"relative_to_direct_fresh_long":limit,"point":best["name"] if best else None})
    return choices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--device",choices=["cpu","mps"],required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output")
    if args.device=="mps" and not torch.backends.mps.is_available():
        parser.error("MPS unavailable")
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK","0")!="0":
        parser.error("Disable CPU fallback")
    config = json.loads(args.config.read_text());torch.set_num_threads(config["threads"])
    run,folder = ROOT/config["hybrid_run"],ROOT/config["reference_replay"]
    hybrid,selection = json.loads((run/"results.json").read_text()),json.loads((run/"selection.json").read_text())
    reference = json.loads((folder/"results.json").read_text());stream = dict(np.load(folder/"replay.npz"))
    if checksum(run/"selection.json")!=hybrid["selection_sha256"]:
        raise ValueError("Hybrid selection changed")
    models,configs,sources = load_pair(hybrid["config"],"256",args.device)
    if sources!=hybrid["sources"]:
        raise ValueError("Models differ from the evaluated hybrid")
    for p,h in hybrid["reference_sha256"].items():
        if checksum(ROOT/p)!=h:
            raise ValueError("Reference changed")
    sources_used = {str(p.relative_to(ROOT)):checksum(p) for p in (run/"results.json",run/"selection.json",folder/"replay.npz",
        folder/"results.json",ROOT/"scripts/profile_sasrec_hybrid.py",ROOT/"scripts/replay_hybrid_sasrec.py",
        ROOT/"scripts/blend_sasrec.py",ROOT/"scripts/sasrec.py",ROOT/"scripts/replay_sasrec_policies.py",args.config.resolve())}
    service = Serving(models,configs,args.device,config.get("recent_input_length"))
    def state_digest(model):
        value = hashlib.sha256()
        for name,tensor in model.state_dict().items():
            value.update(name.encode());value.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        return value.hexdigest()
    states_before = {role:state_digest(m) for role,m in models.items()}
    pool = np.flatnonzero(stream["split_idx"]==1)
    draw_sets = [np.random.default_rng(config["seed"]+batch).choice(pool,size=(steps*repeat,batch),replace=True)
                 for batch,steps,repeat in zip(config["batch_sizes"],config["steps"],config["repeats"])]
    chosen = np.unique(np.concatenate([d.ravel() for d in draw_sets]))
    lookup = np.full(len(stream["users"]),-1,dtype=np.int64);lookup[chosen] = np.arange(len(chosen))
    vectors = {role:encode_queries(m,service.inputs(role,stream["queries"][chosen]),args.device,256)[0]
               for role,m in models.items()}
    alphas = sorted({p["alpha"] for p in selection["policies"] if 0<p["alpha"]<1})
    fresh_alpha = selection["policies"][0]["alpha"]
    calibration = {}
    for batch,steps,repeat,warmup,draws in zip(config["batch_sizes"],config["steps"],config["repeats"],config["warmup"],draw_sets):
        calibration[str(batch)] = calibrate(service,stream,draws,vectors,lookup,steps,repeat,warmup,alphas,fresh_alpha,config["seed"])
    paths = calibration["1"]
    decisions = {}
    for split_idx,split in enumerate(hybrid["config"]["splits"],1):
        take = stream["split_idx"]==split_idx
        part = {k:stream[k][take] for k in ("users","timestamp","history","split_idx")}
        values = {}
        for p in reference["policies"]:
            trials = []
            for _ in range(3):
                started = time.perf_counter();schedule(part,p,256)
                trials.append((time.perf_counter()-started)*1000/take.sum())
            values[p["name"]] = float(np.median(trials))
        decisions[split] = values
    results = {s:{"points":cost_rows(hybrid,reference,s,paths,decisions[s])} for s in hybrid["config"]["splits"]}
    for b in results.values():
        b["frontiers"] = {m:frontier(b["points"],m) for m in ("ndcg@10","recall@100")}
    picked = budgets(results["controller_train"]["points"],config["cost_budgets_relative_to_fresh_long"])
    checks = list(dict.fromkeys(["long:always","long:never"]+[f"hybrid:{p}" for p in config["mixed_policies"]]+
                               [r["point"] for r in picked if r["point"] and not r["point"].startswith("recent:")]))
    mixed = {}
    for idx,split in enumerate(hybrid["config"]["splits"],1):
        pool = np.flatnonzero(stream["split_idx"]==idx)
        n = config["mixed_steps"]*config["mixed_repeats"]
        take = np.random.default_rng(config["seed"]+idx+100).choice(pool,n,replace=True)
        by_name = {p["name"]:i for i,p in enumerate(reference["policies"])}
        selected = [by_name[name.split(":",1)[1]] for name in checks]
        refs = stream["refs"][selected][:,take]
        needed = np.unique(refs)
        states,_ = encode_queries(models["long"],stream["queries"][needed],args.device,256)
        state_lookup = {int(v):i for i,v in enumerate(needed)}
        mixed[split] = {}
        for label,j,old in zip(checks,selected,refs):
            family,name = label.split(":",1)
            alpha = 0 if family=="long" else selection["policies"][j]["alpha"]
            cache = states[torch.as_tensor([state_lookup[int(v)] for v in old],device=args.device)].cpu().numpy().copy()
            updates = stream["refresh"][j,take]
            for i in range(min(16,n)):
                service.serve(stream["queries"][take[i:i+1]],cache[i:i+1],bool(updates[i]),alpha,name=="always");sync(args.device)
            trials = []
            for rep in range(config["mixed_repeats"]):
                started = time.perf_counter()
                for i in range(rep*config["mixed_steps"],(rep+1)*config["mixed_steps"]):
                    service.serve(stream["queries"][take[i:i+1]],cache[i:i+1],bool(updates[i]),alpha,name=="always");sync(args.device)
                trials.append((time.perf_counter()-started)*1000/config["mixed_steps"])
            expected = estimated_cost(family,name,alpha,float(updates.mean()),0,paths)["estimated_ms_per_request"]
            mixed[split][label] = {"alpha":alpha,"requests":n,"sample_refreshes":int(updates.sum()) if alpha<1 else 0,
                "repeat_mean_ms_per_request":trials,"median_ms_per_request":float(np.median(trials)),"predicted_sample_ms":expected,
                "measured_minus_predicted_percent":100*(float(np.median(trials))/expected-1)}
    probes = {}
    for batch in config["batch_sizes"]:
        q = stream["queries"][draw_sets[config["batch_sizes"].index(batch)][0]]
        cache = np.zeros((batch,64),dtype=np.float32)
        traces = []
        service.serve(q,cache,True,fresh_alpha,False,traces)
        probes[str(batch)] = traces
    # Independent re-encoding of stale/fresh windows verifies the timed serving path.
    rng = np.random.default_rng(config["seed"])
    target_rows = rng.choice(len(stream["targets"]),32,replace=False)
    q = stream["target_request"][target_rows];p = rng.integers(len(reference["policies"]),size=len(q))
    old = stream["refs"][p,q];alpha = np.array([selection["policies"][j]["alpha"] for j in p])
    expected = naive_ranks(models,configs,stream["queries"],old,q,stream["targets"][target_rows],alpha,args.device,1e-8)
    values,_ = encode_queries(models["long"],stream["queries"][old],args.device,32)
    actual = []
    for i in range(len(q)):
        s = service.scores(stream["queries"][q[i:i+1]],values[i:i+1].cpu().numpy().copy(),False,float(alpha[i]))
        ids = torch.arange(2,models["long"].item.num_embeddings,device=args.device)
        y = torch.as_tensor(stream["targets"][target_rows[i:i+1]],dtype=torch.long,device=args.device)
        actual.append(int(score_ranks(s,ids,y).cpu()[0]))
    actual = np.array(actual)
    if metrics(actual)!=metrics(expected) or np.max(np.abs(actual-expected))>1:
        raise ValueError("Timed serving differs from the quality oracle")
    for p,h in sources_used.items():
        if checksum(ROOT/p)!=h:
            raise ValueError(f"Source changed: {p}")
    if {role:state_digest(m) for role,m in models.items()}!=states_before:
        raise ValueError("Frozen serving models changed")
    for s in sources.values():
        if checksum(ROOT/s["run"]/"train-mps/best.pt")!=s["checkpoint_sha256"]:
            raise ValueError("Serving checkpoint changed")
    data = ROOT/configs["long"]["data"]
    if any(checksum(data/p)!=h for p,h in hybrid["data_files_sha256"].items()):
        raise ValueError("Serving data changed")
    write(args.output,{"config":config,"device":args.device,"sources":sources,"calibration":calibration,"decisions_ms":decisions,
        "results":results,"ct_budget_selection":picked,"mixed_checks":mixed,"memory_probes":probes,
        "serving_oracle":{"pairs":len(q),"same_metrics":True,"exact_rank_matches":int((actual==expected).sum()),"maximum_rank_difference":int(np.abs(actual-expected).max())},
        "selected_input_windows":len(chosen),"parameters":{k:sum(p.numel() for p in m.parameters()) for k,m in models.items()},
        "logical_cache_bytes":256*64*4,"source_sha256":sources_used,"torch":torch.__version__,"numpy":np.__version__,
        "frozen_state_sha256":states_before,"no_test_events_read":True,
        "input_widths":{role:service.inputs(role,stream["queries"][:1]).shape[1] for role in models},
        "python":platform.python_version(),"macos":platform.mac_ver()[0],**memory(args.device)})


if __name__=="__main__":
    main()
