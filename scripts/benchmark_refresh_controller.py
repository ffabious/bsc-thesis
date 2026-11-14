"""Interleaved B=1 measurements of plain serving and complete controller paths."""

import time

import numpy as np
import torch

from sasrec import sync
from refresh_controller import make_predictor


@torch.inference_mode()
def measure_cases(model,vectors,stream,features,controller,cases,config,device):
    n=config["benchmark_steps"]*config["benchmark_repeats"];rng=np.random.default_rng(config["seed"])
    table=model.item.weight[2:].T;predict=make_predictor(controller)
    operations=[];metadata={};trials={c["name"]:[] for c in cases}
    def prepare(case):
        take=rng.choice(case["pool"],n,replace=True)
        cache=vectors[torch.as_tensor(case["refs"][take],device=device)].cpu().numpy().copy()
        def operation(k):
            i=take[k];update=bool(case["refresh"][i])
            if "threshold" in case:
                previous=int(case["before"][i])
                actual=previous<0 or predict(features.row(i,previous))>case["threshold"]
                assert actual==update
            if update:
                query=model(torch.as_tensor(stream["queries"][i:i+1],dtype=torch.long,device=device));cache[k:k+1]=query.cpu().numpy()
            else:query=torch.as_tensor(cache[k:k+1],device=device)
            return (query@table).topk(100,dim=1).indices.cpu()
        metadata[case["name"]]={"sample_requests":n,"sample_refreshes":int(case["refresh"][take].sum()),
            "unique_sample_requests":int(len(np.unique(take))),"controller_features_included":"threshold" in case}
        return operation
    for case in cases:operations.append(prepare(case))
    for operation in operations:
        for k in range(config["benchmark_warmup"]):operation(k%n);sync(device)
    order=[]
    for repeat in range(config["benchmark_repeats"]):
        chosen=rng.permutation(len(cases));order.append([cases[j]["name"] for j in chosen])
        for j in chosen:
            started=time.perf_counter()
            for k in range(repeat*config["benchmark_steps"],(repeat+1)*config["benchmark_steps"]):operations[j](k);sync(device)
            trials[cases[j]["name"]].append(1000*(time.perf_counter()-started)/config["benchmark_steps"])
    return {"cases":{name:{**metadata[name],"repeat_mean_ms":values,"median_mean_ms":float(np.median(values))}
        for name,values in trials.items()},"interleaved_order":order}


def coupled_branches(model,vectors,stream,features,controller,names,thresholds,refs,refresh,before,pool,config,device):
    q=len(stream["queries"]);fresh=np.arange(q);cases=[]
    for branch,update in [("refresh_write_serve",True),("cache_hit_serve",False)]:
        cases.append({"name":f"plain/{branch}","pool":pool,"refs":fresh,"refresh":np.full(q,update)})
        for j,name in enumerate(names):
            subset=pool[refresh[j,pool]==update]
            if len(subset):
                cases.append({"name":f"{name}/{branch}","pool":subset,"refs":refs[j],"refresh":refresh[j],
                    "before":before[j],"threshold":thresholds[j]})
    measured=measure_cases(model,vectors,stream,features,controller,cases,config,device)
    branches={}
    for name in ["plain",*names]:
        branches[name]={branch:measured["cases"].get(f"{name}/{branch}",measured["cases"][f"plain/{branch}"])
            for branch in ["refresh_write_serve","cache_hit_serve"]}
    return {"plain":branches.pop("plain"),"controllers":branches,"interleaved_order":measured["interleaved_order"],
        "scope":"Controller branches include scalar CPU features/prediction and MPS serving; no separate CPU decision charge"}
