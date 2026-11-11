#!/usr/bin/env python3
"""Select a teacher on model_valid, warm-start CE/KD students, compare frozen models."""

import argparse
import hashlib
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from blend_sasrec import checksum,load_pair,normalize_scores,score_ranks,select
from compare_sasrec import paired_intervals
from profile_sasrec_hybrid import Serving
from replay_sasrec import quality
from run_sasrec import ROOT,memory,model_args,train,write
from sasrec import SASRec,ranks,sync


class Teacher:
    def __init__(self,models,alpha,temperature=2):
        if not 0<=alpha<=1 or temperature<=0:
            raise ValueError("Invalid teacher alpha/temperature")
        self.models,self.alpha,self.temperature = models,alpha,temperature
        for model in models.values():model.eval().requires_grad_(False)

    @torch.no_grad()
    def logits(self,sequence):
        long = self.models["long"](sequence)@self.models["long"].item.weight[2:].T
        if self.alpha==0:
            return long
        scale = long.std(dim=1,unbiased=False,keepdim=True).clamp_min(1e-8)
        recent = self.models["recent"](sequence[:,-2:])@self.models["recent"].item.weight[2:].T
        normalize_scores(long);normalize_scores(recent)
        # Affine rescaling keeps the selected ensemble ranks and matches long-logit units.
        return ((1-self.alpha)*long+self.alpha*recent)*scale

    def targets(self,sequence):
        return F.log_softmax(self.logits(sequence)/self.temperature,dim=1)


def objectives(logits,target,teacher_log_prob,weight,temperature):
    ce = F.cross_entropy(logits,target-2)
    kd = F.kl_div(F.log_softmax(logits/temperature,dim=1),teacher_log_prob,
                  log_target=True,reduction="batchmean")*temperature**2
    return ce+weight*kd,ce,kd


class DistillationStep:
    def __init__(self,teacher,weight):
        self.teacher,self.weight = teacher,weight
        self.total,self.count = None,0

    def __call__(self,model,optimizer,sequence,target,negative,reduction,objective):
        if objective!="full_ce":raise ValueError("Distillation requires full CE")
        optimizer.zero_grad(set_to_none=True)
        logits = model(sequence)@model.item.weight[2:].T
        value,ce,kd = objectives(logits,target,self.teacher.targets(sequence),self.weight,self.teacher.temperature)
        value.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.0);optimizer.step()
        values = torch.stack([ce.detach(),kd.detach()])*len(target)
        self.total = values if self.total is None else self.total+values
        self.count += len(target)
        return value.detach()

    def summary(self):
        ce,kd = (self.total/self.count).tolist()
        self.total,self.count = None,0
        return {"ce":ce,"kl_t2":kd,"weighted_kl_t2":self.weight*kd}


def read(path):
    return json.loads(path.read_text())


def label(weight):
    return "control" if weight==0 else f"kd-{weight:g}"


def validation(result):
    return result["initial_validation"] if result["best_epoch"]==0 else result["epochs"][result["best_epoch"]-1]["validation"]


def load_student(folder,device):
    trained = read(folder/"training.json");path = folder/"best.pt"
    if checksum(path)!=trained["checkpoint_sha256"]:raise ValueError("Student checkpoint changed")
    saved = torch.load(path,map_location="cpu",weights_only=True)
    model = SASRec(saved["items"],**model_args(saved["config"]))
    model.load_state_dict(saved["weights"])
    return model.to(device).eval().requires_grad_(False),saved["config"]


def source_hashes(config_path):
    paths = [config_path.resolve()]+[ROOT/"scripts"/p for p in ["distill_sasrec.py","run_sasrec.py","sasrec.py","blend_sasrec.py"]]
    return {str(p.relative_to(ROOT)):checksum(p) for p in paths}


def check_hashes(hashes):
    if any(checksum(ROOT/p)!=h for p,h in hashes.items()):raise ValueError("Source/input changed")


def run_training(protocol,root,weight,device,config_path,resume):
    started = time.perf_counter()
    teacher_selection = read(root/"teacher/selection.json")
    if teacher_selection["protocol"]!=protocol["teacher"]:raise ValueError("Teacher protocol changed")
    selection_hash = checksum(root/"teacher/selection.json")
    config = {**read(ROOT/protocol["base_config"]),"initial_checkpoint":protocol["initial_checkpoint"],
        "initial_checkpoint_sha256":protocol["initial_checkpoint_sha256"],"distillation_weight":weight,
        "distillation_temperature":protocol["temperature"],"teacher_alpha":teacher_selection["alpha"],
        "teacher_selection_sha256":selection_hash,"experiment_protocol_sha256":checksum(config_path)}
    hashes = source_hashes(config_path);data = ROOT/config["data"]
    used = {p:checksum(data/p) for p in ["manifest.json","events/train.parquet","events/model_valid.parquet","items.parquet","users.parquet"]}
    update = None
    if weight:
        models,_,sources = load_pair(protocol["teacher"],"256",device)
        if sources!=teacher_selection["sources"]:raise ValueError("Teacher sources changed")
        update = DistillationStep(Teacher(models,teacher_selection["alpha"],protocol["temperature"]),weight)
    # Teacher setup precedes seeding: control/KD use identical initial dropout and batch RNG.
    torch.manual_seed(config["seed"])
    if device=="mps":torch.mps.manual_seed(config["seed"])
    folder = root/label(weight)/"train-mps"
    setup_seconds = time.perf_counter()-started
    result = train(config,data,device,folder,resume,update)
    if checksum(root/"teacher/selection.json")!=selection_hash:raise ValueError("Teacher selection changed")
    check_hashes(hashes);check_hashes({str((data/p).relative_to(ROOT)):h for p,h in used.items()})
    for src in teacher_selection["sources"].values():
        if checksum(ROOT/src["run"]/"train-mps/best.pt")!=src["checkpoint_sha256"]:raise ValueError("Teacher checkpoint changed")
    result.update(config=config,device=device,torch=torch.__version__,numpy=np.__version__,python=platform.python_version(),
        macos=platform.mac_ver()[0],data_manifest_sha256=used["manifest.json"],source_sha256=hashes,
        used_data_files_sha256=used,teacher_sources=teacher_selection["sources"],no_holdout_events_read=True,
        setup_seconds=setup_seconds,total_run_seconds=time.perf_counter()-started)
    write(folder/"training.json",result)
    print(json.dumps({"run":label(weight),"best_epoch":result["best_epoch"],"epochs":len(result["epochs"]),
        "validation":validation(result),"elapsed_seconds":result["elapsed_seconds"]}),flush=True)


def lock_selection(protocol,root):
    path = root/"selection.json"
    if path.exists():raise FileExistsError("Model selection already locked")
    rows = []
    for weight in protocol["weights"]:
        folder = root/label(weight)/"train-mps";r = read(folder/"training.json")
        if (r["stop_reason"]!="validation_patience" or r["config"]["distillation_weight"]!=weight or
                checksum(folder/"best.pt")!=r["checkpoint_sha256"]):raise ValueError("Training incomplete or changed")
        rows.append({"weight":weight,"run":str(folder.relative_to(ROOT)),"best_epoch":r["best_epoch"],
            "metrics":validation(r),"checkpoint_sha256":r["checkpoint_sha256"]})
    winner = max([r for r in rows if r["weight"]>0],key=lambda r:(r["metrics"]["ndcg@10"],-r["weight"]))
    result = {"selected_on":"model_valid","criterion":"maximum best-epoch NDCG@10; ties smaller nonzero weight",
        "student":winner,"control":rows[0],"candidates":rows,"teacher_selection_sha256":checksum(root/"teacher/selection.json"),
        "locked_before_holdout_read":True}
    write(path,result)
    return result


@torch.inference_mode()
def evaluate(protocol,root,device):
    if (root/"evaluation.json").exists():raise FileExistsError("Evaluation already exists")
    selected = lock_selection(protocol,root);selection_hash = checksum(root/"selection.json")
    teacher_selection = read(root/"teacher/selection.json")
    models,_,sources = load_pair(protocol["teacher"],"256",device)
    if sources!=teacher_selection["sources"]:raise ValueError("Teacher changed")
    teacher = Teacher(models,teacher_selection["alpha"],protocol["temperature"])
    folder = ROOT/protocol["reference_replay"]
    stream = dict(np.load(folder/"replay.npz"));reference = read(folder/"results.json")
    if sources["long"]["checkpoint_sha256"]!=reference["checkpoint_sha256"]:raise ValueError("Replay checkpoint differs")
    q = stream["target_request"];x = stream["queries"][q];y = stream["targets"].astype(np.int64);users = stream["users"][q]
    values = {"original":stream["ranks"][0]}
    for name in ["control","student"]:
        model,config = load_student(ROOT/selected[name]["run"],device)
        values[name] = ranks(model,x,y,device,256)
        del model
    ids = torch.arange(2,models["long"].item.num_embeddings,device=device);parts = []
    for start in range(0,len(y),256):
        sequence = torch.as_tensor(x[start:start+256],dtype=torch.long,device=device)
        target = torch.as_tensor(y[start:start+256],dtype=torch.long,device=device)
        parts.append(score_ranks(teacher.logits(sequence),ids,target).cpu().numpy())
    values["teacher"] = np.concatenate(parts)
    results = {}
    for idx,split in enumerate(protocol["evaluation_splits"],1):
        take = stream["split_idx"][q]==idx
        current = {k:v[take] for k,v in values.items()}
        last = x[take,-1]==y[take];seen = (x[take]==y[take,None]).any(1)
        results[split] = {"targets":int(take.sum()),"users":int(len(np.unique(users[take]))),
            "metrics":{k:quality(v) for k,v in current.items()},
            "student_minus":{k:paired_intervals(current["student"],v,users[take],42) for k,v in current.items() if k!="student"},
            "control_minus_original":paired_intervals(current["control"],current["original"],users[take],42),
            "strata":{segment:{"targets":int(mask.sum()),"metrics":{k:quality(v[mask]) for k,v in current.items()}}
                      for segment,mask in [("last_item",last),("other_in_window",seen&~last),("outside_window",~seen)]}}
    if checksum(root/"selection.json")!=selection_hash:raise ValueError("Selection changed")
    np.savez_compressed(root/"evaluation-ranks.npz",**values,targets=y,users=users,split_idx=stream["split_idx"][q])
    write(root/"evaluation.json",{"selection_sha256":selection_hash,"selection":selected,"teacher_alpha":teacher.alpha,
        "results":results,"catalog_items":sources["long"]["items"],"sources":sources,"device":device,"no_test_events_read":True,
        "reference_sha256":{str((folder/p).relative_to(ROOT)):checksum(folder/p) for p in ["replay.npz","results.json"]}})


@torch.inference_mode()
def profile(protocol,root,device):
    destination = root/f"serving-{device}.json"
    if destination.exists():raise FileExistsError("Serving profile already exists")
    selected = read(root/"selection.json");selection_hash = checksum(root/"selection.json")
    teacher_selection = read(root/"teacher/selection.json")
    models,configs,sources = load_pair(protocol["teacher"],"256",device)
    alpha = teacher_selection["alpha"]
    services = {"original":Serving(models,configs,device,2),"teacher":Serving(models,configs,device,2)}
    for name in ["control","student"]:
        model,config = load_student(ROOT/selected[name]["run"],device)
        services[name] = Serving({"long":model,"recent":models["recent"]},{"long":config,"recent":configs["recent"]},device,2)
    stream = dict(np.load(ROOT/protocol["reference_replay"]/"replay.npz"));pool = np.flatnonzero(stream["split_idx"]==1)
    measurements,probes = {},{}
    for batch,steps,repeats,warmup in [(1,128,5,16),(256,8,3,2)]:
        draws = np.random.default_rng(42+batch).choice(pool,(steps*repeats,batch),replace=True)
        cache = np.zeros((batch,64),dtype=np.float32);values = {name:[] for name in services}
        def operation(name,i):
            return services[name].serve(stream["queries"][draws[i]],cache,True,alpha if name=="teacher" else 0,True)
        for name in services:
            for i in range(warmup):operation(name,i%len(draws));sync(device)
        rng = np.random.default_rng(42)
        for rep in range(repeats):
            for name in rng.permutation(list(services)):
                sync(device);started = time.perf_counter()
                for i in range(rep*steps,(rep+1)*steps):operation(name,i);sync(device)
                values[name].append((time.perf_counter()-started)*1000/steps/batch)
            print(f"serving {device} B={batch}, repeat {rep+1}/{repeats}",flush=True)
        measurements[str(batch)] = {k:{"repeat_mean_ms_per_request":v,"median_ms_per_request":float(np.median(v))} for k,v in values.items()}
        probes[str(batch)] = {}
        for name,service in services.items():
            trace = [];before = memory(device)
            service.serve(stream["queries"][draws[0]],cache,True,alpha if name=="teacher" else 0,True,trace)
            probes[str(batch)][name] = {"before":before,"stages":trace}
    if checksum(root/"selection.json")!=selection_hash:raise ValueError("Selection changed")
    write(destination,{"measurements":measurements,"memory_probes":probes,"selection_sha256":selection_hash,
        "teacher_alpha":alpha,"device":device,"teacher_sources":sources,"parameters_single":sum(p.numel() for p in models["long"].parameters()),
        "timing_scope":"complete fresh serving: token preparation/transfers, encode, catalog scoring, optional zscore/mix, final top100 CPU result",
        "memory_scope":"common benchmark process holds original/control/student/recent models; stage snapshots are not deployment peaks",
        "no_test_events_read":True,**memory(device)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=["teacher","train","evaluate","profile"])
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--device",choices=["cpu","mps"],required=True)
    parser.add_argument("--weight",type=float,default=0)
    parser.add_argument("--resume",action="store_true")
    args = parser.parse_args();protocol = read(args.config)
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK","0")!="0":parser.error("Disable CPU fallback")
    if args.device=="mps" and not torch.backends.mps.is_available():parser.error("MPS unavailable")
    if args.weight not in protocol["weights"]:parser.error("Weight not in the fixed grid")
    torch.set_num_threads(protocol["teacher"]["threads"])
    if args.mode=="teacher":select(protocol["teacher"],"256",args.device,args.output/"teacher")
    elif args.mode=="train":run_training(protocol,args.output,args.weight,args.device,args.config,args.resume)
    elif args.mode=="evaluate":evaluate(protocol,args.output,args.device)
    else:profile(protocol,args.output,args.device)


if __name__=="__main__":main()
