#!/usr/bin/env python3
"""Restore the original CE initializer and verify a fresh KD student's state/RNG."""

import argparse
import ast
import hashlib
import json
import platform
import shutil
import subprocess
import types
from pathlib import Path

import numpy as np
import torch

from blend_sasrec import checksum
from run_sasrec import ROOT,model_args,state_digest,write
from sasrec import SASRec,examples,metrics,ranks


def digest_rng(value):
    return hashlib.sha256(value.cpu().numpy().tobytes()).hexdigest()


def definition(tree,name,member=None):
    node=next(n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name==name)
    return next(n for n in node.body if isinstance(n,ast.FunctionDef) and n.name==member) if member else node


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--device",choices=["cpu","mps"],required=True)
    args=parser.parse_args();root=args.output.resolve()
    if (root/"initialization.json").exists():parser.error("Initialization already verified")
    protocol=json.loads(args.config.read_text());base=json.loads((ROOT/protocol["base_config"]).read_text())
    control=ROOT/protocol["control_run"];trained=json.loads((control/"training.json").read_text())
    if (base!=trained["config"] or checksum(control/"best.pt")!=protocol["control_checkpoint_sha256"] or
            trained["torch"]!=torch.__version__ or trained["numpy"]!=np.__version__ or trained["device"]!=args.device):
        raise ValueError("Ready control configuration/runtime/checkpoint differs")
    registry=json.loads((ROOT/protocol["reference_source_registry"]).read_text())
    archive=root/"initialization-reference";archive.mkdir(parents=True,exist_ok=True)
    sources={}
    for name in ["sasrec.py","run_sasrec.py"]:
        content=subprocess.run(["git","show",f"{protocol['reference_source_commit']}:scripts/{name}"],cwd=ROOT,check=True,capture_output=True).stdout
        path=archive/name;path.write_bytes(content)
        if checksum(path)!=registry[f"scripts/{name}"]:raise ValueError("Archived source differs from the original run manifest")
        sources[f"scripts/{name}"]=checksum(path)
    old=ast.parse((archive/"sasrec.py").read_text());current=ast.parse((ROOT/"scripts/sasrec.py").read_text())
    for name in ["SASRec","Block"]:
        if ast.dump(definition(old,name,"__init__"))!=ast.dump(definition(current,name,"__init__")):raise ValueError("Model constructor changed")
    for name in ["examples","negatives","batch","loss"]:
        if ast.dump(definition(old,name))!=ast.dump(definition(current,name)):raise ValueError(f"Control recipe changed: {name}")
    old_loop=ast.parse((archive/"run_sasrec.py").read_text());current_loop=ast.parse((ROOT/"scripts/run_sasrec.py").read_text())
    for name in ["optimizer_for","step"]:
        if ast.dump(definition(old_loop,name))!=ast.dump(definition(current_loop,name)):raise ValueError(f"CE optimization changed: {name}")
    legacy=types.ModuleType("restored_ce_sasrec")
    exec(compile((archive/"sasrec.py").read_text(),str(archive/"sasrec.py"),"exec"),legacy.__dict__)
    torch.set_num_threads(base["threads"])
    items=json.loads((ROOT/base["data"]/"manifest.json").read_text())["candidate_items"]
    models=[];rng=[]
    for cls in [legacy.SASRec,SASRec]:
        torch.manual_seed(base["seed"])
        if args.device=="mps":torch.mps.manual_seed(base["seed"])
        models.append(cls(items,**model_args(base)))
        rng.append({"cpu":digest_rng(torch.get_rng_state()),"mps":digest_rng(torch.mps.get_rng_state()) if args.device=="mps" else None})
    left,right=[m.state_dict() for m in models]
    if set(left)!=set(right) or any(not torch.equal(v,right[k]) for k,v in left.items()) or rng[0]!=rng[1]:
        raise ValueError("Fresh student initialization differs from restored CE")
    last=torch.load(control/"last.pt",weights_only=True,map_location="cpu")
    if last["config"]!=base or last["epoch"]!=len(trained["epochs"]) or digest_rng(last["torch_rng"])!=rng[0]["cpu"]:
        raise ValueError("CPU RNG does not match the saved CE run")
    del last
    data=ROOT/base["data"];x,y,_=examples(data,"model_valid",base["length"])
    values=[ranks(m.to(args.device),x,y,args.device,base["eval_batch_size"]) for m in models]
    if any(metrics(v)!=trained["initial_validation"] for v in values):raise ValueError("Original initial validation not reproduced")
    teacher=ROOT/protocol["teacher_selection_source"]
    if checksum(teacher)!=protocol["teacher_selection_sha256"]:raise ValueError("Fixed teacher selection changed")
    selected=json.loads(teacher.read_text())
    if selected["alpha"]!=protocol["fixed_teacher_alpha"] or selected["protocol"]!=protocol["teacher"]:raise ValueError("Teacher protocol differs")
    (root/"teacher").mkdir(exist_ok=True);shutil.copyfile(teacher,root/"teacher/selection.json")
    write(root/"initialization.json",{"control_run":protocol["control_run"],"control_checkpoint_sha256":checksum(control/"best.pt"),
        "control_last_sha256":checksum(control/"last.pt"),"reference_source_commit":protocol["reference_source_commit"],
        "archived_source_sha256":sources,"same_all_tensors":True,"tensors":len(left),"parameters":sum(p.numel() for p in models[0].parameters()),
        "initial_state_sha256":state_digest(models[0]),"initial_rng_sha256":rng[0]["cpu"],"initial_mps_rng_sha256":rng[0]["mps"],
        "matches_saved_control_cpu_rng":True,"initial_validation":metrics(values[1]),"reference_initial_validation":trained["initial_validation"],
        "initial_rank_exact_matches":int((values[0]==values[1]).sum()),"initial_rank_max_delta":int(np.abs(values[0]-values[1]).max()),
        "validation_examples":len(y),"teacher_selection_sha256":checksum(teacher),"data_manifest_sha256":trained["data_manifest_sha256"],
        "torch":torch.__version__,"numpy":np.__version__,"python":platform.python_version(),"device":args.device,
        "current_source_sha256":{str(p.relative_to(ROOT)):checksum(p) for p in [Path(__file__).resolve(),ROOT/"scripts/sasrec.py",ROOT/"scripts/run_sasrec.py"]},
        "no_ct_pv_test_events_read":True,"method":"CE initial checkpoint was not saved; restored from archived source/config/seed and checked against saved CPU RNG and initial validation"})
    print(f"Verified all {len(left)} initial tensors and {len(y)} validation targets; CE control can be reused",flush=True)


if __name__=="__main__":main()
