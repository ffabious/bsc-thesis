#!/usr/bin/env python3
"""Prepare a paired seed repeat, then freeze its newly trained CE50 teacher."""

import argparse
import hashlib
import json

import numpy as np
import torch

from blend_sasrec import checksum
from distill_sasrec import load_student,validation
from replay_sasrec import quality
from run_sasrec import ROOT,model_args,state_digest,write
from sasrec import SASRec,examples,ranks


def read(path):
    return json.loads(path.read_text())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=["initial","teacher"])
    args=parser.parse_args()
    root=ROOT/"data/runs/paired-seed-repeat";ce_config=ROOT/"configs/experiments/paired-repeat-ce50.json"
    student_config=ROOT/"configs/experiments/paired-repeat-student.json"
    base=read(ROOT/"configs/experiments/sasrec-256-full-ce.json");base["seed"]=43
    if not torch.backends.mps.is_available():raise RuntimeError("Local MPS is required")
    torch.set_num_threads(base["threads"])
    if args.mode=="initial":
        root.mkdir(parents=True,exist_ok=False)
        torch.manual_seed(43);torch.mps.manual_seed(43)
        data=ROOT/base["data"];count=read(data/"manifest.json")["candidate_items"]
        model=SASRec(count,**model_args(base))
        rng=lambda x:hashlib.sha256(x.cpu().numpy().tobytes()).hexdigest()
        initial={"state":state_digest(model),"cpu_rng":rng(torch.get_rng_state()),"mps_rng":rng(torch.mps.get_rng_state())}
        torch.save({"weights":model.state_dict(),"seed":43,"items":count,"model_args":model_args(base)},root/"initial.pt")
        base.update(expected_initial_state_sha256=initial["state"],expected_initial_rng_sha256=initial["cpu_rng"],
            expected_initial_mps_rng_sha256=initial["mps_rng"])
        write(ce_config,base)
        source=[ROOT/"scripts/run_sasrec.py",ROOT/"scripts/sasrec.py",ce_config]
        files=["manifest.json","events/train.parquet","events/model_valid.parquet","items.parquet","users.parquet"]
        write(root/"setup.json",{"seed":43,"initial":initial,"initial_checkpoint_sha256":checksum(root/"initial.pt"),
            "tensors":len(model.state_dict()),"parameters":sum(p.numel() for p in model.parameters()),
            "source_sha256":{str(p.relative_to(ROOT)):checksum(p) for p in source},
            "used_data_files_sha256":{name:checksum(data/name) for name in files},
            "torch":torch.__version__,"numpy":np.__version__,"no_ct_pv_test_events_read":True})
        print("Saved random initializer and guarded CE50 config for the paired repeat",flush=True)
        return
    setup=read(root/"setup.json");folder=root/"ce50/train-mps";trained=read(folder/"training.json")
    config=read(ce_config);data=ROOT/config["data"]
    if trained["stop_reason"]!="validation_patience" or checksum(folder/"best.pt")!=trained["checkpoint_sha256"]:
        raise ValueError("CE50 training incomplete or checkpoint changed")
    for path,h in setup["source_sha256"].items():
        if checksum(ROOT/path)!=h:raise ValueError("CE training source/config changed")
    for path,h in setup["used_data_files_sha256"].items():
        if checksum(data/path)!=h:raise ValueError("CE input changed")
    for name,key in [("initial_state_sha256","state"),("initial_rng_sha256","cpu_rng"),("initial_mps_rng_sha256","mps_rng")]:
        if trained[name]!=setup["initial"][key]:raise ValueError("Actual CE initializer differs")
    trained.update(source_sha256=setup["source_sha256"],used_data_files_sha256=setup["used_data_files_sha256"],
        no_ct_pv_test_events_read=True)
    write(folder/"training.json",trained)
    student=root/"student"
    if (student/"fixed-teacher.json").exists():raise FileExistsError("Teacher is already fixed")
    protocol=read(ROOT/"configs/experiments/student-temperature-control.json")
    for key in list(protocol):
        if key.startswith("comparison_") or key in ["reference_label","student_label"]:del protocol[key]
    protocol.update(base_config=str(ce_config.relative_to(ROOT)),control_run=str(folder.relative_to(ROOT)),
        control_checkpoint_sha256=trained["checkpoint_sha256"])
    protocol["teacher"]["runs"]["256"]["long"]=str((root/"ce50").relative_to(ROOT))
    protocol["teacher"]["seed"]=43
    model,_=load_student(folder,"mps");x,y,u=examples(data,"model_valid",50)
    rank=ranks(model,x,y,"mps",config["eval_batch_size"]);actual=quality(rank)
    if any(actual[k]!=v for k,v in validation(trained).items()):raise ValueError("Teacher validation not reproduced")
    source={"run":str((root/"ce50").relative_to(ROOT)),"checkpoint_sha256":trained["checkpoint_sha256"],
        "epoch":trained["best_epoch"],"data_manifest_sha256":trained["data_manifest_sha256"],"items":model.item.num_embeddings-2}
    fixed={"pilot":"256","alpha":0.0,"selected_on":"fixed_CE_control_before_student_training", "examples":len(y),
        "users":len(np.unique(u)),"catalog_items":source["items"],"curve":[{"alpha":0.0,"metrics":actual}],
        "sources":{"long":source},"protocol":protocol["teacher"],"endpoint_validation_reproduced":True,
        "no_ct_pv_test_events_read":True}
    write(student/"fixed-teacher.json",fixed)
    protocol["teacher_selection_source"]=str((student/"fixed-teacher.json").relative_to(ROOT))
    protocol["teacher_selection_sha256"]=checksum(student/"fixed-teacher.json")
    write(student_config,protocol)
    (student/"teacher").mkdir(exist_ok=True)
    np.savez_compressed(student/"teacher/model-valid.npz",ranks=rank,targets=y,users=u)
    print(json.dumps({"teacher_best_epoch":trained["best_epoch"],"validation":actual,"student_config":str(student_config)}),flush=True)


if __name__=="__main__":main()
