#!/usr/bin/env python3
"""Probe the frozen encoder's repeat losses using only its last two inputs."""

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"scripts"))
from blend_sasrec import checksum
from compare_sasrec import paired_intervals
from distill_sasrec import load_student
from replay_sasrec import quality
from run_sasrec import write
from sasrec import ranks


@torch.inference_mode()
def main():
    assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK","0")=="0"
    torch.set_num_threads(4)
    root=ROOT/"reports/experiments/sasrec-256-distillation-audit"
    output=root/"context-check.json"
    if output.exists():raise FileExistsError(output)
    local=ROOT/"data/diagnostics/sasrec-256-distillation-audit"
    source=local/"events.npz";v=dict(np.load(source));take=v["stratum"]==0
    v={key:a[take] for key,a in v.items()};assert (v["queries"][:,-1]==v["targets"]).all()
    selection=json.loads((ROOT/"data/runs/sasrec-256-distillation-cold/selection.json").read_text())
    for name in ["control","student"]:
        model,_=load_student(ROOT/selection[name]["run"],"mps")
        v[name+"_short"]=ranks(model,v["queries"][:,-2:],v["targets"],"mps",256)
        del model
    result={"no_training":True,"no_test_events_read":True,"source_events_sha256":checksum(source),
        "source_sha256":checksum(Path(__file__).resolve()),"interpretation":"Input ablation of frozen history50 models; not a retrained or deployable repeat detector", "splits":{}}
    for idx,split in enumerate(["model_valid","controller_train","policy_valid"]):
        m=v["split_idx"]==idx
        result["splits"][split]={"targets":int(m.sum()),"models":{name:quality(v[name][m])
            for name in ["control","student","control_short","student_short"]},
            "student_minus_control_short":paired_intervals(v["student_short"][m],v["control_short"][m],v["users"][m],42)}
    np.savez_compressed(local/"context-check.npz",**v)
    write(output,result);print(json.dumps(result,indent=2))


if __name__=="__main__":main()
