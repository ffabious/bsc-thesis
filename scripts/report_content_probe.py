#!/usr/bin/env python3
"""Audit and publish the prespecified validation gate without starting a replay."""

import shutil
from pathlib import Path

import numpy as np

from blend_sasrec import checksum
from content_refresh_features import CONTENT_FEATURES
from plot_sasrec_replay import plt,save
from report_controller_models import read
from run_sasrec import ROOT,write


def main():
    root=ROOT/"data/runs/controller-content4-ce1k";out=ROOT/"reports/experiments/content-controller-probe";out.mkdir(parents=True,exist_ok=True)
    probe=read(root/"probe.json");gate=read(root/"validation-gate.json");curve=read(root/"model-training.json");model=read(root/"model.json")
    reference=ROOT/probe["config"]["validation_reference"]["path"];old=read(reference.parent/"model-training.json")
    for path,h in probe["source_sha256"].items():assert checksum(ROOT/path)==h,path
    assert not any("final-user-holdout" in path for path in probe["source_sha256"])
    assert not probe["replay_performed"] and probe["policy_valid_predictions_not_evaluated"]
    assert not (root/"results.json").exists() and not (root/"replay.npz").exists()
    assert model["features"]==CONTENT_FEATURES and model["parameters"]==1505
    assert curve["stop_reason"]=="validation_patience" and curve["best_epoch"]==min(curve["epochs"],key=lambda row:row["validation_mse"])["epoch"]
    np.testing.assert_allclose(gate["new_mse"],curve["best_validation_mse"],rtol=0,atol=1e-8)
    np.testing.assert_allclose(gate["reference_mse"],old["best_validation_mse"],rtol=0,atol=1e-8)
    np.testing.assert_allclose(gate["relative_improvement"],1-gate["new_mse"]/gate["reference_mse"],rtol=0,atol=1e-15)
    assert gate["minimum_relative_improvement"]==.01 and not gate["passed"]
    new_rows=dict(np.load(root/"training-examples.npz"));old_rows=dict(np.load(reference.parent/"training-examples.npz"))
    for field in new_rows:np.testing.assert_array_equal(new_rows[field],old_rows[field])
    verification={"same_training_requests_states_labels_weights":True,"validation_gate_matches_independent_torch_curves":True,
        "best_epoch_restored":True,"parameters":model["parameters"],"features":len(model["features"]),
        "no_new_pv_replay_or_serving":True,"previous_user_holdout_not_an_input":True,
        "source_sha256":{str(path.relative_to(ROOT)):checksum(path) for path in [Path(__file__).resolve(),root/"probe.json",root/"model.json",root/"model-training.json",root/"validation-gate.json",root/"training-examples.npz",reference,reference.parent/"model-training.json",reference.parent/"training-examples.npz"]}}
    write(out/"verification.json",verification)
    for filename in ["probe.json","validation-gate.json","model.json","model-training.json"]:shutil.copyfile(root/filename,out/filename)
    shutil.copyfile(reference.parent/"model-training.json",out/"reference-training.json")
    fig,ax=plt.subplots(figsize=(7,4))
    for label,data,color in [("8 age/activity features",old,"#777777"),("12 with item cues",curve,"#168469")]:
        ax.plot([row["epoch"] for row in data["epochs"]],[row["validation_mse"] for row in data["epochs"]],"o-",label=label,c=color)
    ax.axhline(gate["reference_mse"]*.99,color="#c87220",linestyle="--",label="Required 1% improvement")
    ax.set(xlabel="Epoch",ylabel="Weighted early-validation MSE",title="Development CT days 8–10, fixed gate");ax.grid(alpha=.2);ax.legend(frameon=False)
    fig.tight_layout();save(fig,out,"validation-curve")
    print({"gate":gate,"best_epoch":curve["best_epoch"],"epochs":len(curve["epochs"]),"feature_info":probe["feature_info"],"fit":probe["fit"],"memory":probe["memory"],"verification":verification})


if __name__=="__main__":main()
