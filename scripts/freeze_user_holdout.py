#!/usr/bin/env python3
"""Freeze a user-disjoint final protocol using only train eligibility and saved CT results."""

import argparse
import hashlib
import json
from pathlib import Path

import duckdb
import numpy as np

from blend_sasrec import checksum
from report_student_maintenance import cost_matched_mixture
from run_sasrec import ROOT,write


def read(path):return json.loads(path.read_text())


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--config",type=Path,required=True)
    args=parser.parse_args();config=read(args.config);output=ROOT/config["output"];report=ROOT/config["report"]
    if output.exists() or report.exists():parser.error("Choose new frozen protocol outputs")
    datasets={};training_paths=[]
    for path in sorted((ROOT/"data/runs").rglob("training.json")):
        metadata=read(path);data=metadata.get("config",{}).get("data")
        if data:datasets.setdefault(data,[]).append(str(path.relative_to(ROOT)));training_paths.append(path)
    assert set(datasets)<=set(config["exclude_user_datasets"]),"An additional trained cohort must be excluded"
    pool=ROOT/config["pool_data"];vocabulary=ROOT/config["model_data"];manifest=read(pool/"manifest.json")
    with duckdb.connect(config={"threads":2,"memory_limit":"512MB"}) as db:
        users=db.execute("SELECT uid,train_events FROM read_parquet(?) WHERE train_events>=20 ORDER BY uid",[str(pool/"users.parquet")]).fetchall()
        excluded=set()
        for data in config["exclude_user_datasets"]:
            excluded.update(uid for uid, in db.execute("SELECT uid FROM read_parquet(?)",[str(ROOT/data/"users.parquet")]).fetchall())
    eligible=[(uid,count) for uid,count in users if uid not in excluded]
    eligible.sort(key=lambda row:hashlib.sha256(f"{config['cohort_seed']}:{row[0]}".encode()).digest())
    selected=eligible[:config["users"]];assert len(selected)==config["users"] and not ({uid for uid,_ in selected}&excluded)
    controller_root=ROOT/config["controller"];controller=read(controller_root/"results.json");grid_path=ROOT/config["grid"]
    grid=read(grid_path)["policies"];checkpoint=ROOT/config["checkpoint"];model_manifest=read(vocabulary/"manifest.json")
    assert checksum(checkpoint)==controller["checkpoint_sha256"]==config["checkpoint_sha256"]
    assert checksum(vocabulary/"manifest.json")==read(checkpoint.with_name("training.json"))["data_manifest_sha256"]
    policies=[]
    for pick in controller["selection"]["budgets"]:
        if pick["controller"] is None:continue
        candidate=next(row for row in controller["selection"]["candidates"] if row["name"]==pick["controller"])
        mixture=cost_matched_mixture(controller["selection"]["tuning_simple_rows"],pick["tuning_controller"]["cost"]["estimated_ms_per_request"])
        policies.append({"ct_budget":pick["relative_budget"],"name":pick["controller"],"threshold":candidate["threshold"],
            "matched_simple":pick["matched_simple"],"expected_mixture":mixture})
    assert len(policies)==2 and [p["ct_budget"] for p in policies]==[.75,.9]
    sources=[args.config.resolve(),Path(__file__).resolve(),checkpoint,pool/"manifest.json",pool/"users.parquet",vocabulary/"manifest.json",vocabulary/"items.parquet",grid_path]
    sources += [ROOT/data/"users.parquet" for data in config["exclude_user_datasets"]]
    sources += [controller_root/name for name in ["results.json","model.json","selection.json","policy-selection.json"]]
    hashes={str(path.relative_to(ROOT)):checksum(path) for path in sources}
    output.mkdir(parents=True);report.mkdir(parents=True)
    text="uid,user_idx,train_events\n"+"".join(f"{uid},{i},{count}\n" for i,(uid,count) in enumerate(selected))
    (output/"selected-users.csv").write_text(text)
    protocol={"config":config,"source_sha256":hashes,"cohort":{"users":len(selected),"eligible_outside_development":len(eligible),
        "excluded_development_users":len(excluded),"overlap":0,"selection":"sha256(seed:uid), ascending digest; train_events>=20; no holdout activity/labels used",
        "selected_users_sha256":checksum(output/"selected-users.csv")},"completed_training_audit":datasets,
        "selection_metadata_sha256":{str(path.relative_to(ROOT)):checksum(path) for path in training_paths},
        "periods":manifest["periods"],"candidate_items":model_manifest["candidate_items"],"policies":grid,"controllers":policies,
        "controller_model_sha256":checksum(controller_root/"model.json"),"vocabulary_sha256":checksum(vocabulary/"items.parquet"),
        "primary_analysis":"Two NDCG@10 controller-minus-fixed-simple comparisons at CT budgets75/90%; user-cluster bootstrap97.5% per comparison (Bonferroni over two primary comparisons)",
        "secondary_analysis":"Recall@100/Hit@1, paired fresh losses, fixed CT expected-mixture references and descriptive full-grid frontier; pointwise95% CI where reported",
        "histories":"All closed-audio listens retained; fixed CE1k vocabulary, unseen items=UNK; current timestamp group excluded; cached state reset at test start",
        "target":"Positive listens played_ratio_pct>50 with strict past and known CE1k item; coverage of excluded unknown targets reported",
        "no_refit_or_policy_selection_on_new_users":True,"new_user_test_events_read":False,
        "old_test_limitation":"Earlier encoder experiments read test on development users; old test is not an untouched end-to-end holdout. New cohort is user-disjoint, with warm histories and frozen item vocabulary."}
    write(report/"protocol.json",protocol);write(output/"protocol.json",protocol)
    print(json.dumps({"cohort":protocol["cohort"],"controllers":policies,"protocol_sha256":checksum(report/"protocol.json")},indent=2))


if __name__=="__main__":main()
