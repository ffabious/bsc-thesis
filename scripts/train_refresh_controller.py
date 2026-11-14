#!/usr/bin/env python3
"""Fit a CPU refresh-benefit model, tune on late CT, then replay fixed choices on PV."""

import argparse
import copy
import json
import time
from pathlib import Path

import duckdb
import numpy as np
import torch

from blend_sasrec import checksum
from benchmark_refresh_controller import coupled_branches,measure_cases
from distill_sasrec import load_student
from controller_horizon import replace_gain
from list_cache import list_ranks
from refresh_controller import HistoryFeatures,fit_ridge,predict_matrix,rollout,training_examples
from mlp_refresh_controller import fit_mlp
from replay_sasrec import encode_queries,quality
from replay_sasrec_policies import frontier,policy_cost,rank_reusing,select_budgets
from run_sasrec import ROOT,state_digest,write
from sasrec import ranks


def read(path):return json.loads(path.read_text())


def region_rows(names,rank,target_ids,refresh,region,stream,decisions,branches):
    requests=int(region.sum());targets=region[stream["target_request"][target_ids]]
    result=[]
    for j,name in enumerate(names):
        updates=int(refresh[j,np.flatnonzero(region)].sum())
        costs=branches[j] if isinstance(branches,list) else branches
        result.append({"name":name,"metrics":quality(rank[j,targets]),"requests":requests,
            "targets":int(targets.sum()),"refreshes":updates,"refresh_fraction":updates/requests,
            "cost":policy_cost(updates,requests,decisions[j],costs)})
    return result


def measured_rollout(stream,features,model,threshold,start,end,repeats):
    rollout(stream,features,model,threshold,start,end)  # Warm the CPU path.
    trials=[];saved=None
    for _ in range(repeats):
        started=time.perf_counter();current=rollout(stream,features,model,threshold,start,end)
        trials.append(1000*(time.perf_counter()-started)/(end-start))
        if saved is not None:assert all(np.array_equal(x,y) for x,y in zip(current,saved))
        saved=current
    return saved,{"median_ms_per_request":float(np.median(trials)),"repeat_ms_per_request":trials}


def reprice(block,branches):
    block=copy.deepcopy(block)
    for row in block["policies"]:
        row["cost"]=policy_cost(row["refreshes"],row["requests"],row["cost"]["decision_ms_per_request"],branches)
    block["frontiers"]={key:frontier(block["policies"],key) for key in ["ndcg@10","recall@100"]}
    return block


def validate_mixed(model,vectors,stream,features,controller,names,thresholds,refs,updates,before,simple_names,baseline,calibration,pool,config,device):
    cases=[]
    for j,name in enumerate(names):
        cases.append({"name":f"controller/{name}","pool":pool,"refs":refs[j],"refresh":updates[j],"before":before[j],"threshold":thresholds[j]})
    for name in simple_names:
        j=next(i for i,p in enumerate(baseline["policies"]) if p["name"]==name)
        cases.append({"name":f"simple/{name}","pool":pool,"refs":stream["refs"][j],"refresh":stream["refresh"][j]})
    measured=measure_cases(model,vectors,stream,features,controller,cases,config,device)
    for key,row in measured["cases"].items():
        kind,name=key.split("/");branches=calibration["controllers"][name] if kind=="controller" else calibration["plain"]
        predicted=policy_cost(row["sample_refreshes"],row["sample_requests"],0,branches)["estimated_ms_per_request"]
        row.update(predicted_sample_ms=predicted,measured_minus_predicted_percent=100*(row["median_mean_ms"]/predicted-1))
    return measured


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=ROOT/"configs/experiments/student43-controller.json")
    parser.add_argument("--output",type=Path,default=ROOT/"data/runs/student43-controller")
    parser.add_argument("--device",choices=["mps","cpu"],default="mps")
    args=parser.parse_args();config=read(args.config);torch.set_num_threads(config["threads"])
    if args.output.exists():parser.error("Choose a new controller output")
    base=ROOT/config["policy_replay"];baseline=read(base/"results.json")
    assert baseline["device"]==args.device and baseline["config"]["cost_budgets_relative_to_always"]==config["budgets"]
    assert checksum(base/"results.json")==config["policy_results_sha256"] and checksum(base/"replay.npz")==config["policy_arrays_sha256"]
    stream=dict(np.load(base/"replay.npz"));data=ROOT/baseline["config"]["data"]
    periods=read(data/"manifest.json")["periods"];period=next(p for p in periods if p["split"]=="controller_train")
    fit_cutoff=period["start"]+config["fit_days"]*86400
    cutoff=fit_cutoff+config.get("early_stop_days",0)*86400;ct=stream["split_idx"]==1;ct_end=int(ct.sum())
    assert np.array_equal(np.flatnonzero(ct),np.arange(ct_end)) and cutoff<period["end"]
    fit=ct&(stream["timestamp"]<fit_cutoff);early=ct&(stream["timestamp"]>=fit_cutoff)&(stream["timestamp"]<cutoff);valid=ct&(stream["timestamp"]>=cutoff)
    with duckdb.connect(config={"threads":config["threads"],"memory_limit":"512MB"}) as db:
        prefix=db.execute("SELECT user_idx,timestamp FROM read_parquet(?) WHERE timestamp>=? AND timestamp<? ORDER BY user_idx,timestamp",
            [str(data/"events/model_valid.parquet"),period["start"]-86400,period["start"]]).fetchnumpy()
    features=HistoryFeatures(stream,prefix["user_idx"],prefix["timestamp"])
    hashes={str((data/p).relative_to(ROOT)):h for p,h in baseline["used_data_files_sha256"].items()}
    sources=[args.config.resolve(),Path(__file__).resolve(),ROOT/"scripts/refresh_controller.py",base/"results.json",base/"replay.npz",
        ROOT/"scripts/replay_sasrec_policies.py",ROOT/"scripts/replay_sasrec.py",ROOT/"scripts/sasrec.py",ROOT/"scripts/run_sasrec.py",ROOT/"scripts/benchmark_refresh_controller.py",ROOT/"scripts/mlp_refresh_controller.py",
        ROOT/baseline["config"]["checkpoint"],data/"manifest.json"]
    labels=None
    if config.get("training_labels"):
        path=ROOT/config["training_labels"]
        assert checksum(path)==config["training_labels_sha256"]
        labels=dict(np.load(path));sources += [path,ROOT/"scripts/controller_horizon.py"]
    top100=None
    if baseline["config"].get("top100_cache"):
        path=ROOT/baseline["config"]["top100_cache"]
        assert checksum(path)==baseline["top100_cache_sha256"]
        top100=np.load(path);sources += [path,ROOT/"scripts/list_cache.py"]
    hashes.update({str(p.relative_to(ROOT)):checksum(p) for p in sources})
    for p,h in hashes.items():assert checksum(ROOT/p)==h,p
    started=time.perf_counter();rows=training_examples(stream,stream["ranks"],stream["refs"],fit)
    if labels is not None:rows=replace_gain(rows,labels,"fit")
    x=features.matrix(rows["requests"],rows["previous"])
    training=None
    if config.get("learner")=="mlp":
        early_rows=training_examples(stream,stream["ranks"],stream["refs"],early)
        if labels is not None:early_rows=replace_gain(early_rows,labels,"early")
        vx=features.matrix(early_rows["requests"],early_rows["previous"])
        controller,training=fit_mlp(x,rows["gain"],rows["weights"],vx,early_rows["gain"],early_rows["weights"],config)
    else:controller=fit_ridge(x,rows["gain"],rows["weights"],config["ridge"])
    prediction=predict_matrix(controller,x);order=np.argsort(prediction)
    thresholds=np.interp(config["threshold_quantiles"],np.cumsum(rows["weights"][order])/rows["weights"].sum(),prediction[order])
    choices=[]
    for quantile,threshold in zip(config["threshold_quantiles"],thresholds):
        if not any(c["threshold"]==float(threshold) for c in choices):
            choices.append({"name":f"controller_q{round(100*quantile):03d}","threshold":float(threshold),"fit_quantile":quantile})
    args.output.mkdir(parents=True);write(args.output/"model.json",controller)
    if training:write(args.output/"model-training.json",training)
    np.savez_compressed(args.output/"training-examples.npz",**rows)
    fit_info={"cutoff":cutoff,"fit_cutoff":fit_cutoff,"early_stop_requests":int(early.sum()),"fit_requests":int(fit.sum()),"tuning_requests":int(valid.sum()),"contexts":len(x),
        "weighted_requests":float(rows["weights"].sum()),"weighted_positive_gain_fraction":float(np.average(rows["gain"]>0,weights=rows["weights"])),
        "weighted_negative_gain_fraction":float(np.average(rows["gain"]<0,weights=rows["weights"])),"elapsed_seconds":time.perf_counter()-started}
    description={k:controller[k] for k in ["kind","parameters","best_epoch","epochs"]} if controller.get("kind")=="mlp" else controller
    print(json.dumps({"fit":fit_info,"model":description,"candidates":choices}),flush=True);del x,prediction
    ct_refs=[];ct_updates=[];ct_before=[];decision=[]
    for choice in choices:
        state,timing=measured_rollout(stream,features,controller,choice["threshold"],0,ct_end,config["decision_repeats"])
        r,f,b=state;ct_refs.append(r);ct_updates.append(f);ct_before.append(b);decision.append(timing)
        print(json.dumps({"candidate":choice["name"],"ct_refreshes":int(f.sum()),"decision":timing}),flush=True)
    ct_refs,ct_updates,ct_before=np.array(ct_refs),np.array(ct_updates),np.array(ct_before)
    model,_=load_student((ROOT/baseline["config"]["checkpoint"]).parent,args.device);before_state=state_digest(model)
    assert before_state==baseline["frozen_state_sha256"]
    vectors,encoding=encode_queries(model,stream["queries"],args.device,baseline["config"]["batch_size"])
    ct_targets=np.flatnonzero(ct[stream["target_request"]]);pv_targets=np.flatnonzero(~ct[stream["target_request"]])
    if top100 is None:
        ct_ranks,ct_scoring=rank_reusing(model,vectors,ct_refs[:,stream["target_request"][ct_targets]],stream["targets"][ct_targets],args.device,256,
            (stream["refs"][:,stream["target_request"][ct_targets]],stream["ranks"][:,ct_targets]))
    else:
        ct_ranks=list_ranks(top100,ct_refs[:,stream["target_request"][ct_targets]],stream["targets"][ct_targets])
        ct_scoring={"source":"precomputed exact-top100 lists","additional_model_scoring":False}
    ct_calibration=coupled_branches(model,vectors,stream,features,controller,[c["name"] for c in choices],
        [c["threshold"] for c in choices],ct_refs,ct_updates,ct_before,np.flatnonzero(ct),config,args.device)
    candidate_rows=region_rows([c["name"] for c in choices],ct_ranks,ct_targets,ct_updates,valid,stream,
        [0.]*len(choices),[ct_calibration["controllers"][c["name"]] for c in choices])
    simple_names=[p["name"] for p in baseline["policies"]]
    simple_rows=region_rows(simple_names,stream["ranks"][:,ct_targets],ct_targets,stream["refresh"],valid,stream,
        [p["cost"]["decision_ms_per_request"] for p in baseline["results"]["controller_train"]["policies"]],ct_calibration["plain"])
    repriced_ct=reprice(baseline["results"]["controller_train"],ct_calibration["plain"])
    policy_selection={"source_split":"controller_train","criterion":baseline["selection"]["criterion"],
        "budgets":select_budgets(repriced_ct["policies"],config["budgets"])}
    write(args.output/"policy-selection.json",policy_selection)
    policy_selection_hash=checksum(args.output/"policy-selection.json")
    always=simple_rows[0];selected=select_budgets([always,*candidate_rows],config["budgets"])
    for row in selected:
        row["controller"]=row.pop("policy")
        if row["controller"]:
            current=next(p for p in candidate_rows if p["name"]==row["controller"])
            cap=current["cost"]["estimated_ms_per_request"]/always["cost"]["estimated_ms_per_request"]
            row["matched_simple"]=select_budgets(simple_rows,[cap])[0]["policy"]
            row["tuning_controller"]=current
            row["tuning_simple"]=next(p for p in simple_rows if p["name"]==row["matched_simple"])
        else:row["matched_simple"]=None
    selection={"source":"last four days of controller_train","fit_days":config["fit_days"],"cutoff":cutoff,
        "criterion":"max NDCG, ties Recall then lower full cost; simple comparator has CT tuning cost <= controller",
        "budgets":selected,"candidates":choices,"tuning_controller_rows":candidate_rows,"tuning_simple_rows":simple_rows,
        "policy_valid_not_used_for_selection":True,"model_sha256":checksum(args.output/"model.json")}
    write(args.output/"selection.json",selection);selection_hash=checksum(args.output/"selection.json")
    print(json.dumps({"selected":selected}),flush=True)
    names=list(dict.fromkeys(p["controller"] for p in selected if p["controller"]))
    assert names,"No controller candidate meets any agreed budget"
    indices=[next(i for i,c in enumerate(choices) if c["name"]==name) for name in names]
    refs=[];updates=[];before=[];pv_decisions=[]
    for j in indices:
        state,timing=measured_rollout(stream,features,controller,choices[j]["threshold"],ct_end,len(ct),config["decision_repeats"])
        r,f,b=state;refs.append(np.r_[ct_refs[j],r]);updates.append(np.r_[ct_updates[j],f]);before.append(np.r_[ct_before[j],b]);pv_decisions.append(timing)
    refs,updates,before=np.array(refs),np.array(updates),np.array(before)
    assert np.all(stream["users"][refs]==stream["users"][None])
    assert np.all(stream["split_idx"][refs]==stream["split_idx"][None])
    assert np.all(stream["last_input_timestamp"][refs]<stream["timestamp"][None])
    if top100 is None:
        pv_ranks,pv_scoring=rank_reusing(model,vectors,refs[:,stream["target_request"][pv_targets]],stream["targets"][pv_targets],args.device,256,
            (stream["refs"][:,stream["target_request"][pv_targets]],stream["ranks"][:,pv_targets]))
    else:
        pv_ranks=list_ranks(top100,refs[:,stream["target_request"][pv_targets]],stream["targets"][pv_targets])
        pv_scoring={"source":"precomputed exact-top100 lists","additional_model_scoring":False}
    all_ranks=np.empty((len(names),len(stream["targets"])),dtype=np.int32);all_ranks[:,ct_targets]=ct_ranks[indices];all_ranks[:,pv_targets]=pv_ranks
    thresholds=[choices[j]["threshold"] for j in indices]
    pv_calibration=coupled_branches(model,vectors,stream,features,controller,names,thresholds,refs,updates,before,np.flatnonzero(~ct),config,args.device)
    calibrations={"controller_train":ct_calibration,"policy_valid":pv_calibration};summary={};mixed={}
    simple_checks=list(dict.fromkeys(["always","never",*[p["policy"] for p in policy_selection["budgets"] if p["policy"]],
        *[p["matched_simple"] for p in selected if p["matched_simple"]]]))
    for split,region,target_ids,rank in [("controller_train",ct,ct_targets,all_ranks[:,ct_targets]),("policy_valid",~ct,pv_targets,all_ranks[:,pv_targets])]:
        calibration=calibrations[split]
        summary[split]=region_rows(names,rank,target_ids,updates,region,stream,[0.]*len(names),[calibration["controllers"][n] for n in names])
        mixed[split]=validate_mixed(model,vectors,stream,features,controller,names,thresholds,refs,updates,before,simple_checks,
            baseline,calibration,np.flatnonzero(region),config,args.device)
    p,t=np.unravel_index(np.random.default_rng(config["seed"]).choice(all_ranks.size,256,replace=False),all_ranks.shape)
    cpu,_=load_student((ROOT/baseline["config"]["checkpoint"]).parent,"cpu")
    oracle=ranks(cpu,stream["queries"][refs[p,stream["target_request"][t]]],stream["targets"][t],"cpu",256)
    assert quality(oracle)==quality(all_ranks[p,t]);assert state_digest(model)==before_state
    assert selection_hash==checksum(args.output/"selection.json")
    assert policy_selection_hash==checksum(args.output/"policy-selection.json")
    for path,h in hashes.items():assert checksum(ROOT/path)==h,path
    np.savez_compressed(args.output/"replay.npz",refs=refs,refresh=updates,before=before,ranks=all_ranks,names=np.array(names),
        targets=stream["targets"],target_request=stream["target_request"],target_event_ids=stream["target_event_ids"])
    np.savez_compressed(args.output/"ct-candidates.npz",refs=ct_refs,refresh=ct_updates,ranks=ct_ranks)
    repriced=copy.deepcopy(baseline)
    repriced.update(results={"controller_train":repriced_ct,"policy_valid":reprice(baseline["results"]["policy_valid"],pv_calibration["plain"])},
        branch_costs={s:c["plain"] for s,c in calibrations.items()},selection=policy_selection,
        mixed_workload_checks={s:{k:v for k,v in b["cases"].items() if k.startswith("simple/")} for s,b in mixed.items()},
        cost_recalibrated_with_controller=True,cost_recalibration_source_sha256=hashes,selection_sha256=policy_selection_hash)
    write(args.output/"policy-results.json",repriced)
    oracle_info={"pairs":256,"same_metrics":True,"exact_ranks":int((oracle==all_ranks[p,t]).sum()),"max_rank_delta":int(np.abs(oracle-all_ranks[p,t]).max())}
    if top100 is not None:
        inside=oracle<=100;assert np.array_equal(inside,all_ranks[p,t]<=100)
        oracle_info={"pairs":256,"same_metrics":True,"same_top100_membership":True,"max_top100_rank_delta":int(np.abs(oracle[inside]-all_ranks[p,t][inside]).max(initial=0))}
    result={"config":config,"model":controller,"fit":fit_info,"selection":selection,"selection_sha256":selection_hash,
        "policy_selection_sha256":policy_selection_hash,"results":summary,"calibration":calibrations,
        "decision_costs":{"controller_train":{c["name"]:d for c,d in zip(choices,decision)},"policy_valid":dict(zip(names,pv_decisions))},
        "mixed_workload_checks":mixed,"checkpoint_sha256":baseline["checkpoint_sha256"],"frozen_state_unchanged":True,
        "cpu_oracle":oracle_info,"rank_scope":baseline.get("rank_scope","Full catalog ranks"),
        "execution":{"encode_seconds":encoding,"ct_scoring":ct_scoring,"pv_scoring":pv_scoring},"source_sha256":hashes,
        "replay_sha256":checksum(args.output/"replay.npz"),"total_seconds":time.perf_counter()-started,
        "no_test_events_read":True,"policy_valid_not_used_for_training_or_selection":True,
        "cost_scope":"Interleaved complete B=1 branches: controller scalar CPU features/prediction + MPS serving; no separate CPU charge; history loading/network excluded"}
    write(args.output/"results.json",result);print(json.dumps({"completed":True,"seconds":result["total_seconds"],"results":summary,"mixed":mixed}),flush=True)


if __name__=="__main__":main()
