#!/usr/bin/env python3
"""Build one fixed-horizon refresh label from frozen CT predictions, reusing ranks."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from blend_sasrec import checksum
from controller_horizon import horizon_design,horizon_gain
from distill_sasrec import load_student
from refresh_controller import training_examples
from replay_sasrec import encode_queries,quality,rank_vectors
from run_sasrec import ROOT,state_digest,write
from sasrec import ranks


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--device",choices=["cpu","mps"],default="mps")
    parser.add_argument("--plan-only",action="store_true")
    args=parser.parse_args();config=json.loads(args.config.read_text());torch.set_num_threads(config["threads"])
    base=ROOT/config["policy_replay"];stream=dict(np.load(base/"replay.npz"));meta=json.loads((base/"results.json").read_text())
    assert checksum(base/"replay.npz")==config["policy_arrays_sha256"]
    assert checksum(base/"results.json")==config["policy_results_sha256"]
    data=ROOT/meta["config"]["data"]
    period=next(p for p in json.loads((data/"manifest.json").read_text())["periods"] if p["split"]=="controller_train")
    cut=period["start"]+config["fit_days"]*86400;end=cut+config["early_stop_days"]*86400
    ct=stream["split_idx"]==1
    regions={"fit":ct&(stream["timestamp"]<cut),"early":ct&(stream["timestamp"]>=cut)&(stream["timestamp"]<end)}
    checkpoint=ROOT/meta["config"]["checkpoint"]
    assert checksum(checkpoint)==meta["checkpoint_sha256"]
    width=torch.load(checkpoint,map_location="cpu",weights_only=True)["items"]+2
    paths=[args.config.resolve(),Path(__file__).resolve(),base/"results.json",base/"replay.npz",checkpoint,data/"manifest.json"]
    paths += [ROOT/"scripts"/name for name in ["controller_horizon.py","refresh_controller.py","replay_sasrec.py","sasrec.py","distill_sasrec.py"]]
    hashes={str(p.relative_to(ROOT)):checksum(p) for p in paths};started=time.perf_counter()
    rows={name:training_examples(stream,stream["ranks"],stream["refs"],region) for name,region in regions.items()}
    designs={name:horizon_design(stream,rows[name],region,config["horizon_requests"],width) for name,region in regions.items()}
    keys=np.unique(np.concatenate([design[field] for design in designs.values() for field in ["fresh","stale"]]))
    old_keys,first=np.unique(stream["refs"][:,stream["target_request"]].astype(np.int64)*width+stream["targets"][None],return_index=True)
    old_rank=stream["ranks"].ravel()[first];pos=np.searchsorted(old_keys,keys).clip(max=len(old_keys)-1);reuse=old_keys[pos]==keys
    plan={"unique_pairs":len(keys),"reused_pairs":int(reuse.sum()),"new_pairs":int((~reuse).sum()),
        "contexts":{name:len(row["gain"]) for name,row in rows.items()},"horizon":config["horizon_requests"],"fit_end":cut,"early_end":end}
    print(json.dumps(plan),flush=True)
    if args.plan_only:return
    output=ROOT/config["output"]
    if output.exists():parser.error("Choose a new label output")
    model,_=load_student(checkpoint.parent,args.device);state=state_digest(model)
    # Only training/early-validation snapshots are encoded; PV never contributes labels.
    used=np.unique(keys//width);mapping=np.full(len(stream["queries"]),-1,dtype=np.int32);mapping[used]=np.arange(len(used))
    vectors,encoding=encode_queries(model,stream["queries"][used],args.device,256)
    rank=np.empty(len(keys),dtype=np.int32);rank[reuse]=old_rank[pos[reuse]]
    missing=keys[~reuse]
    values,timing=rank_vectors(model,vectors,mapping[missing//width][None],missing%width,args.device,256);rank[~reuse]=values[0]
    saved={};statistics={}
    for name,row in rows.items():
        gain=horizon_gain(designs[name],keys,rank);weights=row["weights"]
        saved.update({f"{name}_{field}":row[field] for field in ["requests","previous"]});saved[f"{name}_gain"]=gain
        statistics[name]={"weighted_positive_fraction":float(np.average(gain>0,weights=weights)),
            "weighted_negative_fraction":float(np.average(gain<0,weights=weights)),"mean_gain":float(np.average(gain,weights=weights)),
            "short_horizon_contexts":int((designs[name]["requests_in_horizon"]<config["horizon_requests"]).sum())}
    # An independent direct CPU oracle checks pairs used to construct the new labels.
    chosen=np.random.default_rng(42).choice(len(keys),256,replace=False);cpu,_=load_student(checkpoint.parent,"cpu")
    oracle=ranks(cpu,stream["queries"][keys[chosen]//width],keys[chosen]%width,"cpu",256)
    assert quality(oracle)==quality(rank[chosen]) and state_digest(model)==state
    for path,h in hashes.items():assert checksum(ROOT/path)==h,path
    output.mkdir(parents=True);np.savez_compressed(output/"labels.npz",**saved);np.savez_compressed(output/"pairs.npz",keys=keys,ranks=rank)
    result={"config":config,"plan":plan,"statistics":statistics,"source_sha256":hashes,"label_sha256":checksum(output/"labels.npz"),
        "execution":{"encoding_seconds":encoding,"scoring":timing,"total_seconds":time.perf_counter()-started},
        "cpu_oracle":{"same_metrics":True,"exact_ranks":int((oracle==rank[chosen]).sum()),"max_rank_delta":int(np.abs(oracle-rank[chosen]).max())},
        "labels_only_ct_fit_and_early":True,"no_test_events_read":True,
        "target_definition":"Mean of summed positive-target NDCG rewards over current and next four same-user requests, holding each compared snapshot fixed; truncated at region end"}
    write(output/"results.json",result);print(json.dumps(result),flush=True)


if __name__=="__main__":main()
