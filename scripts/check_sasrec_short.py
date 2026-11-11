#!/usr/bin/env python3
"""Check native suffix inference against padded inference and frozen CT/PV ranks."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from blend_sasrec import checksum,load_pair,normalize_scores
from profile_sasrec_hybrid import Serving
from replay_hybrid_sasrec import assemble,mixed_ranks,pair_keys,summarize
from replay_sasrec import encode_queries,quality,rank_vectors
from run_sasrec import ROOT,memory,write


def rank_difference(actual,expected):
    delta = np.abs(actual.astype(np.int64)-expected.astype(np.int64))
    return {"pairs":actual.size,"exact_matches":int((delta==0).sum()),
            "maximum_rank_difference":int(delta.max()),
            "metric_difference":{k:quality(actual)[k]-v for k,v in quality(expected).items()}}


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hybrid-run",type=Path,required=True)
    parser.add_argument("--device",choices=["cpu","mps"],required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--full-replay",action="store_true",help="Recompute every selected mixture; keep CT alpha fixed")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output directory")
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK","0")!="0":
        parser.error("Disable MPS CPU fallback")
    run = ROOT/args.hybrid_run
    original = json.loads((run/"results.json").read_text())
    selection = json.loads((run/"selection.json").read_text())
    config = original["config"];torch.set_num_threads(config["threads"])
    folder = ROOT/config["reference_replay"]
    base = dict(np.load(folder/"replay.npz"))
    reference = json.loads((folder/"results.json").read_text())
    old = dict(np.load(run/"ranks.npz"))
    paths = [run/"results.json",run/"selection.json",run/"ranks.npz",folder/"replay.npz",folder/"results.json"]
    paths += [ROOT/"scripts"/name for name in ["check_sasrec_short.py","sasrec.py","profile_sasrec_hybrid.py",
        "replay_hybrid_sasrec.py","replay_sasrec.py","blend_sasrec.py"]]
    hashes = {str(p.relative_to(ROOT)):checksum(p) for p in paths}
    for p,h in original["reference_sha256"].items():
        if checksum(ROOT/p)!=h:
            raise ValueError("Reference replay changed")
    if checksum(run/"selection.json")!=original["selection_sha256"]:
        raise ValueError("Frozen alpha selection changed")
    models,configs,sources = load_pair(config,"256",args.device)
    if sources!=original["sources"]:
        raise ValueError("Checkpoints differ from the original replay")
    full = Serving(models,configs,args.device)
    short = Serving(models,configs,args.device,recent_input_length=2)
    vectors = {}
    vectors["recent"],_ = encode_queries(models["recent"],short.inputs("recent",base["queries"]),args.device,256)
    expected,_ = encode_queries(models["recent"],full.inputs("recent",base["queries"]),args.device,256)
    # Fixed before execution; test numerical equality separately from ranking ties.
    atol,rtol = 2e-5,1e-4
    torch.testing.assert_close(vectors["recent"],expected,atol=atol,rtol=rtol)
    checks = {"atol":atol,"rtol":rtol,"input_width":2,"positions":[48,49],
              "request_vectors":{"pairs":len(expected),"max_abs_error":float((vectors["recent"]-expected).abs().max())}}
    sample = np.random.default_rng(42).choice(len(expected),256,replace=False)
    sample = torch.as_tensor(sample,device=args.device)
    left,right = vectors["recent"][sample]@models["recent"].item.weight[2:].T,expected[sample]@models["recent"].item.weight[2:].T
    torch.testing.assert_close(left,right,atol=atol,rtol=rtol)
    checks["catalog_scores"] = {"requests":len(sample),"candidates":left.shape[1],"max_abs_error":float((left-right).abs().max())}
    normalize_scores(left);normalize_scores(right)
    torch.testing.assert_close(left,right,atol=atol,rtol=rtol)
    checks["normalized_scores_max_abs_error"] = float((left-right).abs().max())
    del expected,left,right
    recent,timing = rank_vectors(models["recent"],vectors["recent"],base["target_request"][None],base["targets"],args.device,256)
    recent = recent[0]
    results,selected = {},None
    if args.full_replay:
        vectors["long"],_ = encode_queries(models["long"],base["queries"],args.device,256)
        selected = np.empty_like(base["ranks"])
    for idx,split in enumerate(config["splits"],1):
        take = base["split_idx"][base["target_request"]]==idx
        checks[split] = {"recent":rank_difference(recent[take],old["recent"][take])}
        if args.full_replay:
            request = base["target_request"][take]
            keys,inverse = pair_keys(base["refs"][:,request],request,base["targets"][take],len(base["queries"]),models["long"].item.num_embeddings)
            inverse = inverse.reshape(100,-1)
            alphas = sorted({p["alpha"] for p in selection["policies"] if 0<p["alpha"]<1})
            needed = np.zeros((len(alphas),len(keys)),dtype=bool)
            for i,p in enumerate(selection["policies"]):
                if p["alpha"] in alphas:
                    needed[alphas.index(p["alpha"]),inverse[i]] = True
            grid,_ = mixed_ranks(models,vectors,keys,len(base["queries"]),alphas,args.device,256,1e-8,needed)
            choices = []
            for i,(p,ref) in enumerate(zip(selection["policies"],reference["results"][split]["policies"])):
                selected[i,take] = assemble(p["alpha"],alphas,grid,inverse[i],base["ranks"][i,take],recent[take])
                choices.append({**ref,"alpha":p["alpha"]})
            checks[split]["hybrid"] = {p["name"]:rank_difference(selected[i,take],old["hybrid"][i,take])
                                       for i,p in enumerate(selection["policies"])}
            results[split] = summarize(base,base,selected[:,take],recent[take],idx,choices,config["seed"])
        print(json.dumps({"split":split,"recent_equivalence":checks[split]["recent"],
            "fresh_hybrid":results.get(split,{}).get("policies",[{}])[0].get("hybrid_metrics")}),flush=True)
    data = ROOT/configs["recent"]["data"]
    if any(checksum(data/p)!=h for p,h in original["data_files_sha256"].items()):
        raise ValueError("Input data changed")
    if any(checksum(ROOT/p)!=h for p,h in hashes.items()):
        raise ValueError("Input or source changed during the check")
    for src in sources.values():
        if checksum(ROOT/src["run"]/"train-mps/best.pt")!=src["checkpoint_sha256"]:
            raise ValueError("Frozen checkpoint changed")
    write(args.output/"equivalence.json",{"checks":checks,"device":args.device,"sources":sources,
        "source_sha256":hashes,"data_files_sha256":original["data_files_sha256"],"no_test_events_read":True,**memory(args.device)})
    if args.full_replay:
        args.output.joinpath("selection.json").write_bytes((run/"selection.json").read_bytes())
        np.savez_compressed(args.output/"ranks.npz",hybrid=selected,recent=recent)
        write(args.output/"results.json",{**original,"results":results,"verification":checks,"device":args.device,
            "short_inference":{"recent_input_length":2,"original_run":str(args.hybrid_run)},
            "source_sha256":hashes,"timing":{"recent_scoring":timing},"reused_mixture_files_sha256":{},**memory(args.device)})


if __name__=="__main__":
    main()
