#!/usr/bin/env python3
"""Mix cached long-CE and fresh recent-BCE scores; select on CT and evaluate PV."""

import argparse
import hashlib
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch

from blend_sasrec import checksum, load_pair, normalize_scores, score_ranks, select_alpha, validate_grid
from compare_sasrec import paired_intervals
from replay_sasrec import encode_queries, quality, rank_vectors
from run_sasrec import ROOT, memory, model_args, write
from sasrec import SASRec, examples, mask_context, metrics, ranks, sync


def pair_keys(long_refs, recent_refs, targets, requests, width):
    # Pack in int64: recent-major ordering shares recent logits within each scoring batch.
    return np.unique((recent_refs[None].astype(np.int64)*requests + long_refs.astype(np.int64))*width +
                     targets[None], return_inverse=True)


@torch.inference_mode()
def mixed_ranks(models, vectors, keys, requests, alphas, device, batch_size, epsilon, needed=None):
    width = models["long"].item.num_embeddings
    recent, long, targets = keys // width // requests, (keys // width) % requests, keys % width
    ids = torch.arange(2, width, device=device)
    result = np.full((len(alphas), len(keys)), -1, dtype=np.int32)
    sync(device)
    started = time.perf_counter()
    for start in range(0, len(keys), batch_size):
        end = min(start + batch_size, len(keys))
        if needed is not None and not needed[:,start:end].any():
            continue
        scores = {}
        for role, refs in (("long", long), ("recent", recent)):
            chosen, inverse = np.unique(refs[start:end], return_inverse=True)
            q = vectors[role][torch.as_tensor(chosen, device=device)]
            logits = q @ models[role].item.weight[2:].T
            normalize_scores(logits, epsilon)
            scores[role] = logits[torch.as_tensor(inverse, device=device)]
        target = torch.as_tensor(targets[start:end], device=device)
        batch_ranks = torch.full((len(alphas),end-start),-1,dtype=torch.int32,device=device)
        for i, alpha in enumerate(alphas):
            if needed is None:
                mixed = (1-alpha)*scores["long"] + alpha*scores["recent"]
                batch_ranks[i] = score_ranks(mixed,ids,target)
                continue
            take = np.ones(end-start, dtype=bool) if needed is None else needed[i,start:end]
            if not take.any():
                continue
            index = torch.as_tensor(np.flatnonzero(take), device=device)
            mixed = (1-alpha)*scores["long"][index] + alpha*scores["recent"][index]
            batch_ranks[i,index] = score_ranks(mixed,ids,target[index])
        # Transfer only small ranks once per batch; keep catalog arithmetic on the device.
        result[:,start:end] = batch_ranks.cpu().numpy()
        if (start // batch_size + 1) % 64 == 0:
            print(f"mixed {end}/{len(keys)} unique pairs, elapsed {time.perf_counter()-started:.1f}s",flush=True)
    sync(device)
    return result, time.perf_counter()-started


def assemble(alpha, inner, grid, inverse, long, recent):
    if alpha == 0:
        return long
    if alpha == 1:
        return recent
    result = grid[inner.index(alpha),inverse]
    if (result < 1).any():
        raise ValueError("A selected mixture pair was not scored")
    return result


def summarize(stream, base, selected, recent, split_idx, choices, seed):
    take = stream["split_idx"][stream["target_request"]] == split_idx
    q = stream["target_request"][take]
    users, y = stream["users"][q],stream["targets"][take]
    x = stream["queries"][q]
    last, seen = x[:,-1] == y, (x == y[:,None]).any(1)
    long = base["ranks"][:,take]
    rows = []
    for i,p in enumerate(choices):
        hybrid = selected[i]
        rows.append({"name":p["name"],"family":p["family"],"events":p["events"],"seconds":p["seconds"],
                     "alpha":p["alpha"],"long_refresh_fraction":p["refresh_fraction"],
                     "long_metrics":quality(long[i]),"hybrid_metrics":quality(hybrid),
                     "hybrid_minus":{name:paired_intervals(hybrid,r,users,seed) for name,r in
                                     (("long_cached",long[i]),("long_fresh",long[0]),("recent_only",recent))},
                     "strata":{name:{"examples":int(mask.sum()),"long_metrics":quality(long[i,mask]),
                               "recent_metrics":quality(recent[mask]),"hybrid_metrics":quality(hybrid[mask])}
                               for name,mask in (("last_item",last),("other_in_window",seen & ~last),("outside_window",~seen)) if mask.any()}})
    return {"targets":len(y),"users":len(np.unique(users)),"recent_metrics":quality(recent),"policies":rows}


@torch.inference_mode()
def naive_ranks(models, configs, queries, long_refs, recent_refs, targets, alphas, device, epsilon):
    ids = torch.arange(2,models["long"].item.num_embeddings,device=device)
    logits = {}
    for role,refs in (("long",long_refs),("recent",recent_refs)):
        x = mask_context(queries[refs].copy(),configs[role].get("context_keep"))
        q = models[role](torch.as_tensor(x,dtype=torch.long,device=device))
        logits[role] = q @ models[role].item.weight[2:].T
    y = torch.as_tensor(targets,dtype=torch.long,device=device)
    endpoints = {role:score_ranks(s,ids,y) for role,s in logits.items()}
    for s in logits.values():
        normalize_scores(s,epsilon)
    a = torch.as_tensor(alphas,dtype=logits["long"].dtype,device=device)[:,None]
    result = score_ranks((1-a)*logits["long"]+a*logits["recent"],ids,y)
    result = torch.where(a[:,0] == 0,endpoints["long"],result)
    result = torch.where(a[:,0] == 1,endpoints["recent"],result)
    return result.cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--device",choices=["cpu","mps"],required=True)
    parser.add_argument("--resume",action="store_true",help="Reuse saved mixture grids after verifying keys and frozen selection")
    args = parser.parse_args()
    if args.output.exists() and not args.resume:
        parser.error("Choose a new output directory")
    if args.resume and not (args.output/"selection.json").exists():
        parser.error("Resume requires a frozen selection")
    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS unavailable")
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK","0") != "0":
        parser.error("Disable CPU fallback")
    config = json.loads(args.config.read_text())
    validate_grid(config["alphas"])
    if config["splits"] != ["controller_train","policy_valid"]:
        raise ValueError("Hybrid replay forbids test")
    torch.set_num_threads(config["threads"])
    models,configs,sources = load_pair(config,"256",args.device)
    if {role:s["checkpoint_sha256"] for role,s in sources.items()} != config["checkpoint_sha256"]:
        raise ValueError("Unexpected checkpoints")
    folder = ROOT/config["reference_replay"]
    reference_hashes = {str((folder/f).relative_to(ROOT)):config[key] for f,key in
                       (("results.json","reference_results_sha256"),("replay.npz","reference_ranks_sha256"))}
    if any(checksum(ROOT/p) != h for p,h in reference_hashes.items()):
        raise ValueError("Reference replay changed")
    reference = json.loads((folder/"results.json").read_text())
    base = dict(np.load(folder/"replay.npz"))
    if (reference["checkpoint_sha256"] != sources["long"]["checkpoint_sha256"] or
            any(s["data_manifest_sha256"] != reference["data_manifest_sha256"] for s in sources.values()) or
            reference["config"]["splits"] != config["splits"]):
        raise ValueError("Reference model/data/splits differ")
    data = ROOT/configs["long"]["data"]
    hashes = reference["used_data_files_sha256"]
    if any(checksum(data/p) != h for p,h in hashes.items()):
        raise ValueError("Dataset changed")
    source_hashes = {str(p.relative_to(ROOT)):checksum(p) for p in (args.config.resolve(),
        ROOT/"scripts/replay_hybrid_sasrec.py",ROOT/"scripts/blend_sasrec.py",ROOT/"scripts/replay_sasrec.py",
        ROOT/"scripts/sasrec.py",ROOT/"scripts/run_sasrec.py",ROOT/"scripts/compare_sasrec.py")}

    def model_digest(model):
        h = hashlib.sha256()
        for name,tensor in model.state_dict().items():
            h.update(name.encode());h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()

    states_before = {role:model_digest(m) for role,m in models.items()}
    vectors,timing = {},{}
    for role,model in models.items():
        x = mask_context(base["queries"].copy(),configs[role].get("context_keep"))
        vectors[role],timing[f"encode_{role}_seconds"] = encode_queries(model,x,args.device,config["batch_size"])
    recent,timing["recent_scoring"] = rank_vectors(models["recent"],vectors["recent"],base["target_request"][None],
                                                 base["targets"],args.device,config["batch_size"])
    recent = recent[0]
    checks = {}
    for idx,split in enumerate(config["splits"],1):
        mask = base["split_idx"][base["target_request"]] == idx
        x,y,u = examples(data,split,50)
        rows,q = base["source_row"][mask],base["target_request"][mask]
        if not (np.array_equal(x[rows],base["queries"][q]) and np.array_equal(y[rows],base["targets"][mask]) and
                np.array_equal(u[rows],base["users"][q])):
            raise ValueError("Reference stream differs from the original evaluator")
        oracle = ranks(models["recent"],mask_context(x,2),y,args.device,config["batch_size"])[rows]
        if metrics(oracle) != metrics(recent[mask]):
            raise ValueError("Recent-only metrics differ from the original evaluator")
        checks[split] = {"recent_oracle_same_inputs_and_metrics":True,"exact_rank_matches":int((oracle == recent[mask]).sum()),
                         "maximum_rank_difference":int(np.abs(oracle-recent[mask]).max())}
        print(json.dumps({"split":split,"recent_only":quality(recent[mask])}),flush=True)
    args.output.mkdir(parents=True,exist_ok=args.resume)
    prior_selection = json.loads((args.output/"selection.json").read_text()) if args.resume else None
    if prior_selection and (prior_selection["config"] != config or prior_selection["sources"] != sources):
        raise ValueError("Saved mixture selection uses different configuration or checkpoints")
    timing_path = args.output/"scoring-timing.json"
    saved_timing = json.loads(timing_path.read_text()) if args.resume and timing_path.exists() else {}
    reused_grids = {}
    inner = [a for a in config["alphas"] if 0<a<1]
    results,selection,selected_all = {},None,np.empty_like(base["ranks"])
    for idx,split in enumerate(config["splits"],1):
        mask = base["split_idx"][base["target_request"]] == idx
        req,y = base["target_request"][mask],base["targets"][mask]
        refs = base["refs"][:,req]
        keys,inverse = pair_keys(refs,req,y,len(base["users"]),models["long"].item.num_embeddings)
        inverse = inverse.reshape(refs.shape)
        needed = None
        if selection is not None:
            needed = np.zeros((len(inner),len(keys)),dtype=bool)
            for i,p in enumerate(selection["policies"]):
                if p["alpha"] in inner:
                    needed[inner.index(p["alpha"]),inverse[i]] = True
        grid_path = args.output/f"{split}-mixtures.npz"
        if args.resume and grid_path.exists():
            saved_grid = np.load(grid_path)
            if not (np.array_equal(saved_grid["keys"],keys) and np.array_equal(saved_grid["inverse"],inverse) and
                    np.array_equal(saved_grid["alphas"],inner)):
                raise ValueError("Saved mixture grid differs from the current snapshots/targets/alphas")
            grid,elapsed = saved_grid["ranks"],0
            if grid.shape != (len(inner),len(keys)) or np.any(grid[(np.ones_like(grid,dtype=bool) if needed is None else needed)] < 1):
                raise ValueError("Saved mixture grid is incomplete")
            reused_grids[str(grid_path.resolve().relative_to(ROOT))] = checksum(grid_path)
            timing[split] = {**saved_timing.get(split,{}),"unique_pairs":len(keys),"reused_grid":True}
        else:
            grid,elapsed = mixed_ranks(models,vectors,keys,len(base["users"]),inner,args.device,config["batch_size"],config["epsilon"],needed)
            timing[split] = {"unique_pairs":len(keys),"mixed_scoring_seconds":elapsed,"mixed_alpha_pairs":int((grid>0).sum())}
        curves,chosen = [],[]
        for i,p in enumerate(reference["results"][split]["policies"]):
            curve = [{"alpha":a,"metrics":quality(assemble(a,inner,grid,inverse[i],base["ranks"][i,mask],recent[mask]))}
                     for a in config["alphas"]] if selection is None else None
            alpha = select_alpha(curve) if selection is None else selection["policies"][i]["alpha"]
            curves.append({"name":p["name"],"alpha":alpha,"curve":curve})
            chosen.append({**p,"alpha":alpha})
            selected_all[i,mask] = assemble(alpha,inner,grid,inverse[i],base["ranks"][i,mask],recent[mask])
        if selection is None:
            selection = {"selected_on":"controller_train","criterion":"per-policy maximum NDCG@10; ties smaller alpha",
                         "policies":curves,"config":config,"sources":sources}
            if prior_selection:
                if selection != prior_selection:
                    raise ValueError("Saved CT selection differs from the recomputed grid maximum")
            else:
                write(args.output/"selection.json",selection)
            selection_hash = checksum(args.output/"selection.json")
        results[split] = summarize(base,base,selected_all[:,mask],recent[mask],idx,chosen,config["seed"])
        if not (args.resume and grid_path.exists()):
            np.savez(grid_path,keys=keys,ranks=grid,alphas=inner,inverse=inverse.astype(np.int32))
        write(timing_path,{s:timing[s] for s in config["splits"] if s in timing})
        print(json.dumps({"split":split,"hybrid_fresh":results[split]["policies"][0]["hybrid_metrics"],
                          "alpha_fresh":chosen[0]["alpha"],"unique_pairs":len(keys),"seconds":elapsed}),flush=True)
    cpu_models = {}
    for role,s in sources.items():
        saved = torch.load(ROOT/s["run"]/"train-mps/best.pt",map_location="cpu",weights_only=True)
        m = SASRec(saved["items"],**model_args(saved["config"]));m.load_state_dict(saved["weights"])
        cpu_models[role] = m.eval().requires_grad_(False)
    p,t = np.unravel_index(np.random.default_rng(config["seed"]).choice(selected_all.size,256,replace=False),selected_all.shape)
    q = base["target_request"][t]
    alpha = np.array([selection["policies"][j]["alpha"] for j in p])
    oracle = naive_ranks(cpu_models,configs,base["queries"],base["refs"][p,q],q,base["targets"][t],alpha,"cpu",config["epsilon"])
    if metrics(oracle) != metrics(selected_all[p,t]):
        raise ValueError("CPU mixture metrics differ")
    checks["cpu_mixture"] = {"pairs":len(t),"same_metrics":True,"exact_rank_matches":int((oracle==selected_all[p,t]).sum()),
                             "maximum_rank_difference":int(np.abs(oracle-selected_all[p,t]).max())}
    if checksum(args.output/"selection.json") != selection_hash:
        raise ValueError("Selection changed after PV")
    if {role:model_digest(m) for role,m in models.items()} != states_before:
        raise ValueError("Frozen model changed")
    for p,h in {**source_hashes,**reference_hashes}.items():
        if checksum(ROOT/p) != h:
            raise ValueError(f"Source/reference changed: {p}")
    for p,h in hashes.items():
        if checksum(data/p) != h:
            raise ValueError(f"Data changed: {p}")
    for s in sources.values():
        if checksum(ROOT/s["run"]/"train-mps/best.pt") != s["checkpoint_sha256"]:
            raise ValueError("Checkpoint changed")
    if any(checksum(ROOT/p) != h for p,h in reused_grids.items()):
        raise ValueError("A reused mixture grid changed")
    np.savez(args.output/"ranks.npz",hybrid=selected_all,recent=recent)
    write(args.output/"results.json",{"config":config,"sources":sources,"results":results,"verification":checks,
          "selection_sha256":selection_hash,"timing":timing,"source_sha256":source_hashes,
          "reference_sha256":reference_hashes,"data_files_sha256":hashes,"frozen_state_sha256":states_before,
          "reused_mixture_files_sha256":reused_grids,
          "device":args.device,"torch":torch.__version__,"numpy":np.__version__,"python":platform.python_version(),
          "macos":platform.mac_ver()[0],"cost_scope":"Quality experiment; hybrid serving cost not measured",**memory(args.device)})


if __name__ == "__main__":
    main()
