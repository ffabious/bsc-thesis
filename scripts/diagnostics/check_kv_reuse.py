#!/usr/bin/env python3
"""Test exact projected-KV reuse conditions on the actual frozen SASRec blocks."""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from blend_sasrec import checksum
from distill_sasrec import load_student
from run_sasrec import ROOT,state_digest,write


class KVCache:
    def __init__(self,model,blocks):
        self.model=model;self.blocks=blocks;self.keys=[None]*blocks;self.values=[None]*blocks

    def append(self,token,position,retain=None):
        model=self.model;dim=model.position.embedding_dim
        x=model.input_projection(model.item(torch.tensor([[token]])))*math.sqrt(dim)+model.position.weight[None,position:position+1]
        for i,block in enumerate(model.blocks[:self.blocks]):
            query=block.norm1(x);attention=block.attention;heads=attention.num_heads;head_dim=dim//heads
            weights=attention.in_proj_weight.chunk(3);biases=attention.in_proj_bias.chunk(3)
            q,k,v=[F.linear(value,weight,bias).reshape(1,1,heads,head_dim).transpose(1,2)
                for value,weight,bias in zip([query,x,x],weights,biases)]
            if self.keys[i] is not None:
                old_k,old_v=self.keys[i],self.values[i]
                if retain is not None:old_k,old_v=old_k[:,:,-retain:],old_v[:,:,-retain:]
                k,v=torch.cat([old_k,k],dim=2),torch.cat([old_v,v],dim=2)
            self.keys[i],self.values[i]=k,v
            attended=F.scaled_dot_product_attention(q,k,v,dropout_p=0.).transpose(1,2).reshape(1,1,dim)
            x=query+attention.out_proj(attended);normalized=block.norm2(x);x=normalized+block.ff(normalized)
        return model.output_projection(model.norm(x))[:,0]


def full(model,tokens,positions,blocks):
    x=model.input_projection(model.item(torch.tensor([tokens])))*math.sqrt(model.position.embedding_dim)
    x=x+model.position(torch.tensor([positions]));n=len(tokens);allowed=torch.ones(1,n,n,dtype=torch.bool).tril()
    for block in model.blocks[:blocks]:x=block(x,allowed)
    return model.output_projection(model.norm(x))[:,-1]


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--config",type=Path,required=True)
    args=parser.parse_args();config=json.loads(args.config.read_text());torch.set_num_threads(1)
    checkpoint=ROOT/config["checkpoint"];assert checksum(checkpoint)==config["checkpoint_sha256"]
    output=ROOT/config["output"]
    if output.exists():parser.error("Choose a new feasibility output")
    model,_=load_student(checkpoint.parent,"cpu");state=state_digest(model);layers=len(model.blocks);length=model.length
    paths=[args.config.resolve(),Path(__file__).resolve(),checkpoint,ROOT/"scripts/sasrec.py",ROOT/"scripts/distill_sasrec.py"]
    hashes={str(path.relative_to(ROOT)):checksum(path) for path in paths};rows=[]
    for seed in config["seeds"]:
        tokens=np.random.default_rng(seed).integers(2,model.item.num_embeddings,length+1).tolist()
        def compare(name,actual,expected,equivalent):
            delta=float(torch.max(torch.abs(actual-expected)));matches=torch.allclose(actual,expected,atol=config["atol"],rtol=config["rtol"])
            assert matches==equivalent,(name,delta)
            rows.append({"seed":seed,"case":name,"expected_equivalent":equivalent,"max_absolute_delta":delta,"within_tolerance":matches})
        # Fixed positions, no eviction: old keys/values remain valid under causal attention.
        cache=KVCache(model,layers)
        for n in range(1,9):compare(f"fixed_append_{n}",cache.append(tokens[n-1],n-1),full(model,tokens[:n],list(range(n)),layers),True)
        current=torch.tensor([tokens[:5]])
        compare("reference_current_right_alignment",full(model,tokens[:5],list(range(length-5,length)),layers),model(current),True)
        cache=KVCache(model,layers)
        for j in range(4):cache.append(tokens[j],length-4+j)
        compare("right_aligned_append_reuses_old_positions",cache.append(tokens[4],length-1),model(current),False)
        cache=KVCache(model,layers)
        for j in range(length):cache.append(tokens[j],j)
        window=tokens[1:length+1];expected=model(torch.tensor([window]))
        compare("sliding_window_reuses_shifted_states",cache.append(tokens[length],length-1,retain=length-1),expected,False)
        rebuilt=KVCache(model,layers)
        for j,token in enumerate(window):last=rebuilt.append(token,j)
        compare("rebuild_after_invalidation",last,expected,True)
        # Keeping positions fixed isolates removed context in higher-layer cached states.
        for count in [1,layers]:
            cache=KVCache(model,count)
            for j in range(8):cache.append(tokens[j],j)
            compare(f"fixed_positions_eviction_layers_{count}",cache.append(tokens[8],8,retain=7),
                full(model,tokens[1:9],list(range(1,9)),count),count==1)
    assert state_digest(model)==state
    for path,h in hashes.items():assert checksum(ROOT/path)==h,path
    output.mkdir(parents=True);write(output/"results.json",{"config":config,"cases":rows,"source_sha256":hashes,"frozen_state_unchanged":True,
        "precision":"CPU FP32, dropout disabled, frozen trained weights, synthetic token sequences",
        "projected_kv_bytes_per_user_full_window":2*layers*length*model.position.embedding_dim*4,
        "no_dataset_events_read":True,"scope":"Correctness feasibility only; fixed-position controls change the serving contract and are not a measured speedup of the current right-aligned sliding model"})
    print(json.dumps({"cases":len(rows),"expected_matches":sum(row["within_tolerance"] for row in rows),"expected_mismatches":sum(not row["within_tolerance"] for row in rows)}),flush=True)


if __name__=="__main__":main()
