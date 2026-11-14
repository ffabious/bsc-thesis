"""Counterfactual held-cache rewards within one chronological training region."""

import numpy as np


def horizon_design(stream,rows,region,horizon,width):
    future=np.full((len(region),horizon),-1,dtype=np.int32)
    for user in np.unique(stream["users"][region]):
        ids=np.flatnonzero(region&(stream["users"]==user))
        assert np.all(np.diff(stream["timestamp"][ids])>0)
        for offset in range(min(horizon,len(ids))):
            future[ids[:len(ids)-offset],offset]=ids[offset:]
    assert region[rows["requests"]].all()
    counts=np.bincount(stream["target_request"],minlength=len(region))
    starts=np.cumsum(counts)-counts;order=np.argsort(stream["target_request"],kind="stable")
    owners=[];fresh=[];stale=[]
    for offset in range(horizon):
        next_request=future[rows["requests"],offset];active=np.flatnonzero(next_request>=0)
        size=counts[next_request[active]];owner=np.repeat(active,size)
        positions=np.arange(size.sum())-np.repeat(np.cumsum(size)-size,size)
        targets=order[np.repeat(starts[next_request[active]],size)+positions]
        assert region[stream["target_request"][targets]].all()
        owners.append(owner)
        fresh.append(rows["requests"][owner].astype(np.int64)*width+stream["targets"][targets])
        stale.append(rows["previous"][owner].astype(np.int64)*width+stream["targets"][targets])
    return {"owner":np.concatenate(owners),"fresh":np.concatenate(fresh),"stale":np.concatenate(stale),
        "requests_in_horizon":(future[rows["requests"]]>=0).sum(1)}


def horizon_gain(design,keys,ranks):
    def reward(wanted):
        positions=np.searchsorted(keys,wanted);assert np.array_equal(keys[positions],wanted)
        selected=ranks[positions]
        return np.where(selected<=10,1/np.log2(selected+1),0)
    delta=reward(design["fresh"])-reward(design["stale"])
    total=np.bincount(design["owner"],weights=delta,minlength=len(design["requests_in_horizon"]))
    return total/design["requests_in_horizon"]


def replace_gain(rows,saved,prefix):
    for key in ["requests","previous"]:
        assert np.array_equal(rows[key],saved[f"{prefix}_{key}"]),key
    gain=saved[f"{prefix}_gain"]
    assert gain.shape==rows["gain"].shape and np.isfinite(gain).all()
    return {**rows,"gain":gain}
