"""Canonical CPU top-k payloads and static-catalog cache lookup."""

import numpy as np


def canonical_topk(scores,k=100):
    """Descending score, ascending item index, including ties at the k boundary."""
    values,indices=scores.topk(min(k+1,scores.shape[1]),dim=1)
    values=values.cpu().numpy();indices=indices.cpu().numpy();k=min(k,scores.shape[1])
    order=np.lexsort((indices[:,:k],-values[:,:k]),axis=1)
    result=np.take_along_axis(indices[:,:k],order,axis=1)
    if values.shape[1]>k:
        for row in np.flatnonzero(values[:,k-1]==values[:,k]):
            full=scores[row].cpu().numpy();eligible=np.flatnonzero(full>=values[row,k-1])
            result[row]=eligible[np.lexsort((eligible,-full[eligible]))[:k]]
    return result


def list_ranks(top100,refs,targets):
    output=np.empty(refs.size,dtype=np.int32);states=refs.ravel()
    labels=np.broadcast_to(targets,refs.shape).ravel()-2
    for start in range(0,len(states),8192):
        take=slice(start,start+8192);matches=top100[states[take]]==labels[take,None]
        output[take]=np.where(matches.any(1),matches.argmax(1)+1,1000000)
    return output.reshape(refs.shape)


def previous_states(stream,refs):
    before=np.full(len(refs),-1,dtype=np.int32)
    for split in np.unique(stream["split_idx"]):
        for user in np.unique(stream["users"][stream["split_idx"]==split]):
            ids=np.flatnonzero((stream["split_idx"]==split)&(stream["users"]==user))
            before[ids[1:]]=refs[ids[:-1]]
    return before


def refresh_decision(stream,features,predict,case,i):
    previous=int(case["before"][i])
    if previous<0:return True
    if "threshold" in case:return predict(features.row(i,previous))>case["threshold"]
    policy=case["policy"]
    return (policy["family"]=="always" or
        (policy["events"]>0 and stream["history"][i]-stream["history"][previous]>=policy["events"]) or
        (policy["seconds"]>0 and stream["timestamp"][i]-stream["timestamp"][previous]>=policy["seconds"]))
