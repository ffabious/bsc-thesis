"""Small CPU model of immediate refresh benefit; all features are strict-past."""

import math

import numpy as np

FEATURES=["log_new_events","log_age_seconds","log_activity_1h","log_activity_24h",
          "log_last_gap","log_history","events_x_activity","age_x_activity"]


class HistoryFeatures:
    def __init__(self,stream,prefix_users,prefix_times):
        self.stream=stream;self.times={};self.static=np.empty((len(stream["users"]),4))
        for user in np.unique(stream["users"]):
            take=np.flatnonzero(stream["users"]==user)
            times=np.sort(np.r_[prefix_times[prefix_users==user],
                np.repeat(stream["timestamp"][take],stream["group_size"][take])])
            self.times[int(user)]=times;t=stream["timestamp"][take]
            end=np.searchsorted(times,t,side="left")
            self.static[take]=np.log1p(np.column_stack([end-np.searchsorted(times,t-3600,side="left"),
                end-np.searchsorted(times,t-86400,side="left"),t-stream["last_input_timestamp"][take],stream["history"][take]]))

    def row(self,i,previous):
        # Scalar history lookups and feature extraction also run inside the cost benchmark.
        s=self.stream;t=int(s["timestamp"][i]);times=self.times[int(s["users"][i])]
        end,hour,day=np.searchsorted(times,[t,t-3600,t-86400],side="left")
        activity_hour,activity_day=math.log1p(end-hour),math.log1p(end-day)
        gap=math.log1p(t-int(s["last_input_timestamp"][i]));history=math.log1p(int(s["history"][i]))
        events=math.log1p(int(s["history"][i]-s["history"][previous]))
        age=math.log1p(t-int(s["timestamp"][previous]))
        return np.array([events,age,activity_hour,activity_day,gap,history,events*activity_day,age*activity_day])

    def matrix(self,requests,previous):
        s=self.stream;static=self.static[requests]
        events=np.log1p(s["history"][requests]-s["history"][previous])
        age=np.log1p(s["timestamp"][requests]-s["timestamp"][previous])
        return np.column_stack([events,age,static,events*static[:,1],age*static[:,1]])


def training_examples(stream,ranks,refs,fit_requests):
    """One weight per request, spread across its distinct observed stale states."""
    ids=np.flatnonzero(fit_requests);lookup=np.full(len(fit_requests),-1,dtype=np.int32);lookup[ids]=np.arange(len(ids))
    targets=fit_requests[stream["target_request"]];target_requests=lookup[stream["target_request"][targets]]
    ndcg=np.where(ranks[:,targets]<=10,1/np.log2(ranks[:,targets]+1),0)
    rewards=np.array([np.bincount(target_requests,weights=row,minlength=len(ids)) for row in ndcg])
    states=refs[:,ids];stale=states!=ids[None]
    keys=(ids[None].astype(np.int64)*len(fit_requests)+states)[stale]
    unique,first=np.unique(keys,return_index=True)
    requests,previous=unique//len(fit_requests),unique%len(fit_requests)
    labels=(rewards[[0]]-rewards)[stale][first]
    counts=np.bincount(requests,minlength=len(fit_requests));weights=1/counts[requests]
    return {"requests":requests,"previous":previous,"gain":labels,"weights":weights}


def fit_ridge(x,y,weights,regularization):
    total=weights.sum();mean=np.average(x,axis=0,weights=weights)
    scale=np.sqrt(np.average((x-mean)**2,axis=0,weights=weights)).clip(1e-8)
    z=(x-mean)/scale;center=float(np.average(y,weights=weights))
    beta=np.linalg.solve(z.T@(weights[:,None]*z)/total+regularization*np.eye(x.shape[1]),
                         z.T@(weights*(y-center))/total)
    coefficients=beta/scale;intercept=center-float(mean@coefficients)
    prediction=x@coefficients+intercept
    return {"features":FEATURES,"coefficients":coefficients.tolist(),"intercept":intercept,
        "regularization":regularization,"feature_mean":mean.tolist(),"feature_scale":scale.tolist(),
        "fit_weighted_mse":float(np.average((prediction-y)**2,weights=weights)),
        "constant_weighted_mse":float(np.average((y-center)**2,weights=weights)),"mean_gain":center}


def rollout(stream,features,model,threshold,start=0,end=None):
    end=len(stream["users"]) if end is None else end
    refs=np.empty(end-start,dtype=np.int32);refresh=np.empty(end-start,dtype=bool)
    before=np.empty(end-start,dtype=np.int32);cached=np.full(int(stream["users"].max())+1,-1,dtype=np.int32)
    coefficients=np.array(model["coefficients"]);current_split=-1
    for pos,i in enumerate(range(start,end)):
        split=int(stream["split_idx"][i]);user=int(stream["users"][i])
        if split!=current_split:cached.fill(-1);current_split=split
        previous=int(cached[user]);before[pos]=previous
        update=previous<0 or float(features.row(i,previous)@coefficients+model["intercept"])>threshold
        if update:cached[user]=i
        refs[pos],refresh[pos]=cached[user],update
    return refs,refresh,before
