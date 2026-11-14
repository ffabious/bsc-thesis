"""Small weighted CPU regressor with chronological early stopping."""

import copy
import time

import numpy as np
import torch
from torch import nn

from refresh_controller import FEATURES,predict_matrix


@torch.inference_mode(False)
@torch.enable_grad()
def fit_mlp(x,y,weights,vx,vy,vweights,config):
    torch.manual_seed(config["model_seed"]);rng=np.random.default_rng(config["model_seed"])
    mean=np.average(x,axis=0,weights=weights);scale=np.sqrt(np.average((x-mean)**2,axis=0,weights=weights)).clip(1e-8)
    tx=torch.from_numpy(((x-mean)/scale).astype(np.float32));ty=torch.from_numpy(y.astype(np.float32));tw=torch.from_numpy(weights.astype(np.float32))
    valid=torch.from_numpy(((vx-mean)/scale).astype(np.float32));labels=torch.from_numpy(vy.astype(np.float32));vw=torch.from_numpy(vweights.astype(np.float32))
    model=nn.Sequential(nn.Linear(x.shape[1],32),nn.ReLU(),nn.Linear(32,32),nn.ReLU(),nn.Linear(32,1))
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    best=float("inf");waiting=0;history=[];started=time.perf_counter()
    def evaluate():
        with torch.no_grad():
            total=0.
            for start in range(0,len(valid),8192):
                take=slice(start,start+8192);total+=float((((model(valid[take])[:,0]-labels[take])**2)*vw[take]).sum())
            return total/float(vw.sum())
    for epoch in range(1,config["max_epochs"]+1):
        order=rng.permutation(len(x));total=0.;den=0.
        for start in range(0,len(order),4096):
            take=order[start:start+4096];optimizer.zero_grad(set_to_none=True)
            loss=(((model(tx[take])[:,0]-ty[take])**2)*tw[take]).sum()/tw[take].sum()
            loss.backward();optimizer.step();total+=float(loss.detach())*float(tw[take].sum());den+=float(tw[take].sum())
        score=evaluate();row={"epoch":epoch,"fit_mse":total/den,"validation_mse":score};history.append(row)
        print(row,flush=True)
        if score<best:best=score;selected=copy.deepcopy(model.state_dict());best_epoch=epoch;waiting=0
        else:waiting+=1
        if waiting>=config["patience"]:break
    model.load_state_dict(selected)
    layers=[{"weight":layer.weight.detach().numpy().tolist(),"bias":layer.bias.detach().numpy().tolist()} for layer in model if isinstance(layer,nn.Linear)]
    result={"kind":"mlp","features":config.get("feature_names",FEATURES),"feature_mean":mean.tolist(),"feature_scale":scale.tolist(),"layers":layers,
        "model_seed":config["model_seed"],"best_epoch":best_epoch,"epochs":len(history),"parameters":sum(p.numel() for p in model.parameters())}
    check=np.linspace(0,len(vx)-1,min(len(vx),512),dtype=int)
    with torch.no_grad():expected=model(valid[check])[:,0].numpy()
    delta=float(np.abs(expected-predict_matrix(result,vx[check])).max());assert delta<1e-5
    curve={"epochs":history,"best_epoch":best_epoch,"best_validation_mse":best,"elapsed_seconds":time.perf_counter()-started,
        "stop_reason":"validation_patience" if waiting>=config["patience"] else "epoch_limit","torch_numpy_max_prediction_delta":delta}
    return result,curve
