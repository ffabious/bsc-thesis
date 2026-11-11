#!/usr/bin/env python3
"""Export exploratory history/popularity and user-weight checks from saved ranks."""

import csv
import json
from pathlib import Path

import numpy as np

from audit_cold_distillation import ROOT,NAMES,STRATA,ndcg
from blend_sasrec import checksum
from compare_sasrec import paired_intervals
from run_sasrec import write


def main():
    root=ROOT/"reports/experiments/sasrec-256-distillation-audit"
    local=ROOT/"data/diagnostics/sasrec-256-distillation-audit"
    source=local/"events.npz";context=local/"context-check.npz";v=dict(np.load(source))
    d=ndcg(v["student"])-ndcg(v["control"]);rows=[];slices={}
    pops=[v["pop_rank"]<=1000,(v["pop_rank"]>1000)&(v["pop_rank"]<=10000),v["pop_rank"]>10000]
    for idx,split in enumerate(NAMES):
        base=v["split_idx"]==idx;n=int(base.sum())
        for h,history in enumerate(STRATA):
            for p,pop in enumerate(["top_1000","1001_to_10000","over_10000"]):
                m=base&(v["stratum"]==h)&pops[p]
                rows.append({"split":split,"history":history,"popularity":pop,"targets":int(m.sum()),
                    "ndcg_delta":float(d[m].mean()),"ndcg_contribution":float(d[m].sum()/n),
                    **{f"{name}_ndcg":float(ndcg(v[name][m]).mean()) for name in ["control","student","teacher"]}})
        if idx in [0,2]:
            groups={name:v["stratum"]==h for h,name in enumerate(STRATA)}
            groups["last_item_over_10000"]=(v["stratum"]==0)&pops[2]
            slices[split]={name:paired_intervals(v["student"][base&m],v["control"][base&m],v["users"][base&m],42)
                for name,m in groups.items()}
    common=np.intersect1d(v["users"][v["split_idx"]==0],v["users"][v["split_idx"]==2]);per=[]
    for idx in [0,2]:
        values=[]
        for user in common:
            m=(v["split_idx"]==idx)&(v["users"]==user);values.append([m.sum(),d[m].mean()])
        per.append(np.array(values))
    fixed={name:{"common_users":len(common),"macro_delta":float(per[i][:,1].mean()),
        "own_event_weights":float(np.average(per[i][:,1],weights=per[i][:,0])),
        "other_period_user_weights":float(np.average(per[i][:,1],weights=per[1-i][:,0]))}
        for i,name in enumerate([NAMES[0],NAMES[2]])}
    b=v["split_idx"]==2;last=b&(v["stratum"]==0);outside=b&(v["stratum"]==3)
    counts={name:{model:{"top1":int((v[model][m]==1).sum()),"top100":int((v[model][m]<=100).sum())}
        for model in ["control","student","teacher"]} for name,m in [("all",b),("last_item",last),("outside_window",outside)]}
    short=dict(np.load(context));checks={}
    for idx,split in enumerate(NAMES):
        m=short["split_idx"]==idx
        checks[split]={name:paired_intervals(short[name+"_short"][m],short[name][m],short["users"][m],42)
            for name in ["control","student"]}
    result={"source_events_sha256":checksum(source),"source_context_sha256":checksum(context),
        "source_sha256":checksum(Path(__file__).resolve()),"cross_segments":rows,
        "paired_stratum_intervals":slices,"common_user_weight_checks":fixed,"pv_counts":counts,
        "context_short_minus_long":checks,"exploratory":True,
        "ci_scope":"2000 whole-user resamples per stratum; conditional, pointwise, not adjusted for slice exploration",
        "definitions":{"popularity":"train positive-listen count, descending; ties by item_idx",
            "history":"last item / second-last excluding last / positions 3..50 excluding last two / absent from window"},
        "no_training":True,"no_test_events_read":True}
    write(root/"additional-slices.json",result)
    with (root/"history-popularity.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n");writer.writeheader();writer.writerows(rows)
    print("Saved history/popularity intersections and paired context/user checks")


if __name__=="__main__":main()
