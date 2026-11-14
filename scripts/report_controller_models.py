#!/usr/bin/env python3
"""Compare matched Ridge/MLP refresh controllers without consulting test."""

import json
import shutil
from pathlib import Path

import numpy as np

from blend_sasrec import checksum
from compare_sasrec import paired_intervals
from plot_sasrec_replay import plt,save
from refresh_controller import HistoryFeatures,predict_matrix,training_examples
from report_student_maintenance import cost_matched_mixture,csv_file
from run_sasrec import ROOT,write


def read(path):return json.loads(path.read_text())


def terms(rank):
    return {"ndcg@10":np.where(rank<=10,1/np.log2(rank+1),0),"recall@100":(rank<=100).astype(float)}


def independent_ci(delta,users):
    """User-count bootstrap, independent of the rank-comparison implementation."""
    ids,inverse=np.unique(users,return_inverse=True);counts=np.bincount(inverse)
    draws=np.random.default_rng(42).integers(0,len(ids),(2000,len(ids)))
    multiplicities=np.array([np.bincount(row,minlength=len(ids)) for row in draws])
    sums=np.bincount(inverse,weights=delta)
    return np.quantile((multiplicities@sums)/(multiplicities@counts),[.025,.975]).tolist()


def main():
    names=["ridge8","mlp42","mlp43"];roots={name:ROOT/f"data/runs/controller-{name}" for name in names}
    runs={name:read(root/"results.json") for name,root in roots.items()}
    arrays={name:dict(np.load(root/"replay.npz")) for name,root in roots.items()}
    base=ROOT/runs["ridge8"]["config"]["policy_replay"];stream=dict(np.load(base/"replay.npz"))
    simple=read(base/"results.json");out=ROOT/"reports/experiments/controller-models";out.mkdir(parents=True,exist_ok=True)
    result={"models":{},"comparisons":[],"no_test_events_read":True};table=[];checked=0
    for name,run in runs.items():
        for path,h in run["source_sha256"].items():assert checksum(ROOT/path)==h,path
        assert checksum(roots[name]/"selection.json")==run["selection_sha256"]
        assert checksum(roots[name]/"replay.npz")==run["replay_sha256"]
        assert run["policy_valid_not_used_for_training_or_selection"] and run["no_test_events_read"]
        for key in ["targets","target_request","target_event_ids"]:assert np.array_equal(arrays[name][key],stream[key]),key
        a=arrays[name];assert np.all(stream["users"][a["refs"]]==stream["users"][None])
        assert np.all(stream["split_idx"][a["refs"]]==stream["split_idx"][None])
        assert np.all(stream["last_input_timestamp"][a["refs"]]<stream["timestamp"][None])
        errors=[abs(row["measured_minus_predicted_percent"]) for block in run["mixed_workload_checks"].values() for row in block["cases"].values()]
        training=read(roots[name]/"model-training.json") if name!="ridge8" else None
        if training:
            assert training["best_epoch"]==min(training["epochs"],key=lambda row:row["validation_mse"])["epoch"]
            assert training["stop_reason"]=="validation_patience"
        result["models"][name]={"fit":run["fit"],"training":training,"cpu_oracle":run["cpu_oracle"],
            "mixed_max_error_percent":max(errors),"total_seconds":run["total_seconds"]}
        for filename in ["results.json","selection.json","model.json","policy-selection.json"]:
            shutil.copyfile(roots[name]/filename,out/f"{name}-{filename}")
        if training:shutil.copyfile(roots[name]/"model-training.json",out/f"{name}-training.json")
    # The same chronological early-validation labels are scored by all three learners.
    import duckdb
    data=ROOT/simple["config"]["data"];ct=stream["split_idx"]==1
    fit_end=runs["ridge8"]["fit"]["fit_cutoff"];early_end=runs["ridge8"]["fit"]["cutoff"]
    ct_start=next(p["start"] for p in read(data/"manifest.json")["periods"] if p["split"]=="controller_train")
    with duckdb.connect(config={"threads":4,"memory_limit":"512MB"}) as db:
        prefix=db.execute("SELECT user_idx,timestamp FROM read_parquet(?) WHERE timestamp>=? AND timestamp<?",
            [str(data/"events/model_valid.parquet"),ct_start-86400,ct_start]).fetchnumpy()
    features=HistoryFeatures(stream,prefix["user_idx"],prefix["timestamp"])
    early=ct&(stream["timestamp"]>=fit_end)&(stream["timestamp"]<early_end)
    rows=training_examples(stream,stream["ranks"],stream["refs"],early);x=features.matrix(rows["requests"],rows["previous"])
    fit=dict(np.load(roots["ridge8"]/"training-examples.npz"));constant=np.average(fit["gain"],weights=fit["weights"])
    result["early_validation"]={"contexts":len(x),"weighted_requests":float(rows["weights"].sum()),
        "constant_mse":float(np.average((rows["gain"]-constant)**2,weights=rows["weights"]))}
    for name,run in runs.items():
        other=dict(np.load(roots[name]/"training-examples.npz"))
        assert all(np.array_equal(fit[key],other[key]) for key in fit)
        mse=float(np.average((predict_matrix(run["model"],x)-rows["gain"])**2,weights=rows["weights"]))
        result["early_validation"][name]=mse
        if name!="ridge8":assert abs(mse-result["models"][name]["training"]["best_validation_mse"])<1e-7
    for idx,split in enumerate(["controller_train","policy_valid"],1):
        take=stream["split_idx"][stream["target_request"]]==idx;users=stream["users"][stream["target_request"]][take]
        for name,run in runs.items():
            priced=read(roots[name]/"policy-results.json")["results"][split]["policies"];always=priced[0]["cost"]["estimated_ms_per_request"]
            for selected in run["selection"]["budgets"]:
                if selected["controller"] is None:continue
                controller=next(p for p in run["results"][split] if p["name"]==selected["controller"])
                k=list(arrays[name]["names"]).index(controller["name"]);rank=arrays[name]["ranks"][k,take]
                j=next(j for j,p in enumerate(simple["policies"]) if p["name"]==selected["matched_simple"])
                effect=paired_intervals(rank,stream["ranks"][j,take],users,42)
                comparisons={"simple":effect};current=terms(rank)
                references={"simple":terms(stream["ranks"][j,take])}
                if name!="ridge8":
                    pick=next(p for p in runs["ridge8"]["selection"]["budgets"] if p["relative_budget"]==selected["relative_budget"])
                    r=list(arrays["ridge8"]["names"]).index(pick["controller"]);reference=arrays["ridge8"]["ranks"][r,take]
                    comparisons["ridge8"]=paired_intervals(rank,reference,users,42);references["ridge8"]=terms(reference)
                for reference,effects in comparisons.items():
                    for metric,e in effects.items():
                        np.testing.assert_allclose(independent_ci(current[metric]-references[reference][metric],users),e["cluster_bootstrap_95ci"],rtol=0,atol=1e-14);checked+=1
                row={"split":split,"model":name,"budget":selected["relative_budget"],"controller":controller,
                    "simple":priced[j],"paired":comparisons}
                # Expected randomized simple-policy envelope, chosen only on CT tuning.
                mix=cost_matched_mixture(run["selection"]["tuning_simple_rows"],selected["tuning_controller"]["cost"]["estimated_ms_per_request"])
                if mix:
                    left=next(j for j,p in enumerate(simple["policies"]) if p["name"]==mix["left"])
                    right=next(j for j,p in enumerate(simple["policies"]) if p["name"]==mix["right"]);p=mix["right_probability"]
                    expected={m:(1-p)*v+p*terms(stream["ranks"][right,take])[m] for m,v in terms(stream["ranks"][left,take]).items()}
                    row["mixture"]={**mix,"scope":"Independent per-user randomization in expectation; assignment variance/cost excluded",
                        "paired":{m:{"difference":float(np.mean(v-expected[m])),"cluster_bootstrap_95ci":independent_ci(v-expected[m],users)} for m,v in current.items()}}
                result["comparisons"].append(row)
                flat={"split":split,"model":name,"ct_budget":selected["relative_budget"],"policy":controller["name"],"simple":selected["matched_simple"],
                    "cost_ms":controller["cost"]["estimated_ms_per_request"],"relative_cost":controller["cost"]["estimated_ms_per_request"]/always,
                    "simple_relative_cost":priced[j]["cost"]["estimated_ms_per_request"]/always,"refresh_fraction":controller["refresh_fraction"],**controller["metrics"]}
                for metric,e in effect.items():flat.update({f"delta_{metric}":e["difference"],f"ci_low_{metric}":e["cluster_bootstrap_95ci"][0],f"ci_high_{metric}":e["cluster_bootstrap_95ci"][1]})
                table.append(flat)
    result["verification"]={"independent_paired_intervals":checked,"same_fit_examples":True,"source_and_selection_hashes":True,
        "causal_user_split_time_checks":True,"fit_early_tuning_days":[8,2,4],"model_valid_labels_and_test_unused":True}
    paths=[Path(__file__).resolve(),base/"results.json",base/"replay.npz"]+[root/filename for root in roots.values() for filename in ["results.json","selection.json","replay.npz"]]
    result["source_sha256"]={str(path.relative_to(ROOT)):checksum(path) for path in paths}
    write(out/"comparison.json",result);csv_file(out/"comparison.csv",table)
    fig,axes=plt.subplots(1,2,figsize=(11,4.5));split="policy_valid"
    for ax,metric in zip(axes,["ndcg@10","recall@100"]):
        policies=simple["results"][split]["policies"];always=policies[0]["cost"]["estimated_ms_per_request"]
        ax.scatter([p["cost"]["estimated_ms_per_request"]/always for p in policies],[p["metrics"][metric] for p in policies],s=18,c="#aaa",label="106 simple rules (H1 costs)")
        for name,color in zip(names,["#777777","#168469","#c87220"]):
            rows=[r for r in table if r["model"]==name and r["split"]==split]
            ax.plot([r["relative_cost"] for r in rows],[r[metric] for r in rows],"o--",color=color,label=name)
        ax.set(xlabel="Full cost / own-run always",ylabel=metric,title=f"policy_valid — {metric}",ylim=(0,None));ax.grid(alpha=.2)
    fig.legend(*axes[0].get_legend_handles_labels(),loc="upper center",ncol=4,frameon=False)
    fig.text(.5,.005,"CT-selected thresholds; each run includes its own coupled cost calibration. Test sealed.",ha="center",fontsize=8)
    fig.tight_layout(rect=(0,.035,1,.9));save(fig,out,"frontier")
    print(json.dumps({"early_validation":result["early_validation"],"models":result["models"],"verification":result["verification"],"pv":[r for r in table if r["split"]=="policy_valid"]},indent=2))


if __name__=="__main__":main()
