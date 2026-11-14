#!/usr/bin/env python3
"""Compare maintenance effects relative to each frozen encoder's own fresh quality."""

import json
import shutil
from pathlib import Path

import numpy as np

from blend_sasrec import checksum
from compare_sasrec import paired_intervals
from plot_sasrec_replay import plt,save
from report_controller_models import independent_ci,read,terms
from report_student_maintenance import cost_matched_mixture,csv_file
from run_sasrec import ROOT,write


def main():
    roots={"student43":ROOT/"data/runs/controller-mlp42",**{name:ROOT/f"data/runs/controller-mlp-{name}" for name in ["student42","ce1k"]}}
    out=ROOT/"reports/experiments/policy-transfer";out.mkdir(parents=True,exist_ok=True)
    result={"models":{},"comparisons":[],"no_test_events_read":True};table=[];policy_rows=[];resources=[];checked=0
    for name,root in roots.items():
        run=read(root/"results.json");base=ROOT/run["config"]["policy_replay"];stream=dict(np.load(base/"replay.npz"));current=dict(np.load(root/"replay.npz"))
        quality=read(root/"policy-results.json");training=read(root/"model-training.json");prepared=read(base/"results.json")
        if name=="student43":assert checksum(root/"results.json")==checksum(ROOT/"reports/experiments/controller-models/mlp42-results.json")
        else:
            for metadata in [run,prepared]:
                for path,h in metadata["source_sha256"].items():assert checksum(ROOT/path)==h,path
        assert checksum(root/"selection.json")==run["selection_sha256"] and checksum(root/"replay.npz")==run["replay_sha256"]
        assert training["stop_reason"]=="validation_patience" and training["best_epoch"]==min(training["epochs"],key=lambda row:row["validation_mse"])["epoch"]
        assert run["fit"]["fit_cutoff"]==23667205 and run["fit"]["cutoff"]==23840005
        for field in ["targets","target_request","target_event_ids"]:np.testing.assert_array_equal(stream[field],current[field])
        refs=current["refs"]
        assert np.all(stream["users"][refs]==stream["users"][None]) and np.all(stream["split_idx"][refs]==stream["split_idx"][None])
        assert np.all(stream["last_input_timestamp"][refs]<stream["timestamp"][None])
        errors=[abs(row["measured_minus_predicted_percent"]) for b in run["mixed_workload_checks"].values() for row in b["cases"].values()]
        result["models"][name]={"fit":run["fit"],"training":training,"controller_seconds":run["total_seconds"],
            "preparation_seconds":prepared.get("execution",prepared.get("offline_execution")),"cpu_oracle":run["cpu_oracle"],"mixed_max_error_percent":max(errors),
            "rank_scope":run.get("rank_scope","Full ranks"),"budget_selection":run["selection"]["budgets"],"splits":{}}
        for idx,split in enumerate(["controller_train","policy_valid"],1):
            take=stream["split_idx"][stream["target_request"]]==idx;users=stream["users"][stream["target_request"]][take]
            simple=quality["results"][split]["policies"];fresh=simple[0];always=fresh["cost"]["estimated_ms_per_request"]
            result["models"][name]["splits"][split]={"requests":fresh["requests"],"targets":int(take.sum()),"positive_users":len(np.unique(users)),"fresh":fresh["metrics"],
                "never_cost_fraction":simple[1]["cost"]["estimated_ms_per_request"]/always}
            for row in simple:policy_rows.append({"model":name,"split":split,"policy":row["name"],"refresh_fraction":row["refresh_fraction"],
                "relative_cost":row["cost"]["estimated_ms_per_request"]/always,**row["metrics"]})
            for pick in run["selection"]["budgets"]:
                if pick["controller"] is None:continue
                k=list(current["names"]).index(pick["controller"]);rank=current["ranks"][k,take]
                row=next(p for p in run["results"][split] if p["name"]==pick["controller"])
                j=next(j for j,p in enumerate(quality["policies"]) if p["name"]==pick["matched_simple"])
                references={"simple":stream["ranks"][j,take],"fresh":stream["ranks"][0,take]}
                effects={key:paired_intervals(rank,value,users,42) for key,value in references.items()}
                for reference,block in effects.items():
                    for metric,effect in block.items():
                        np.testing.assert_allclose(independent_ci(terms(rank)[metric]-terms(references[reference])[metric],users),effect["cluster_bootstrap_95ci"],rtol=0,atol=1e-14);checked+=1
                comparison={"model":name,"split":split,"ct_budget":pick["relative_budget"],"controller":row,"simple":simple[j],"fresh":fresh["metrics"],"paired":effects}
                mix=cost_matched_mixture(run["selection"]["tuning_simple_rows"],pick["tuning_controller"]["cost"]["estimated_ms_per_request"])
                if mix:
                    left=next(i for i,p in enumerate(quality["policies"]) if p["name"]==mix["left"]);right=next(i for i,p in enumerate(quality["policies"]) if p["name"]==mix["right"]);p=mix["right_probability"]
                    expected={metric:(1-p)*value+p*terms(stream["ranks"][right,take])[metric] for metric,value in terms(stream["ranks"][left,take]).items()}
                    comparison["expected_mixture"]={**mix,"paired":{metric:{"difference":float((value-expected[metric]).mean()),"cluster_bootstrap_95ci":independent_ci(value-expected[metric],users)} for metric,value in terms(rank).items()}}
                result["comparisons"].append(comparison)
                flat={"model":name,"split":split,"ct_budget":pick["relative_budget"],"controller":pick["controller"],"simple":pick["matched_simple"],
                    "relative_cost":row["cost"]["estimated_ms_per_request"]/always,"simple_relative_cost":simple[j]["cost"]["estimated_ms_per_request"]/always,
                    "refresh_fraction":row["refresh_fraction"],**row["metrics"]}
                for metric,effect in effects["simple"].items():flat.update({f"retained_{metric}":row["metrics"][metric]/fresh["metrics"][metric],
                    f"delta_{metric}":effect["difference"],f"ci_low_{metric}":effect["cluster_bootstrap_95ci"][0],f"ci_high_{metric}":effect["cluster_bootstrap_95ci"][1]})
                table.append(flat)
        if name!="student43":
            direct=read(root/"direct-serving.json")
            for path,h in direct["source_sha256"].items():assert checksum(ROOT/path)==h,path
            result["models"][name]["direct_serving_seconds"]=direct["total_seconds"]
            for split,block in direct["results"].items():
                vector={row["name"]:row for row in block["vector"]}
                for regime,rows in block.items():
                    always=next(row["cost_ms"] for row in rows if row["name"]=="always")
                    for row in rows:
                        pair=direct["measurements"][split]["cases"][f"vector/{row['name']}"];sample=direct["measurements"][split]["cases"][f"{regime}/{row['name']}"]
                        for key in ["sample_requests","sample_refreshes","unique_sample_requests"]:assert sample[key]==pair[key]
                        np.testing.assert_allclose(row["cost_ms"],np.mean(sample["repeat_mean_ms"]),rtol=0,atol=1e-14)
                        assert abs(row["refresh_fraction"]-row["sample_refresh_fraction"])<=.5/sample["sample_requests"]+1e-12
                        resources.append({"model":name,"split":split,"regime":regime,"policy":row["name"],"refresh_fraction":row["refresh_fraction"],
                            "cost_ms":row["cost_ms"],"relative_cost":row["cost_ms"]/always,"cost_over_vector_same_policy":row["cost_ms"]/vector[row["name"]]["cost_ms"],**row["metrics"]})
            shutil.copyfile(root/"direct-serving.json",out/f"{name}-direct-serving.json")
            for filename in ["results.json","model.json","model-training.json","selection.json"]:shutil.copyfile(root/filename,out/f"{name}-{filename}")
            write(out/f"{name}-reference.json",{k:prepared[k] for k in ["config","source_sha256","used_data_files_sha256","data_manifest_sha256","checkpoint_sha256","rank_scope","cpu_oracle","execution","memory","top100_cache_sha256","replay_sha256"]})
    csv_file(out/"controller-comparison.csv",table);csv_file(out/"policies.csv",policy_rows);csv_file(out/"direct-resources.csv",resources)
    result["verification"]={"independent_paired_intervals":checked,"source_data_checkpoint_hashes":True,"same_ct_8_2_4":True,
        "same_events_for_each_paired_comparison":True,"causal_refs_and_frozen_encoder":True,"top100_scope_explicit":True,"no_test_events_read":True}
    result["source_sha256"]={str(path.relative_to(ROOT)):checksum(path) for path in [Path(__file__).resolve(),*[root/filename for root in roots.values() for filename in ["results.json","replay.npz","selection.json"]]]}
    write(out/"comparison.json",result)
    fig,axes=plt.subplots(1,2,figsize=(10,4.5))
    for ax,metric in zip(axes,["ndcg@10","recall@100"]):
        for name,color in zip(roots,["#777777","#168469","#c87220"]):
            rows=[row for row in table if row["model"]==name and row["split"]=="policy_valid"]
            ax.plot([row["relative_cost"] for row in rows],[row[f"retained_{metric}"] for row in rows],"o--",color=color,label=name)
        ax.set(xlabel="Coupled vector cost / own always",ylabel=f"Retained {metric} / own fresh",ylim=(0,1.05));ax.grid(alpha=.2)
    fig.legend(*axes[0].get_legend_handles_labels(),loc="upper center",ncol=3,frameon=False)
    fig.text(.5,.005,"PV, CT-selected controllers. Different populations/catalogs; raw cross-model quality is not compared. Test sealed.",ha="center",fontsize=8)
    fig.tight_layout(rect=(0,.035,1,.9));save(fig,out,"quality-retention")
    print(json.dumps({"verification":result["verification"],"pv":[row for row in table if row["split"]=="policy_valid"]},indent=2))


if __name__=="__main__":main()
