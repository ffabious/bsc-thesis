#!/usr/bin/env python3
"""Publish static-catalog cache equivalence and directly measured frontiers."""

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from blend_sasrec import checksum
from plot_sasrec_replay import plt,save
from report_controller_models import read
from report_student_maintenance import csv_file
from run_sasrec import ROOT,write


def main():
    root=ROOT/"data/runs/static-list-cache";out=ROOT/"reports/experiments/static-list-cache";out.mkdir(parents=True,exist_ok=True)
    equiv=read(root/"equivalence.json");direct=read(root/"benchmark-direct.json");probe=read(root/"benchmark.json")
    for metadata in [equiv,direct]:
        for path,h in metadata["source_sha256"].items():
            actual=root/"direct-benchmark-driver.py" if metadata is direct and path=="scripts/benchmark_list_cache.py" else ROOT/path
            assert checksum(actual)==h,path
    for name,h in equiv["artifacts_sha256"].items():assert checksum(root/name)==h,name
    assert checksum(root/"initial-benchmark-driver.py")==probe["source_sha256"]["scripts/benchmark_list_cache.py"]
    protected=[root/name for name in ["ranks.npz","benchmark-direct.json"]];before={path:checksum(path) for path in protected}
    guard=subprocess.run([sys.executable,str(ROOT/"scripts/benchmark_list_cache.py"),"--config",str(ROOT/"configs/experiments/static-list-cache.json"),"--direct"],cwd=ROOT,capture_output=True,text=True)
    assert guard.returncode==2 and "already exists" in guard.stderr
    assert all(checksum(path)==h for path,h in before.items())
    assert all(row["old"]==row["cached_top100"] for row in equiv["comparisons"])
    assert direct["selection"]["policy_valid"]==direct["selection"]["controller_train"]
    costs=[];selected=[];fraction_errors=[]
    for split,block in direct["results"].items():
        vector={row["name"]:row for row in block["vector"]["policies"]}
        for regime,summary in block.items():
            always=summary["policies"][0]["cost"]["estimated_ms_per_request"]
            for row in summary["policies"]:
                measured=direct["measurements"][split]["cases"][f"{regime}/{row['name']}"]
                np.testing.assert_allclose(row["cost"]["estimated_ms_per_request"],np.mean(measured["repeat_mean_ms"]),rtol=0,atol=1e-14)
                paired=direct["measurements"][split]["cases"][f"vector/{row['name']}"]
                for key in ["sample_requests","sample_refreshes","unique_sample_requests"]:assert measured[key]==paired[key]
                error=abs(row["refresh_fraction"]-row["sample_refresh_fraction"]);fraction_errors.append(error)
                assert error<=.5/measured["sample_requests"]+1e-12
                costs.append({"split":split,"regime":regime,"policy":row["name"],"refresh_fraction":row["refresh_fraction"],
                    "sample_refresh_fraction":row["sample_refresh_fraction"],"cost_ms":row["cost"]["estimated_ms_per_request"],
                    "cost_over_own_always":row["cost"]["estimated_ms_per_request"]/always,
                    "cost_over_vector_same_policy":row["cost"]["estimated_ms_per_request"]/vector[row["name"]]["cost"]["estimated_ms_per_request"],**row["metrics"]})
            for pick in direct["selection"]["controller_train"][regime]:
                row=next(p for p in costs if p["split"]==split and p["regime"]==regime and p["policy"]==pick["policy"])
                selected.append({"ct_budget":pick["relative_budget"],**row})
    rows=[{"split":split,"path":name,"measured_ms":row["median_mean_ms"],"predicted_ms":row["predicted_ms"],"error_percent":row["error_percent"]}
        for split,block in probe["mixed"].items() for name,row in block["cases"].items()]
    csv_file(out/"diagnostic-mixed-errors.csv",rows);csv_file(out/"policies.csv",costs);csv_file(out/"selected.csv",selected)
    verification={"same_quality_rows":len(equiv["comparisons"]),"canonical_rank_checks":equiv["canonical_same_score_rank_checks"],
        "paired_workload_cases":sum(len(block["cases"]) for block in direct["measurements"].values()),
        "max_sample_refresh_fraction_error":max(fraction_errors),"all_direct_cost_means_checked":True,
        "frozen_ct_selection_on_pv":True,"historical_probe_source_snapshot_verified":True,
        "existing_output_guard_preserves_rank_and_result_files":True,"restored_rank_file_matches_original_sha256":True,
        "no_test_events_read":True,"source_sha256":{str(path.relative_to(ROOT)):checksum(path) for path in [Path(__file__).resolve(),root/"equivalence.json",root/"benchmark-direct.json",root/"benchmark.json",root/"initial-benchmark-driver.py"]}}
    write(out/"verification.json",verification)
    for filename in ["equivalence.json","benchmark-direct.json","benchmark.json","initial-benchmark-driver.py","direct-benchmark-driver.py"]:shutil.copyfile(root/filename,out/filename)
    fig,axes=plt.subplots(2,2,figsize=(11,7))
    for i,split in enumerate(["controller_train","policy_valid"]):
        denominator=direct["results"][split]["vector"]["policies"][0]["cost"]["estimated_ms_per_request"]
        for ax,metric in zip(axes[i],["ndcg@10","recall@100"]):
            for regime,color in [("vector","#888888"),("list","#168469")]:
                block=direct["results"][split][regime];rows=block["policies"]
                ax.scatter([p["cost"]["estimated_ms_per_request"]/denominator for p in rows],[p["metrics"][metric] for p in rows],s=16,c=color,alpha=.5,label=regime)
                frontier=sorted([p for p in rows if p["name"] in block["frontiers"][metric]],key=lambda p:p["cost"]["estimated_ms_per_request"])
                ax.plot([p["cost"]["estimated_ms_per_request"]/denominator for p in frontier],[p["metrics"][metric] for p in frontier],c=color,linewidth=1)
            ax.set(title=f"{split} — {metric}",xlabel="Direct full cost / vector always",ylabel="Quality",ylim=(0,None));ax.grid(alpha=.2)
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc="upper center",ncol=2,frameon=False)
    fig.text(.5,.005,"Static full catalog; identical quality, paired B=1 workloads, frozen H2 controller thresholds. Test sealed.",ha="center",fontsize=9)
    fig.tight_layout(rect=(0,.035,1,.95));save(fig,out,"frontier")
    print({"verification":verification,"pv":[row for row in selected if row["split"]=="policy_valid"]})


if __name__=="__main__":main()
