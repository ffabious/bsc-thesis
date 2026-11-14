"""Integrity checks shared by the frozen final-cohort preparation and evaluation."""

import json
import subprocess

from blend_sasrec import checksum
from run_sasrec import ROOT


def verified_protocol():
    path=ROOT/"reports/experiments/final-user-holdout/protocol.json";relative=str(path.relative_to(ROOT))
    committed=subprocess.run(["git","show",f"HEAD:{relative}"],cwd=ROOT,check=True,capture_output=True).stdout
    if committed!=path.read_bytes():raise ValueError("Final protocol must match its committed version before reading holdout")
    protocol=json.loads(committed);root=ROOT/protocol["config"]["output"]
    assert checksum(root/"protocol.json")==checksum(path)
    assert checksum(root/"selected-users.csv")==protocol["cohort"]["selected_users_sha256"]
    for source,h in protocol["source_sha256"].items():assert checksum(ROOT/source)==h,source
    commit=subprocess.run(["git","log","-1","--format=%H","--",relative],cwd=ROOT,check=True,capture_output=True,text=True).stdout.strip()
    return protocol,commit
