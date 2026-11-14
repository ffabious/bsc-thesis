#!/usr/bin/env python3
"""Retokenize only the committed final users using the frozen CE1k vocabulary."""

import json
import shutil
import time
from pathlib import Path

import duckdb

from blend_sasrec import checksum
from final_holdout import verified_protocol
from run_sasrec import ROOT,write


def main():
    protocol,commit=verified_protocol();config=protocol["config"];root=ROOT/config["output"];output=root/"dataset"
    if output.exists():raise FileExistsError("Final dataset already exists")
    pool=ROOT/config["pool_data"];vocabulary=ROOT/config["model_data"];source=json.loads((pool/"manifest.json").read_text())
    expected={row["path"]:row["sha256"] for row in source["files"]};started=time.perf_counter()
    output.mkdir();(output/"events").mkdir();coverage={};files=[]
    with duckdb.connect(config={"threads":config["threads"],"memory_limit":"2GB"}) as db:
        db.read_csv(str(root/"selected-users.csv")).create_view("selected")
        db.read_parquet(str(vocabulary/"items.parquet")).create_view("vocabulary")
        db.execute("COPY (SELECT uid::UINTEGER uid,user_idx::UINTEGER user_idx,train_events::UBIGINT train_events FROM selected ORDER BY user_idx) TO ? (FORMAT PARQUET, COMPRESSION ZSTD)",[str(output/"users.parquet")])
        assert db.execute("SELECT count(DISTINCT uid) FROM selected").fetchone()[0]==256
        for period in protocol["periods"]:
            split=period["split"];relative=f"events/{split}.parquet";path=pool/relative;target=output/relative
            assert checksum(path)==expected[relative],relative
            # Filter users before vocabulary enrichment; retain every event, including UNK.
            db.execute("""COPY (SELECT e.* EXCLUDE(user_idx,item_idx,is_target),
                u.user_idx::UINTEGER user_idx,coalesce(v.item_idx,1)::UINTEGER item_idx,
                e.is_positive AND coalesce(v.item_idx,1)>1 AND e.history_size>=1 AS is_target
                FROM read_parquet(?) e JOIN selected u USING(uid)
                LEFT JOIN vocabulary v USING(item_id) ORDER BY u.user_idx,e.timestamp,e.event_id)
                TO ? (FORMAT PARQUET, COMPRESSION ZSTD)""",[str(path),str(target)])
            row=db.execute("""SELECT count(*),count(DISTINCT uid),count(*) FILTER(WHERE item_idx=1),
                count(*) FILTER(WHERE is_positive AND history_size>=1),
                count(*) FILTER(WHERE is_positive AND history_size>=1 AND item_idx=1),
                count(*) FILTER(WHERE is_target),count(*) FILTER(WHERE is_positive!=(played_ratio_pct>50))
                FROM read_parquet(?)""",[str(target)]).fetchone()
            assert row[-1]==0 and checksum(path)==expected[relative]
            coverage[split]=dict(zip(["listens","users","unknown_history_items","positive_with_history","unknown_positive_with_history","targets"],row[:-1]))
            coverage[split]["known_positive_fraction"]=row[5]/row[3] if row[3] else None
            print(json.dumps({"split":split,**coverage[split]}),flush=True)
        assert db.execute("SELECT count(*) FROM read_parquet(?) e ANTI JOIN selected u USING(uid)",[str(output/"events/train.parquet")]).fetchone()[0]==0
    shutil.copyfile(vocabulary/"items.parquet",output/"items.parquet")
    assert checksum(output/"items.parquet")==protocol["vocabulary_sha256"]
    for path in sorted(output.rglob("*.parquet")):
        files.append({"path":str(path.relative_to(output)),"bytes":path.stat().st_size,"sha256":checksum(path)})
    manifest={"selected_users":256,"candidate_items":protocol["candidate_items"],"periods":protocol["periods"],"files":files,
        "coverage":coverage,"freeze_commit":commit,"protocol_sha256":checksum(root/"protocol.json"),"source_pool":config["pool_data"],
        "vocabulary_reference":config["model_data"],"vocabulary_unchanged":True,"weights_or_thresholds_updated":False,
        "source_sha256":{str(path.relative_to(ROOT)):checksum(path) for path in [Path(__file__).resolve(),ROOT/"scripts/final_holdout.py",pool/"manifest.json",root/"protocol.json",root/"selected-users.csv"]},
        "elapsed_seconds":time.perf_counter()-started}
    write(output/"manifest.json",manifest);write(ROOT/config["report"]/"data-preparation.json",manifest)


if __name__=="__main__":main()
