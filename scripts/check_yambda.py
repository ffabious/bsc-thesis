#!/usr/bin/env python3
"""Independently check exported Yambda files, vocabulary and strict-past history."""

import argparse
import hashlib
import json
from pathlib import Path

import duckdb


def check(root):
    manifest = json.loads((root / "manifest.json").read_text())
    for file in manifest["files"]:
        path = root / file["path"]
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != file["sha256"] or path.stat().st_size != file["bytes"]:
            raise ValueError(f"Output checksum/size mismatch: {path}")
    with duckdb.connect(config={"threads": 4, "memory_limit": "4GB",
                               "temp_directory": str(root / ".check-spill")}) as db:
        db.read_parquet([str(root / "events" / f"{p['split']}.parquet")
                         for p in manifest["periods"]],
                        filename=True, file_row_number=True).create_view("events")
        db.read_parquet(str(root / "items.parquet")).create_view("items")
        queries = {
            "unsorted_rows": """SELECT count(*) FROM (
                SELECT timestamp < lag(timestamp) OVER
                    (PARTITION BY filename ORDER BY file_row_number) AS bad FROM events
                ) WHERE bad""",
            "incorrect_history_groups": """SELECT count(*) FROM (
                SELECT *, coalesce(sum(n) OVER (PARTITION BY uid ORDER BY timestamp
                    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING), 0) AS expected
                FROM (SELECT uid, timestamp, count(*) AS n, min(history_size) AS lo,
                    max(history_size) AS hi FROM events GROUP BY uid, timestamp)
                ) WHERE lo != expected OR hi != expected""",
            "duplicate_event_ids": "SELECT count(*) - count(DISTINCT event_id) FROM events",
            "unmapped_known_items": """SELECT count(*) FROM events e
                LEFT JOIN items i USING (item_id) WHERE
                (e.item_idx = 1 AND i.item_idx IS NOT NULL) OR
                (e.item_idx > 1 AND (i.item_idx IS NULL OR e.item_idx != i.item_idx))""",
        }
        result = {name: db.execute(sql).fetchall()[0][0] for name, sql in queries.items()}
        if any(result.values()):
            raise ValueError(f"Invalid exported data: {result}")
        for name, sql in {
            "repeated_user_item_events": "SELECT count(*) - count(DISTINCT (uid, item_id)) FROM events",
            "organic_share": "SELECT avg(is_organic::DOUBLE) FROM events",
        }.items():
            result[name] = db.execute(sql).fetchall()[0][0]
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = json.dumps(check(args.data), indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result)
    print(result, end="")
