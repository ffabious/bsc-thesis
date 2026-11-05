#!/usr/bin/env python3
"""Download a pinned Yambda file and prepare a local chronological replay dataset."""

import argparse
import hashlib
import json
import platform
import shutil
import time
import urllib.request
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
DAY = 86400


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def download(source, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        url = (f"https://huggingface.co/datasets/{source['repo']}/resolve/"
               f"{source['revision']}/{source['path']}?download=true")
        partial = path.with_suffix(".part")
        print(f"Downloading {source['bytes'] / 1e6:.1f} MB", flush=True)
        with urllib.request.urlopen(url, timeout=120) as response, partial.open("wb") as out:
            shutil.copyfileobj(response, out, length=1024 * 1024)
        if partial.stat().st_size != source["bytes"] or sha256(partial) != source["sha256"]:
            raise ValueError(f"Download checksum/size mismatch: {partial}")
        partial.rename(path)
    if path.stat().st_size != source["bytes"] or sha256(path) != source["sha256"]:
        raise ValueError(f"Source checksum/size mismatch: {path}")


def rows(db, sql, params=()):
    result = db.execute(sql, params)
    names = [column[0] for column in result.description]
    return [dict(zip(names, row)) for row in result.fetchall()]


def periods(start, last, holdouts):
    end = last + 5
    train_end = end - sum(holdouts.values()) * DAY
    if train_end <= start or any(days <= 0 for days in holdouts.values()):
        raise ValueError("Holdout windows must be positive and leave a nonempty train period")
    result = [("train", start, train_end)]
    for name, days in holdouts.items():
        result.append((name, result[-1][2], result[-1][2] + days * DAY))
    return result


def prepare(source, output, config, user_limit=0, threads=4, memory="4GB"):
    if output.exists():
        raise FileExistsError(f"Choose a new output directory: {output}")
    started = time.perf_counter()
    output.mkdir(parents=True)
    db = duckdb.connect(config={"threads": threads, "memory_limit": memory,
                                "temp_directory": str(output / ".spill")})
    try:
        db.read_parquet(str(source), file_row_number=True).create_view("raw")
        schema = {row[0]: row[1] for row in db.execute("DESCRIBE raw").fetchall()}
        required = {"uid", "item_id", "timestamp", "event_type", "is_organic",
                    "played_ratio_pct", "track_length_seconds"}
        if not required <= schema.keys():
            raise ValueError(f"Missing columns: {required - schema.keys()}")
        audit = rows(db, """
            SELECT count(*) AS events, count(DISTINCT uid) AS users,
                   count(DISTINCT item_id) AS items, min(timestamp) AS min_timestamp,
                   max(timestamp) AS max_timestamp,
                   count(*) FILTER (WHERE uid IS NULL OR item_id IS NULL
                     OR timestamp IS NULL OR timestamp < 0 OR timestamp % 5 != 0
                     OR is_organic IS NULL OR is_organic NOT IN (0, 1)
                     OR event_type IS NULL OR event_type NOT IN
                        ('listen', 'like', 'dislike', 'unlike', 'undislike')
                     OR (event_type = 'listen' AND (played_ratio_pct IS NULL
                         OR played_ratio_pct < 0 OR track_length_seconds IS NULL))) AS invalid,
                   count(*) FILTER (WHERE event_type = 'listen'
                                     AND played_ratio_pct > 100) AS ratios_over_100
            FROM raw
        """)[0]
        if not audit["events"] or audit["invalid"]:
            raise ValueError(f"Empty source or invalid events: {audit}")
        audit["event_types"] = rows(db, "SELECT event_type, count(*) AS events FROM raw GROUP BY 1 ORDER BY 1")
        windows = periods(audit["min_timestamp"], audit["max_timestamp"], config["holdout_days"])
        train_end = windows[0][2]
        db.execute("""
            CREATE VIEW listens AS SELECT file_row_number AS event_id, uid, item_id,
                timestamp::BIGINT AS timestamp, is_organic, played_ratio_pct,
                track_length_seconds FROM raw WHERE event_type = 'listen'
        """)
        candidates = db.execute("""
            SELECT uid, count(*) AS train_events FROM listens WHERE timestamp < ?
            GROUP BY uid HAVING count(*) >= ? ORDER BY uid
        """, [train_end, config["min_train_events"]]).fetchall()
        if not candidates:
            raise ValueError("No users satisfy the train-only history filter")
        # Sampling cannot depend on validation/test activity or DuckDB's hash version.
        selected = sorted(candidates, key=lambda row: hashlib.sha256(
            f"{config['seed']}:{row[0]}".encode()).digest())
        selected = sorted(selected[:user_limit] if user_limit else selected)
        db.execute("CREATE TABLE users(uid UINTEGER, user_idx UINTEGER, train_events UBIGINT)")
        db.executemany("INSERT INTO users VALUES (?, ?, ?)",
                       [(uid, index, count) for index, (uid, count) in enumerate(selected)])
        db.execute("""
            CREATE TABLE items AS SELECT item_id,
                (row_number() OVER (ORDER BY item_id) + 1)::UINTEGER AS item_idx,
                count(*) AS train_events, min(timestamp) AS first_timestamp
            FROM listens JOIN users USING (uid) WHERE timestamp < ? GROUP BY item_id
        """, [train_end])
        db.execute("CREATE TABLE periods(split VARCHAR, start_ts BIGINT, end_ts BIGINT)")
        db.executemany("INSERT INTO periods VALUES (?, ?, ?)", windows)
        print(f"Preparing {len(selected)} users; train ends at relative second {train_end}", flush=True)
        # RANGE excludes the whole tied timestamp group, unlike a ROWS-based counter.
        db.execute("""
            CREATE TABLE prepared AS SELECT l.*, u.user_idx,
                coalesce(i.item_idx, 1)::UINTEGER AS item_idx, p.split,
                count(*) OVER (PARTITION BY l.uid ORDER BY l.timestamp
                    RANGE BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS history_size
            FROM listens l JOIN users u USING (uid) LEFT JOIN items i USING (item_id)
            JOIN periods p ON l.timestamp >= p.start_ts AND l.timestamp < p.end_ts
        """)
        db.execute(f"""
            CREATE VIEW events AS SELECT *, played_ratio_pct > {int(config['positive_ratio_gt'])}
                AS is_positive, played_ratio_pct > {int(config['positive_ratio_gt'])}
                AND item_idx > 1 AND history_size >= {int(config['min_history'])} AS is_target
            FROM prepared
        """)
        summary = rows(db, """
            SELECT split, count(*) AS events, count(DISTINCT uid) AS users,
                count(*) FILTER (WHERE is_positive) AS positives,
                count(*) FILTER (WHERE is_target) AS targets,
                count(*) FILTER (WHERE is_positive AND item_idx = 1) AS unknown_targets,
                count(*) FILTER (WHERE is_positive AND history_size = 0) AS no_history_positives,
                min(timestamp) AS min_timestamp, max(timestamp) AS max_timestamp
            FROM events GROUP BY split ORDER BY min(timestamp)
        """)
        if len(summary) != len(windows) or any(row["targets"] == 0 for row in summary):
            raise ValueError(f"Each period needs eligible targets: {summary}")
        ties = rows(db, """
            SELECT count(*) AS groups, coalesce(sum(n), 0)::BIGINT AS events
            FROM (SELECT count(*) AS n FROM events GROUP BY uid, timestamp HAVING count(*) > 1)
        """)[0]
        distributions = rows(db, """
            SELECT quantile_cont(n, [0.5, 0.9, 0.99]) AS user_history_quantiles
            FROM (SELECT uid, count(*) AS n FROM events GROUP BY uid)
        """)[0]
        expected = db.execute("SELECT count(*) FROM listens JOIN users USING (uid)").fetchall()[0][0]
        if sum(row["events"] for row in summary) != expected:
            raise ValueError("Split row counts do not cover all selected listening events")
        (output / "events").mkdir()
        for name, start, end in windows:
            target = output / "events" / f"{name}.parquet"
            db.sql("SELECT * EXCLUDE (split) FROM events WHERE split = ? ORDER BY timestamp, uid, event_id",
                   params=[name]).write_parquet(str(target), compression="zstd")
            db.read_parquet(str(target)).create_view("written", replace=True)
            invalid = db.execute("""
                SELECT count(*) - count(DISTINCT event_id) +
                    count(*) FILTER (WHERE timestamp < ? OR timestamp >= ?)
                FROM written
            """, [start, end]).fetchall()[0][0]
            if invalid:
                raise ValueError(f"Invalid exported split: {name}")
            print(f"Written {name}", flush=True)
        for table in ("items", "users"):
            db.sql(f"SELECT * FROM {table} ORDER BY {table[:-1]}_idx").write_parquet(
                str(output / f"{table}.parquet"), compression="zstd")
        manifest = {
            "source": config.get("source"), "source_sha256": sha256(source),
            "pipeline_sha256": sha256(Path(__file__)), "config": config,
            "python": platform.python_version(), "duckdb": duckdb.__version__,
            "machine": platform.machine(), "threads": threads, "memory_limit": memory,
            "requested_users": user_limit, "selected_users": len(selected),
            "train_eligible_users": len(candidates),
            "candidate_items": db.execute("SELECT count(*) FROM items").fetchall()[0][0],
            "periods": [{"split": n, "start": a, "end": b} for n, a, b in windows],
            "source_audit": audit, "splits": summary, "timestamp_ties": ties,
            **distributions,
            "files": [{"path": str(p.relative_to(output)), "bytes": p.stat().st_size,
                       "sha256": sha256(p)} for p in sorted(output.rglob("*.parquet"))],
            "elapsed_seconds": round(time.perf_counter() - started, 2),
        }
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        return manifest
    finally:
        db.close()
        shutil.rmtree(output / ".spill", ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/data/yambda50m.json")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--users", type=int, default=0, help="0 = all train-eligible users")
    parser.add_argument("--output", type=Path, default=ROOT / "data/processed/yambda50m")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--memory", default="4GB")
    args = parser.parse_args()
    if args.users < 0 or args.threads < 1:
        parser.error("--users must be nonnegative and --threads positive")
    config = json.loads(args.config.read_text())
    source = ROOT / "data/raw/yambda50m/multi_event.parquet"
    if args.download or args.download_only:
        download(config["source"], source)
    if args.download_only:
        return
    if not source.exists() or sha256(source) != config["source"]["sha256"]:
        parser.error("Missing or mismatched source; use --download to fetch the pinned file")
    result = prepare(source, args.output, config, args.users, args.threads, args.memory)
    print(f"Done: {result['selected_users']} users, {result['candidate_items']} items, "
          f"{result['elapsed_seconds']} s; {args.output}")


if __name__ == "__main__":
    main()
