#!/usr/bin/env python3
"""Prepare strict, fixed audio tables for a train-only Yambda vocabulary."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import duckdb
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source_file(source, download=False):
    raw = ROOT / "data/raw/yambda-audio/embeddings.parquet"
    if download:
        from huggingface_hub import hf_hub_download
        hf_hub_download(repo_id=source["repo"], repo_type="dataset", revision=source["revision"],
                        filename=source["path"], local_dir=raw.parent, token=False,
                        cache_dir=ROOT / ".cache/huggingface")
    if not raw.exists() or raw.stat().st_size != source["bytes"] or digest(raw) != source["sha256"]:
        raise ValueError("Missing or mismatched audio source; run with --download")
    return raw


def item_index(raw, source):
    path = raw.parent / "items.parquet"
    if path.exists():
        cached = json.loads((raw.parent / "manifest.json").read_text())
        if cached["source"] != source or cached["sha256"] != digest(path):
            raise ValueError("Cached audio item index uses a different source or checksum")
    else:
        with duckdb.connect(config={"threads": 4, "memory_limit": "2GB"}) as db:
            db.sql("SELECT DISTINCT item_id FROM read_parquet(?) ORDER BY item_id",
                   params=[str(raw)]).write_parquet(str(path), compression="zstd")
    result = {"source": source, "sha256": digest(path), "full_source_sha256_verified": True}
    (raw.parent / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    return path


def prepare(data, output, source, raw):
    if output.exists():
        raise FileExistsError(f"Choose a new output directory: {output}")
    started = time.perf_counter()
    output.mkdir(parents=True)
    with duckdb.connect(config={"threads": 4,
                               "memory_limit": "4GB", "temp_directory": str(output / ".spill")}) as db:
        db.read_parquet(str(data / "items.parquet")).create_view("items")
        print(f"Extracting fixed audio vectors for {data}", flush=True)
        values = db.execute(f"""
            SELECT i.item_idx, e.{source['column']} AS vector FROM items i
            LEFT JOIN read_parquet(?) e USING (item_id) ORDER BY i.item_idx
        """, [str(raw)]).fetchnumpy()
    count = len(values["item_idx"])
    if not np.array_equal(values["item_idx"], np.arange(2, count + 2)):
        raise ValueError("Audio join duplicated or reordered the item vocabulary")
    matrix = np.zeros((count + 2, source["dimension"]), dtype=np.float32)
    for index, vector in zip(values["item_idx"], values["vector"]):
        if vector is None or vector is np.ma.masked:
            raise ValueError("Missing audio: rebuild all splits with --audio-items first")
        vector = np.asarray(vector, dtype=np.float32)
        if vector.shape != (source["dimension"],) or not np.isfinite(vector).all():
            raise ValueError(f"Invalid audio vector for item index {index}")
        norm = np.linalg.norm(vector)
        if norm <= 0:
            raise ValueError(f"Zero audio vector for item index {index}")
        matrix[index] = vector / norm
    np.save(output / "vectors.npy", matrix)
    result = {"source": source, "full_source_sha256_verified": True,
              "items_sha256": digest(data / "items.parquet"), "items": count,
              "covered_items": count, "missing_items": 0,
              "dimension": source["dimension"], "missing_policy": "not allowed; PAD/unused UNK zero",
              "vectors_sha256": digest(output / "vectors.npy"),
              "elapsed_seconds": time.perf_counter() - started}
    (output / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--index-only", action="store_true")
    parser.add_argument("--data", type=Path, default=ROOT / "data/processed/yambda50m-pilot-audio")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    source = json.loads((ROOT / "configs/data/yambda-audio.json").read_text())
    raw = source_file(source, args.download)
    if args.index_only:
        print(item_index(raw, source))
    else:
        print(json.dumps(prepare(args.data, args.output or args.data / "audio", source, raw), indent=2))
