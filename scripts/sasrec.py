"""SASRec blocks, strictly historical examples and full-catalog ranking."""

import json
import math
import duckdb
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class Block(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(dim, eps=1e-8)
        self.norm2 = nn.LayerNorm(dim, eps=1e-8)
        self.attention = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Dropout(dropout),
                                nn.Linear(dim, dim), nn.Dropout(dropout))

    def forward(self, x, allowed):
        query = self.norm1(x)
        attended, _ = self.attention(query, x, x, need_weights=False,
                                    attn_mask=~allowed.repeat_interleave(self.heads, dim=0))
        x = query + attended
        normalized = self.norm2(x)
        return normalized + self.ff(normalized)


class SASRec(nn.Module):
    def __init__(self, items, length=50, dim=64, heads=2, blocks=2, dropout=0.2,
                 item_dim=None, freeze_items=False):
        super().__init__()
        self.length = length
        item_dim = item_dim or dim
        self.item = nn.Embedding(items + 2, item_dim, padding_idx=0)
        self.item.weight.requires_grad_(not freeze_items)
        self.input_projection = nn.Linear(item_dim, dim, bias=False) if item_dim != dim else nn.Identity()
        self.output_projection = nn.Linear(dim, item_dim, bias=False) if item_dim != dim else nn.Identity()
        self.position = nn.Embedding(length, dim)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([Block(dim, heads, dropout) for _ in range(blocks)])
        self.norm = nn.LayerNorm(dim, eps=1e-8)
        self.register_buffer("causal", torch.ones(length, length, dtype=torch.bool).tril())
        nn.init.normal_(self.item.weight, std=0.02)
        nn.init.normal_(self.position.weight, std=0.02)
        with torch.no_grad():
            self.item.weight[0].zero_()

    def encode(self, sequence):
        valid = sequence != 0
        allowed = self.causal[None] & valid[:, None, :]
        # Padding queries see themselves, avoiding all-masked softmax rows and NaN gradients.
        allowed = allowed | (~valid[:, :, None] & torch.eye(
            self.length, device=sequence.device, dtype=torch.bool)[None])
        x = self.input_projection(self.item(sequence)) * math.sqrt(self.position.embedding_dim)
        x = self.dropout(x + self.position.weight[None]) * valid[:, :, None]
        for block in self.blocks:
            x = block(x, allowed) * valid[:, :, None]
        return self.norm(x) * valid[:, :, None]

    def forward(self, sequence):
        return self.output_projection(self.encode(sequence)[:, -1])


def examples(data, split, length):
    """One prefix per eligible target; all tied targets share a strict-past prefix."""
    manifest = json.loads((data / "manifest.json").read_text())
    names = [p["split"] for p in manifest["periods"]]
    files = [str(data / "events" / f"{name}.parquet") for name in names[:names.index(split) + 1]]
    with duckdb.connect(config={"threads": 4, "memory_limit": "2GB"}) as db:
        db.read_parquet(files, filename=True).create_view("events")
        values = db.execute("""
            SELECT user_idx, item_idx, history_size, is_target AND filename = ? AS target
            FROM events ORDER BY user_idx, timestamp, event_id
        """, [files[-1]]).fetchnumpy()
    users = values["user_idx"]
    starts = np.flatnonzero(np.r_[True, users[1:] != users[:-1]])
    contexts, targets, owners = [], [], []
    for start, end in zip(starts, np.r_[starts[1:], len(users)]):
        chosen = np.flatnonzero(values["target"][start:end]) + start
        padded = np.pad(values["item_idx"][start:end], (length, 0))
        windows = np.lib.stride_tricks.sliding_window_view(padded, length)
        contexts.append(windows[values["history_size"][chosen]])
        targets.append(values["item_idx"][chosen])
        owners.append(users[chosen])
    return (np.concatenate(contexts).astype(np.int32),
            np.concatenate(targets).astype(np.int64), np.concatenate(owners))


def negatives(targets, count, items, rng):
    # Uniform known items except the current positive; no future/history-based filtering.
    sampled = rng.integers(2, items + 1, size=(len(targets), count), dtype=np.int64)
    return sampled + (sampled >= targets[:, None])


def batch(x, y, indices, device):
    return (torch.as_tensor(x[indices], dtype=torch.long, device=device),
            torch.as_tensor(y[indices], dtype=torch.long, device=device))


def loss(model, x, y, negative):
    state = model(x)
    positive = (state * model.item(y)).sum(-1)
    other = (state[:, None] * model.item(negative)).sum(-1)
    return F.softplus(-positive).mean() + F.softplus(other).mean()


def sync(device):
    if device == "mps":
        torch.mps.synchronize()


@torch.inference_mode()
def ranks(model, x, y, device, batch_size):
    model.eval()
    output = []
    ids = torch.arange(2, model.item.num_embeddings, device=device)
    for start in range(0, len(y), batch_size):
        sequence, target = batch(x, y, slice(start, start + batch_size), device)
        scores = model(sequence) @ model.item.weight[2:].T
        actual = scores.gather(1, (target - 2)[:, None])
        # Deterministic tie break: smaller item index first. PAD/UNK never enter the catalog.
        rank = 1 + (scores > actual).sum(1, dtype=torch.int32)
        rank += ((scores == actual) & (ids[None] < target[:, None])).sum(1, dtype=torch.int32)
        output.append(rank.cpu().numpy())
    return np.concatenate(output)


def metrics(rank):
    return {"ndcg@10": float(np.where(rank <= 10, 1 / np.log2(rank + 1), 0).mean()),
            "recall@100": float((rank <= 100).mean())}


def most_popular(data, targets):
    with duckdb.connect() as db:
        values = db.execute("""
            SELECT item_idx, count(*) AS n FROM read_parquet(?) WHERE is_positive
            GROUP BY item_idx ORDER BY n DESC, item_idx
        """, [str(data / "events/train.parquet")]).fetchall()
    count = json.loads((data / "manifest.json").read_text())["candidate_items"]
    ordered = [row[0] for row in values]
    seen = set(ordered)
    ordered.extend(item for item in range(2, count + 2) if item not in seen)
    rank = np.zeros(count + 2, dtype=np.int32)
    rank[ordered] = np.arange(1, count + 1)
    return rank[targets]
