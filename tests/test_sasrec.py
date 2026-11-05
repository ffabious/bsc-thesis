import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import duckdb
import numpy as np
import torch

from scripts.sasrec import SASRec, examples, loss, negatives, ranks


class SASRecTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)

    def test_prefixes_exclude_current_timestamp_and_cross_split_boundaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "events").mkdir()
            (root / "manifest.json").write_text(json.dumps({"periods": [
                {"split": "train"}, {"split": "model_valid"}]}))
            with duckdb.connect() as db:
                db.execute("""CREATE TABLE events(user_idx INTEGER, item_idx INTEGER,
                    timestamp INTEGER, event_id INTEGER, history_size INTEGER, is_target BOOLEAN)""")
                db.executemany("INSERT INTO events VALUES (?, ?, ?, ?, ?, ?)", [
                    (0, 2, 0, 0, 0, False), (0, 3, 5, 1, 1, True),
                    (0, 4, 5, 2, 1, True), (0, 5, 10, 3, 3, True),
                    (1, 6, 0, 4, 0, False), (1, 2, 5, 5, 1, True),
                    (0, 1, 15, 6, 4, False), (0, 3, 20, 7, 5, True)])
                for split, condition in [("train", "timestamp < 15"), ("model_valid", "timestamp >= 15")]:
                    db.sql(f"SELECT * FROM events WHERE {condition}").write_parquet(str(root / "events" / f"{split}.parquet"))
            x, y, _ = examples(root, "train", 3)
            np.testing.assert_array_equal(x, [[0, 0, 2], [0, 0, 2], [2, 3, 4], [0, 0, 6]])
            np.testing.assert_array_equal(y, [3, 4, 5, 2])
            x, y, _ = examples(root, "model_valid", 3)
            np.testing.assert_array_equal(x, [[4, 5, 1]])
            np.testing.assert_array_equal(y, [3])

    def test_attention_is_causal_and_padding_gradients_are_finite(self):
        model = SASRec(8, length=4, dim=8, heads=2, blocks=2, dropout=0).eval()
        first = torch.tensor([[0, 2, 3, 4], [0, 0, 0, 0]])
        second = first.clone()
        second[0, -1] = 7
        torch.testing.assert_close(model.encode(first)[:, :-1], model.encode(second)[:, :-1])
        self.assertTrue(torch.equal(model.encode(first)[1], torch.zeros(4, 8)))
        value = loss(model, first, torch.tensor([5, 6]), torch.tensor([[6, 7], [2, 3]]))
        value.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        self.assertTrue(torch.equal(model.item.weight.grad[0], torch.zeros(8)))

    def test_negative_sampling_excludes_only_the_current_positive(self):
        target = np.array([2, 3, 4, 5])
        sampled = negatives(target, 1000, 4, np.random.default_rng(42))
        self.assertTrue(np.all(sampled != target[:, None]))
        for y, values in zip(target, sampled):
            self.assertEqual(set(values), set(range(2, 6)) - {y})

    def test_full_catalog_rank_excludes_special_tokens_and_breaks_ties(self):
        model = SASRec(4, length=3, dim=2, heads=1, blocks=1, dropout=0)
        with torch.no_grad():
            model.item.weight.copy_(torch.tensor([[100., 0], [100., 0], [1., 0],
                                                  [1., 0], [0.5, 0], [-1., 0]]))
        with patch.object(model, "forward", return_value=torch.tensor([[1., 0], [1., 0]])):
            rank = ranks(model, np.ones((2, 3), dtype=np.int32), np.array([3, 5]), "cpu", 2)
        np.testing.assert_array_equal(rank, [2, 4])

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS unavailable")
    def test_cpu_mps_forward_and_backward_agree(self):
        cpu = SASRec(12, length=5, dim=8, heads=2, blocks=2, dropout=0).eval()
        mps = copy.deepcopy(cpu).to("mps")
        x = torch.tensor([[0, 0, 2, 3, 4], [2, 5, 6, 7, 8]])
        y = torch.tensor([5, 9])
        neg = torch.tensor([[6, 7], [10, 11]])
        torch.testing.assert_close(cpu(x), mps(x.to("mps")).cpu(), atol=1e-5, rtol=1e-4)
        left, right = loss(cpu, x, y, neg), loss(mps, x.to("mps"), y.to("mps"), neg.to("mps"))
        left.backward()
        right.backward()
        torch.testing.assert_close(left, right.cpu(), atol=1e-5, rtol=1e-4)
        for a, b in zip(cpu.parameters(), mps.parameters()):
            torch.testing.assert_close(a.grad, b.grad.cpu(), atol=1e-5, rtol=1e-3)


if __name__ == "__main__":
    unittest.main()
