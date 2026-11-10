import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

import duckdb
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from scripts.replay_sasrec import cutoffs, encode_queries, load_replay, rank_vectors, settings, summarize
from scripts.sasrec import SASRec, examples, ranks


class ReplayTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)
        self.splits = ["controller_train", "policy_valid"]
        self.conditions = settings({"event_lags": [0, 1, 2], "time_lags_seconds": [0, 5, 15]})

    def dataset(self, root, future_item=9):
        (root / "events").mkdir(exist_ok=True)
        periods = ["train", "model_valid", *self.splits, "test"]
        (root / "manifest.json").write_text(json.dumps({"periods": [{"split": s} for s in periods]}))
        # user, item, time, strict-past count, target, split
        rows = [(0,2,0,0,False,0), (0,3,10,1,False,0), (0,4,10,1,False,0),
                (0,5,20,3,False,1), (0,6,30,4,True,2), (0,7,30,4,True,2),
                (0,8,40,6,True,2), (0,future_item,50,7,True,3), (0,10,60,8,True,3),
                (1,11,0,0,False,0), (1,12,10,1,False,0), (1,13,30,2,True,2), (1,14,50,3,True,3)]
        with duckdb.connect() as db:
            db.execute("CREATE TABLE e(user_idx INTEGER,item_idx INTEGER,timestamp BIGINT,history_size BIGINT,is_target BOOLEAN,split INTEGER,event_id BIGINT)")
            db.executemany("INSERT INTO e VALUES (?,?,?,?,?,?,?)", [(*r, i) for i, r in enumerate(rows)])
            for i, name in enumerate(periods[:-1]):
                db.sql(f"SELECT * EXCLUDE(split) FROM e WHERE split={i}").write_parquet(str(root / "events" / f"{name}.parquet"))
        (root / "events/test.parquet").write_bytes(b"Replay must never read this poisoned test file")

    def test_cutoffs_keep_timestamp_groups_whole_and_time_uses_strict_past(self):
        times = np.array([0,10,10,20,30,30,40,50])
        cuts, age = cutoffs(times, np.array([4,6]), np.array([30,40]), self.conditions)
        np.testing.assert_array_equal(cuts, [[4,6], [3,4], [1,4], [4,6], [3,4]])
        np.testing.assert_array_equal(age, [[0,0], [10,10], [20,10], [5,5], [15,15]])
        with self.assertRaisesRegex(ValueError, "entire current"):
            cutoffs(times, np.array([5]), np.array([30]), self.conditions)

    def test_loader_uses_one_fixed_cohort_and_matches_existing_fresh_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.dataset(root)
            replay, coverage = load_replay(root, self.splits, 3, self.conditions)
            self.assertEqual(coverage["controller_train"]["all_targets"], 4)
            self.assertEqual(coverage["controller_train"]["included"], 3)
            self.assertEqual(coverage["controller_train"]["excluded_users"], 1)
            self.assertEqual(coverage["policy_valid"]["included"], 3)
            for i, split in enumerate(self.splits, 1):
                x, y, users = examples(root, split, 3)
                mask = replay["split_idx"] == i
                take = replay["source_row"][mask]
                np.testing.assert_array_equal(replay["queries"][replay["refs"][0,mask]], x[take])
                np.testing.assert_array_equal(replay["targets"][mask], y[take])
                np.testing.assert_array_equal(replay["users"][mask], users[take])
            # Both targets in the current tied group must receive the same historical snapshot.
            np.testing.assert_array_equal(replay["refs"][:,0], replay["refs"][:,1])
            self.assertTrue(np.all(replay["query_users"][replay["refs"]] == replay["users"]))
            with self.assertRaisesRegex(ValueError, "restricted"):
                load_replay(root, ["test"], 3, self.conditions)

    def test_future_events_do_not_change_controller_train_snapshots(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.dataset(root)
            first, _ = load_replay(root, self.splits, 3, self.conditions)
            self.dataset(root, future_item=15)
            second, _ = load_replay(root, self.splits, 3, self.conditions)
            for replay in (first, second):
                mask = replay["split_idx"] == 1
                replay["controller_inputs"] = replay["queries"][replay["refs"][:,mask]]
            np.testing.assert_array_equal(first["controller_inputs"], second["controller_inputs"])

    def test_cached_vectors_match_naive_recomputation_for_every_lag(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.dataset(root)
            replay, _ = load_replay(root, self.splits, 3, self.conditions)
            model = SASRec(16, length=3, dim=8, heads=2, blocks=1, dropout=0).eval()
            vector, _ = encode_queries(model, replay["queries"], "cpu", 4)
            actual, info = rank_vectors(model, vector, replay["refs"], replay["targets"], "cpu", 4)
            for i, refs in enumerate(replay["refs"]):
                expected = ranks(model, replay["queries"][refs], replay["targets"], "cpu", 3)
                np.testing.assert_array_equal(actual[i], expected)
            self.assertLess(info["unique_state_target_pairs"], actual.size)
            summary = summarize(replay, actual, self.conditions, self.splits, 42)
            fresh = summary["controller_train"]["conditions"][0]
            self.assertEqual(fresh["stale_minus_fresh"]["ndcg@10"]["difference"], 0)
            self.assertIsNone(fresh["strata"]["past_day_50_plus"]["metrics"])

    def test_ranking_excludes_special_tokens_and_preserves_tie_order(self):
        model = SASRec(4, length=3, dim=2, heads=1, blocks=1, dropout=0).eval()
        with torch.no_grad():
            model.item.weight.zero_()
            model.item.weight[:2].fill_(1000)
        refs, targets = np.array([[0,0], [0,1]]), np.array([3,5])
        actual, info = rank_vectors(model, torch.zeros(2,2), refs, targets, "cpu", 3)
        np.testing.assert_array_equal(actual, [[2,4], [2,4]])
        self.assertEqual(info["unique_state_target_pairs"], 3)

    def test_large_snapshot_catalog_product_does_not_overflow_int32(self):
        model = SASRec(50000, length=1, dim=2, heads=1, blocks=1, dropout=0).eval()
        with torch.no_grad():
            model.item.weight.zero_()
        refs = np.array([[42948,49999]], dtype=np.int32)
        targets = np.array([123,49999], dtype=np.int32)
        actual, _ = rank_vectors(model, torch.zeros(50000,2), refs, targets, "cpu", 2)
        np.testing.assert_array_equal(actual, [[122,49998]])

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS unavailable")
    def test_cpu_mps_snapshot_ranks_agree(self):
        model = SASRec(12, length=3, dim=8, heads=2, blocks=1, dropout=0).eval()
        other = copy.deepcopy(model).to("mps")
        x = np.array([[0,2,3], [3,4,5], [6,7,8]], dtype=np.int32)
        refs, targets = np.array([[0,1], [1,2], [0,1]]), np.array([4,9])
        cpu, _ = encode_queries(model, x, "cpu", 3)
        mps, _ = encode_queries(other, x, "mps", 3)
        torch.testing.assert_close(cpu, mps.cpu(), atol=1e-5, rtol=1e-4)
        a, _ = rank_vectors(model, cpu, refs, targets, "cpu", 2)
        b, _ = rank_vectors(other, mps, refs, targets, "mps", 2)
        np.testing.assert_array_equal(a, b)


if __name__ == "__main__":
    unittest.main()
