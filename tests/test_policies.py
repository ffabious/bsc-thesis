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

from replay_sasrec import encode_queries, rank_vectors
from replay_sasrec_policies import frontier, load_requests, policies, policy_cost, rank_reusing, schedule, select_budgets
from sasrec import SASRec, examples, ranks


class PolicyTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)
        self.splits = ["controller_train", "policy_valid"]

    def dataset(self, root, future=9, flip_labels=False):
        (root / "events").mkdir(exist_ok=True)
        periods = ["train", "model_valid", *self.splits, "test"]
        (root / "manifest.json").write_text(json.dumps({"periods": [{"split": s} for s in periods]}))
        # Nonpositive requests and tied targets must affect history, but never current decisions.
        rows = [(0,2,0,0,False,0), (0,3,5,1,False,0), (0,4,10,2,False,1),
                (0,5,20,3,False,2), (0,6,30,4,True,2), (0,7,30,4,True,2), (0,8,40,6,True,2),
                (0,future,50,7,True,3), (0,10,60,8,True,3), (1,11,0,0,False,0),
                (1,12,20,1,True,2), (1,13,35,2,False,2), (1,14,50,3,True,3)]
        if flip_labels:
            rows = [(*r[:4], not r[4], r[5]) for r in rows]
        with duckdb.connect() as db:
            db.execute("CREATE TABLE e(user_idx INTEGER,item_idx INTEGER,timestamp BIGINT,history_size BIGINT,is_target BOOLEAN,split INTEGER,event_id BIGINT)")
            db.executemany("INSERT INTO e VALUES (?,?,?,?,?,?,?)", [(*r,i) for i,r in enumerate(rows)])
            for i, split in enumerate(periods[:-1]):
                db.sql(f"SELECT * EXCLUDE(split) FROM e WHERE split={i}").write_parquet(str(root / "events" / f"{split}.parquet"))
        (root / "events/test.parquet").write_bytes(b"Test must never be read")

    def test_stream_keeps_nonpositive_requests_and_whole_groups(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.dataset(root)
            stream = load_requests(root, self.splits, 3)
            np.testing.assert_array_equal(stream["timestamp"], [20,20,30,35,40,50,50,60])
            np.testing.assert_array_equal(stream["group_size"], [1,1,2,1,1,1,1,1])
            np.testing.assert_array_equal(stream["queries"][2], [3,4,5])
            self.assertEqual(len(stream["targets"]), 7)
            for i, split in enumerate(self.splits, 1):
                x,y,u = examples(root, split, 3)
                mask = stream["split_idx"][stream["target_request"]] == i
                take = stream["target_request"][mask]
                source = stream["source_row"][mask]
                np.testing.assert_array_equal(x[source], stream["queries"][take])
                np.testing.assert_array_equal(y[source], stream["targets"][mask])
                np.testing.assert_array_equal(u[source], stream["users"][take])
            with self.assertRaisesRegex(ValueError, "forbidden"):
                load_requests(root, ["test"], 3)

    def test_exact_schedules_include_initialization_and_reset_at_split(self):
        expected = {"always": [0,1,2,3,4,5,6,7], "never": [0,1,0,1,0,5,6,5],
                    "events_2": [0,1,0,1,4,5,6,5], "time_15": [0,1,0,3,4,5,6,5],
                    "hybrid_2_15": [0,1,0,3,4,5,6,5], "hybrid_2_10": [0,1,2,3,4,5,6,7]}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.dataset(root)
            stream = load_requests(root, self.splits, 3)
            for p in policies({"event_thresholds": [2], "time_thresholds_seconds": [10,15]}):
                if p["name"] not in expected:
                    continue
                refs, refresh = schedule(stream, p, 2)
                np.testing.assert_array_equal(refs, expected[p["name"]])
                np.testing.assert_array_equal(refresh, refs == np.arange(8))
                self.assertTrue(refresh[[0,1,5,6]].all())
                self.assertTrue(np.all(stream["split_idx"][refs] == stream["split_idx"]))
            # Event thresholds can be exceeded by a tied group; no mid-group update occurs.
            refs, _ = schedule(stream, {"family": "events", "events": 2, "seconds": 0}, 2)
            self.assertEqual(stream["history"][4] - stream["history"][refs[2]], 3)

    def test_labels_and_future_items_do_not_change_past_decisions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.dataset(root)
            first = load_requests(root, self.splits, 3)
            self.dataset(root, future=15, flip_labels=True)
            second = load_requests(root, self.splits, 3)
            for key in ("users", "timestamp", "history", "split_idx"):
                np.testing.assert_array_equal(first[key], second[key])
            np.testing.assert_array_equal(first["queries"][:5], second["queries"][:5])
            for p in policies({"event_thresholds": [2], "time_thresholds_seconds": [15]}):
                a,b = schedule(first,p,2), schedule(second,p,2)
                np.testing.assert_array_equal(a[0],b[0])
                np.testing.assert_array_equal(a[1],b[1])

    def test_cached_policy_ranks_match_naive_fresh_recomputation_of_selected_windows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.dataset(root)
            stream = load_requests(root,self.splits,3)
            choices = policies({"event_thresholds": [2], "time_thresholds_seconds": [15]})
            refs = np.array([schedule(stream,p,2)[0] for p in choices])[:,stream["target_request"]]
            model = SASRec(16,length=3,dim=8,heads=2,blocks=1,dropout=0).eval()
            vectors,_ = encode_queries(model,stream["queries"],"cpu",4)
            actual,_ = rank_vectors(model,vectors,refs,stream["targets"],"cpu",4)
            for i, row in enumerate(refs):
                expected = ranks(model,stream["queries"][row],stream["targets"],"cpu",3)
                np.testing.assert_array_equal(actual[i],expected)

    def test_cost_counts_paid_initialization_without_double_charging_refresh(self):
        branches = {"refresh_write_serve": {"median_mean_ms": 2}, "cache_hit_serve": {"median_mean_ms": 1}}
        c = policy_cost(2,5,.1,branches)
        self.assertAlmostEqual(c["estimated_total_ms"],7.5)
        self.assertAlmostEqual(c["estimated_ms_per_request"],1.5)
        self.assertEqual(c["refresh_branch_total_ms"],4)
        self.assertEqual(c["hit_branch_total_ms"],3)
        self.assertEqual(policy_cost(5,5,0,branches)["estimated_total_ms"],10)

    def test_rank_reuse_matches_naive_ranks_and_skips_already_scored_pairs(self):
        model = SASRec(12,length=3,dim=8,heads=2,blocks=1,dropout=0).eval()
        x = np.array([[0,2,3],[3,4,5],[6,7,8]],dtype=np.int32)
        vectors,_ = encode_queries(model,x,"cpu",3)
        y = np.array([4,9,10])
        old_refs = np.array([[0,1,1],[0,0,1]],dtype=np.int32)
        old_rank,_ = rank_vectors(model,vectors,old_refs,y,"cpu",3)
        refs = np.array([[0,1,2],[1,0,2],[0,1,1]],dtype=np.int32)
        actual,info = rank_reusing(model,vectors,refs,y,"cpu",3,(old_refs,old_rank))
        for i,row in enumerate(refs):
            np.testing.assert_array_equal(actual[i],ranks(model,x[row],y,"cpu",3))
        self.assertEqual(info["reused_state_target_pairs"],4)
        self.assertEqual(info["computed_state_target_pairs"],2)
        reused,all_info = rank_reusing(model,vectors,old_refs,y,"cpu",3,(old_refs,old_rank))
        np.testing.assert_array_equal(reused,old_rank)
        self.assertEqual(all_info["computed_state_target_pairs"],0)

    def test_frontier_and_ct_budget_selection(self):
        def row(name,cost,ndcg,recall):
            return {"name":name,"cost":{"estimated_ms_per_request":cost},"metrics":{"ndcg@10":ndcg,"recall@100":recall}}
        rows = [row("always",2,.06,.3),row("never",1,.02,.1),row("a",1.5,.05,.2),
                row("b",1.5,.04,.25),row("dominated",1.8,.04,.2)]
        self.assertEqual(frontier(rows,"ndcg@10"),["always","never","a"])
        self.assertEqual(frontier(rows,"recall@100"),["always","never","b"])
        self.assertEqual(select_budgets(rows,[.4,.6,.75]),[
            {"relative_budget":.4,"policy":None},{"relative_budget":.6,"policy":"never"},
            {"relative_budget":.75,"policy":"a"}])
        with self.assertRaises(ValueError):
            policies({"event_thresholds":[0],"time_thresholds_seconds":[15]})

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS unavailable")
    def test_policy_rank_oracle_cpu_mps(self):
        model = SASRec(12,length=3,dim=8,heads=2,blocks=1,dropout=0).eval()
        other = copy.deepcopy(model).to("mps")
        x = np.array([[0,2,3],[3,4,5],[6,7,8]],dtype=np.int32)
        refs,y = np.array([[0,1,2],[0,1,1]]),np.array([4,9,10])
        a,_ = encode_queries(model,x,"cpu",3)
        b,_ = encode_queries(other,x,"mps",3)
        cpu,_ = rank_vectors(model,a,refs,y,"cpu",3)
        mps,_ = rank_vectors(other,b,refs,y,"mps",3)
        np.testing.assert_array_equal(cpu,mps)


if __name__ == "__main__":
    unittest.main()
