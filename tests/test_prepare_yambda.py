import hashlib
import tempfile
import unittest
from pathlib import Path

import duckdb

from scripts.prepare_yambda import DAY, periods, prepare


class PrepareYambdaTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = {
            "holdout_days": {"model_valid": 1, "controller_train": 1,
                             "policy_valid": 1, "test": 1},
            "positive_ratio_gt": 50, "min_history": 1,
            "min_train_events": 1, "seed": 42,
        }
        # User 2 appears only in holdout; item 99 is never seen during train.
        events = [(1, 10, 0, 100), (1, 10, 5, 100), (1, 10, 5, 100),
                  (1, 20, 10, 140), (1, 11, DAY, 50),
                  (1, 99, 2 * DAY, 100), (1, 10, 2 * DAY, 100),
                  (1, 11, 2 * DAY + 5, 51), (2, 999, 3 * DAY, 100),
                  (3, 30, 0, 100)]
        events += [(uid, item, day * DAY, 100)
                   for uid, item in [(1, 10), (3, 30)] for day in (3, 4, 5)]
        events += [(3, 30, 2 * DAY, 100), (1, 10, 6 * DAY - 5, 100)]
        self.db = duckdb.connect()
        self.addCleanup(self.db.close)
        self.db.execute("""
            CREATE TABLE raw(uid UINTEGER, item_id UINTEGER, timestamp BIGINT,
                event_type VARCHAR, is_organic UTINYINT, played_ratio_pct INTEGER,
                track_length_seconds INTEGER)
        """)
        self.db.executemany("INSERT INTO raw VALUES (?, ?, ?, 'listen', 1, ?, 200)", events)
        self.db.execute("INSERT INTO raw VALUES (1, 888, ?, 'like', 1, NULL, NULL)", [2 * DAY])
        self.source = self.root / "raw.parquet"

    def run_prepare(self, name="out", **kwargs):
        self.db.sql("SELECT * FROM raw").write_parquet(str(self.source), overwrite=True)
        output = self.root / name
        manifest = prepare(self.source, output, self.config, threads=1, memory="256MB", **kwargs)
        return output, manifest

    def test_period_boundaries_and_timestamp_groups(self):
        output, manifest = self.run_prepare()
        self.assertEqual(manifest["periods"][0]["end"], 2 * DAY)
        self.assertEqual(manifest["source_audit"]["ratios_over_100"], 1)
        train = self.db.read_parquet(str(output / "events/train.parquet"))
        # Neither simultaneous repeat can see the other as preceding history.
        self.assertEqual(train.filter("uid = 1 AND timestamp = 5").project("history_size").fetchall(),
                         [(1,), (1,)])
        valid = self.db.read_parquet(str(output / "events/model_valid.parquet"))
        self.assertEqual(valid.filter("uid = 1 AND timestamp = 172800").project("history_size").fetchall(),
                         [(5,), (5,)])
        self.assertEqual(train.filter("played_ratio_pct = 50").project("is_positive").fetchall(), [(False,)])
        self.assertTrue(valid.filter("played_ratio_pct = 51").project("is_target").fetchall()[0][0])
        self.assertEqual(sum(row["events"] for row in manifest["splits"]), 17)

    def test_train_only_vocabulary_and_unknown_history(self):
        output, manifest = self.run_prepare()
        self.assertEqual(self.db.read_parquet(str(output / "items.parquet")).project("item_id").fetchall(),
                         [(10,), (11,), (20,), (30,)])
        users = self.db.read_parquet(str(output / "users.parquet")).project("uid").fetchall()
        self.assertEqual(users, [(1,), (3,)])
        valid = self.db.read_parquet(str(output / "events/model_valid.parquet"))
        self.assertEqual(valid.filter("item_id = 99").project("item_idx, is_target").fetchall(), [(1, False)])
        self.assertEqual(valid.filter("uid = 1 AND timestamp = 172805").project("history_size").fetchall(), [(7,)])
        self.assertEqual(manifest["candidate_items"], 4)
        self.assertEqual(manifest["splits"][0]["targets"], 3)

    def test_sample_depends_only_on_train_users_and_seed(self):
        first, _ = self.run_prepare("first", user_limit=1)
        expected = min([1, 3], key=lambda uid: hashlib.sha256(f"42:{uid}".encode()).digest())
        self.assertEqual(self.db.read_parquet(str(first / "users.parquet")).project("uid").fetchall()[0][0], expected)
        self.db.execute("INSERT INTO raw VALUES (999, 666, ?, 'listen', 1, 100, 200)", [5 * DAY])
        second, _ = self.run_prepare("second", user_limit=1)
        self.assertEqual((first / "users.parquet").read_bytes(), (second / "users.parquet").read_bytes())
        self.assertEqual((first / "items.parquet").read_bytes(), (second / "items.parquet").read_bytes())

    def test_rejects_bad_timestamps_and_existing_outputs(self):
        self.db.execute("UPDATE raw SET timestamp = 1 WHERE timestamp = 0")
        with self.assertRaisesRegex(ValueError, "invalid events"):
            self.run_prepare()
        with self.assertRaises(FileExistsError):
            self.run_prepare()

    def test_rejects_insufficient_time_range(self):
        with self.assertRaises(ValueError):
            periods(0, DAY, self.config["holdout_days"])


if __name__ == "__main__":
    unittest.main()
