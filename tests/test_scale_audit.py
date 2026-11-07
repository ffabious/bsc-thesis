import json
import sys
import tempfile
import unittest
from pathlib import Path

import duckdb
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from diagnostics.audit_pilot_scale import aligned_examples, repeat_distribution
from diagnostics.audit_pilot_training import vector_ranks


class ScaleAuditTest(unittest.TestCase):
    def test_alignment_keeps_reference_users_and_targets_with_expanded_histories(self):
        with tempfile.TemporaryDirectory() as tmp:
            small, large = Path(tmp) / "small", Path(tmp) / "large"
            with duckdb.connect() as db:
                for folder, users, rows in [
                    (small, [(7, 0), (8, 1)], [(7, 2, 0, 0, 0, False), (7, 3, 1, 1, 5, True),
                         (7, 2, 2, 3, 15, True), (8, 2, 0, 4, 0, False), (8, 3, 1, 5, 15, True)]),
                    (large, [(7, 9), (8, 10), (9, 11)], [(7, 2, 0, 0, 0, False), (7, 4, 1, 1, 5, True),
                         (7, 3, 2, 2, 10, True), (7, 2, 3, 3, 15, True), (8, 2, 0, 4, 0, False),
                         (8, 4, 1, 5, 15, True), (9, 3, 0, 6, 0, False), (9, 3, 1, 7, 15, True)])]:
                    (folder / "events").mkdir(parents=True)
                    (folder / "manifest.json").write_text(json.dumps({"periods":[{"split":"train"},{"split":"test"}]}))
                    db.execute("CREATE OR REPLACE TABLE users(uid INTEGER, user_idx INTEGER)")
                    db.executemany("INSERT INTO users VALUES (?, ?)", users)
                    db.sql("SELECT * FROM users").write_parquet(str(folder / "users.parquet"))
                    db.execute("""CREATE OR REPLACE TABLE events(uid INTEGER, item_idx INTEGER,
                        history_size INTEGER, event_id INTEGER, timestamp INTEGER, is_target BOOLEAN)""")
                    db.executemany("INSERT INTO events VALUES (?, ?, ?, ?, ?, ?)", rows)
                    for name, condition in [("train", "timestamp < 10"), ("test", "timestamp >= 10")]:
                        db.sql(f"SELECT * FROM events WHERE {condition}").write_parquet(str(folder / "events" / f"{name}.parquet"))
                x, y, users, ids = aligned_examples(large, small, "test", 3)
                np.testing.assert_array_equal(x, [[2, 4, 3], [0, 0, 2]])
                np.testing.assert_array_equal(y, [2, 4])
                np.testing.assert_array_equal(users, [0, 1])
                np.testing.assert_array_equal(ids, [3, 5])
                repeats = repeat_distribution(large, small)
                added = next(r for r in repeats if r["split"] == "test" and r["cohort"] == "added_users")
                self.assertEqual(added["last_token_share"], 1)

    def test_vector_scoring_ties_and_candidate_norm_ablation(self):
        query = torch.tensor([[1., 0], [1., 0]])
        vectors = torch.tensor([[1., 0], [1., 0], [3., 1]])
        np.testing.assert_array_equal(vector_ranks(query, vectors, np.array([0, 1])), [2, 3])
        unit = torch.nn.functional.normalize(vectors, dim=1)
        np.testing.assert_array_equal(vector_ranks(query, unit, np.array([0, 1])), [1, 2])


if __name__ == "__main__":
    unittest.main()
