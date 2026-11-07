import sys
import tempfile
import unittest
from pathlib import Path

import duckdb
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from compare_sasrec import item_mapping, paired_intervals


class PilotComparisonTest(unittest.TestCase):
    def test_mapping_uses_item_ids_and_rejects_missing_tracks(self):
        with tempfile.TemporaryDirectory() as tmp:
            old, new = Path(tmp) / "old", Path(tmp) / "new"
            old.mkdir()
            new.mkdir()
            with duckdb.connect() as db:
                db.sql("SELECT * FROM (VALUES (20, 2), (40, 3)) t(item_id, item_idx)").write_parquet(str(old / "items.parquet"))
                db.sql("SELECT * FROM (VALUES (10, 2), (20, 3), (30, 4), (40, 5)) t(item_id, item_idx)").write_parquet(str(new / "items.parquet"))
                np.testing.assert_array_equal(item_mapping(old, new), [0, 1, 3, 5])
                db.sql("SELECT * FROM (VALUES (10, 2), (20, 3)) t(item_id, item_idx)").write_parquet(str(new / "items.parquet"), overwrite=True)
                with self.assertRaisesRegex(ValueError, "does not cover"):
                    item_mapping(old, new)

    def test_identical_predictions_have_zero_paired_difference(self):
        rank = np.array([1, 2, 30, 120])
        result = paired_intervals(rank, rank, np.array([1, 1, 2, 2]), 42)
        for values in result.values():
            self.assertEqual(values["difference"], 0)
            self.assertEqual(values["cluster_bootstrap_95ci"], [0, 0])


if __name__ == "__main__":
    unittest.main()
