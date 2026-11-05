import tempfile
import unittest
from pathlib import Path

import duckdb
import numpy as np

from scripts.prepare_audio_embeddings import item_index, prepare


class AudioEmbeddingTest(unittest.TestCase):
    def test_cached_item_index_cannot_change_source_silently(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / 'raw.parquet'
            with duckdb.connect() as db:
                db.sql('SELECT unnest([2, 3, 3]) AS item_id').write_parquet(str(raw))
            path = item_index(raw, {'revision': 'one'})
            self.assertEqual(item_index(raw, {'revision': 'one'}), path)
            with self.assertRaisesRegex(ValueError, 'different source'):
                item_index(raw, {'revision': 'two'})

    def test_complete_vectors_are_aligned_and_normalized(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with duckdb.connect() as db:
                db.sql("SELECT * FROM (VALUES (100, 2), (200, 3)) t(item_id, item_idx)").write_parquet(str(root / 'items.parquet'))
                db.sql("SELECT * FROM (VALUES (200, [0., 2.]), (100, [3., 4.])) t(item_id, normalized_embed)").write_parquet(str(root / 'raw.parquet'))
            result = prepare(root, root / 'out', {'column': 'normalized_embed', 'dimension': 2}, root / 'raw.parquet')
            np.testing.assert_allclose(np.load(root / 'out/vectors.npy'), [[0, 0], [0, 0], [.6, .8], [0, 1]])
            self.assertEqual(result['missing_items'], 0)

    def test_missing_vectors_are_rejected_without_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with duckdb.connect() as db:
                db.sql("SELECT * FROM (VALUES (100, 2), (200, 3)) t(item_id, item_idx)").write_parquet(str(root / 'items.parquet'))
                db.sql("SELECT 100 AS item_id, [3., 4.] AS normalized_embed").write_parquet(str(root / 'raw.parquet'))
            with self.assertRaisesRegex(ValueError, 'Missing audio'):
                prepare(root, root / 'out', {'column': 'normalized_embed', 'dimension': 2}, root / 'raw.parquet')


if __name__ == '__main__':
    unittest.main()
