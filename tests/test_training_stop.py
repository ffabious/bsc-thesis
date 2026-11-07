import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_sasrec


class TrainingStopTest(unittest.TestCase):
    def run_tiny_training(self, epochs, validation):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "manifest.json").write_text(json.dumps({"candidate_items": 4}))
            config = {"length": 3, "dim": 4, "heads": 1, "blocks": 1, "dropout": 0,
                      "seed": 42, "batch_size": 2, "negatives": 2, "learning_rate": .001,
                      "eval_batch_size": 2, "epochs": epochs, "patience": 2}
            examples = (np.array([[0, 2, 3], [0, 3, 4]], dtype=np.int32),
                        np.array([4, 5]), np.array([0, 1]))
            with patch.object(run_sasrec, "examples", return_value=examples), \
                 patch.object(run_sasrec, "ranks", side_effect=[np.array([r, r]) for r in validation]), \
                 patch.object(run_sasrec, "most_popular", return_value=np.array([2, 3])):
                result = run_sasrec.train(config, root, "cpu", root / "training")
            saved = torch.load(root / "training/best.pt", map_location="cpu", weights_only=True)
            return result, saved

    def test_no_epoch_limit_stops_on_validation_and_keeps_best(self):
        # Initial ranks, epoch 1 best, then two worse epochs.
        result, saved = self.run_tiny_training(None, [4, 1, 2, 3])
        self.assertEqual(result["stop_reason"], "validation_patience")
        self.assertEqual(len(result["epochs"]), 3)
        self.assertEqual(result["best_epoch"], 1)
        self.assertEqual(saved["epoch"], 1)

    def test_explicit_epoch_limit_still_works(self):
        result, saved = self.run_tiny_training(2, [4, 3, 2])
        self.assertEqual(result["stop_reason"], "epoch_limit")
        self.assertEqual(len(result["epochs"]), 2)
        self.assertEqual(saved["epoch"], 2)


if __name__ == "__main__":
    unittest.main()
