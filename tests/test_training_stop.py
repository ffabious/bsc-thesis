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
    def run_tiny_training(self, epochs, validation, context_keep=None):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "manifest.json").write_text(json.dumps({"candidate_items": 4}))
            config = {"length": 3, "dim": 4, "heads": 1, "blocks": 1, "dropout": 0,
                      "seed": 42, "batch_size": 2, "negatives": 2, "learning_rate": .001,
                      "eval_batch_size": 2, "epochs": epochs, "patience": 2}
            if context_keep is not None:
                config["context_keep"] = context_keep
            examples = (np.array([[0, 2, 3], [0, 3, 4]], dtype=np.int32),
                        np.array([4, 5]), np.array([0, 1]))
            with patch.object(run_sasrec, "examples", side_effect=lambda *args: tuple(a.copy() for a in examples)), \
                 patch.object(run_sasrec, "step", wraps=run_sasrec.step) as steps, \
                 patch.object(run_sasrec, "ranks", side_effect=[np.array([r, r]) for r in validation]) as evaluations, \
                 patch.object(run_sasrec, "most_popular", return_value=np.array([2, 3])):
                result = run_sasrec.train(config, root, "cpu", root / "training")
                if context_keep is not None:
                    for call in steps.call_args_list:
                        self.assertEqual(call.args[2].shape[1], config["length"])
                        self.assertFalse(call.args[2][:, :-context_keep].any())
                    for call in evaluations.call_args_list:
                        self.assertEqual(call.args[1].shape[1], config["length"])
                        self.assertFalse(call.args[1][:, :-context_keep].any())
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

    def test_recent_context_is_used_in_training_and_validation(self):
        result, saved = self.run_tiny_training(None, [4, 1, 2, 3], context_keep=1)
        self.assertEqual(result["stop_reason"], "validation_patience")
        self.assertEqual(saved["config"]["length"], 3)
        self.assertEqual(saved["config"]["context_keep"], 1)


if __name__ == "__main__":
    unittest.main()
