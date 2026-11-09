import hashlib
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


    def test_validation_evaluation_uses_no_test_examples(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = {"length": 3, "dim": 4, "heads": 1, "blocks": 1,
                      "dropout": 0, "eval_batch_size": 2, "seed": 42}
            manifest = root / "manifest.json"
            manifest.write_text('{"candidate_items": 4}')
            model = run_sasrec.SASRec(4, **run_sasrec.model_args(config))
            checkpoint = root / "best.pt"
            torch.save({"weights": model.state_dict(), "config": config, "items": 4, "epoch": 1}, checkpoint)
            (root / "training.json").write_text(json.dumps({
                "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                "data_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()}))
            arrays = np.array([[0, 2, 3]], dtype=np.int32), np.array([4]), np.array([0])
            with patch.object(run_sasrec, "examples", return_value=arrays) as read_examples, \
                 patch.object(run_sasrec, "most_popular", return_value=np.array([2])):
                result = run_sasrec.evaluate(config, root, "cpu", checkpoint, root / "valid.json", "model_valid")
            read_examples.assert_called_once_with(root, "model_valid", 3)
            self.assertEqual(result["split"], "model_valid")
            with self.assertRaisesRegex(ValueError, "Evaluation split"):
                run_sasrec.evaluate(config, root, "cpu", checkpoint, root / "invalid.json", "train")


if __name__ == "__main__":
    unittest.main()
