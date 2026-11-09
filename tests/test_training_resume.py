import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_sasrec


class TrainingResumeTest(unittest.TestCase):
    def test_mid_epoch_resume_reproduces_uninterrupted_training(self):
        torch.set_num_threads(1)
        config = {"length": 3, "dim": 8, "item_dim": 4, "ff_dim": 16,
                  "heads": 2, "blocks": 1, "dropout": .2, "seed": 42,
                  "batch_size": 2, "negatives": 2, "learning_rate": .001,
                  "eval_batch_size": 2, "epochs": None, "patience": 2,
                  "checkpoint_steps": 1, "loss_reduction": "joint_mean"}
        arrays = (np.array([[0, 2, 3], [0, 3, 4]] * 3, dtype=np.int32),
                  np.array([4, 5] * 3), np.array([0, 1] * 3))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "manifest.json").write_text('{"candidate_items": 4}')
            with patch.object(run_sasrec, "examples", side_effect=lambda *args: tuple(a.copy() for a in arrays)), \
                 patch.object(run_sasrec, "most_popular", return_value=np.array([2, 3])):
                torch.manual_seed(42)
                with patch.object(run_sasrec, "ranks", side_effect=[np.array([r]) for r in [4, 1, 2, 3]]):
                    run_sasrec.train(config, root, "cpu", root / "full")
                original_step, calls = run_sasrec.step, 0

                def interrupted_step(*args):
                    nonlocal calls
                    calls += 1
                    if calls == 3:
                        raise RuntimeError("simulated interruption")
                    return original_step(*args)

                torch.manual_seed(42)
                with patch.object(run_sasrec, "step", side_effect=interrupted_step), \
                     patch.object(run_sasrec, "ranks", return_value=np.array([4])):
                    with self.assertRaisesRegex(RuntimeError, "simulated"):
                        run_sasrec.train(config, root, "cpu", root / "resumed")
                torch.manual_seed(999)
                with patch.object(run_sasrec, "ranks", side_effect=[np.array([r]) for r in [1, 2, 3]]):
                    result = run_sasrec.train(config, root, "cpu", root / "resumed", resume=True)
            self.assertEqual(result["stop_reason"], "validation_patience")
            left = torch.load(root / "full/last.pt", weights_only=True)
            right = torch.load(root / "resumed/last.pt", weights_only=True)
            for name in left["weights"]:
                self.assertTrue(torch.equal(left["weights"][name], right["weights"][name]), name)
            self.assertEqual([r["loss"] for r in left["result"]["epochs"]],
                             [r["loss"] for r in right["result"]["epochs"]])
