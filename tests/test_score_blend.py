import copy
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from blend_sasrec import blended_ranks, normalize_scores, score_ranks, select_alpha, validate_grid
from sasrec import SASRec


class ScoreBlendTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)

    def test_normalization_is_invariant_to_query_scale_and_offset(self):
        values = torch.tensor([[1., 2., 4.], [3., 3., 3.]])
        expected = normalize_scores(values.clone())
        scaled = values * torch.tensor([[5.], [.2]]) + torch.tensor([[17.], [-9.]])
        torch.testing.assert_close(normalize_scores(scaled), expected)
        self.assertTrue(torch.isfinite(expected).all())
        torch.testing.assert_close(expected[1], torch.zeros(3))

    def test_selection_uses_ndcg_then_smaller_alpha(self):
        curve = [{"alpha": a, "metrics": {"ndcg@10": n, "recall@100": r}}
                 for a, n, r in [(0., .1, .9), (.5, .2, .3), (1., .2, .8)]]
        self.assertEqual(select_alpha(curve), .5)
        validate_grid([0., .5, 1.])
        for values in ([.1, 1.], [0., .5], [0., .5, .5, 1.], [0., float("nan"), 1.]):
            with self.assertRaises(ValueError):
                validate_grid(values)

    def test_score_ties_use_item_id(self):
        ids = torch.tensor([2, 4, 7])
        scores = torch.tensor([[1., 1., 0.], [2., 2., 2.]])
        torch.testing.assert_close(score_ranks(scores, ids, torch.tensor([4, 7])), torch.tensor([2, 3], dtype=torch.int32))

    def test_grid_uses_each_models_own_vectors_and_mask(self):
        models = {key: SASRec(3, length=3, dim=2, heads=1, blocks=1, dropout=0).eval()
                  for key in ("long", "recent")}
        vectors = {"long": np.array([[2., 0.], [0., 1.], [-1., 0.]], dtype=np.float32),
                   "recent": np.array([[0., 1.], [-2., 0.], [0., -1.]], dtype=np.float32)}
        queries = {"long": np.array([[1., .2], [.1, 1.]], dtype=np.float32),
                   "recent": np.array([[1., .3], [.5, -.2]], dtype=np.float32)}
        for key, model in models.items():
            with torch.no_grad():
                model.item.weight[2:].copy_(torch.from_numpy(vectors[key]))
        x, y = np.array([[2, 3, 4], [3, 4, 2]], dtype=np.int32), np.array([2, 4])
        configs = {"long": {}, "recent": {"context_keep": 2}}
        with patch.object(models["long"], "forward", return_value=torch.from_numpy(queries["long"])) as long_call, \
             patch.object(models["recent"], "forward", return_value=torch.from_numpy(queries["recent"])) as recent_call:
            actual = blended_ranks(models, configs, x, y, "cpu", 2, [0., .5, 1.])
        raw = {key: queries[key] @ vectors[key].T for key in models}
        normalized = {key: (v - v.mean(1, keepdims=True)) / v.std(1, keepdims=True) for key, v in raw.items()}
        expected = []
        for values in (raw["long"], (normalized["long"] + normalized["recent"]) / 2, raw["recent"]):
            expected.append([sorted(range(3), key=lambda j: (-float(row[j]), j)).index(int(target - 2)) + 1
                             for row, target in zip(values, y)])
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(long_call.call_args.args[0].numpy(), x)
        np.testing.assert_array_equal(recent_call.call_args.args[0][:, 0].numpy(), [0, 0])
        np.testing.assert_array_equal(x[:, 0], [2, 3])

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS unavailable")
    def test_cpu_and_mps_mixture_ranks_agree(self):
        models = {key: SASRec(6, length=3, dim=4, heads=1, blocks=1, dropout=0).eval()
                  for key in ("long", "recent")}
        configs = {"long": {}, "recent": {"context_keep": 2}}
        x, y = np.array([[2, 3, 4], [5, 6, 7]], dtype=np.int32), np.array([3, 7])
        cpu = blended_ranks(models, configs, x, y, "cpu", 2, [0., .5, 1.])
        mps = blended_ranks({key: copy.deepcopy(value).to("mps") for key, value in models.items()},
                            configs, x, y, "mps", 2, [0., .5, 1.])
        np.testing.assert_array_equal(cpu, mps)


if __name__ == "__main__":
    unittest.main()
