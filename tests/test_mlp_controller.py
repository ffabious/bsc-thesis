import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from mlp_refresh_controller import fit_mlp
from refresh_controller import make_predictor,predict_matrix


class MLPControllerTest(unittest.TestCase):
    def test_numpy_batch_and_scalar_normalized_serving_match(self):
        rng=np.random.default_rng(42)
        model={"kind":"mlp","feature_mean":np.arange(8).tolist(),"feature_scale":[2.]*8,
            "layers":[{"weight":rng.normal(size=(4,8)).tolist(),"bias":[.1]*4},
                      {"weight":rng.normal(size=(1,4)).tolist(),"bias":[.2]}]}
        x=rng.normal(size=(12,8));predict=make_predictor(model)
        np.testing.assert_allclose(predict_matrix(model,x),[predict(row) for row in x],rtol=1e-6,atol=1e-6)

    @torch.inference_mode()
    def test_cpu_training_restores_validation_selected_epoch(self):
        torch.set_num_threads(1);rng=np.random.default_rng(42);x=rng.normal(size=(80,8));y=x[:,0]*.1
        model,curve=fit_mlp(x[:60],y[:60],np.ones(60),x[60:],y[60:],np.ones(20),
            {"model_seed":42,"max_epochs":3,"patience":2})
        self.assertEqual(model["parameters"],1377)
        self.assertEqual(curve["best_epoch"],min(curve["epochs"],key=lambda row:row["validation_mse"])["epoch"])
        prediction=predict_matrix(model,x[60:]);loss=float(np.mean((prediction-y[60:])**2))
        self.assertAlmostEqual(loss,curve["best_validation_mse"],places=6)


if __name__=="__main__":unittest.main()
