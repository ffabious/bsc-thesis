import copy
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))

from blend_sasrec import score_ranks
from profile_sasrec_hybrid import Serving,estimated_cost
from replay_hybrid_sasrec import naive_ranks
from sasrec import SASRec


class HybridServingTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(19)
        self.models = {r:SASRec(12,length=3,dim=8,heads=2,blocks=1,dropout=0).eval() for r in ("long","recent")}
        self.configs = {"long":{},"recent":{"context_keep":2}}
        self.service = Serving(self.models,self.configs,"cpu")
        self.x = np.array([[2,3,4],[5,6,7],[7,8,9]],dtype=np.int32)

    def test_cached_long_and_fresh_recent_match_independent_reencoding(self):
        old = np.array([0,1]);now = np.array([1,2]);y = np.array([6,10],dtype=np.uint32)
        cache = self.service.encode("long",self.x[old]).numpy().copy()
        for alpha in (0,.5,1):
            scores = self.service.scores(self.x[now],cache.copy(),False,alpha)
            actual = score_ranks(scores,torch.arange(2,14),torch.tensor(y.astype(np.int64))).numpy()
            expected = naive_ranks(self.models,self.configs,self.x,old,now,y,np.full(2,alpha),"cpu",1e-8)
            np.testing.assert_array_equal(actual,expected)

    def test_refresh_writes_cache_and_direct_fresh_avoids_write(self):
        cache = np.zeros((2,8),dtype=np.float32)
        result = self.service.scores(self.x[:2],cache,True,.5)
        np.testing.assert_allclose(cache,self.service.encode("long",self.x[:2]).numpy(),atol=1e-6)
        direct_cache = np.zeros_like(cache)
        direct = self.service.scores(self.x[:2],direct_cache,True,.5,True)
        torch.testing.assert_close(result,direct)
        np.testing.assert_array_equal(direct_cache,0)

    def test_alpha_endpoints_skip_unused_encoder_and_normalization(self):
        cache = self.service.encode("long",self.x[:2]).numpy().copy()
        with patch.object(self.models["recent"],"forward",side_effect=AssertionError("recent must be skipped")):
            self.service.scores(self.x[:2],cache,False,0)
        original = cache.copy()
        with patch.object(self.models["long"],"forward",side_effect=AssertionError("long must be skipped")):
            self.service.scores(self.x[:2],cache,True,1)
        np.testing.assert_array_equal(cache,original)

    def test_full_serving_returns_final_topk_only(self):
        cache = np.zeros((2,8),dtype=np.float32)
        result = self.service.serve(self.x[:2],cache,True,.5)
        self.assertEqual(result.shape,(2,12))
        for row in result.numpy():
            self.assertEqual(set(row),set(range(12)))

    def test_cost_charges_refreshes_and_skips_unused_long_or_cache_writes(self):
        series = lambda v:{"repeat_mean_ms_per_request":[v,v,v]}
        paths = {"long_direct":series(1),"long_refresh":series(2),"long_hit":series(.5),
                 "recent_direct":series(.8),"hybrid_direct":series(1.8),
                 "hybrid_refresh_0.5":series(3),"hybrid_hit_0.5":series(1.5)}
        cost = lambda f,p,a,r:estimated_cost(f,p,a,r,.1,paths)["estimated_ms_per_request"]
        self.assertAlmostEqual(cost("hybrid","time_300",.5,.2),1.9)
        self.assertEqual(cost("hybrid","always",.5,1),1.8)
        self.assertEqual(cost("long","always",0,1),1)
        self.assertEqual(cost("hybrid","time_300",1,.8),.8)

    @unittest.skipUnless(torch.backends.mps.is_available(),"MPS unavailable")
    def test_full_serving_cpu_mps_scores_and_cache_write_agree(self):
        other = Serving({r:copy.deepcopy(m).to("mps") for r,m in self.models.items()},self.configs,"mps")
        a,b = np.zeros((2,8),dtype=np.float32),np.zeros((2,8),dtype=np.float32)
        cpu = self.service.scores(self.x[:2],a,True,.5)
        mps = other.scores(self.x[:2],b,True,.5)
        torch.testing.assert_close(cpu,mps.cpu(),atol=1e-5,rtol=1e-4)
        np.testing.assert_allclose(a,b,atol=1e-5,rtol=1e-4)


if __name__=="__main__":
    unittest.main()
