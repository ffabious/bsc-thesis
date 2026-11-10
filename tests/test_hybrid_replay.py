import copy
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))

from replay_hybrid_sasrec import assemble, mixed_ranks, naive_ranks, pair_keys
from replay_sasrec import encode_queries
from sasrec import SASRec, mask_context


class HybridReplayTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(17)
        self.models = {role:SASRec(12,length=3,dim=8,heads=2,blocks=1,dropout=0).eval()
                       for role in ("long","recent")}
        self.configs = {"long":{},"recent":{"context_keep":2}}
        self.x = np.array([[0,2,3],[3,4,5],[5,6,7],[7,8,9]],dtype=np.int32)
        self.refs = np.array([[0,1,2],[1,1,0]],dtype=np.int32)
        self.fresh = np.array([3,2,3],dtype=np.int32)
        self.y = np.array([4,9,10])
        self.alphas = [.2,.5,.8]

    def vectors(self,models,device):
        return {role:encode_queries(model,mask_context(self.x.copy(),self.configs[role].get("context_keep")),device,4)[0]
                for role,model in models.items()}

    def test_distinct_cached_long_and_fresh_recent_inputs_match_naive_scoring(self):
        keys,inverse = pair_keys(self.refs,self.fresh,self.y,len(self.x),14)
        inverse = inverse.reshape(self.refs.shape)
        grid,_ = mixed_ranks(self.models,self.vectors(self.models,"cpu"),keys,len(self.x),self.alphas,"cpu",3,1e-8)
        for i,alpha in enumerate(self.alphas):
            for j,refs in enumerate(self.refs):
                expected = naive_ranks(self.models,self.configs,self.x,refs,self.fresh,self.y,
                                       np.full(len(self.y),alpha),"cpu",1e-8)
                np.testing.assert_array_equal(grid[i,inverse[j]],expected)
        self.assertEqual(len(keys),5)

    def test_needed_mask_skips_unselected_alpha_pairs(self):
        keys,_ = pair_keys(self.refs,self.fresh,self.y,len(self.x),14)
        needed = np.zeros((3,len(keys)),dtype=bool)
        needed[0,0],needed[2,-1] = True,True
        grid,_ = mixed_ranks(self.models,self.vectors(self.models,"cpu"),keys,len(self.x),self.alphas,"cpu",3,1e-8,needed)
        self.assertTrue(np.all(grid[~needed] == -1))
        self.assertTrue(np.all(grid[needed] >= 1))

    def test_full_grid_and_explicit_all_needed_mask_agree(self):
        keys,_ = pair_keys(self.refs,self.fresh,self.y,len(self.x),14)
        vectors = self.vectors(self.models,"cpu")
        direct,_ = mixed_ranks(self.models,vectors,keys,len(self.x),self.alphas,"cpu",3,1e-8)
        masked,_ = mixed_ranks(self.models,vectors,keys,len(self.x),self.alphas,"cpu",3,1e-8,
                               np.ones((len(self.alphas),len(keys)),dtype=bool))
        np.testing.assert_array_equal(direct,masked)

    def test_assembly_preserves_raw_endpoints_and_rejects_unscored_pairs(self):
        long,recent = np.array([1,3]),np.array([4,2])
        grid = np.array([[3,-1],[1,2],[4,2]])
        np.testing.assert_array_equal(assemble(0,self.alphas,grid,np.array([0,1]),long,recent),long)
        np.testing.assert_array_equal(assemble(1,self.alphas,grid,np.array([0,1]),long,recent),recent)
        np.testing.assert_array_equal(assemble(.5,self.alphas,grid,np.array([0,1]),long,recent),[1,2])
        with self.assertRaisesRegex(ValueError,"not scored"):
            assemble(.2,self.alphas,grid,np.array([0,1]),long,recent)

    def test_naive_oracle_accepts_unsigned_item_ids_from_parquet(self):
        unsigned = naive_ranks(self.models,self.configs,self.x,self.refs[0],self.fresh,self.y.astype(np.uint32),
                               np.full(len(self.y),.5),"cpu",1e-8)
        signed = naive_ranks(self.models,self.configs,self.x,self.refs[0],self.fresh,self.y,
                             np.full(len(self.y),.5),"cpu",1e-8)
        np.testing.assert_array_equal(unsigned,signed)

    @unittest.skipUnless(torch.backends.mps.is_available(),"MPS unavailable")
    def test_cpu_mps_cached_long_and_fresh_recent_ranks_agree(self):
        other = {role:copy.deepcopy(model).to("mps") for role,model in self.models.items()}
        keys,_ = pair_keys(self.refs,self.fresh,self.y,len(self.x),14)
        cpu,_ = mixed_ranks(self.models,self.vectors(self.models,"cpu"),keys,len(self.x),self.alphas,"cpu",3,1e-8)
        mps,_ = mixed_ranks(other,self.vectors(other,"mps"),keys,len(self.x),self.alphas,"mps",3,1e-8)
        np.testing.assert_array_equal(cpu,mps)


if __name__ == "__main__":
    unittest.main()
