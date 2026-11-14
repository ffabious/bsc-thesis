import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from list_cache import canonical_topk,list_ranks,previous_states


class ListCacheTest(unittest.TestCase):
    def test_ties_including_boundary_follow_item_order(self):
        scores=torch.tensor([[4.,3.,3.,3.,1.],[0.,0.,0.,0.,0.]])
        np.testing.assert_array_equal(canonical_topk(scores,3),[[0,1,2],[0,1,2]])
        np.testing.assert_array_equal(canonical_topk(scores,5),[[0,1,2,3,4],[0,1,2,3,4]])

    def test_ranks_and_cache_state_boundaries(self):
        top=np.array([[0,1,2],[3,2,1]])
        np.testing.assert_array_equal(list_ranks(top,np.array([[0,1,0]]),np.array([2,4,8])),[[1,2,1000000]])
        stream={"split_idx":np.array([1,1,1,2,2]),"users":np.array([0,1,0,0,1])}
        np.testing.assert_array_equal(previous_states(stream,np.array([0,1,0,3,4])),[-1,-1,0,-1,-1])


if __name__=="__main__":unittest.main()
