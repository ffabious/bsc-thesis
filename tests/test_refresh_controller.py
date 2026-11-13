import copy
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from refresh_controller import HistoryFeatures,fit_ridge,rollout,training_examples
from train_refresh_controller import region_rows
from report_student_maintenance import cost_matched_mixture


class RefreshControllerTest(unittest.TestCase):
    def setUp(self):
        self.stream={"users":np.array([0,1,0,0,1]),"timestamp":np.array([10,12,20,30,32]),
            "history":np.array([2,1,3,4,2]),"last_input_timestamp":np.array([5,0,10,20,12]),
            "group_size":np.ones(5,dtype=int),"split_idx":np.array([1,1,1,2,2]),
            "target_request":np.array([0,0,2,4])}
        self.prefix_users=np.array([0,0,1]);self.prefix_times=np.array([0,5,0])
        self.features=HistoryFeatures(self.stream,self.prefix_users,self.prefix_times)

    def test_vectorized_training_and_measured_scalar_features_match(self):
        matrix=self.features.matrix(np.array([2,3,4]),np.array([0,2,1]))
        for i,(request,previous) in enumerate([(2,0),(3,2),(4,1)]):
            np.testing.assert_allclose(matrix[i],self.features.row(request,previous),atol=1e-14)
        self.assertAlmostEqual(matrix[0,2],np.log1p(3))  # Current group at t=20 is excluded.

    def test_future_groups_do_not_change_earlier_features(self):
        stream=copy.deepcopy(self.stream);stream["timestamp"][3:]+=100000
        other=HistoryFeatures(stream,self.prefix_users,self.prefix_times)
        np.testing.assert_array_equal(other.row(2,0),self.features.row(2,0))

    def test_closed_loop_initializes_and_resets_cache_without_using_labels(self):
        model={"coefficients":[0.]*8,"intercept":-1.}
        refs,refresh,before=rollout(self.stream,self.features,model,0.)
        np.testing.assert_array_equal(refs,[0,1,0,3,4]);np.testing.assert_array_equal(refresh,[1,1,0,1,1])
        np.testing.assert_array_equal(before,[-1,-1,0,-1,-1])
        stream=copy.deepcopy(self.stream);stream["target_request"]=np.array([],dtype=int)
        np.testing.assert_array_equal(rollout(stream,self.features,model,0.)[0],refs)
        model["intercept"]=1.
        np.testing.assert_array_equal(rollout(self.stream,self.features,model,0.)[0],np.arange(5))

    def test_training_keeps_zero_reward_requests_and_deduplicates_states(self):
        stream={"target_request":self.stream["target_request"]}
        refs=np.array([np.arange(5),[0,0,0,3,3],[0,0,0,3,3]])
        rank=np.array([[1,11,1,101],[101,101,11,101],[101,101,11,101]])
        rows=training_examples(stream,rank,refs,self.stream["split_idx"]==1)
        np.testing.assert_array_equal(rows["requests"],[1,2]);np.testing.assert_array_equal(rows["gain"],[0.,1.])
        np.testing.assert_array_equal(rows["weights"],[1.,1.])
        rank[:,-1]=1
        poisoned=training_examples(stream,rank,refs,self.stream["split_idx"]==1)
        for key in rows:np.testing.assert_array_equal(rows[key],poisoned[key])

    def test_weighted_ridge_recovers_linear_prediction(self):
        rng=np.random.default_rng(42);x=rng.normal(size=(80,8));beta=np.arange(8)/8;y=x@beta+.1
        model=fit_ridge(x,y,np.ones(80),1e-12)
        np.testing.assert_allclose(x@model["coefficients"]+model["intercept"],y,atol=1e-10)

    def test_cost_uses_complete_controller_specific_branches_without_double_charge(self):
        branches=[{"refresh_write_serve":{"median_mean_ms":3},"cache_hit_serve":{"median_mean_ms":2}},
                  {"refresh_write_serve":{"median_mean_ms":4},"cache_hit_serve":{"median_mean_ms":1}}]
        rows=region_rows(["a","b"],np.array([[1,2],[2,1]]),np.arange(2),np.array([[1,0],[1,1]],dtype=bool),
            np.ones(2,dtype=bool),{"target_request":np.arange(2)},[0.,0.],branches)
        self.assertEqual(rows[0]["cost"]["estimated_ms_per_request"],2.5)
        self.assertEqual(rows[1]["cost"]["estimated_ms_per_request"],4.)

    def test_ct_mixture_matches_cost_and_can_improve_discrete_grid(self):
        rows=[{"name":name,"cost":{"estimated_ms_per_request":cost},"metrics":{"ndcg@10":score}}
              for name,cost,score in [("a",1,.02),("b",2,.06),("c",1.5,.03)]]
        mixture=cost_matched_mixture(rows,1.5)
        self.assertEqual((mixture["left"],mixture["right"]),("a","b"))
        self.assertEqual(mixture["right_probability"],.5);self.assertAlmostEqual(mixture["ct_ndcg"],.04)


if __name__=="__main__":unittest.main()
