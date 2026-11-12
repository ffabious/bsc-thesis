import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from replay_encoder_pairs import compare_pair,paired_effect


class StalenessComparisonTest(unittest.TestCase):
    def test_user_bootstrap_preserves_event_weighting(self):
        effect=paired_effect(np.array([[1.,1.,1.,-1.]]),np.array([0,0,0,1]))[0]
        self.assertEqual(effect["difference"],.5)
        self.assertEqual(effect["cluster_bootstrap_95ci"],[-1.,1.])

    def test_fresh_advantage_alone_is_not_lower_staleness_loss(self):
        ce=np.array([[1,101],[101,101]])
        student=np.array([[1,1],[1,101]])
        result=compare_pair(student,ce,np.array([0,1]))
        for stats in result.values():
            self.assertEqual(stats["student_minus_ce"][1]["difference"],.5)
            self.assertEqual(stats["loss_reduction_vs_ce"][0]["cluster_bootstrap_95ci"],[0.,0.])
            self.assertEqual(stats["loss_reduction_vs_ce"][1]["difference"],0.)

    def test_positive_loss_reduction_means_student_keeps_more_quality(self):
        ce=np.array([[1,1],[101,101]])
        student=np.array([[1,1],[1,101]])
        result=compare_pair(student,ce,np.array([0,1]))
        for stats in result.values():
            self.assertEqual(stats["loss_reduction_vs_ce"][1]["difference"],.5)
            self.assertEqual(stats["loss_reduction_vs_ce"][1]["cluster_bootstrap_95ci"],[0.,1.])


if __name__=="__main__":unittest.main()
