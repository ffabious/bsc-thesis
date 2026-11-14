import copy
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from content_refresh_features import ContentFeatures
from tests import test_refresh_controller as controller_tests


class ContentFeatureTest(unittest.TestCase):
    def setUp(self):
        base=controller_tests.RefreshControllerTest();base.setUp();self.stream=base.stream
        self.stream["queries"]=np.array([[0,0,2,3],[0,0,0,4],[0,2,3,5],[2,3,5,3],[0,0,4,4]])
        self.embeddings=np.zeros((8,2),dtype=np.float32);self.embeddings[3]=[1,0];self.embeddings[5]=[0,1];self.embeddings[4]=[1,1]
        self.prefix_users=base.prefix_users;self.prefix_times=base.prefix_times
        self.features=ContentFeatures(self.stream,self.prefix_users,self.prefix_times,self.embeddings,np.arange(8))

    def test_content_cues_and_scalar_matrix_paths(self):
        matrix=self.features.matrix(np.array([2,3,4]),np.array([0,2,1]))
        for j,(request,previous) in enumerate([(2,0),(3,2),(4,1)]):np.testing.assert_allclose(matrix[j],self.features.row(request,previous),rtol=1e-6,atol=1e-7)
        np.testing.assert_allclose(matrix[:,8:],[[0,0,0,np.log(6)],[0,1,0,np.log(4)],[1,1,1,np.log(5)]],atol=1e-7)

    def test_future_queries_or_labels_do_not_change_current_cues(self):
        other=copy.deepcopy(self.stream);other["queries"][3:]=99;other["targets"]=np.array([999999])
        features=ContentFeatures(other,self.prefix_users,self.prefix_times,self.embeddings,np.arange(8))
        np.testing.assert_array_equal(features.row(2,0),self.features.row(2,0))


if __name__=="__main__":unittest.main()
