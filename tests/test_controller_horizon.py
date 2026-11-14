import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from controller_horizon import horizon_design,horizon_gain,replace_gain


class HorizonLabelsTest(unittest.TestCase):
    def test_no_crossing_users_or_training_boundary_and_zero_target_requests_count(self):
        stream={"users":np.array([0,1,0,1,0,0]),"timestamp":np.array([1,1,2,2,3,4]),
            "target_request":np.array([0,2,2,4,5]),"targets":np.array([2,3,4,5,6])}
        region=np.array([True,True,True,True,True,False])
        rows={"requests":np.array([2,3]),"previous":np.array([0,1]),"gain":np.zeros(2)}
        design=horizon_design(stream,rows,region,5,10)
        np.testing.assert_array_equal(design["requests_in_horizon"],[2,1])
        np.testing.assert_array_equal(design["owner"],[0,0,0])
        np.testing.assert_array_equal(design["fresh"],[23,24,25])
        keys=np.unique(np.r_[design["fresh"],design["stale"]]);rank=np.where(keys>=20,1,1000)
        np.testing.assert_allclose(horizon_gain(design,keys,rank),[1.5,0])
        saved={f"fit_{key}":value for key,value in rows.items()};saved["fit_gain"]=np.array([1.5,0])
        np.testing.assert_array_equal(replace_gain(rows,saved,"fit")["gain"],[1.5,0])
        saved["fit_previous"]=np.array([1,0])
        with self.assertRaises(AssertionError):replace_gain(rows,saved,"fit")


if __name__=="__main__":unittest.main()
