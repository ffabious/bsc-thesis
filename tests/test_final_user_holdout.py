import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import duckdb
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from final_holdout import verified_protocol
from replay_sasrec_policies import load_requests,request_stream


class FinalHoldoutTest(unittest.TestCase):
    def test_test_reader_preserves_tied_group_causality_and_past_history(self):
        with tempfile.TemporaryDirectory() as temporary,duckdb.connect() as db:
            root=Path(temporary);(root/"events").mkdir();periods=["train","model_valid","controller_train","policy_valid","test"]
            (root/"manifest.json").write_text(json.dumps({"periods":[{"split":name} for name in periods]}))
            db.execute("CREATE TABLE rows(user_idx INTEGER,item_idx INTEGER,timestamp BIGINT,event_id BIGINT,history_size BIGINT,is_target BOOLEAN,period VARCHAR)")
            db.executemany("INSERT INTO rows VALUES (?,?,?,?,?,?,?)",[
                (0,2,1,1,0,False,"train"),(0,3,2,2,1,True,"train"),
                (0,4,10,3,2,True,"test"),(0,5,10,4,2,True,"test"),(0,2,11,5,4,True,"test"),
                (1,3,1,6,0,False,"train"),(1,2,10,7,1,True,"test")])
            for name in periods:
                db.execute("COPY (SELECT * EXCLUDE(period) FROM rows WHERE period=?) TO ? (FORMAT PARQUET)",[name,str(root/f"events/{name}.parquet")])
            with self.assertRaises(ValueError):load_requests(root,["test"],4)
            stream=request_stream(root,["test"],4)
            np.testing.assert_array_equal(stream["queries"],[[0,0,2,3],[0,0,0,3],[2,3,4,5]])
            np.testing.assert_array_equal(stream["target_request"],[0,0,2,1])
            np.testing.assert_array_equal(stream["targets"],[4,5,2,2])
            self.assertTrue((stream["last_input_timestamp"]<stream["timestamp"]).all())

    def test_changed_protocol_blocks_final_access(self):
        with patch("final_holdout.subprocess.run",return_value=SimpleNamespace(stdout=b"{}")):
            with self.assertRaisesRegex(ValueError,"committed version"):verified_protocol()


if __name__=="__main__":unittest.main()
