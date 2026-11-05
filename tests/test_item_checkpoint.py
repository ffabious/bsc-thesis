import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_sasrec import make_model, optimizer_for, step
from sasrec import SASRec


class ItemCheckpointTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "manifest.json").write_text('{"candidate_items": 8}')
        source = SASRec(8, length=4, dim=8, heads=2, blocks=1, dropout=0)
        self.weights = {k: v.detach().clone() for k, v in source.state_dict().items()}
        path = self.root / "best.pt"
        torch.save({"weights": self.weights, "items": 8}, path)
        checksum = hashlib.sha256(path.read_bytes()).hexdigest()
        (self.root / "training.json").write_text(json.dumps({
            "checkpoint_sha256": checksum,
            "data_manifest_sha256": hashlib.sha256((self.root / "manifest.json").read_bytes()).hexdigest()}))
        self.config = {"item_checkpoint": str(path), "item_checkpoint_sha256": checksum,
                       "freeze_items": True, "length": 4, "dim": 8, "heads": 2,
                       "blocks": 1, "dropout": 0, "learning_rate": 0.01}

    def test_only_item_table_transfers_and_stays_frozen(self):
        torch.manual_seed(42)
        fresh = SASRec(8, length=4, dim=8, heads=2, blocks=1, dropout=0)
        torch.manual_seed(42)
        model = make_model(self.config, self.root, 8)
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, self.weights[name] if name == "item.weight"
                                        else fresh.state_dict()[name]))
        before = model.position.weight.detach().clone()
        optimizer = optimizer_for(model, self.config)
        step(model, optimizer, torch.tensor([[0, 2, 3, 4]]), torch.tensor([5]), torch.tensor([[6, 7]]))
        self.assertTrue(torch.equal(model.item.weight, self.weights["item.weight"]))
        self.assertIsNone(model.item.weight.grad)
        self.assertFalse(torch.equal(model.position.weight, before))

    def test_rejects_changed_source_or_dataset(self):
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            make_model(dict(self.config, item_checkpoint_sha256="wrong"), self.root, 8)
        (self.root / "manifest.json").write_text('{"candidate_items": 9}')
        with self.assertRaisesRegex(ValueError, "different dataset"):
            make_model(self.config, self.root, 8)

    def test_rejects_ambiguous_source_and_dimension_mismatch(self):
        with self.assertRaisesRegex(ValueError, "exactly one"):
            make_model(dict(self.config, item_features="audio"), self.root, 8)
        with self.assertRaisesRegex(ValueError, "shape/values"):
            make_model(dict(self.config, item_dim=16), self.root, 8)


if __name__ == "__main__":
    unittest.main()
