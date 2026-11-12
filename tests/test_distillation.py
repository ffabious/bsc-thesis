import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import run_sasrec
from blend_sasrec import normalize_scores
from distill_sasrec import DistillationStep,Teacher,objectives
from distill_sasrec import lock_selection,load_teachers
from sasrec import SASRec


class DistillationTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(21)

    def models(self):
        return {k:SASRec(12,length=3,dim=8,heads=2,blocks=1,dropout=0).eval() for k in ["long","recent"]}

    def test_teacher_uses_native_recent_and_preserves_ensemble_ranks(self):
        models = self.models();teacher = Teacher(models,.4,2)
        x = torch.tensor([[0,2,3],[4,5,5],[0,0,6]])
        with torch.no_grad():
            long = models["long"](x)@models["long"].item.weight[2:].T
            recent = models["recent"](x[:,-2:])@models["recent"].item.weight[2:].T
            scale = long.std(dim=1,unbiased=False,keepdim=True)
            expected = (.6*normalize_scores(long)+.4*normalize_scores(recent))*scale
        shapes = []
        hook = models["recent"].register_forward_pre_hook(lambda module,args:shapes.append(tuple(args[0].shape)))
        actual = teacher.logits(x);hook.remove()
        torch.testing.assert_close(actual,expected)
        self.assertEqual(shapes,[(3,2)])
        self.assertFalse(teacher.targets(x).requires_grad)
        with torch.no_grad():
            torch.testing.assert_close(Teacher(models,0).logits(x),models["long"](x)@models["long"].item.weight[2:].T)

    def test_loss_and_gradients_match_explicit_kl_and_zero_weight_ce(self):
        for temperature in [1,2]:
            logits = torch.randn(3,5,requires_grad=True);target = torch.tensor([2,4,6])
            logp = F.log_softmax(torch.randn(3,5)/temperature,dim=1)
            actual,ce,kd = objectives(logits,target,logp,.25,temperature)
            expected_kd = (logp.exp()*(logp-F.log_softmax(logits/temperature,dim=1))).sum()/3*temperature**2
            expected = F.cross_entropy(logits,target-2)+.25*expected_kd
            torch.testing.assert_close(actual,expected);torch.testing.assert_close(kd,expected_kd)
            torch.testing.assert_close(torch.autograd.grad(actual,logits,retain_graph=True)[0],torch.autograd.grad(expected,logits,retain_graph=True)[0])
            zero,_,_ = objectives(logits,target,logp,0,temperature)
            torch.testing.assert_close(zero,ce)
            torch.testing.assert_close(torch.autograd.grad(zero,logits,retain_graph=True)[0],torch.autograd.grad(ce,logits)[0])

    def test_single_teacher_loads_no_recent_branch_and_uses_raw_ce_logits(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            root=Path(tmp);(root/"manifest.json").write_text('{"candidate_items":12}')
            folder=root/"train-mps";folder.mkdir()
            config={"data":str(root.relative_to(run_sasrec.ROOT)),"length":3,"dim":8,"heads":2,"blocks":1,"dropout":0}
            model=SASRec(12,**run_sasrec.model_args(config))
            torch.save({"weights":model.state_dict(),"config":config,"items":12,"epoch":4},folder/"best.pt")
            trained={"checkpoint_sha256":hashlib.sha256((folder/"best.pt").read_bytes()).hexdigest(),"config":config,
                "data_manifest_sha256":hashlib.sha256((root/"manifest.json").read_bytes()).hexdigest(),"best_epoch":4}
            (folder/"training.json").write_text(json.dumps(trained))
            protocol={"teacher":{"runs":{"256":{"long":str(root.relative_to(run_sasrec.ROOT))}}},"fixed_teacher_alpha":0}
            with patch("distill_sasrec.load_pair",side_effect=AssertionError("Unused pair must not load")):
                models,configs,sources=load_teachers(protocol,"cpu")
            self.assertEqual(set(models),{"long"});self.assertEqual(set(sources),{"long"})
            x=torch.tensor([[0,2,3],[4,5,5]])
            for temperature in [1,2]:
                teacher=Teacher(models,0,temperature)
                with torch.no_grad():raw=models["long"](x)@models["long"].item.weight[2:].T
                torch.testing.assert_close(teacher.logits(x),raw,atol=0,rtol=0)
                torch.testing.assert_close(teacher.targets(x),F.log_softmax(raw/temperature,dim=1),atol=0,rtol=0)
            self.assertTrue(all(p.grad is None and not p.requires_grad for p in models["long"].parameters()))

    def test_training_updates_student_but_not_teacher(self):
        models = self.models();teacher = Teacher(models,.4)
        before = {role:{k:v.clone() for k,v in m.state_dict().items()} for role,m in models.items()}
        student = copy.deepcopy(models["long"]).requires_grad_(True)
        optimizer = torch.optim.Adam(student.parameters(),lr=.001,betas=(.9,.98))
        update = DistillationStep(teacher,.25)
        value = update(student,optimizer,torch.tensor([[0,2,3],[4,5,5]]),torch.tensor([4,6]),None,"joint_mean","full_ce")
        self.assertTrue(torch.isfinite(value));self.assertFalse(torch.equal(student.item.weight,models["long"].item.weight))
        for role,m in models.items():
            for k,v in m.state_dict().items():self.assertTrue(torch.equal(v,before[role][k]))
            self.assertTrue(all(p.grad is None for p in m.parameters()))
        summary = update.summary()
        self.assertAlmostEqual(summary["ce"]+summary["weighted_kl_t2"],value.item(),places=5)

    def test_warm_start_loads_all_weights_and_keeps_epoch_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp);(root/"manifest.json").write_text('{"candidate_items":4}')
            config = {"length":3,"dim":8,"heads":2,"blocks":1,"dropout":0,"seed":42,"batch_size":2,
                "negatives":2,"learning_rate":.001,"eval_batch_size":2,"epochs":None,"patience":2,"objective":"full_ce"}
            original = SASRec(4,**run_sasrec.model_args(config));folder = root/"source";folder.mkdir()
            path = folder/"best.pt";torch.save({"weights":original.state_dict(),"config":config,"items":4,"epoch":4},path)
            h = hashlib.sha256(path.read_bytes()).hexdigest()
            (folder/"training.json").write_text(json.dumps({"checkpoint_sha256":h,
                "data_manifest_sha256":hashlib.sha256((root/"manifest.json").read_bytes()).hexdigest()}))
            config.update(initial_checkpoint=str(path),initial_checkpoint_sha256=h)
            loaded = run_sasrec.make_model(config,root,4)
            for k,v in original.state_dict().items():self.assertTrue(torch.equal(v,loaded.state_dict()[k]))
            arrays = np.array([[0,2,3],[0,3,4]],dtype=np.int32),np.array([4,5]),np.array([0,1])
            with patch.object(run_sasrec,"examples",return_value=arrays),patch.object(run_sasrec,"most_popular",return_value=np.array([2,3])), \
                 patch.object(run_sasrec,"ranks",side_effect=[np.array([r,r]) for r in [1,2,3]]):
                result = run_sasrec.train(config,root,"cpu",root/"trained")
            self.assertEqual(result["best_epoch"],0);self.assertEqual(len(result["epochs"]),2)
            self.assertEqual(result["stop_reason"],"validation_patience")
            best = torch.load(root/"trained/best.pt",weights_only=True)
            for k,v in original.state_dict().items():self.assertTrue(torch.equal(v,best["weights"][k]))
            config["initial_checkpoint_sha256"] = "wrong"
            with self.assertRaisesRegex(ValueError,"Initial checkpoint checksum"):
                run_sasrec.make_model(config,root,4)

    def test_selection_exports_relative_output_paths_before_holdout_reads(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            root=Path(tmp);(root/"teacher").mkdir();(root/"teacher/selection.json").write_text("{}")
            for name,weight in [("control",0),("kd-0.25",.25),("kd-1",1)]:
                folder=root/name/"train-mps";folder.mkdir(parents=True);(folder/"best.pt").write_bytes(b"checkpoint")
                data={"stop_reason":"validation_patience","config":{"distillation_weight":weight},"best_epoch":0,
                    "initial_validation":{"ndcg@10":.1,"recall@100":.2},
                    "checkpoint_sha256":hashlib.sha256((folder/"best.pt").read_bytes()).hexdigest()}
                (folder/"training.json").write_text(json.dumps(data))
            selected=lock_selection({"weights":[0,.25,1]},root.relative_to(Path.cwd()))
            self.assertEqual(selected["student"]["weight"],.25)
            self.assertTrue(selected["locked_before_holdout_read"])
            self.assertTrue((run_sasrec.ROOT/selected["student"]["run"]).is_absolute())

    def test_ready_control_reused_without_a_new_control_training(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            root=Path(tmp);(root/"teacher").mkdir();(root/"teacher/selection.json").write_text("{}")
            for name,weight in [("ready-control",0),("kd-1",1)]:
                folder=root/name/"train-mps";folder.mkdir(parents=True);(folder/"best.pt").write_bytes(b"checkpoint")
                data={"stop_reason":"validation_patience","config":{} if weight==0 else {"distillation_weight":weight},
                    "best_epoch":1,"initial_validation":{"ndcg@10":.01,"recall@100":.02},
                    "epochs":[{"validation":{"ndcg@10":.1,"recall@100":.2}}],
                    "checkpoint_sha256":hashlib.sha256((folder/"best.pt").read_bytes()).hexdigest()}
                (folder/"training.json").write_text(json.dumps(data))
            protocol={"weights":[1],"control_run":str((root/"ready-control/train-mps").relative_to(run_sasrec.ROOT)),
                      "control_checkpoint_sha256":hashlib.sha256(b"checkpoint").hexdigest()}
            selected=lock_selection(protocol,root)
            self.assertTrue(selected["control"]["reused"])
            self.assertEqual(len(selected["candidates"]),2)
            self.assertEqual(selected["student"]["weight"],1)

    def test_cold_state_rng_guard_does_not_enable_warm_epoch_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/"manifest.json").write_text('{"candidate_items":4}')
            config={"length":3,"dim":8,"heads":2,"blocks":1,"dropout":0,"seed":42,"batch_size":2,
                    "negatives":2,"learning_rate":.001,"eval_batch_size":2,"epochs":None,"patience":2,"objective":"full_ce"}
            torch.manual_seed(42);model=SASRec(4,**run_sasrec.model_args(config))
            config.update(expected_initial_state_sha256=run_sasrec.state_digest(model),
                          expected_initial_rng_sha256=hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest())
            arrays=np.array([[0,2,3],[0,3,4]],dtype=np.int32),np.array([4,5]),np.array([0,1])
            with patch.object(run_sasrec,"examples",return_value=arrays),patch.object(run_sasrec,"most_popular",return_value=np.array([2,3])):
                torch.manual_seed(42)
                with patch.object(run_sasrec,"ranks",side_effect=[np.array([r,r]) for r in [1,2,3,4]]):
                    result=run_sasrec.train(config,root,"cpu",root/"trained")
                self.assertEqual(result["best_epoch"],1)
                self.assertEqual(result["initial_state_sha256"],config["expected_initial_state_sha256"])
                config["expected_initial_state_sha256"]="wrong";torch.manual_seed(42)
                with self.assertRaisesRegex(ValueError,"Initial model state differs"):
                    run_sasrec.train(config,root,"cpu",root/"bad")

    @unittest.skipUnless(torch.backends.mps.is_available(),"MPS unavailable")
    def test_cpu_mps_distillation_gradients_agree(self):
        models = self.models();student = copy.deepcopy(models["long"]).requires_grad_(True)
        other = copy.deepcopy(student).to("mps")
        updates = [DistillationStep(Teacher(models,.4),.25),
                   DistillationStep(Teacher({k:copy.deepcopy(m).to("mps") for k,m in models.items()},.4),.25)]
        x,y = torch.tensor([[0,2,3],[4,5,5]]),torch.tensor([4,6]);values = []
        for model,update,device in zip([student,other],updates,["cpu","mps"]):
            optimizer = torch.optim.Adam(model.parameters(),lr=.001,betas=(.9,.98))
            values.append(update(model,optimizer,x.to(device),y.to(device),None,"joint_mean","full_ce"))
        torch.testing.assert_close(values[0],values[1].cpu(),atol=1e-5,rtol=1e-4)
        for a,b in zip(student.parameters(),other.parameters()):
            torch.testing.assert_close(a.grad,b.grad.cpu(),atol=1e-5,rtol=1e-3)


if __name__=="__main__":unittest.main()
