from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from nano_train.rl_opd_admission import (
    distillation_kl,
    load_config,
    reinforce_loss,
    rollout_prediction_logits,
    validate_config,
    verifier_reward,
)
from scripts.preregister_rl_opd_admission_v1 import build_receipt
from scripts.render_rl_opd_admission_v1 import admission_gates


ROOT = Path(__file__).resolve().parents[1]
RL_CONFIG = ROOT / "configs/admission/qwen35_4b_rl_admission_v1.json"
OPD_CONFIG = ROOT / "configs/admission/qwen35_4b_opd_admission_v1.json"


class RLOPDAdmissionTests(unittest.TestCase):
    def test_configs_freeze_distinct_rl_and_opd_contracts(self):
        rl = load_config(RL_CONFIG)
        opd = load_config(OPD_CONFIG)
        self.assertEqual(rl.mode, "rl")
        self.assertEqual(opd.mode, "opd")
        self.assertIsNone(rl.teacher_model_path)
        self.assertEqual(opd.teacher_model_path, "../../../models/Qwen3.5-9B")
        self.assertEqual(rl.max_steps, 2)
        self.assertEqual(opd.max_steps, 2)
        self.assertEqual(len(rl.train_tasks), rl.max_steps)
        self.assertEqual(len(opd.train_tasks), opd.max_steps)
        self.assertEqual(rl.reference_kl_weight, 0.02)
        self.assertEqual(opd.reference_kl_weight, 0.0)
        self.assertFalse(rl.policy["quality_claim_allowed"])
        self.assertFalse(opd.policy["benchmark_access_allowed_after_smoke"])

    def test_config_rejects_posthoc_method_mutation(self):
        raw = json.loads(RL_CONFIG.read_text(encoding="utf-8"))
        for key, value, message in (
            ("max_steps", 3, "max_steps"),
            ("learning_rate", 0.00002, "learning_rate"),
            ("rollout_temperature", 0.7, "rollout_temperature"),
        ):
            with self.subTest(key=key):
                altered = copy.deepcopy(raw)
                altered[key] = value
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "config.json"
                    path.write_text(json.dumps(altered), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, message):
                        load_config(path)

    def test_config_rejects_benchmark_or_quality_policy_change(self):
        raw = json.loads(OPD_CONFIG.read_text(encoding="utf-8"))
        config = load_config(OPD_CONFIG)
        for key in ("contains_benchmark_rows", "quality_claim_allowed"):
            with self.subTest(key=key):
                altered = copy.deepcopy(config.__dict__)
                altered["policy"] = dict(raw["policy"])
                altered["policy"][key] = True
                with self.assertRaisesRegex(
                    ValueError,
                    "policy differs",
                ):
                    validate_config(type(config)(**altered))

    def test_verifier_reward_is_exact_and_format_sensitive(self):
        self.assertEqual(verifier_reward("FINAL: 42", "42"), 1.0)
        self.assertEqual(verifier_reward("FINAL: 41", "42"), -0.25)
        self.assertEqual(verifier_reward("42", "42"), -1.0)
        self.assertEqual(verifier_reward("FINAL 42", "42"), -1.0)

    def test_rollout_logits_align_next_token_predictions(self):
        logits = torch.arange(1 * 7 * 11).reshape(1, 7, 11)
        selected = rollout_prediction_logits(
            logits,
            prompt_length=4,
            rollout_length=3,
        )
        self.assertTrue(torch.equal(selected, logits[:, 3:6, :]))
        with self.assertRaisesRegex(ValueError, "shape differs"):
            rollout_prediction_logits(
                logits,
                prompt_length=7,
                rollout_length=2,
            )

    def test_reinforce_loss_updates_sampled_token_logits(self):
        logits = torch.randn(1, 3, 7, requires_grad=True)
        rollout = torch.tensor([[1, 2, 3]])
        loss = reinforce_loss(logits, rollout, reward=1.0)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(float(logits.grad.abs().sum()), 0.0)

    def test_reinforce_accepts_cloned_inference_rollout_ids(self):
        with torch.inference_mode():
            inference_rollout = torch.tensor([[1, 2, 3]])
        rollout = inference_rollout.detach().clone()
        logits = torch.randn(1, 3, 7, requires_grad=True)
        loss = reinforce_loss(logits, rollout, reward=-0.25)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_opd_kl_detaches_teacher_and_updates_student(self):
        teacher = torch.randn(1, 3, 7, requires_grad=True)
        student = torch.randn(1, 3, 7, requires_grad=True)
        loss = distillation_kl(teacher, student)
        loss.backward()
        self.assertIsNone(teacher.grad)
        self.assertIsNotNone(student.grad)
        self.assertTrue(torch.isfinite(student.grad).all())
        equal = distillation_kl(teacher.detach(), teacher.detach().clone())
        self.assertAlmostEqual(float(equal), 0.0, places=6)

    def test_preregistration_records_real_token_lengths(self):
        identity = {
            "student": {
                "config_sha256": "a" * 64,
                "index_sha256": "b" * 64,
                "shards": [],
            },
            "teacher": None,
        }
        contamination = {
            "synthetic_prompts": 4,
            "exact_normalized_prompt_overlap": 0,
            "benchmark_labels_loaded": False,
            "benchmark_outputs_loaded": False,
            "passed": True,
        }
        with (
            mock.patch(
                "scripts.preregister_rl_opd_admission_v1.model_identity_checks",
                return_value=identity,
            ),
            mock.patch(
                "scripts.preregister_rl_opd_admission_v1.build_contamination_audit",
                return_value=contamination,
            ),
        ):
            receipt = build_receipt()
        for experiment in receipt["experiments"]:
            lengths = experiment["synthetic_tasks"]["prompt_lengths"]
            self.assertTrue(all(20 <= length <= 128 for length in lengths))
            self.assertEqual(
                experiment["synthetic_tasks"]["maximum_prompt_tokens"],
                max(lengths),
            )
        self.assertFalse(receipt["execution_boundary"]["training_started"])
        self.assertFalse(receipt["acceptance"]["benchmark_allowed"])

    def test_public_admission_gates_fail_closed(self):
        metrics = {
            "training": {
                "optimizer_steps": 2,
                "loss_curve": [{}, {}],
                "all_losses_finite": True,
                "all_gradient_norms_finite": True,
            },
            "adapter_effect": {"logits_changed": True},
            "identity": {"adapter_sha256": "a" * 64},
            "contamination_audit": {
                "passed": True,
                "exact_normalized_prompt_overlap": 0,
                "benchmark_labels_loaded": False,
                "benchmark_outputs_loaded": False,
                "canary_or_holdout_loaded": False,
            },
            "failure_receipt_exists": False,
        }
        reload = {
            "finite_adapter_tensors": 32,
            "nonfinite_adapter_tensors": 0,
            "reload_success": True,
            "probe_logits_exact": True,
            "adapter_sha256": "a" * 64,
        }
        self.assertTrue(
            all(
                admission_gates(
                    metrics,
                    reload,
                    runtime_failure_exists=False,
                ).values()
            )
        )
        altered = copy.deepcopy(metrics)
        altered["contamination_audit"]["benchmark_outputs_loaded"] = True
        self.assertFalse(
            admission_gates(
                altered,
                reload,
                runtime_failure_exists=False,
            )["contamination_audit_passed"]
        )
        self.assertFalse(
            admission_gates(
                metrics,
                reload,
                runtime_failure_exists=True,
            )["runtime_failure_receipt_absent"]
        )


if __name__ == "__main__":
    unittest.main()
