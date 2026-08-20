from __future__ import annotations

import copy
import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path

import torch

from nano_train.anchor_policy_replay import (
    aggregated_policy_kl,
    build_dataset,
    build_step_schedule,
    compress_policy,
    contamination_audit,
    load_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs/anchor_policy_replay/"
    "qwen35_anchor_policy_replay_v1.json"
)
RENDER_PATH = ROOT / "scripts/render_anchor_policy_replay_v1.py"
RENDER_SPEC = importlib.util.spec_from_file_location(
    "render_anchor_policy_replay_v1",
    RENDER_PATH,
)
RENDER_MODULE = importlib.util.module_from_spec(RENDER_SPEC)
assert RENDER_SPEC.loader is not None
RENDER_SPEC.loader.exec_module(RENDER_MODULE)


class AnchorPolicyReplayTests(unittest.TestCase):
    def test_dataset_is_balanced_disjoint_and_deterministic(self):
        config = load_config(CONFIG)
        first = build_dataset(config)
        second = build_dataset(config)
        self.assertEqual(first, second)
        self.assertEqual(len(first["train_pairs"]), 256)
        self.assertEqual(len(first["dev_pairs"]), 256)
        for key in ("train_pairs", "dev_pairs"):
            counts = {}
            for row in first[key]:
                counts[row["family"]] = counts.get(row["family"], 0) + 1
            self.assertEqual(set(counts.values()), {64})
        self.assertEqual(
            {row["expression"] for row in first["train_pairs"]}
            & {row["expression"] for row in first["dev_pairs"]},
            set(),
        )

    def test_schedules_isolate_anchor_policy_kl(self):
        config = load_config(CONFIG)
        dataset = build_dataset(config)
        control = build_step_schedule(config, dataset, "control")
        treatment = build_step_schedule(config, dataset, "treatment")
        self.assertEqual(len(control), 512)
        self.assertEqual(len(treatment), 512)
        self.assertEqual(
            [row["pair_id"] for row in control],
            [row["pair_id"] for row in treatment],
        )
        self.assertEqual(
            [row["kind"] for row in control[::2]],
            ["full_consistency"] * 256,
        )
        self.assertEqual(
            [row["kind"] for row in treatment[::2]],
            ["full_consistency"] * 256,
        )
        self.assertEqual(
            [row["kind"] for row in control[1::2]],
            ["final_ce_only"] * 256,
        )
        self.assertEqual(
            [row["kind"] for row in treatment[1::2]],
            ["final_ce_plus_anchor_policy_kl"] * 256,
        )

    def test_compressed_policy_normalizes_and_identical_kl_is_zero(self):
        torch.manual_seed(7)
        logits = torch.randn(3, 128, dtype=torch.float32)
        teacher = compress_policy(logits, top_k=16, temperature=1.0)
        for row in teacher:
            total = sum(math.exp(value) for value in row["top_logprobs"])
            total += math.exp(row["other_logprob"])
            self.assertAlmostEqual(total, 1.0, places=12)
        loss = aggregated_policy_kl(logits, teacher, temperature=1.0)
        self.assertAlmostEqual(float(loss), 0.0, places=5)

    def test_compressed_policy_stays_normalized_when_top_mass_is_near_one(self):
        logits = torch.full((2, 256), -30.0, dtype=torch.float32)
        logits[:, :16] = torch.linspace(30.0, 15.0, 16)
        teacher = compress_policy(logits, top_k=16, temperature=1.0)
        for row in teacher:
            top_mass = sum(
                math.exp(value) for value in row["top_logprobs"]
            )
            other_mass = math.exp(row["other_logprob"])
            self.assertLess(top_mass, 1.0)
            self.assertGreater(other_mass, 0.0)
            self.assertAlmostEqual(top_mass + other_mass, 1.0, places=12)

    def test_policy_kl_is_positive_and_has_finite_gradient(self):
        torch.manual_seed(11)
        teacher_logits = torch.randn(2, 96, dtype=torch.float32)
        teacher = compress_policy(
            teacher_logits,
            top_k=12,
            temperature=1.0,
        )
        student = (teacher_logits + 0.5 * torch.randn_like(teacher_logits))
        student.requires_grad_(True)
        loss = aggregated_policy_kl(student, teacher, temperature=1.0)
        self.assertGreater(float(loss.detach()), 0)
        loss.backward()
        self.assertIsNotNone(student.grad)
        self.assertTrue(torch.isfinite(student.grad).all())
        self.assertGreater(float(student.grad.abs().sum()), 0)

    def test_contamination_audit_passes_all_prior_surfaces(self):
        config = load_config(CONFIG)
        audit = contamination_audit(config, build_dataset(config))
        self.assertTrue(audit["passed"])
        self.assertEqual(audit["observed_quality_prompt_overlap"], 0)
        self.assertEqual(audit["benchmark_prompt_overlap"], 0)
        self.assertEqual(
            audit["benchmark_rows_hashed"],
            {"gsm8k": 1319, "mmlu": 14042, "gpqa_diamond": 198},
        )

    def test_config_rejects_posthoc_changes(self):
        raw = json.loads(CONFIG.read_text(encoding="utf-8"))
        for key, value, error in (
            ("train_range_offset", 61000, "train_range_offset"),
            ("max_steps_per_arm", 256, "max_steps_per_arm"),
            ("anchor_policy_top_k", 32, "anchor_policy_top_k"),
            (
                "treatment_anchor_policy_kl_weight",
                0.5,
                "treatment_anchor_policy_kl_weight",
            ),
            (
                "treatment_second_step",
                "final_ce_only",
                "treatment_second_step",
            ),
        ):
            with self.subTest(key=key):
                altered = copy.deepcopy(raw)
                altered[key] = value
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "config.json"
                    path.write_text(json.dumps(altered), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, error):
                        load_config(path)

    def test_gate_requires_zero_losses_and_fewer_than_control(self):
        family_metrics = {
            family: {"cases": 64, "correct": 1, "parse_failures": 0}
            for family in (
                "exact_division",
                "mixed_products",
                "nested_offset",
                "repeated_operand",
            )
        }
        baseline = {
            "accuracy": 4 / 256,
            "parse_failures": 0,
            "by_family": copy.deepcopy(family_metrics),
        }
        treatment_post = copy.deepcopy(baseline)
        treatment_post["accuracy"] = 16 / 256
        for row in treatment_post["by_family"].values():
            row["correct"] = 4
        control_post = copy.deepcopy(baseline)
        control_post["accuracy"] = 14 / 256
        metrics = {
            "training": {
                "all_components_finite": True,
                "loss_curve": [
                    {
                        "anchor_policy_kl": 0.1,
                        "objective": 1.0,
                        "gradient_norm": 1.0,
                    }
                ],
            },
            "failure_receipt_exists": False,
            "identity": {
                "adapter_sha256": "a" * 64,
                "teacher_cache_sha256": "c" * 64,
            },
            "baseline_dev": baseline,
            "post_dev": treatment_post,
            "comparison": {
                "candidate_accuracy": 16 / 256,
                "baseline_accuracy": 4 / 256,
                "paired_bootstrap_95_ci": [1 / 256, 20 / 256],
                "mcnemar_exact_p": 0.001,
                "paired_counts": {
                    "candidate_only": 12,
                    "baseline_only": 0,
                },
            },
        }
        treatment = copy.deepcopy(metrics)
        control = copy.deepcopy(metrics)
        control["post_dev"] = control_post
        control["comparison"]["candidate_accuracy"] = 14 / 256
        control["comparison"]["paired_counts"] = {
            "candidate_only": 12,
            "baseline_only": 2,
        }
        reload = {
            "reload_success": True,
            "metrics_exact": True,
            "generations_exact": True,
            "adapter_sha256": "a" * 64,
        }
        cache_receipt = {
            "identity": {"teacher_cache_sha256": "c" * 64},
            "summary": {
                "all_probabilities_finite": True,
                "all_probability_sums_within_1e_5": True,
            },
        }
        gates = RENDER_MODULE.admission_gates(
            control,
            treatment,
            {"control": reload, "treatment": reload},
            cache_receipt,
        )
        self.assertTrue(all(gates.values()))
        treatment["comparison"]["paired_counts"]["baseline_only"] = 1
        gates = RENDER_MODULE.admission_gates(
            control,
            treatment,
            {"control": reload, "treatment": reload},
            cache_receipt,
        )
        self.assertFalse(gates["treatment_anchor_maximum_losses"])
        self.assertTrue(gates["treatment_losses_lt_control"])


if __name__ == "__main__":
    unittest.main()
