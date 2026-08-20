from __future__ import annotations

import copy
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
            self.assertAlmostEqual(total, 1.0, places=6)
        loss = aggregated_policy_kl(logits, teacher, temperature=1.0)
        self.assertAlmostEqual(float(loss), 0.0, places=5)

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


if __name__ == "__main__":
    unittest.main()
