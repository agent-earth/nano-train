from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from nano_train.synthetic_quality import (
    FAMILIES,
    build_cases,
    case_contract,
    load_config,
    paired_comparison,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/evaluation/qwen35_synthetic_quality_v1.json"


class SyntheticQualityTests(unittest.TestCase):
    def test_cases_are_deterministic_balanced_and_fresh_range(self):
        config = load_config(CONFIG)
        first = build_cases(config)
        second = build_cases(config)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 96)
        self.assertEqual(
            {
                family: sum(row["family"] == family for row in first)
                for family in FAMILIES
            },
            dict.fromkeys(FAMILIES, 24),
        )
        self.assertEqual(len({row["case_id"] for row in first}), 96)
        self.assertEqual(len({row["prompt"] for row in first}), 96)
        contract = case_contract(first)
        self.assertEqual(contract["case_count"], 96)
        self.assertNotIn("prompt", contract["cases"][0])
        self.assertNotIn("expected", contract["cases"][0])

    def test_config_freezes_arm_order_and_generation(self):
        config = load_config(CONFIG)
        self.assertEqual(
            [arm["arm_id"] for arm in config.model_arms],
            ["base4", "rl4", "opd4", "base9"],
        )
        self.assertEqual(config.batch_size, 8)
        self.assertEqual(config.max_new_tokens, 32)
        self.assertEqual(config.temperature, 0.0)
        self.assertFalse(config.policy["training_eligible"])

    def test_config_rejects_posthoc_case_or_budget_change(self):
        raw = json.loads(CONFIG.read_text(encoding="utf-8"))
        for key, value, error in (
            ("cases_per_family", 25, "cases_per_family"),
            ("batch_size", 4, "batch_size"),
            ("max_new_tokens", 64, "max_new_tokens"),
        ):
            with self.subTest(key=key):
                altered = copy.deepcopy(raw)
                altered[key] = value
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "config.json"
                    path.write_text(json.dumps(altered), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, error):
                        load_config(path)

    def test_paired_comparison_detects_clear_gain(self):
        baseline = [
            {"case_id": f"case-{index}", "correct": index < 40}
            for index in range(96)
        ]
        candidate = [
            {"case_id": f"case-{index}", "correct": index < 56}
            for index in range(96)
        ]
        comparison = paired_comparison(
            candidate,
            baseline,
            bootstrap_samples=10_000,
            bootstrap_seed=20260820,
        )
        self.assertAlmostEqual(comparison["delta"], 16 / 96)
        self.assertGreater(comparison["paired_bootstrap_95_ci"][0], 0)
        self.assertLess(comparison["mcnemar_exact_p"], 0.05)
        self.assertEqual(
            comparison["paired_counts"],
            {
                "candidate_only": 16,
                "baseline_only": 0,
                "both_correct": 40,
                "both_wrong": 40,
            },
        )

    def test_paired_comparison_rejects_case_mismatch(self):
        with self.assertRaisesRegex(ValueError, "case sets differ"):
            paired_comparison(
                [{"case_id": "a", "correct": True}],
                [{"case_id": "b", "correct": True}],
                bootstrap_samples=10,
                bootstrap_seed=1,
            )


if __name__ == "__main__":
    unittest.main()
