from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from nano_train.scaled_quality import (
    build_dataset,
    load_config,
    public_dataset_contract,
)
from nano_train.synthetic_quality import (
    build_cases as build_forbidden_cases,
    load_config as load_forbidden_config,
)
from scripts.render_scaled_quality_sft_v1 import acceptance_gates


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/scaled_quality/qwen35_scaled_quality_sft_v1.json"
FORBIDDEN = ROOT / "configs/evaluation/qwen35_synthetic_quality_v1.json"


class ScaledQualityTests(unittest.TestCase):
    def test_dataset_is_balanced_disjoint_and_deterministic(self):
        config = load_config(CONFIG)
        first = build_dataset(config)
        second = build_dataset(config)
        self.assertEqual(first, second)
        self.assertEqual(len(first["train"]), 512)
        self.assertEqual(len(first["dev"]), 96)
        self.assertEqual(
            {row["expression"] for row in first["train"]}
            & {row["expression"] for row in first["dev"]},
            set(),
        )
        for split, expected in (("train", 128), ("dev", 24)):
            counts = {}
            for row in first[split]:
                counts[row["family"]] = counts.get(row["family"], 0) + 1
            self.assertEqual(set(counts.values()), {expected})

    def test_dataset_excludes_observed_quality_prompts(self):
        config = load_config(CONFIG)
        dataset = build_dataset(config)
        forbidden = build_forbidden_cases(load_forbidden_config(FORBIDDEN))
        actual = {
            row["prompt"] for row in (*dataset["train"], *dataset["dev"])
        }
        observed = {row["prompt"] for row in forbidden}
        self.assertEqual(actual & observed, set())

    def test_public_contract_excludes_expression_and_target(self):
        contract = public_dataset_contract(build_dataset(load_config(CONFIG)))
        for split in ("train", "dev"):
            self.assertTrue(contract[split])
            self.assertEqual(
                set(contract[split][0]),
                {"case_id", "family", "prompt_sha256", "target_sha256"},
            )

    def test_config_freezes_exact_one_epoch_exposure(self):
        config = load_config(CONFIG)
        self.assertEqual(config.max_steps, 128)
        self.assertEqual(config.batch_size, 4)
        self.assertEqual(
            config.max_steps * config.batch_size,
            config.train_cases_per_family * 4,
        )
        self.assertEqual(config.learning_rate, 0.00005)
        self.assertFalse(config.policy["benchmark_access_after_result"])

    def test_config_rejects_posthoc_scale_change(self):
        raw = json.loads(CONFIG.read_text(encoding="utf-8"))
        for key, value, error in (
            ("max_steps", 64, "max_steps"),
            ("batch_size", 2, "batch_size"),
            ("learning_rate", 0.0001, "learning_rate"),
            ("dev_range_offset", 4000, "dev_range_offset"),
        ):
            with self.subTest(key=key):
                altered = copy.deepcopy(raw)
                altered[key] = value
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "config.json"
                    path.write_text(json.dumps(altered), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, error):
                        load_config(path)

    def test_result_gate_rejects_directional_two_win_result(self):
        by_family = {
            family: {"correct": 0, "cases": 24, "parse_failures": 0}
            for family in (
                "repeated_operand",
                "mixed_products",
                "exact_division",
                "nested_offset",
            )
        }
        post_by_family = copy.deepcopy(by_family)
        post_by_family["repeated_operand"]["correct"] = 1
        post_by_family["exact_division"]["correct"] = 1
        metrics = {
            "training": {
                "all_losses_finite": True,
                "all_gradient_norms_finite": True,
            },
            "failure_receipt_exists": False,
            "comparison": {
                "baseline_accuracy": 0.0,
                "candidate_accuracy": 2 / 96,
                "paired_bootstrap_95_ci": [0.0, 5 / 96],
                "mcnemar_exact_p": 0.5,
                "paired_counts": {
                    "candidate_only": 2,
                    "baseline_only": 0,
                },
            },
            "baseline_dev": {
                "by_family": by_family,
                "parse_failures": 0,
            },
            "post_sft_dev": {
                "by_family": post_by_family,
                "parse_failures": 0,
            },
            "identity": {"adapter_sha256": "a" * 64},
        }
        reload = {
            "reload_success": True,
            "metrics_exact": True,
            "generations_exact": True,
            "adapter_sha256": "a" * 64,
        }
        gates = acceptance_gates(metrics, reload)
        self.assertTrue(gates["post_accuracy_gt_baseline"])
        self.assertTrue(gates["maximum_baseline_only_losses"])
        self.assertFalse(gates["paired_bootstrap_ci_lower_gt_zero"])
        self.assertFalse(gates["exact_mcnemar_p_lt_005"])
        self.assertFalse(gates["minimum_candidate_only_wins"])
        self.assertFalse(all(gates.values()))


if __name__ == "__main__":
    unittest.main()
