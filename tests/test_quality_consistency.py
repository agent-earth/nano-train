from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from nano_train.quality_consistency import (
    benchmark_prompt_hashes,
    build_dataset,
    dataset_prompt_hashes,
    forbidden_prompt_hashes,
    load_config,
    normalized_dataset_prompt_hashes,
    public_contract,
)
from scripts.render_quality_consistency_v1 import acceptance_gates


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT / "configs/quality_consistency/qwen35_quality_consistency_v1.json"
)


class QualityConsistencyTests(unittest.TestCase):
    def test_dataset_is_balanced_disjoint_and_deterministic(self):
        config = load_config(CONFIG)
        first = build_dataset(config)
        second = build_dataset(config)
        self.assertEqual(first, second)
        self.assertEqual(len(first["train_pairs"]), 256)
        self.assertEqual(len(first["dev_pairs"]), 192)
        for key, expected in (("train_pairs", 64), ("dev_pairs", 48)):
            counts = {}
            for row in first[key]:
                counts[row["family"]] = counts.get(row["family"], 0) + 1
            self.assertEqual(set(counts.values()), {expected})
        self.assertEqual(
            {row["expression"] for row in first["train_pairs"]}
            & {row["expression"] for row in first["dev_pairs"]},
            set(),
        )

    def test_dataset_excludes_all_observed_quality_prompts(self):
        config = load_config(CONFIG)
        self.assertEqual(
            dataset_prompt_hashes(build_dataset(config))
            & forbidden_prompt_hashes(config),
            set(),
        )
        benchmark, counts = benchmark_prompt_hashes(config)
        self.assertEqual(
            counts,
            {"gsm8k": 1319, "mmlu": 14042, "gpqa_diamond": 198},
        )
        self.assertEqual(
            normalized_dataset_prompt_hashes(build_dataset(config))
            & benchmark,
            set(),
        )

    def test_public_contract_excludes_raw_content(self):
        contract = public_contract(build_dataset(load_config(CONFIG)))
        for key in ("train_pairs", "dev_pairs"):
            self.assertTrue(contract[key])
            self.assertEqual(
                set(contract[key][0]),
                {
                    "pair_id",
                    "family",
                    "process_prompt_sha256",
                    "final_prompt_sha256",
                    "process_target_sha256",
                    "final_target_sha256",
                },
            )

    def test_config_freezes_changed_mechanism(self):
        config = load_config(CONFIG)
        self.assertEqual(config.max_steps, 256)
        self.assertEqual(config.train_pairs_per_family, 64)
        self.assertEqual(config.process_ce_weight, 0.5)
        self.assertEqual(config.final_ce_weight, 0.5)
        self.assertEqual(config.consistency_weight, 1.0)
        self.assertTrue(config.teacher_detach)
        self.assertFalse(config.policy["benchmark_access_after_result"])

    def test_config_rejects_posthoc_method_change(self):
        raw = json.loads(CONFIG.read_text(encoding="utf-8"))
        for key, value, error in (
            ("max_steps", 128, "max_steps"),
            ("learning_rate", 0.0001, "learning_rate"),
            ("consistency_weight", 0.5, "consistency_weight"),
            ("teacher_detach", False, "teacher_detach"),
        ):
            with self.subTest(key=key):
                altered = copy.deepcopy(raw)
                altered[key] = value
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "config.json"
                    path.write_text(json.dumps(altered), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, error):
                        load_config(path)

    def test_gate_rejects_significant_gain_with_two_losses(self):
        by_family = {
            family: {"correct": 0, "cases": 48, "parse_failures": 0}
            for family in (
                "repeated_operand",
                "mixed_products",
                "exact_division",
                "nested_offset",
            )
        }
        baseline_family = copy.deepcopy(by_family)
        baseline_family["repeated_operand"]["correct"] = 2
        post_family = copy.deepcopy(by_family)
        post_family["repeated_operand"]["correct"] = 5
        post_family["exact_division"]["correct"] = 10
        metrics = {
            "training": {"all_components_finite": True},
            "failure_receipt_exists": False,
            "comparison": {
                "baseline_accuracy": 2 / 192,
                "candidate_accuracy": 15 / 192,
                "paired_bootstrap_95_ci": [5 / 192, 21 / 192],
                "mcnemar_exact_p": 0.002349853515625,
                "paired_counts": {
                    "candidate_only": 15,
                    "baseline_only": 2,
                },
            },
            "baseline_dev": {
                "by_family": baseline_family,
                "parse_failures": 0,
            },
            "post_dev": {
                "by_family": post_family,
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
        self.assertTrue(gates["paired_bootstrap_ci_lower_gt_zero"])
        self.assertTrue(gates["exact_mcnemar_p_lt_005"])
        self.assertTrue(gates["minimum_candidate_only_wins"])
        self.assertFalse(gates["maximum_baseline_only_losses"])
        self.assertFalse(all(gates.values()))


if __name__ == "__main__":
    unittest.main()
