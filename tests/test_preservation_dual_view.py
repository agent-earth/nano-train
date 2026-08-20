from __future__ import annotations

import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from nano_train.preservation_dual_view import (
    build_dataset,
    build_step_schedule,
    contamination_audit,
    load_config,
    public_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs/preservation_dual_view/"
    "qwen35_preservation_dual_view_v1.json"
)
RENDER_PATH = ROOT / "scripts/render_preservation_dual_view_v1.py"
RENDER_SPEC = importlib.util.spec_from_file_location(
    "render_preservation_dual_view_v1",
    RENDER_PATH,
)
RENDER_MODULE = importlib.util.module_from_spec(RENDER_SPEC)
assert RENDER_SPEC.loader is not None
RENDER_SPEC.loader.exec_module(RENDER_MODULE)


class PreservationDualViewTests(unittest.TestCase):
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

    def test_schedules_match_steps_and_isolate_replay_factor(self):
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
            ["repeat_full_consistency"] * 256,
        )
        self.assertEqual(
            [row["kind"] for row in treatment[1::2]],
            ["final_ce_only"] * 256,
        )

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

    def test_public_contract_excludes_raw_content(self):
        contract = public_contract(build_dataset(load_config(CONFIG)))
        for key in ("train_pairs", "dev_pairs"):
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

    def test_config_rejects_posthoc_method_changes(self):
        raw = json.loads(CONFIG.read_text(encoding="utf-8"))
        for key, value, error in (
            ("train_range_offset", 41000, "train_range_offset"),
            ("max_steps_per_arm", 256, "max_steps_per_arm"),
            ("learning_rate", 0.0001, "learning_rate"),
            ("replay_final_ce_weight", 1.0, "replay_final_ce_weight"),
            ("treatment_second_step", "repeat_full_consistency", "treatment"),
        ):
            with self.subTest(key=key):
                altered = copy.deepcopy(raw)
                altered[key] = value
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "config.json"
                    path.write_text(json.dumps(altered), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, error):
                        load_config(path)

    def test_gate_requires_zero_treatment_losses_and_fewer_than_control(self):
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
                        "objective": 1.0,
                        "gradient_norm": 1.0,
                    }
                ],
            },
            "failure_receipt_exists": False,
            "identity": {"adapter_sha256": "a" * 64},
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
        gates = RENDER_MODULE.admission_gates(
            control,
            treatment,
            {"control": reload, "treatment": reload},
        )
        self.assertTrue(all(gates.values()))
        treatment["comparison"]["paired_counts"]["baseline_only"] = 1
        gates = RENDER_MODULE.admission_gates(
            control,
            treatment,
            {"control": reload, "treatment": reload},
        )
        self.assertFalse(gates["treatment_anchor_maximum_losses"])
        self.assertTrue(gates["treatment_losses_lt_control"])


if __name__ == "__main__":
    unittest.main()
