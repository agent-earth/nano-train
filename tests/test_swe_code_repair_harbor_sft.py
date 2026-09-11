from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from nano_train.swe_code_repair_harbor_sft import (
    admission_gates,
    build_training_contract,
    compare_actionable,
    harbor_training_dataset,
    load_config,
)
from nano_train.swe_code_repair_qualification import (
    build_qualification_contract,
    load_config as load_qualification_config,
)
from nano_train.swe_code_repair_sft import build_selection_contract
from scripts.preregister_swe_code_repair_harbor_sft_v2 import build_receipt
from scripts.render_swe_code_repair_harbor_sft_v2 import build_report


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/sft/swe_code_repair_harbor_prompt_v2.json"


def result_row(case_id: str, *, band: str, actionable: bool) -> dict:
    return {
        "sample_id_sha256": case_id,
        "task_family": band,
        "terminus_parser_valid": actionable,
        "terminus_actionable_or_complete": actionable,
        "generation_budget_exhausted": False,
    }


class SWECodeRepairHarborSFTTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config(CONFIG)
        cls.contract = build_training_contract(cls.config)

    def test_config_freezes_single_variable_prompt_alignment(self):
        self.assertEqual(self.config.max_steps, 40)
        self.assertEqual(self.config.gradient_accumulation_steps, 4)
        self.assertEqual(self.config.learning_rate, 0.00005)
        self.assertEqual(self.config.lora_targets, ("q_proj", "v_proj"))
        self.assertEqual(self.config.max_length, 2816)
        self.assertEqual(sum(self.config.train_rows_by_band.values()), 160)
        self.assertEqual(sum(self.config.dev_rows_by_band.values()), 24)

    def test_training_ids_match_v1_and_third_dev_is_fresh(self):
        source_selection = build_selection_contract(
            self.contract["source_config"]
        )
        qualification = build_qualification_contract(
            load_qualification_config(self.config.qualification_v1_config_path)
        )
        self.assertEqual(
            self.contract["train_sample_ids"],
            source_selection["train_sample_ids"],
        )
        self.assertEqual(
            self.contract["train_sample_ids_sha256"],
            source_selection["train_sample_ids_sha256"],
        )
        excluded = set(source_selection["dev_sample_ids"]) | set(
            qualification["selected_dev_ids"]
        )
        self.assertEqual(len(excluded), 36)
        self.assertEqual(len(self.contract["dev_sample_ids"]), 24)
        self.assertFalse(set(self.contract["dev_sample_ids"]) & excluded)

    def test_harbor_training_dataset_uses_one_user_prompt_and_target(self):
        dataset = harbor_training_dataset(
            self.contract["selected_train"][:1],
            prompt_template=Path(
                self.config.terminus_prompt_template_path
            ).read_text(encoding="utf-8"),
            split="train",
        )
        self.assertEqual(len(dataset["samples"]), 1)
        sample = dataset["samples"][0]
        self.assertEqual(
            [message["role"] for message in sample["messages"]],
            ["user", "assistant"],
        )
        self.assertIn("Format your response as JSON", sample["messages"][0]["content"])
        self.assertIn("Task Description:", sample["messages"][0]["content"])
        self.assertEqual(sample["split"], "train")

    def test_config_rejects_method_mutation(self):
        raw = json.loads(CONFIG.read_text(encoding="utf-8"))
        raw["learning_rate"] = 0.0001
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "freezes learning_rate"):
                load_config(path)

    def test_comparison_and_admission_require_no_v1_regression(self):
        bands = ["short"] * 8 + ["medium"] * 8 + ["long"] * 8
        v1 = [
            result_row(str(index), band=band, actionable=index != 0)
            for index, band in enumerate(bands)
        ]
        v2 = [
            result_row(str(index), band=band, actionable=True)
            for index, band in enumerate(bands)
        ]
        comparison = compare_actionable(v1, v2)
        self.assertEqual(
            comparison,
            {"candidate_only": 1, "baseline_only": 0, "both": 23, "neither": 0},
        )
        gates = admission_gates(
            base_loss=1.0,
            v1_loss=0.9,
            v2_loss=0.8,
            v1_rows=v1,
            v2_rows=v2,
            config=self.config,
        )
        self.assertTrue(all(gates.values()))

        regressed = copy.deepcopy(v2)
        regressed[8]["terminus_actionable_or_complete"] = False
        regression_gates = admission_gates(
            base_loss=1.0,
            v1_loss=0.9,
            v2_loss=0.8,
            v1_rows=v1,
            v2_rows=regressed,
            config=self.config,
        )
        self.assertFalse(regression_gates["maximum_v1_only_losses"])
        self.assertFalse(regression_gates["per_band_non_regression_vs_v1"])

    def test_preregister_is_deterministic_and_closed(self):
        first = build_receipt()
        second = build_receipt()
        self.assertEqual(first, second)
        self.assertEqual(first["selection"]["train_samples"], 160)
        self.assertEqual(first["selection"]["dev_samples"], 24)
        self.assertEqual(
            first["selection"]["overlap_with_v1_and_qualification_dev"],
            0,
        )
        self.assertLessEqual(
            first["selection"]["maximum_full_sequence_tokens"],
            self.config.max_length,
        )
        self.assertTrue(
            first["decision_boundary"][
                "passing_unlocks_only_fresh8_sft_screening"
            ]
        )
        self.assertFalse(first["decision_boundary"]["complete_500_allowed"])
        self.assertTrue(
            first["execution_boundary"]["this_commit_only_preregisters"]
        )

    def test_public_report_admits_only_fresh8_without_private_content(self):
        report = build_report()
        self.assertEqual(
            report["decision"]["verdict"],
            "admit_to_frozen_fresh8_screening",
        )
        self.assertTrue(
            report["decision"]["candidate_admitted_for_fresh8"]
        )
        self.assertGreater(
            report["fresh_heldout"]["loss"][
                "v2_relative_improvement_vs_base"
            ],
            0.14,
        )
        self.assertGreater(
            report["fresh_heldout"]["loss"][
                "v2_relative_improvement_vs_v1"
            ],
            0.10,
        )
        self.assertEqual(
            report["fresh_heldout"]["structure"]["harbor_sft_v2"][
                "terminus_parser_valid"
            ],
            24,
        )
        self.assertTrue(report["reload"]["loss_exact"])
        self.assertTrue(report["reload"]["generations_exact"])
        self.assertFalse(report["decision"]["complete_500_allowed"])
        serialized = json.dumps(report).lower()
        self.assertNotIn("messages", serialized)
        self.assertNotIn("problem_statement", serialized)
        self.assertNotIn("keystrokes\":", serialized)


if __name__ == "__main__":
    unittest.main()
