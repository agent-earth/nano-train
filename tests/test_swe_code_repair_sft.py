from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from nano_train.swe_code_repair_sft import (
    build_selection_contract,
    load_config,
    token_band,
    valid_json_action,
    validate_release,
)
from scripts.preregister_swe_code_repair_sft_v1 import build_receipt


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/sft/swe_code_repair_smoke_v1.json"


class SWECodeRepairSFTTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config(CONFIG)
        cls.selection = build_selection_contract(cls.config)

    def test_config_freezes_single_candidate_smoke(self):
        self.assertEqual(
            self.config.experiment_id,
            "evoloop-swe-code-repair-sft-smoke-v1",
        )
        self.assertEqual(self.config.max_steps, 40)
        self.assertEqual(self.config.gradient_accumulation_steps, 4)
        self.assertEqual(self.config.lora_targets, ("q_proj", "v_proj"))
        self.assertEqual(sum(self.config.train_rows_by_band.values()), 160)
        self.assertEqual(sum(self.config.dev_rows_by_band.values()), 12)

    def test_selection_is_deterministic_disjoint_and_band_balanced(self):
        second = build_selection_contract(self.config)
        self.assertEqual(
            self.selection["train_sample_ids"],
            second["train_sample_ids"],
        )
        self.assertEqual(
            self.selection["dev_sample_ids"],
            second["dev_sample_ids"],
        )
        self.assertEqual(len(self.selection["train_sample_ids"]), 160)
        self.assertEqual(len(self.selection["dev_sample_ids"]), 12)
        self.assertFalse(
            set(self.selection["train_sample_ids"])
            & set(self.selection["dev_sample_ids"])
        )
        train_bands = {
            band: sum(
                token_band(int(row["token_count"])) == band
                for row in self.selection["selected_train"]
            )
            for band in ("short", "medium", "long")
        }
        dev_bands = {
            band: sum(
                token_band(int(row["token_count"])) == band
                for row in self.selection["selected_dev"]
            )
            for band in ("short", "medium", "long")
        }
        self.assertEqual(train_bands, self.config.train_rows_by_band)
        self.assertEqual(dev_bands, self.config.dev_rows_by_band)

    def test_json_action_validator_accepts_dataset_order_and_rejects_noise(self):
        target = self.selection["selected_train"][0]["messages"][-1]["content"]
        self.assertTrue(valid_json_action(target))
        self.assertTrue(
            valid_json_action(
                json.dumps(
                    {
                        "plan": "Inspect and patch.",
                        "commands": [
                            {
                                "keystrokes": "git diff --check\n",
                                "duration": 1,
                            }
                        ],
                        "analysis": "A production edit is required.",
                    }
                )
            )
        )
        self.assertFalse(valid_json_action("not json"))
        self.assertFalse(
            valid_json_action(
                json.dumps(
                    {
                        "analysis": "x",
                        "plan": "y",
                        "commands": [],
                    }
                )
            )
        )
        self.assertFalse(
            valid_json_action(
                json.dumps(
                    {
                        "analysis": "x",
                        "plan": "y",
                        "commands": [
                            {"keystrokes": "true", "duration": 0}
                        ],
                    }
                )
            )
        )

    def test_config_rejects_posthoc_mutation(self):
        raw = json.loads(CONFIG.read_text(encoding="utf-8"))
        raw["max_steps"] = 41
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "freezes max_steps"):
                load_config(path)

    def test_release_tampering_fails_closed(self):
        release = copy.deepcopy(self.selection["release"])
        release["training_boundary"][
            "contains_swe_bench_verified_content"
        ] = True
        with self.assertRaisesRegex(ValueError, "not admitted"):
            validate_release(release, self.config)
        release = copy.deepcopy(self.selection["release"])
        release["accepted"]["train_samples"] -= 1
        with self.assertRaisesRegex(ValueError, "not admitted"):
            validate_release(release, self.config)
        release = copy.deepcopy(self.selection["release"])
        release["checks"]["verified_problem_exclusion_pass"] = False
        with self.assertRaisesRegex(ValueError, "not admitted"):
            validate_release(release, self.config)

    def test_dataset_identity_tampering_fails_closed(self):
        altered = copy.deepcopy(self.config.__dict__)
        altered["dataset_file_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "dataset identity differs"):
            build_selection_contract(type(self.config)(**altered))

    def test_preregister_is_deterministic_and_closed(self):
        first = build_receipt()
        second = build_receipt()
        self.assertEqual(first, second)
        self.assertEqual(first["project_name"], "EvoLoop")
        self.assertTrue(
            first["execution_boundary"]["this_commit_only_preregisters"]
        )
        self.assertFalse(first["decision_boundary"]["complete_500_allowed"])
        self.assertEqual(
            first["selection"]["train_sample_ids_sha256"],
            self.selection["train_sample_ids_sha256"],
        )


if __name__ == "__main__":
    unittest.main()
