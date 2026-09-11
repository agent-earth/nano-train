from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from nano_train.swe_code_repair_sft import (
    build_selection_contract,
    diagnose_json_action_output,
    load_config,
    token_band,
    valid_json_action,
    validate_release,
    validate_reload,
)
from scripts.preregister_swe_code_repair_sft_v1 import build_receipt
from scripts.render_swe_code_repair_sft_v1 import build_report


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

    def test_json_action_diagnostics_do_not_store_output_text(self):
        valid = json.dumps(
            {
                "analysis": "x",
                "plan": "y",
                "commands": [{"keystrokes": "true\n", "duration": 1}],
            }
        )
        exact = diagnose_json_action_output(valid)
        self.assertEqual(exact["parse_status"], "exact_valid")
        self.assertEqual(exact["prefix_class"], "json_object")
        self.assertNotIn(valid, json.dumps(exact))
        fenced = diagnose_json_action_output(f"```json\n{valid}\n```")
        self.assertEqual(fenced["parse_status"], "json_decode_error")
        self.assertEqual(fenced["prefix_class"], "markdown_fence")
        self.assertTrue(fenced["recoverable_json_action_substring"])

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

    def test_reload_accepts_rejected_candidate_for_reproducibility(self):
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory)
            adapter = output_root / "adapter"
            adapter.mkdir()
            (adapter / "adapter_config.json").write_text("{}")
            generations = [{"sample_id": "dev-1", "json_action_valid": False}]
            (output_root / "validation_generations.json").write_text(
                json.dumps(generations),
                encoding="utf-8",
            )
            config = type(self.config)(
                **{**self.config.__dict__, "output_dir": str(output_root)}
            )
            from nano_train.sft import sha256_tree

            (output_root / "metrics.json").write_text(
                json.dumps(
                    {
                        "experiment_id": config.experiment_id,
                        "dataset_file_sha256": config.dataset_file_sha256,
                        "release_manifest_sha256": (
                            config.release_manifest_sha256
                        ),
                        "model_config_sha256": config.model_config_sha256,
                        "adapter_sha256": sha256_tree(adapter),
                        "candidate_admitted_for_fresh8": False,
                        "post_dev_loss": 1.0,
                        "generation_validation": {
                            "samples": 1,
                            "json_action_valid": 0,
                            "json_action_valid_rate": 0.0,
                            "by_band": {},
                        },
                    }
                ),
                encoding="utf-8",
            )
            fake_tokenizer = mock.Mock(eos_token_id=1, pad_token_id=1)
            fake_model = mock.Mock()
            fake_peft_model = mock.Mock()
            with (
                mock.patch(
                    "nano_train.swe_code_repair_sft.build_selection_contract",
                    return_value={
                        "selected_train": [],
                        "selected_dev": [],
                        "train_sample_ids": [],
                        "dev_sample_ids": ["dev-1"],
                    },
                ),
                mock.patch(
                    "nano_train.swe_code_repair_sft.AutoTokenizer.from_pretrained",
                    return_value=fake_tokenizer,
                ),
                mock.patch(
                    "nano_train.swe_code_repair_sft.tokenize_samples",
                    return_value=[mock.Mock(sample_id="dev-1")],
                ),
                mock.patch(
                    "nano_train.swe_code_repair_sft.Qwen3_5ForCausalLM.from_pretrained",
                    return_value=fake_model,
                ),
                mock.patch.object(fake_model, "to", return_value=fake_model),
                mock.patch(
                    "nano_train.swe_code_repair_sft.PeftModel.from_pretrained",
                    return_value=fake_peft_model,
                ),
                mock.patch.object(
                    fake_peft_model,
                    "to",
                    return_value=fake_peft_model,
                ),
                mock.patch(
                    "nano_train.swe_code_repair_sft.mean_teacher_forced_loss",
                    return_value=1.0,
                ),
                mock.patch(
                    "nano_train.swe_code_repair_sft.generate_validation",
                    return_value=(
                        {
                            "samples": 1,
                            "json_action_valid": 0,
                            "json_action_valid_rate": 0.0,
                            "by_band": {},
                        },
                        generations,
                    ),
                ),
                mock.patch(
                    "nano_train.swe_code_repair_sft.torch.cuda.is_available",
                    return_value=True,
                ),
                mock.patch(
                    "nano_train.swe_code_repair_sft.torch.cuda.max_memory_allocated",
                    return_value=0,
                ),
            ):
                receipt = validate_reload(config)
            self.assertTrue(receipt["reload_success"])
            self.assertFalse(
                json.loads(
                    (output_root / "metrics.json").read_text(encoding="utf-8")
                )["candidate_admitted_for_fresh8"]
            )

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

    def test_public_report_recomputes_rejection_without_private_content(self):
        report = build_report()
        self.assertEqual(
            report["decision"]["verdict"],
            "reject_standard_sft_v1",
        )
        self.assertFalse(
            report["decision"]["candidate_admitted_for_fresh8"]
        )
        self.assertGreater(
            report["heldout"]["relative_dev_loss_improvement"],
            0.10,
        )
        self.assertEqual(
            report["heldout"]["adapter_terminus_parser"][
                "terminus_parser_valid"
            ],
            0,
        )
        serialized = json.dumps(report).lower()
        self.assertNotIn("messages", serialized)
        self.assertNotIn("problem_statement", serialized)
        self.assertNotIn("keystrokes\":", serialized)


if __name__ == "__main__":
    unittest.main()
