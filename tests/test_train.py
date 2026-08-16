from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from nano_train.config import load_sft_smoke_config
from nano_train.data import (
    TokenizedSample,
    collate_samples,
    semantic_output_valid,
    tokenize_samples,
)
from nano_train.sft import (
    _assert_finite_gradients,
    _assert_finite_loss,
    _assert_finite_parameters,
    _batch_order,
    _scheduler_scale,
    _write_failure,
    evaluate_exact,
)
from scripts.run_generation_budget_audit import (
    load_config as load_audit_config,
    validate_contract as validate_audit_contract,
    verify_identity as verify_audit_identity,
)


class FakeTokenizer:
    eos_token = "<eos>"
    pad_token_id = 0

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
        enable_thinking,
    ):
        self.assertions = (tokenize, add_generation_prompt, enable_thinking)
        return "|".join(message["content"] for message in messages) + "|assistant:"

    def __call__(self, text, *, add_special_tokens=False):
        return SimpleNamespace(input_ids=[ord(char) % 251 + 1 for char in text])


class TrainTests(unittest.TestCase):
    def test_config_rejects_non_smoke_step_count(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": "nano_train_sft_smoke_v1",
                        "experiment_id": "x",
                        "model_path": "model",
                        "dataset_path": "data",
                        "output_dir": "out",
                        "seed": 1,
                        "dtype": "float16",
                        "max_length": 128,
                        "max_steps": 41,
                        "batch_size": 1,
                        "gradient_accumulation_steps": 1,
                        "learning_rate": 0.001,
                        "weight_decay": 0.0,
                        "warmup_steps": 1,
                        "lora_r": 4,
                        "lora_alpha": 8,
                        "lora_dropout": 0.0,
                        "lora_targets": ["q_proj"],
                        "generation_max_new_tokens": 8,
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "40 optimizer steps"):
                load_sft_smoke_config(path)

    def test_config_accepts_float32_smoke(self):
        config = load_sft_smoke_config(
            "configs/sft/format_contract_smoke_v2.json"
        )
        self.assertEqual(config.dtype, "float32")
        self.assertEqual(config.max_steps, 20)

    def test_v3_changes_only_dataset_identity_fields(self):
        v2 = load_sft_smoke_config(
            "configs/sft/format_contract_smoke_v2.json"
        )
        v3 = load_sft_smoke_config(
            "configs/sft/format_contract_smoke_v3.json"
        )
        excluded = {"experiment_id", "dataset_path", "output_dir"}
        for field in v2.__dataclass_fields__:
            if field in excluded:
                continue
            self.assertEqual(getattr(v2, field), getattr(v3, field), field)

    def test_v4_changes_only_semantic_objective_fields(self):
        v3 = load_sft_smoke_config(
            "configs/sft/format_contract_smoke_v3.json"
        )
        v4 = load_sft_smoke_config(
            "configs/sft/semantic_arithmetic_smoke_v4.json"
        )
        excluded = {
            "experiment_id",
            "dataset_path",
            "output_dir",
            "generation_max_new_tokens",
        }
        for field in v3.__dataclass_fields__:
            if field in excluded:
                continue
            self.assertEqual(getattr(v3, field), getattr(v4, field), field)
        self.assertEqual(v4.generation_max_new_tokens, 32)

    def test_v5_changes_only_step_count_and_identity(self):
        v4 = load_sft_smoke_config(
            "configs/sft/semantic_arithmetic_smoke_v4.json"
        )
        v5 = load_sft_smoke_config(
            "configs/sft/semantic_arithmetic_smoke_v5.json"
        )
        excluded = {"experiment_id", "output_dir", "max_steps"}
        for field in v4.__dataclass_fields__:
            if field in excluded:
                continue
            self.assertEqual(getattr(v4, field), getattr(v5, field), field)
        self.assertEqual(v4.max_steps, 20)
        self.assertEqual(v5.max_steps, 40)

    def test_v6_changes_only_process_objective_fields(self):
        v5 = load_sft_smoke_config(
            "configs/sft/semantic_arithmetic_smoke_v5.json"
        )
        v6 = load_sft_smoke_config(
            "configs/sft/arithmetic_process_smoke_v6.json"
        )
        excluded = {
            "experiment_id",
            "dataset_path",
            "output_dir",
            "max_length",
            "generation_max_new_tokens",
        }
        for field in v5.__dataclass_fields__:
            if field in excluded:
                continue
            self.assertEqual(getattr(v5, field), getattr(v6, field), field)
        self.assertEqual(v6.max_length, 192)
        self.assertEqual(v6.generation_max_new_tokens, 80)

    def test_v7_changes_only_mixed_safety_intervention_fields(self):
        v6 = load_sft_smoke_config(
            "configs/sft/arithmetic_process_smoke_v6.json"
        )
        v7 = load_sft_smoke_config(
            "configs/sft/hard_preservation_smoke_v7.json"
        )
        excluded = {
            "experiment_id",
            "dataset_path",
            "output_dir",
            "generation_max_new_tokens",
            "max_steps",
        }
        for field in v6.__dataclass_fields__:
            if field in excluded:
                continue
            self.assertEqual(getattr(v6, field), getattr(v7, field), field)
        self.assertEqual(v7.max_steps, 20)
        self.assertEqual(v7.generation_max_new_tokens, 128)

    def test_v8_changes_only_step_count_and_identity(self):
        v7 = load_sft_smoke_config(
            "configs/sft/hard_preservation_smoke_v7.json"
        )
        v8 = load_sft_smoke_config(
            "configs/sft/hard_preservation_smoke_v8.json"
        )
        excluded = {"experiment_id", "output_dir", "max_steps"}
        for field in v7.__dataclass_fields__:
            if field in excluded:
                continue
            self.assertEqual(getattr(v7, field), getattr(v8, field), field)
        self.assertEqual(v7.max_steps, 20)
        self.assertEqual(v8.max_steps, 40)

    def test_v9_changes_only_step_count_and_identity(self):
        v8 = load_sft_smoke_config(
            "configs/sft/hard_preservation_smoke_v8.json"
        )
        v9 = load_sft_smoke_config(
            "configs/sft/hard_preservation_smoke_v9.json"
        )
        excluded = {"experiment_id", "output_dir", "max_steps"}
        for field in v8.__dataclass_fields__:
            if field in excluded:
                continue
            self.assertEqual(getattr(v8, field), getattr(v9, field), field)
        self.assertEqual(v8.max_steps, 40)
        self.assertEqual(v9.max_steps, 30)

    def test_tokenize_masks_prompt_and_keeps_assistant(self):
        dataset = {
            "samples": [
                {
                    "sample_id": "synthetic-a",
                    "split": "train",
                    "format_family": "final_choice",
                    "messages": [
                        {"role": "system", "content": "system"},
                        {"role": "user", "content": "question"},
                        {"role": "assistant", "content": "FINAL: A"},
                    ],
                }
            ]
        }
        tokenizer = FakeTokenizer()
        sample = tokenize_samples(dataset, tokenizer, max_length=256)[0]
        self.assertTrue(all(value == -100 for value in sample.labels[: len(sample.prompt_ids)]))
        self.assertTrue(all(value != -100 for value in sample.labels[len(sample.prompt_ids) :]))
        self.assertEqual(tokenizer.assertions, (False, True, False))
        self.assertEqual(sample.task_family, "")

    def test_evaluate_exact_reports_family_metrics(self):
        class FakeModel:
            def eval(self):
                return None

            def generate(self, *, input_ids, **kwargs):
                return torch.cat(
                    [input_ids, torch.tensor([[9]], device=input_ids.device)],
                    dim=1,
                )

        class DecodeTokenizer:
            eos_token_id = 2
            pad_token_id = 0

            def decode(self, token_ids, *, skip_special_tokens):
                return "ok"

        samples = [
            TokenizedSample(
                "a",
                "validation",
                [1, 9],
                [-100, 9],
                [1],
                "ok",
                "final_numeric",
                None,
                "family_a",
            ),
            TokenizedSample(
                "b",
                "validation",
                [1, 9],
                [-100, 9],
                [1],
                "ok",
                "final_numeric",
                None,
                "family_b",
            ),
        ]
        metrics, rows = evaluate_exact(
            FakeModel(),
            DecodeTokenizer(),
            samples,
            device=torch.device("cpu"),
            max_new_tokens=1,
        )
        self.assertEqual(metrics["exact"], 2)
        self.assertEqual(metrics["semantic_exact"], 2)
        self.assertEqual(
            metrics["by_family"],
            {
                "family_a": {
                    "samples": 1,
                    "exact": 1,
                    "semantic_exact": 1,
                    "exact_failure_sample_ids": [],
                    "semantic_failure_sample_ids": [],
                },
                "family_b": {
                    "samples": 1,
                    "exact": 1,
                    "semantic_exact": 1,
                    "exact_failure_sample_ids": [],
                    "semantic_failure_sample_ids": [],
                },
            },
        )
        self.assertEqual(
            [row["task_family"] for row in rows],
            ["family_a", "family_b"],
        )

    def test_collator_masks_padding(self):
        first = TokenizedSample(
            "a", "train", [1, 2], [-100, 2], [1], "x", "final_numeric", None
        )
        second = TokenizedSample(
            "b", "train", [3], [3], [], "y", "final_numeric", None
        )
        batch = collate_samples([first, second], pad_token_id=0)
        self.assertEqual(batch["input_ids"].tolist(), [[1, 2], [3, 0]])
        self.assertEqual(batch["labels"].tolist(), [[-100, 2], [3, -100]])
        self.assertEqual(batch["attention_mask"].tolist(), [[1, 1], [1, 0]])

    def test_batch_order_and_scheduler_are_deterministic(self):
        samples = [
            TokenizedSample(
                str(index),
                "train",
                [index],
                [index],
                [],
                str(index),
                "final_numeric",
                None,
            )
            for index in range(8)
        ]
        self.assertEqual(_batch_order(samples, 7), _batch_order(samples, 7))
        self.assertNotEqual(_batch_order(samples, 7), _batch_order(samples, 8))
        self.assertEqual(_scheduler_scale(0, 2, 20), 0.5)
        self.assertEqual(_scheduler_scale(1, 2, 20), 1.0)
        self.assertGreater(_scheduler_scale(2, 2, 20), _scheduler_scale(19, 2, 20))

    def test_nonfinite_guards_and_failure_receipt(self):
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        parameter.grad = torch.tensor([float("nan")])
        with self.assertRaisesRegex(FloatingPointError, "gradients"):
            _assert_finite_gradients([parameter], step=2)
        with self.assertRaisesRegex(FloatingPointError, "loss"):
            _assert_finite_loss(torch.tensor(float("nan")), step=2)
        parameter.data.fill_(float("inf"))
        with self.assertRaisesRegex(FloatingPointError, "parameters"):
            _assert_finite_parameters([parameter], step=2)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_failure(
                root,
                step=2,
                stage="gradient",
                error=FloatingPointError("bad gradient"),
            )
            receipt = json.loads((root / "failure.json").read_text())
        self.assertEqual(receipt["optimizer_step"], 2)
        self.assertEqual(receipt["stage"], "gradient")
        self.assertFalse(receipt["adapter_saved"])
        self.assertFalse(receipt["post_validation_run"])

    def test_semantic_trace_accepts_equivalent_safe_expression(self):
        sample = TokenizedSample(
            "trace",
            "validation",
            [1],
            [1],
            [],
            "CALC: 7 + 5 * 3 = 22\nFINAL: 22",
            "trace_numeric",
            {
                "kind": "safe_ast_arithmetic_v1",
                "expression": "7 + 5 * 3",
                "expected_result": "22",
            },
        )
        self.assertTrue(
            semantic_output_valid(
                sample,
                "CALC: (5 * 3) + 7 = 22\nFINAL: 22",
            )
        )
        self.assertFalse(
            semantic_output_valid(
                sample,
                "CALC: (5 * 3) + 7 = 21\nFINAL: 21",
            )
        )
        self.assertFalse(
            semantic_output_valid(
                sample,
                "CALC: __import__('os').system('id') = 22\nFINAL: 22",
            )
        )

    def test_process_trace_verifies_every_intermediate_step(self):
        sample = TokenizedSample(
            "process",
            "validation",
            [1],
            [1],
            [],
            (
                "STEP 1: 5 * 3 = 15\n"
                "STEP 2: 7 + 15 = 22\n"
                "FINAL: 22"
            ),
            "process_trace_numeric",
            {
                "kind": "safe_ast_arithmetic_process_v2",
                "source_expression": "7 + 5 * 3",
                "steps": [
                    {"expression": "5 * 3", "expected_result": "15"},
                    {"expression": "7 + 15", "expected_result": "22"},
                ],
                "expected_result": "22",
            },
        )
        self.assertTrue(
            semantic_output_valid(
                sample,
                (
                    "STEP 1: (5 * 3) = 15\n"
                    "STEP 2: (7 + 15) = 22\n"
                    "FINAL: 22"
                ),
            )
        )
        self.assertFalse(
            semantic_output_valid(
                sample,
                (
                    "STEP 1: 5 * 3 = 14\n"
                    "STEP 2: 7 + 14 = 21\n"
                    "FINAL: 21"
                ),
            )
        )
        self.assertFalse(
            semantic_output_valid(
                sample,
                (
                    "STEP 1: 5 * 3 = 15\n"
                    "STEP 2: 8 + 14 = 22\n"
                    "FINAL: 22"
                ),
            )
        )
        self.assertFalse(
            semantic_output_valid(
                sample,
                (
                    "STEP 1: __import__('os').system('id') = 15\n"
                    "STEP 2: 7 + 15 = 22\n"
                    "FINAL: 22"
                ),
            )
        )

    def test_reasoning_numeric_accepts_safe_equivalent_work(self):
        sample = TokenizedSample(
            "reasoning",
            "validation",
            [1],
            [1],
            [],
            "WORK: 1 + 12 + 12 * 2 = 37\nFINAL: 37",
            "reasoning_numeric",
            {
                "kind": "safe_ast_reasoning_numeric_v1",
                "expression": "1 + 12 + 12 * 2",
                "expected_result": "37",
            },
        )
        self.assertTrue(
            semantic_output_valid(
                sample,
                "WORK: 12 * 2 + 12 + 1 = 37\nFINAL: 37",
            )
        )
        self.assertFalse(
            semantic_output_valid(
                sample,
                "WORK: 12 * 2 + 12 = 36\nFINAL: 36",
            )
        )
        self.assertFalse(
            semantic_output_valid(
                sample,
                "WORK: __import__('os').system('id') = 37\nFINAL: 37",
            )
        )
        self.assertFalse(
            semantic_output_valid(
                sample,
                "WORK: 12 * 2 + 12 + 1 = 37 FINAL: 37",
            )
        )

    def test_v5_budget_audit_identity_and_contract(self):
        config = load_audit_config(
            Path(
                "configs/audits/"
                "semantic_arithmetic_v5_budget_audit_v1.json"
            )
        )
        identity = verify_audit_identity(config)
        self.assertEqual(identity, config["expected"])
        validation = [
            TokenizedSample(
                "case-a",
                "validation",
                [1],
                [-100, 1, 2, 3],
                [1],
                "target",
                "trace_numeric",
                {
                    "kind": "safe_ast_arithmetic_v1",
                    "expression": "1 + 1",
                    "expected_result": "2",
                },
            )
        ]
        tokenizer = mock.Mock()
        tokenizer.return_value = SimpleNamespace(input_ids=[1, 2, 3])
        contract = validate_audit_contract(
            config,
            tokenizer,
            validation,
            {"post_sft": [{"sample_id": "case-a"}]},
        )
        self.assertTrue(contract["source_case_set_matches"])
        self.assertTrue(contract["budget_above_target_with_eos_max"])

        short_budget = {**config, "generation_max_new_tokens": 3}
        with self.assertRaisesRegex(ValueError, "must exceed"):
            validate_audit_contract(
                short_budget,
                tokenizer,
                validation,
                {"post_sft": [{"sample_id": "case-a"}]},
            )

        with self.assertRaisesRegex(ValueError, "case sets differ"):
            validate_audit_contract(
                config,
                tokenizer,
                validation,
                {"post_sft": [{"sample_id": "different-case"}]},
            )


if __name__ == "__main__":
    unittest.main()
