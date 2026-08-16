from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

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


if __name__ == "__main__":
    unittest.main()
