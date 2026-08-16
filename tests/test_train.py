from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from nano_train.config import load_sft_smoke_config
from nano_train.data import TokenizedSample, collate_samples, tokenize_samples
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
                        "max_steps": 21,
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
            with self.assertRaisesRegex(ValueError, "20 optimizer steps"):
                load_sft_smoke_config(path)

    def test_config_accepts_float32_smoke(self):
        config = load_sft_smoke_config(
            "configs/sft/format_contract_smoke_v2.json"
        )
        self.assertEqual(config.dtype, "float32")
        self.assertEqual(config.max_steps, 20)

    def test_tokenize_masks_prompt_and_keeps_assistant(self):
        dataset = {
            "samples": [
                {
                    "sample_id": "synthetic-a",
                    "split": "train",
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
        first = TokenizedSample("a", "train", [1, 2], [-100, 2], [1], "x")
        second = TokenizedSample("b", "train", [3], [3], [], "y")
        batch = collate_samples([first, second], pad_token_id=0)
        self.assertEqual(batch["input_ids"].tolist(), [[1, 2], [3, 0]])
        self.assertEqual(batch["labels"].tolist(), [[-100, 2], [3, -100]])
        self.assertEqual(batch["attention_mask"].tolist(), [[1, 1], [1, 0]])

    def test_batch_order_and_scheduler_are_deterministic(self):
        samples = [
            TokenizedSample(str(index), "train", [index], [index], [], str(index))
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


if __name__ == "__main__":
    unittest.main()
