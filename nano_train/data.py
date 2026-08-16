from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass(frozen=True)
class TokenizedSample:
    sample_id: str
    split: str
    input_ids: list[int]
    labels: list[int]
    prompt_ids: list[int]
    target: str


def load_analog_dataset(path: str | Path) -> dict[str, Any]:
    dataset = json.loads(Path(path).read_text(encoding="utf-8"))
    if dataset.get("schema_version") != "nano_analog_dataset_v1":
        raise ValueError("unsupported analog dataset schema")
    policy = dataset.get("policy", {})
    if (
        policy.get("source_split") != "non_eval_analog_only"
        or policy.get("training_allowed") is not True
        or policy.get("contains_benchmark_content") is not False
    ):
        raise ValueError("analog dataset is not eligible for SFT smoke")
    samples = dataset.get("samples", [])
    if not samples:
        raise ValueError("analog dataset contains no samples")
    if any(sample.get("training_eligible") is not True for sample in samples):
        raise ValueError("analog dataset contains an ineligible sample")
    return dataset


def tokenize_samples(
    dataset: dict[str, Any],
    tokenizer: Any,
    *,
    max_length: int,
) -> list[TokenizedSample]:
    result = []
    for sample in dataset["samples"]:
        messages = sample["messages"]
        prompt = tokenizer.apply_chat_template(
            messages[:-1],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        target = str(messages[-1]["content"])
        full = prompt + target + tokenizer.eos_token
        prompt_ids = tokenizer(
            prompt,
            add_special_tokens=False,
        ).input_ids
        input_ids = tokenizer(
            full,
            add_special_tokens=False,
        ).input_ids
        if len(input_ids) > max_length:
            raise ValueError(
                f"sample exceeds max_length: {sample['sample_id']} "
                f"{len(input_ids)}>{max_length}"
            )
        labels = [-100] * len(prompt_ids) + input_ids[len(prompt_ids) :]
        if len(labels) != len(input_ids) or not any(label != -100 for label in labels):
            raise ValueError(f"invalid assistant mask: {sample['sample_id']}")
        result.append(
            TokenizedSample(
                sample_id=str(sample["sample_id"]),
                split=str(sample["split"]),
                input_ids=input_ids,
                labels=labels,
                prompt_ids=prompt_ids,
                target=target,
            )
        )
    return result


def collate_samples(
    samples: list[TokenizedSample],
    *,
    pad_token_id: int,
) -> dict[str, torch.Tensor]:
    max_length = max(len(sample.input_ids) for sample in samples)
    input_ids = []
    labels = []
    attention_mask = []
    for sample in samples:
        padding = max_length - len(sample.input_ids)
        input_ids.append(sample.input_ids + [pad_token_id] * padding)
        labels.append(sample.labels + [-100] * padding)
        attention_mask.append([1] * len(sample.input_ids) + [0] * padding)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
    }
