#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import peft
import torch
import transformers
from transformers import AutoConfig, AutoTokenizer

from nano_train.config import load_sft_smoke_config
from nano_train.data import load_analog_dataset, tokenize_samples
from nano_train.sft import sha256_file


ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    "format_contract_smoke_v1.json": {
        "config_sha256": "09bbf842ea2a335e283385eeea18d352f9311dc5747e86da9be9b58bfdae2d93",
        "dataset_sha256": "46f2128f219db7011d5db95b5ca3a97029b57f5ac959e194860b4c0f4ba3ad53",
        "model_config_sha256": "ddc63e1c717afa86c865bb5e01313d89d72bb53b97ad4a8a03ba8510c0621670",
    },
    "format_contract_smoke_v2.json": {
        "config_sha256": "62cc5189cb048fd1a2b4070ffdd27b0a18c3363df1ae8dfa244a381401646207",
        "dataset_sha256": "46f2128f219db7011d5db95b5ca3a97029b57f5ac959e194860b4c0f4ba3ad53",
        "model_config_sha256": "ddc63e1c717afa86c865bb5e01313d89d72bb53b97ad4a8a03ba8510c0621670",
    },
    "format_contract_smoke_v3.json": {
        "config_sha256": "fee61ad70cec96368849b6873e7f261dbfc822dc82af7d206cfdb29b58edbfdd",
        "dataset_sha256": "95d8e3e8a173960fd8604f284bae0243e74f4c924c96b719252c8c9a6525f001",
        "model_config_sha256": "ddc63e1c717afa86c865bb5e01313d89d72bb53b97ad4a8a03ba8510c0621670",
    },
    "semantic_arithmetic_smoke_v4.json": {
        "config_sha256": "a162cc982896b16d5f3f1bdb79ba455f24b629ec95cc149b289e90e0b6ffab04",
        "dataset_sha256": "d226f243051b7d2d2d4db4d5a596b871032fa44d71b296586f879559a8781c09",
        "model_config_sha256": "ddc63e1c717afa86c865bb5e01313d89d72bb53b97ad4a8a03ba8510c0621670",
    },
}
EXPECTED_SPLITS = {
    "format_contract_smoke_v1.json": {"train": 102, "validation": 26},
    "format_contract_smoke_v2.json": {"train": 102, "validation": 26},
    "format_contract_smoke_v3.json": {"train": 128, "validation": 32},
    "semantic_arithmetic_smoke_v4.json": {"train": 160, "validation": 32},
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/sft/format_contract_smoke_v1.json",
    )
    args = parser.parse_args()
    config_path = (ROOT / args.config).resolve()
    if config_path.name not in EXPECTED:
        raise SystemExit(f"no frozen identity for config: {config_path.name}")
    expected_identity = EXPECTED[config_path.name]
    config = load_sft_smoke_config(config_path)
    dataset_path = (ROOT / config.dataset_path).resolve()
    model_path = (ROOT / config.model_path).resolve()
    dataset = load_analog_dataset(dataset_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    samples = tokenize_samples(
        dataset,
        tokenizer,
        max_length=config.max_length,
    )
    counts = Counter(sample.split for sample in samples)
    if counts != EXPECTED_SPLITS[config_path.name]:
        raise SystemExit(f"unexpected split counts: {counts}")
    max_tokens = max(len(sample.input_ids) for sample in samples)
    if max_tokens > config.max_length:
        raise SystemExit("tokenized dataset exceeds max_length")
    target_lengths = [
        sum(label != -100 for label in sample.labels) for sample in samples
    ]
    if min(target_lengths) < 2:
        raise SystemExit("assistant mask contains an empty target")

    model_config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    if model_config.model_type != "qwen3_5":
        raise SystemExit(f"unexpected model type: {model_config.model_type}")
    text_layers = int(model_config.text_config.num_hidden_layers)
    if text_layers != 32:
        raise SystemExit(f"unexpected text layer count: {text_layers}")
    targets = set(config.lora_targets)
    required_targets = {"q_proj", "v_proj", "gate_proj", "up_proj", "down_proj"}
    if targets != required_targets:
        raise SystemExit(f"unexpected LoRA targets: {targets}")

    actual = {
        "config_sha256": sha256_file(config_path),
        "dataset_sha256": sha256_file(dataset_path),
        "model_config_sha256": sha256_file(model_path / "config.json"),
    }
    if actual != expected_identity:
        raise SystemExit(
            f"identity mismatch: actual={actual}, expected={expected_identity}"
        )

    print(
        json.dumps(
            {
                "schema_version": "nano_train_sft_smoke_preflight_v1",
                "experiment_id": config.experiment_id,
                "identity": actual,
                "samples": len(samples),
                "splits": dict(sorted(counts.items())),
                "max_input_tokens": max_tokens,
                "assistant_target_tokens": {
                    "min": min(target_lengths),
                    "max": max(target_lengths),
                },
                "model_type": model_config.model_type,
                "text_layers": text_layers,
                "dtype": config.dtype,
                "max_steps": config.max_steps,
                "effective_batch_size": (
                    config.batch_size * config.gradient_accumulation_steps
                ),
                "lora_targets": sorted(targets),
                "dependencies": {
                    "torch": torch.__version__,
                    "transformers": transformers.__version__,
                    "peft": peft.__version__,
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
