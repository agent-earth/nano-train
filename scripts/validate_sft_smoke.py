#!/usr/bin/env python3

from __future__ import annotations

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
CONFIG = ROOT / "configs/sft/format_contract_smoke_v1.json"
EXPECTED = {
    "config_sha256": "09bbf842ea2a335e283385eeea18d352f9311dc5747e86da9be9b58bfdae2d93",
    "dataset_sha256": "46f2128f219db7011d5db95b5ca3a97029b57f5ac959e194860b4c0f4ba3ad53",
    "model_config_sha256": "ddc63e1c717afa86c865bb5e01313d89d72bb53b97ad4a8a03ba8510c0621670"
}


def main() -> None:
    config = load_sft_smoke_config(CONFIG)
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
    if counts != {"train": 102, "validation": 26}:
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
        "config_sha256": sha256_file(CONFIG),
        "dataset_sha256": sha256_file(dataset_path),
        "model_config_sha256": sha256_file(model_path / "config.json"),
    }
    if actual != EXPECTED:
        raise SystemExit(f"identity mismatch: actual={actual}, expected={EXPECTED}")

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
