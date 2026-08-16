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
from nano_train.sft import _batch_order, sha256_file


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
    "semantic_arithmetic_smoke_v5.json": {
        "config_sha256": "89e48fa387851e06a9394253e3bbdc345d7a0e84d963015be67e2ae8183fad38",
        "dataset_sha256": "d226f243051b7d2d2d4db4d5a596b871032fa44d71b296586f879559a8781c09",
        "model_config_sha256": "ddc63e1c717afa86c865bb5e01313d89d72bb53b97ad4a8a03ba8510c0621670",
    },
    "arithmetic_process_smoke_v6.json": {
        "config_sha256": "f8ab480d0195527b3fe8d98bb49ee377ba444257dcfe203de50c720d06624447",
        "dataset_sha256": "0e53fb3d05fb60569a4109da05b66d93c1158f734495e0126a55cf195c41653a",
        "model_config_sha256": "ddc63e1c717afa86c865bb5e01313d89d72bb53b97ad4a8a03ba8510c0621670",
    },
    "hard_preservation_smoke_v7.json": {
        "config_sha256": "787649d577e3978311c968b9d886ae2188a2a6ff9fcc7f6c00e79f5bfb896c08",
        "dataset_sha256": "ac07e10f1e04ca8ddec74aefc8df9b00475f7fc9c57345ff98182dd8e4c8bae9",
        "model_config_sha256": "ddc63e1c717afa86c865bb5e01313d89d72bb53b97ad4a8a03ba8510c0621670",
    },
    "hard_preservation_smoke_v8.json": {
        "config_sha256": "1d74ff3fb8a6bd9d87a63d73d19af6b3f21dde4831742bfe7681a9628556039e",
        "dataset_sha256": "ac07e10f1e04ca8ddec74aefc8df9b00475f7fc9c57345ff98182dd8e4c8bae9",
        "model_config_sha256": "ddc63e1c717afa86c865bb5e01313d89d72bb53b97ad4a8a03ba8510c0621670",
    },
    "hard_preservation_smoke_v9.json": {
        "config_sha256": "b83a98b345a96971207c4883db507024283a1c7ea073b853bc7e30c6f28ff7f1",
        "dataset_sha256": "ac07e10f1e04ca8ddec74aefc8df9b00475f7fc9c57345ff98182dd8e4c8bae9",
        "model_config_sha256": "ddc63e1c717afa86c865bb5e01313d89d72bb53b97ad4a8a03ba8510c0621670",
    },
    "hard_preservation_smoke_v10.json": {
        "config_sha256": "49c5d50572bb568235fd25e4ad5882b381facc795e6131196423f829985c8910",
        "dataset_sha256": "ac07e10f1e04ca8ddec74aefc8df9b00475f7fc9c57345ff98182dd8e4c8bae9",
        "model_config_sha256": "ddc63e1c717afa86c865bb5e01313d89d72bb53b97ad4a8a03ba8510c0621670",
    },
    "targeted_preservation_smoke_v11.json": {
        "config_sha256": "9a971cb46a1f5c21164d6117bef40aedfcb7170e9e82604bb7400c942a2be593",
        "dataset_sha256": "ab51a1be5f45d7f71796fbf98ef6cce83ff9cb0f0a756fed01cb1e7aea55651d",
        "model_config_sha256": "ddc63e1c717afa86c865bb5e01313d89d72bb53b97ad4a8a03ba8510c0621670",
    },
}
EXPECTED_SPLITS = {
    "format_contract_smoke_v1.json": {"train": 102, "validation": 26},
    "format_contract_smoke_v2.json": {"train": 102, "validation": 26},
    "format_contract_smoke_v3.json": {"train": 128, "validation": 32},
    "semantic_arithmetic_smoke_v4.json": {"train": 160, "validation": 32},
    "semantic_arithmetic_smoke_v5.json": {"train": 160, "validation": 32},
    "arithmetic_process_smoke_v6.json": {"train": 160, "validation": 32},
    "hard_preservation_smoke_v7.json": {"train": 160, "validation": 32},
    "hard_preservation_smoke_v8.json": {"train": 160, "validation": 32},
    "hard_preservation_smoke_v9.json": {"train": 160, "validation": 32},
    "hard_preservation_smoke_v10.json": {"train": 160, "validation": 32},
    "targeted_preservation_smoke_v11.json": {"train": 160, "validation": 32},
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
    examples_seen = (
        config.max_steps
        * config.batch_size
        * config.gradient_accumulation_steps
    )
    train_samples = [sample for sample in samples if sample.split == "train"]
    order = _batch_order(train_samples, config.seed)
    visited_indices = [
        order[index % len(order)] for index in range(examples_seen)
    ]
    unique_examples_seen = len(set(visited_indices))
    if (
        config_path.name
        in {
            "semantic_arithmetic_smoke_v5.json",
            "arithmetic_process_smoke_v6.json",
        }
        and (
            examples_seen != counts["train"]
            or unique_examples_seen != counts["train"]
        )
    ):
        raise SystemExit(
            "full-coverage smoke must expose exactly one train-set equivalent"
        )
    if config_path.name == "arithmetic_process_smoke_v6.json":
        if any(
            sample.format_family != "process_trace_numeric"
            or (sample.verifier or {}).get("kind")
            != "safe_ast_arithmetic_process_v2"
            for sample in samples
        ):
            raise SystemExit("v6 requires only verified process trace samples")
        if max(target_lengths) >= config.generation_max_new_tokens:
            raise SystemExit(
                "v6 generation budget must exceed target length including EOS"
            )
    if config_path.name in {
        "hard_preservation_smoke_v7.json",
        "hard_preservation_smoke_v8.json",
        "hard_preservation_smoke_v9.json",
        "hard_preservation_smoke_v10.json",
        "targeted_preservation_smoke_v11.json",
    }:
        expected_families = {
            "train": {
                "capability_preservation_choice": 40,
                "capability_preservation_numeric": 80,
                "semantic_arithmetic_process": 40,
            },
            "validation": {
                "capability_preservation_choice": 8,
                "capability_preservation_numeric": 16,
                "semantic_arithmetic_process": 8,
            },
        }
        for split, expected_counts in expected_families.items():
            actual_counts = Counter(
                sample.task_family
                for sample in samples
                if sample.split == split
            )
            if dict(sorted(actual_counts.items())) != expected_counts:
                raise SystemExit(
                    f"v7 family counts differ for {split}: {actual_counts}"
                )
        expected_exposure = {
            "hard_preservation_smoke_v7.json": 80,
            "hard_preservation_smoke_v8.json": 160,
            "hard_preservation_smoke_v9.json": 120,
            "hard_preservation_smoke_v10.json": 128,
            "targeted_preservation_smoke_v11.json": 128,
        }[config_path.name]
        if (
            examples_seen != expected_exposure
            or unique_examples_seen != expected_exposure
        ):
            raise SystemExit(
                f"{config_path.name} must expose exactly "
                f"{expected_exposure} unique train samples"
            )
        exposed_family_counts = Counter(
            train_samples[index].task_family
            for index in visited_indices
        )
        expected_exposed_families = {
            "hard_preservation_smoke_v7.json": {
                "capability_preservation_choice": 20,
                "capability_preservation_numeric": 42,
                "semantic_arithmetic_process": 18,
            },
            "hard_preservation_smoke_v8.json": {
                "capability_preservation_choice": 40,
                "capability_preservation_numeric": 80,
                "semantic_arithmetic_process": 40,
            },
            "hard_preservation_smoke_v9.json": {
                "capability_preservation_choice": 27,
                "capability_preservation_numeric": 63,
                "semantic_arithmetic_process": 30,
            },
            "hard_preservation_smoke_v10.json": {
                "capability_preservation_choice": 30,
                "capability_preservation_numeric": 66,
                "semantic_arithmetic_process": 32,
            },
            "targeted_preservation_smoke_v11.json": {
                "capability_preservation_choice": 30,
                "capability_preservation_numeric": 66,
                "semantic_arithmetic_process": 32,
            },
        }[config_path.name]
        if (
            dict(sorted(exposed_family_counts.items()))
            != expected_exposed_families
        ):
            raise SystemExit(
                f"{config_path.name} exposure differs: "
                f"{exposed_family_counts}"
            )
        if max(target_lengths) >= config.generation_max_new_tokens:
            raise SystemExit(
                "v7 generation budget must exceed target length including EOS"
            )
        if dataset.get("policy", {}).get("sealed_canary_used_for_training") is not False:
            raise SystemExit("v7 dataset must exclude the sealed canary")
        if config_path.name == "targeted_preservation_smoke_v11.json":
            base_path = ROOT / "../nano-data-pipeline/datasets/hard_preservation_mix_v5.json"
            base = load_analog_dataset(base_path)
            base_validation = [
                sample
                for sample in base["samples"]
                if sample["split"] == "validation"
            ]
            targeted_validation = [
                sample
                for sample in dataset["samples"]
                if sample["split"] == "validation"
            ]
            if targeted_validation != base_validation:
                raise SystemExit("v11 development rows must be byte-identical to v5")
            if (
                dataset.get("source", {}).get("replacement_count") != 16
                or dataset.get("policy", {}).get("observed_validation_reused")
                is not True
                or dataset.get("policy", {}).get("validation_role")
                != "development_gate_only"
            ):
                raise SystemExit("v11 targeted-data boundary is invalid")
            rows_by_id = {
                sample["sample_id"]: sample for sample in dataset["samples"]
            }
            exposed_targeted = sum(
                (
                    rows_by_id[train_samples[index].sample_id]["generation_rule"]
                    == "targeted_host_two_count_v6"
                )
                for index in visited_indices
            )
            if exposed_targeted != 13:
                raise SystemExit(
                    f"v11 must expose 13 targeted rows, got {exposed_targeted}"
                )

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
                "examples_seen": examples_seen,
                "unique_examples_seen": unique_examples_seen,
                "training_coverage_equivalents": examples_seen / counts["train"],
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
