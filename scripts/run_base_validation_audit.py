#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoTokenizer, Qwen3_5ForCausalLM

from nano_train.data import load_analog_dataset, tokenize_samples
from nano_train.sft import evaluate_exact, set_seed, sha256_file


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_FIELDS = {
    "schema_version",
    "audit_id",
    "model_path",
    "dataset_path",
    "output_dir",
    "seed",
    "max_length",
    "generation_max_new_tokens",
    "expected",
}


def resolve(path: str) -> Path:
    return (ROOT / path).resolve()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = resolve(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if set(config) != EXPECTED_FIELDS:
        raise SystemExit("invalid base validation audit fields")
    if config["schema_version"] != "nano_train_base_validation_audit_v1":
        raise SystemExit("unsupported base validation audit schema")
    model_path = resolve(config["model_path"])
    dataset_path = resolve(config["dataset_path"])
    actual = {
        "dataset_sha256": sha256_file(dataset_path),
        "model_config_sha256": sha256_file(model_path / "config.json"),
    }
    if actual != config["expected"]:
        raise SystemExit(
            f"base validation identity mismatch: actual={actual}, "
            f"expected={config['expected']}"
        )
    if not torch.cuda.is_available():
        raise SystemExit("base validation audit requires CUDA")
    set_seed(int(config["seed"]))
    dataset = load_analog_dataset(dataset_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    samples = tokenize_samples(
        dataset,
        tokenizer,
        max_length=int(config["max_length"]),
    )
    validation = [
        sample for sample in samples if sample.split == "validation"
    ]
    if len(validation) != 32:
        raise SystemExit(f"expected 32 validation samples: {len(validation)}")
    target_lengths = [
        sum(label != -100 for label in sample.labels)
        for sample in validation
    ]
    if max(target_lengths) >= int(config["generation_max_new_tokens"]):
        raise SystemExit("generation budget does not cover target plus EOS")
    model = Qwen3_5ForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
    ).cuda()
    metrics, rows = evaluate_exact(
        model,
        tokenizer,
        validation,
        device=torch.device("cuda"),
        max_new_tokens=int(config["generation_max_new_tokens"]),
    )
    dataset_rows = {
        row["sample_id"]: row for row in dataset["samples"]
    }
    by_family = {}
    for family in sorted(
        {dataset_rows[row["sample_id"]]["task_family"] for row in rows}
    ):
        subset = [
            row
            for row in rows
            if dataset_rows[row["sample_id"]]["task_family"] == family
        ]
        by_family[family] = {
            "samples": len(subset),
            "exact": sum(row["exact"] for row in subset),
            "semantic_exact": sum(
                row["semantic_valid"] for row in subset
            ),
            "exact_failure_sample_ids": [
                row["sample_id"] for row in subset if not row["exact"]
            ],
            "semantic_failure_sample_ids": [
                row["sample_id"]
                for row in subset
                if not row["semantic_valid"]
            ],
        }
    output_lengths = [
        len(
            tokenizer(
                str(row["output"]),
                add_special_tokens=False,
            ).input_ids
        )
        for row in rows
    ]
    output_root = resolve(config["output_dir"])
    output_root.mkdir(parents=True, exist_ok=True)
    generations_path = output_root / "generations.json"
    generations_path.write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    result = {
        "schema_version": "nano_train_base_validation_audit_result_v1",
        "audit_id": config["audit_id"],
        "config_sha256": sha256_file(config_path),
        "identity": actual,
        "validation": metrics,
        "by_family": by_family,
        "contract": {
            "validation_samples": len(validation),
            "max_input_tokens": max(len(sample.input_ids) for sample in samples),
            "target_plus_eos_token_max": max(target_lengths),
            "generation_max_new_tokens": int(
                config["generation_max_new_tokens"]
            ),
            "maximum_output_tokens": max(output_lengths),
            "outputs_at_generation_cap": sum(
                length >= int(config["generation_max_new_tokens"])
                for length in output_lengths
            ),
            "output_token_histogram": dict(
                sorted(Counter(output_lengths).items())
            ),
        },
        "artifacts": {
            "generations_sha256": sha256_file(generations_path),
        },
        "hardware": {
            "gpu_name": torch.cuda.get_device_name(),
            "peak_allocated_gib": (
                torch.cuda.max_memory_allocated() / 2**30
            ),
        },
        "training_performed": False,
    }
    result_path = output_root / "result.json"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
