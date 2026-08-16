#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoTokenizer, Qwen3_5ForCausalLM

from nano_train.data import load_analog_dataset, tokenize_samples
from nano_train.sft import (
    evaluate_exact,
    set_seed,
    sha256_file,
    sha256_tree,
)


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_FIELDS = {
    "schema_version",
    "audit_id",
    "source_pre_registration_revision",
    "source_result_revision",
    "source_config_path",
    "source_metrics_path",
    "source_generations_path",
    "source_reload_path",
    "model_path",
    "adapter_path",
    "dataset_path",
    "output_dir",
    "seed",
    "generation_max_new_tokens",
    "expected",
}
EXPECTED_HASH_FIELDS = {
    "source_config_sha256",
    "source_metrics_sha256",
    "source_generations_sha256",
    "source_reload_sha256",
    "model_config_sha256",
    "adapter_tree_sha256",
    "dataset_sha256",
}


def resolve(path: str) -> Path:
    return (ROOT / path).resolve()


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    unknown = set(config) - EXPECTED_FIELDS
    missing = EXPECTED_FIELDS - set(config)
    if unknown or missing:
        raise ValueError(
            f"invalid audit config fields: unknown={sorted(unknown)}, "
            f"missing={sorted(missing)}"
        )
    if config["schema_version"] != "nano_train_generation_budget_audit_v1":
        raise ValueError("unsupported generation budget audit schema")
    if config["generation_max_new_tokens"] <= 0:
        raise ValueError("generation_max_new_tokens must be positive")
    expected = config["expected"]
    if set(expected) != EXPECTED_HASH_FIELDS:
        raise ValueError("invalid expected hash fields")
    if any(len(value) != 64 for value in expected.values()):
        raise ValueError("every expected identity must be a SHA256")
    return config


def verify_identity(config: dict) -> dict:
    paths = {
        "source_config_sha256": resolve(config["source_config_path"]),
        "source_metrics_sha256": resolve(config["source_metrics_path"]),
        "source_generations_sha256": resolve(
            config["source_generations_path"]
        ),
        "source_reload_sha256": resolve(config["source_reload_path"]),
        "model_config_sha256": resolve(config["model_path"]) / "config.json",
        "dataset_sha256": resolve(config["dataset_path"]),
    }
    actual = {
        name: sha256_file(path) for name, path in paths.items()
    }
    actual["adapter_tree_sha256"] = sha256_tree(
        resolve(config["adapter_path"])
    )
    if actual != config["expected"]:
        raise ValueError(
            f"audit identity mismatch: actual={actual}, "
            f"expected={config['expected']}"
        )
    return actual


def validate_contract(
    config: dict,
    tokenizer,
    validation,
    source_generations: dict,
) -> dict:
    source_rows = source_generations["post_sft"]
    source_ids = [row["sample_id"] for row in source_rows]
    validation_ids = [sample.sample_id for sample in validation]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("source generations contain duplicate validation IDs")
    if set(source_ids) != set(validation_ids):
        raise ValueError("source and audit validation case sets differ")
    target_content_lengths = [
        len(
            tokenizer(
                sample.target,
                add_special_tokens=False,
            ).input_ids
        )
        for sample in validation
    ]
    target_with_eos_lengths = [
        sum(label != -100 for label in sample.labels)
        for sample in validation
    ]
    budget = int(config["generation_max_new_tokens"])
    if budget <= max(target_with_eos_lengths):
        raise ValueError(
            "audit budget must exceed maximum target length including EOS"
        )
    return {
        "validation_samples": len(validation),
        "source_case_set_matches": True,
        "target_content_token_max": max(target_content_lengths),
        "target_with_eos_token_max": max(target_with_eos_lengths),
        "generation_max_new_tokens": budget,
        "budget_above_target_with_eos_max": True,
    }


def load_base_model(config: dict) -> Qwen3_5ForCausalLM:
    return Qwen3_5ForCausalLM.from_pretrained(
        resolve(config["model_path"]),
        local_files_only=True,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
    ).cuda()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = resolve(args.config)
    config = load_config(config_path)
    identity = verify_identity(config)
    if not torch.cuda.is_available():
        raise SystemExit("generation budget audit requires CUDA")
    set_seed(int(config["seed"]))

    tokenizer = AutoTokenizer.from_pretrained(
        resolve(config["adapter_path"]),
        local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset = load_analog_dataset(resolve(config["dataset_path"]))
    validation = [
        sample
        for sample in tokenize_samples(dataset, tokenizer, max_length=128)
        if sample.split == "validation"
    ]
    source_generations = json.loads(
        resolve(config["source_generations_path"]).read_text(
            encoding="utf-8"
        )
    )
    contract = validate_contract(
        config,
        tokenizer,
        validation,
        source_generations,
    )

    device = torch.device("cuda")
    base_model = load_base_model(config)
    base_metrics, base_rows = evaluate_exact(
        base_model,
        tokenizer,
        validation,
        device=device,
        max_new_tokens=int(config["generation_max_new_tokens"]),
    )
    del base_model
    torch.cuda.empty_cache()

    adapter_model = load_base_model(config)
    adapter_model = PeftModel.from_pretrained(
        adapter_model,
        resolve(config["adapter_path"]),
        is_trainable=False,
    ).cuda()
    adapter_metrics, adapter_rows = evaluate_exact(
        adapter_model,
        tokenizer,
        validation,
        device=device,
        max_new_tokens=int(config["generation_max_new_tokens"]),
    )
    output_root = resolve(config["output_dir"])
    output_root.mkdir(parents=True, exist_ok=True)
    generations_path = output_root / "generations.json"
    generations_path.write_text(
        json.dumps(
            {
                "base": base_rows,
                "unchanged_v5_adapter": adapter_rows,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    source_metrics = json.loads(
        resolve(config["source_metrics_path"]).read_text(encoding="utf-8")
    )
    result = {
        "schema_version": "nano_train_generation_budget_audit_result_v1",
        "audit_id": config["audit_id"],
        "config_sha256": sha256_file(config_path),
        "identity": identity,
        "contract": contract,
        "source_official_result": {
            "experiment_id": source_metrics["experiment_id"],
            "generation_max_new_tokens": source_metrics["config"][
                "generation_max_new_tokens"
            ],
            "baseline_validation": source_metrics["baseline_validation"],
            "post_sft_validation": source_metrics["post_sft_validation"],
            "official_score_changed": False,
        },
        "audit_validation": {
            "base": base_metrics,
            "unchanged_v5_adapter": adapter_metrics,
        },
        "generations_sha256": sha256_file(generations_path),
        "peak_allocated_gib": (
            torch.cuda.max_memory_allocated(device) / 2**30
        ),
        "training_performed": False,
        "adapter_modified": False,
        "official_score_changed": False,
        "benchmark_evaluation_allowed": False,
        "rl_allowed": False,
    }
    result_path = output_root / "result.json"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
