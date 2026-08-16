#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoTokenizer, Qwen3_5ForCausalLM

from nano_train.data import load_analog_dataset, tokenize_samples
from nano_train.sft import evaluate_exact, sha256_file, sha256_tree


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    config_path = Path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "experiment_id",
        "model_path",
        "model_config_sha256",
        "dataset_path",
        "dataset_sha256",
        "preservation_adapter",
        "preservation_adapter_tree_sha256",
        "capability_adapter",
        "capability_adapter_tree_sha256",
        "preservation_weight",
        "capability_weight",
        "adapter_output",
        "receipt_output",
        "evaluation_output",
        "max_length",
        "generation_max_new_tokens",
    }
    if set(config) != required:
        raise SystemExit("composition config fields differ from contract")
    if config["schema_version"] != "nano_train_lora_composition_smoke_v1":
        raise SystemExit("unsupported composition schema")
    if (
        config["preservation_weight"] != 0.75
        or config["capability_weight"] != 0.25
    ):
        raise SystemExit("composition weights differ from pre-registration")
    model_path = Path(config["model_path"])
    dataset_path = Path(config["dataset_path"])
    adapter_path = Path(config["adapter_output"])
    receipt_path = Path(config["receipt_output"])
    if sha256_file(model_path / "config.json") != config["model_config_sha256"]:
        raise SystemExit("model config identity mismatch")
    if sha256_file(dataset_path) != config["dataset_sha256"]:
        raise SystemExit("dataset identity mismatch")
    if (
        sha256_tree(Path(config["preservation_adapter"]))
        != config["preservation_adapter_tree_sha256"]
        or sha256_tree(Path(config["capability_adapter"]))
        != config["capability_adapter_tree_sha256"]
    ):
        raise SystemExit("source adapter identity mismatch")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (
        receipt["formula"] != "0.75 * delta_v11 + 0.25 * delta_v15"
        or receipt["target_rank"] != 16
        or receipt["target_alpha"] != 16
        or receipt["max_block_error"] != 0.0
        or receipt["output"]["adapter_tree_sha256"] != sha256_tree(adapter_path)
    ):
        raise SystemExit("composition receipt does not authorize evaluation")
    if not torch.cuda.is_available():
        raise SystemExit("composition evaluation requires CUDA")

    tokenizer = AutoTokenizer.from_pretrained(adapter_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset = load_analog_dataset(dataset_path)
    validation = [
        sample
        for sample in tokenize_samples(
            dataset,
            tokenizer,
            max_length=int(config["max_length"]),
        )
        if sample.split == "validation"
    ]
    model = Qwen3_5ForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
    ).cuda()
    model = PeftModel.from_pretrained(
        model,
        adapter_path,
        is_trainable=False,
    ).cuda()
    metrics, rows = evaluate_exact(
        model,
        tokenizer,
        validation,
        device=torch.device("cuda"),
        max_new_tokens=int(config["generation_max_new_tokens"]),
    )
    result = {
        "schema_version": "nano_train_lora_composition_evaluation_v1",
        "experiment_id": config["experiment_id"],
        "config_sha256": sha256_file(config_path),
        "composition_receipt_sha256": sha256_file(receipt_path),
        "adapter_tree_sha256": sha256_tree(adapter_path),
        "dataset_sha256": sha256_file(dataset_path),
        "model_config_sha256": sha256_file(model_path / "config.json"),
        "validation": metrics,
        "generations": rows,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "sealed_canary_run": False,
        "prior_full_suite_run": False,
        "independent_holdout_run": False,
        "independent_holdout_prompts_loaded": False,
        "independent_holdout_references_loaded": False,
    }
    output = Path(args.output or config["evaluation_output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
