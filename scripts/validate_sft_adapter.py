#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoTokenizer, Qwen3_5ForCausalLM

from nano_train.config import load_sft_smoke_config
from nano_train.data import (
    load_analog_dataset,
    load_execution_target_dataset,
    load_skill_release_dataset,
    tokenize_samples,
)
from nano_train.sft import evaluate_exact, sha256_tree


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_sft_smoke_config(args.config)
    if not torch.cuda.is_available():
        raise SystemExit("adapter validation requires CUDA")

    output_root = Path(config.output_dir)
    adapter = output_root / "adapter"
    if not adapter.is_dir():
        raise SystemExit(f"missing adapter: {adapter}")
    if (output_root / "failure.json").exists():
        raise SystemExit("refusing to validate an adapter with a failure receipt")

    dtype = {
        "float16": torch.float16,
        "float32": torch.float32,
    }[config.dtype]
    tokenizer = AutoTokenizer.from_pretrained(adapter, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if config.dataset_schema == "skill_release_jsonl_v1":
        dataset = load_skill_release_dataset(
            config.dataset_path,
            config.release_manifest_path or "",
            train_samples_per_family=config.train_samples_per_family or 0,
            validation_samples_per_family=(
                config.validation_samples_per_family or 0
            ),
            validation_start_per_family=(
                config.validation_start_per_family
            ),
        )
    elif config.dataset_schema == "execution_target_json_v1":
        dataset = load_execution_target_dataset(
            config.dataset_path,
            config.release_manifest_path or "",
        )
    else:
        dataset = load_analog_dataset(config.dataset_path)
    validation = [
        sample
        for sample in tokenize_samples(
            dataset,
            tokenizer,
            max_length=config.max_length,
        )
        if sample.split == "validation"
    ]

    model = Qwen3_5ForCausalLM.from_pretrained(
        config.model_path,
        local_files_only=True,
        dtype=dtype,
        low_cpu_mem_usage=True,
    ).cuda()
    model = PeftModel.from_pretrained(
        model,
        adapter,
        is_trainable=False,
    ).cuda()
    metrics, _ = evaluate_exact(
        model,
        tokenizer,
        validation,
        device=torch.device("cuda"),
        max_new_tokens=config.generation_max_new_tokens,
    )
    receipt = {
        "schema_version": "nano_train_adapter_reload_v1",
        "experiment_id": config.experiment_id,
        "adapter_sha256": sha256_tree(adapter),
        "reload_success": True,
        "validation": metrics,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
    }
    path = output_root / "reload_validation.json"
    path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
