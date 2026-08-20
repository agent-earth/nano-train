#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoTokenizer, Qwen3_5ForCausalLM

from nano_train.data import load_analog_dataset, tokenize_samples
from nano_train.router_negative_diversity import load_config
from nano_train.sft import evaluate_exact, sha256_file, sha256_tree


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/sft/router_negative_diversity_v2.json",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    if not torch.cuda.is_available():
        raise SystemExit("router reload validation requires CUDA")

    output_root = Path(config.output_dir)
    adapter = output_root / "adapter"
    metrics_path = output_root / "metrics.json"
    generations_path = output_root / "generations.json"
    if (
        not adapter.is_dir()
        or not metrics_path.is_file()
        or not generations_path.is_file()
    ):
        raise SystemExit("router reload validation artifacts are incomplete")
    if (output_root / "failure.json").exists():
        raise SystemExit("refusing to validate an adapter with a failure receipt")

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    source_generations = json.loads(
        generations_path.read_text(encoding="utf-8")
    )
    tokenizer = AutoTokenizer.from_pretrained(
        adapter,
        local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
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
    if len(validation) != 1536:
        raise ValueError("router reload validation row count differs")

    dtype = {
        "float16": torch.float16,
        "float32": torch.float32,
    }[config.dtype]
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
    validation_metrics, validation_rows = evaluate_exact(
        model,
        tokenizer,
        validation,
        device=torch.device("cuda"),
        max_new_tokens=config.generation_max_new_tokens,
    )
    reload_generations = output_root / "reload_generations.json"
    reload_generations.write_text(
        json.dumps(validation_rows, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    generations_exact = validation_rows == source_generations["post_sft"]
    metrics_exact = validation_metrics == metrics["post_sft_validation"]
    if not generations_exact or not metrics_exact:
        raise ValueError("router independent reload differs from post SFT")
    receipt = {
        "schema_version": "nano_train_router_reload_v2",
        "experiment_id": config.experiment_id,
        "adapter_sha256": sha256_tree(adapter),
        "source_generations_sha256": sha256_file(generations_path),
        "reload_generations_sha256": sha256_file(reload_generations),
        "reload_success": True,
        "metrics_exact": True,
        "generations_exact": True,
        "validation": validation_metrics,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
    }
    receipt_path = output_root / "reload_validation.json"
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
