#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from safetensors import safe_open
from transformers import AutoTokenizer, Qwen3_5ForCausalLM

from nano_train.continuation import load_config
from nano_train.data import load_analog_dataset, tokenize_samples
from nano_train.sft import evaluate_exact, sha256_file, sha256_tree


def tensor_audit(anchor: Path, candidate: Path) -> dict[str, int]:
    with safe_open(anchor, framework="pt", device="cpu") as left, safe_open(
        candidate, framework="pt", device="cpu"
    ) as right:
        if set(left.keys()) != set(right.keys()):
            raise ValueError("adapter tensor keys differ")
        a_changed = 0
        b_changed = 0
        nonfinite = 0
        for key in left.keys():
            candidate_tensor = right.get_tensor(key)
            changed = not bool(
                torch.equal(left.get_tensor(key), candidate_tensor)
            )
            if ".lora_A." in key:
                a_changed += int(changed)
            elif ".lora_B." in key:
                b_changed += int(changed)
            else:
                raise ValueError(f"unexpected adapter tensor: {key}")
            nonfinite += int(not bool(torch.isfinite(candidate_tensor).all()))
    return {
        "a_tensors_changed": a_changed,
        "b_tensors_changed": b_changed,
        "nonfinite_tensors": nonfinite,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = Path(args.config)
    config = load_config(config_path)
    if not torch.cuda.is_available():
        raise SystemExit("adapter reload validation requires CUDA")

    output_root = Path(config.output_dir)
    adapter = output_root / "adapter"
    metrics_path = output_root / "metrics.json"
    if not adapter.is_dir() or not metrics_path.is_file():
        raise SystemExit("continuation artifacts are incomplete")
    if (output_root / "failure.json").exists():
        raise SystemExit("refusing adapter with a failure receipt")
    if sha256_file(Path(config.model_path) / "config.json") != (
        config.model_config_sha256
    ):
        raise SystemExit("model config identity mismatch")
    if sha256_file(Path(config.dataset_path)) != config.dataset_sha256:
        raise SystemExit("dataset identity mismatch")
    if sha256_tree(Path(config.anchor_adapter)) != (
        config.anchor_adapter_tree_sha256
    ):
        raise SystemExit("anchor identity mismatch")

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(adapter, local_files_only=True)
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
    model = Qwen3_5ForCausalLM.from_pretrained(
        config.model_path,
        local_files_only=True,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
    ).cuda()
    model = PeftModel.from_pretrained(
        model,
        adapter,
        is_trainable=False,
    ).cuda()
    validation_metrics, _ = evaluate_exact(
        model,
        tokenizer,
        validation,
        device=torch.device("cuda"),
        max_new_tokens=config.generation_max_new_tokens,
    )
    if validation_metrics != metrics["post_validation"]:
        raise SystemExit("reload validation differs from training result")

    audit = tensor_audit(
        Path(config.anchor_adapter) / "adapter_model.safetensors",
        adapter / "adapter_model.safetensors",
    )
    receipt = {
        "schema_version": "nano_train_anchored_adapter_reload_v1",
        "experiment_id": config.experiment_id,
        "config_sha256": sha256_file(config_path),
        "anchor_adapter_tree_sha256": sha256_tree(Path(config.anchor_adapter)),
        "adapter_tree_sha256": sha256_tree(adapter),
        "reload_success": True,
        "reload_matches_training": True,
        "validation": validation_metrics,
        "tensor_audit": audit,
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
