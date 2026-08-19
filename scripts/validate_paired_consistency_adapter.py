#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoTokenizer, Qwen3_5ForCausalLM

from nano_train.data import tokenize_samples
from nano_train.paired_consistency import build_selection_contract, load_config
from nano_train.sft import evaluate_exact, sha256_file, sha256_tree


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = Path(args.config)
    config = load_config(config_path)
    if not torch.cuda.is_available():
        raise SystemExit("paired consistency adapter validation requires CUDA")
    output_root = Path(config.output_dir)
    adapter = output_root / "adapter"
    if not adapter.is_dir():
        raise SystemExit(f"missing adapter: {adapter}")
    if (output_root / "failure.json").exists():
        raise SystemExit("refusing to validate an adapter with a failure receipt")

    selection = build_selection_contract(config)
    tokenizer = AutoTokenizer.from_pretrained(adapter, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenized = tokenize_samples(
        selection["dataset"],
        tokenizer,
        max_length=config.max_length,
    )
    by_id = {sample.sample_id: sample for sample in tokenized}
    heldout = [
        by_id[sample_id] for sample_id in selection["heldout_sample_ids"]
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
    metrics, rows = evaluate_exact(
        model,
        tokenizer,
        heldout,
        device=torch.device("cuda"),
        max_new_tokens=config.generation_max_new_tokens,
    )
    receipt = {
        "schema_version": "nano_train_paired_consistency_reload_v1",
        "experiment_id": config.experiment_id,
        "config_sha256": sha256_file(config_path),
        "adapter_sha256": sha256_tree(adapter),
        "heldout_sample_id_sha256": selection["hashes"][
            "heldout_sample_id_sha256"
        ],
        "reload_success": True,
        "validation": metrics,
        "generations": rows,
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
