#!/usr/bin/env python3

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

from nano_train.orca_math_sft import (
    PREREGISTER_SCHEMA,
    build_selection_contract,
    load_config,
)
from nano_train.sft import sha256_file


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/sft/orca_math_smoke_v1.json"
OUTPUT = (
    ROOT / "docs/experiments/orca_math_sft_smoke_v1.preregister.json"
)
MARKDOWN = ROOT / "docs/experiments/orca_math_sft_smoke_v1.md"


def build_receipt() -> dict:
    config = load_config(CONFIG)
    selection = build_selection_contract(config)
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_path,
        local_files_only=True,
    )
    lengths = {}
    target_lengths = {}
    for row in selection["selected_train"] + selection["selected_dev"]:
        messages = row["messages"]
        prompt = tokenizer.apply_chat_template(
            messages[:-1],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        target = messages[-1]["content"] + tokenizer.eos_token
        prompt_ids = tokenizer(
            prompt,
            add_special_tokens=False,
        ).input_ids
        target_ids = tokenizer(
            target,
            add_special_tokens=False,
        ).input_ids
        lengths[row["sample_id"]] = len(prompt_ids) + len(target_ids)
        target_lengths[row["sample_id"]] = len(target_ids)
    if max(lengths.values()) > config.max_length:
        raise ValueError("Orca Math SFT selected row exceeds max_length")
    train_strata = Counter(
        row["stratum"] for row in selection["selected_train"]
    )
    dev_strata = Counter(
        row["stratum"] for row in selection["selected_dev"]
    )
    if (
        dict(train_strata) != config.train_rows_by_stratum
        or dict(dev_strata) != config.dev_rows_by_stratum
    ):
        raise ValueError("Orca Math SFT selected strata differ")
    return {
        "schema_version": PREREGISTER_SCHEMA,
        "experiment_id": config.experiment_id,
        "identity": {
            "config_sha256": sha256_file(CONFIG),
            "model_config_sha256": config.model_config_sha256,
            "dataset_file_sha256": config.dataset_file_sha256,
            "release_manifest_sha256": config.release_manifest_sha256,
        },
        "selection": {
            "train_samples": len(selection["selected_train"]),
            "dev_samples": len(selection["selected_dev"]),
            "train_by_stratum": dict(sorted(train_strata.items())),
            "dev_by_stratum": dict(sorted(dev_strata.items())),
            "train_sample_ids": selection["train_sample_ids"],
            "dev_sample_ids": selection["dev_sample_ids"],
            "train_sample_ids_sha256": selection[
                "train_sample_ids_sha256"
            ],
            "dev_sample_ids_sha256": selection["dev_sample_ids_sha256"],
            "selected_full_sequence_max": max(lengths.values()),
            "selected_target_max": max(target_lengths.values()),
        },
        "training": {
            "model": "Qwen3.5-4B",
            "dtype": config.dtype,
            "seed": config.seed,
            "max_length": config.max_length,
            "optimizer_steps": config.max_steps,
            "micro_batch_size": config.batch_size,
            "gradient_accumulation_steps": (
                config.gradient_accumulation_steps
            ),
            "unique_train_exposures": len(selection["selected_train"]),
            "learning_rate": config.learning_rate,
            "warmup_steps": config.warmup_steps,
            "weight_decay": config.weight_decay,
            "lora_targets": list(config.lora_targets),
            "lora_r": config.lora_r,
            "lora_alpha": config.lora_alpha,
            "gradient_checkpointing": config.gradient_checkpointing,
        },
        "evaluation": {
            "scorer": (
                "last non-empty line must full-match FINAL: <number>; "
                "compare exact rational numeric values"
            ),
            "generation_max_new_tokens": (
                config.generation_max_new_tokens
            ),
            "generation_batch_size": config.generation_batch_size,
            "bootstrap_samples": config.bootstrap_samples,
            "bootstrap_seed": config.bootstrap_seed,
            "exact_mcnemar_alpha": config.alpha,
            "minimum_candidate_only_wins": (
                config.minimum_candidate_only_wins
            ),
            "admission": (
                "positive point delta, positive paired-bootstrap lower "
                "bound, exact McNemar p below alpha, at least six "
                "candidate-only wins, candidate-only > baseline-only, "
                "and every difficulty stratum non-regressing"
            ),
        },
        "decision_boundary": {
            "larger_sft_allowed": False,
            "benchmark_allowed": False,
            "rl_or_opd_allowed": False,
            "forbidden_after_observation": [
                "sample_selection_change",
                "step_change",
                "learning_rate_change",
                "seed_change",
                "lora_scope_change",
                "generation_budget_change",
                "generation_batch_size_change",
                "parser_change",
                "scorer_change",
                "statistical_threshold_change",
                "adapter_weight_change",
            ],
        },
        "execution_boundary": {
            "training_started": False,
            "model_generation_started": False,
            "this_commit_only_preregisters": True,
        },
        "claim_boundary": (
            "This receipt freezes one local synthetic SFT smoke before "
            "training or generation. It is not benchmark, holdout, RL, or "
            "model-quality evidence."
        ),
    }


def render_markdown(receipt: dict) -> str:
    selection = receipt["selection"]
    training = receipt["training"]
    evaluation = receipt["evaluation"]
    return f"""# Orca Math SFT Smoke v1 Pre-Registration

## Frozen Run

- Train: {selection['train_samples']} unique rows across 40 optimizer steps
  and four gradient-accumulation micro-batches;
- development: {selection['dev_samples']} untouched rows;
- max selected sequence:
  {selection['selected_full_sequence_max']} / {training['max_length']} tokens;
- model: {training['model']}, FP32 q/v-only LoRA r={training['lora_r']};
- generation: greedy, batch {evaluation['generation_batch_size']},
  up to {evaluation['generation_max_new_tokens']} new tokens;
- scorer: strict final-line numeric equivalence;
- statistics: {evaluation['bootstrap_samples']:,} paired bootstrap samples
  and exact McNemar alpha {evaluation['exact_mcnemar_alpha']}.

## Admission

{evaluation['admission']}.

Passing unlocks only a separately reviewed next step. Benchmark, independent
holdout, RL, OPD, and post-hoc tuning remain closed.

## Boundary

This commit selects no benchmark rows and starts no training or generation.
"""


def main() -> None:
    receipt = build_receipt()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    MARKDOWN.write_text(render_markdown(receipt), encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
