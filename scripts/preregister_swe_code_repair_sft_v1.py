#!/usr/bin/env python3

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

from nano_train.sft import sha256_file
from nano_train.swe_code_repair_sft import (
    PREREGISTER_SCHEMA,
    build_selection_contract,
    load_config,
    token_band,
    valid_json_action,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/sft/swe_code_repair_smoke_v1.json"
OUTPUT = (
    ROOT
    / "docs/experiments/evoloop_swe_code_repair_sft_smoke_v1.preregister.json"
)
MARKDOWN = (
    ROOT / "docs/experiments/evoloop_swe_code_repair_sft_smoke_v1.md"
)


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
        if not valid_json_action(messages[-1]["content"]):
            raise ValueError("selected target is not a valid JSON action")
        if token_band(int(row["token_count"])) not in {
            "short",
            "medium",
            "long",
        }:
            raise ValueError("selected row has an unsupported token band")
    if max(lengths.values()) > config.max_length:
        raise ValueError("selected SWE code-repair row exceeds max_length")

    train_bands = Counter(
        token_band(int(row["token_count"]))
        for row in selection["selected_train"]
    )
    dev_bands = Counter(
        token_band(int(row["token_count"]))
        for row in selection["selected_dev"]
    )
    if (
        dict(train_bands) != config.train_rows_by_band
        or dict(dev_bands) != config.dev_rows_by_band
    ):
        raise ValueError("selected SWE code-repair bands differ")

    return {
        "schema_version": PREREGISTER_SCHEMA,
        "experiment_id": config.experiment_id,
        "project_name": "EvoLoop",
        "identity": {
            "config_sha256": sha256_file(CONFIG),
            "model_config_sha256": config.model_config_sha256,
            "dataset_file_sha256": config.dataset_file_sha256,
            "release_manifest_sha256": config.release_manifest_sha256,
        },
        "selection": {
            "train_samples": len(selection["selected_train"]),
            "dev_samples": len(selection["selected_dev"]),
            "train_by_band": dict(sorted(train_bands.items())),
            "dev_by_band": dict(sorted(dev_bands.items())),
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
            "heldout_measure": "teacher_forced_cross_entropy",
            "generation": "greedy",
            "generation_max_new_tokens": (
                config.generation_max_new_tokens
            ),
            "json_action_scorer": (
                "exact JSON object with non-empty analysis and plan plus "
                "one or more positive-duration keystroke commands"
            ),
            "admission": {
                "minimum_dev_loss_relative_improvement": (
                    config.minimum_dev_loss_relative_improvement
                ),
                "minimum_json_action_valid_rate": (
                    config.minimum_json_action_valid_rate
                ),
                "all_losses_finite": True,
                "all_gradient_norms_finite": True,
                "adapter_saved": True,
                "failure_receipt_absent": True,
                "independent_reload_loss_and_generations_exact": True,
            },
        },
        "decision_boundary": {
            "fresh8_allowed_only_after_all_admission_gates": True,
            "complete_500_allowed": False,
            "rl_or_opd_allowed": False,
            "forbidden_after_observation": [
                "sample_selection_change",
                "step_change",
                "learning_rate_change",
                "seed_change",
                "lora_scope_change",
                "generation_budget_change",
                "json_action_scorer_change",
                "adapter_weight_change",
            ],
        },
        "execution_boundary": {
            "training_started": False,
            "model_generation_started": False,
            "this_commit_only_preregisters": True,
        },
        "claim_boundary": (
            "This receipt freezes one local SWE-specific LoRA smoke using "
            "SWE-bench train rows that passed exact and near Verified-500 "
            "exclusion. Passing admits only frozen fresh-8 screening; it "
            "does not prove SWE-bench improvement or model superiority."
        ),
    }


def render_markdown(receipt: dict) -> str:
    selection = receipt["selection"]
    training = receipt["training"]
    admission = receipt["evaluation"]["admission"]
    return f"""# EvoLoop SWE Code-Repair SFT Smoke v1

## Frozen Run

- Model: Qwen3.5-4B, FP32 q/v-only LoRA r={training['lora_r']}.
- Training: {selection['train_samples']} unique rows, {training['optimizer_steps']}
  optimizer steps, accumulation {training['gradient_accumulation_steps']}.
- Development: {selection['dev_samples']} held-out rows, four per length band.
- Longest selected sequence:
  {selection['selected_full_sequence_max']} / {training['max_length']} tokens.
- Dataset and release identities are pinned in the machine-readable receipt.
- Every selected target is a valid Terminus JSON action.

## Admission

The adapter reaches fresh-8 only if held-out teacher-forced loss improves by at
least {admission['minimum_dev_loss_relative_improvement']:.1%}, at least
{admission['minimum_json_action_valid_rate']:.0%} of held-out generations are
valid JSON actions, all numerical and artifact gates pass, and an independent
reload reproduces both held-out loss and all 12 generation receipts.

## Boundary

The source is the public SWE-bench train split. The data release separately
proved zero exact instance, problem, patch, and near-problem overlap with all
500 SWE-bench Verified cases. This smoke does not itself prove benchmark
improvement. Full-500, RL, OPD, and post-hoc tuning remain closed.
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
