#!/usr/bin/env python3

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

from nano_train.sft import sha256_file
from nano_train.swe_code_repair_harbor_sft import (
    PREREGISTER_SCHEMA,
    build_training_contract,
    harbor_training_dataset,
    load_config,
)
from nano_train.swe_code_repair_qualification import build_harbor_prompts


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/sft/swe_code_repair_harbor_prompt_v2.json"
OUTPUT = (
    ROOT
    / "docs/experiments/evoloop_swe_code_repair_harbor_sft_v2.preregister.json"
)
MARKDOWN = ROOT / "docs/experiments/evoloop_swe_code_repair_harbor_sft_v2.md"


def build_receipt() -> dict:
    config = load_config(CONFIG)
    contract = build_training_contract(config)
    source_config = contract["source_config"]
    tokenizer = AutoTokenizer.from_pretrained(
        source_config.model_path,
        local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    prompt_template = Path(config.terminus_prompt_template_path).read_text(
        encoding="utf-8"
    )
    dataset = {
        "dataset_id": "evoloop-swe-code-repair-harbor-sft-v2",
        "samples": (
            harbor_training_dataset(
                contract["selected_train"],
                prompt_template=prompt_template,
                split="train",
            )["samples"]
            + harbor_training_dataset(
                contract["selected_dev"],
                prompt_template=prompt_template,
                split="validation",
            )["samples"]
        ),
    }
    lengths = {}
    target_lengths = {}
    for sample in dataset["samples"]:
        prompt = tokenizer.apply_chat_template(
            sample["messages"][:-1],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        target = sample["messages"][-1]["content"] + tokenizer.eos_token
        prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
        target_ids = tokenizer(target, add_special_tokens=False).input_ids
        lengths[sample["sample_id"]] = len(prompt_ids) + len(target_ids)
        target_lengths[sample["sample_id"]] = len(target_ids)
    if max(lengths.values()) > config.max_length:
        raise ValueError("Harbor-prompt SFT selected row exceeds max_length")
    prompts = build_harbor_prompts(
        tokenizer,
        contract["selected_dev"],
        prompt_template,
    )
    if max(len(row["prompt_ids"]) for row in prompts) > 7_168:
        raise ValueError("Harbor-prompt SFT dev prompt exceeds serving budget")
    train_bands = Counter(
        sample["task_family"]
        for sample in dataset["samples"]
        if sample["split"] == "train"
    )
    dev_bands = Counter(
        sample["task_family"]
        for sample in dataset["samples"]
        if sample["split"] == "validation"
    )
    if (
        dict(train_bands) != config.train_rows_by_band
        or dict(dev_bands) != config.dev_rows_by_band
    ):
        raise ValueError("Harbor-prompt SFT selected bands differ")
    return {
        "schema_version": PREREGISTER_SCHEMA,
        "experiment_id": config.experiment_id,
        "project_name": "EvoLoop",
        "identity": {
            "config_sha256": sha256_file(CONFIG),
            "source_sft_config_sha256": config.source_sft_config_sha256,
            "source_v1_metrics_sha256": config.source_v1_metrics_sha256,
            "source_v1_reload_sha256": config.source_v1_reload_sha256,
            "source_v1_adapter_sha256": config.source_v1_adapter_sha256,
            "qualification_v1_config_sha256": (
                config.qualification_v1_config_sha256
            ),
            "qualification_v1_preregister_sha256": (
                config.qualification_v1_preregister_sha256
            ),
            "qualification_v1_metrics_sha256": (
                config.qualification_v1_metrics_sha256
            ),
            "terminus_parser_sha256": config.terminus_parser_sha256,
            "terminus_prompt_template_sha256": (
                config.terminus_prompt_template_sha256
            ),
        },
        "selection": {
            "train_samples": len(contract["train_sample_ids"]),
            "dev_samples": len(contract["dev_sample_ids"]),
            "train_by_band": dict(sorted(train_bands.items())),
            "dev_by_band": dict(sorted(dev_bands.items())),
            "train_sample_ids": contract["train_sample_ids"],
            "dev_sample_ids": contract["dev_sample_ids"],
            "train_sample_ids_sha256": contract["train_sample_ids_sha256"],
            "dev_sample_ids_sha256": contract["dev_sample_ids_sha256"],
            "excluded_prior_dev_ids_sha256": contract[
                "excluded_dev_ids_sha256"
            ],
            "overlap_with_v1_and_qualification_dev": 0,
            "maximum_full_sequence_tokens": max(lengths.values()),
            "maximum_target_tokens": max(target_lengths.values()),
            "maximum_harbor_dev_prompt_tokens": max(
                len(row["prompt_ids"]) for row in prompts
            ),
        },
        "single_variable_change": {
            "changed": [
                "training prompt is the exact pinned Harbor first-turn template",
                "max_length increases from 2048 to 2816 solely to avoid truncating 34 of the unchanged 160 training rows",
            ],
            "unchanged": [
                "160 training sample IDs and order",
                "40 optimizer steps",
                "four micro-batches per step",
                "learning rate and scheduler",
                "seed",
                "FP32",
                "q/v-only LoRA r=8 alpha=16",
                "source dataset and release",
                "greedy decoding and 768-token output budget",
            ],
        },
        "evaluation": {
            "arms": [
                "Qwen3.5-4B base",
                "standard-prompt SFT v1",
                "Harbor-prompt SFT v2",
            ],
            "dev": "third fresh 24-row release-dev set",
            "loss": "teacher-forced cross entropy under the Harbor prompt",
            "parser": "exact pinned Terminus2 JSON parser",
            "generation": "greedy",
            "max_output_tokens": config.generation_max_new_tokens,
        },
        "admission": {
            "minimum_v2_loss_relative_improvement_vs_base": (
                config.minimum_v2_loss_relative_improvement_vs_base
            ),
            "minimum_v2_loss_relative_improvement_vs_v1": (
                config.minimum_v2_loss_relative_improvement_vs_v1
            ),
            "minimum_v2_parser_valid": config.minimum_v2_parser_valid,
            "maximum_v1_only_losses": config.maximum_v1_only_losses,
            "per_band_non_regression_vs_v1": True,
            "v2_generation_budget_not_exhausted": True,
            "all_losses_and_gradients_finite": True,
            "independent_reload_exact": True,
            "all_gates_required": True,
        },
        "decision_boundary": {
            "passing_unlocks_only_fresh8_sft_screening": True,
            "complete_500_allowed": False,
            "rl_or_opd_allowed": False,
            "forbidden_after_observation": [
                "sample_selection_change",
                "prompt_change",
                "max_length_change",
                "step_change",
                "learning_rate_change",
                "seed_change",
                "lora_scope_change",
                "generation_budget_change",
                "parser_change",
                "admission_gate_change",
                "adapter_weight_change",
            ],
        },
        "execution_boundary": {
            "training_started": False,
            "model_generation_started": False,
            "this_commit_only_preregisters": True,
        },
        "claim_boundary": (
            "This receipt freezes one Harbor-prompt-aligned SFT experiment "
            "before training or generation. Passing admits only frozen "
            "fresh-8 screening and does not establish SWE-bench superiority."
        ),
    }


def render_markdown(receipt: dict) -> str:
    selection = receipt["selection"]
    admission = receipt["admission"]
    return f"""# EvoLoop Harbor-Prompt SFT v2

## Single Variable

Train the same 160 SWE code-repair rows, in the same order and with the same
optimizer, seed, and q/v-only LoRA as v1. The supervision prompt now exactly
matches Harbor Terminus2. `max_length` increases from 2048 to 2816 only because
the full Harbor prompt would otherwise truncate 34 of the unchanged rows.

## Frozen Evidence

- Train rows: {selection['train_samples']} ({selection['train_by_band']});
- fresh dev rows: {selection['dev_samples']} ({selection['dev_by_band']});
- overlap with both prior observed dev sets: 0;
- max full train/dev sequence: {selection['maximum_full_sequence_tokens']} /
  2816 tokens;
- max Harbor dev prompt: {selection['maximum_harbor_dev_prompt_tokens']} /
  7168 tokens.

## Admission

All gates are required: v2 loss improves at least
{admission['minimum_v2_loss_relative_improvement_vs_base']:.0%} versus base and
{admission['minimum_v2_loss_relative_improvement_vs_v1']:.0%} versus v1,
{admission['minimum_v2_parser_valid']}/24 outputs are parser-valid, there are at
most {admission['maximum_v1_only_losses']} v1-only structural wins, no band
regresses, no v2 output exhausts its budget, all training values are finite, and
independent reload is exact.

## Boundary

Passing allows only the frozen fresh-8 SFT screen. It does not support a
SWE-bench score claim or a complete-500 launch.
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
