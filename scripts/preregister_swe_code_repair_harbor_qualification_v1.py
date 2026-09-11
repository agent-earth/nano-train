#!/usr/bin/env python3

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

from nano_train.sft import sha256_file
from nano_train.swe_code_repair_qualification import (
    PREREGISTER_SCHEMA,
    build_harbor_prompts,
    build_qualification_contract,
    load_config,
)
from nano_train.swe_code_repair_sft import token_band


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT / "configs/eval/swe_code_repair_harbor_qualification_v1.json"
)
OUTPUT = (
    ROOT
    / "docs/experiments/"
    "evoloop_swe_code_repair_harbor_qualification_v1.preregister.json"
)
MARKDOWN = (
    ROOT
    / "docs/experiments/evoloop_swe_code_repair_harbor_qualification_v1.md"
)


def build_receipt() -> dict:
    config = load_config(CONFIG)
    contract = build_qualification_contract(config)
    source_config = contract["source_config"]
    tokenizer = AutoTokenizer.from_pretrained(
        config.adapter_path,
        local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    prompt_template = Path(config.terminus_prompt_template_path).read_text(
        encoding="utf-8"
    )
    prompts = build_harbor_prompts(
        tokenizer,
        contract["selected_dev"],
        prompt_template,
    )
    full_lengths = {}
    target_lengths = {}
    for row in contract["selected_dev"]:
        messages = row["messages"]
        training_prompt = tokenizer.apply_chat_template(
            messages[:-1],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        target = messages[-1]["content"] + tokenizer.eos_token
        full_lengths[row["sample_id"]] = len(
            tokenizer(
                training_prompt + target,
                add_special_tokens=False,
            ).input_ids
        )
        target_lengths[row["sample_id"]] = len(
            tokenizer(target, add_special_tokens=False).input_ids
        )
    if max(full_lengths.values()) > source_config.max_length:
        raise ValueError("fresh qualification target exceeds SFT max_length")
    if max(len(row["prompt_ids"]) for row in prompts) > config.max_input_tokens:
        raise ValueError("fresh qualification Harbor prompt exceeds budget")
    bands = Counter(
        token_band(int(row["token_count"]))
        for row in contract["selected_dev"]
    )
    if dict(bands) != config.dev_rows_by_band:
        raise ValueError("fresh qualification bands differ")
    return {
        "schema_version": PREREGISTER_SCHEMA,
        "experiment_id": config.experiment_id,
        "project_name": "EvoLoop",
        "identity": {
            "config_sha256": sha256_file(CONFIG),
            "source_sft_config_sha256": config.source_sft_config_sha256,
            "source_metrics_sha256": config.source_metrics_sha256,
            "source_reload_sha256": config.source_reload_sha256,
            "adapter_sha256": config.adapter_sha256,
            "terminus_parser_sha256": config.terminus_parser_sha256,
            "terminus_prompt_template_sha256": (
                config.terminus_prompt_template_sha256
            ),
        },
        "selection": {
            "dev_samples": len(contract["selected_dev_ids"]),
            "dev_by_band": dict(sorted(bands.items())),
            "dev_sample_ids": contract["selected_dev_ids"],
            "dev_sample_ids_sha256": contract["selected_dev_ids_sha256"],
            "prior_v1_dev_sample_ids_sha256": config.prior_dev_ids_sha256,
            "overlap_with_prior_v1_dev": 0,
            "maximum_training_style_sequence_tokens": max(
                full_lengths.values()
            ),
            "maximum_target_tokens": max(target_lengths.values()),
            "maximum_harbor_prompt_tokens": max(
                len(row["prompt_ids"]) for row in prompts
            ),
        },
        "protocol": {
            "arms": [
                "Qwen3.5-4B base",
                "Qwen3.5-4B plus frozen adapter",
            ],
            "prompt": "exact pinned Terminus2 first-turn JSON template",
            "parser": "exact pinned Terminus2 JSON parser",
            "temperature": 0,
            "thinking": False,
            "max_input_tokens": config.max_input_tokens,
            "max_output_tokens": config.max_output_tokens,
            "measurements": [
                "teacher-forced loss on original training-style messages",
                "Terminus parser validity",
                "actionable command or completion",
                "paired adapter-only wins and base-only losses",
                "per-band non-regression",
                "generation budget exhaustion",
            ],
        },
        "admission": {
            "minimum_relative_loss_improvement": (
                config.minimum_relative_loss_improvement
            ),
            "minimum_adapter_parser_valid": (
                config.minimum_adapter_parser_valid
            ),
            "minimum_adapter_only_wins": (
                config.minimum_adapter_only_wins
            ),
            "maximum_base_only_losses": config.maximum_base_only_losses,
            "per_band_non_regression": True,
            "adapter_generation_budget_not_exhausted": True,
            "source_reload_exact": True,
            "all_gates_required": True,
        },
        "decision_boundary": {
            "passing_unlocks_only_fresh8_sft_screening": True,
            "complete_500_allowed": False,
            "rl_or_opd_allowed": False,
            "no_training_or_adapter_change": True,
            "forbidden_after_observation": [
                "case_selection_change",
                "prompt_change",
                "parser_change",
                "loss_threshold_change",
                "parser_valid_threshold_change",
                "paired_win_threshold_change",
                "per_band_gate_change",
                "generation_budget_change",
                "adapter_weight_change",
            ],
        },
        "execution_boundary": {
            "qualification_started": False,
            "model_generation_started": False,
            "this_commit_only_preregisters": True,
        },
        "claim_boundary": (
            "This is a fresh local qualification of an unchanged adapter. "
            "Passing admits only the already frozen fresh-8 SWE-bench screen; "
            "it does not establish benchmark improvement or superiority."
        ),
    }


def render_markdown(receipt: dict) -> str:
    selection = receipt["selection"]
    admission = receipt["admission"]
    return f"""# EvoLoop SWE Code-Repair Harbor Qualification v1

## Frozen Evidence

- 24 previously unobserved release-dev rows: 8 short, 8 medium, 8 long;
- zero overlap with the 12 development rows used by SFT smoke v1;
- unchanged base model and unchanged adapter;
- exact Harbor Terminus2 prompt template and JSON parser pinned by SHA256;
- greedy decoding, 7168 input tokens, 768 output tokens, thinking disabled.

## Admission

The adapter reaches fresh-8 only if all conditions pass:

- held-out loss improves by at least
  {admission['minimum_relative_loss_improvement']:.0%};
- at least {admission['minimum_adapter_parser_valid']}/24 outputs are accepted
  by the exact Harbor parser;
- at least {admission['minimum_adapter_only_wins']} paired adapter-only format
  repair and at most {admission['maximum_base_only_losses']} base-only loss;
- no short, medium, or long band regresses;
- no adapter output exhausts the 768-token budget;
- the source adapter has an exact independent reload receipt.

## Boundary

No training occurs in this experiment. Passing only allows fresh-8 SFT
screening; it does not support a SWE-bench performance claim or a complete-500
launch.
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
