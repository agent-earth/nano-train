#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from nano_train.data import tokenize_samples
from nano_train.paired_consistency import (
    JSON_FAMILIES,
    build_selection_contract,
    build_step_schedule,
    load_config,
    supervised_target_labels,
)
from nano_train.sft import sha256_file


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs/paired_consistency/execution_target_consistency_v1.json"
)
OUTPUT = (
    ROOT
    / "docs/experiments/execution_target_paired_consistency_v1.preregister.json"
)


def main() -> None:
    config = load_config(CONFIG)
    selection = build_selection_contract(config)
    schedule = build_step_schedule(selection)
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_path,
        local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenized = tokenize_samples(
        selection["dataset"],
        tokenizer,
        max_length=config.max_length,
    )
    by_id = {sample.sample_id: sample for sample in tokenized}
    selected_ids = {
        sample_id
        for pair in selection["pair_schedule"]
        for sample_id in (
            pair["process_sample_id"],
            pair["final_sample_id"],
        )
    } | set(selection["json_schedule"])
    full_lengths = {
        sample_id: len(by_id[sample_id].input_ids)
        for sample_id in selected_ids
    }
    target_lengths = {
        sample_id: sum(label != -100 for label in by_id[sample_id].labels)
        for sample_id in selected_ids
    }
    heldout_lengths = {
        sample_id: len(by_id[sample_id].input_ids)
        for sample_id in selection["heldout_sample_ids"]
    }
    heldout_family = {}
    raw_by_id = selection["raw_by_id"]
    for sample_id in selection["heldout_sample_ids"]:
        row = raw_by_id[sample_id]
        key = row["view"] if row["pair_id"] else row["task_family"]
        heldout_family[key] = heldout_family.get(key, 0) + 1
    if heldout_family != {
        "process": 24,
        "final": 24,
        **{family: 8 for family in JSON_FAMILIES},
    }:
        raise ValueError("paired consistency heldout composition differs")
    pair_suffix_alignment = []
    for pair in selection["pair_schedule"]:
        process_labels = supervised_target_labels(
            torch.tensor(
                [by_id[pair["process_sample_id"]].labels]
            )
        )
        final_labels = supervised_target_labels(
            torch.tensor(
                [by_id[pair["final_sample_id"]].labels]
            )
        )
        aligned = (
            process_labels.shape[0] >= final_labels.shape[0]
            and torch.equal(
                process_labels[-final_labels.shape[0] :],
                final_labels,
            )
        )
        pair_suffix_alignment.append(
            {"pair_id": pair["pair_id"], "aligned": aligned}
        )
    if not all(row["aligned"] for row in pair_suffix_alignment):
        raise ValueError("paired consistency target suffix alignment fails")

    result = {
        "schema_version": "nano_train_paired_consistency_preregister_v1",
        "experiment_id": config.experiment_id,
        "identity": {
            "config_sha256": sha256_file(CONFIG),
            "dataset_file_sha256": config.dataset_file_sha256,
            "dataset_canonical_sha256": config.dataset_canonical_sha256,
            "release_manifest_sha256": config.release_manifest_sha256,
            "prior_standard_config_sha256": (
                config.prior_standard_config_sha256
            ),
            "model_config_sha256": config.model_config_sha256,
        },
        "selection": {
            "heldout_samples": len(selection["heldout_sample_ids"]),
            "heldout_sample_ids": selection["heldout_sample_ids"],
            "heldout_composition": heldout_family,
            "pair_schedule": selection["pair_schedule"],
            "json_schedule": selection["json_schedule"],
            "step_schedule": schedule,
            "hashes": selection["hashes"],
            "selected_full_sequence_max": max(full_lengths.values()),
            "selected_target_max": max(target_lengths.values()),
            "heldout_full_sequence_max": max(heldout_lengths.values()),
            "pair_suffix_alignment": pair_suffix_alignment,
            "pair_suffix_alignment_pass": True,
        },
        "objective": {
            "pair_step_total": (
                "0.5 * process_ce + 0.5 * final_ce + "
                "1.0 * KL(detach(process_final_logits) || final_logits)"
            ),
            "json_step_total": "json_ce",
            "teacher_detach": config.teacher_detach,
            "temperature": config.consistency_temperature,
            "process_ce_weight": config.process_ce_weight,
            "final_ce_weight": config.final_ce_weight,
            "consistency_weight": config.consistency_weight,
            "pair_steps": config.train_pair_count,
            "json_steps": len(selection["json_schedule"]),
            "total_steps": config.max_steps,
            "gradient_policy": (
                "process CE backward, detach aligned teacher suffix logits, "
                "then final CE plus KL backward before one optimizer step"
            ),
        },
        "training": {
            "model": "Qwen3.5-4B",
            "dtype": config.dtype,
            "seed": config.seed,
            "max_length": config.max_length,
            "generation_max_new_tokens": config.generation_max_new_tokens,
            "learning_rate": config.learning_rate,
            "warmup_steps": config.warmup_steps,
            "lora_targets": list(config.lora_targets),
            "lora_r": config.lora_r,
            "lora_alpha": config.lora_alpha,
            "gradient_checkpointing": config.gradient_checkpointing,
        },
        "decision_rule": {
            "method_accepted": (
                "stable_and_reloadable AND aggregate_verified_delta > 0 "
                "AND final_view_verified_delta > 0 "
                "AND pair_both_verified_delta > 0 "
                "AND every_JSON_family_post_verified >= baseline_verified"
            ),
            "larger_training_allowed": False,
            "benchmark_allowed": False,
            "independent_holdout_allowed": False,
            "rl_allowed": False,
            "forbidden_after_observation": [
                "loss_weight_search",
                "temperature_search",
                "teacher_detach_search",
                "heldout_selection_search",
                "pair_schedule_search",
                "json_schedule_search",
                "step_search",
                "learning_rate_search",
                "seed_search",
                "lora_scope_search",
                "prompt_search",
                "parser_search",
                "adapter_weight_search",
            ],
        },
        "execution_boundary": {
            "training_started": False,
            "model_generation_started": False,
            "this_commit_only_preregisters": True,
        },
        "claim_boundary": (
            "This receipt freezes the consistency objective, fresh heldout, "
            "training schedule, identities, and decision gates before any "
            "model training or generation."
        ),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
