#!/usr/bin/env python3

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from nano_train.data import tokenize_samples
from nano_train.paired_consistency import (
    JSON_FAMILIES,
    PairedConsistencyConfig,
    build_selection_contract,
    build_step_schedule,
    frozen_method_contract,
    load_config,
    supervised_target_labels,
)
from nano_train.sft import sha256_file


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs/paired_consistency/consistency_replication_v1.json"
)
OUTPUT = (
    ROOT
    / "docs/experiments/consistency_replication_v1.preregister.json"
)
DATA_COMMIT = "2d3cfae"


def _pair_suffix_alignment(
    pairs: list[dict[str, str]],
    by_id: dict[str, Any],
) -> list[dict[str, Any]]:
    result = []
    for pair in pairs:
        process_labels = supervised_target_labels(
            torch.tensor([by_id[pair["process_sample_id"]].labels])
        )
        final_labels = supervised_target_labels(
            torch.tensor([by_id[pair["final_sample_id"]].labels])
        )
        aligned = (
            process_labels.shape[0] >= final_labels.shape[0]
            and torch.equal(
                process_labels[-final_labels.shape[0] :],
                final_labels,
            )
        )
        result.append({"pair_id": pair["pair_id"], "aligned": aligned})
    return result


def _heldout_composition(
    sample_ids: list[str],
    raw_by_id: dict[str, dict[str, Any]],
) -> dict[str, int]:
    composition = Counter()
    for sample_id in sample_ids:
        row = raw_by_id[sample_id]
        key = row["view"] if row["pair_id"] else row["task_family"]
        composition[key] += 1
    return dict(sorted(composition.items()))


def build_receipt(
    config_path: Path = CONFIG,
    *,
    tokenizer: Any | None = None,
) -> dict[str, Any]:
    config = load_config(config_path)
    prior_config = load_config(config.prior_standard_config_path)
    method_contract = frozen_method_contract(config)
    prior_method_contract = frozen_method_contract(prior_config)
    if method_contract != prior_method_contract:
        raise ValueError("replication method differs from consistency v1")

    selection = build_selection_contract(config)
    schedule = build_step_schedule(selection)
    if tokenizer is None:
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
    heldout_ids = set(selection["heldout_sample_ids"])
    selected_lengths = {
        sample_id: len(by_id[sample_id].input_ids)
        for sample_id in selected_ids
    }
    selected_target_lengths = {
        sample_id: sum(
            label != -100 for label in by_id[sample_id].labels
        )
        for sample_id in selected_ids
    }
    heldout_lengths = {
        sample_id: len(by_id[sample_id].input_ids)
        for sample_id in heldout_ids
    }
    heldout_composition = _heldout_composition(
        selection["heldout_sample_ids"],
        selection["raw_by_id"],
    )
    expected_composition = {
        "final": config.heldout_pair_count,
        "process": config.heldout_pair_count,
        **{
            family: config.heldout_json_per_family
            for family in JSON_FAMILIES
        },
    }
    if heldout_composition != dict(sorted(expected_composition.items())):
        raise ValueError("replication heldout composition differs")

    train_alignment = _pair_suffix_alignment(
        selection["pair_schedule"],
        by_id,
    )
    heldout_alignment = _pair_suffix_alignment(
        selection["heldout_pair_schedule"],
        by_id,
    )
    if not all(
        row["aligned"] for row in train_alignment + heldout_alignment
    ):
        raise ValueError("replication target suffix alignment fails")

    release = json.loads(
        Path(config.release_manifest_path).read_text(encoding="utf-8")
    )
    if release["accepted"] != {
        "rows": 1152,
        "train_rows": 640,
        "dev_rows": 512,
        "train_pairs": 192,
        "dev_pairs": 192,
        "train_json_by_family": {
            family: 64 for family in JSON_FAMILIES
        },
        "dev_json_by_family": {
            family: 32 for family in JSON_FAMILIES
        },
        "train_tokens": 405007,
    }:
        raise ValueError("replication release composition differs")

    return {
        "schema_version": (
            "nano_train_paired_consistency_replication_preregister_v1"
        ),
        "experiment_id": config.experiment_id,
        "identity": {
            "data_commit": DATA_COMMIT,
            "config_sha256": sha256_file(config_path),
            "dataset_file_sha256": config.dataset_file_sha256,
            "dataset_canonical_sha256": (
                config.dataset_canonical_sha256
            ),
            "release_manifest_sha256": config.release_manifest_sha256,
            "source_result_sha256": release["source"][
                "source_result_sha256"
            ],
            "prior_consistency_config_sha256": (
                config.prior_standard_config_sha256
            ),
            "model_config_sha256": config.model_config_sha256,
        },
        "method_lock": {
            "matches_execution_target_consistency_v1": True,
            "frozen_fields": method_contract,
            "objective": {
                "pair_step_total": (
                    "0.5 * process_ce + 0.5 * final_ce + "
                    "1.0 * KL(detach(process_final_logits) || final_logits)"
                ),
                "json_step_total": "json_ce",
                "gradient_policy": (
                    "process CE backward, detach aligned teacher suffix "
                    "logits, then final CE plus KL backward before one "
                    "optimizer step"
                ),
            },
        },
        "release": {
            "release_id": release["release_id"],
            "accepted": release["accepted"],
            "checks_passed": sum(release["checks"].values()),
            "checks_total": len(release["checks"]),
            "training_unblocked": release["training_unblocked"],
            "scope": (
                "release eligibility only; it is not evidence that the "
                "model method improves"
            ),
        },
        "selection": {
            "train_pair_steps": len(selection["pair_schedule"]),
            "train_json_steps": len(selection["json_schedule"]),
            "optimizer_steps": len(schedule),
            "pair_schedule": selection["pair_schedule"],
            "json_schedule": selection["json_schedule"],
            "step_schedule": schedule,
            "heldout_samples": len(selection["heldout_sample_ids"]),
            "heldout_pairs": len(selection["heldout_pair_schedule"]),
            "heldout_sample_ids": selection["heldout_sample_ids"],
            "heldout_pair_schedule": selection["heldout_pair_schedule"],
            "heldout_composition": heldout_composition,
            "hashes": selection["hashes"],
            "selected_full_sequence_max": max(
                selected_lengths.values()
            ),
            "selected_target_max": max(
                selected_target_lengths.values()
            ),
            "heldout_full_sequence_max": max(
                heldout_lengths.values()
            ),
            "max_length_bound_pass": max(
                max(selected_lengths.values()),
                max(heldout_lengths.values()),
            )
            <= config.max_length,
            "train_pair_suffix_alignment": train_alignment,
            "heldout_pair_suffix_alignment": heldout_alignment,
            "pair_suffix_alignment_pass": True,
            "train_heldout_overlap": len(selected_ids & heldout_ids),
        },
        "significance": {
            "bootstrap_samples": config.bootstrap_samples,
            "bootstrap_seeds": {
                "aggregate": config.aggregate_bootstrap_seed,
                "final": config.final_bootstrap_seed,
                "pair": config.pair_bootstrap_seed,
                "json": config.json_bootstrap_seed,
            },
            "mcnemar_alpha": config.mcnemar_alpha,
            "acceptance_gates": {
                "aggregate_ci_lower_gt_zero": (
                    config.require_ci_lower_positive
                ),
                "final_ci_lower_gt_zero": (
                    config.require_ci_lower_positive
                ),
                "pair_ci_lower_gt_zero": (
                    config.require_ci_lower_positive
                ),
                "aggregate_exact_mcnemar_p_lt": config.mcnemar_alpha,
                "final_exact_mcnemar_p_lt": config.mcnemar_alpha,
                "pair_exact_mcnemar_p_lt": config.mcnemar_alpha,
                "minimum_final_only_wins": (
                    config.minimum_final_only_wins
                ),
                "maximum_final_only_losses": (
                    config.maximum_final_only_losses
                ),
                "every_json_family_non_regression": True,
            },
        },
        "decision_rule": {
            "method_accepted": (
                "stable_and_reloadable AND aggregate/final/pair bootstrap "
                "CI lower bounds > 0 AND aggregate/final/pair exact "
                "McNemar p < 0.05 AND final-only wins >= 6 AND final-only "
                "losses = 0 AND every JSON family is non-regressing"
            ),
            "larger_training_allowed": False,
            "benchmark_allowed": False,
            "independent_holdout_allowed": False,
            "rl_allowed": False,
            "forbidden_after_observation": [
                "objective_search",
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
                "threshold_search",
            ],
        },
        "execution_boundary": {
            "training_started": False,
            "model_generation_started": False,
            "this_commit_only_preregisters": True,
        },
        "claim_boundary": (
            "This receipt freezes one unchanged consistency-v1 replication "
            "on 192 fresh development pairs and 128 fresh JSON rows before "
            "any model training or generation. It is local synthetic "
            "replication evidence, not benchmark or independent-holdout "
            "evidence."
        ),
    }


def main() -> None:
    receipt = build_receipt()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
