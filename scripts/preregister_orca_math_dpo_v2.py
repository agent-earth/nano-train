#!/usr/bin/env python3

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

from nano_train.orca_math_dpo_suffix import (
    build_selection,
    load_config,
    tokenize_suffix_pair,
)
from nano_train.sft import sha256_file


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/preference_orca_math_dpo_v2.json"
OUTPUT = (
    ROOT / "docs/experiments/orca_math_verifier_dpo_suffix_v2.preregister.json"
)


def build_receipt() -> dict:
    config = load_config(CONFIG)
    selection = build_selection(config)
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_path, local_files_only=True
    )
    suffix = []
    for row in selection["train"]:
        chosen, rejected, common = tokenize_suffix_pair(
            tokenizer, row, max_length=config.max_length
        )
        suffix.append(
            {
                "sample_id": row["sample_id"],
                "common_target_tokens": common,
                "chosen_supervised_tokens": sum(
                    label != -100 for label in chosen.labels
                ),
                "rejected_supervised_tokens": sum(
                    label != -100 for label in rejected.labels
                ),
            }
        )
    return {
        "schema_version": "nano_train_orca_math_dpo_suffix_preregister_v2",
        "experiment_id": config.experiment_id,
        "identity": {
            "config_sha256": sha256_file(CONFIG),
            "model_config_sha256": config.model_config_sha256,
            "dataset_file_sha256": config.dataset_file_sha256,
            "release_manifest_sha256": config.release_manifest_sha256,
            "prior_dpo_preregister_sha256": (
                config.prior_dpo_preregister_sha256
            ),
            "prior_dpo_result_sha256": config.prior_dpo_result_sha256,
        },
        "selection": {
            "train_pairs": len(selection["train"]),
            "dev_rows": len(selection["dev"]),
            "train_ids": selection["train_ids"],
            "dev_ids": selection["dev_ids"],
            "train_ids_sha256": selection["train_ids_sha256"],
            "dev_ids_sha256": selection["dev_ids_sha256"],
            "prior_ids_sha256": selection["prior_ids_sha256"],
            "train_by_stratum": dict(
                sorted(
                    Counter(
                        row["stratum"] for row in selection["train"]
                    ).items()
                )
            ),
            "dev_by_stratum": dict(
                sorted(
                    Counter(row["stratum"] for row in selection["dev"]).items()
                )
            ),
            "suffix_receipts": suffix,
        },
        "objective": {
            "kind": "base_reference_dpo_differing_suffix_only",
            "beta": config.beta,
            "optimizer_steps": config.max_steps,
            "learning_rate": config.learning_rate,
            "seed": config.seed,
            "only_method_change_from_v1": (
                "mask all shared target tokens before the first differing "
                "chosen/rejected token"
            ),
        },
        "evaluation": {
            "fresh_dev_rows": 192,
            "generation_max_new_tokens": config.generation_max_new_tokens,
            "generation_batch_size": config.generation_batch_size,
            "bootstrap_samples": config.bootstrap_samples,
            "bootstrap_seed": config.bootstrap_seed,
            "exact_mcnemar_alpha": config.alpha,
            "minimum_candidate_only_wins": config.minimum_candidate_only_wins,
        },
        "decision_boundary": {
            "benchmark_allowed": False,
            "larger_training_allowed": False,
            "rerun_or_tuning_allowed": False,
        },
        "execution_boundary": {
            "training_started": False,
            "generation_started": False,
            "this_commit_only_preregisters": True,
        },
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
