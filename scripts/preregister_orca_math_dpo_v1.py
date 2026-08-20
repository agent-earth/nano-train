#!/usr/bin/env python3

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

from nano_train.orca_math_dpo import build_selection, load_config
from nano_train.sft import sha256_file


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/preference_orca_math_dpo_v1.json"
OUTPUT = (
    ROOT / "docs/experiments/orca_math_verifier_dpo_v1.preregister.json"
)


def build_receipt() -> dict:
    config = load_config(CONFIG)
    selection = build_selection(config)
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_path, local_files_only=True
    )
    lengths = []
    for row in selection["train"]:
        for target in (row["chosen"], row["rejected"]):
            prompt = tokenizer.apply_chat_template(
                row["prompt_messages"],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            lengths.append(
                len(
                    tokenizer(
                        prompt + target + tokenizer.eos_token,
                        add_special_tokens=False,
                    ).input_ids
                )
            )
    if max(lengths) > config.max_length:
        raise ValueError("Orca Math DPO selected sequence exceeds max_length")
    return {
        "schema_version": "nano_train_orca_math_dpo_preregister_v1",
        "experiment_id": config.experiment_id,
        "identity": {
            "config_sha256": sha256_file(CONFIG),
            "model_config_sha256": config.model_config_sha256,
            "dataset_file_sha256": config.dataset_file_sha256,
            "release_manifest_sha256": config.release_manifest_sha256,
            "prior_sft_result_sha256": config.prior_sft_result_sha256,
        },
        "selection": {
            "train_pairs": len(selection["train"]),
            "dev_rows": len(selection["dev"]),
            "train_ids": selection["train_ids"],
            "dev_ids": selection["dev_ids"],
            "train_ids_sha256": selection["train_ids_sha256"],
            "dev_ids_sha256": selection["dev_ids_sha256"],
            "train_by_stratum": dict(
                sorted(Counter(row["stratum"] for row in selection["train"]).items())
            ),
            "dev_by_stratum": dict(
                sorted(Counter(row["stratum"] for row in selection["dev"]).items())
            ),
            "max_train_sequence": max(lengths),
        },
        "objective": {
            "kind": "base_reference_dpo",
            "loss": (
                "-logsigmoid(beta * ((policy_chosen-policy_rejected) - "
                "(base_chosen-base_rejected)))"
            ),
            "sequence_score": "mean target-token log probability",
            "beta": config.beta,
            "reference": "same frozen Qwen3.5-4B with LoRA disabled",
            "optimizer_steps": config.max_steps,
            "learning_rate": config.learning_rate,
            "seed": config.seed,
            "lora_targets": list(config.lora_targets),
        },
        "evaluation": {
            "fresh_dev_rows": 192,
            "generation_max_new_tokens": config.generation_max_new_tokens,
            "generation_batch_size": config.generation_batch_size,
            "scorer": "strict FINAL numeric equivalence",
            "bootstrap_samples": config.bootstrap_samples,
            "bootstrap_seed": config.bootstrap_seed,
            "exact_mcnemar_alpha": config.alpha,
            "minimum_candidate_only_wins": config.minimum_candidate_only_wins,
            "admission": (
                "positive point delta, positive bootstrap lower bound, "
                "McNemar p<0.05, at least six wins, wins>losses, and every "
                "stratum non-regressing"
            ),
        },
        "decision_boundary": {
            "benchmark_allowed": False,
            "larger_training_allowed": False,
            "rerun_or_tuning_allowed": False,
            "forbidden_after_observation": [
                "pair_selection_change",
                "dev_selection_change",
                "beta_change",
                "step_change",
                "learning_rate_change",
                "seed_change",
                "lora_scope_change",
                "parser_change",
                "scorer_change",
                "statistical_threshold_change",
            ],
        },
        "execution_boundary": {
            "training_started": False,
            "generation_started": False,
            "this_commit_only_preregisters": True,
        },
        "claim_boundary": (
            "This receipt freezes one fresh local verifier-guided DPO smoke. "
            "It is not benchmark, holdout, 9B, 27B, or model-quality evidence."
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
