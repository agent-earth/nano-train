#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

from nano_train.config import load_sft_smoke_config
from nano_train.data import load_execution_target_dataset, tokenize_samples
from nano_train.sft import _sample_scheduled_batch_order, sha256_file


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/sft/execution_target_paired_smoke_v1.json"
OUTPUT = (
    ROOT
    / "docs/experiments/execution_target_paired_sft_smoke_v1.preregister.json"
)
JSON_FAMILIES = {
    "coding-and-validation",
    "planning-and-state",
    "skill-routing-and-reflection",
    "tool-use-and-recovery",
}


def main() -> None:
    config = load_sft_smoke_config(CONFIG)
    dataset = load_execution_target_dataset(
        config.dataset_path,
        config.release_manifest_path or "",
    )
    raw = json.loads(Path(config.dataset_path).read_text(encoding="utf-8"))
    raw_by_id = {row["sample_id"]: row for row in raw["samples"]}
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_path,
        local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenized = tokenize_samples(
        dataset,
        tokenizer,
        max_length=config.max_length,
    )
    train = [sample for sample in tokenized if sample.split == "train"]
    validation = [
        sample for sample in tokenized if sample.split == "validation"
    ]
    order = _sample_scheduled_batch_order(
        train,
        config.train_sample_schedule,
    )
    actual_ids = [train[index].sample_id for index in order]
    if actual_ids != list(config.train_sample_schedule):
        raise ValueError("resolved sample schedule differs from config")

    selected_raw = [raw_by_id[sample_id] for sample_id in actual_ids]
    selected_views = Counter(row["view"] for row in selected_raw)
    selected_json = Counter(
        row["task_family"]
        for row in selected_raw
        if row["view"] == "json_preservation"
    )
    selected_pairs: dict[str, set[str]] = {}
    for row in selected_raw:
        if row["pair_id"]:
            selected_pairs.setdefault(row["pair_id"], set()).add(row["view"])
    if (
        selected_views
        != Counter({"process": 10, "final": 10, "json_preservation": 20})
        or set(selected_json) != JSON_FAMILIES
        or any(count != 5 for count in selected_json.values())
        or len(selected_pairs) != 10
        or any(views != {"process", "final"} for views in selected_pairs.values())
    ):
        raise ValueError("sample schedule violates paired exposure contract")

    validation_ids = sorted(sample.sample_id for sample in validation)
    target_lengths = {
        row["sample_id"]: len(
            tokenizer(
                row["messages"][-1]["content"],
                add_special_tokens=False,
            ).input_ids
        )
        for row in raw["samples"]
    }
    result = {
        "schema_version": "nano_train_execution_target_preregister_v1",
        "experiment_id": config.experiment_id,
        "identity": {
            "data_commit": "c2e07e9",
            "config_sha256": sha256_file(CONFIG),
            "dataset_file_sha256": dataset["release"]["dataset_file_sha256"],
            "dataset_canonical_sha256": dataset["release"][
                "dataset_canonical_sha256"
            ],
            "release_manifest_sha256": dataset["release"]["sha256"],
            "model_config_sha256": sha256_file(
                Path(config.model_path) / "config.json"
            ),
        },
        "data": {
            "train_rows": len(train),
            "validation_rows": len(validation),
            "scheduled_train_rows": len(actual_ids),
            "scheduled_train_sample_ids": actual_ids,
            "scheduled_train_sample_id_sha256": hashlib.sha256(
                "\n".join(actual_ids).encode("utf-8")
            ).hexdigest(),
            "scheduled_views": dict(sorted(selected_views.items())),
            "scheduled_json_by_family": dict(sorted(selected_json.items())),
            "scheduled_complete_pairs": len(selected_pairs),
            "validation_sample_ids": validation_ids,
            "validation_sample_id_sha256": hashlib.sha256(
                "\n".join(validation_ids).encode("utf-8")
            ).hexdigest(),
            "full_sequence_max": max(len(sample.input_ids) for sample in tokenized),
            "target_max": max(target_lengths.values()),
        },
        "training": {
            "model": "Qwen3.5-4B",
            "steps": config.max_steps,
            "learning_rate": config.learning_rate,
            "seed": config.seed,
            "dtype": config.dtype,
            "max_length": config.max_length,
            "generation_max_new_tokens": config.generation_max_new_tokens,
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
                "train_sample_schedule_search",
                "step_search",
                "learning_rate_search",
                "seed_search",
                "max_length_search",
                "generation_budget_search",
                "lora_target_search",
                "prompt_search",
                "parser_search",
                "adapter_weight_search",
                "route_search",
            ],
        },
        "claim_boundary": (
            "This receipt freezes one local SFT smoke before any model "
            "generation on the 80-row development split. Passing or failing "
            "is local synthetic evidence only."
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
