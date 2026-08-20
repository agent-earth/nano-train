#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

from nano_train.data import load_analog_dataset, tokenize_samples
from nano_train.router_negative_diversity import (
    AUDIT_SHA256,
    CONTRACT_SHA256,
    DATASET_CANONICAL_SHA256,
    DATASET_SHA256,
    RELEASE_SHA256,
    load_config,
    verify_data_release,
)
from nano_train.sft import _batch_order, sha256_file


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/sft/router_negative_diversity_v2.json"
DATASET = (
    ROOT.parent
    / "nano-data-pipeline-fullstack-traex-03/datasets/"
    "qwen35_router_negative_diversity_v2.json"
)
RELEASE = (
    ROOT.parent
    / "nano-data-pipeline-fullstack-traex-03/manifests/"
    "qwen35_router_negative_diversity_v2.release.json"
)
PREREG = (
    ROOT
    / "docs/experiments/qwen35_router_negative_diversity_sft_v2.preregister.json"
)
MARKDOWN = (
    ROOT / "docs/experiments/qwen35_router_negative_diversity_sft_v2.md"
)
DATA_COMMIT = "9374a9f67ab17623f4caf842f9d2546851aa60b2"


def git_revision() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()


def build_receipt() -> dict:
    config = load_config(CONFIG)
    if (
        Path(config.dataset_path).resolve() != DATASET.resolve()
        or sha256_file(DATASET) != DATASET_SHA256
        or sha256_file(RELEASE) != RELEASE_SHA256
    ):
        raise ValueError("router negative diversity SFT data differs")
    release_identity = verify_data_release(config)
    release = json.loads(RELEASE.read_text(encoding="utf-8"))
    dataset = load_analog_dataset(DATASET)
    raw_by_id = {row["sample_id"]: row for row in dataset["samples"]}
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
    train = [row for row in tokenized if row.split == "train"]
    validation = [row for row in tokenized if row.split == "validation"]
    order = _batch_order(train, config.seed)
    exposure_count = (
        config.max_steps
        * config.batch_size
        * config.gradient_accumulation_steps
    )
    exposure = [train[order[index % len(order)]] for index in range(exposure_count)]
    exposure_labels = Counter(row.task_family for row in exposure)
    exposure_subtypes = Counter(
        raw_by_id[row.sample_id]["negative_subtype"]
        for row in exposure
        if row.task_family == "router_c"
    )
    validation_labels = Counter(row.task_family for row in validation)
    validation_subtypes = Counter(
        raw_by_id[row.sample_id]["negative_subtype"]
        for row in validation
        if row.task_family == "router_c"
    )
    expected_subtypes = {
        "box_total",
        "remaining_stock",
        "paired_average",
        "single_operation",
        "weighted_total",
        "quotient_remainder",
        "time_conversion",
        "percentage_change",
    }
    if (
        len(train) != 6144
        or len(validation) != 1536
        or exposure_labels
        != Counter({"router_c": 60, "router_a": 52, "router_b": 48})
        or set(exposure_subtypes) != expected_subtypes
        or min(exposure_subtypes.values()) < 3
        or validation_labels
        != Counter({"router_a": 512, "router_b": 512, "router_c": 512})
        or validation_subtypes != Counter(dict.fromkeys(expected_subtypes, 64))
    ):
        raise ValueError("router negative diversity SFT exposure differs")
    exposure_ids = [row.sample_id for row in exposure]
    validation_ids = sorted(row.sample_id for row in validation)
    return {
        "schema_version": (
            "nano_train_router_negative_diversity_preregister_v2"
        ),
        "experiment_id": config.experiment_id,
        "identity": {
            "code_revision": git_revision(),
            "config_sha256": sha256_file(CONFIG),
            "data_commit": DATA_COMMIT,
            "dataset_file_sha256": DATASET_SHA256,
            "dataset_canonical_sha256": DATASET_CANONICAL_SHA256,
            "release_sha256": release_identity["sha256"],
            "audit_sha256": AUDIT_SHA256,
            "contract_sha256": CONTRACT_SHA256,
            "model_config_sha256": sha256_file(
                Path(config.model_path) / "config.json"
            ),
            "tokenizer_file_sha256": release["source"][
                "tokenizer_file_sha256"
            ],
        },
        "data": {
            "train_rows": len(train),
            "validation_rows": len(validation),
            "train_tokens": release["accepted"]["train_tokens"],
            "validation_by_label": dict(sorted(validation_labels.items())),
            "validation_c_by_subtype": dict(
                sorted(validation_subtypes.items())
            ),
            "validation_sample_ids_sha256": hashlib.sha256(
                "\n".join(validation_ids).encode()
            ).hexdigest(),
            "scheduled_exposures": len(exposure),
            "scheduled_exposure_by_label": dict(
                sorted(exposure_labels.items())
            ),
            "scheduled_c_exposure_by_subtype": dict(
                sorted(exposure_subtypes.items())
            ),
            "scheduled_exposure_ids_sha256": hashlib.sha256(
                "\n".join(exposure_ids).encode()
            ).hexdigest(),
            "full_sequence_max": max(len(row.input_ids) for row in tokenized),
            "target_max": max(
                len(
                    tokenizer(
                        row.target,
                        add_special_tokens=False,
                    ).input_ids
                )
                for row in tokenized
            ),
        },
        "training": {
            "recipe_source": "router-classification-sft-smoke-v1",
            "model": "Qwen3.5-4B",
            "steps": config.max_steps,
            "effective_batch_size": (
                config.batch_size * config.gradient_accumulation_steps
            ),
            "learning_rate": config.learning_rate,
            "seed": config.seed,
            "dtype": config.dtype,
            "max_length": config.max_length,
            "generation_max_new_tokens": config.generation_max_new_tokens,
            "lora_targets": list(config.lora_targets),
            "lora_r": config.lora_r,
            "lora_alpha": config.lora_alpha,
        },
        "acceptance": {
            "finite_loss_gradients_and_adapter": True,
            "actual_exposure_ids_exact": True,
            "independent_reload_metrics_and_generations_exact": True,
            "aggregate_post_exact_gt_baseline": True,
            "router_a_post_exact_at_least_480_of_512": True,
            "router_b_post_exact_at_least_480_of_512": True,
            "router_c_post_exact_at_least_496_of_512": True,
            "every_c_subtype_post_exact_at_least_60_of_64": True,
            "every_label_non_regression": True,
            "every_c_subtype_non_regression": True,
            "serving_namespace_remap_required": True,
            "fresh_router_integration_preregistration_allowed_after_pass": True,
            "benchmark_allowed_after_pass": False,
            "canary_allowed_after_pass": False,
            "holdout_allowed_after_pass": False,
            "rl_allowed_after_pass": False,
        },
        "decision_policy": {
            "forbidden_after_observation": [
                "dataset_change",
                "sample_schedule_change",
                "step_change",
                "learning_rate_change",
                "seed_change",
                "lora_target_change",
                "lora_rank_change",
                "generation_budget_change",
                "prompt_change",
                "label_or_subtype_parser_change",
                "adapter_checkpoint_selection",
                "adapter_weight_change",
                "second_training_run",
                "reuse_integration_v1_or_v2_rows_or_outputs",
            ],
            "passed": (
                "Publish local SFT evidence, build a content-identical vLLM "
                "namespace remap, prove serving parity, and separately "
                "pre-register a fresh integration."
            ),
            "failed": (
                "Publish negative evidence and stop this recipe. Do not tune "
                "or rerun on the observed validation split."
            ),
        },
        "execution_boundary": {
            "training_started": False,
            "model_generation_started": False,
            "adapter_exists": False,
            "metrics_exist": False,
            "integration_rows_or_outputs_loaded": False,
            "benchmark_accessed": False,
            "canary_accessed": False,
            "holdout_accessed": False,
            "this_commit_only_preregisters": True,
        },
        "claim_boundary": (
            "This pre-registers one bounded 40-step router SFT run on "
            "deterministic synthetic data. Passing is not benchmark or final "
            "routing evidence and does not authorize a second run."
        ),
    }


def render_markdown(receipt: dict) -> str:
    return f"""# Qwen3.5 Router Negative-Diversity SFT v2

## Data

- Train：{receipt['data']['train_rows']:,}，tokens：
  {receipt['data']['train_tokens']:,}；
- Dev：{receipt['data']['validation_rows']:,}，A/B/C 各512；
- C dev：8 subtypes 各64；
- 40 steps × effective batch 4 = 160 exposures；
- exposure A/B/C：
  `{json.dumps(receipt['data']['scheduled_exposure_by_label'], sort_keys=True)}`；
- C exposure：
  `{json.dumps(receipt['data']['scheduled_c_exposure_by_subtype'], sort_keys=True)}`。

## Frozen Recipe

- FP32 expanded LoRA：q/v/gate/up/down，r=8，alpha=16；
- LR 2e-4，40 steps，seed 20260827；
- max length 256；generation 8 tokens；
- 保持 framework smoke 上限，不扩大到长训练。

## Acceptance

- finite loss/gradients/adapter；
- exact exposure IDs；
- independent reload metrics + 1,536 generations exact；
- A/B ≥480/512，C ≥496/512；
- 每个 C subtype ≥60/64；
- label 和 subtype 全部 non-regression；
- vLLM namespace remap + serving parity 必须另行通过。

通过也只允许另行预注册 fresh integration；benchmark/canary/holdout/RL
继续关闭。

## Boundary

- config SHA：`{receipt['identity']['config_sha256']}`；
- dataset SHA：`{receipt['identity']['dataset_file_sha256']}`；
- release SHA：`{receipt['identity']['release_sha256']}`；
- exposure SHA：`{receipt['data']['scheduled_exposure_ids_sha256']}`；
- training/model generation started：false；
- adapter/metrics exists：false。
"""


def main() -> None:
    receipt = build_receipt()
    PREREG.parent.mkdir(parents=True, exist_ok=True)
    PREREG.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    MARKDOWN.write_text(render_markdown(receipt), encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
