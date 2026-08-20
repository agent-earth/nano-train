#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

from nano_train.data import load_analog_dataset, tokenize_samples
from nano_train.router_classification import (
    load_config,
    verify_data_release,
)
from nano_train.sft import _batch_order, sha256_file


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/sft/router_classification_smoke_v1.json"
DATASET = (
    ROOT.parent
    / "nano-data-pipeline-fullstack-traex-03/datasets/"
    "qwen35_router_classification_v1.json"
)
RELEASE = (
    ROOT.parent
    / "nano-data-pipeline-fullstack-traex-03/manifests/"
    "qwen35_router_classification_v1.release.json"
)
PREREG = (
    ROOT
    / "docs/experiments/qwen35_router_classification_sft_v1.preregister.json"
)
MARKDOWN = (
    ROOT / "docs/experiments/qwen35_router_classification_sft_v1.md"
)
DATA_COMMIT = "c00231e616fd2bd6c69a46226745c0ab737bf823"
DATASET_SHA256 = (
    "dacd3663639fe9ddc054865b87afdd0c918f0fddb12c8c9355819d4bbce95d65"
)
RELEASE_SHA256 = (
    "fb265e125e181056856a196322cf5da3b1d7d890d60ad653839d2707ebe3781d"
)


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
        raise ValueError("router SFT data identity differs")
    release_identity = verify_data_release(config)
    release = json.loads(RELEASE.read_text(encoding="utf-8"))
    dataset = load_analog_dataset(DATASET)
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
    validation_labels = Counter(row.task_family for row in validation)
    if (
        len(train) != 768
        or len(validation) != 192
        or set(exposure_labels) != {"router_a", "router_b", "router_c"}
        or min(exposure_labels.values()) < 40
        or validation_labels
        != Counter({"router_a": 64, "router_b": 64, "router_c": 64})
    ):
        raise ValueError("router SFT exposure contract differs")
    validation_ids = sorted(row.sample_id for row in validation)
    exposure_ids = [row.sample_id for row in exposure]
    return {
        "schema_version": "nano_train_router_classification_preregister_v1",
        "experiment_id": config.experiment_id,
        "identity": {
            "code_revision": git_revision(),
            "config_sha256": sha256_file(CONFIG),
            "data_commit": DATA_COMMIT,
            "dataset_file_sha256": DATASET_SHA256,
            "dataset_canonical_sha256": release["source"][
                "dataset_canonical_sha256"
            ],
            "release_sha256": release_identity["sha256"],
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
            "validation_by_label": dict(sorted(validation_labels.items())),
            "validation_sample_ids_sha256": hashlib.sha256(
                "\n".join(validation_ids).encode()
            ).hexdigest(),
            "scheduled_exposures": len(exposure),
            "scheduled_exposure_by_label": dict(
                sorted(exposure_labels.items())
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
            "recipe_source": "format-contract-sft-smoke-v3",
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
            "independent_reload_exact_metrics": True,
            "aggregate_post_exact_gt_baseline": True,
            "router_a_post_exact_at_least_48_of_64": True,
            "router_b_post_exact_at_least_48_of_64": True,
            "router_c_post_exact_at_least_60_of_64": True,
            "every_label_non_regression": True,
            "next_fresh_router_integration_preregistration_allowed_after_pass": True,
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
                "label_parser_change",
                "adapter_checkpoint_selection",
                "adapter_weight_change",
                "second_training_run",
            ],
            "passed": (
                "Publish local SFT smoke evidence and separately pre-register "
                "fresh router integration with the exact adapter identity."
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
            "benchmark_accessed": False,
            "canary_accessed": False,
            "holdout_accessed": False,
            "this_commit_only_preregisters": True,
        },
        "claim_boundary": (
            "This pre-registers one bounded router SFT smoke. Passing is local "
            "synthetic classification evidence only and does not establish "
            "benchmark or final routing quality."
        ),
    }


def render_markdown(receipt: dict) -> str:
    return f"""# Qwen3.5 Router Classification SFT Smoke v1

## Data

- Train：768；
- Dev：192，A/B/C 各64；
- 40 steps × effective batch 4 = 160 exposures；
- exposure by label：
  `{json.dumps(receipt['data']['scheduled_exposure_by_label'], sort_keys=True)}`；
- max sequence：{receipt['data']['full_sequence_max']}；
- target max：{receipt['data']['target_max']}。

## Frozen Recipe

- parent recipe：format-contract-sft-smoke-v3；
- FP32；
- expanded LoRA：q/v/gate/up/down，r=8，alpha=16；
- LR 2e-4，40 steps，seed 20260824；
- max length 256；generation 8 tokens。

## Acceptance

- finite loss/gradients/adapter；
- independent reload exact metrics；
- aggregate exact improves；
- A/B each ≥48/64；
- C ≥60/64；
- every label non-regression。

通过也只允许另行预注册 fresh router integration；benchmark/canary/holdout/RL
继续关闭。

## Boundary

- config SHA：`{receipt['identity']['config_sha256']}`；
- dataset SHA：`{receipt['identity']['dataset_file_sha256']}`；
- release SHA：`{receipt['identity']['release_sha256']}`；
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
