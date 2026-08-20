#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from transformers import AutoTokenizer

from nano_train.scaled_quality import (
    build_dataset,
    load_config,
    public_dataset_contract,
)
from nano_train.sft import sha256_file
from nano_train.synthetic_quality import (
    build_cases as build_forbidden_cases,
    case_contract as forbidden_case_contract,
    load_config as load_forbidden_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/scaled_quality/qwen35_scaled_quality_sft_v1.json"
JSON_OUTPUT = (
    ROOT / "docs/experiments/qwen35_scaled_quality_sft_v1.preregister.json"
)
MARKDOWN_OUTPUT = (
    ROOT / "docs/experiments/qwen35_scaled_quality_sft_v1.md"
)


def git_revision() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()


def build_receipt() -> dict:
    config = load_config(CONFIG)
    dataset = build_dataset(config)
    contract = public_dataset_contract(dataset)
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_path,
        local_files_only=True,
    )
    lengths = []
    train_tokens = 0
    for row in (*dataset["train"], *dataset["dev"]):
        text = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": config.system_prompt},
                {"role": "user", "content": row["prompt"]},
                {"role": "assistant", "content": row["target"]},
            ],
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        length = len(tokenizer(text, add_special_tokens=False).input_ids)
        lengths.append(length)
        if row["split"] == "train":
            train_tokens += length
    if max(lengths) > config.max_length:
        raise ValueError("scaled quality sequence exceeds max_length")
    forbidden_path = Path(config.forbidden_evaluation_config_path)
    if sha256_file(forbidden_path) != config.forbidden_evaluation_config_sha256:
        raise ValueError("forbidden evaluation config identity differs")
    forbidden = build_forbidden_cases(load_forbidden_config(forbidden_path))
    if (
        forbidden_case_contract(forbidden)["case_contract_sha256"]
        != config.forbidden_evaluation_contract_sha256
    ):
        raise ValueError("forbidden evaluation contract identity differs")
    forbidden_hashes = {
        hashlib.sha256(row["prompt"].encode()).hexdigest()
        for row in forbidden
    }
    train_hashes = {
        row["prompt_sha256"] for row in contract["train"]
    }
    dev_hashes = {
        row["prompt_sha256"] for row in contract["dev"]
    }
    if train_hashes & dev_hashes or (train_hashes | dev_hashes) & forbidden_hashes:
        raise ValueError("scaled quality prompt overlap detected")
    return {
        "schema_version": "nano_train_scaled_quality_sft_preregister_v1",
        "experiment_id": config.experiment_id,
        "identity": {
            "code_revision": git_revision(),
            "config_sha256": sha256_file(CONFIG),
            "model_config_sha256": config.model_config_sha256,
            "model_index_sha256": config.model_index_sha256,
            "forbidden_evaluation_config_sha256": (
                config.forbidden_evaluation_config_sha256
            ),
            "forbidden_evaluation_contract_sha256": (
                config.forbidden_evaluation_contract_sha256
            ),
        },
        "dataset_contract": contract,
        "counts": {
            "train_rows": len(dataset["train"]),
            "dev_rows": len(dataset["dev"]),
            "train_rows_per_family": config.train_cases_per_family,
            "dev_rows_per_family": config.dev_cases_per_family,
            "tokenizer_counted_train_tokens": train_tokens,
            "maximum_sequence_tokens": max(lengths),
            "train_dev_prompt_overlap": 0,
            "observed_quality_prompt_overlap": 0,
        },
        "training": {
            "model": "Qwen3.5-4B",
            "dtype": config.dtype,
            "seed": config.seed,
            "batch_size": config.batch_size,
            "optimizer_steps": config.max_steps,
            "epochs": 1,
            "every_train_row_exposed_exactly_once": True,
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "lora_r": config.lora_r,
            "lora_alpha": config.lora_alpha,
            "lora_targets": list(config.lora_targets),
            "gradient_checkpointing": config.gradient_checkpointing,
        },
        "evaluation": {
            "dev_rows": len(dataset["dev"]),
            "generation_batch_size": config.generation_batch_size,
            "generation_max_new_tokens": config.generation_max_new_tokens,
            "temperature": 0.0,
            "strict_exact_target": "FINAL: <integer>",
            "bootstrap_samples": config.bootstrap_samples,
            "bootstrap_seed": config.bootstrap_seed,
        },
        "acceptance": {
            "finite_and_reloadable": True,
            "post_accuracy_gt_baseline": True,
            "paired_bootstrap_ci_lower_gt_zero": True,
            "exact_mcnemar_p_lt": 0.05,
            "minimum_candidate_only_wins": 12,
            "maximum_baseline_only_losses": 0,
            "every_family_non_regression": True,
            "parse_failures_non_regression": True,
            "benchmark_allowed_after_pass": False,
            "canary_allowed_after_pass": False,
        },
        "decision_policy": {
            "forbidden_after_observation": [
                "dataset_change",
                "train_dev_split_change",
                "step_change",
                "batch_change",
                "learning_rate_change",
                "seed_change",
                "lora_scope_change",
                "prompt_change",
                "target_change",
                "generation_budget_change",
                "parser_change",
                "adapter_weight_change",
                "benchmark_access",
                "canary_access",
                "holdout_access",
            ],
            "passed": (
                "Publish synthetic quality evidence and separately pre-register "
                "any canary treatment."
            ),
            "failed": (
                "Preserve negative evidence and change mechanism or data on a "
                "new train/dev surface; do not tune on observed dev."
            ),
        },
        "execution_boundary": {
            "training_started": False,
            "model_generation_started": False,
            "dev_observed": False,
            "benchmark_accessed": False,
            "this_commit_only_preregisters": True,
        },
        "claim_boundary": (
            "This is a benchmark-free scaled SFT quality experiment on new "
            "programmatic train/dev data. Passing would establish only fresh "
            "synthetic transfer, not benchmark or final 4B superiority."
        ),
    }


def render_markdown(receipt: dict) -> str:
    counts = receipt["counts"]
    training = receipt["training"]
    return f"""# Qwen3.5 Scaled Quality SFT v1

## 假设

两步 RL/OPD 只证明机制可运行，没有改变 96-case correctness。下一实验只改变
训练规模和监督覆盖：用新生成的 512 条 SFT train rows，训练 128 steps，
检查 96 条未触碰 dev 是否出现显著提升。

## 数据

- train：`{counts['train_rows']}` rows，每个 family
  `{counts['train_rows_per_family']}`；
- dev：`{counts['dev_rows']}` rows，每个 family
  `{counts['dev_rows_per_family']}`；
- Qwen3.5 tokenizer train tokens：
  `{counts['tokenizer_counted_train_tokens']:,}`；
- maximum sequence：`{counts['maximum_sequence_tokens']}`；
- train/dev prompt overlap：0；
- 与已观察 96-case quality suite prompt overlap：0；
- benchmark/canary/holdout rows 和 outputs：0。

public receipt 只保存 case IDs、family、prompt SHA 和 target SHA；raw expression、
target、generation、adapter 都在 ignored `artifacts/`。

## 训练

- Qwen3.5-4B FP32；
- q/v LoRA r={training['lora_r']} alpha={training['lora_alpha']}；
- batch={training['batch_size']}；
- steps={training['optimizer_steps']}；
- exactly one epoch，每个 train row 恰好曝光一次；
- LR={training['learning_rate']}；
- seed={training['seed']}。

## 验收

- finite + independent reload；
- post accuracy > baseline；
- paired bootstrap 95% CI 下界 > 0；
- exact McNemar p < 0.05；
- 至少 12 wins、0 losses；
- 每个 family 不回退；
- parse failures 不回退。

即使通过，也只建立 fresh synthetic quality evidence；canary、benchmark 和
independent holdout 仍需单独预注册。

## 执行边界

```json
{json.dumps(receipt['execution_boundary'], indent=2, sort_keys=True)}
```
"""


def main() -> None:
    receipt = build_receipt()
    JSON_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    JSON_OUTPUT.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    MARKDOWN_OUTPUT.write_text(render_markdown(receipt), encoding="utf-8")
    print(
        json.dumps(
            {
                "experiment_id": receipt["experiment_id"],
                "config_sha256": receipt["identity"]["config_sha256"],
                "counts": receipt["counts"],
                "execution_boundary": receipt["execution_boundary"],
                "json_output": str(JSON_OUTPUT),
                "markdown_output": str(MARKDOWN_OUTPUT),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
