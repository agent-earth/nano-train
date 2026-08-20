#!/usr/bin/env python3

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from transformers import AutoTokenizer

from nano_train.quality_consistency import (
    benchmark_prompt_hashes,
    build_dataset,
    dataset_prompt_hashes,
    forbidden_prompt_hashes,
    load_config,
    normalized_dataset_prompt_hashes,
    public_contract,
)
from nano_train.sft import sha256_file, sha256_tree


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT / "configs/quality_consistency/qwen35_quality_consistency_v1.json"
)
JSON_OUTPUT = (
    ROOT / "docs/experiments/qwen35_quality_consistency_v1.preregister.json"
)
MARKDOWN_OUTPUT = (
    ROOT / "docs/experiments/qwen35_quality_consistency_v1.md"
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
    contract = public_contract(dataset)
    if sha256_tree(Path(config.anchor_adapter_path)) != config.anchor_adapter_sha256:
        raise ValueError("quality consistency anchor identity differs")
    observed_overlap = (
        dataset_prompt_hashes(dataset) & forbidden_prompt_hashes(config)
    )
    benchmark, benchmark_counts = benchmark_prompt_hashes(config)
    benchmark_overlap = (
        normalized_dataset_prompt_hashes(dataset) & benchmark
    )
    if observed_overlap or benchmark_overlap:
        raise ValueError("quality consistency contamination detected")
    tokenizer = AutoTokenizer.from_pretrained(
        config.anchor_adapter_path,
        local_files_only=True,
    )
    maximum = 0
    aligned = 0
    train_tokens = 0
    for pair in (*dataset["train_pairs"], *dataset["dev_pairs"]):
        target_ids = {}
        for view in ("process", "final"):
            prompt = tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": config.system_prompt},
                    {"role": "user", "content": pair[f"{view}_prompt"]},
                ],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            full = prompt + pair[f"{view}_target"] + tokenizer.eos_token
            prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
            full_ids = tokenizer(full, add_special_tokens=False).input_ids
            target_ids[view] = full_ids[len(prompt_ids) :]
            maximum = max(maximum, len(full_ids))
            if pair in dataset["train_pairs"]:
                train_tokens += len(full_ids)
        if (
            len(target_ids["process"]) >= len(target_ids["final"])
            and target_ids["process"][-len(target_ids["final"]) :]
            == target_ids["final"]
        ):
            aligned += 1
    total_pairs = len(dataset["train_pairs"]) + len(dataset["dev_pairs"])
    if aligned != total_pairs or maximum > config.max_length:
        raise ValueError("quality consistency tokenization contract differs")
    return {
        "schema_version": "nano_train_quality_consistency_preregister_v1",
        "experiment_id": config.experiment_id,
        "identity": {
            "code_revision": git_revision(),
            "config_sha256": sha256_file(CONFIG),
            "model_config_sha256": config.model_config_sha256,
            "model_index_sha256": config.model_index_sha256,
            "anchor_adapter_sha256": config.anchor_adapter_sha256,
        },
        "dataset_contract": contract,
        "counts": {
            "train_pairs": len(dataset["train_pairs"]),
            "dev_final_only_cases": len(dataset["dev_pairs"]),
            "train_pairs_per_family": config.train_pairs_per_family,
            "dev_cases_per_family": config.dev_cases_per_family,
            "train_full_sequence_tokens": train_tokens,
            "maximum_sequence_tokens": maximum,
            "pair_suffix_alignment_passed": aligned,
            "observed_quality_prompt_overlap": 0,
            "benchmark_prompt_overlap": 0,
            "benchmark_rows_hashed": benchmark_counts,
        },
        "objective": {
            "total": (
                "0.5 * process_ce + 0.5 * final_ce + "
                "1.0 * KL(detach(process_final_logits) || final_logits)"
            ),
            "process_ce_weight": config.process_ce_weight,
            "final_ce_weight": config.final_ce_weight,
            "consistency_weight": config.consistency_weight,
            "temperature": config.consistency_temperature,
            "teacher_detach": config.teacher_detach,
            "optimizer_steps": config.max_steps,
            "learning_rate": config.learning_rate,
        },
        "evaluation": {
            "dev_cases": len(dataset["dev_pairs"]),
            "final_only": True,
            "generation_batch_size": config.generation_batch_size,
            "generation_max_new_tokens": config.generation_max_new_tokens,
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
                "pair_count_change",
                "step_change",
                "learning_rate_change",
                "loss_weight_change",
                "teacher_detach_change",
                "temperature_change",
                "prompt_change",
                "target_change",
                "generation_budget_change",
                "parser_change",
                "anchor_adapter_change",
                "benchmark_access",
                "canary_access",
                "holdout_access",
            ],
            "passed": (
                "Publish synthetic consistency evidence and separately "
                "pre-register any canary treatment."
            ),
            "failed": (
                "Preserve negative evidence and change mechanism/data on a "
                "new surface; do not tune on observed dev."
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
            "This experiment changes supervision mechanism on fresh synthetic "
            "pairs and untouched final-only dev. Passing would establish only "
            "synthetic transfer, not benchmark or final 4B superiority."
        ),
    }


def render_markdown(receipt: dict) -> str:
    counts = receipt["counts"]
    return f"""# Qwen3.5 Quality Consistency v1

## 变化

上一轮 answer-only SFT 只有 2 个 fresh-dev wins。这里不增加 dose，而是更换
监督机制：每个 train task 同时提供 verified process view 和 final-only view，
并用 detached-teacher KL 把 process final logits 约束到 final-only logits。

## 数据

- train pairs：`{counts['train_pairs']}`，每个 family
  `{counts['train_pairs_per_family']}`；
- untouched final-only dev：`{counts['dev_final_only_cases']}`，每个 family
  `{counts['dev_cases_per_family']}`；
- train full-sequence tokens：`{counts['train_full_sequence_tokens']:,}`；
- maximum sequence：`{counts['maximum_sequence_tokens']}`；
- process/final suffix alignment：
  `{counts['pair_suffix_alignment_passed']}` pairs 全部通过；
- 与已观察 quality surfaces prompt overlap：0；
- 与 GSM8K/MMLU/GPQA 题面 overlap：0；
- benchmark/canary/holdout rows、labels、outputs：0。

## Objective

`0.5 * process CE + 0.5 * final CE + 1.0 * KL(detach(process final logits) || final logits)`

- temperature 1.0；
- 256 pair optimizer steps；
- LR 5e-5；
- anchor 固定为 scaled-SFT adapter
  `{receipt['identity']['anchor_adapter_sha256']}`。

## Gate

- finite + independent reload；
- post accuracy > baseline；
- paired bootstrap CI lower > 0；
- exact McNemar p < 0.05；
- 至少 12 wins、0 losses；
- every-family 和 parse non-regression。

即使通过，benchmark/canary/holdout 仍需单独预注册。

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
