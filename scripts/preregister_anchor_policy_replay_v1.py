#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from transformers import AutoTokenizer

from nano_train.anchor_policy_replay import (
    ARMS,
    build_dataset,
    build_step_schedule,
    contamination_audit,
    load_config,
    public_contract,
    verify_identity,
)
from nano_train.sft import sha256_file


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs/anchor_policy_replay/"
    "qwen35_anchor_policy_replay_v1.json"
)
JSON_OUTPUT = (
    ROOT
    / "docs/experiments/"
    "qwen35_anchor_policy_replay_v1.preregister.json"
)
MARKDOWN_OUTPUT = (
    ROOT / "docs/experiments/qwen35_anchor_policy_replay_v1.md"
)


def git_revision() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()


def schedule_sha256(schedule: list[dict[str, str]]) -> str:
    canonical = json.dumps(
        schedule,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def build_receipt() -> dict:
    config = load_config(CONFIG)
    identity = verify_identity(config)
    dataset = build_dataset(config)
    contract = public_contract(dataset)
    audit = contamination_audit(config, dataset)
    if not audit["passed"]:
        raise ValueError("anchor policy replay contamination detected")
    schedules = {
        arm: build_step_schedule(config, dataset, arm) for arm in ARMS
    }
    if (
        [row["pair_id"] for row in schedules["control"]]
        != [row["pair_id"] for row in schedules["treatment"]]
    ):
        raise ValueError("anchor policy replay arm pair order differs")
    tokenizer = AutoTokenizer.from_pretrained(
        config.anchor_adapter_path,
        local_files_only=True,
    )
    maximum = 0
    aligned = 0
    train_full_sequence_tokens = 0
    train_final_target_tokens = 0
    for split in ("train_pairs", "dev_pairs"):
        for pair in dataset[split]:
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
                prompt_ids = tokenizer(
                    prompt,
                    add_special_tokens=False,
                ).input_ids
                full_ids = tokenizer(
                    full,
                    add_special_tokens=False,
                ).input_ids
                target_ids[view] = full_ids[len(prompt_ids) :]
                maximum = max(maximum, len(full_ids))
                if split == "train_pairs":
                    train_full_sequence_tokens += len(full_ids)
                    if view == "final":
                        train_final_target_tokens += len(target_ids[view])
            if (
                len(target_ids["process"]) >= len(target_ids["final"])
                and target_ids["process"][-len(target_ids["final"]) :]
                == target_ids["final"]
            ):
                aligned += 1
    total_pairs = len(dataset["train_pairs"]) + len(dataset["dev_pairs"])
    if aligned != total_pairs or maximum > config.max_length:
        raise ValueError("anchor policy replay tokenization differs")
    schedule_counts = {
        arm: {
            kind: sum(row["kind"] == kind for row in schedules[arm])
            for kind in sorted({row["kind"] for row in schedules[arm]})
        }
        for arm in ARMS
    }
    return {
        "schema_version": (
            "nano_train_anchor_policy_replay_preregister_v1"
        ),
        "experiment_id": config.experiment_id,
        "identity": {
            "code_revision": git_revision(),
            "config_sha256": sha256_file(CONFIG),
            "model_identity": identity,
            "schedule_sha256": {
                arm: schedule_sha256(schedules[arm]) for arm in ARMS
            },
        },
        "dataset_contract": contract,
        "counts": {
            "train_pairs": len(dataset["train_pairs"]),
            "dev_final_only_cases": len(dataset["dev_pairs"]),
            "train_pairs_per_family": config.train_pairs_per_family,
            "dev_cases_per_family": config.dev_cases_per_family,
            "train_full_sequence_tokens": train_full_sequence_tokens,
            "train_final_target_tokens": train_final_target_tokens,
            "maximum_sequence_tokens": maximum,
            "pair_suffix_alignment_passed": aligned,
            "schedule_counts": schedule_counts,
        },
        "contamination_audit": audit,
        "teacher_cache_contract": {
            "schema_version": "nano_train_anchor_policy_cache_v1",
            "path_is_ignored": True,
            "public_receipt_path": config.teacher_cache_receipt_path,
            "train_pairs": len(dataset["train_pairs"]),
            "final_target_positions_only": True,
            "top_k": config.anchor_policy_top_k,
            "residual_other_bucket": True,
            "temperature": config.anchor_policy_temperature,
            "teacher_model": "frozen_anchor_qwen35_4b_adapter",
            "teacher_uses_training_target_prefix": True,
            "teacher_uses_evaluation_expected_answer": False,
            "teacher_uses_case_correctness": False,
            "teacher_uses_observed_quality_outputs": False,
            "cache_generated_after_preregistration": True,
            "cache_generation_precedes_arm_training": True,
        },
        "arms": {
            "control": {
                "first_step_per_pair": "full_consistency",
                "second_step_per_pair": config.control_second_step,
                "anchor_policy_kl_weight": (
                    config.control_anchor_policy_kl_weight
                ),
                "optimizer_steps": config.max_steps_per_arm,
            },
            "treatment": {
                "first_step_per_pair": "full_consistency",
                "second_step_per_pair": config.treatment_second_step,
                "anchor_policy_kl_weight": (
                    config.treatment_anchor_policy_kl_weight
                ),
                "optimizer_steps": config.max_steps_per_arm,
            },
        },
        "objective": {
            "full_consistency": (
                "0.5 * process_ce + 0.5 * final_ce + "
                "1.0 * KL(detach(process_final_logits) || final_logits)"
            ),
            "control_replay": "0.5 * final_ce",
            "treatment_replay": (
                "0.5 * final_ce + 1.0 * "
                "KL(anchor_top64_plus_other || student_top64_plus_other)"
            ),
            "same_anchor_seed_data_order_ce_lr_and_total_steps": True,
            "isolated_factor": "anchor_policy_kl_on_replay_step",
        },
        "evaluation": {
            "shared_dev_cases": len(dataset["dev_pairs"]),
            "final_only": True,
            "generation_batch_size": config.generation_batch_size,
            "generation_max_new_tokens": config.generation_max_new_tokens,
            "bootstrap_samples": config.bootstrap_samples,
            "bootstrap_seed": config.bootstrap_seed,
            "baseline_rows_must_match_across_arms": True,
        },
        "acceptance": {
            "teacher_cache_finite_and_identity_verified": True,
            "both_arms_finite_and_reloadable": True,
            "treatment_accuracy_gt_anchor": True,
            "treatment_anchor_bootstrap_ci_lower_gt_zero": True,
            "treatment_anchor_exact_mcnemar_p_lt": 0.05,
            "treatment_anchor_minimum_wins": 12,
            "treatment_anchor_maximum_losses": 0,
            "treatment_every_family_non_regression_vs_anchor": True,
            "treatment_parse_non_regression_vs_anchor": True,
            "treatment_accuracy_gte_control": True,
            "treatment_losses_lt_control": True,
            "canary_allowed_after_pass": True,
            "benchmark_allowed_after_pass": False,
        },
        "decision_policy": {
            "forbidden_after_observation": [
                "dataset_change",
                "cache_change",
                "top_k_change",
                "temperature_change",
                "kl_weight_change",
                "schedule_change",
                "step_change",
                "learning_rate_change",
                "ce_weight_change",
                "prompt_change",
                "target_change",
                "generation_budget_change",
                "parser_change",
                "anchor_adapter_change",
                "arm_rerun",
                "third_arm",
                "benchmark_access",
                "canary_access_before_pass",
                "holdout_access",
            ],
            "passed": (
                "Publish matched ablation and consume only the exact treatment "
                "adapter in the existing pre-registered 211-case canary."
            ),
            "failed": (
                "Preserve both-arm evidence and do not tune or rerun on this "
                "dev surface."
            ),
        },
        "execution_boundary": {
            "teacher_cache_started": False,
            "training_started": False,
            "model_generation_started": False,
            "dev_observed": False,
            "benchmark_accessed": False,
            "canary_accessed": False,
            "this_commit_only_preregisters": True,
        },
        "claim_boundary": (
            "Passing establishes only fresh synthetic treatment admission and "
            "permits the frozen 211-case canary. It is not complete benchmark, "
            "holdout, or final 4B/9B superiority evidence."
        ),
    }


def render_markdown(receipt: dict) -> str:
    counts = receipt["counts"]
    return f"""# Qwen3.5 Anchor-Policy Replay v1

## 假设

上一 matched ablation 证明 gold final replay 显著提升净质量，但 control/treatment
都丢掉同一个 anchor win。这里在全新 60k/70k 数值区间加入 frozen anchor
policy KL，直接约束 replay step 的决策分布。

## 数据

- train pairs：`{counts['train_pairs']}`；
- untouched final-only dev：`{counts['dev_final_only_cases']}`；
- 每个 family train/dev：`{counts['train_pairs_per_family']}` /
  `{counts['dev_cases_per_family']}`；
- train full-sequence tokens：`{counts['train_full_sequence_tokens']:,}`；
- train final target tokens：`{counts['train_final_target_tokens']:,}`；
- maximum sequence：`{counts['maximum_sequence_tokens']}`；
- suffix alignment：`{counts['pair_suffix_alignment_passed']}`；
- 与六个已观察 local surfaces overlap：0；
- 与完整 GSM8K/MMLU/GPQA prompts overlap：0。

## Teacher cache

- teacher：冻结的相同 4B anchor adapter；
- 只缓存 train final-view supervised token positions；
- 每个位置 top-64 log probabilities + one residual other bucket；
- temperature 1.0；
- 采用 train target teacher-forcing 前缀，但不读取 dev/benchmark expected answer、
  correctness 或 observed outputs；
- raw cache 只写 ignored artifacts，公开收据只保存 identity/统计。

## Matched arms

- control replay：`0.5 final CE`；
- treatment replay：`0.5 final CE + 1.0 anchor-policy KL`；
- 两臂共享 anchor、seed、data、pair order、CE、LR、512 steps、prompt、parser
  和 generation budget；
- 唯一隔离因素：replay step 上的 anchor-policy KL。

## Gate

- teacher cache finite + identity verified；
- 两臂 finite + independent reload exact；
- treatment vs anchor：显著、至少 12 wins、0 losses、family/parse
  non-regression；
- treatment accuracy >= control；
- treatment anchor-only losses < control；
- baseline rows 跨两臂逐条一致。

通过只允许 treatment 进入已预注册 211-case canary；完整 benchmark 仍关闭。

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
                "teacher_cache_contract": receipt["teacher_cache_contract"],
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
