#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from transformers import AutoTokenizer

from nano_train.preservation_dual_view import (
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
    / "configs/preservation_dual_view/"
    "qwen35_preservation_dual_view_v1.json"
)
JSON_OUTPUT = (
    ROOT
    / "docs/experiments/"
    "qwen35_preservation_dual_view_v1.preregister.json"
)
MARKDOWN_OUTPUT = (
    ROOT
    / "docs/experiments/"
    "qwen35_preservation_dual_view_v1.md"
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
        raise ValueError("preservation dual-view contamination detected")
    schedules = {
        arm: build_step_schedule(config, dataset, arm) for arm in ARMS
    }
    if (
        [row["pair_id"] for row in schedules["control"]]
        != [row["pair_id"] for row in schedules["treatment"]]
    ):
        raise ValueError("preservation dual-view arm pair order differs")
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
        raise ValueError("preservation dual-view tokenization differs")
    schedule_counts = {
        arm: {
            kind: sum(row["kind"] == kind for row in schedules[arm])
            for kind in sorted({row["kind"] for row in schedules[arm]})
        }
        for arm in ARMS
    }
    return {
        "schema_version": (
            "nano_train_preservation_dual_view_preregister_v1"
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
        "arms": {
            "control": {
                "first_step_per_pair": "full_consistency",
                "second_step_per_pair": config.control_second_step,
                "optimizer_steps": config.max_steps_per_arm,
                "purpose": (
                    "Matched-dose control: repeat the complete process/final/"
                    "KL objective twice for every pair."
                ),
            },
            "treatment": {
                "first_step_per_pair": "full_consistency",
                "second_step_per_pair": config.treatment_second_step,
                "optimizer_steps": config.max_steps_per_arm,
                "purpose": (
                    "Preservation treatment: keep one complete objective step "
                    "and replace only the second step with final-only replay."
                ),
            },
        },
        "objective": {
            "full_consistency": (
                "0.5 * process_ce + 0.5 * final_ce + "
                "1.0 * KL(detach(process_final_logits) || final_logits)"
            ),
            "final_replay": "0.5 * final_ce",
            "learning_rate": config.learning_rate,
            "teacher_detach": config.teacher_detach,
            "temperature": config.consistency_temperature,
            "same_anchor_seed_data_order_and_total_steps": True,
            "isolated_factor": "second_step_objective",
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
                "schedule_change",
                "arm_change",
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
                "arm_rerun",
                "third_arm",
                "benchmark_access",
                "canary_access_before_pass",
                "holdout_access",
            ],
            "passed": (
                "Publish the matched ablation and consume only the treatment "
                "adapter identity in the existing pre-registered 211-case "
                "canary."
            ),
            "failed": (
                "Preserve both-arm negative evidence and do not tune on this "
                "dev surface."
            ),
        },
        "execution_boundary": {
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
    return f"""# Qwen3.5 Preservation-Aware Dual-View SFT v1

## 假设

旧 consistency adapter 在 fresh dev 上从 2/192 提升到 15/192，但覆盖掉 2
个 anchor wins。这里不使用旧 dev 调参，而是在全新数值区间做 matched two-arm
ablation，检验遗忘是否来自第二次 process/KL 压力。

## 数据

- train pairs：`{counts['train_pairs']}`；
- untouched final-only dev：`{counts['dev_final_only_cases']}`；
- 每个 family train/dev：`{counts['train_pairs_per_family']}` /
  `{counts['dev_cases_per_family']}`；
- train full-sequence tokens：`{counts['train_full_sequence_tokens']:,}`；
- train final target tokens：`{counts['train_final_target_tokens']:,}`；
- maximum sequence：`{counts['maximum_sequence_tokens']}`；
- suffix alignment：`{counts['pair_suffix_alignment_passed']}`；
- 与五个已观察 local surfaces overlap：0；
- 与完整 GSM8K/MMLU/GPQA prompts overlap：0。

## Matched arms

- control：每个 pair 连续执行两次完整
  `0.5 process CE + 0.5 final CE + 1.0 detached-teacher KL`；
- treatment：第一次相同，第二次替换为 `0.5 final CE`；
- 两臂共享 anchor、seed、data、pair order、LR、总步数 512、prompt、parser、
  generation budget；
- 唯一隔离因素：second-step objective。

## Gate

- 两臂 finite 且 independent reload exact；
- treatment vs anchor：accuracy 提升、CI lower > 0、McNemar p < 0.05、
  至少 12 wins、0 losses、family/parse non-regression；
- treatment accuracy >= control；
- treatment anchor-only losses < control anchor-only losses；
- baseline rows 在两臂间逐条完全一致。

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
                "arms": receipt["arms"],
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
