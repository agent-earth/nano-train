#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from transformers import AutoTokenizer

from nano_train.rl_opd_admission import (
    build_contamination_audit,
    load_config,
    model_identity_checks,
    sha256_file,
    task_prompt,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = (
    ROOT / "configs/admission/qwen35_4b_rl_admission_v1.json",
    ROOT / "configs/admission/qwen35_4b_opd_admission_v1.json",
)
JSON_OUTPUT = (
    ROOT / "docs/experiments/qwen35_rl_opd_admission_v1.preregister.json"
)
MARKDOWN_OUTPUT = (
    ROOT / "docs/experiments/qwen35_rl_opd_admission_v1.md"
)


def git_revision() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()


def build_receipt() -> dict:
    configs = [load_config(path) for path in CONFIGS]
    if [config.mode for config in configs] != ["rl", "opd"]:
        raise ValueError("RL/OPD admission config order differs")
    receipts = []
    student_tokenizer = AutoTokenizer.from_pretrained(
        configs[0].student_model_path,
        local_files_only=True,
    )
    teacher_tokenizer = AutoTokenizer.from_pretrained(
        configs[1].teacher_model_path,
        local_files_only=True,
    )
    tokenizer_contract = {
        "student_class": type(student_tokenizer).__name__,
        "teacher_class": type(teacher_tokenizer).__name__,
        "student_vocab_size": len(student_tokenizer),
        "teacher_vocab_size": len(teacher_tokenizer),
        "vocab_size_equal": len(student_tokenizer) == len(teacher_tokenizer),
        "eos_token_id_equal": (
            student_tokenizer.eos_token_id
            == teacher_tokenizer.eos_token_id
        ),
        "pad_token_id_equal": (
            student_tokenizer.pad_token_id
            == teacher_tokenizer.pad_token_id
        ),
    }
    if not all(
        tokenizer_contract[key]
        for key in (
            "vocab_size_equal",
            "eos_token_id_equal",
            "pad_token_id_equal",
        )
    ):
        raise ValueError("student and teacher tokenizers differ")
    for path, config in zip(CONFIGS, configs):
        prompts = [
            {
                "task_id": task["task_id"],
                "prompt": task_prompt(config, task),
                "expected": task["expected"],
            }
            for task in (*config.train_tasks, *config.probe_tasks)
        ]
        prompt_lengths = [
            len(
                student_tokenizer(
                    student_tokenizer.apply_chat_template(
                        [
                            {
                                "role": "system",
                                "content": config.system_prompt,
                            },
                            {"role": "user", "content": row["prompt"]},
                        ],
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    ),
                    add_special_tokens=False,
                ).input_ids
            )
            for row in prompts
        ]
        if min(prompt_lengths) < 20 or max(prompt_lengths) > 128:
            raise ValueError("synthetic prompt token lengths differ")
        identities = model_identity_checks(config)
        contamination = build_contamination_audit(config)
        receipts.append(
            {
                "experiment_id": config.experiment_id,
                "mode": config.mode,
                "config_path": str(path.relative_to(ROOT)),
                "config_sha256": sha256_file(path),
                "model_identities": identities,
                "contamination_audit": contamination,
                "synthetic_tasks": {
                    "train": list(config.train_tasks),
                    "probe": list(config.probe_tasks),
                    "prompt_lengths": prompt_lengths,
                    "maximum_prompt_tokens": max(prompt_lengths),
                },
                "training": {
                    "optimizer_steps": config.max_steps,
                    "seed": config.seed,
                    "student_dtype": config.student_dtype,
                    "teacher_dtype": config.teacher_dtype,
                    "learning_rate": config.learning_rate,
                    "lora_r": config.lora_r,
                    "lora_alpha": config.lora_alpha,
                    "lora_targets": list(config.lora_targets),
                    "rollout_temperature": config.rollout_temperature,
                    "rollout_top_p": config.rollout_top_p,
                    "rollout_max_new_tokens": (
                        config.rollout_max_new_tokens
                    ),
                    "reference_kl_weight": (
                        config.reference_kl_weight
                    ),
                },
                "objective": (
                    "REINFORCE exact-verifier reward plus detached base-policy "
                    "KL on student-generated rollouts."
                    if config.mode == "rl"
                    else "KL from frozen Qwen3.5-9B teacher logits to "
                    "Qwen3.5-4B student logits on 4B on-policy rollouts."
                ),
            }
        )
    return {
        "schema_version": "nano_train_rl_opd_admission_preregister_v1",
        "campaign_id": "qwen35-rl-opd-admission-v1",
        "identity": {
            "code_revision": git_revision(),
            "config_sha256": {
                row["mode"]: row["config_sha256"] for row in receipts
            },
        },
        "tokenizer_contract": tokenizer_contract,
        "experiments": receipts,
        "acceptance": {
            "each_experiment": {
                "optimizer_steps_exact": 2,
                "all_losses_finite": True,
                "all_gradient_norms_finite": True,
                "nonzero_finite_adapter_tensors": True,
                "probe_logits_changed_from_base": True,
                "independent_reload_probe_logits_exact": True,
                "failure_receipt_absent": True,
                "contamination_audit_passed": True,
            },
            "joint_admission": (
                "RL and OPD each pass every finite, effect, reload, identity, "
                "and contamination gate."
            ),
            "benchmark_allowed": False,
            "quality_claim_allowed": False,
            "larger_training_allowed": False,
        },
        "decision_policy": {
            "forbidden_after_observation": [
                "synthetic_task_change",
                "reward_change",
                "teacher_change",
                "rollout_temperature_change",
                "rollout_top_p_change",
                "rollout_budget_change",
                "optimizer_change",
                "learning_rate_change",
                "step_change",
                "seed_change",
                "lora_scope_change",
                "adapter_weight_change",
                "benchmark_access",
                "canary_access",
                "independent_holdout_access",
            ],
            "failed_smoke": (
                "Preserve failure evidence and redesign the method on a new "
                "synthetic surface; do not tune on observed smoke outputs."
            ),
            "passed_smoke": (
                "Establishes implementation admission only. It does not "
                "establish model quality or unblock benchmarks."
            ),
        },
        "execution_boundary": {
            "training_started": False,
            "model_generation_started": False,
            "rl_started": False,
            "opd_started": False,
            "this_commit_only_preregisters": True,
        },
        "claim_boundary": (
            "This receipt freezes two benchmark-free implementation admission "
            "smokes before generation or optimization. Passing proves only "
            "that the RL and OPD mechanisms run, update finite LoRA weights, "
            "and reload reproducibly."
        ),
    }


def render_markdown(receipt: dict) -> str:
    rows = []
    for experiment in receipt["experiments"]:
        training = experiment["training"]
        rows.append(
            f"| `{experiment['mode']}` | {training['optimizer_steps']} | "
            f"`{training['student_dtype']}` | `{training['teacher_dtype']}` | "
            f"`{experiment['config_sha256']}` |"
        )
    return f"""# Qwen3.5 RL / OPD Admission v1

## 这次要验证什么

这不是 benchmark 实验，也不是模型质量实验。只验证两件事：

1. Qwen3.5-4B 能否在 exact verifier 奖励下完成真实 RL 更新；
2. Qwen3.5-4B 能否对自己的 on-policy rollout，接受冻结
   Qwen3.5-9B token 分布的蒸馏更新。

| Mode | Steps | Student dtype | Teacher dtype | Config SHA256 |
| --- | ---: | --- | --- | --- |
{chr(10).join(rows)}

## 数据和污染边界

- 每个实验只有 2 条 train synthetic arithmetic 和 2 条 probe。
- synthetic prompt 与完整 GSM8K 1,319、MMLU 14,042、GPQA 198 题面做
  normalized exact-hash 对比，重叠为 0。
- 污染审计只读取 benchmark 题面列，不读取标签；不读取任何 benchmark、
  canary、holdout 模型输出。
- raw rollout、adapter、metrics 写入 ignored `artifacts/`，不会提交。

## 固定机制

- RL：4B 采样 rollout，exact verifier 给 `+1/-0.25/-1` reward，
  优化 REINFORCE loss，并用 detached base-policy KL 约束。
- OPD：4B 采样 rollout；冻结 9B 在 GPU1 对同一 token sequence 输出 logits；
  4B 在 GPU0 最小化 teacher→student KL。
- 两个实验都从 fresh base 4B 开始，不串联 adapter。
- FP32 student、q/v LoRA r=8 alpha=16、2 steps、LR 1e-5、
  seed 20260820、temperature 0.8、top-p 0.95、12-token rollout 全部冻结。

## 通过条件

- 恰好 2 个 optimizer steps；
- 所有 loss、gradient norm、adapter tensor 有限；
- adapter 使固定 probe logits 发生变化；
- 独立 reload 后 probe logits SHA256 逐字一致；
- failure receipt 不存在；
- 污染审计通过。

通过只说明实现可用，不说明能力提升，不自动开放 benchmark、canary、holdout
或更大训练。

## 禁止事项

{chr(10).join(f"- `{item}`" for item in receipt['decision_policy']['forbidden_after_observation'])}

## 执行边界

```json
{json.dumps(receipt['execution_boundary'], indent=2, sort_keys=True)}
```
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-output", default=str(JSON_OUTPUT))
    parser.add_argument("--markdown-output", default=str(MARKDOWN_OUTPUT))
    args = parser.parse_args()
    receipt = build_receipt()
    json_output = Path(args.json_output)
    markdown_output = Path(args.markdown_output)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_output.write_text(
        render_markdown(receipt),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "campaign_id": receipt["campaign_id"],
                "config_sha256": receipt["identity"]["config_sha256"],
                "tokenizer_contract": receipt["tokenizer_contract"],
                "execution_boundary": receipt["execution_boundary"],
                "json_output": str(json_output),
                "markdown_output": str(markdown_output),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
