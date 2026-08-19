#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path

from safetensors import safe_open

from nano_train.config import load_sft_smoke_config
from nano_train.data import (
    load_skill_release_dataset,
    skill_release_output_valid,
)
from nano_train.sft import sha256_file


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts/skill-release-expanded-lora-sft-v3"
BOUNDED_ARTIFACTS = ROOT / "artifacts/skill-release-bounded-dose-sft-v2"
CONFIG = ROOT / "configs/sft/skill_release_expanded_lora_v3.json"
RESCORE = ROOT / (
    "docs/results/skill_release_expanded_lora_sft_v3.rescore.public.json"
)
BOUNDED_DOSE = ROOT / (
    "docs/results/skill_release_bounded_dose_sft_v2.public.json"
)
PUBLIC_JSON = ROOT / (
    "docs/results/skill_release_expanded_lora_sft_v3.public.json"
)
REPORT = ROOT / "docs/results/skill_release_expanded_lora_sft_v3.md"


def main() -> None:
    metrics = json.loads((ARTIFACTS / "metrics.json").read_text())
    reload = json.loads((ARTIFACTS / "reload_validation.json").read_text())
    rescore = json.loads(RESCORE.read_text())
    bounded_dose = json.loads(BOUNDED_DOSE.read_text())
    config = load_sft_smoke_config(CONFIG)
    dataset = load_skill_release_dataset(
        config.dataset_path,
        config.release_manifest_path or "",
        train_samples_per_family=config.train_samples_per_family or 0,
        validation_samples_per_family=(
            config.validation_samples_per_family or 0
        ),
    )
    validation = {
        row["sample_id"]: row
        for row in dataset["samples"]
        if row["split"] == "validation"
    }
    generations = json.loads(
        (ARTIFACTS / "generations.json").read_text()
    )
    bounded_generations = json.loads(
        (BOUNDED_ARTIFACTS / "generations.json").read_text()
    )
    generation_arms = {
        "baseline": {
            row["sample_id"]: row for row in generations["baseline"]
        },
        "qv_only": {
            row["sample_id"]: row
            for row in bounded_generations["post_sft"]
        },
        "expanded": {
            row["sample_id"]: row for row in generations["post_sft"]
        },
    }
    adapter_path = ARTIFACTS / "adapter/adapter_model.safetensors"
    with safe_open(adapter_path, framework="pt", device="cpu") as handle:
        tensors = list(handle.keys())
        nonfinite = sum(
            not handle.get_tensor(name).isfinite().all().item()
            for name in tensors
        )

    reload_matches = reload["validation"] == metrics["post_sft_validation"]
    finite_losses = all(
        isinstance(row["loss"], (int, float)) and row["loss"] >= 0
        for row in metrics["loss_curve"]
    )
    stable = (
        finite_losses
        and nonfinite == 0
        and reload_matches
        and reload["adapter_sha256"] == metrics["adapter_sha256"]
    )
    baseline = rescore["arms"]["baseline"]
    post = rescore["arms"]["post_sft"]
    family_non_regression = all(
        post["by_family"][family]["verified"]
        >= baseline["by_family"][family]["verified"]
        for family in baseline["by_family"]
    )
    positive_delta = rescore["verified_delta"] > 0
    bounded_evaluation = bounded_dose["evaluation"]
    reasoning_family = "verified-reasoning"
    lost_reasoning_ids = []
    for sample_id, source in validation.items():
        if source["task_family"] != reasoning_family:
            continue
        qv_valid = skill_release_output_valid(
            source["task_family"],
            source.get("task_spec"),
            source.get("verifier"),
            generation_arms["qv_only"][sample_id]["output"],
        )
        expanded_valid = skill_release_output_valid(
            source["task_family"],
            source.get("task_spec"),
            source.get("verifier"),
            generation_arms["expanded"][sample_id]["output"],
        )
        if qv_valid and not expanded_valid:
            lost_reasoning_ids.append(sample_id)
    lost_reasoning_ids.sort()
    representative_id = lost_reasoning_ids[0]
    representative = validation[representative_id]

    report = {
        "schema_version": "nano_train_skill_release_expanded_lora_public_v1",
        "experiment_id": metrics["experiment_id"],
        "identity": {
            "config_sha256": sha256_file(CONFIG),
            "dataset_sha256": metrics["dataset"]["sha256"],
            "release_manifest_sha256": metrics["dataset"]["release"]["sha256"],
            "adapter_sha256": metrics["adapter_sha256"],
        },
        "method": {
            "control": "skill-release-bounded-dose-sft-v2",
            "frozen_fields": [
                "model",
                "release_data",
                "80_train_20_dev_split",
                "20_steps",
                "learning_rate",
                "seed",
                "precision",
                "sequence_length",
                "evaluation",
            ],
            "control_lora_targets": ["q_proj", "v_proj"],
            "treatment_lora_targets": metrics["config"]["lora_targets"],
        },
        "data": {
            "train_samples": metrics["dataset"]["train_samples"],
            "validation_samples": metrics["dataset"]["validation_samples"],
            "families": sorted(post["by_family"]),
            "max_length": metrics["config"]["max_length"],
        },
        "training": {
            "steps": metrics["config"]["max_steps"],
            "loss_curve": metrics["loss_curve"],
            "peak_allocated_gib": metrics["hardware"]["peak_allocated_gib"],
            "wall_seconds": metrics["wall_seconds"],
            "trainable_parameters": metrics["model"]["trainable_parameters"],
        },
        "evaluation": {
            "baseline": baseline,
            "post_sft": post,
            "verified_delta": rescore["verified_delta"],
            "family_non_regression": family_non_regression,
            "changed_output_count": post["changed_output_count"],
            "raw_exact_delta": post["exact"] - baseline["exact"],
            "reload_matches": reload_matches,
            "reload_peak_allocated_gib": reload["peak_allocated_gib"],
        },
        "control_comparison": {
            "qv_only_verified": bounded_evaluation["post_sft"]["verified"],
            "expanded_lora_verified": post["verified"],
            "verified_delta": (
                post["verified"]
                - bounded_evaluation["post_sft"]["verified"]
            ),
            "qv_only_changed_outputs": bounded_evaluation[
                "changed_output_count"
            ],
            "expanded_lora_changed_outputs": post["changed_output_count"],
            "qv_only_trainable_parameters": bounded_dose["training"][
                "trainable_parameters"
            ],
            "expanded_lora_trainable_parameters": metrics["model"][
                "trainable_parameters"
            ],
        },
        "failure_analysis": {
            "regressed_family": reasoning_family,
            "baseline_verified": baseline["by_family"][reasoning_family][
                "verified"
            ],
            "post_sft_verified": post["by_family"][reasoning_family][
                "verified"
            ],
            "regressed_sample_ids": lost_reasoning_ids,
            "representative_regression": {
                "sample_id": representative_id,
                "expression": representative["task_spec"]["expression"],
                "expected": representative["messages"][-1]["content"],
                "baseline": generation_arms["baseline"][
                    representative_id
                ]["output"],
                "qv_only": generation_arms["qv_only"][
                    representative_id
                ]["output"],
                "expanded_lora": generation_arms["expanded"][
                    representative_id
                ]["output"],
                "interpretation": (
                    "The expanded adapter emits the multiplication "
                    "intermediate result and omits the final subtraction."
                ),
            },
        },
        "adapter_validation": {
            "tensor_count": len(tensors),
            "nonfinite_tensors": nonfinite,
            "reload_success": reload["reload_success"],
            "stable": stable,
        },
        "artifacts": {
            "metrics_sha256": sha256_file(ARTIFACTS / "metrics.json"),
            "generations_sha256": sha256_file(
                ARTIFACTS / "generations.json"
            ),
            "reload_validation_sha256": sha256_file(
                ARTIFACTS / "reload_validation.json"
            ),
            "rescore_sha256": sha256_file(RESCORE),
        },
        "decision": {
            "accepted_local_smoke": stable,
            "expanded_lora_method_accepted": (
                stable and positive_delta and family_non_regression
            ),
            "positive_dev_delta": positive_delta,
            "family_non_regression": family_non_regression,
            "larger_training_allowed": False,
            "benchmark_allowed": False,
            "holdout_allowed": False,
            "rl_allowed": False,
            "next_action": (
                "拒绝在这版数据上使用 expanded-LoRA。不要根据已经看过的 "
                "dev 结果继续调 dose、LR、seed、adapter weight、parser 或 "
                "prompt。保留 q/v-only 对照，下一轮在新的本地评估面上预注册"
                " reasoning-preservation objective 或 verified-execution "
                "method。"
            ),
        },
        "claim_boundary": (
            "Expanded-LoRA 的训练数值正常、adapter 可以独立重载，也确实改变"
            "了更多输出；但 corrected verified dev 从 17/20 降到 16/20，"
            "verified-reasoning 从 1/4 降到 0/4。raw exact 从 5/20 升到 "
            "12/20 只说明格式和模板更接近目标，不代表质量提升。本结果拒绝"
            "该方法，不允许据此扩大训练、访问 benchmark/holdout 或启动 RL。"
        ),
    }
    PUBLIC_JSON.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    REPORT.write_text(render_markdown(report), encoding="utf-8")


def render_markdown(report: dict) -> str:
    baseline = report["evaluation"]["baseline"]
    post = report["evaluation"]["post_sft"]
    control = report["control_comparison"]
    failure = report["failure_analysis"]
    decision = report["decision"]
    return f"""# Skill Release Expanded-LoRA SFT v3

## 这次具体改了什么

这是一个单变量实验。对照组只训练 `q_proj` 和 `v_proj`；本次把 LoRA
target 扩展为 `q_proj`、`v_proj`、`gate_proj`、`up_proj`、`down_proj`。
模型、数据、80 train / 20 frozen dev、20 steps、LR、seed、FP32、max
length 和评估方式全部不变。

## 训练是否正常

- 训练参数：{report['training']['trainable_parameters']:,}；
- 训练耗时：{report['training']['wall_seconds']:.2f} 秒；
- 显存峰值：{report['training']['peak_allocated_gib']:.2f} GiB；
- Adapter tensors：{report['adapter_validation']['tensor_count']}；
- Non-finite tensors：{report['adapter_validation']['nonfinite_tensors']}；
- 独立进程 reload 与训练后输出一致：
  {str(report['evaluation']['reload_matches']).lower()}。

所以训练和保存流程是稳定的，下面的负结果不是 adapter 损坏或 reload
失败造成的。

## 正确指标

- Family verifier：{baseline['verified']}/{baseline['samples']} →
  {post['verified']}/{post['samples']}，delta
  {report['evaluation']['verified_delta']:+d}；
- 字符串 exact：{baseline['exact']}/{baseline['samples']} →
  {post['exact']}/{post['samples']}，delta
  {report['evaluation']['raw_exact_delta']:+d}；
- 改变输出：{report['evaluation']['changed_output_count']}/
  {post['samples']}；
- Family non-regression：
  {str(report['evaluation']['family_non_regression']).lower()}。

字符串 exact 上升，是因为 JSON 类输出更贴近固定模板；它没有变成真实
正确性提升。corrected family verifier 反而下降 1 题。

## 与 q/v-only 对照

- q/v-only：{control['qv_only_trainable_parameters']:,} 个可训练参数，
  changed {control['qv_only_changed_outputs']}/20，verified
  {control['qv_only_verified']}/20；
- expanded-LoRA：{control['expanded_lora_trainable_parameters']:,} 个可训练
  参数，changed {control['expanded_lora_changed_outputs']}/20，verified
  {control['expanded_lora_verified']}/20；
- 相对 q/v-only 的 verified delta：
  {control['verified_delta']:+d}。

扩展 LoRA 让更多输出发生变化，但没有带来更高正确率。

## 失败样例

`verified-reasoning` 从 {failure['baseline_verified']}/4 降到
{failure['post_sft_verified']}/4。代表样例：

- 表达式：`{failure['representative_regression']['expression']}`；
- 正确答案：`{failure['representative_regression']['expected']}`；
- base 4B：`{failure['representative_regression']['baseline']}`；
- q/v-only：`{failure['representative_regression']['qv_only']}`；
- expanded-LoRA：`{failure['representative_regression']['expanded_lora']}`。

`276` 是 `(68 + 24) * 3` 的中间值。expanded-LoRA 忽略了最后的 `-24`，
把原本唯一正确的 reasoning case 改错。

## 决策

- expanded_lora_method_accepted：
  {str(decision['expanded_lora_method_accepted']).lower()}；
- larger_training_allowed：
  {str(decision['larger_training_allowed']).lower()}；
- benchmark / holdout / RL：全部关闭。

下一步：{decision['next_action']}

## Evidence

- config SHA256: `{report['identity']['config_sha256']}`;
- dataset SHA256: `{report['identity']['dataset_sha256']}`;
- adapter SHA256: `{report['identity']['adapter_sha256']}`;
- metrics SHA256: `{report['artifacts']['metrics_sha256']}`;
- generations SHA256: `{report['artifacts']['generations_sha256']}`;
- reload SHA256: `{report['artifacts']['reload_validation_sha256']}`;
- rescore SHA256: `{report['artifacts']['rescore_sha256']}`.

## 结论边界

{report['claim_boundary']}
"""


if __name__ == "__main__":
    main()
