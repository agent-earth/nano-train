#!/usr/bin/env python3

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from safetensors import safe_open

from nano_train.config import load_sft_smoke_config
from nano_train.data import evaluate_arithmetic, load_skill_release_dataset
from nano_train.sft import sha256_file


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts/skill-release-reasoning-preservation-sft-v4"
CONFIG = ROOT / "configs/sft/skill_release_reasoning_preservation_v4.json"
PREREGISTER = ROOT / (
    "docs/experiments/"
    "skill_release_reasoning_preservation_sft_v4.preregister.json"
)
RESCORE = ROOT / (
    "docs/results/"
    "skill_release_reasoning_preservation_sft_v4.rescore.public.json"
)
PUBLIC_JSON = ROOT / (
    "docs/results/skill_release_reasoning_preservation_sft_v4.public.json"
)
REPORT = ROOT / (
    "docs/results/skill_release_reasoning_preservation_sft_v4.md"
)


def main() -> None:
    config = load_sft_smoke_config(CONFIG)
    preregister = json.loads(PREREGISTER.read_text())
    metrics = json.loads((ARTIFACTS / "metrics.json").read_text())
    reload = json.loads((ARTIFACTS / "reload_validation.json").read_text())
    rescore = json.loads(RESCORE.read_text())
    generations = json.loads((ARTIFACTS / "generations.json").read_text())
    dataset = load_skill_release_dataset(
        config.dataset_path,
        config.release_manifest_path or "",
        train_samples_per_family=config.train_samples_per_family or 0,
        validation_samples_per_family=(
            config.validation_samples_per_family or 0
        ),
        validation_start_per_family=config.validation_start_per_family,
    )
    validation = {
        row["sample_id"]: row
        for row in dataset["samples"]
        if row["split"] == "validation"
    }
    generation_arms = {
        arm: {row["sample_id"]: row for row in rows}
        for arm, rows in generations.items()
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

    actual_schedule = [
        row["task_families"][0] for row in metrics["train_exposure"]
    ]
    schedule_matches = (
        actual_schedule == preregister["method"]["train_family_schedule"]
    )
    baseline = rescore["arms"]["baseline"]
    post = rescore["arms"]["post_sft"]
    json_families = sorted(
        family
        for family in baseline["by_family"]
        if family != "verified-reasoning"
    )
    json_non_regression = all(
        post["by_family"][family]["verified"]
        >= baseline["by_family"][family]["verified"]
        for family in json_families
    )
    positive_delta = rescore["verified_delta"] > 0

    reasoning_cases = []
    for sample_id, source in validation.items():
        if source["task_family"] != "verified-reasoning":
            continue
        expression = source["task_spec"]["expression"]
        baseline_output = generation_arms["baseline"][sample_id]["output"]
        post_output = generation_arms["post_sft"][sample_id]["output"]
        reasoning_cases.append(
            {
                "sample_id": sample_id,
                "expression": expression,
                "expected": f"FINAL: {evaluate_arithmetic(expression)}",
                "baseline": baseline_output,
                "post_sft": post_output,
                "changed": baseline_output != post_output,
            }
        )
    reasoning_cases.sort(key=lambda row: row["sample_id"])

    report = {
        "schema_version": (
            "nano_train_skill_release_reasoning_preservation_public_v1"
        ),
        "experiment_id": metrics["experiment_id"],
        "identity": {
            "preregister_commit": "4628044",
            "preregister_sha256": sha256_file(PREREGISTER),
            "config_sha256": sha256_file(CONFIG),
            "dataset_sha256": metrics["dataset"]["sha256"],
            "release_manifest_sha256": metrics["dataset"]["release"]["sha256"],
            "adapter_sha256": metrics["adapter_sha256"],
        },
        "method": {
            "control": preregister["method"]["control"],
            "frozen": preregister["method"]["frozen"],
            "changed": preregister["method"]["changed"],
            "validation_selection_rule": preregister["fresh_validation"][
                "selection_rule"
            ],
            "fresh_validation_sample_id_sha256": preregister[
                "fresh_validation"
            ]["sample_id_sha256"],
            "fresh_validation_overlap": {
                key: preregister["fresh_validation"][key]
                for key in (
                    "train_id_overlap",
                    "observed_dev_id_overlap",
                    "train_semantic_overlap",
                    "observed_dev_semantic_overlap",
                )
            },
            "planned_train_exposure_by_family": preregister["method"][
                "train_exposure_by_family"
            ],
            "actual_train_exposure_by_family": dict(
                sorted(Counter(actual_schedule).items())
            ),
            "schedule_matches": schedule_matches,
        },
        "training": {
            "steps": metrics["config"]["max_steps"],
            "train_samples": metrics["dataset"]["train_samples"],
            "validation_samples": metrics["dataset"]["validation_samples"],
            "trainable_parameters": metrics["model"]["trainable_parameters"],
            "loss_curve": metrics["loss_curve"],
            "peak_allocated_gib": metrics["hardware"]["peak_allocated_gib"],
            "wall_seconds": metrics["wall_seconds"],
        },
        "evaluation": {
            "baseline": baseline,
            "post_sft": post,
            "verified_delta": rescore["verified_delta"],
            "changed_output_count": post["changed_output_count"],
            "json_family_non_regression": json_non_regression,
            "reload_matches": reload_matches,
            "reload_peak_allocated_gib": reload["peak_allocated_gib"],
        },
        "failure_analysis": {
            "reasoning_baseline_verified": baseline["by_family"][
                "verified-reasoning"
            ]["verified"],
            "reasoning_post_verified": post["by_family"][
                "verified-reasoning"
            ]["verified"],
            "reasoning_changed_output_count": sum(
                row["changed"] for row in reasoning_cases
            ),
            "reasoning_cases": reasoning_cases,
            "mechanism": (
                "Ten answer-only reasoning exposures do not teach the "
                "multi-step arithmetic procedure. Three fresh reasoning "
                "outputs are byte-identical and the fourth changes only from "
                "FINAL: 342 to FINAL: 344 while the verified target is 467."
            ),
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
            "accepted_local_smoke": stable and schedule_matches,
            "reasoning_preservation_method_accepted": (
                stable
                and schedule_matches
                and positive_delta
                and json_non_regression
            ),
            "positive_dev_delta": positive_delta,
            "json_family_non_regression": json_non_regression,
            "larger_training_allowed": False,
            "benchmark_allowed": False,
            "independent_holdout_allowed": False,
            "rl_allowed": False,
            "next_action": (
                "拒绝只增加 answer-only reasoning 样本频次的方案。不要根据"
                "已经看到的结果继续调整 schedule、offset、dose、LR、seed、"
                "prompt、parser、adapter weight 或 route。下一轮必须改变监督"
                "结构，例如使用 verifier 检查的过程轨迹，并换一组新的冻结"
                "本地评估数据。"
            ),
        },
        "claim_boundary": (
            "这次预注册实验只证明：固定 schedule 的 q/v-only SFT 可以稳定"
            "训练，并在一组 fresh synthetic dev 上保住四个 JSON family。"
            "corrected verified 指标没有提升，reasoning 也没有提升。因此不能"
            "扩大训练，不能访问 benchmark 或 independent holdout，也不能"
            "启动 RL。"
        ),
    }
    PUBLIC_JSON.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    REPORT.write_text(render_markdown(report), encoding="utf-8")


def render_markdown(report: dict) -> str:
    method = report["method"]
    training = report["training"]
    evaluation = report["evaluation"]
    failure = report["failure_analysis"]
    decision = report["decision"]
    case_lines = "\n".join(
        (
            f"- `{row['expression']}`：expected `{row['expected']}`，"
            f"base `{row['baseline']}`，SFT `{row['post_sft']}`。"
        )
        for row in failure["reasoning_cases"]
    )
    planned_exposure = "、".join(
        f"{family} {count} 次"
        for family, count in method[
            "planned_train_exposure_by_family"
        ].items()
    )
    actual_exposure = "、".join(
        f"{family} {count} 次"
        for family, count in method[
            "actual_train_exposure_by_family"
        ].items()
    )
    return f"""# Skill Release Reasoning-Preservation SFT v4

## 这次具体做了什么

这是一个预注册的单方法实验：

- 保持 Qwen3.5-4B、release、80 条 train、q/v-only LoRA、20 steps、LR、
  seed、FP32、max length 和 verifier 不变；
- 把 20 steps 的训练暴露固定为 reasoning 10 次，四个 JSON family 合计
  10 次保护性 replay；
- fresh dev 固定取每个 family 在 release 顺序中的第 5–8 条，共 20 条；
- fresh dev 与训练集、旧 20 题在 sample ID 和 semantic hash 上的 overlap
  都是 0；
- 配置和 decision rule 已在 commit
  `{report['identity']['preregister_commit']}` 中先提交，再运行模型。

## 合同是否真的执行

- planned exposure：{planned_exposure}；
- actual exposure：{actual_exposure}；
- schedule matches：{str(method['schedule_matches']).lower()}；
- fresh dev ID SHA256：
  `{method['fresh_validation_sample_id_sha256']}`。

## 训练稳定性

- 可训练参数：{training['trainable_parameters']:,}；
- 训练耗时：{training['wall_seconds']:.2f} 秒；
- 训练显存峰值：{training['peak_allocated_gib']:.2f} GiB；
- reload 显存峰值：{evaluation['reload_peak_allocated_gib']:.2f} GiB；
- adapter tensors：{report['adapter_validation']['tensor_count']}；
- non-finite tensors：{report['adapter_validation']['nonfinite_tensors']}；
- 独立 reload 与训练后结果一致：
  {str(evaluation['reload_matches']).lower()}。

训练和保存流程正常，负结果不是运行故障。

## Corrected Verified Dev

- aggregate：{evaluation['baseline']['verified']}/20 →
  {evaluation['post_sft']['verified']}/20，delta
  {evaluation['verified_delta']:+d}；
- changed outputs：{evaluation['changed_output_count']}/20；
- 四个 JSON family：全部 4/4 → 4/4；
- verified-reasoning：
  {failure['reasoning_baseline_verified']}/4 →
  {failure['reasoning_post_verified']}/4。

## 为什么没有提升

{case_lines}

4 个 fresh reasoning case 中只有
{failure['reasoning_changed_output_count']} 个输出发生变化，而且仍然错误。
10 次 answer-only reasoning 暴露没有教会多步算术过程；它只产生了局部 token
偏移。

## 决策

- reasoning_preservation_method_accepted：
  {str(decision['reasoning_preservation_method_accepted']).lower()}；
- larger_training_allowed：
  {str(decision['larger_training_allowed']).lower()}；
- benchmark / independent holdout / RL：全部关闭。

下一步：{decision['next_action']}

## Evidence

- config SHA256: `{report['identity']['config_sha256']}`;
- preregister SHA256: `{report['identity']['preregister_sha256']}`;
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
