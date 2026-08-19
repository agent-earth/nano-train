#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path

from safetensors import safe_open

from nano_train.sft import sha256_file


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts/execution-target-paired-sft-smoke-v1"
CONFIG = ROOT / "configs/sft/execution_target_paired_smoke_v1.json"
PREREGISTER = ROOT / (
    "docs/experiments/execution_target_paired_sft_smoke_v1.preregister.json"
)
RESCORE = ROOT / (
    "docs/results/execution_target_paired_sft_smoke_v1.rescore.public.json"
)
PUBLIC_JSON = ROOT / (
    "docs/results/execution_target_paired_sft_smoke_v1.public.json"
)
REPORT = ROOT / "docs/results/execution_target_paired_sft_smoke_v1.md"


def main() -> None:
    preregister = json.loads(PREREGISTER.read_text())
    metrics = json.loads((ARTIFACTS / "metrics.json").read_text())
    reload = json.loads((ARTIFACTS / "reload_validation.json").read_text())
    rescore = json.loads(RESCORE.read_text())
    generations = json.loads((ARTIFACTS / "generations.json").read_text())
    baseline_rows = {
        row["sample_id"]: row for row in generations["baseline"]
    }
    post_rows = {
        row["sample_id"]: row for row in generations["post_sft"]
    }
    raw_dataset = json.loads(
        Path(metrics["config"]["dataset_path"]).read_text(encoding="utf-8")
    )
    raw_dev = {
        row["sample_id"]: row
        for row in raw_dataset["samples"]
        if row["split"] == "dev"
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
    actual_schedule = [
        row["sample_ids"][0] for row in metrics["train_exposure"]
    ]
    schedule_matches = (
        actual_schedule
        == preregister["data"]["scheduled_train_sample_ids"]
    )
    stable = (
        finite_losses
        and nonfinite == 0
        and reload_matches
        and reload["adapter_sha256"] == metrics["adapter_sha256"]
        and schedule_matches
    )

    baseline = rescore["arms"]["baseline"]
    post = rescore["arms"]["post_sft"]
    json_families = {
        "coding-and-validation",
        "planning-and-state",
        "skill-routing-and-reflection",
        "tool-use-and-recovery",
    }
    json_non_regression = all(
        post["by_family"][family]["verified"]
        >= baseline["by_family"][family]["verified"]
        for family in json_families
    )
    aggregate_positive = rescore["verified_delta"] > 0
    final_positive = (
        post["by_view"]["final"]["verified"]
        > baseline["by_view"]["final"]["verified"]
    )
    pair_positive = rescore["pair_both_verified_delta"] > 0
    method_accepted = (
        stable
        and aggregate_positive
        and final_positive
        and pair_positive
        and json_non_regression
    )

    changed_final_cases = []
    for sample_id, source in raw_dev.items():
        if source["view"] != "final":
            continue
        baseline_output = baseline_rows[sample_id]["output"]
        post_output = post_rows[sample_id]["output"]
        if baseline_output != post_output:
            changed_final_cases.append(
                {
                    "sample_id": sample_id,
                    "pair_id": source["pair_id"],
                    "expression": source["task_spec"]["expression"],
                    "expected": source["messages"][-1]["content"],
                    "baseline": baseline_output,
                    "post_sft": post_output,
                }
            )
    changed_final_cases.sort(key=lambda row: row["pair_id"])

    report = {
        "schema_version": (
            "nano_train_execution_target_paired_sft_public_v1"
        ),
        "experiment_id": metrics["experiment_id"],
        "identity": {
            "preregister_commit": "09f5747",
            "preregister_sha256": sha256_file(PREREGISTER),
            "config_sha256": sha256_file(CONFIG),
            "dataset_file_sha256": rescore["dataset_file_sha256"],
            "dataset_canonical_sha256": rescore[
                "dataset_canonical_sha256"
            ],
            "release_manifest_sha256": rescore[
                "release_manifest_sha256"
            ],
            "adapter_sha256": metrics["adapter_sha256"],
        },
        "method": {
            "scheduled_train_rows": len(actual_schedule),
            "scheduled_train_sample_id_sha256": preregister["data"][
                "scheduled_train_sample_id_sha256"
            ],
            "validation_sample_id_sha256": preregister["data"][
                "validation_sample_id_sha256"
            ],
            "scheduled_views": preregister["data"]["scheduled_views"],
            "scheduled_json_by_family": preregister["data"][
                "scheduled_json_by_family"
            ],
            "scheduled_complete_pairs": preregister["data"][
                "scheduled_complete_pairs"
            ],
            "schedule_matches": schedule_matches,
        },
        "training": {
            "steps": metrics["config"]["max_steps"],
            "train_rows_available": metrics["dataset"]["train_samples"],
            "validation_rows": metrics["dataset"]["validation_samples"],
            "trainable_parameters": metrics["model"][
                "trainable_parameters"
            ],
            "loss_curve": metrics["loss_curve"],
            "peak_allocated_gib": metrics["hardware"][
                "peak_allocated_gib"
            ],
            "wall_seconds": metrics["wall_seconds"],
        },
        "evaluation": {
            "baseline": baseline,
            "post_sft": post,
            "verified_delta": rescore["verified_delta"],
            "pair_both_verified_delta": rescore[
                "pair_both_verified_delta"
            ],
            "final_view_verified_delta": (
                post["by_view"]["final"]["verified"]
                - baseline["by_view"]["final"]["verified"]
            ),
            "process_view_verified_delta": (
                post["by_view"]["process"]["verified"]
                - baseline["by_view"]["process"]["verified"]
            ),
            "json_verified_delta": (
                post["by_view"]["json_preservation"]["verified"]
                - baseline["by_view"]["json_preservation"]["verified"]
            ),
            "json_family_non_regression": json_non_regression,
            "changed_output_count": post["changed_output_count"],
            "reload_matches": reload_matches,
            "reload_peak_allocated_gib": reload["peak_allocated_gib"],
        },
        "failure_analysis": {
            "baseline_process_only_pairs": baseline["pair_summary"][
                "process_only_verified"
            ],
            "post_process_only_pairs": post["pair_summary"][
                "process_only_verified"
            ],
            "baseline_both_verified_pairs": baseline["pair_summary"][
                "both_verified"
            ],
            "post_both_verified_pairs": post["pair_summary"][
                "both_verified"
            ],
            "changed_final_case_count": len(changed_final_cases),
            "changed_final_cases": changed_final_cases,
            "mechanism": (
                "The base model already executes all 24 process views. "
                "Standard SFT changes some final-only outputs but never makes "
                "one correct, so the paired supervision does not transfer the "
                "verified process result into the final-only contract."
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
            "accepted_local_smoke": stable,
            "paired_execution_method_accepted": method_accepted,
            "aggregate_positive": aggregate_positive,
            "final_view_positive": final_positive,
            "pair_both_verified_positive": pair_positive,
            "json_family_non_regression": json_non_regression,
            "larger_training_allowed": False,
            "benchmark_allowed": False,
            "independent_holdout_allowed": False,
            "rl_allowed": False,
            "next_action": (
                "拒绝把标准 SFT 直接用于 paired views 的 execution transfer。"
                "不要根据这组 dev 继续搜索 steps、LR、seed、schedule、LoRA "
                "scope、prompt、parser 或 adapter weight。下一轮训练前先设计"
                "显式 consistency/distillation objective，把 process view 的"
                "正确最终值约束到 final-only logits。"
            ),
        },
        "claim_boundary": (
            "adapter 训练稳定，aggregate verified 增加 3 分，但全部来自 JSON "
            "routing 改善。final-only execution 仍是 0/24，both-verified "
            "pairs 仍是 0/24。因此本方法被拒绝，不允许扩大训练、访问 "
            "benchmark 或 independent holdout，也不允许启动 RL。"
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
    baseline = evaluation["baseline"]
    post = evaluation["post_sft"]
    changed_examples = "\n".join(
        (
            f"- `{row['expression']}`：expected `{row['expected']}`，"
            f"base `{row['baseline']}`，SFT `{row['post_sft']}`。"
        )
        for row in failure["changed_final_cases"][:6]
    )
    return f"""# Execution-Target Paired SFT Smoke v1

## 这次具体做了什么

- 数据：`skill-sft-execution-target-paired-v1`，512 train / 80 dev；
- 训练：Qwen3.5-4B，q/v-only LoRA，FP32，40 steps；
- 40 条 train exposure 包含 10 个完整 process/final pair，以及四个 JSON
  family 各 5 条；
- max sequence 704，generation budget 160；
- 配置、40 个 train ID、80 个 dev ID 和 decision rule 已在 commit
  `{report['identity']['preregister_commit']}` 中先提交；
- 训练后独立 reload 全部 80 条 dev。

## 合同是否执行

- schedule matches：{str(method['schedule_matches']).lower()}；
- scheduled complete pairs：{method['scheduled_complete_pairs']}；
- scheduled views：{method['scheduled_views']}；
- scheduled JSON：{method['scheduled_json_by_family']}。

## 稳定性

- 可训练参数：{training['trainable_parameters']:,}；
- 训练显存峰值：{training['peak_allocated_gib']:.2f} GiB；
- reload 显存峰值：{evaluation['reload_peak_allocated_gib']:.2f} GiB；
- 训练耗时：{training['wall_seconds']:.2f} 秒；
- adapter tensors：{report['adapter_validation']['tensor_count']}；
- non-finite tensors：{report['adapter_validation']['nonfinite_tensors']}；
- 独立 reload 一致：{str(evaluation['reload_matches']).lower()}。

## Corrected Dev

- aggregate verified：{baseline['verified']}/80 →
  {post['verified']}/80，delta {evaluation['verified_delta']:+d}；
- JSON verified：{baseline['by_view']['json_preservation']['verified']}/32 →
  {post['by_view']['json_preservation']['verified']}/32，delta
  {evaluation['json_verified_delta']:+d}；
- process verified：{baseline['by_view']['process']['verified']}/24 →
  {post['by_view']['process']['verified']}/24；
- final-only verified：{baseline['by_view']['final']['verified']}/24 →
  {post['by_view']['final']['verified']}/24；
- both-verified pairs：{baseline['pair_summary']['both_verified']}/24 →
  {post['pair_summary']['both_verified']}/24；
- changed outputs：{evaluation['changed_output_count']}/80。

aggregate 的 +3 全部来自 `skill-routing-and-reflection` 4/8 → 7/8。
execution 的 final-only 和 paired gate 都没有提升。

## 为什么没迁移

Base 4B 的 process view 已经是 24/24，但对应 final-only view 是 0/24。
训练后仍然是 24 个 process-only pair，0 个 both-verified pair。

发生变化但仍错误的 final-only 样例：

{changed_examples}

标准 SFT 同时看过两种 view，但没有把 process 中已经正确的最终值迁移到
final-only 输出。

## 决策

- paired_execution_method_accepted：
  {str(decision['paired_execution_method_accepted']).lower()}；
- aggregate_positive：{str(decision['aggregate_positive']).lower()}；
- final_view_positive：{str(decision['final_view_positive']).lower()}；
- pair_both_verified_positive：
  {str(decision['pair_both_verified_positive']).lower()}；
- larger training / benchmark / independent holdout / RL：全部关闭。

下一步：{decision['next_action']}

## Evidence

- config SHA256: `{report['identity']['config_sha256']}`;
- dataset canonical SHA256:
  `{report['identity']['dataset_canonical_sha256']}`;
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
