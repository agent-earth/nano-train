#!/usr/bin/env python3

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from nano_train.sft import sha256_file


ROOT = Path(__file__).resolve().parents[1]
PREREG = (
    ROOT / "docs/experiments/qwen35_rl_opd_admission_v1.preregister.json"
)
PUBLIC_JSON = ROOT / "docs/results/qwen35_rl_opd_admission_v1.public.json"
MARKDOWN = ROOT / "docs/results/qwen35_rl_opd_admission_v1.md"
EXPERIMENTS = {
    "rl": ROOT / "artifacts/qwen35-4b-rl-admission-v1",
    "opd": ROOT / "artifacts/qwen35-4b-opd-admission-v1",
}


def git_revision() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()


def _load(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def admission_gates(
    metrics: dict,
    reload: dict,
    *,
    runtime_failure_exists: bool,
) -> dict[str, bool]:
    steps = metrics["training"]["loss_curve"]
    return {
        "optimizer_steps_exact": (
            metrics["training"]["optimizer_steps"] == 2
            and len(steps) == 2
        ),
        "all_losses_finite": metrics["training"]["all_losses_finite"],
        "all_gradient_norms_finite": metrics["training"][
            "all_gradient_norms_finite"
        ],
        "adapter_logits_changed": metrics["adapter_effect"][
            "logits_changed"
        ],
        "finite_adapter_tensors": (
            reload["finite_adapter_tensors"] == 32
            and reload["nonfinite_adapter_tensors"] == 0
        ),
        "independent_reload_exact": (
            reload["reload_success"]
            and reload["probe_logits_exact"]
            and reload["adapter_sha256"]
            == metrics["identity"]["adapter_sha256"]
        ),
        "contamination_audit_passed": (
            metrics["contamination_audit"]["passed"]
            and metrics["contamination_audit"][
                "exact_normalized_prompt_overlap"
            ]
            == 0
            and metrics["contamination_audit"][
                "benchmark_labels_loaded"
            ]
            is False
            and metrics["contamination_audit"][
                "benchmark_outputs_loaded"
            ]
            is False
            and metrics["contamination_audit"][
                "canary_or_holdout_loaded"
            ]
            is False
        ),
        "runtime_failure_receipt_absent": (
            metrics["failure_receipt_exists"] is False
            and not runtime_failure_exists
        ),
    }


def build_report() -> dict:
    prereg = _load(PREREG)
    if (
        prereg["schema_version"]
        != "nano_train_rl_opd_admission_preregister_v1"
        or prereg["execution_boundary"]["training_started"] is not False
    ):
        raise ValueError("RL/OPD preregistration identity differs")
    experiments = {}
    for mode, root in EXPERIMENTS.items():
        metrics_path = root / "metrics.json"
        reload_path = root / "reload_validation.json"
        metrics = _load(metrics_path)
        reload = _load(reload_path)
        if (
            metrics["schema_version"]
            != "nano_train_rl_opd_admission_result_v1"
            or reload["schema_version"]
            != "nano_train_rl_opd_admission_reload_v1"
            or metrics["mode"] != mode
            or reload["mode"] != mode
        ):
            raise ValueError(f"{mode} admission result identity differs")
        steps = metrics["training"]["loss_curve"]
        gates = admission_gates(
            metrics,
            reload,
            runtime_failure_exists=(root / "failure.json").exists(),
        )
        experiments[mode] = {
            "experiment_id": metrics["experiment_id"],
            "identity": {
                "adapter_sha256": metrics["identity"]["adapter_sha256"],
                "metrics_sha256": sha256_file(metrics_path),
                "reload_sha256": sha256_file(reload_path),
                "student_model_config_sha256": metrics["identity"][
                    "student_model_config_sha256"
                ],
                "student_model_index_sha256": metrics["identity"][
                    "student_model_index_sha256"
                ],
                "teacher_model_config_sha256": metrics["identity"][
                    "teacher_model_config_sha256"
                ],
                "teacher_model_index_sha256": metrics["identity"][
                    "teacher_model_index_sha256"
                ],
                "trajectory_sha256": metrics["identity"][
                    "trajectories_sha256"
                ],
            },
            "training": {
                "optimizer_steps": metrics["training"]["optimizer_steps"],
                "trainable_parameters": metrics["training"][
                    "trainable_parameters"
                ],
                "total_losses": [
                    row["total_loss"] for row in steps
                ],
                "gradient_norms": [
                    row["gradient_norm"] for row in steps
                ],
                "reward_counts": {
                    str(value): metrics["raw"]["rewards"].count(value)
                    for value in sorted(set(metrics["raw"]["rewards"]))
                },
                "trajectory_rows": metrics["raw"]["trajectory_rows"],
            },
            "adapter_effect": metrics["adapter_effect"],
            "reload": {
                "finite_adapter_tensors": reload[
                    "finite_adapter_tensors"
                ],
                "nonfinite_adapter_tensors": reload[
                    "nonfinite_adapter_tensors"
                ],
                "probe_logits_exact": reload["probe_logits_exact"],
                "reload_success": reload["reload_success"],
            },
            "contamination_audit": metrics["contamination_audit"],
            "hardware": metrics["hardware"],
            "gates": gates,
            "admitted": all(gates.values()),
        }
    failed_attempt = _load(
        EXPERIMENTS["rl"] / "attempt-1.failure.json"
    )
    if (
        failed_attempt["optimizer_steps_completed"] != 0
        or failed_attempt["adapter_saved"] is not False
        or failed_attempt["treatment_changed"] is not False
    ):
        raise ValueError("RL implementation failure receipt differs")
    both_admitted = all(row["admitted"] for row in experiments.values())
    return {
        "schema_version": "nano_train_rl_opd_admission_public_v1",
        "campaign_id": "qwen35-rl-opd-admission-v1",
        "identity": {
            "preregister_sha256": sha256_file(PREREG),
            "preregister_revision": prereg["identity"]["code_revision"],
            "result_revision": git_revision(),
            "config_sha256": prereg["identity"]["config_sha256"],
        },
        "experiments": experiments,
        "failed_attempts": [
            {
                "mode": "rl",
                "attempt": 1,
                "failure_stage": failed_attempt["failure_stage"],
                "root_cause": failed_attempt["root_cause"],
                "optimizer_steps_completed": 0,
                "adapter_saved": False,
                "treatment_changed": False,
                "failure_receipt_sha256": sha256_file(
                    EXPERIMENTS["rl"] / "attempt-1.failure.json"
                ),
            }
        ],
        "decision": {
            "rl_implementation_admitted": experiments["rl"]["admitted"],
            "opd_implementation_admitted": experiments["opd"]["admitted"],
            "joint_implementation_admitted": both_admitted,
            "model_quality_improvement_established": False,
            "benchmark_allowed": False,
            "canary_allowed": False,
            "independent_holdout_allowed": False,
            "larger_training_allowed": False,
            "next_action": (
                "Keep RL/OPD available as versioned mechanisms, but wait for "
                "the peer paired-consistency replication and a separately "
                "pre-registered quality experiment before any scale-up or "
                "benchmark use."
            ),
        },
        "claim_boundary": (
            "RL and OPD pass implementation admission only. Two synthetic "
            "optimizer steps, finite adapters, changed probe logits, and "
            "exact reload do not establish a capability or benchmark gain."
        ),
    }


def render_markdown(report: dict) -> str:
    rows = []
    for mode in ("rl", "opd"):
        experiment = report["experiments"][mode]
        rows.append(
            f"| {mode.upper()} | "
            f"{experiment['training']['optimizer_steps']} | "
            f"{experiment['training']['total_losses']} | "
            f"{experiment['training']['gradient_norms']} | "
            f"{experiment['reload']['finite_adapter_tensors']}/32 | "
            f"{str(experiment['reload']['probe_logits_exact']).lower()} | "
            f"{str(experiment['admitted']).lower()} |"
        )
    rl = report["experiments"]["rl"]
    opd = report["experiments"]["opd"]
    failed = report["failed_attempts"][0]
    return f"""# Qwen3.5 RL / OPD Admission v1 Result

## 结论

RL 和 OPD 都通过了**实现准入**，但没有做 benchmark，也没有证明模型能力提升。

| Mode | Steps | Loss | Gradient norm | Finite tensors | Reload exact | Admitted |
| --- | ---: | --- | --- | ---: | --- | --- |
{chr(10).join(rows)}

## 具体做了什么

### RL

- Qwen3.5-4B 在 GPU0 上生成 2 条 synthetic arithmetic rollout；
- exact verifier 对两条都给 `+1`；
- 执行 2 次 REINFORCE + detached base-policy KL 更新；
- 917,504 个 q/v LoRA 参数参与训练；
- probe logits SHA 从
  `{rl['adapter_effect']['before_probe_logits_sha256']}` 变为
  `{rl['adapter_effect']['after_probe_logits_sha256']}`；
- 独立 reload 后 logits SHA 完全一致；
- peak GPU memory `{rl['hardware']['student_peak_allocated_gib']:.2f}` GiB。

### OPD

- fresh Qwen3.5-4B 在 GPU0 生成 2 条 on-policy rollout；
- 冻结 Qwen3.5-9B 在 GPU1 对相同 token sequence 输出 teacher logits；
- 4B 执行 2 次 teacher→student KL 更新；
- 917,504 个 q/v LoRA 参数参与训练；
- probe logits SHA 从
  `{opd['adapter_effect']['before_probe_logits_sha256']}` 变为
  `{opd['adapter_effect']['after_probe_logits_sha256']}`；
- 独立 reload 后 logits SHA 完全一致；
- student/teacher peak GPU memory
  `{opd['hardware']['student_peak_allocated_gib']:.2f}` /
  `{opd['hardware']['teacher_peak_allocated_gib']:.2f}` GiB。

## 污染审计

两个实验都只用 4 条新 synthetic prompt（2 train + 2 probe）。这些 prompt 与：

- GSM8K 1,319 个题面；
- MMLU 14,042 个题面；
- GPQA-Diamond 198 个题面；

做 normalized exact-hash 比较，重叠为 0。审计没有读取 benchmark label、模型
output、canary 或 independent holdout。

## 失败样例

RL attempt 1 在第一个 optimizer step 前失败：

- stage：`{failed['failure_stage']}`；
- root cause：{failed['root_cause']}；
- optimizer steps：0；
- adapter saved：false；
- treatment changed：false。

修复仅把 inference-mode rollout token tensor clone 成普通 tensor，未改变冻结的
task、reward、teacher、seed、LR、steps、LoRA、temperature、top-p 或 budget。
attempt 2 使用原 config SHA 成功。

## Evidence

- preregistration SHA：`{report['identity']['preregister_sha256']}`；
- RL metrics SHA：`{rl['identity']['metrics_sha256']}`；
- RL reload SHA：`{rl['identity']['reload_sha256']}`；
- RL adapter tree SHA：`{rl['identity']['adapter_sha256']}`；
- OPD metrics SHA：`{opd['identity']['metrics_sha256']}`；
- OPD reload SHA：`{opd['identity']['reload_sha256']}`；
- OPD adapter tree SHA：`{opd['identity']['adapter_sha256']}`。

## 不代表什么

这次只证明两套机制可以真实更新、finite、产生 adapter effect 并独立 reload。
它不证明训练后回答更好，不开放 benchmark/canary/holdout，也不允许扩大训练。
"""


def main() -> None:
    report = build_report()
    PUBLIC_JSON.parent.mkdir(parents=True, exist_ok=True)
    PUBLIC_JSON.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    MARKDOWN.write_text(render_markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "campaign_id": report["campaign_id"],
                "decision": report["decision"],
                "public_json": str(PUBLIC_JSON),
                "markdown": str(MARKDOWN),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
