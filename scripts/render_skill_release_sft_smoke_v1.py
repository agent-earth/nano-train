#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path

from safetensors import safe_open

from nano_train.sft import sha256_file


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts/skill-release-long-sequence-sft-smoke-v1"
CONFIG = ROOT / "configs/sft/skill_release_long_sequence_smoke_v1.json"
PUBLIC_JSON = ROOT / "docs/results/skill_release_long_sequence_sft_smoke_v1.public.json"
REPORT = ROOT / "docs/results/skill_release_long_sequence_sft_smoke_v1.md"
RESCORE = ROOT / (
    "docs/results/"
    "skill_release_long_sequence_sft_smoke_v1.rescore.public.json"
)


def main() -> None:
    metrics = json.loads((ARTIFACTS / "metrics.json").read_text())
    reload = json.loads((ARTIFACTS / "reload_validation.json").read_text())
    rescore = json.loads(RESCORE.read_text())
    adapter_path = ARTIFACTS / "adapter/adapter_model.safetensors"
    with safe_open(adapter_path, framework="pt", device="cpu") as handle:
        tensors = list(handle.keys())
        nonfinite = sum(
            not handle.get_tensor(name).isfinite().all().item()
            for name in tensors
        )
    losses = metrics["loss_curve"]
    baseline = metrics["baseline_validation"]
    post = metrics["post_sft_validation"]
    reload_matches = reload["validation"] == post
    stable = (
        all(row["loss"] >= 0 for row in losses)
        and nonfinite == 0
        and reload_matches
        and reload["adapter_sha256"] == metrics["adapter_sha256"]
    )
    corrected_baseline = rescore["arms"]["baseline"]["verified"]
    corrected_post = rescore["arms"]["post_sft"]["verified"]
    improved = corrected_post > corrected_baseline
    release = metrics["dataset"]["release"]
    report = {
        "schema_version": "nano_train_skill_release_sft_smoke_public_v1",
        "experiment_id": metrics["experiment_id"],
        "identity": {
            "config_sha256": sha256_file(CONFIG),
            "dataset_sha256": metrics["dataset"]["sha256"],
            "release_manifest_sha256": release["sha256"],
            "release_accepted_jsonl_sha256": release[
                "accepted_jsonl_sha256"
            ],
            "adapter_sha256": metrics["adapter_sha256"],
        },
        "data": {
            "dataset_id": metrics["dataset"]["dataset_id"],
            "train_samples": metrics["dataset"]["train_samples"],
            "validation_samples": metrics["dataset"]["validation_samples"],
            "max_length": metrics["config"]["max_length"],
            "families": sorted(post["by_family"]),
        },
        "training": {
            "steps": metrics["config"]["max_steps"],
            "gradient_checkpointing": metrics["config"][
                "gradient_checkpointing"
            ],
            "loss_curve": losses,
            "trainable_parameters": metrics["model"][
                "trainable_parameters"
            ],
            "peak_allocated_gib": metrics["hardware"][
                "peak_allocated_gib"
            ],
            "wall_seconds": metrics["wall_seconds"],
        },
        "evaluation": {
            "original_string_scorer": {
                "baseline": baseline,
                "post_sft": post,
            },
            "corrected_family_verifier": {
                "baseline_verified": corrected_baseline,
                "post_verified": corrected_post,
                "samples": rescore["arms"]["baseline"]["samples"],
                "verified_delta": rescore["verified_delta"],
                "by_family_baseline": rescore["arms"]["baseline"][
                    "by_family"
                ],
                "by_family_post": rescore["arms"]["post_sft"]["by_family"],
                "changed_output_count": rescore["arms"]["post_sft"][
                    "changed_output_count"
                ],
            },
            "reload_matches": reload_matches,
            "reload_peak_allocated_gib": reload["peak_allocated_gib"],
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
        "scorer_correction": {
            "reason": (
                "The original nano-train semantic scorer reduced skill-release "
                "JSON tasks to string equality, so equivalent JSON key order "
                "was incorrectly marked wrong. The corrected scorer executes "
                "the frozen family verifier from task_spec."
            ),
            "raw_metrics_modified": False,
        },
        "decision": {
            "accepted_local_smoke": stable,
            "quality_improved": improved,
            "scale_allowed": stable and improved,
            "benchmark_allowed": False,
            "holdout_allowed": False,
            "rl_allowed": False,
            "next_action": (
                "Keep full training closed. Increase only the bounded local "
                "training dose and synthetic dev sample count under a new "
                "pre-registered config; require a positive dev delta before "
                "any benchmark or RL work."
            ),
        },
        "claim_boundary": (
            "This smoke proves the 4B long-sequence LoRA path is finite, fits "
            "one V100, saves and reloads reproducibly, and consumes the "
            "released JSONL. It does not establish quality or benchmark uplift."
        ),
    }
    PUBLIC_JSON.parent.mkdir(parents=True, exist_ok=True)
    PUBLIC_JSON.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    REPORT.write_text(render_markdown(report), encoding="utf-8")


def render_markdown(report: dict) -> str:
    training = report["training"]
    evaluation = report["evaluation"]
    original = evaluation["original_string_scorer"]
    corrected = evaluation["corrected_family_verifier"]
    decision = report["decision"]
    losses = ", ".join(
        f"{row['loss']:.6f}" for row in training["loss_curve"]
    )
    return f"""# Skill Release Long-Sequence SFT Smoke v1

## 做了什么

- 模型：Qwen3.5-4B；
- 数据：`skill-sft-10k-10m-v2` release；
- 每个 family 取 2 条 train + 1 条 dev，共 10 / 5 条；
- 最大序列长度：{report['data']['max_length']}；
- LoRA：q_proj + v_proj，4 optimizer steps；
- gradient checkpointing：开启；
- 独立重新加载 adapter 并复跑同一 dev。

## 稳定性

- Loss：{losses}；
- Train peak：{training['peak_allocated_gib']:.2f} GiB；
- Reload peak：{evaluation['reload_peak_allocated_gib']:.2f} GiB；
- Adapter tensors：{report['adapter_validation']['tensor_count']}；
- Non-finite tensors：{report['adapter_validation']['nonfinite_tensors']}；
- Reload 与进程内结果一致：{str(evaluation['reload_matches']).lower()}。

## 质量

- 原始 string exact（保留，不改写）：
  {original['baseline']['exact']}/5 → {original['post_sft']['exact']}/5；
- 修正后的 family verifier：
  {corrected['baseline_verified']}/{corrected['samples']} →
  {corrected['post_verified']}/{corrected['samples']}；
- Verified delta：{corrected['verified_delta']:+d}；
- 改变输出：{corrected['changed_output_count']}/{corrected['samples']}。

这轮没有质量提升。它只证明长序列训练路径可运行、显存可承受、adapter 可保存并
独立重载。

## Scorer 更正

旧 scorer 把 JSON 输出退化为字符串完全一致，因此 key 顺序不同也会被误报失败。
现在按 release 的 `task_spec + verifier` 重算。Raw metrics 没有修改；更正结果作为
独立 public receipt 保存。

## 决策

- accepted_local_smoke：{str(decision['accepted_local_smoke']).lower()}；
- quality_improved：{str(decision['quality_improved']).lower()}；
- scale_allowed：{str(decision['scale_allowed']).lower()}；
- benchmark_allowed：false；
- holdout_allowed：false；
- rl_allowed：false。

下一步：{decision['next_action']}

## Evidence

- config SHA256: `{report['identity']['config_sha256']}`;
- dataset SHA256: `{report['identity']['dataset_sha256']}`;
- release manifest SHA256:
  `{report['identity']['release_manifest_sha256']}`;
- adapter SHA256: `{report['identity']['adapter_sha256']}`;
- metrics SHA256: `{report['artifacts']['metrics_sha256']}`;
- reload SHA256: `{report['artifacts']['reload_validation_sha256']}`;
- rescore SHA256: `{report['artifacts']['rescore_sha256']}`.

## 结论边界

{report['claim_boundary']}
"""


if __name__ == "__main__":
    main()
