#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path

from safetensors import safe_open

from nano_train.sft import sha256_file


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts/skill-release-bounded-dose-sft-v2"
CONFIG = ROOT / "configs/sft/skill_release_bounded_dose_v2.json"
RESCORE = ROOT / (
    "docs/results/skill_release_bounded_dose_sft_v2.rescore.public.json"
)
PUBLIC_JSON = ROOT / (
    "docs/results/skill_release_bounded_dose_sft_v2.public.json"
)
REPORT = ROOT / "docs/results/skill_release_bounded_dose_sft_v2.md"


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
    reload_matches = reload["validation"] == metrics["post_sft_validation"]
    finite_losses = all(
        isinstance(row["loss"], (int, float))
        and row["loss"] >= 0
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
    report = {
        "schema_version": "nano_train_skill_release_bounded_dose_public_v1",
        "experiment_id": metrics["experiment_id"],
        "identity": {
            "config_sha256": sha256_file(CONFIG),
            "dataset_sha256": metrics["dataset"]["sha256"],
            "release_manifest_sha256": metrics["dataset"]["release"]["sha256"],
            "adapter_sha256": metrics["adapter_sha256"],
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
        "decision": {
            "accepted_local_smoke": stable,
            "positive_dev_delta": positive_delta,
            "family_non_regression": family_non_regression,
            "larger_training_allowed": (
                stable and positive_delta and family_non_regression
            ),
            "benchmark_allowed": False,
            "holdout_allowed": False,
            "rl_allowed": False,
            "next_action": (
                "Do not increase dose or search LR/seed. The adapter changes "
                "outputs but does not improve verified dev. Investigate the "
                "remaining verified-reasoning failures and choose one method-"
                "level intervention with protected JSON families."
            ),
        },
        "claim_boundary": (
            "The bounded dose proves stable optimization and adapter effect, "
            "but verified dev remains unchanged. It does not justify larger "
            "training, benchmark access, or quality claims."
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
    decision = report["decision"]
    losses = ", ".join(
        f"{row['loss']:.6f}" for row in report["training"]["loss_curve"]
    )
    return f"""# Skill Release Bounded-Dose SFT v2

## 配置

- 80 train / 20 frozen dev，五个 family 均衡；
- 20 optimizer steps；
- max length {report['data']['max_length']}；
- LoRA q_proj + v_proj；
- 模型、LR、seed 与 4-step smoke 相同，只增加 dose 和 dev 数。

## 稳定性

- Loss：{losses}；
- Peak：{report['training']['peak_allocated_gib']:.2f} GiB；
- Adapter tensors：{report['adapter_validation']['tensor_count']}；
- Non-finite tensors：{report['adapter_validation']['nonfinite_tensors']}；
- Independent reload 一致：{str(report['evaluation']['reload_matches']).lower()}。

## Verified Dev

- Baseline：{baseline['verified']}/{baseline['samples']}；
- Post-SFT：{post['verified']}/{post['samples']}；
- Delta：{report['evaluation']['verified_delta']:+d}；
- 改变输出：{report['evaluation']['changed_output_count']}/{post['samples']}；
- Family non-regression：
  {str(report['evaluation']['family_non_regression']).lower()}。

Adapter 确实改变了 9/20 个输出，但 verified 分数没有提高。

## 决策

- accepted_local_smoke：{str(decision['accepted_local_smoke']).lower()}；
- positive_dev_delta：{str(decision['positive_dev_delta']).lower()}；
- larger_training_allowed：{str(decision['larger_training_allowed']).lower()}；
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
