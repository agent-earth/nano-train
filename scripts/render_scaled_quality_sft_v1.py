#!/usr/bin/env python3

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

from nano_train.scaled_quality import load_config
from nano_train.sft import sha256_file


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/scaled_quality/qwen35_scaled_quality_sft_v1.json"
PREREG = (
    ROOT / "docs/experiments/qwen35_scaled_quality_sft_v1.preregister.json"
)
ARTIFACTS = ROOT / "artifacts/qwen35-scaled-quality-sft-v1"
PUBLIC_JSON = ROOT / "docs/results/qwen35_scaled_quality_sft_v1.public.json"
MARKDOWN = ROOT / "docs/results/qwen35_scaled_quality_sft_v1.md"


def git_revision() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()


def acceptance_gates(
    metrics: dict,
    reload: dict,
) -> dict[str, bool]:
    comparison = metrics["comparison"]
    baseline = metrics["baseline_dev"]
    post = metrics["post_sft_dev"]
    return {
        "finite_training": (
            metrics["training"]["all_losses_finite"]
            and metrics["training"]["all_gradient_norms_finite"]
            and not metrics["failure_receipt_exists"]
        ),
        "independent_reload_exact": (
            reload["reload_success"]
            and reload["metrics_exact"]
            and reload["generations_exact"]
            and reload["adapter_sha256"]
            == metrics["identity"]["adapter_sha256"]
        ),
        "post_accuracy_gt_baseline": (
            comparison["candidate_accuracy"] > comparison["baseline_accuracy"]
        ),
        "paired_bootstrap_ci_lower_gt_zero": (
            comparison["paired_bootstrap_95_ci"][0] > 0
        ),
        "exact_mcnemar_p_lt_005": (
            comparison["mcnemar_exact_p"] < 0.05
        ),
        "minimum_candidate_only_wins": (
            comparison["paired_counts"]["candidate_only"] >= 12
        ),
        "maximum_baseline_only_losses": (
            comparison["paired_counts"]["baseline_only"] == 0
        ),
        "every_family_non_regression": all(
            post["by_family"][family]["correct"]
            >= baseline["by_family"][family]["correct"]
            for family in post["by_family"]
        ),
        "parse_failures_non_regression": (
            post["parse_failures"] <= baseline["parse_failures"]
        ),
    }


def build_report() -> dict:
    config = load_config(CONFIG)
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    metrics_path = ARTIFACTS / "metrics.json"
    reload_path = ARTIFACTS / "reload_validation.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    reload = json.loads(reload_path.read_text(encoding="utf-8"))
    if (
        prereg["schema_version"]
        != "nano_train_scaled_quality_sft_preregister_v1"
        or prereg["execution_boundary"]["training_started"] is not False
        or metrics["schema_version"]
        != "nano_train_scaled_quality_sft_result_v1"
        or reload["schema_version"] != "nano_train_scaled_quality_reload_v1"
    ):
        raise ValueError("scaled quality result identity differs")
    gates = acceptance_gates(metrics, reload)
    losses = metrics["training"]["loss_curve"]
    if (
        len(losses) != config.max_steps
        or not all(
            math.isfinite(row["loss"])
            and math.isfinite(row["gradient_norm"])
            for row in losses
        )
    ):
        raise ValueError("scaled quality loss curve differs")
    wins = [
        row["case_id"]
        for row in json.loads(
            (ARTIFACTS / "generations.json").read_text(encoding="utf-8")
        )["post_sft"]
        if row["correct"]
    ]
    return {
        "schema_version": "nano_train_scaled_quality_sft_public_v1",
        "experiment_id": config.experiment_id,
        "identity": {
            "preregister_sha256": sha256_file(PREREG),
            "preregister_revision": prereg["identity"]["code_revision"],
            "result_revision": git_revision(),
            "config_sha256": sha256_file(CONFIG),
            "metrics_sha256": sha256_file(metrics_path),
            "reload_sha256": sha256_file(reload_path),
            "adapter_sha256": metrics["identity"]["adapter_sha256"],
            "generations_sha256": metrics["identity"][
                "generations_sha256"
            ],
            "train_case_ids_sha256": metrics["dataset_contract"]["identity"][
                "train_case_ids_sha256"
            ],
            "dev_case_ids_sha256": metrics["dataset_contract"]["identity"][
                "dev_case_ids_sha256"
            ],
        },
        "data": {
            "train_rows": metrics["training"]["train_rows"],
            "dev_rows": metrics["baseline_dev"]["cases"],
            "unique_train_exposures": metrics["training"][
                "unique_exposures"
            ],
            "observed_quality_prompt_overlap": prereg["counts"][
                "observed_quality_prompt_overlap"
            ],
            "train_dev_prompt_overlap": prereg["counts"][
                "train_dev_prompt_overlap"
            ],
            "benchmark_canary_holdout_rows_or_outputs": 0,
        },
        "training": {
            "optimizer_steps": metrics["training"]["optimizer_steps"],
            "trainable_parameters": metrics["training"][
                "trainable_parameters"
            ],
            "first_loss": losses[0],
            "last_loss": losses[-1],
            "minimum_loss": min(row["loss"] for row in losses),
            "maximum_gradient_norm": max(
                row["gradient_norm"] for row in losses
            ),
            "all_losses_finite": metrics["training"][
                "all_losses_finite"
            ],
            "all_gradient_norms_finite": metrics["training"][
                "all_gradient_norms_finite"
            ],
        },
        "evaluation": {
            "baseline_dev": metrics["baseline_dev"],
            "post_sft_dev": metrics["post_sft_dev"],
            "comparison": metrics["comparison"],
            "candidate_only_win_case_ids": wins,
        },
        "reload": reload,
        "hardware": metrics["hardware"],
        "decision": {
            "gates": gates,
            "scaled_sft_quality_candidate_admitted": all(gates.values()),
            "benchmark_allowed": False,
            "canary_allowed": False,
            "independent_holdout_allowed": False,
            "further_tuning_on_observed_dev_allowed": False,
            "next_action": (
                "Reject this scaled SFT candidate because the directional 2/96 "
                "gain is not significant and is below the frozen 12-win gate. "
                "Do not tune on this dev; change the supervision mechanism or "
                "use a separately generated train/dev surface."
            ),
        },
        "claim_boundary": (
            "The result shows two fresh synthetic execution wins after scaled "
            "SFT but does not establish a stable quality gain, benchmark gain, "
            "or permission to access canary/holdout."
        ),
    }


def render_markdown(report: dict) -> str:
    evaluation = report["evaluation"]
    comparison = evaluation["comparison"]
    gates = report["decision"]["gates"]
    return f"""# Qwen3.5 Scaled Quality SFT v1 Result

## 结论

候选被拒绝。512 条新 train rows、128 steps 的 SFT 在 untouched dev 上从
`{evaluation['baseline_dev']['correct']}/96` 提升到
`{evaluation['post_sft_dev']['correct']}/96`，但只有 2 wins / 0 losses：

- delta `{comparison['delta']:+.4f}`；
- 95% bootstrap CI
  `[{comparison['paired_bootstrap_95_ci'][0]:+.4f},
  {comparison['paired_bootstrap_95_ci'][1]:+.4f}]`；
- exact McNemar `p={comparison['mcnemar_exact_p']}`。

CI 下界仍为 0，p 不显著，并且未达到预注册的 12-win gate。

## 训练

- 512 train rows，512 unique exposures；
- 128 optimizer steps，batch 4，exactly one epoch；
- trainable LoRA parameters：
  `{report['training']['trainable_parameters']:,}`；
- first loss：`{report['training']['first_loss']['loss']:.6f}`；
- last loss：`{report['training']['last_loss']['loss']:.6f}`；
- minimum loss：`{report['training']['minimum_loss']:.6f}`；
- maximum gradient norm：
  `{report['training']['maximum_gradient_norm']:.6f}`；
- peak GPU memory：`{report['hardware']['peak_allocated_gib']:.2f}` GiB；
- independent reload：
  `{str(report['reload']['reload_success']).lower()}`。

## Fresh dev

| Family | Base | Post-SFT |
| --- | ---: | ---: |
{chr(10).join(
    f"| {family} | {evaluation['baseline_dev']['by_family'][family]['correct']}/24 | "
    f"{evaluation['post_sft_dev']['by_family'][family]['correct']}/24 |"
    for family in sorted(evaluation['baseline_dev']['by_family'])
)}

两条 win 分别来自 exact-division 和 repeated-operand。public report 只保留
case IDs，不包含 raw expression、target 或 output。

## Gate

```json
{json.dumps(gates, indent=2, sort_keys=True)}
```

## Evidence

- preregistration SHA：`{report['identity']['preregister_sha256']}`；
- metrics SHA：`{report['identity']['metrics_sha256']}`；
- reload SHA：`{report['identity']['reload_sha256']}`；
- adapter tree SHA：`{report['identity']['adapter_sha256']}`；
- generations SHA：`{report['identity']['generations_sha256']}`。

## 下一步

保留这 2 个 directionally correct wins，但拒绝当前 adapter。不要在已观察的
96 dev rows 上增加 steps、改 LR、seed、LoRA、prompt、parser 或 adapter
weight。下一轮必须更换监督机制，或使用全新的 train/dev surface。

benchmark、canary 和 independent holdout 继续关闭。
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
                "experiment_id": report["experiment_id"],
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
