#!/usr/bin/env python3

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

from nano_train.quality_consistency import load_config
from nano_train.sft import sha256_file


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/quality_consistency/qwen35_quality_consistency_v1.json"
PREREG = (
    ROOT / "docs/experiments/qwen35_quality_consistency_v1.preregister.json"
)
ARTIFACTS = ROOT / "artifacts/qwen35-quality-consistency-v1"
PUBLIC_JSON = ROOT / "docs/results/qwen35_quality_consistency_v1.public.json"
MARKDOWN = ROOT / "docs/results/qwen35_quality_consistency_v1.md"


def git_revision() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()


def acceptance_gates(metrics: dict, reload: dict) -> dict[str, bool]:
    comparison = metrics["comparison"]
    baseline = metrics["baseline_dev"]
    post = metrics["post_dev"]
    return {
        "finite_training": (
            metrics["training"]["all_components_finite"]
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
        "exact_mcnemar_p_lt_005": comparison["mcnemar_exact_p"] < 0.05,
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
    generations_path = ARTIFACTS / "generations.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    reload = json.loads(reload_path.read_text(encoding="utf-8"))
    generations = json.loads(generations_path.read_text(encoding="utf-8"))
    if (
        prereg["schema_version"]
        != "nano_train_quality_consistency_preregister_v1"
        or prereg["execution_boundary"]["training_started"] is not False
        or metrics["schema_version"] != "nano_train_quality_consistency_result_v1"
        or reload["schema_version"] != "nano_train_quality_consistency_reload_v1"
    ):
        raise ValueError("quality consistency result identity differs")
    gates = acceptance_gates(metrics, reload)
    curve = metrics["training"]["loss_curve"]
    if len(curve) != config.max_steps or not all(
        all(
            math.isfinite(row[key])
            for key in (
                "process_ce",
                "final_ce",
                "consistency_kl",
                "gradient_norm",
            )
        )
        for row in curve
    ):
        raise ValueError("quality consistency curve differs")
    baseline = {
        row["case_id"]: row for row in generations["baseline"]
    }
    post = {row["case_id"]: row for row in generations["post"]}
    if set(baseline) != set(post):
        raise ValueError("quality consistency generation cases differ")
    wins = sorted(
        case_id
        for case_id in baseline
        if post[case_id]["correct"] and not baseline[case_id]["correct"]
    )
    losses = sorted(
        case_id
        for case_id in baseline
        if baseline[case_id]["correct"] and not post[case_id]["correct"]
    )
    if (
        len(wins) != metrics["comparison"]["paired_counts"]["candidate_only"]
        or len(losses)
        != metrics["comparison"]["paired_counts"]["baseline_only"]
    ):
        raise ValueError("quality consistency discordant cases differ")
    return {
        "schema_version": "nano_train_quality_consistency_public_v1",
        "experiment_id": config.experiment_id,
        "identity": {
            "preregister_sha256": sha256_file(PREREG),
            "preregister_revision": prereg["identity"]["code_revision"],
            "result_revision": git_revision(),
            "config_sha256": sha256_file(CONFIG),
            "metrics_sha256": sha256_file(metrics_path),
            "reload_sha256": sha256_file(reload_path),
            "adapter_sha256": metrics["identity"]["adapter_sha256"],
            "anchor_adapter_sha256": metrics["identity"][
                "anchor_adapter_sha256"
            ],
            "generations_sha256": metrics["identity"][
                "generations_sha256"
            ],
            "train_pair_ids_sha256": metrics["dataset_contract"]["identity"][
                "train_pair_ids_sha256"
            ],
            "dev_pair_ids_sha256": metrics["dataset_contract"]["identity"][
                "dev_pair_ids_sha256"
            ],
        },
        "data": {
            "train_pairs": metrics["training"]["train_pairs"],
            "dev_final_only_cases": metrics["baseline_dev"]["cases"],
            "observed_quality_prompt_overlap": prereg["counts"][
                "observed_quality_prompt_overlap"
            ],
            "benchmark_prompt_overlap": prereg["counts"][
                "benchmark_prompt_overlap"
            ],
            "benchmark_canary_holdout_rows_or_outputs": 0,
        },
        "training": {
            "optimizer_steps": metrics["training"]["optimizer_steps"],
            "all_components_finite": metrics["training"][
                "all_components_finite"
            ],
            "first_step": curve[0],
            "last_step": curve[-1],
            "minimum_consistency_kl": min(
                row["consistency_kl"] for row in curve
            ),
            "maximum_gradient_norm": max(
                row["gradient_norm"] for row in curve
            ),
        },
        "evaluation": {
            "baseline_dev": metrics["baseline_dev"],
            "post_dev": metrics["post_dev"],
            "comparison": metrics["comparison"],
            "candidate_only_win_case_ids": wins,
            "baseline_only_loss_case_ids": losses,
        },
        "reload": reload,
        "hardware": metrics["hardware"],
        "decision": {
            "gates": gates,
            "statistically_significant_local_improvement": (
                gates["post_accuracy_gt_baseline"]
                and gates["paired_bootstrap_ci_lower_gt_zero"]
                and gates["exact_mcnemar_p_lt_005"]
            ),
            "quality_consistency_candidate_admitted": all(gates.values()),
            "benchmark_allowed": False,
            "canary_allowed": False,
            "independent_holdout_allowed": False,
            "further_tuning_on_observed_dev_allowed": False,
            "next_action": (
                "Preserve this first significant local consistency gain, but "
                "do not promote because two baseline-only losses violate the "
                "frozen zero-loss gate. Use a separately pre-registered "
                "conservative adapter-routing or rollback policy on a new "
                "fresh local surface; do not tune consistency on observed dev."
            ),
        },
        "claim_boundary": (
            "The adapter establishes a significant fresh synthetic final-only "
            "gain but fails the stricter zero-loss admission rule. It does not "
            "authorize canary, benchmark, or holdout access."
        ),
    }


def render_markdown(report: dict) -> str:
    evaluation = report["evaluation"]
    comparison = evaluation["comparison"]
    decision = report["decision"]
    return f"""# Qwen3.5 Quality Consistency v1 Result

## 结论

这是当前第一个**统计显著**的 fresh local quality gain：

- baseline `{evaluation['baseline_dev']['correct']}/192`；
- consistency `{evaluation['post_dev']['correct']}/192`；
- delta `{comparison['delta']:+.4f}`；
- 95% CI
  `[{comparison['paired_bootstrap_95_ci'][0]:+.4f},
  {comparison['paired_bootstrap_95_ci'][1]:+.4f}]`；
- exact McNemar `p={comparison['mcnemar_exact_p']}`；
- 15 wins / 2 losses。

但候选仍被拒绝，因为预注册要求 0 baseline-only losses。

## Family

| Family | Baseline | Post |
| --- | ---: | ---: |
{chr(10).join(
    f"| {family} | {evaluation['baseline_dev']['by_family'][family]['correct']}/48 | "
    f"{evaluation['post_dev']['by_family'][family]['correct']}/48 |"
    for family in sorted(evaluation['baseline_dev']['by_family'])
)}

## 训练

- 256 fresh train pairs / 256 optimizer steps；
- process CE、final CE、detached-teacher KL 全部 finite；
- first step：`{report['training']['first_step']}`；
- last step：`{report['training']['last_step']}`；
- peak GPU memory：`{report['hardware']['peak_allocated_gib']:.2f}` GiB；
- independent reload：
  `{str(report['reload']['reload_success']).lower()}`。

## Gate

```json
{json.dumps(decision['gates'], indent=2, sort_keys=True)}
```

- significant local improvement：
  `{str(decision['statistically_significant_local_improvement']).lower()}`；
- candidate admitted：
  `{str(decision['quality_consistency_candidate_admitted']).lower()}`。

## Discordant cases

- wins：`{evaluation['candidate_only_win_case_ids']}`；
- losses：`{evaluation['baseline_only_loss_case_ids']}`。

这里只公开 case IDs，不公开 expression、target 或 model output。

## Evidence

- preregistration SHA：`{report['identity']['preregister_sha256']}`；
- metrics SHA：`{report['identity']['metrics_sha256']}`；
- reload SHA：`{report['identity']['reload_sha256']}`；
- anchor adapter SHA：`{report['identity']['anchor_adapter_sha256']}`；
- consistency adapter SHA：`{report['identity']['adapter_sha256']}`；
- generations SHA：`{report['identity']['generations_sha256']}`。

## 下一步

保留显著提升，但不访问 canary/benchmark。下一步只允许在全新 local surface
上预注册 conservative adapter routing / rollback，使 consistency 只在可安全
获益的 family/condition 生效并要求 0 losses。禁止在已观察 dev 上调整
consistency weight、steps、LR、seed、prompt、parser 或 adapter weight。
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
