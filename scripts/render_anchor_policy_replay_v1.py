#!/usr/bin/env python3

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

from nano_train.anchor_policy_replay import ARMS, FAMILIES, load_config
from nano_train.sft import sha256_file
from nano_train.synthetic_quality import paired_comparison


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs/anchor_policy_replay/"
    "qwen35_anchor_policy_replay_v1.json"
)
PREREG = (
    ROOT
    / "docs/experiments/"
    "qwen35_anchor_policy_replay_v1.preregister.json"
)
CACHE_RECEIPT = (
    ROOT
    / "docs/experiments/"
    "qwen35_anchor_policy_teacher_cache_v1.public.json"
)
ARTIFACTS = ROOT / "artifacts/qwen35-anchor-policy-replay-v1"
PUBLIC_JSON = (
    ROOT / "docs/results/qwen35_anchor_policy_replay_v1.public.json"
)
MARKDOWN = ROOT / "docs/results/qwen35_anchor_policy_replay_v1.md"


def git_revision() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()


def arm_is_finite_and_reloadable(
    metrics: dict,
    reload: dict,
) -> bool:
    curve = metrics["training"]["loss_curve"]
    return (
        metrics["training"]["all_components_finite"]
        and not metrics["failure_receipt_exists"]
        and bool(curve)
        and all(
            all(
                math.isfinite(value)
                for key, value in row.items()
                if key
                in {
                    "process_ce",
                    "final_ce",
                    "consistency_kl",
                    "anchor_policy_kl",
                    "objective",
                    "gradient_norm",
                }
            )
            for row in curve
        )
        and reload["reload_success"]
        and reload["metrics_exact"]
        and reload["generations_exact"]
        and reload["adapter_sha256"]
        == metrics["identity"]["adapter_sha256"]
    )


def admission_gates(
    control: dict,
    treatment: dict,
    reloads: dict[str, dict],
    cache_receipt: dict,
) -> dict[str, bool]:
    treatment_comparison = treatment["comparison"]
    control_comparison = control["comparison"]
    treatment_metrics = treatment["post_dev"]
    anchor_metrics = treatment["baseline_dev"]
    cache_sha256 = cache_receipt["identity"]["teacher_cache_sha256"]
    return {
        "teacher_cache_finite_and_identity_verified": (
            cache_receipt["summary"]["all_probabilities_finite"]
            and cache_receipt["summary"][
                "all_probability_sums_within_1e_5"
            ]
            and all(
                metrics["identity"]["teacher_cache_sha256"] == cache_sha256
                for metrics in (control, treatment)
            )
        ),
        "both_arms_finite_and_reloadable": all(
            arm_is_finite_and_reloadable(metrics, reloads[arm])
            for arm, metrics in (
                ("control", control),
                ("treatment", treatment),
            )
        ),
        "treatment_accuracy_gt_anchor": (
            treatment_comparison["candidate_accuracy"]
            > treatment_comparison["baseline_accuracy"]
        ),
        "treatment_anchor_bootstrap_ci_lower_gt_zero": (
            treatment_comparison["paired_bootstrap_95_ci"][0] > 0
        ),
        "treatment_anchor_exact_mcnemar_p_lt_005": (
            treatment_comparison["mcnemar_exact_p"] < 0.05
        ),
        "treatment_anchor_minimum_wins": (
            treatment_comparison["paired_counts"]["candidate_only"] >= 12
        ),
        "treatment_anchor_maximum_losses": (
            treatment_comparison["paired_counts"]["baseline_only"] == 0
        ),
        "treatment_every_family_non_regression_vs_anchor": all(
            treatment_metrics["by_family"][family]["correct"]
            >= anchor_metrics["by_family"][family]["correct"]
            for family in FAMILIES
        ),
        "treatment_parse_non_regression_vs_anchor": (
            treatment_metrics["parse_failures"]
            <= anchor_metrics["parse_failures"]
        ),
        "treatment_accuracy_gte_control": (
            treatment["post_dev"]["accuracy"]
            >= control["post_dev"]["accuracy"]
        ),
        "treatment_losses_lt_control": (
            treatment_comparison["paired_counts"]["baseline_only"]
            < control_comparison["paired_counts"]["baseline_only"]
        ),
    }


def _case_sets(generations: dict[str, list[dict]]) -> set[str]:
    baseline = {row["case_id"] for row in generations["baseline"]}
    post = {row["case_id"] for row in generations["post"]}
    if baseline != post:
        raise ValueError("anchor policy replay arm case sets differ")
    return baseline


def _discordant(
    candidate_rows: list[dict],
    baseline_rows: list[dict],
) -> dict[str, list[str]]:
    candidate = {row["case_id"]: row for row in candidate_rows}
    baseline = {row["case_id"]: row for row in baseline_rows}
    if set(candidate) != set(baseline):
        raise ValueError("anchor policy replay discordant sets differ")
    return {
        "candidate_only_win_case_ids": sorted(
            case_id
            for case_id in candidate
            if candidate[case_id]["correct"]
            and not baseline[case_id]["correct"]
        ),
        "baseline_only_loss_case_ids": sorted(
            case_id
            for case_id in candidate
            if baseline[case_id]["correct"]
            and not candidate[case_id]["correct"]
        ),
    }


def build_report() -> dict:
    config = load_config(CONFIG)
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    cache_receipt = json.loads(
        CACHE_RECEIPT.read_text(encoding="utf-8")
    )
    metrics_paths = {
        arm: ARTIFACTS / arm / "metrics.json" for arm in ARMS
    }
    reload_paths = {
        arm: ARTIFACTS / arm / "reload_validation.json" for arm in ARMS
    }
    generations_paths = {
        arm: ARTIFACTS / arm / "generations.json" for arm in ARMS
    }
    metrics = {
        arm: json.loads(path.read_text(encoding="utf-8"))
        for arm, path in metrics_paths.items()
    }
    reloads = {
        arm: json.loads(path.read_text(encoding="utf-8"))
        for arm, path in reload_paths.items()
    }
    generations = {
        arm: json.loads(path.read_text(encoding="utf-8"))
        for arm, path in generations_paths.items()
    }
    if (
        prereg["schema_version"]
        != "nano_train_anchor_policy_replay_preregister_v1"
        or prereg["execution_boundary"]["training_started"] is not False
        or prereg["identity"]["config_sha256"] != sha256_file(CONFIG)
        or cache_receipt["schema_version"]
        != "nano_train_anchor_policy_cache_public_v1"
        or any(
            metrics[arm]["schema_version"]
            != "nano_train_anchor_policy_replay_result_v1"
            or metrics[arm]["arm_id"] != arm
            or reloads[arm]["schema_version"]
            != "nano_train_anchor_policy_replay_reload_v1"
            or reloads[arm]["arm_id"] != arm
            for arm in ARMS
        )
    ):
        raise ValueError("anchor policy replay result identity differs")
    contracts = [metrics[arm]["dataset_contract"] for arm in ARMS]
    if (
        contracts[0] != contracts[1]
        or contracts[0] != prereg["dataset_contract"]
    ):
        raise ValueError("anchor policy replay data contract differs")
    expected_case_ids = {
        row["pair_id"]
        for row in prereg["dataset_contract"]["dev_pairs"]
    }
    for arm in ARMS:
        if (
            metrics[arm]["identity"]["schedule_sha256"]
            != prereg["identity"]["schedule_sha256"][arm]
            or metrics[arm]["training"]["optimizer_steps"]
            != config.max_steps_per_arm
            or not metrics[arm]["contamination_audit"]["passed"]
            or _case_sets(generations[arm]) != expected_case_ids
        ):
            raise ValueError(
                f"anchor policy replay {arm} execution differs"
            )
    if generations["control"]["baseline"] != generations["treatment"][
        "baseline"
    ]:
        raise ValueError("anchor policy replay baselines are not exact")
    treatment_vs_control = paired_comparison(
        generations["treatment"]["post"],
        generations["control"]["post"],
        bootstrap_samples=config.bootstrap_samples,
        bootstrap_seed=config.bootstrap_seed,
    )
    gates = admission_gates(
        metrics["control"],
        metrics["treatment"],
        reloads,
        cache_receipt,
    )
    discordant = {
        arm: _discordant(
            generations[arm]["post"],
            generations[arm]["baseline"],
        )
        for arm in ARMS
    }
    discordant["treatment_vs_control"] = _discordant(
        generations["treatment"]["post"],
        generations["control"]["post"],
    )
    admitted = all(gates.values())
    return {
        "schema_version": "nano_train_anchor_policy_replay_public_v1",
        "experiment_id": config.experiment_id,
        "identity": {
            "result_revision": git_revision(),
            "preregister_revision": prereg["identity"]["code_revision"],
            "preregister_sha256": sha256_file(PREREG),
            "config_sha256": sha256_file(CONFIG),
            "cache_receipt_sha256": sha256_file(CACHE_RECEIPT),
            "teacher_cache_sha256": cache_receipt["identity"][
                "teacher_cache_sha256"
            ],
            "case_contract_identity": prereg["dataset_contract"][
                "identity"
            ],
            "arms": {
                arm: {
                    "metrics_sha256": sha256_file(metrics_paths[arm]),
                    "reload_sha256": sha256_file(reload_paths[arm]),
                    "generations_sha256": sha256_file(
                        generations_paths[arm]
                    ),
                    "adapter_sha256": metrics[arm]["identity"][
                        "adapter_sha256"
                    ],
                    "schedule_sha256": metrics[arm]["identity"][
                        "schedule_sha256"
                    ],
                }
                for arm in ARMS
            },
        },
        "data": {
            "train_pairs": len(
                prereg["dataset_contract"]["train_pairs"]
            ),
            "dev_final_only_cases": len(
                prereg["dataset_contract"]["dev_pairs"]
            ),
            "observed_quality_prompt_overlap": prereg[
                "contamination_audit"
            ]["observed_quality_prompt_overlap"],
            "benchmark_prompt_overlap": prereg["contamination_audit"][
                "benchmark_prompt_overlap"
            ],
            "benchmark_canary_holdout_rows_or_outputs": 0,
            "baseline_rows_exact_across_arms": True,
        },
        "teacher_cache": {
            "summary": cache_receipt["summary"],
            "contract": cache_receipt["teacher_contract"],
        },
        "arms": {
            arm: {
                "training": {
                    "optimizer_steps": metrics[arm]["training"][
                        "optimizer_steps"
                    ],
                    "full_consistency_steps": metrics[arm]["training"][
                        "full_consistency_steps"
                    ],
                    "final_replay_steps": metrics[arm]["training"][
                        "final_replay_steps"
                    ],
                    "anchor_policy_kl_weight": metrics[arm]["training"][
                        "anchor_policy_kl_weight"
                    ],
                    "all_components_finite": metrics[arm]["training"][
                        "all_components_finite"
                    ],
                },
                "baseline_dev": metrics[arm]["baseline_dev"],
                "post_dev": metrics[arm]["post_dev"],
                "comparison_vs_anchor": metrics[arm]["comparison"],
                "reload": reloads[arm],
                "hardware": metrics[arm]["hardware"],
            }
            for arm in ARMS
        },
        "treatment_vs_control": treatment_vs_control,
        "discordant_cases": discordant,
        "decision": {
            "gates": gates,
            "treatment_admitted": admitted,
            "canary_allowed": admitted,
            "benchmark_allowed": False,
            "independent_holdout_allowed": False,
            "further_tuning_on_observed_dev_allowed": False,
            "next_action": (
                "Consume only the exact treatment adapter identity in the "
                "existing pre-registered 211-case canary."
                if admitted
                else "Reject this treatment and preserve both-arm evidence; "
                "do not tune or rerun on this dev surface."
            ),
        },
        "claim_boundary": (
            "This matched fresh synthetic ablation can admit only the frozen "
            "treatment adapter to the existing 211-case canary. It is not "
            "complete benchmark, holdout, or final 4B/9B superiority evidence."
        ),
    }


def render_markdown(report: dict) -> str:
    decision = report["decision"]
    rows = []
    for arm, payload in report["arms"].items():
        comparison = payload["comparison_vs_anchor"]
        rows.append(
            f"| {arm} | {payload['post_dev']['correct']}/256 | "
            f"{comparison['delta']:+.4f} | "
            f"{comparison['paired_counts']['candidate_only']} | "
            f"{comparison['paired_counts']['baseline_only']} | "
            f"{comparison['mcnemar_exact_p']:.6g} |"
        )
    treatment_vs_control = report["treatment_vs_control"]
    return f"""# Qwen3.5 Anchor-Policy Replay v1 Result

## 结论

- treatment admitted：
  `{str(decision['treatment_admitted']).lower()}`；
- 211-case canary allowed：
  `{str(decision['canary_allowed']).lower()}`；
- complete benchmark allowed：`false`；
- tuning/rerun on observed dev：`false`。

## Matched arms vs anchor

| Arm | Correct | Delta | Wins | Losses | McNemar p |
| --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(rows)}

两臂 baseline rows 逐条完全一致；唯一隔离因素是 treatment replay step 上
weight=1.0 的 frozen anchor top-64+other policy KL。

## Treatment vs control

- delta `{treatment_vs_control['delta']:+.4f}`；
- paired bootstrap 95% CI
  `[{treatment_vs_control['paired_bootstrap_95_ci'][0]:+.4f},
  {treatment_vs_control['paired_bootstrap_95_ci'][1]:+.4f}]`；
- exact McNemar `p={treatment_vs_control['mcnemar_exact_p']}`；
- wins/losses：
  `{treatment_vs_control['paired_counts']['candidate_only']}/
  {treatment_vs_control['paired_counts']['baseline_only']}`。

## Frozen gates

```json
{json.dumps(decision['gates'], indent=2, sort_keys=True)}
```

## Evidence

- prereg SHA：`{report['identity']['preregister_sha256']}`；
- config SHA：`{report['identity']['config_sha256']}`；
- cache receipt / raw SHA：
  `{report['identity']['cache_receipt_sha256']}` /
  `{report['identity']['teacher_cache_sha256']}`；
- control metrics/reload/generations：
  `{report['identity']['arms']['control']['metrics_sha256']}` /
  `{report['identity']['arms']['control']['reload_sha256']}` /
  `{report['identity']['arms']['control']['generations_sha256']}`；
- treatment metrics/reload/generations：
  `{report['identity']['arms']['treatment']['metrics_sha256']}` /
  `{report['identity']['arms']['treatment']['reload_sha256']}` /
  `{report['identity']['arms']['treatment']['generations_sha256']}`。

公开报告只包含聚合指标、case IDs 和 SHA，不包含 prompt、target、output 或
teacher logits。通过只允许 treatment 进入已预注册 211-case canary；完整 benchmark
与 independent holdout 继续关闭。
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
