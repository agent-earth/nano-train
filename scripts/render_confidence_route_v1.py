#!/usr/bin/env python3

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

from nano_train.confidence_route import (
    FAMILIES,
    combine,
    comparison,
    contamination_audit,
    build_cases,
    load_config,
    routed_metrics,
)
from nano_train.sft import sha256_file


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/confidence_route/qwen35_confidence_route_v1.json"
PREREG = (
    ROOT / "docs/experiments/qwen35_confidence_route_v1.preregister.json"
)
ARTIFACTS = ROOT / "artifacts/qwen35-confidence-route-v1"
PUBLIC_JSON = ROOT / "docs/results/qwen35_confidence_route_v1.public.json"
MARKDOWN = ROOT / "docs/results/qwen35_confidence_route_v1.md"


def git_revision() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()


def load_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def all_scores_finite(*score_rows: list[dict]) -> bool:
    expected_fields = {
        "anchor_candidate_mean_logprob",
        "consistency_candidate_mean_logprob",
    }
    return all(
        set(row).issuperset(expected_fields)
        and all(math.isfinite(row[field]) for field in expected_fields)
        for rows in score_rows
        for row in rows
    )


def admission_gates(
    paired: dict,
    candidate_metrics: dict,
    anchor_metrics: dict,
    *,
    scores_finite: bool,
) -> dict[str, bool]:
    return {
        "routed_accuracy_gt_anchor": (
            paired["candidate_accuracy"] > paired["baseline_accuracy"]
        ),
        "paired_bootstrap_ci_lower_gt_zero": (
            paired["paired_bootstrap_95_ci"][0] > 0
        ),
        "exact_mcnemar_p_lt_005": paired["mcnemar_exact_p"] < 0.05,
        "minimum_candidate_only_wins": (
            paired["paired_counts"]["candidate_only"] >= 6
        ),
        "maximum_anchor_only_losses": (
            paired["paired_counts"]["baseline_only"] == 0
        ),
        "every_family_non_regression": all(
            candidate_metrics["by_family"][family]["correct"]
            >= anchor_metrics["by_family"][family]["correct"]
            for family in FAMILIES
        ),
        "parse_failures_non_regression": (
            candidate_metrics["parse_failures"]
            <= anchor_metrics["parse_failures"]
        ),
        "all_scores_finite": scores_finite,
    }


def build_report() -> dict:
    config = load_config(CONFIG)
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    cases = build_cases(config)
    audit = contamination_audit(config, cases)
    generation_paths = {
        arm: ARTIFACTS / "generation" / arm / "cases.jsonl"
        for arm in ("anchor", "consistency")
    }
    generation_summary_paths = {
        arm: ARTIFACTS / "generation" / arm / "summary.json"
        for arm in ("anchor", "consistency")
    }
    score_paths = {
        arm: ARTIFACTS / "scores" / arm / "scores.jsonl"
        for arm in ("anchor", "consistency")
    }
    score_summary_paths = {
        arm: ARTIFACTS / "scores" / arm / "summary.json"
        for arm in ("anchor", "consistency")
    }
    generation = {
        arm: load_rows(path) for arm, path in generation_paths.items()
    }
    generation_summaries = {
        arm: json.loads(path.read_text(encoding="utf-8"))
        for arm, path in generation_summary_paths.items()
    }
    scores = {arm: load_rows(path) for arm, path in score_paths.items()}
    score_summaries = {
        arm: json.loads(path.read_text(encoding="utf-8"))
        for arm, path in score_summary_paths.items()
    }
    expected_case_ids = {row["case_id"] for row in cases}
    all_case_sets = [
        {row["case_id"] for row in rows}
        for rows in (*generation.values(), *scores.values())
    ]
    if (
        prereg["schema_version"]
        != "nano_train_confidence_route_preregister_v1"
        or prereg["identity"]["config_sha256"] != sha256_file(CONFIG)
        or prereg["case_contract"]["case_contract_sha256"]
        != generation_summaries["anchor"]["case_contract"][
            "case_contract_sha256"
        ]
        or prereg["case_contract"]["case_contract_sha256"]
        != generation_summaries["consistency"]["case_contract"][
            "case_contract_sha256"
        ]
        or any(case_ids != expected_case_ids for case_ids in all_case_sets)
        or any(len(rows) != len(cases) for rows in generation.values())
        or any(len(rows) != len(cases) for rows in scores.values())
        or any(
            summary["rows"] != len(cases)
            for summary in score_summaries.values()
        )
        or not audit["passed"]
    ):
        raise ValueError("confidence route result identity differs")

    routed, route_summary = combine(config)
    routed_path = ARTIFACTS / "routed" / "cases.jsonl"
    routed_path.parent.mkdir(parents=True, exist_ok=True)
    routed_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in routed),
        encoding="utf-8",
    )
    anchor_metrics = generation_summaries["anchor"]["metrics"]
    candidate_metrics = routed_metrics(routed)
    paired = comparison(config, routed, generation["anchor"])
    scores_finite = (
        all_scores_finite(*scores.values())
        and all(
            summary["all_scores_finite"]
            for summary in score_summaries.values()
        )
    )
    gates = admission_gates(
        paired,
        candidate_metrics,
        anchor_metrics,
        scores_finite=scores_finite,
    )
    anchor_by_id = {
        row["case_id"]: row for row in generation["anchor"]
    }
    routed_by_id = {row["case_id"]: row for row in routed}
    wins = sorted(
        case_id
        for case_id in expected_case_ids
        if routed_by_id[case_id]["correct"]
        and not anchor_by_id[case_id]["correct"]
    )
    losses = sorted(
        case_id
        for case_id in expected_case_ids
        if anchor_by_id[case_id]["correct"]
        and not routed_by_id[case_id]["correct"]
    )
    admitted = all(gates.values())
    return {
        "schema_version": "nano_train_confidence_route_public_v1",
        "experiment_id": config.experiment_id,
        "identity": {
            "preregister_sha256": sha256_file(PREREG),
            "preregister_revision": prereg["identity"]["code_revision"],
            "result_revision": git_revision(),
            "config_sha256": sha256_file(CONFIG),
            "case_contract_sha256": prereg["case_contract"][
                "case_contract_sha256"
            ],
            "generation": {
                arm: {
                    "summary_sha256": sha256_file(
                        generation_summary_paths[arm]
                    ),
                    "raw_sha256": sha256_file(generation_paths[arm]),
                    "adapter_sha256": generation_summaries[arm]["identity"][
                        "adapter_sha256"
                    ],
                }
                for arm in ("anchor", "consistency")
            },
            "scores": {
                arm: {
                    "summary_sha256": sha256_file(score_summary_paths[arm]),
                    "raw_sha256": sha256_file(score_paths[arm]),
                    "adapter_sha256": score_summaries[arm][
                        "adapter_sha256"
                    ],
                }
                for arm in ("anchor", "consistency")
            },
            "routed_raw_sha256": sha256_file(routed_path),
        },
        "selector": {
            **route_summary,
            "uses_case_correctness": False,
            "uses_model_output_for_feedback": False,
            "scores_include_eos": True,
            "all_scores_finite": scores_finite,
        },
        "arms": {
            "anchor": anchor_metrics,
            "consistency": generation_summaries["consistency"]["metrics"],
            "routed": candidate_metrics,
        },
        "comparison": paired,
        "discordant_cases": {
            "candidate_only_win_case_ids": wins,
            "anchor_only_loss_case_ids": losses,
        },
        "contamination_audit": audit,
        "decision": {
            "gates": gates,
            "local_route_admitted": admitted,
            "canary_allowed": admitted,
            "benchmark_allowed": False,
            "independent_holdout_allowed": False,
            "route_tuning_allowed": False,
            "next_action": (
                "Consume only these exact selector, score, generation, and "
                "adapter identities in the pre-registered 211-case canary."
                if admitted
                else "Reject this selector and preserve the result; do not "
                "tune on these cases."
            ),
        },
        "claim_boundary": (
            "This fresh synthetic result can admit only the frozen selector "
            "to the pre-registered 211-case canary. It is not benchmark, "
            "holdout, or final 4B/9B superiority evidence."
        ),
    }


def render_markdown(report: dict) -> str:
    paired = report["comparison"]
    decision = report["decision"]
    selector = report["selector"]
    return f"""# Qwen3.5 Normalized-Confidence Route v1 Result

## 结论

- local route admitted：
  `{str(decision['local_route_admitted']).lower()}`；
- 211-case canary allowed：
  `{str(decision['canary_allowed']).lower()}`；
- complete benchmark allowed：`false`；
- route tuning on observed cases：`false`。

## Arms

| Arm | Correct | Accuracy | Parse failures |
| --- | ---: | ---: | ---: |
{chr(10).join(
    f"| {arm} | {metrics['correct']}/256 | {metrics['accuracy']:.4f} | "
    f"{metrics['parse_failures']} |"
    for arm, metrics in report['arms'].items()
)}

## Routed vs anchor

- delta `{paired['delta']:+.4f}`；
- paired bootstrap 95% CI
  `[{paired['paired_bootstrap_95_ci'][0]:+.4f},
  {paired['paired_bootstrap_95_ci'][1]:+.4f}]`；
- exact McNemar `p={paired['mcnemar_exact_p']}`；
- wins/losses：
  `{paired['paired_counts']['candidate_only']}/
  {paired['paired_counts']['baseline_only']}`。

## Frozen selector

- anchor routes：`{selector['anchor_routes']}`；
- consistency routes：`{selector['consistency_routes']}`；
- tie fallback：`{selector['tie_policy']}`；
- all cross-model candidate scores finite：
  `{str(selector['all_scores_finite']).lower()}`；
- expected answer / correctness feedback：`false`。

```json
{json.dumps(decision['gates'], indent=2, sort_keys=True)}
```

## Discordant cases

- wins：`{report['discordant_cases']['candidate_only_win_case_ids']}`；
- losses：`{report['discordant_cases']['anchor_only_loss_case_ids']}`。

只公开 case IDs、SHA 和聚合指标，不公开 prompt、target、output 或逐条 score。

## Evidence

- prereg SHA：`{report['identity']['preregister_sha256']}`；
- config SHA：`{report['identity']['config_sha256']}`；
- anchor generation raw：
  `{report['identity']['generation']['anchor']['raw_sha256']}`；
- consistency generation raw：
  `{report['identity']['generation']['consistency']['raw_sha256']}`；
- anchor score raw：
  `{report['identity']['scores']['anchor']['raw_sha256']}`；
- consistency score raw：
  `{report['identity']['scores']['consistency']['raw_sha256']}`；
- routed raw：`{report['identity']['routed_raw_sha256']}`。

若 gate 通过，下一步只能用完全相同的 selector、candidate、score、adapter identities
进入已预注册 211-case canary；不能修改 threshold、tie、prompt、parser、budget
或 adapter。无论通过与否，完整 benchmark 和 independent holdout 都保持关闭。
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
