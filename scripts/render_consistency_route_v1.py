#!/usr/bin/env python3

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from nano_train.consistency_route import (
    FAMILIES,
    compare_routed,
    load_config,
    load_rows,
    routed_rows,
)
from nano_train.sft import sha256_file


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/consistency_route/qwen35_consistency_route_v1.json"
PREREG = (
    ROOT / "docs/experiments/qwen35_consistency_route_v1.preregister.json"
)
ARTIFACTS = ROOT / "artifacts/qwen35-consistency-route-v1"
PUBLIC_JSON = ROOT / "docs/results/qwen35_consistency_route_v1.public.json"
MARKDOWN = ROOT / "docs/results/qwen35_consistency_route_v1.md"


def git_revision() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()


def admission_gates(
    comparison: dict,
    routed_metrics: dict,
    anchor_metrics: dict,
) -> dict[str, bool]:
    return {
        "routed_accuracy_gt_anchor": (
            comparison["candidate_accuracy"] > comparison["baseline_accuracy"]
        ),
        "paired_bootstrap_ci_lower_gt_zero": (
            comparison["paired_bootstrap_95_ci"][0] > 0
        ),
        "exact_mcnemar_p_lt_005": comparison["mcnemar_exact_p"] < 0.05,
        "minimum_candidate_only_wins": (
            comparison["paired_counts"]["candidate_only"] >= 6
        ),
        "maximum_anchor_only_losses": (
            comparison["paired_counts"]["baseline_only"] == 0
        ),
        "every_family_non_regression": all(
            routed_metrics["by_family"][family]["correct"]
            >= anchor_metrics["by_family"][family]["correct"]
            for family in FAMILIES
        ),
        "parse_failures_non_regression": (
            routed_metrics["parse_failures"]
            <= anchor_metrics["parse_failures"]
        ),
    }


def _metrics(rows: list[dict]) -> dict:
    by_family = {}
    for family in FAMILIES:
        selected = [row for row in rows if row["family"] == family]
        by_family[family] = {
            "cases": len(selected),
            "correct": sum(row["correct"] for row in selected),
            "parse_failures": sum(
                row["parse_failure"] for row in selected
            ),
        }
    correct = sum(row["correct"] for row in rows)
    return {
        "cases": len(rows),
        "correct": correct,
        "accuracy": correct / len(rows),
        "parse_failures": sum(row["parse_failure"] for row in rows),
        "by_family": by_family,
    }


def build_report() -> dict:
    config = load_config(CONFIG)
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    anchor_summary_path = ARTIFACTS / "anchor/summary.json"
    consistency_summary_path = ARTIFACTS / "consistency/summary.json"
    anchor_cases_path = ARTIFACTS / "anchor/cases.jsonl"
    consistency_cases_path = ARTIFACTS / "consistency/cases.jsonl"
    anchor_summary = json.loads(
        anchor_summary_path.read_text(encoding="utf-8")
    )
    consistency_summary = json.loads(
        consistency_summary_path.read_text(encoding="utf-8")
    )
    anchor = load_rows(anchor_cases_path)
    consistency = load_rows(consistency_cases_path)
    if (
        prereg["schema_version"]
        != "nano_train_consistency_route_preregister_v1"
        or prereg["execution_boundary"]["evaluation_started"] is not False
        or anchor_summary["schema_version"]
        != "nano_train_consistency_route_result_v1"
        or consistency_summary["schema_version"]
        != "nano_train_consistency_route_result_v1"
        or anchor_summary["case_contract"]["case_contract_sha256"]
        != prereg["case_contract"]["case_contract_sha256"]
        or consistency_summary["case_contract"]["case_contract_sha256"]
        != prereg["case_contract"]["case_contract_sha256"]
    ):
        raise ValueError("consistency route result identity differs")
    routed = routed_rows(config, anchor, consistency)
    routed_path = ARTIFACTS / "routed/cases.jsonl"
    routed_path.parent.mkdir(parents=True, exist_ok=True)
    routed_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in routed),
        encoding="utf-8",
    )
    comparison = compare_routed(config, routed, anchor)
    routed_metrics = _metrics(routed)
    gates = admission_gates(
        comparison,
        routed_metrics,
        anchor_summary["metrics"],
    )
    anchor_by_id = {row["case_id"]: row for row in anchor}
    routed_by_id = {row["case_id"]: row for row in routed}
    wins = sorted(
        case_id
        for case_id in anchor_by_id
        if routed_by_id[case_id]["correct"]
        and not anchor_by_id[case_id]["correct"]
    )
    losses = sorted(
        case_id
        for case_id in anchor_by_id
        if anchor_by_id[case_id]["correct"]
        and not routed_by_id[case_id]["correct"]
    )
    route_counts = {
        route: sum(row["route"] == route for row in routed)
        for route in ("anchor", "consistency")
    }
    return {
        "schema_version": "nano_train_consistency_route_public_v1",
        "experiment_id": config.experiment_id,
        "identity": {
            "preregister_sha256": sha256_file(PREREG),
            "preregister_revision": prereg["identity"]["code_revision"],
            "result_revision": git_revision(),
            "config_sha256": sha256_file(CONFIG),
            "case_contract_sha256": prereg["case_contract"][
                "case_contract_sha256"
            ],
            "anchor_summary_sha256": sha256_file(anchor_summary_path),
            "consistency_summary_sha256": sha256_file(
                consistency_summary_path
            ),
            "anchor_raw_sha256": sha256_file(anchor_cases_path),
            "consistency_raw_sha256": sha256_file(consistency_cases_path),
            "routed_raw_sha256": sha256_file(routed_path),
            "anchor_adapter_sha256": config.anchor_adapter_sha256,
            "consistency_adapter_sha256": (
                config.consistency_adapter_sha256
            ),
        },
        "route": {
            "routed_family": config.routed_family,
            "route_counts": route_counts,
            "uses_model_confidence": False,
            "uses_model_output": False,
            "uses_expected_answer": False,
        },
        "arms": {
            "anchor": anchor_summary["metrics"],
            "consistency": consistency_summary["metrics"],
            "routed": routed_metrics,
        },
        "comparison": comparison,
        "discordant_cases": {
            "candidate_only_win_case_ids": wins,
            "anchor_only_loss_case_ids": losses,
        },
        "contamination_audit": anchor_summary["contamination_audit"],
        "decision": {
            "gates": gates,
            "local_route_admitted": all(gates.values()),
            "canary_allowed": all(gates.values()),
            "benchmark_allowed": False,
            "independent_holdout_allowed": False,
            "route_tuning_allowed": False,
            "next_action": (
                "If admitted, consume this exact route and adapter identities "
                "as the local dependency for the already pre-registered "
                "211-case treatment canary; do not change route or adapters."
                if all(gates.values())
                else "Reject this route and preserve the result; do not tune "
                "on these cases."
            ),
        },
        "claim_boundary": (
            "This result establishes only fresh synthetic local route "
            "admission. It is not benchmark, canary, holdout, or final 4B/9B "
            "superiority evidence."
        ),
    }


def render_markdown(report: dict) -> str:
    comparison = report["comparison"]
    decision = report["decision"]
    return f"""# Qwen3.5 Conservative Consistency Route v1 Result

## 结论

- local route admitted：
  `{str(decision['local_route_admitted']).lower()}`；
- canary allowed：`{str(decision['canary_allowed']).lower()}`；
- benchmark allowed：false。

## Arms

| Arm | Correct | Accuracy | Parse failures |
| --- | ---: | ---: | ---: |
{chr(10).join(
    f"| {arm} | {metrics['correct']}/256 | {metrics['accuracy']:.4f} | "
    f"{metrics['parse_failures']} |"
    for arm, metrics in report['arms'].items()
)}

## Routed vs anchor

- delta `{comparison['delta']:+.4f}`；
- 95% CI
  `[{comparison['paired_bootstrap_95_ci'][0]:+.4f},
  {comparison['paired_bootstrap_95_ci'][1]:+.4f}]`；
- exact McNemar `p={comparison['mcnemar_exact_p']}`；
- wins/losses：
  `{comparison['paired_counts']['candidate_only']}/
  {comparison['paired_counts']['baseline_only']}`。

## Route

- exact-division：consistency；
- 其他 192 cases：anchor fallback；
- confidence/output/expected-answer routing：false。

```json
{json.dumps(decision['gates'], indent=2, sort_keys=True)}
```

## Discordant cases

- wins：`{report['discordant_cases']['candidate_only_win_case_ids']}`；
- losses：`{report['discordant_cases']['anchor_only_loss_case_ids']}`。

只公开 case IDs，不公开 prompt、target 或 output。

## Evidence

- prereg SHA：`{report['identity']['preregister_sha256']}`；
- anchor summary/raw：
  `{report['identity']['anchor_summary_sha256']}` /
  `{report['identity']['anchor_raw_sha256']}`；
- consistency summary/raw：
  `{report['identity']['consistency_summary_sha256']}` /
  `{report['identity']['consistency_raw_sha256']}`；
- routed raw：`{report['identity']['routed_raw_sha256']}`。

若 gate 通过，下一步只能消费完全相同的 route 和 adapter identities 进入已预注册
211-case canary；不能修改 route、threshold、prompt、parser 或 adapter。
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
