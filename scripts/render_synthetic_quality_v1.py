#!/usr/bin/env python3

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from nano_train.sft import sha256_file
from nano_train.synthetic_quality import (
    FAMILIES,
    candidate_admission_gates,
    load_config,
    load_rows,
    paired_comparison,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/evaluation/qwen35_synthetic_quality_v1.json"
PREREG = (
    ROOT / "docs/experiments/qwen35_synthetic_quality_v1.preregister.json"
)
ARTIFACTS = ROOT / "artifacts/qwen35-synthetic-quality-v1"
PUBLIC_JSON = ROOT / "docs/results/qwen35_synthetic_quality_v1.public.json"
MARKDOWN = ROOT / "docs/results/qwen35_synthetic_quality_v1.md"
ARM_ORDER = ("base4", "rl4", "opd4", "base9")


def git_revision() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()


def build_report() -> dict:
    config = load_config(CONFIG)
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    if (
        prereg["schema_version"]
        != "nano_train_synthetic_quality_preregister_v1"
        or prereg["execution_boundary"]["evaluation_started"] is not False
    ):
        raise ValueError("synthetic quality preregistration differs")
    summaries = {}
    rows = {}
    for arm in ARM_ORDER:
        summary_path = ARTIFACTS / arm / "summary.json"
        cases_path = ARTIFACTS / arm / "cases.jsonl"
        summaries[arm] = json.loads(summary_path.read_text(encoding="utf-8"))
        rows[arm] = load_rows(cases_path)
        summary = summaries[arm]
        if (
            summary["schema_version"]
            != "nano_train_synthetic_quality_result_v1"
            or summary["arm_id"] != arm
            or summary["case_contract"]["case_contract_sha256"]
            != prereg["case_contract"]["case_contract_sha256"]
            or summary["raw"]["cases_sha256"] != sha256_file(cases_path)
            or summary["contamination_audit"]["passed"] is not True
        ):
            raise ValueError(f"synthetic quality arm validation failed: {arm}")
    comparisons = {}
    for candidate in ("rl4", "opd4", "base9"):
        comparisons[f"{candidate}_vs_base4"] = paired_comparison(
            rows[candidate],
            rows["base4"],
            bootstrap_samples=config.bootstrap_samples,
            bootstrap_seed=config.bootstrap_seed,
        )
    arms = {}
    for arm in ARM_ORDER:
        summary = summaries[arm]
        arms[arm] = {
            "metrics": summary["metrics"],
            "identity": {
                "summary_sha256": sha256_file(
                    ARTIFACTS / arm / "summary.json"
                ),
                "raw_cases_sha256": summary["raw"]["cases_sha256"],
                "model_config_sha256": summary["identity"][
                    "model_config_sha256"
                ],
                "model_index_sha256": summary["identity"][
                    "model_index_sha256"
                ],
                "adapter_sha256": (
                    summary["identity"]["adapter"]["sha256"]
                    if summary["identity"]["adapter"]
                    else None
                ),
            },
            "hardware": summary["hardware"],
            "wall_seconds": summary["wall_seconds"],
        }
    decisions = {}
    baseline = arms["base4"]["metrics"]
    for candidate in ("rl4", "opd4"):
        comparison = comparisons[f"{candidate}_vs_base4"]
        metrics = arms[candidate]["metrics"]
        gates = candidate_admission_gates(
            comparison,
            metrics,
            baseline,
        )
        decisions[candidate] = {
            "gates": gates,
            "admitted_synthetic_quality_candidate": all(gates.values()),
        }
    return {
        "schema_version": "nano_train_synthetic_quality_public_v1",
        "experiment_id": config.experiment_id,
        "identity": {
            "preregister_sha256": sha256_file(PREREG),
            "preregister_revision": prereg["identity"]["code_revision"],
            "result_revision": git_revision(),
            "config_sha256": sha256_file(CONFIG),
            "case_contract_sha256": prereg["case_contract"][
                "case_contract_sha256"
            ],
            "case_ids_sha256": prereg["case_contract"]["case_ids_sha256"],
        },
        "arms": arms,
        "comparisons": comparisons,
        "decision": {
            "candidates": decisions,
            "any_two_step_adapter_admitted": any(
                row["admitted_synthetic_quality_candidate"]
                for row in decisions.values()
            ),
            "benchmark_allowed": False,
            "canary_allowed": False,
            "holdout_allowed": False,
            "larger_training_allowed_from_this_result": False,
            "model_quality_scope": "fresh_synthetic_arithmetic_only",
            "next_action": (
                "Reject two-step RL and OPD as quality candidates. Preserve "
                "their implementation admission, wait for peer consistency, "
                "and pre-register a scaled quality-training intervention on "
                "new synthetic train/dev data rather than tuning on these "
                "96 evaluation cases."
            ),
        },
        "claim_boundary": (
            "The four-arm comparison is valid only for this fresh synthetic "
            "arithmetic suite. It is not benchmark, canary, holdout, or final "
            "4B/9B superiority evidence."
        ),
    }


def render_markdown(report: dict) -> str:
    arm_rows = []
    for arm in ARM_ORDER:
        metrics = report["arms"][arm]["metrics"]
        arm_rows.append(
            f"| `{arm}` | {metrics['correct']}/96 | "
            f"{metrics['accuracy']:.4f} | {metrics['parse_failures']} |"
        )
    comparison_rows = []
    for name, comparison in report["comparisons"].items():
        comparison_rows.append(
            f"| `{name}` | {comparison['delta']:+.4f} | "
            f"[{comparison['paired_bootstrap_95_ci'][0]:+.4f}, "
            f"{comparison['paired_bootstrap_95_ci'][1]:+.4f}] | "
            f"{comparison['mcnemar_exact_p']:.6g} | "
            f"{comparison['paired_counts']['candidate_only']}/"
            f"{comparison['paired_counts']['baseline_only']} |"
        )
    rl = report["decision"]["candidates"]["rl4"]
    opd = report["decision"]["candidates"]["opd4"]
    return f"""# Qwen3.5 Synthetic Quality Ablation v1 Result

## 结论

两步 RL 和两步 OPD 都**没有**成为质量候选：

- RL：`{str(rl['admitted_synthetic_quality_candidate']).lower()}`；
- OPD：`{str(opd['admitted_synthetic_quality_candidate']).lower()}`。

它们的实现准入仍然成立，但不能据此扩大训练或访问 benchmark。

## 四臂结果

| Arm | Correct | Accuracy | Parse failures |
| --- | ---: | ---: | ---: |
{chr(10).join(arm_rows)}

## Paired 比较

| Comparison | Delta | 95% bootstrap CI | McNemar p | Wins/Losses |
| --- | ---: | --- | ---: | ---: |
{chr(10).join(comparison_rows)}

## 具体观察

- base4 只有 3/96，三个正确样例都来自 exact-division family；
- rl4 同样是 3/96；raw output SHA 与 base4 不同，说明 adapter 改了输出，
  但没有改正确率；
- opd4 也是 3/96，且 96 条 raw output 与 base4 逐字相同；
- base9 提供相同 case、prompt、parser、budget 下的 reference。

因此 “probe logits changed” 只能证明 adapter 生效，不能证明质量提升。这次
fresh evaluation 正好区分了实现证据与质量证据。

## Candidate gates

```json
{json.dumps(report['decision']['candidates'], indent=2, sort_keys=True)}
```

## Evidence

- preregistration SHA：`{report['identity']['preregister_sha256']}`；
- case contract SHA：`{report['identity']['case_contract_sha256']}`；
{chr(10).join(
    f"- {arm} summary/raw SHA：`{report['arms'][arm]['identity']['summary_sha256']}` / "
    f"`{report['arms'][arm]['identity']['raw_cases_sha256']}`；"
    for arm in ARM_ORDER
)}

## 下一步

保留 RL/OPD 的可运行机制，但拒绝把两步 adapter 当成质量候选。不要在这
96 cases 上改 task、prompt、reward、teacher、steps、LR、LoRA、budget 或
parser。等待 peer consistency replication，同时只允许另行预注册使用新
synthetic train/dev 数据的 scaled quality intervention。

本结果不是 benchmark、canary、independent holdout 或最终 4B/9B superiority。
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
