#!/usr/bin/env python3

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from nano_train.consistency_route import (
    build_cases,
    contamination_audit,
    load_config,
    public_contract,
    verify_identity,
)
from nano_train.sft import sha256_file, sha256_tree


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/consistency_route/qwen35_consistency_route_v1.json"
JSON_OUTPUT = (
    ROOT / "docs/experiments/qwen35_consistency_route_v1.preregister.json"
)
MARKDOWN_OUTPUT = ROOT / "docs/experiments/qwen35_consistency_route_v1.md"


def git_revision() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()


def build_receipt() -> dict:
    config = load_config(CONFIG)
    cases = build_cases(config)
    contract = public_contract(cases)
    audit = contamination_audit(config, cases)
    model_identity = verify_identity(config)
    if (
        sha256_tree(Path(config.anchor_adapter_path))
        != config.anchor_adapter_sha256
        or sha256_tree(Path(config.consistency_adapter_path))
        != config.consistency_adapter_sha256
    ):
        raise ValueError("consistency route adapter identity differs")
    family_counts = {
        family: sum(row["family"] == family for row in cases)
        for family in sorted({row["family"] for row in cases})
    }
    if set(family_counts.values()) != {config.cases_per_family}:
        raise ValueError("consistency route family counts differ")
    return {
        "schema_version": "nano_train_consistency_route_preregister_v1",
        "experiment_id": config.experiment_id,
        "identity": {
            "code_revision": git_revision(),
            "config_sha256": sha256_file(CONFIG),
            "model_config_sha256": config.model_config_sha256,
            "model_index_sha256": config.model_index_sha256,
            "anchor_adapter_sha256": config.anchor_adapter_sha256,
            "consistency_adapter_sha256": config.consistency_adapter_sha256,
            "model_identity": model_identity,
        },
        "case_contract": contract,
        "family_counts": family_counts,
        "contamination_audit": audit,
        "route": {
            "routed_family": config.routed_family,
            "routed_family_arm": "consistency",
            "all_other_families_arm": "anchor",
            "uses_model_confidence": False,
            "uses_model_output": False,
            "uses_expected_answer": False,
            "rule_frozen_before_generation": True,
        },
        "generation": {
            "arm_order": ["anchor", "consistency"],
            "batch_size": config.batch_size,
            "max_new_tokens": config.max_new_tokens,
            "temperature": config.temperature,
            "chat_template_enable_thinking": False,
        },
        "acceptance": {
            "routed_accuracy_gt_anchor": True,
            "paired_bootstrap_ci_lower_gt_zero": True,
            "exact_mcnemar_p_lt": 0.05,
            "minimum_candidate_only_wins": 6,
            "maximum_anchor_only_losses": 0,
            "every_family_non_regression": True,
            "parse_failures_non_regression": True,
            "benchmark_allowed_after_pass": False,
            "canary_allowed_after_pass": True,
        },
        "decision_policy": {
            "forbidden_after_observation": [
                "case_change",
                "route_family_change",
                "route_threshold_addition",
                "prompt_change",
                "parser_change",
                "budget_change",
                "adapter_change",
                "decoding_change",
                "benchmark_access",
                "canary_access_before_pass",
                "holdout_access",
            ],
            "passed": (
                "Publish local route evidence and consume it as the local "
                "admission dependency for the pre-registered 211-case canary."
            ),
            "failed": (
                "Preserve evidence and do not tune routing on this surface."
            ),
        },
        "execution_boundary": {
            "model_generation_started": False,
            "evaluation_started": False,
            "benchmark_accessed": False,
            "canary_accessed": False,
            "this_commit_only_preregisters": True,
        },
        "claim_boundary": (
            "This fresh local evaluation tests one target-blind family route. "
            "Passing can admit a frozen canary arm, not establish benchmark or "
            "final 4B superiority."
        ),
    }


def render_markdown(receipt: dict) -> str:
    return f"""# Qwen3.5 Conservative Consistency Route v1

## 假设

Consistency 在 fresh dev 上有 15 wins / 2 losses。两个 losses 都来自
repeated-operand，而 10/15 wins 来自 exact-division。因此预注册一个最简单、
target-blind 的结构 route：

- exact-division → consistency adapter；
- 其他三个 family → anchor adapter。

规则只看程序化 family，不看 model confidence、output、expected answer 或已观察
case 结果。

## Fresh evaluation

- 4 families × 64 = 256 cases；
- case contract SHA：
  `{receipt['case_contract']['case_contract_sha256']}`；
- 与所有已观察 synthetic quality prompts overlap=0；
- 与完整 GSM8K/MMLU/GPQA 题面 overlap=0；
- benchmark/canary/holdout rows 和 outputs=0；
- raw cases/output 仅写 ignored `artifacts/`。

## Arms

1. anchor adapter；
2. consistency adapter；
3. routed candidate 在 report 阶段按 frozen family 规则组合。

两条原始 arm 都必须先完整运行；不能根据 anchor 结果改变 route。

## Gate

- routed accuracy > anchor；
- paired bootstrap CI lower > 0；
- exact McNemar p < 0.05；
- 至少 6 wins、0 losses；
- every-family 和 parse non-regression。

通过只允许消费为已预注册 211-case canary 的本地 admission dependency；不直接
开放完整 benchmark 或 independent holdout。

## 执行边界

```json
{json.dumps(receipt['execution_boundary'], indent=2, sort_keys=True)}
```
"""


def main() -> None:
    receipt = build_receipt()
    JSON_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    JSON_OUTPUT.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    MARKDOWN_OUTPUT.write_text(render_markdown(receipt), encoding="utf-8")
    print(
        json.dumps(
            {
                "experiment_id": receipt["experiment_id"],
                "config_sha256": receipt["identity"]["config_sha256"],
                "case_contract": receipt["case_contract"],
                "route": receipt["route"],
                "execution_boundary": receipt["execution_boundary"],
                "json_output": str(JSON_OUTPUT),
                "markdown_output": str(MARKDOWN_OUTPUT),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
