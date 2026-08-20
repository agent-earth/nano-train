#!/usr/bin/env python3

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from nano_train.confidence_route import (
    build_cases,
    contamination_audit,
    load_config,
    public_contract,
    verify_identity,
)
from nano_train.sft import sha256_file


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/confidence_route/qwen35_confidence_route_v1.json"
JSON_OUTPUT = (
    ROOT / "docs/experiments/qwen35_confidence_route_v1.preregister.json"
)
MARKDOWN_OUTPUT = ROOT / "docs/experiments/qwen35_confidence_route_v1.md"


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
    identity = verify_identity(config)
    family_counts = {
        family: sum(row["family"] == family for row in cases)
        for family in sorted({row["family"] for row in cases})
    }
    if set(family_counts.values()) != {config.cases_per_family}:
        raise ValueError("confidence route family counts differ")
    return {
        "schema_version": "nano_train_confidence_route_preregister_v1",
        "experiment_id": config.experiment_id,
        "identity": {
            "code_revision": git_revision(),
            "config_sha256": sha256_file(CONFIG),
            "model_identity": identity,
        },
        "case_contract": contract,
        "family_counts": family_counts,
        "contamination_audit": audit,
        "stages": [
            "generate_anchor_candidate",
            "generate_consistency_candidate",
            "score_both_candidates_with_anchor",
            "score_both_candidates_with_consistency",
            "combine_by_frozen_relative_logprob_rule",
        ],
        "selector": {
            "name": config.selector,
            "anchor_relative": (
                "logp_consistency_model(anchor_candidate) - "
                "logp_anchor_model(anchor_candidate)"
            ),
            "consistency_relative": (
                "logp_consistency_model(consistency_candidate) - "
                "logp_anchor_model(consistency_candidate)"
            ),
            "select_consistency_when": (
                "consistency_relative > anchor_relative"
            ),
            "tie_policy": config.tie_policy,
            "score": "mean next-token log probability including EOS",
            "uses_expected_answer": False,
            "uses_case_correctness": False,
            "uses_observed_quality_outputs": False,
            "threshold_search": False,
        },
        "generation": {
            "arm_order": ["anchor", "consistency"],
            "generation_batch_size": config.generation_batch_size,
            "scoring_batch_size": config.scoring_batch_size,
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
            "all_scores_finite": True,
            "benchmark_allowed_after_pass": False,
            "canary_allowed_after_pass": True,
        },
        "decision_policy": {
            "forbidden_after_observation": [
                "case_change",
                "selector_formula_change",
                "threshold_addition",
                "tie_policy_change",
                "score_normalization_change",
                "prompt_change",
                "parser_change",
                "budget_change",
                "adapter_change",
                "benchmark_access",
                "canary_access_before_pass",
                "holdout_access",
            ],
            "passed": (
                "Publish local route evidence and consume exact identities as "
                "the local dependency for the pre-registered 211-case canary."
            ),
            "failed": "Preserve evidence and do not tune on these cases.",
        },
        "execution_boundary": {
            "generation_started": False,
            "scoring_started": False,
            "evaluation_started": False,
            "benchmark_accessed": False,
            "canary_accessed": False,
            "this_commit_only_preregisters": True,
        },
        "claim_boundary": (
            "This fresh local selector evaluation can admit one frozen canary "
            "route only. It is not benchmark, holdout, or final superiority "
            "evidence."
        ),
    }


def render_markdown(receipt: dict) -> str:
    selector = receipt["selector"]
    return f"""# Qwen3.5 Normalized-Confidence Route v1

## 目的

Family route 在 fresh cases 上不稳定。这里改用 target-blind 的双模型相对
likelihood selector；规则在生成前冻结，不使用 expected answer 或 correctness。

## Fresh surface

- 4 families × 64 = 256 cases；
- case contract SHA：
  `{receipt['case_contract']['case_contract_sha256']}`；
- 与所有已观察 synthetic quality prompts overlap=0；
- 与完整 GSM8K/MMLU/GPQA 题面 overlap=0；
- raw candidates/scores 只写 ignored `artifacts/`。

## 五阶段

1. anchor 生成完整 256 candidates；
2. consistency 生成完整 256 candidates；
3. anchor 对两份 candidates 计算 mean token logprob；
4. consistency 对两份 candidates 计算 mean token logprob；
5. 按 frozen relative-logprob rule 组合。

## Selector

- anchor relative：`{selector['anchor_relative']}`；
- consistency relative：`{selector['consistency_relative']}`；
- consistency 仅在 `consistency_relative > anchor_relative` 时生效；
- 平局回退 anchor；
- score 包含 EOS，按 candidate token 数取 mean；
- 无 threshold、confidence search、expected answer、correctness 或 model-output
  feedback。

## Gate

- routed accuracy > anchor；
- CI lower > 0；
- McNemar p < 0.05；
- 至少 6 wins / 0 losses；
- family/parse non-regression；
- 所有 logprob finite。

通过只允许进入已预注册 211-case canary，不直接开放完整 benchmark。

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
                "selector": receipt["selector"],
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
