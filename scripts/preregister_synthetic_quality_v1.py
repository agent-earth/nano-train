#!/usr/bin/env python3

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from nano_train.sft import sha256_file
from nano_train.synthetic_quality import (
    build_cases,
    case_contract,
    contamination_audit,
    load_config,
    verify_arm_identity,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/evaluation/qwen35_synthetic_quality_v1.json"
JSON_OUTPUT = (
    ROOT / "docs/experiments/qwen35_synthetic_quality_v1.preregister.json"
)
MARKDOWN_OUTPUT = (
    ROOT / "docs/experiments/qwen35_synthetic_quality_v1.md"
)


def git_revision() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()


def build_receipt() -> dict:
    config = load_config(CONFIG)
    cases = build_cases(config)
    contract = case_contract(cases)
    audit = contamination_audit(config, cases)
    identities = {
        arm["arm_id"]: verify_arm_identity(arm)
        for arm in config.model_arms
    }
    family_counts = {
        family: sum(case["family"] == family for case in cases)
        for family in sorted({case["family"] for case in cases})
    }
    if set(family_counts.values()) != {config.cases_per_family}:
        raise ValueError("synthetic quality family counts differ")
    return {
        "schema_version": "nano_train_synthetic_quality_preregister_v1",
        "experiment_id": config.experiment_id,
        "identity": {
            "code_revision": git_revision(),
            "config_sha256": sha256_file(CONFIG),
            "model_arms": identities,
        },
        "case_contract": contract,
        "family_counts": family_counts,
        "contamination_audit": audit,
        "generation": {
            "arm_order": [arm["arm_id"] for arm in config.model_arms],
            "batch_size": config.batch_size,
            "max_new_tokens": config.max_new_tokens,
            "temperature": config.temperature,
            "chat_template_enable_thinking": False,
            "raw_output_dir": config.output_dir,
        },
        "comparison": {
            "baseline_arm": "base4",
            "candidate_arms": ["rl4", "opd4"],
            "reference_arm": "base9",
            "bootstrap_samples": config.bootstrap_samples,
            "bootstrap_seed": config.bootstrap_seed,
            "admission": (
                "candidate delta > 0 AND bootstrap CI lower > 0 AND exact "
                "McNemar p < 0.05 AND every family correct >= base4 family "
                "correct AND parse failures <= base4 parse failures"
            ),
        },
        "decision_policy": {
            "each_arm_runs_even_if_prior_arm_fails": True,
            "forbidden_after_observation": [
                "case_change",
                "prompt_change",
                "parser_change",
                "budget_change",
                "batch_change",
                "adapter_change",
                "decoding_change",
                "benchmark_access",
                "canary_access",
                "holdout_access",
            ],
            "passed_candidate": (
                "Synthetic quality evidence only; separately pre-register any "
                "training or benchmark treatment."
            ),
            "failed_candidate": (
                "Preserve negative evidence; do not tune on these 96 cases."
            ),
        },
        "execution_boundary": {
            "model_generation_started": False,
            "evaluation_started": False,
            "benchmark_accessed": False,
            "this_commit_only_preregisters": True,
        },
        "claim_boundary": (
            "This fresh 96-case synthetic evaluation tests whether the two-step "
            "RL and OPD adapters change arithmetic quality. It is not a public "
            "benchmark, canary, independent holdout, or final model claim."
        ),
    }


def render_markdown(receipt: dict) -> str:
    return f"""# Qwen3.5 Synthetic Quality Ablation v1

## 目的

用从未生成过输出的 96 个 synthetic arithmetic cases，比较：

1. base Qwen3.5-4B；
2. 两步 RL adapter；
3. 两步 OPD adapter；
4. base Qwen3.5-9B。

这一步回答“adapter 是否真的改变正确率”，不是 benchmark 评测。

## 冻结 case

- 4 个 family，每类 24 个，共 96 个；
- case contract SHA：`{receipt['case_contract']['case_contract_sha256']}`；
- case IDs SHA：`{receipt['case_contract']['case_ids_sha256']}`；
- public receipt 只保存 case ID、family、prompt SHA 和 expected SHA；
- raw expression、expected 和模型 output 只写 ignored `artifacts/`。

## 污染边界

- 与完整 GSM8K 1,319、MMLU 14,042、GPQA 198 题面做 normalized exact-hash；
- overlap 为 0；
- 不读取 benchmark label、output、canary 或 independent holdout；
- 所有 96 cases `training_eligible=false`。

## 生成合同

- arm order：`base4 → rl4 → opd4 → base9`；
- greedy decoding；
- thinking disabled；
- batch size 8；
- max new tokens 32；
- strict parser：整段输出必须恰好匹配 `FINAL: <integer>`。

## Candidate gate

RL/OPD 相对 base4 必须同时满足：

- overall delta > 0；
- paired bootstrap 95% CI 下界 > 0；
- exact McNemar `p < 0.05`；
- 每个 family correct 不低于 base4；
- parse failures 不多于 base4。

每个 arm 都必须运行，不能根据前一个结果跳过。观察后禁止更改 case、prompt、
parser、budget、batch、adapter 或 decoding。

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
