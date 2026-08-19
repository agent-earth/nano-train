#!/usr/bin/env python3

from __future__ import annotations

import json
import math
import random
import statistics
from pathlib import Path

from safetensors import safe_open

from nano_train.data import execution_target_output_valid
from nano_train.paired_consistency import build_selection_contract, load_config
from nano_train.sft import sha256_file


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts/execution-target-paired-consistency-v1"
CONFIG = (
    ROOT
    / "configs/paired_consistency/execution_target_consistency_v1.json"
)
PREREGISTER = ROOT / (
    "docs/experiments/execution_target_paired_consistency_v1.preregister.json"
)
RESCORE = ROOT / (
    "docs/results/execution_target_paired_consistency_v1.rescore.public.json"
)
PUBLIC_JSON = ROOT / (
    "docs/results/execution_target_paired_consistency_v1.public.json"
)
REPORT = ROOT / "docs/results/execution_target_paired_consistency_v1.md"


def _comparison(
    records: list[tuple[str, bool, bool]],
    *,
    seed: str,
) -> dict:
    values = [int(post) - int(baseline) for _, baseline, post in records]
    count = len(values)
    randomizer = random.Random(seed)
    samples = 10_000
    estimates = sorted(
        sum(values[randomizer.randrange(count)] for _ in range(count)) / count
        for _ in range(samples)
    )
    wins = sum(not baseline and post for _, baseline, post in records)
    losses = sum(baseline and not post for _, baseline, post in records)
    discordant = wins + losses
    tail = min(wins, losses)
    p_value = (
        min(
            1.0,
            2.0
            * sum(math.comb(discordant, index) for index in range(tail + 1))
            / (2**discordant),
        )
        if discordant
        else 1.0
    )
    return {
        "samples": count,
        "delta": sum(values) / count,
        "wins": wins,
        "losses": losses,
        "paired_bootstrap_95_ci": [
            estimates[int(samples * 0.025)],
            estimates[int(samples * 0.975)],
        ],
        "mcnemar_exact_p": p_value,
        "bootstrap_samples": samples,
        "bootstrap_seed": seed,
    }


def main() -> None:
    config = load_config(CONFIG)
    selection = build_selection_contract(config)
    preregister = json.loads(PREREGISTER.read_text())
    metrics = json.loads((ARTIFACTS / "metrics.json").read_text())
    reload = json.loads((ARTIFACTS / "reload_validation.json").read_text())
    rescore = json.loads(RESCORE.read_text())
    generations = json.loads((ARTIFACTS / "generations.json").read_text())
    baseline_rows = {
        row["sample_id"]: row for row in generations["baseline"]
    }
    post_rows = {
        row["sample_id"]: row for row in generations["post_sft"]
    }

    records = []
    for sample_id in selection["heldout_sample_ids"]:
        source = selection["raw_by_id"][sample_id]
        baseline_valid = execution_target_output_valid(
            source["task_family"],
            source["view"],
            source.get("task_spec"),
            source.get("verifier"),
            baseline_rows[sample_id]["output"],
        )
        post_valid = execution_target_output_valid(
            source["task_family"],
            source["view"],
            source.get("task_spec"),
            source.get("verifier"),
            post_rows[sample_id]["output"],
        )
        records.append((source["view"], baseline_valid, post_valid))
    comparisons = {
        "aggregate": _comparison(
            records,
            seed="paired-consistency-v1-all",
        ),
        "final": _comparison(
            [record for record in records if record[0] == "final"],
            seed="paired-consistency-v1-final",
        ),
        "json": _comparison(
            [
                record
                for record in records
                if record[0] == "json_preservation"
            ],
            seed="paired-consistency-v1-json",
        ),
    }
    pair_records = []
    pair_ids = sorted(
        {
            selection["raw_by_id"][sample_id]["pair_id"]
            for sample_id in selection["heldout_sample_ids"]
            if selection["raw_by_id"][sample_id]["pair_id"]
        }
    )
    fixed_pairs = []
    for pair_id in pair_ids:
        pair_sample_ids = [
            sample_id
            for sample_id in selection["heldout_sample_ids"]
            if selection["raw_by_id"][sample_id].get("pair_id") == pair_id
        ]

        def pair_valid(rows: dict[str, dict]) -> bool:
            return all(
                execution_target_output_valid(
                    selection["raw_by_id"][sample_id]["task_family"],
                    selection["raw_by_id"][sample_id]["view"],
                    selection["raw_by_id"][sample_id].get("task_spec"),
                    selection["raw_by_id"][sample_id].get("verifier"),
                    rows[sample_id]["output"],
                )
                for sample_id in pair_sample_ids
            )

        baseline_valid = pair_valid(baseline_rows)
        post_valid = pair_valid(post_rows)
        pair_records.append((pair_id, baseline_valid, post_valid))
        if not baseline_valid and post_valid:
            final_id = next(
                sample_id
                for sample_id in pair_sample_ids
                if selection["raw_by_id"][sample_id]["view"] == "final"
            )
            process_id = next(
                sample_id
                for sample_id in pair_sample_ids
                if selection["raw_by_id"][sample_id]["view"] == "process"
            )
            source = selection["raw_by_id"][final_id]
            fixed_pairs.append(
                {
                    "pair_id": pair_id,
                    "expression": source["task_spec"]["expression"],
                    "expected": source["messages"][-1]["content"],
                    "baseline": baseline_rows[final_id]["output"],
                    "post_sft": post_rows[final_id]["output"],
                    "process_output": post_rows[process_id]["output"],
                }
            )
    comparisons["pair_both"] = _comparison(
        pair_records,
        seed="paired-consistency-v1-pair",
    )

    adapter_path = ARTIFACTS / "adapter/adapter_model.safetensors"
    with safe_open(adapter_path, framework="pt", device="cpu") as handle:
        tensors = list(handle.keys())
        nonfinite = sum(
            not handle.get_tensor(name).isfinite().all().item()
            for name in tensors
        )
    reload_matches = (
        metrics["post_validation"] == reload["validation"]
        and generations["post_sft"] == reload["generations"]
        and metrics["adapter_sha256"] == reload["adapter_sha256"]
        and metrics["selection"]["hashes"]["heldout_sample_id_sha256"]
        == reload["heldout_sample_id_sha256"]
    )
    stable = (
        not metrics["failure_receipt_exists"]
        and nonfinite == 0
        and reload_matches
        and all(
            math.isfinite(float(row["total_loss"]))
            for row in metrics["loss_curve"]
        )
    )
    pair_losses = [
        row for row in metrics["loss_curve"] if row["kind"] == "pair"
    ]
    loss_summary = {}
    for key in ("process_ce", "final_ce", "consistency_kl", "total_loss"):
        values = [float(row[key]) for row in pair_losses]
        loss_summary[key] = {
            "first_five_mean": statistics.mean(values[:5]),
            "last_five_mean": statistics.mean(values[-5:]),
            "minimum": min(values),
            "maximum": max(values),
        }

    baseline = rescore["arms"]["baseline"]
    post = rescore["arms"]["post_sft"]
    json_families = (
        "coding-and-validation",
        "planning-and-state",
        "skill-routing-and-reflection",
        "tool-use-and-recovery",
    )
    json_non_regression = all(
        post["by_family"][family]["verified"]
        >= baseline["by_family"][family]["verified"]
        for family in json_families
    )
    direction_gate_passed = (
        stable
        and rescore["verified_delta"] > 0
        and (
            post["by_view"]["final"]["verified"]
            > baseline["by_view"]["final"]["verified"]
        )
        and rescore["pair_both_verified_delta"] > 0
        and json_non_regression
    )
    statistically_supported = (
        comparisons["aggregate"]["paired_bootstrap_95_ci"][0] > 0
        and comparisons["aggregate"]["mcnemar_exact_p"] < 0.05
        and comparisons["final"]["paired_bootstrap_95_ci"][0] > 0
        and comparisons["final"]["mcnemar_exact_p"] < 0.05
    )
    report = {
        "schema_version": "nano_train_paired_consistency_public_v1",
        "experiment_id": metrics["experiment_id"],
        "identity": {
            "preregister_commit": "00aa03b",
            "validation_tools_commit": "2d60919",
            "config_sha256": sha256_file(CONFIG),
            "preregister_sha256": sha256_file(PREREGISTER),
            "dataset_file_sha256": config.dataset_file_sha256,
            "dataset_canonical_sha256": config.dataset_canonical_sha256,
            "heldout_sample_id_sha256": selection["hashes"][
                "heldout_sample_id_sha256"
            ],
            "adapter_sha256": metrics["adapter_sha256"],
        },
        "method": {
            "objective": preregister["objective"],
            "training": preregister["training"],
            "selection_hashes": selection["hashes"],
            "heldout_samples": len(selection["heldout_sample_ids"]),
            "pair_steps": len(selection["pair_schedule"]),
            "json_steps": len(selection["json_schedule"]),
        },
        "training": {
            "trainable_parameters": metrics["trainable_parameters"],
            "peak_allocated_gib": metrics["peak_allocated_gib"],
            "wall_seconds": metrics["wall_seconds"],
            "loss_summary": loss_summary,
            "loss_curve": metrics["loss_curve"],
        },
        "evaluation": {
            "baseline": baseline,
            "post_sft": post,
            "verified_delta": rescore["verified_delta"],
            "pair_both_verified_delta": rescore[
                "pair_both_verified_delta"
            ],
            "json_family_non_regression": json_non_regression,
            "changed_output_count": post["changed_output_count"],
            "comparisons": comparisons,
            "reload_matches": reload_matches,
            "reload_peak_allocated_gib": reload["peak_allocated_gib"],
        },
        "mechanism_evidence": {
            "fixed_pairs": fixed_pairs,
            "fixed_pair_count": len(fixed_pairs),
            "interpretation": (
                "The explicit consistency objective produces one genuine "
                "process-to-final transfer on a fresh pair. The effect is "
                "directionally correct but sparse and statistically "
                "inconclusive."
            ),
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
            "accepted_local_method_smoke": direction_gate_passed,
            "statistically_supported": statistically_supported,
            "larger_training_allowed": False,
            "benchmark_allowed": False,
            "independent_holdout_allowed": False,
            "rl_allowed": False,
            "next_action": (
                "保留 consistency v1，作为第一个方向正确的 execution-transfer "
                "方法。不要在这组 heldout 上调参。下一步使用完全相同的 "
                "objective 预注册更大的 fresh local replication，样本量必须"
                "足够检验显著性；在此之前不允许扩大训练或访问 benchmark。"
            ),
        },
        "claim_boundary": (
            "Consistency v1 通过了预注册的本地方向 gate，并修复了 1 个 fresh "
            "final-only execution pair；但 aggregate 和 final 的置信区间都"
            "包含 0，McNemar 检验也不显著。这是机制证据，不是稳定提升证据，"
            "不能据此扩大训练、访问 benchmark/holdout 或启动 RL。"
        ),
    }
    PUBLIC_JSON.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    REPORT.write_text(render_markdown(report), encoding="utf-8")


def render_markdown(report: dict) -> str:
    evaluation = report["evaluation"]
    baseline = evaluation["baseline"]
    post = evaluation["post_sft"]
    comparisons = evaluation["comparisons"]
    mechanism = report["mechanism_evidence"]
    decision = report["decision"]
    fixed = mechanism["fixed_pairs"][0]
    return f"""# Paired Consistency v1

## 做了什么

- Qwen3.5-4B，q/v-only LoRA，FP32，40 steps；
- 20 个 pair step 与 20 个 JSON step 交替；
- pair loss：
  `0.5*process_ce + 0.5*final_ce + 1.0*KL(detach(process)||final)`；
- fresh heldout：24 个完整 pair + 四个 JSON family 各 8 条，共 80 条；
- config、loss、train schedule、heldout 和 decision rule 已在 commit
  `{report['identity']['preregister_commit']}` 中先冻结；
- training 前明确记录 `training_started=false`。

## 稳定性

- 可训练参数：{report['training']['trainable_parameters']:,}；
- 训练显存峰值：{report['training']['peak_allocated_gib']:.2f} GiB；
- reload 显存峰值：{evaluation['reload_peak_allocated_gib']:.2f} GiB；
- 训练耗时：{report['training']['wall_seconds']:.2f} 秒；
- adapter tensors：{report['adapter_validation']['tensor_count']}；
- non-finite tensors：{report['adapter_validation']['nonfinite_tensors']}；
- 独立 reload metrics 和 80 条 generations 逐字一致：
  {str(evaluation['reload_matches']).lower()}。

## Corrected Fresh Heldout

- aggregate verified：{baseline['verified']}/80 →
  {post['verified']}/80，delta {evaluation['verified_delta']:+d}；
- process：{baseline['by_view']['process']['verified']}/24 →
  {post['by_view']['process']['verified']}/24；
- final-only：{baseline['by_view']['final']['verified']}/24 →
  {post['by_view']['final']['verified']}/24；
- both-verified pairs：{baseline['pair_summary']['both_verified']}/24 →
  {post['pair_summary']['both_verified']}/24；
- JSON：{baseline['by_view']['json_preservation']['verified']}/32 →
  {post['by_view']['json_preservation']['verified']}/32；
- 四个 JSON family 均 non-regression：
  {str(evaluation['json_family_non_regression']).lower()}。

## 真实修复

- 表达式：`{fixed['expression']}`；
- process view：最后得到 `{fixed['expected']}`；
- base final-only：`{fixed['baseline']}`；
- consistency final-only：`{fixed['post_sft']}`。

这是第一个在 fresh pair 上观察到的 process→final 正向迁移。

## 不确定性

- aggregate：5 wins / 1 loss，delta
  {comparisons['aggregate']['delta']:+.4f}，95% CI
  [{comparisons['aggregate']['paired_bootstrap_95_ci'][0]:+.4f},
  {comparisons['aggregate']['paired_bootstrap_95_ci'][1]:+.4f}]，
  McNemar p={comparisons['aggregate']['mcnemar_exact_p']:.5f}；
- final-only：1 win / 0 loss，delta
  {comparisons['final']['delta']:+.4f}，95% CI
  [{comparisons['final']['paired_bootstrap_95_ci'][0]:+.4f},
  {comparisons['final']['paired_bootstrap_95_ci'][1]:+.4f}]，
  McNemar p={comparisons['final']['mcnemar_exact_p']:.1f}；
- both-verified pair：1 win / 0 loss，指标同 final-only。

方向是正的，但置信区间下界为 0，统计检验不显著。

## 决策

- accepted_local_method_smoke：
  {str(decision['accepted_local_method_smoke']).lower()}；
- statistically_supported：
  {str(decision['statistically_supported']).lower()}；
- larger training / benchmark / independent holdout / RL：全部关闭。

下一步：{decision['next_action']}

## Evidence

- config SHA256: `{report['identity']['config_sha256']}`;
- heldout SHA256:
  `{report['identity']['heldout_sample_id_sha256']}`;
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
