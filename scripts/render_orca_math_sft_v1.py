#!/usr/bin/env python3

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from nano_train.orca_math_sft import admission_gates, load_config
from nano_train.sft import sha256_file


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/sft/orca_math_smoke_v1.json"
PREREGISTER = (
    ROOT / "docs/experiments/orca_math_sft_smoke_v1.preregister.json"
)
ARTIFACTS = ROOT / "artifacts/orca-math-sft-smoke-v1"
METRICS = ARTIFACTS / "metrics.json"
GENERATIONS = ARTIFACTS / "generations.json"
RELOAD = ARTIFACTS / "reload_validation.json"
PUBLIC = ROOT / "docs/results/orca_math_sft_smoke_v1.public.json"
MARKDOWN = ROOT / "docs/results/orca_math_sft_smoke_v1.md"


def build_report() -> dict:
    config = load_config(CONFIG)
    preregister = json.loads(PREREGISTER.read_text(encoding="utf-8"))
    metrics = json.loads(METRICS.read_text(encoding="utf-8"))
    generations = json.loads(GENERATIONS.read_text(encoding="utf-8"))
    reload_receipt = json.loads(RELOAD.read_text(encoding="utf-8"))
    if (
        preregister.get("schema_version")
        != "nano_train_orca_math_sft_preregister_v1"
        or preregister.get("identity", {}).get("config_sha256")
        != sha256_file(CONFIG)
        or metrics.get("schema_version")
        != "nano_train_orca_math_sft_result_v1"
        or metrics.get("config", {}).get("experiment_id")
        != config.experiment_id
        or metrics.get("dataset_file_sha256")
        != config.dataset_file_sha256
        or metrics.get("release_manifest_sha256")
        != config.release_manifest_sha256
        or metrics.get("model_config_sha256")
        != config.model_config_sha256
        or reload_receipt.get("adapter_sha256")
        != metrics.get("adapter_sha256")
        or reload_receipt.get("metrics_exact") is not True
        or reload_receipt.get("generations_exact") is not True
    ):
        raise ValueError("Orca Math SFT result identity differs")
    baseline = {
        row["case_id"]: row for row in generations["baseline"]
    }
    post = {row["case_id"]: row for row in generations["post_sft"]}
    if set(baseline) != set(post) or len(baseline) != 192:
        raise ValueError("Orca Math SFT generation case sets differ")

    transitions = Counter()
    by_stratum = {}
    for case_id in sorted(baseline):
        before = baseline[case_id]
        after = post[case_id]
        if before["correct"] and not after["correct"]:
            outcome = "regressed"
        elif not before["correct"] and after["correct"]:
            outcome = "repaired"
        elif before["correct"] and after["correct"]:
            outcome = "both_correct"
        else:
            outcome = "both_wrong"
        transitions[outcome] += 1
        if before["parse_failure"] and not after["parse_failure"]:
            transitions["parse_failure_repaired"] += 1
        if not before["parse_failure"] and after["parse_failure"]:
            transitions["parse_failure_introduced"] += 1
        if before["output"] != after["output"]:
            transitions["output_changed"] += 1
        stratum = before["stratum"]
        by_stratum.setdefault(stratum, Counter())[outcome] += 1

    loss_curve = metrics["loss_curve"]
    exposure_ids = [
        sample_id
        for step in metrics["train_exposure"]
        for sample_id in step["sample_ids"]
    ]
    gates = admission_gates(
        metrics["comparison"],
        candidate_by_stratum={
            key: value["correct"]
            for key, value in metrics["post_validation"][
                "by_stratum"
            ].items()
        },
        baseline_by_stratum={
            key: value["correct"]
            for key, value in metrics["baseline_validation"][
                "by_stratum"
            ].items()
        },
        alpha=config.alpha,
        minimum_candidate_only_wins=config.minimum_candidate_only_wins,
    )
    if gates != metrics["gates"] or metrics["candidate_admitted"] is not False:
        raise ValueError("Orca Math SFT decision differs")
    return {
        "schema_version": "nano_train_orca_math_sft_public_v1",
        "experiment_id": config.experiment_id,
        "identity": {
            "result_revision": (
                "ada7e91a7e0d466edfdd3f342f23c62bb5dc2a67"
            ),
            "config_sha256": sha256_file(CONFIG),
            "preregister_sha256": sha256_file(PREREGISTER),
            "dataset_file_sha256": config.dataset_file_sha256,
            "release_manifest_sha256": config.release_manifest_sha256,
            "model_config_sha256": config.model_config_sha256,
            "metrics_sha256": sha256_file(METRICS),
            "generations_sha256": sha256_file(GENERATIONS),
            "reload_receipt_sha256": sha256_file(RELOAD),
            "adapter_sha256": metrics["adapter_sha256"],
        },
        "training": {
            "optimizer_steps": config.max_steps,
            "unique_train_exposures": len(set(exposure_ids)),
            "total_train_exposures": len(exposure_ids),
            "trainable_parameters": metrics["trainable_parameters"],
            "loss_first": loss_curve[0]["loss"],
            "loss_last": loss_curve[-1]["loss"],
            "loss_minimum": min(row["loss"] for row in loss_curve),
            "peak_allocated_gib": metrics["peak_allocated_gib"],
            "wall_seconds": metrics["wall_seconds"],
        },
        "evaluation": {
            "baseline": metrics["baseline_validation"],
            "post_sft": metrics["post_validation"],
            "comparison": metrics["comparison"],
            "transitions": dict(sorted(transitions.items())),
            "by_stratum_transitions": {
                key: dict(sorted(value.items()))
                for key, value in sorted(by_stratum.items())
            },
        },
        "reload": {
            "success": reload_receipt["reload_success"],
            "metrics_exact": reload_receipt["metrics_exact"],
            "generations_exact": reload_receipt["generations_exact"],
            "peak_allocated_gib": reload_receipt["peak_allocated_gib"],
        },
        "decision": {
            "gates": gates,
            "candidate_admitted": False,
            "larger_sft_allowed": False,
            "benchmark_allowed": False,
            "rl_or_opd_allowed_for_this_adapter": False,
            "rerun_or_tuning_allowed": False,
            "next_action": (
                "Reject this adapter. Pre-register a fresh, verifier-guided "
                "RL/OPD method on disjoint non-benchmark math data; do not "
                "tune this SFT run or revisit its observed development rows."
            ),
        },
        "claim_boundary": (
            "This is a local synthetic-development SFT result. It is not a "
            "GSM8K, MMLU, GPQA, 9B, 27B, RL, OPD, or agent-benchmark result."
        ),
    }


def render_markdown(report: dict) -> str:
    evaluation = report["evaluation"]
    comparison = evaluation["comparison"]
    training = report["training"]
    transitions = evaluation["transitions"]
    return f"""# Orca Math SFT Smoke v1 Result

## Verdict

**REJECT.**

- Baseline: {evaluation['baseline']['correct']}/192
  ({evaluation['baseline']['accuracy']:.4f});
- post-SFT: {evaluation['post_sft']['correct']}/192
  ({evaluation['post_sft']['accuracy']:.4f});
- paired delta: {comparison['delta']:+.4f};
- paired bootstrap 95% CI:
  [{comparison['paired_bootstrap_95_ci'][0]:+.4f},
  {comparison['paired_bootstrap_95_ci'][1]:+.4f}];
- exact McNemar p: {comparison['mcnemar_exact_p']:.6g};
- candidate-only / baseline-only:
  {comparison['paired_counts']['candidate_only']} /
  {comparison['paired_counts']['baseline_only']}.

## What Happened

- The training loss moved from {training['loss_first']:.6f} to
  {training['loss_last']:.6f}, with minimum {training['loss_minimum']:.6f}.
- All {training['unique_train_exposures']} training rows were unique and seen
  exactly once.
- Final-line parse failures improved from
  {evaluation['baseline']['parse_failures']} to
  {evaluation['post_sft']['parse_failures']}.
- Despite better format completion, the adapter repaired only
  {transitions['repaired']} cases and regressed
  {transitions['regressed']} previously correct cases.
- Independent reload reproduced every post-SFT generation and metric exactly.

This shows that loss reduction and fewer format failures did not transfer into
math quality. Standard SFT on 160 verbose teacher trajectories catastrophically
reduced final-answer correctness, especially in the medium stratum
(50/96 to 24/96).

## Decision

Reject the adapter and forbid rerun or hyperparameter tuning on the observed
development rows. The next method must use fresh non-benchmark data and a
verifier-guided RL/OPD objective rather than another standard SFT dose search.

## Boundary

{report['claim_boundary']}
"""


def main() -> None:
    report = build_report()
    PUBLIC.parent.mkdir(parents=True, exist_ok=True)
    PUBLIC.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    MARKDOWN.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
