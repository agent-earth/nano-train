#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path

from nano_train.orca_math_dpo_suffix import load_config
from nano_train.sft import sha256_file


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/preference_orca_math_dpo_v2.json"
PREREGISTER = (
    ROOT
    / "docs/experiments/orca_math_verifier_dpo_suffix_v2.preregister.json"
)
ARTIFACTS = ROOT / "artifacts/orca-math-verifier-dpo-suffix-v2"
METRICS = ARTIFACTS / "metrics.json"
GENERATIONS = ARTIFACTS / "generations.json"
RELOAD = ARTIFACTS / "reload_validation.json"
FAILURE = ARTIFACTS / "attempt-001.failure.json"
PUBLIC = (
    ROOT / "docs/results/orca_math_verifier_dpo_suffix_v2.public.json"
)
MARKDOWN = ROOT / "docs/results/orca_math_verifier_dpo_suffix_v2.md"


def build_report() -> dict:
    config = load_config(CONFIG)
    preregister = json.loads(PREREGISTER.read_text(encoding="utf-8"))
    metrics = json.loads(METRICS.read_text(encoding="utf-8"))
    generations = json.loads(GENERATIONS.read_text(encoding="utf-8"))
    reload_receipt = json.loads(RELOAD.read_text(encoding="utf-8"))
    failure = json.loads(FAILURE.read_text(encoding="utf-8"))
    if (
        preregister.get("schema_version")
        != "nano_train_orca_math_dpo_suffix_preregister_v2"
        or preregister.get("identity", {}).get("config_sha256")
        != sha256_file(CONFIG)
        or metrics.get("schema_version")
        != "nano_train_orca_math_dpo_suffix_result_v2"
        or metrics.get("adapter_sha256")
        != reload_receipt.get("adapter_sha256")
        or reload_receipt.get("metrics_exact") is not True
        or reload_receipt.get("generations_exact") is not True
        or failure.get("optimizer_steps_completed") != 0
        or failure.get("adapter_saved") is not False
        or failure.get("config_changed") is not False
        or failure.get("objective_changed") is not False
    ):
        raise ValueError("suffix DPO v2 identity differs")
    baseline = {
        row["case_id"]: row for row in generations["baseline"]
    }
    post = {row["case_id"]: row for row in generations["post_dpo"]}
    if set(baseline) != set(post) or len(baseline) != 192:
        raise ValueError("suffix DPO v2 generation case sets differ")
    changed = sum(
        baseline[case_id]["output"] != post[case_id]["output"]
        for case_id in baseline
    )
    correctness_changed = sum(
        baseline[case_id]["correct"] != post[case_id]["correct"]
        for case_id in baseline
    )
    parse_status_changed = sum(
        baseline[case_id]["parse_failure"] != post[case_id]["parse_failure"]
        for case_id in baseline
    )
    curve = metrics["loss_curve"]
    return {
        "schema_version": "nano_train_orca_math_dpo_suffix_public_v2",
        "experiment_id": config.experiment_id,
        "identity": {
            "result_revision": (
                "12a6f3c79ad3c81aece639f2dba2fd3943d10265"
            ),
            "config_sha256": sha256_file(CONFIG),
            "preregister_sha256": sha256_file(PREREGISTER),
            "dataset_file_sha256": config.dataset_file_sha256,
            "release_manifest_sha256": config.release_manifest_sha256,
            "prior_dpo_preregister_sha256": (
                config.prior_dpo_preregister_sha256
            ),
            "prior_dpo_result_sha256": config.prior_dpo_result_sha256,
            "metrics_sha256": sha256_file(METRICS),
            "generations_sha256": sha256_file(GENERATIONS),
            "reload_receipt_sha256": sha256_file(RELOAD),
            "failure_receipt_sha256": sha256_file(FAILURE),
            "adapter_sha256": metrics["adapter_sha256"],
        },
        "failed_attempt": failure,
        "training": {
            "optimizer_steps": len(curve),
            "trainable_parameters": metrics["trainable_parameters"],
            "loss_first": curve[0]["loss"],
            "loss_last": curve[-1]["loss"],
            "loss_minimum": min(row["loss"] for row in curve),
            "advantage_first": curve[0]["advantage"],
            "advantage_last": curve[-1]["advantage"],
            "advantage_maximum": max(
                row["advantage"] for row in curve
            ),
            "supervised_tokens_min": min(
                min(
                    row["chosen_supervised_tokens"],
                    row["rejected_supervised_tokens"],
                )
                for row in curve
            ),
            "supervised_tokens_max": max(
                max(
                    row["chosen_supervised_tokens"],
                    row["rejected_supervised_tokens"],
                )
                for row in curve
            ),
            "all_losses_finite": metrics["all_losses_finite"],
            "all_gradient_norms_finite": metrics[
                "all_gradient_norms_finite"
            ],
            "peak_allocated_gib": metrics["peak_allocated_gib"],
            "wall_seconds": metrics["wall_seconds"],
        },
        "evaluation": {
            "baseline": metrics["baseline_validation"],
            "post_dpo": metrics["post_validation"],
            "comparison": metrics["comparison"],
            "changed_outputs": changed,
            "correctness_changed": correctness_changed,
            "parse_status_changed": parse_status_changed,
        },
        "reload": {
            "success": reload_receipt["reload_success"],
            "metrics_exact": reload_receipt["metrics_exact"],
            "generations_exact": reload_receipt["generations_exact"],
            "peak_allocated_gib": reload_receipt["peak_allocated_gib"],
        },
        "decision": {
            "gates": metrics["gates"],
            "candidate_admitted": False,
            "benchmark_allowed": False,
            "larger_training_allowed": False,
            "rerun_or_tuning_allowed": False,
            "next_action": (
                "Stop preference-training dose search. Return to a frozen "
                "base-model harness and pre-register full-solve "
                "self-consistency on fresh non-benchmark development."
            ),
        },
        "mechanism_conclusion": (
            "Masking shared trajectory tokens increased preference advantage "
            "by about two orders of magnitude versus v1, but the low-dose "
            "adapter changed only four output strings and changed no score or "
            "parse status. The training signal is now targeted but not large "
            "enough to move measured behavior under the frozen smoke; "
            "post-hoc dose or LR tuning is forbidden."
        ),
        "claim_boundary": (
            "This is a local synthetic-development preference-optimization "
            "result. It is not a GSM8K, MMLU, GPQA, 9B, 27B, or agent-"
            "benchmark result."
        ),
    }


def render_markdown(report: dict) -> str:
    evaluation = report["evaluation"]
    comparison = evaluation["comparison"]
    training = report["training"]
    return f"""# Orca Math Suffix DPO v2 Result

## Verdict

**REJECT: STRONGER MARGIN, ZERO BEHAVIOR CHANGE.**

- Baseline / post-DPO: {evaluation['baseline']['correct']}/192 ->
  {evaluation['post_dpo']['correct']}/192;
- changed outputs: {evaluation['changed_outputs']}/192;
- correctness / parse-status changes:
  {evaluation['correctness_changed']} /
  {evaluation['parse_status_changed']};
- paired delta: {comparison['delta']:+.4f};
- candidate-only / baseline-only:
  {comparison['paired_counts']['candidate_only']} /
  {comparison['paired_counts']['baseline_only']}.

## Training

- 32 fresh pairs and 192 fresh dev rows, all disjoint from DPO v1;
- only {training['supervised_tokens_min']}-
  {training['supervised_tokens_max']} differing suffix tokens per arm were
  scored;
- preference advantage {training['advantage_first']:.9g} ->
  {training['advantage_last']:.9g}, maximum
  {training['advantage_maximum']:.9g};
- all losses and gradient norms finite;
- independent reload reproduced all generations exactly.

Attempt 1 failed before any optimizer step because two reference forwards were
kept live simultaneously and exhausted GPU memory. The repair used
mathematically equivalent split backward coefficients; config, selection,
objective, and thresholds were unchanged.

## Conclusion

{report['mechanism_conclusion']}

Stop this preference-training family. The next experiment returns to the frozen
base model and tests full-solve self-consistency on fresh local data.

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
