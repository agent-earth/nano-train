#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts/continuation/anchored-v1-choice-replay-v2"
CONFIG = ROOT / "configs/continuation/anchored_v1_choice_replay_v2.json"
METRICS = ARTIFACTS / "metrics.json"
GENERATIONS = ARTIFACTS / "generations.json"
RELOAD = ARTIFACTS / "reload_validation.json"
PRE_REGISTRATION_REVISION = "277b46f"
PRE_REGISTRATION_TREE = "818dcce54a6fff99d8a20bf14dded061a4e06d42"
DATA_REVISION = "744965a"
CONFIG_SHA256 = (
    "afb70e3c2a7008bc4c6175ed1d988a5"
    "d99b14465320a2c18918848728bfaad16"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def family_score(metrics: dict, family: str) -> int:
    return int(metrics["by_family"][family]["semantic_exact"])


def main() -> None:
    if sha256_file(CONFIG) != CONFIG_SHA256:
        raise SystemExit("choice replay config differs from pre-registration")
    metrics = json.loads(METRICS.read_text(encoding="utf-8"))
    generations = json.loads(GENERATIONS.read_text(encoding="utf-8"))
    reload = json.loads(RELOAD.read_text(encoding="utf-8"))
    baseline = metrics["baseline_validation"]
    post = metrics["post_validation"]
    audit = reload["tensor_audit"]

    if reload["validation"] != post or not reload["reload_matches_training"]:
        raise SystemExit("independent reload does not reproduce training result")
    if reload["adapter_tree_sha256"] != metrics["adapter_sha256"]:
        raise SystemExit("adapter identity differs between receipts")

    baseline_rows = generations["baseline"]
    post_rows = generations["post"]
    if [row["sample_id"] for row in baseline_rows] != [
        row["sample_id"] for row in post_rows
    ]:
        raise SystemExit("baseline and post rows are not aligned")
    changed = [
        {
            "sample_id": before["sample_id"],
            "task_family": before["task_family"],
            "target": before["target"],
            "baseline_output": before["output"],
            "post_output": after["output"],
            "baseline_exact": before["exact"],
            "post_exact": after["exact"],
        }
        for before, after in zip(baseline_rows, post_rows)
        if before["output"] != after["output"]
    ]

    numeric = "capability_preservation_numeric"
    choice = "capability_preservation_choice"
    process = "semantic_arithmetic_process"
    checks = {
        "baseline_reproduces_anchored_v1": (
            baseline["exact"] == 22
            and baseline["semantic_exact"] == 25
            and family_score(baseline, numeric) == 11
            and family_score(baseline, choice) == 6
            and family_score(baseline, process) == 8
        ),
        "strict_at_least_22": post["exact"] >= 22,
        "semantic_at_least_25": post["semantic_exact"] >= 25,
        "numeric_at_least_11": family_score(post, numeric) >= 11,
        "choice_at_least_7": family_score(post, choice) >= 7,
        "process_equals_8": family_score(post, process) == 8,
        "relative_b_drift_at_most_0_06": metrics["relative_drift_l2"] <= 0.06,
        "lora_a_unchanged": audit["a_tensors_changed"] == 0,
        "all_lora_b_changed": audit["b_tensors_changed"] == 112,
        "all_tensors_finite": audit["nonfinite_tensors"] == 0,
        "reload_matches_training": reload["reload_matches_training"],
        "no_failure_receipt": not metrics["failure_receipt_exists"],
    }
    passed = all(checks.values())
    if passed:
        raise SystemExit("choice replay unexpectedly passes its frozen gate")
    if [key for key, value in checks.items() if not value] != [
        "choice_at_least_7"
    ]:
        raise SystemExit("choice replay failed outside the expected choice gate")

    losses_finite = all(
        math.isfinite(float(row[key]))
        for row in metrics["loss_curve"]
        for key in ("ce_loss", "anchor_penalty", "total_loss", "learning_rate")
    )
    if not losses_finite:
        raise SystemExit("nonfinite optimization metric")

    report = {
        "schema_version": "nano_train_public_anchored_choice_replay_v2",
        "experiment_id": metrics["experiment_id"],
        "pre_registration_revision": PRE_REGISTRATION_REVISION,
        "pre_registration_tree": PRE_REGISTRATION_TREE,
        "data_revision": DATA_REVISION,
        "passed": False,
        "identity": {
            "config_sha256": CONFIG_SHA256,
            "model_config_sha256": metrics["model_config_sha256"],
            "dataset_id": metrics["dataset"]["dataset_id"],
            "dataset_sha256": metrics["dataset"]["sha256"],
            "anchor_adapter_tree_sha256": (
                metrics["anchor_adapter_tree_sha256"]
            ),
            "adapter_tree_sha256": metrics["adapter_sha256"],
        },
        "method": {
            "max_steps": metrics["config"]["max_steps"],
            "effective_batch_size": (
                metrics["config"]["batch_size"]
                * metrics["config"]["gradient_accumulation_steps"]
            ),
            "examples_seen": metrics["training_exposure"]["examples_seen"],
            "generation_rule_counts": metrics["training_exposure"][
                "generation_rule_counts"
            ],
            "learning_rate": metrics["config"]["learning_rate"],
            "anchor_penalty_coefficient": metrics["config"][
                "anchor_penalty_coefficient"
            ],
            "trainable_lora_b_only": metrics["trainable_lora_b_only"],
            "frozen_lora_a": metrics["frozen_lora_a"],
            "trainable_parameters": metrics["trainable_parameters"],
            "relative_drift_l2": metrics["relative_drift_l2"],
        },
        "baseline_validation": baseline,
        "post_validation": post,
        "local_gate": checks,
        "case_delta": {
            "outputs_changed": len(changed),
            "exact_labels_changed": sum(
                before["exact"] != after["exact"]
                for before, after in zip(baseline_rows, post_rows)
            ),
            "semantic_labels_changed": sum(
                before["semantic_valid"] != after["semantic_valid"]
                for before, after in zip(baseline_rows, post_rows)
            ),
            "changed_rows": changed,
        },
        "optimization": {
            "all_metrics_finite": losses_finite,
            "loss_curve": metrics["loss_curve"],
            "training_peak_allocated_gib": metrics["peak_allocated_gib"],
            "wall_seconds": metrics["wall_seconds"],
            "failure_receipt_exists": metrics["failure_receipt_exists"],
        },
        "validation": {
            "reload_matches_training": reload["reload_matches_training"],
            "reload_peak_allocated_gib": reload["peak_allocated_gib"],
            "a_tensors_changed": audit["a_tensors_changed"],
            "b_tensors_changed": audit["b_tensors_changed"],
            "nonfinite_tensors": audit["nonfinite_tensors"],
        },
        "evaluation_boundary": {
            "local_role": "development_gate_only",
            "sealed_canary_run": False,
            "prior_full_suite_run": False,
            "independent_holdout_run": False,
            "independent_holdout_prompts_loaded": False,
            "independent_holdout_references_loaded": False,
            "independent_quality_claim_allowed": False,
        },
        "artifacts": {
            "metrics_sha256": sha256_file(METRICS),
            "generations_sha256": sha256_file(GENERATIONS),
            "reload_validation_sha256": sha256_file(RELOAD),
            "adapter_tree_sha256": metrics["adapter_sha256"],
        },
        "decision": {
            "accepted_local_smoke": False,
            "sealed_canary_allowed": False,
            "prior_full_suite_allowed": False,
            "independent_holdout_allowed": False,
            "merge_allowed": False,
            "scale_allowed": False,
            "rl_allowed": False,
            "further_choice_replay_dose_search_allowed": False,
            "next_action": (
                "Reject supervised choice replay at the frozen dose and "
                "preserve anchored-v1. Replan toward a separately "
                "pre-registered generic contract-aware harness intervention; "
                "do not tune replay dose on this development split."
            ),
        },
    }

    changed_row = changed[0]
    markdown = f"""# Anchored-v1 Choice Replay Continuation v2 Result

## Result

V2 is numerically stable but fails its frozen local choice gate.

- baseline and post strict / semantic: {baseline['exact']}/32 /
  {baseline['semantic_exact']}/32 and {post['exact']}/32 /
  {post['semantic_exact']}/32;
- post numeric / choice / process semantic:
  {family_score(post, numeric)}/16, {family_score(post, choice)}/8,
  {family_score(post, process)}/8;
- relative LoRA B drift: {metrics['relative_drift_l2']:.6f};
- training / reload peak memory:
  {metrics['peak_allocated_gib']:.2f} / {reload['peak_allocated_gib']:.2f} GiB;
- independent reload exactly reproduces metrics and failure IDs.

The only failed gate is choice >=7/8: the result remains 6/8.

## Mechanism Evidence

All 112 LoRA A tensors remain byte-identical, all 112 B tensors change, and
all adapter tensors are finite. Of 32 development outputs, exactly one changes:
`{changed_row['sample_id']}` moves from `{changed_row['baseline_output']}` to
`{changed_row['post_output']}` while the synthetic target is
`{changed_row['target']}`. The update moves a choice decision boundary but does
not fix a case; strict, semantic, numeric, choice, and process scores are all
unchanged.

Stop this supervised replay path. Do not search a larger replay dose on the
same development split.

## Decision

Reject v2 and preserve anchored-v1. The sealed canary, old full-development
suite, and independent holdout were not run. The holdout remains unread.

## Reproduction Identity

- pre-registration revision / tree: `{PRE_REGISTRATION_REVISION}` /
  `{PRE_REGISTRATION_TREE}`;
- data revision: `{DATA_REVISION}`;
- config SHA256: `{CONFIG_SHA256}`;
- dataset SHA256: `{metrics['dataset']['sha256']}`;
- anchor adapter tree SHA256: `{metrics['anchor_adapter_tree_sha256']}`;
- candidate adapter tree SHA256: `{metrics['adapter_sha256']}`;
- metrics SHA256: `{sha256_file(METRICS)}`;
- generations SHA256: `{sha256_file(GENERATIONS)}`;
- reload receipt SHA256: `{sha256_file(RELOAD)}`.
"""

    output = ROOT / "docs/results"
    output.mkdir(parents=True, exist_ok=True)
    (output / "anchored_v1_choice_replay_continuation_v2.public.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "anchored_v1_choice_replay_continuation_v2.md").write_text(
        markdown,
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "passed": False,
                "failed_gate": "choice_at_least_7",
                "post_choice": family_score(post, choice),
                "sealed_canary_allowed": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
