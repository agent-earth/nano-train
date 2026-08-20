#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path

from nano_train.orca_math_dpo import load_config
from nano_train.sft import sha256_file


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/preference_orca_math_dpo_v1.json"
PREREGISTER = (
    ROOT / "docs/experiments/orca_math_verifier_dpo_v1.preregister.json"
)
ARTIFACTS = ROOT / "artifacts/orca-math-verifier-dpo-v1"
METRICS = ARTIFACTS / "metrics.json"
GENERATIONS = ARTIFACTS / "generations.json"
RELOAD = ARTIFACTS / "reload_validation.json"
PUBLIC = ROOT / "docs/results/orca_math_verifier_dpo_v1.public.json"
MARKDOWN = ROOT / "docs/results/orca_math_verifier_dpo_v1.md"


def build_report() -> dict:
    config = load_config(CONFIG)
    preregister = json.loads(PREREGISTER.read_text(encoding="utf-8"))
    metrics = json.loads(METRICS.read_text(encoding="utf-8"))
    generations = json.loads(GENERATIONS.read_text(encoding="utf-8"))
    reload_receipt = json.loads(RELOAD.read_text(encoding="utf-8"))
    if (
        preregister.get("schema_version")
        != "nano_train_orca_math_dpo_preregister_v1"
        or preregister.get("identity", {}).get("config_sha256")
        != sha256_file(CONFIG)
        or metrics.get("schema_version")
        != "nano_train_orca_math_dpo_result_v1"
        or metrics.get("config", {}).get("experiment_id")
        != config.experiment_id
        or metrics.get("adapter_sha256")
        != reload_receipt.get("adapter_sha256")
        or reload_receipt.get("metrics_exact") is not True
        or reload_receipt.get("generations_exact") is not True
    ):
        raise ValueError("Orca Math DPO identity differs")
    baseline = {
        row["case_id"]: row for row in generations["baseline"]
    }
    post = {row["case_id"]: row for row in generations["post_dpo"]}
    if set(baseline) != set(post) or len(baseline) != 192:
        raise ValueError("Orca Math DPO generation case sets differ")
    changed = sum(
        baseline[case_id]["output"] != post[case_id]["output"]
        for case_id in baseline
    )
    loss_curve = metrics["loss_curve"]
    return {
        "schema_version": "nano_train_orca_math_dpo_public_v1",
        "experiment_id": config.experiment_id,
        "identity": {
            "result_revision": (
                "3dfadc03d9296d0b230786cd1627c1e3decb7fdc"
            ),
            "config_sha256": sha256_file(CONFIG),
            "preregister_sha256": sha256_file(PREREGISTER),
            "dataset_file_sha256": config.dataset_file_sha256,
            "release_manifest_sha256": config.release_manifest_sha256,
            "prior_sft_result_sha256": config.prior_sft_result_sha256,
            "metrics_sha256": sha256_file(METRICS),
            "generations_sha256": sha256_file(GENERATIONS),
            "reload_receipt_sha256": sha256_file(RELOAD),
            "adapter_sha256": metrics["adapter_sha256"],
        },
        "training": {
            "optimizer_steps": len(loss_curve),
            "trainable_parameters": metrics["trainable_parameters"],
            "loss_first": loss_curve[0]["loss"],
            "loss_last": loss_curve[-1]["loss"],
            "loss_minimum": min(row["loss"] for row in loss_curve),
            "advantage_first": loss_curve[0]["advantage"],
            "advantage_last": loss_curve[-1]["advantage"],
            "advantage_maximum": max(
                row["advantage"] for row in loss_curve
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
                "Reject full-trajectory mean-log-probability DPO. "
                "Pre-register a FINAL-token-only preference objective on "
                "fresh, disjoint preference pairs and fresh local dev."
            ),
        },
        "mechanism_conclusion": (
            "Chosen and rejected targets differ only in the FINAL suffix. "
            "Averaging log probability over the long shared trajectory "
            "diluted that signal: the final preference advantage remained "
            "near zero and no development output changed."
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
    return f"""# Orca Math Verifier DPO v1 Result

## Verdict

**REJECT: STABLE NO-OP.**

- Baseline / post-DPO: {evaluation['baseline']['correct']}/192 ->
  {evaluation['post_dpo']['correct']}/192;
- changed outputs: {evaluation['changed_outputs']}/192;
- paired delta: {comparison['delta']:+.4f};
- 95% CI:
  [{comparison['paired_bootstrap_95_ci'][0]:+.4f},
  {comparison['paired_bootstrap_95_ci'][1]:+.4f}];
- candidate-only / baseline-only:
  {comparison['paired_counts']['candidate_only']} /
  {comparison['paired_counts']['baseline_only']}.

## Training

- 32 fresh preference pairs, one optimizer step each;
- loss {training['loss_first']:.9f} ->
  {training['loss_last']:.9f};
- preference advantage {training['advantage_first']:.9g} ->
  {training['advantage_last']:.9g};
- all losses and gradient norms finite;
- independent reload reproduced all 192 generations exactly.

## Conclusion

{report['mechanism_conclusion']}

Do not tune beta, LR, steps, seed, selection, parser, or LoRA scope on this
observed dev. The next experiment must use disjoint pairs and dev rows, and
score only the differing FINAL suffix tokens.

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
