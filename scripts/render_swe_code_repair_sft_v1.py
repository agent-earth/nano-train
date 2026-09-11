#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path

from nano_train.sft import sha256_file
from nano_train.swe_code_repair_sft import load_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/sft/swe_code_repair_smoke_v1.json"
PREREGISTER = (
    ROOT
    / "docs/experiments/evoloop_swe_code_repair_sft_smoke_v1.preregister.json"
)
ARTIFACTS = ROOT / "artifacts/evoloop-swe-code-repair-sft-smoke-v1"
METRICS = ARTIFACTS / "metrics.json"
GENERATIONS = ARTIFACTS / "validation_generations.json"
RELOAD = ARTIFACTS / "reload_validation.json"
DIAGNOSTICS = ARTIFACTS / "generation_diagnostics.json"
PUBLIC = (
    ROOT / "docs/results/evoloop_swe_code_repair_sft_smoke_v1.public.json"
)
MARKDOWN = ROOT / "docs/results/evoloop_swe_code_repair_sft_smoke_v1.md"


def build_report() -> dict:
    config = load_config(CONFIG)
    preregister = json.loads(PREREGISTER.read_text(encoding="utf-8"))
    metrics = json.loads(METRICS.read_text(encoding="utf-8"))
    generations = json.loads(GENERATIONS.read_text(encoding="utf-8"))
    reload_receipt = json.loads(RELOAD.read_text(encoding="utf-8"))
    diagnostics = json.loads(DIAGNOSTICS.read_text(encoding="utf-8"))
    if (
        preregister.get("schema_version")
        != "nano_train_swe_code_repair_sft_preregister_v1"
        or preregister.get("identity", {}).get("config_sha256")
        != sha256_file(CONFIG)
        or metrics.get("schema_version")
        != "nano_train_swe_code_repair_sft_result_v1"
        or metrics.get("experiment_id") != config.experiment_id
        or metrics.get("dataset_file_sha256")
        != config.dataset_file_sha256
        or metrics.get("release_manifest_sha256")
        != config.release_manifest_sha256
        or metrics.get("model_config_sha256")
        != config.model_config_sha256
        or metrics.get("candidate_admitted_for_fresh8") is not False
        or len(generations) != 12
        or reload_receipt.get("adapter_sha256")
        != metrics.get("adapter_sha256")
        or reload_receipt.get("reload_success") is not True
        or reload_receipt.get("generations_exact") is not True
        or reload_receipt.get("generation_metrics_exact") is not True
        or diagnostics.get("adapter_reproduction_exact") is not True
        or diagnostics.get("private_content_recorded") is not False
    ):
        raise ValueError("SWE code-repair SFT result identity differs")
    exposure_ids = [
        sample_id
        for step in metrics["train_exposure"]
        for sample_id in step["sample_ids"]
    ]
    if len(exposure_ids) != 160 or len(set(exposure_ids)) != 160:
        raise ValueError("SWE code-repair SFT exposure differs")
    base_diagnostic = diagnostics["arms"]["base"]["summary"]
    adapter_diagnostic = diagnostics["arms"]["adapter"]["summary"]
    if (
        adapter_diagnostic["terminus_parser_valid"] != 0
        or adapter_diagnostic["terminus_actionable_or_complete"] != 0
    ):
        raise ValueError("SWE code-repair SFT parser decision differs")
    return {
        "schema_version": "nano_train_swe_code_repair_sft_public_v1",
        "experiment_id": config.experiment_id,
        "project_name": "EvoLoop",
        "identity": {
            "training_revision": (
                "e437326503bc9ceaa4fa329201f6e808fbd2763d"
            ),
            "config_sha256": sha256_file(CONFIG),
            "preregister_sha256": sha256_file(PREREGISTER),
            "dataset_file_sha256": config.dataset_file_sha256,
            "release_manifest_sha256": config.release_manifest_sha256,
            "model_config_sha256": config.model_config_sha256,
            "metrics_sha256": sha256_file(METRICS),
            "generations_sha256": sha256_file(GENERATIONS),
            "reload_receipt_sha256": sha256_file(RELOAD),
            "diagnostics_sha256": sha256_file(DIAGNOSTICS),
            "adapter_sha256": metrics["adapter_sha256"],
            "terminus_parser_sha256": diagnostics["identity"][
                "terminus_parser_sha256"
            ],
        },
        "data": {
            "source": "SWE-bench train split, excluding all Verified-500",
            "release_train_rows": 11_000,
            "release_dev_rows": 512,
            "release_train_tokens": 11_861_953,
            "release_repositories": 35,
            "verified_exact_and_near_overlap": 0,
            "selected_train_rows": 160,
            "selected_dev_rows": 12,
            "selected_train_by_band": metrics["selection"]["train_by_band"],
            "selected_dev_by_band": metrics["selection"]["dev_by_band"],
        },
        "training": {
            "optimizer_steps": config.max_steps,
            "unique_train_exposures": len(set(exposure_ids)),
            "trainable_parameters": metrics["trainable_parameters"],
            "loss_first": metrics["loss_curve"][0]["loss"],
            "loss_last": metrics["loss_curve"][-1]["loss"],
            "loss_minimum": min(row["loss"] for row in metrics["loss_curve"]),
            "peak_allocated_gib": metrics["peak_allocated_gib"],
            "wall_seconds": metrics["wall_seconds"],
        },
        "heldout": {
            "baseline_dev_loss": metrics["baseline_dev_loss"],
            "post_sft_dev_loss": metrics["post_dev_loss"],
            "relative_dev_loss_improvement": metrics[
                "relative_dev_loss_improvement"
            ],
            "frozen_gate_results": metrics["gates"],
            "strict_json_action_valid": metrics["generation_validation"][
                "json_action_valid"
            ],
            "strict_json_action_samples": metrics["generation_validation"][
                "samples"
            ],
            "base_terminus_parser": base_diagnostic,
            "adapter_terminus_parser": adapter_diagnostic,
        },
        "reload": {
            "success": reload_receipt["reload_success"],
            "post_dev_loss_exact": (
                reload_receipt["post_dev_loss"]
                == reload_receipt["source_post_dev_loss"]
            ),
            "generation_metrics_exact": reload_receipt[
                "generation_metrics_exact"
            ],
            "generations_exact": reload_receipt["generations_exact"],
            "peak_allocated_gib": reload_receipt["peak_allocated_gib"],
        },
        "decision": {
            "candidate_admitted_for_fresh8": False,
            "fresh8_run_allowed": False,
            "complete_500_allowed": False,
            "rl_or_opd_allowed_for_this_adapter": False,
            "rerun_or_posthoc_tuning_allowed": False,
            "verdict": "reject_standard_sft_v1",
            "mechanism_conclusion": (
                "Standard q/v-only SFT lowered held-out teacher-forced loss "
                "by 10.75% but did not teach the executable Terminus command "
                "schema. The dominant failure moved from command strings to "
                "command/output objects, while Harbor requires "
                "keystrokes/duration objects."
            ),
            "next_action": (
                "Freeze a new train/dev selection and test an explicit "
                "schema-preference or format-balanced objective. Do not "
                "change v1 thresholds or reuse the observed 12-row dev set."
            ),
        },
        "claim_boundary": (
            "This is a local held-out training result, not a SWE-bench score. "
            "It rejects one SFT recipe and supports only the next "
            "mechanism-level experiment."
        ),
    }


def render_markdown(report: dict) -> str:
    training = report["training"]
    heldout = report["heldout"]
    decision = report["decision"]
    return f"""# EvoLoop SWE Code-Repair SFT Smoke v1

## Verdict

**REJECT.** This adapter is reproducible, but it is not eligible for fresh-8.

- Data release: {report['data']['release_train_rows']:,} train rows,
  {report['data']['release_train_tokens']:,} Qwen3.5 tokens, and zero exact or
  near overlap with all 500 SWE-bench Verified cases.
- Training exposure: {training['unique_train_exposures']} unique rows across
  {training['optimizer_steps']} optimizer steps.
- Held-out loss: {heldout['baseline_dev_loss']:.6f} ->
  {heldout['post_sft_dev_loss']:.6f}
  ({heldout['relative_dev_loss_improvement']:+.2%} relative).
- Strict JSON action validity: {heldout['strict_json_action_valid']}/
  {heldout['strict_json_action_samples']}.
- Harbor Terminus parser validity:
  {heldout['adapter_terminus_parser']['terminus_parser_valid']}/12.
- Independent reload reproduced the loss and all 12 generations exactly.

## What Failed

The model usually emitted a JSON-looking response, often inside a Markdown
fence, but the real Harbor parser rejected the command entries. The base model
mostly produced command strings. The LoRA shifted many outputs to
`command`/`output` objects, while Terminus requires `keystrokes`/`duration`
objects. Only one adapter response exhausted the 768-token budget, so simple
truncation is not the main cause.

## Conclusion

{decision['mechanism_conclusion']}

Loss reduction alone is insufficient evidence for agent quality. The next run
must change the supervision objective, use a fresh held-out slice, and preserve
the current v1 artifacts as negative evidence.

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
