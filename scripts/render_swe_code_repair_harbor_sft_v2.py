#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path

from nano_train.sft import sha256_file
from nano_train.swe_code_repair_harbor_sft import load_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/sft/swe_code_repair_harbor_prompt_v2.json"
PREREGISTER = (
    ROOT
    / "docs/experiments/evoloop_swe_code_repair_harbor_sft_v2.preregister.json"
)
ARTIFACTS = ROOT / "artifacts/evoloop-swe-code-repair-harbor-sft-v2"
METRICS = ARTIFACTS / "metrics.json"
ROWS = ARTIFACTS / "structural_rows.json"
RELOAD = ARTIFACTS / "reload_validation.json"
PUBLIC = (
    ROOT / "docs/results/evoloop_swe_code_repair_harbor_sft_v2.public.json"
)
MARKDOWN = ROOT / "docs/results/evoloop_swe_code_repair_harbor_sft_v2.md"


def build_report() -> dict:
    config = load_config(CONFIG)
    preregister = json.loads(PREREGISTER.read_text(encoding="utf-8"))
    metrics = json.loads(METRICS.read_text(encoding="utf-8"))
    rows = json.loads(ROWS.read_text(encoding="utf-8"))
    reload_receipt = json.loads(RELOAD.read_text(encoding="utf-8"))
    if (
        preregister.get("schema_version")
        != "nano_train_swe_code_repair_harbor_sft_preregister_v2"
        or preregister.get("identity", {}).get("config_sha256")
        != sha256_file(CONFIG)
        or metrics.get("schema_version")
        != "nano_train_swe_code_repair_harbor_sft_result_v2"
        or metrics.get("experiment_id") != config.experiment_id
        or metrics.get("candidate_admitted_for_fresh8") is not True
        or metrics.get("adapter_sha256") != reload_receipt.get("adapter_sha256")
        or metrics.get("structural_rows_sha256") != sha256_file(ROWS)
        or reload_receipt.get("reload_success") is not True
        or reload_receipt.get("loss_exact") is not True
        or reload_receipt.get("generations_exact") is not True
        or set(rows) != {"base", "standard_sft_v1", "harbor_sft_v2"}
        or any(len(value) != 24 for value in rows.values())
        or metrics.get("private_content_recorded") is not False
    ):
        raise ValueError("Harbor-prompt SFT v2 result identity differs")
    if not all(metrics["gates"].values()):
        raise ValueError("Harbor-prompt SFT v2 admission gates differ")
    exposure_ids = [
        sample_id
        for step in metrics["training"]["train_exposure"]
        for sample_id in step["sample_ids"]
    ]
    if len(exposure_ids) != 160 or len(set(exposure_ids)) != 160:
        raise ValueError("Harbor-prompt SFT v2 exposure differs")
    return {
        "schema_version": "nano_train_swe_code_repair_harbor_sft_public_v2",
        "experiment_id": config.experiment_id,
        "project_name": "EvoLoop",
        "identity": {
            "training_revision": (
                "f742df63da406bac094a15adf3bbc3fa175903d5"
            ),
            "config_sha256": sha256_file(CONFIG),
            "preregister_sha256": sha256_file(PREREGISTER),
            "metrics_sha256": sha256_file(METRICS),
            "structural_rows_sha256": sha256_file(ROWS),
            "reload_receipt_sha256": sha256_file(RELOAD),
            "adapter_sha256": metrics["adapter_sha256"],
            "source_v1_adapter_sha256": config.source_v1_adapter_sha256,
            "terminus_parser_sha256": config.terminus_parser_sha256,
            "terminus_prompt_template_sha256": (
                config.terminus_prompt_template_sha256
            ),
        },
        "single_variable": {
            "changed": (
                "Training prompt now exactly matches the deployed Harbor "
                "Terminus2 first-turn template."
            ),
            "length_adjustment": (
                "max_length 2048 -> 2816 prevents truncation of 34 unchanged "
                "training rows; the longest selected sequence is 2638."
            ),
            "unchanged": [
                "160 training sample IDs and order",
                "40 optimizer steps and four micro-batches per step",
                "learning rate, scheduler, seed, and FP32",
                "q/v-only LoRA r=8 alpha=16",
                "source dataset and release",
                "greedy decoding and 768-token output budget",
            ],
        },
        "selection": metrics["selection"],
        "training": {
            "optimizer_steps": metrics["training"]["optimizer_steps"],
            "unique_train_exposures": len(set(exposure_ids)),
            "trainable_parameters": metrics["training"][
                "trainable_parameters"
            ],
            "loss_first": metrics["training"]["loss_curve"][0]["loss"],
            "loss_last": metrics["training"]["loss_curve"][-1]["loss"],
            "loss_minimum": min(
                row["loss"] for row in metrics["training"]["loss_curve"]
            ),
            "all_losses_finite": metrics["training"]["all_losses_finite"],
            "all_gradient_norms_finite": metrics["training"][
                "all_gradient_norms_finite"
            ],
            "peak_allocated_gib": metrics["peak_allocated_gib"],
            "wall_seconds": metrics["wall_seconds"],
        },
        "fresh_heldout": {
            "loss": metrics["loss"],
            "structure": metrics["structure"],
        },
        "reload": {
            "success": reload_receipt["reload_success"],
            "loss_exact": reload_receipt["loss_exact"],
            "generations_exact": reload_receipt["generations_exact"],
            "peak_allocated_gib": reload_receipt["peak_allocated_gib"],
        },
        "decision": {
            "gates": metrics["gates"],
            "candidate_admitted_for_fresh8": True,
            "verdict": "admit_to_frozen_fresh8_screening",
            "mechanism_conclusion": (
                "Prompt-aligned SFT preserves perfect 24/24 executable "
                "Terminus responses while lowering fresh held-out loss by "
                "14.51% versus base and 10.26% versus standard-prompt SFT v1. "
                "This supports the prompt-alignment mechanism, not yet "
                "SWE-bench task-resolution improvement."
            ),
            "next_action": (
                "Serve this exact adapter with vLLM and run the frozen "
                "fresh-8 SFT-only arm first. Run harness+SFT only if the "
                "SFT-only result is non-regressing or reveals a specific "
                "complementary failure mode."
            ),
            "complete_500_allowed": False,
            "rl_or_opd_allowed": False,
            "posthoc_tuning_allowed": False,
        },
        "claim_boundary": (
            "This is a fresh local training qualification, not a SWE-bench "
            "score. It admits one exact adapter only to the frozen fresh-8 "
            "development screen."
        ),
    }


def render_markdown(report: dict) -> str:
    loss = report["fresh_heldout"]["loss"]
    structure = report["fresh_heldout"]["structure"]
    training = report["training"]
    return f"""# EvoLoop Harbor-Prompt SFT v2 Result

## Verdict

**ADMIT TO FROZEN FRESH-8 SCREENING.**

- Same {training['unique_train_exposures']} train rows as v1 and the same
  optimizer/LoRA settings; only the deployed prompt shape changed.
- Training completed {training['optimizer_steps']} finite optimizer steps with
  {training['trainable_parameters']:,} trainable parameters.
- Fresh 24-case loss: base {loss['base']:.6f}, standard SFT v1
  {loss['standard_sft_v1']:.6f}, Harbor-prompt SFT v2
  {loss['harbor_sft_v2']:.6f}.
- v2 loss improvement: {loss['v2_relative_improvement_vs_base']:+.2%} versus
  base and {loss['v2_relative_improvement_vs_v1']:+.2%} versus v1.
- Terminus parser validity: base
  {structure['base']['terminus_parser_valid']}/24, v1
  {structure['standard_sft_v1']['terminus_parser_valid']}/24, v2
  {structure['harbor_sft_v2']['terminus_parser_valid']}/24.
- v2 versus v1 structural regressions:
  {structure['v2_vs_v1']['baseline_only']}.
- Independent reload reproduced v2 loss and all 24 structural generation rows.

## What This Proves

The train/deploy prompt mismatch was material. Aligning supervision to the
actual Harbor interaction reduced fresh held-out loss substantially without
breaking the command protocol. All frozen local gates pass.

## What It Does Not Prove

Parser validity and teacher-forced loss are proxies. They do not show that the
agent diagnoses repositories, writes correct patches, or passes hidden tests.
Only official frozen fresh-8 scoring can test that.

## Next

Serve adapter `{report['identity']['adapter_sha256']}` and run the preregistered
fresh-8 SFT-only arm. Do not launch the complete 500 cases from this local
result alone.
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
