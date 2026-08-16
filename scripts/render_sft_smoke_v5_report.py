#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from transformers import AutoTokenizer

from nano_train.data import (
    evaluate_arithmetic,
    format_number,
    tokenize_samples,
)


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts/semantic-arithmetic-sft-smoke-v5"
V4_GENERATIONS = (
    ROOT / "artifacts/semantic-arithmetic-sft-smoke-v4/generations.json"
)
METRICS = ARTIFACTS / "metrics.json"
GENERATIONS = ARTIFACTS / "generations.json"
RELOAD = ARTIFACTS / "reload_validation.json"
ADAPTER = ARTIFACTS / "adapter"
DATASET = (
    ROOT.parent
    / "nano-data-pipeline/datasets/verified_semantic_arithmetic_traces_v3.json"
)
MODEL = ROOT / "../../models/Qwen3.5-4B"
PRE_REGISTRATION_REVISION = "c8e331d"
CONFIG_SHA256 = (
    "89e48fa387851e06a9394253e3bbdc345"
    "d7a0e84d963015be67e2ae8183fad38"
)
TRACE_PATTERN = re.compile(
    (
        r"CALC: (.+) = "
        r"([-+]?(?:[0-9]+(?:\.[0-9]+)?|\.[0-9]+))\n"
        r"FINAL: ([-+]?(?:[0-9]+(?:\.[0-9]+)?|\.[0-9]+))"
    )
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def adapter_summary() -> dict:
    from safetensors import safe_open

    total = 0
    nonfinite = 0
    dtypes: Counter[str] = Counter()
    with safe_open(
        ADAPTER / "adapter_model.safetensors",
        framework="pt",
        device="cpu",
    ) as handle:
        for key in handle.keys():
            tensor = handle.get_tensor(key)
            total += 1
            dtypes[str(tensor.dtype)] += 1
            nonfinite += not bool(tensor.isfinite().all())
    return {
        "tensor_count": total,
        "nonfinite_tensors": nonfinite,
        "dtype_counts": dict(sorted(dtypes.items())),
    }


def classify(sample, row: dict) -> str:
    output = str(row["output"]).strip()
    match = TRACE_PATTERN.fullmatch(output)
    if row["semantic_valid"]:
        return "semantic_valid"
    if match is None:
        return "invalid_trace_grammar"
    expression, calc_result, final_result = match.groups()
    try:
        verified = format_number(evaluate_arithmetic(expression))
    except (SyntaxError, ValueError, ZeroDivisionError, OverflowError):
        return "unsafe_or_invalid_expression"
    expected = str(sample.verifier["expected_result"])
    if calc_result != final_result:
        return "calc_final_mismatch"
    if verified != calc_result:
        return "calc_execution_mismatch"
    if calc_result != expected:
        return "verified_but_wrong_result"
    return "other"


def taxonomy(samples: dict, rows: list[dict]) -> dict:
    counts: Counter[str] = Counter()
    ids: defaultdict[str, list[str]] = defaultdict(list)
    for row in rows:
        category = classify(samples[row["sample_id"]], row)
        counts[category] += 1
        ids[category].append(row["sample_id"])
    return {
        "counts": dict(sorted(counts.items())),
        "case_ids": dict(sorted(ids.items())),
    }


def token_budget_audit(
    tokenizer,
    all_samples: list,
    post_rows: list[dict],
    max_new_tokens: int,
) -> dict:
    split_summary = {}
    for split in ("train", "validation"):
        rows = [sample for sample in all_samples if sample.split == split]
        content_lengths = [
            len(
                tokenizer(
                    sample.target,
                    add_special_tokens=False,
                ).input_ids
            )
            for sample in rows
        ]
        assistant_lengths = [
            sum(label != -100 for label in sample.labels) for sample in rows
        ]
        split_summary[split] = {
            "samples": len(rows),
            "target_content_token_min": min(content_lengths),
            "target_content_token_max": max(content_lengths),
            "target_content_over_generation_cap": sum(
                length > max_new_tokens for length in content_lengths
            ),
            "assistant_with_eos_token_max": max(assistant_lengths),
            "assistant_with_eos_over_generation_cap": sum(
                length > max_new_tokens for length in assistant_lengths
            ),
        }

    validation = {
        sample.sample_id: sample
        for sample in all_samples
        if sample.split == "validation"
    }
    output_at_cap = []
    failed_output_at_cap = []
    over_cap_target_failures = []
    for row in post_rows:
        output_tokens = len(
            tokenizer(
                str(row["output"]),
                add_special_tokens=False,
            ).input_ids
        )
        target_tokens = len(
            tokenizer(
                validation[row["sample_id"]].target,
                add_special_tokens=False,
            ).input_ids
        )
        if output_tokens >= max_new_tokens:
            output_at_cap.append(row["sample_id"])
            if not row["semantic_valid"]:
                failed_output_at_cap.append(row["sample_id"])
        if target_tokens > max_new_tokens and not row["semantic_valid"]:
            over_cap_target_failures.append(row["sample_id"])
    return {
        "generation_max_new_tokens": max_new_tokens,
        "splits": split_summary,
        "post_sft_output_at_cap": len(output_at_cap),
        "post_sft_failed_output_at_cap": len(failed_output_at_cap),
        "post_sft_over_cap_target_failures": len(over_cap_target_failures),
        "post_sft_output_at_cap_case_ids": output_at_cap,
        "post_sft_over_cap_target_failure_case_ids": over_cap_target_failures,
        "official_score_changed": False,
    }


def transition_summary(
    samples: dict,
    v4_rows: list[dict],
    v5_rows: list[dict],
) -> dict:
    v4 = {row["sample_id"]: row for row in v4_rows}
    v5 = {row["sample_id"]: row for row in v5_rows}
    counts: Counter[str] = Counter()
    ids: defaultdict[str, list[str]] = defaultdict(list)
    for sample_id in sorted(samples):
        before = classify(samples[sample_id], v4[sample_id])
        after = classify(samples[sample_id], v5[sample_id])
        transition = f"{before}->{after}"
        counts[transition] += 1
        ids[transition].append(sample_id)
    return {
        "counts": dict(sorted(counts.items())),
        "case_ids": dict(sorted(ids.items())),
    }


def main() -> None:
    raw = json.loads(METRICS.read_text(encoding="utf-8"))
    generations = json.loads(GENERATIONS.read_text(encoding="utf-8"))
    v4_generations = json.loads(
        V4_GENERATIONS.read_text(encoding="utf-8")
    )
    reload = json.loads(RELOAD.read_text(encoding="utf-8"))
    dataset = json.loads(DATASET.read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    all_samples = tokenize_samples(dataset, tokenizer, max_length=128)
    validation = {
        sample.sample_id: sample
        for sample in all_samples
        if sample.split == "validation"
    }
    losses = [float(row["loss"]) for row in raw["loss_curve"]]
    finite = all(math.isfinite(loss) for loss in losses)
    early_mean = statistics.mean(losses[:5])
    late_mean = statistics.mean(losses[-5:])
    mean_decreased = late_mean < early_mean
    adapter = adapter_summary()
    baseline_taxonomy = taxonomy(
        validation,
        generations["baseline"],
    )
    post_taxonomy = taxonomy(
        validation,
        generations["post_sft"],
    )
    transitions = transition_summary(
        validation,
        v4_generations["post_sft"],
        generations["post_sft"],
    )
    budget_audit = token_budget_audit(
        tokenizer,
        all_samples,
        generations["post_sft"],
        raw["config"]["generation_max_new_tokens"],
    )
    semantic_target = raw["post_sft_validation"]["semantic_exact"] == 32
    exact_target = raw["post_sft_validation"]["exact"] >= 30
    semantic_improved = (
        raw["post_sft_validation"]["semantic_exact"]
        > raw["baseline_validation"]["semantic_exact"]
    )
    exact_improved = (
        raw["post_sft_validation"]["exact"]
        > raw["baseline_validation"]["exact"]
    )
    adapter_reload = (
        reload["reload_success"]
        and reload["adapter_sha256"] == raw["adapter_sha256"]
        and reload["validation"] == raw["post_sft_validation"]
    )
    passed = (
        finite
        and mean_decreased
        and semantic_target
        and exact_target
        and semantic_improved
        and exact_improved
        and adapter_reload
        and adapter["nonfinite_tensors"] == 0
        and raw["hardware"]["peak_allocated_gib"] < 28
        and not (ARTIFACTS / "failure.json").exists()
    )
    examples_seen = (
        raw["config"]["max_steps"]
        * raw["config"]["batch_size"]
        * raw["config"]["gradient_accumulation_steps"]
    )
    report = {
        "schema_version": "nano_train_public_sft_smoke_v5",
        "experiment_id": raw["experiment_id"],
        "pre_registration_revision": PRE_REGISTRATION_REVISION,
        "passed": passed,
        "identity": {
            "config_sha256": CONFIG_SHA256,
            "dataset_id": raw["dataset"]["dataset_id"],
            "dataset_sha256": raw["dataset"]["sha256"],
            "model_config_sha256": raw["model"]["config_sha256"],
        },
        "configuration": {
            "seed": raw["config"]["seed"],
            "dtype": raw["config"]["dtype"],
            "max_steps": raw["config"]["max_steps"],
            "effective_batch_size": (
                raw["config"]["batch_size"]
                * raw["config"]["gradient_accumulation_steps"]
            ),
            "examples_seen": examples_seen,
            "unique_examples_seen": raw["dataset"]["train_samples"],
            "train_samples": raw["dataset"]["train_samples"],
            "training_coverage_equivalents": (
                examples_seen / raw["dataset"]["train_samples"]
            ),
            "generation_max_new_tokens": raw["config"][
                "generation_max_new_tokens"
            ],
            "learning_rate": raw["config"]["learning_rate"],
            "warmup_steps": raw["config"]["warmup_steps"],
            "lora_r": raw["config"]["lora_r"],
            "lora_alpha": raw["config"]["lora_alpha"],
            "lora_targets": raw["config"]["lora_targets"],
        },
        "dependencies": raw["dependencies"],
        "hardware": raw["hardware"],
        "wall_seconds": raw["wall_seconds"],
        "baseline_validation": raw["baseline_validation"],
        "post_sft_validation": raw["post_sft_validation"],
        "failure_taxonomy": {
            "baseline": baseline_taxonomy,
            "post_sft": post_taxonomy,
        },
        "v4_to_v5_transitions": transitions,
        "generation_budget_audit": budget_audit,
        "optimization": {
            "steps": len(losses),
            "all_losses_finite": finite,
            "early_five_step_mean": early_mean,
            "late_five_step_mean": late_mean,
            "late_mean_below_early_mean": mean_decreased,
            "minimum_loss": min(losses),
            "loss_curve": raw["loss_curve"],
            "failure_receipt_exists": (ARTIFACTS / "failure.json").exists(),
        },
        "adapter_validation": {
            **adapter,
            "reload_success": reload["reload_success"],
            "reload_validation": reload["validation"],
            "reload_peak_allocated_gib": reload["peak_allocated_gib"],
        },
        "artifacts": {
            "metrics_sha256": sha256_file(METRICS),
            "generations_sha256": sha256_file(GENERATIONS),
            "reload_validation_sha256": sha256_file(RELOAD),
            "adapter_sha256": raw["adapter_sha256"],
        },
        "decision": {
            "accepted": passed,
            "numerical_stability_passed": finite
            and adapter["nonfinite_tensors"] == 0,
            "semantic_validation_improved": semantic_improved,
            "exact_validation_improved": exact_improved,
            "semantic_validation_32_of_32": semantic_target,
            "strict_exact_at_least_30_of_32": exact_target,
            "moving_average_loss_decreased": mean_decreased,
            "adapter_reload_passed": adapter_reload,
            "memory_below_28_gib": raw["hardware"]["peak_allocated_gib"] < 28,
            "generation_budget_contract_valid": False,
            "official_score_reinterpreted": False,
            "benchmark_evaluation_allowed": False,
            "rl_allowed": False,
            "next_action": (
                "Preserve v5 as failed evidence. Pre-register an evaluation-"
                "only audit of the unchanged v5 adapter with a generation "
                "budget above the 37-token target maximum; do not retrain or "
                "alter the official 12/32 result."
            ),
        },
    }
    post_counts = post_taxonomy["counts"]
    transition_counts = transitions["counts"]
    val_audit = budget_audit["splits"]["validation"]
    train_audit = budget_audit["splits"]["train"]
    markdown = f"""# Semantic Arithmetic SFT Smoke v5 Result

## Result

V5 is numerically stable and reaches one complete train-set coverage
equivalent, but fails its frozen validation thresholds.

- baseline strict exact: {raw['baseline_validation']['exact']}/32;
- post strict exact: {raw['post_sft_validation']['exact']}/32;
- baseline semantic valid: {raw['baseline_validation']['semantic_exact']}/32;
- post semantic valid: {raw['post_sft_validation']['semantic_exact']}/32;
- early five-step mean loss: {early_mean:.6f};
- late five-step mean loss: {late_mean:.6f};
- peak training memory: {raw['hardware']['peak_allocated_gib']:.2f} GiB;
- wall time: {raw['wall_seconds']:.1f} seconds;
- independent reload semantic valid: {reload['validation']['semantic_exact']}/32.

All {len(losses)} losses and all {adapter['tensor_count']} FP32 adapter tensors
are finite. Independent reload reproduces both metrics exactly.

## Failure Analysis

Post-SFT taxonomy:

- semantic valid: {post_counts.get('semantic_valid', 0)};
- CALC/FINAL mismatch: {post_counts.get('calc_final_mismatch', 0)};
- CALC execution mismatch: {post_counts.get('calc_execution_mismatch', 0)};
- invalid trace grammar: {post_counts.get('invalid_trace_grammar', 0)}.

Relative to v4, three execution mismatches become semantic-valid, one
semantic-valid case regresses to an execution mismatch, and all 13
CALC/FINAL mismatches remain:

- execution mismatch to semantic valid:
  {transition_counts.get('calc_execution_mismatch->semantic_valid', 0)};
- semantic valid to execution mismatch:
  {transition_counts.get('semantic_valid->calc_execution_mismatch', 0)};
- CALC/FINAL mismatch unchanged:
  {transition_counts.get('calc_final_mismatch->calc_final_mismatch', 0)}.

Full coverage therefore adds only two net semantic-valid cases over v4.

## Generation Budget Audit

The frozen 32-token generation budget is shorter than the target contract:

- train target content: max {train_audit['target_content_token_max']} tokens,
  {train_audit['target_content_over_generation_cap']}/160 above the cap;
- validation target content: max
  {val_audit['target_content_token_max']} tokens,
  {val_audit['target_content_over_generation_cap']}/32 above the cap;
- validation post-SFT outputs at the cap:
  {budget_audit['post_sft_output_at_cap']}/32;
- failed validation outputs at the cap:
  {budget_audit['post_sft_failed_output_at_cap']}/32;
- failures whose canonical target is above the cap:
  {budget_audit['post_sft_over_cap_target_failures']}/32.

All 13 CALC/FINAL mismatches are in the over-cap group. Six arithmetic
execution mismatches remain and are not explained by target truncation.
This audit exposes an evaluation-contract defect; it does not rescore v5 or
turn the failed run into a pass.

## Decision

V5 fails because semantic validation is
{raw['post_sft_validation']['semantic_exact']}/32 rather than 32/32 and strict
exact is {raw['post_sft_validation']['exact']}/32 rather than at least 30/32.
Do not benchmark, merge, scale, or start RL.

Preserve the official 12/32 result. Before another training intervention,
pre-register an evaluation-only audit that loads the unchanged adapter and
raises only the generation budget above the 37-token target-content maximum.
The audit must be reported separately and cannot overwrite v5.

## Reproduction Identity

- pre-registration revision: `{PRE_REGISTRATION_REVISION}`;
- config SHA256: `{report['identity']['config_sha256']}`;
- dataset SHA256: `{report['identity']['dataset_sha256']}`;
- model config SHA256: `{report['identity']['model_config_sha256']}`;
- metrics SHA256: `{report['artifacts']['metrics_sha256']}`;
- generations SHA256: `{report['artifacts']['generations_sha256']}`;
- reload receipt SHA256: `{report['artifacts']['reload_validation_sha256']}`;
- adapter tree SHA256: `{report['artifacts']['adapter_sha256']}`.
"""
    output = ROOT / "docs/results"
    output.mkdir(parents=True, exist_ok=True)
    (output / "semantic_arithmetic_sft_smoke_v5.public.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "semantic_arithmetic_sft_smoke_v5.md").write_text(
        markdown,
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "experiment_id": report["experiment_id"],
                "passed": passed,
                "post_exact": raw["post_sft_validation"]["exact"],
                "post_semantic": raw["post_sft_validation"][
                    "semantic_exact"
                ],
                "validation_targets_over_cap": val_audit[
                    "target_content_over_generation_cap"
                ],
                "failed_outputs_at_cap": budget_audit[
                    "post_sft_failed_output_at_cap"
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
