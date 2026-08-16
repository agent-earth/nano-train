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
ARTIFACTS = ROOT / "artifacts/semantic-arithmetic-sft-smoke-v4"
METRICS = ARTIFACTS / "metrics.json"
GENERATIONS = ARTIFACTS / "generations.json"
RELOAD = ARTIFACTS / "reload_validation.json"
ADAPTER = ARTIFACTS / "adapter"
DATASET = (
    ROOT.parent
    / "nano-data-pipeline/datasets/verified_semantic_arithmetic_traces_v3.json"
)
MODEL = ROOT / "../../models/Qwen3.5-4B"
PRE_REGISTRATION_REVISION = "0fdec2b"
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


def failure_taxonomy() -> dict:
    generations = json.loads(GENERATIONS.read_text(encoding="utf-8"))
    dataset = json.loads(DATASET.read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    samples = {
        sample.sample_id: sample
        for sample in tokenize_samples(dataset, tokenizer, max_length=128)
        if sample.split == "validation"
    }
    result = {}
    for phase in ("baseline", "post_sft"):
        counts: Counter[str] = Counter()
        ids: defaultdict[str, list[str]] = defaultdict(list)
        for row in generations[phase]:
            sample = samples[row["sample_id"]]
            output = str(row["output"]).strip()
            match = TRACE_PATTERN.fullmatch(output)
            if row["semantic_valid"]:
                category = "semantic_valid"
            elif match is None:
                category = "invalid_trace_grammar"
            else:
                expression, calc_result, final_result = match.groups()
                try:
                    verified = format_number(evaluate_arithmetic(expression))
                except (
                    SyntaxError,
                    ValueError,
                    ZeroDivisionError,
                    OverflowError,
                ):
                    category = "unsafe_or_invalid_expression"
                else:
                    expected = str(sample.verifier["expected_result"])
                    if calc_result != final_result:
                        category = "calc_final_mismatch"
                    elif verified != calc_result:
                        category = "calc_execution_mismatch"
                    elif calc_result != expected:
                        category = "verified_but_wrong_result"
                    else:
                        category = "other"
            counts[category] += 1
            ids[category].append(row["sample_id"])
        result[phase] = {
            "counts": dict(sorted(counts.items())),
            "case_ids": dict(sorted(ids.items())),
        }
    return result


def main() -> None:
    raw = json.loads(METRICS.read_text(encoding="utf-8"))
    reload = json.loads(RELOAD.read_text(encoding="utf-8"))
    losses = [float(row["loss"]) for row in raw["loss_curve"]]
    finite = all(math.isfinite(loss) for loss in losses)
    early_mean = statistics.mean(losses[:5])
    late_mean = statistics.mean(losses[-5:])
    mean_decreased = late_mean < early_mean
    adapter = adapter_summary()
    taxonomy = failure_taxonomy()
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
        "schema_version": "nano_train_public_sft_smoke_v4",
        "experiment_id": raw["experiment_id"],
        "pre_registration_revision": PRE_REGISTRATION_REVISION,
        "passed": passed,
        "identity": {
            "config_sha256": (
                "a162cc982896b16d5f3f1bdb79ba455f"
                "24b629ec95cc149b289e90e0b6ffab04"
            ),
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
            "train_samples": raw["dataset"]["train_samples"],
            "training_coverage_equivalents": (
                examples_seen / raw["dataset"]["train_samples"]
            ),
            "generation_max_new_tokens": raw["config"][
                "generation_max_new_tokens"
            ],
            "learning_rate": raw["config"]["learning_rate"],
            "lora_r": raw["config"]["lora_r"],
            "lora_alpha": raw["config"]["lora_alpha"],
            "lora_targets": raw["config"]["lora_targets"],
        },
        "dependencies": raw["dependencies"],
        "hardware": raw["hardware"],
        "baseline_validation": raw["baseline_validation"],
        "post_sft_validation": raw["post_sft_validation"],
        "failure_taxonomy": taxonomy,
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
            "benchmark_evaluation_allowed": False,
            "rl_allowed": False,
            "next_action": (
                "Preserve v4 as directional semantic evidence. Pre-register "
                "one full-coverage 40-step FP32 trace SFT because v4 exposed "
                "only 80 examples for 160 train samples; do not change data."
            ),
        },
    }
    post_counts = taxonomy["post_sft"]["counts"]
    markdown = f"""# Semantic Arithmetic SFT Smoke v4 Result

## Result

V4 is numerically stable and improves verified semantic accuracy, but fails
its pre-registered validation thresholds.

- baseline strict exact: {raw['baseline_validation']['exact']}/32;
- post strict exact: {raw['post_sft_validation']['exact']}/32;
- baseline semantic valid: {raw['baseline_validation']['semantic_exact']}/32;
- post semantic valid: {raw['post_sft_validation']['semantic_exact']}/32;
- early five-step mean loss: {early_mean:.6f};
- late five-step mean loss: {late_mean:.6f};
- peak training memory: {raw['hardware']['peak_allocated_gib']:.2f} GiB;
- independent reload semantic valid: {reload['validation']['semantic_exact']}/32.

All {adapter['tensor_count']} FP32 adapter tensors are finite. Semantic
validation improves by six cases and strict exact improves by seven.

## Remaining Failure

Post-SFT taxonomy:

- semantic valid: {post_counts.get('semantic_valid', 0)};
- CALC/FINAL mismatch: {post_counts.get('calc_final_mismatch', 0)};
- CALC execution mismatch: {post_counts.get('calc_execution_mismatch', 0)};
- invalid trace grammar: {post_counts.get('invalid_trace_grammar', 0)}.

V4 exposes only {examples_seen} training examples for
{raw['dataset']['train_samples']} train samples, or
{examples_seen / raw['dataset']['train_samples']:.2f} epoch equivalents. It
does not cover one full pass through the dataset.

## Decision

V4 fails because semantic validation is
{raw['post_sft_validation']['semantic_exact']}/32 rather than 32/32 and strict
exact is {raw['post_sft_validation']['exact']}/32 rather than at least 30/32.
Do not benchmark, merge, scale, or start RL.

The next experiment may change only optimizer-step count from 20 to 40 so the
effective batch exposes exactly 160 examples, one train-set equivalent.
Data, validation, FP32, seed, LoRA, LR, and all gates must remain frozen.

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
    (output / "semantic_arithmetic_sft_smoke_v4.public.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "semantic_arithmetic_sft_smoke_v4.md").write_text(
        markdown,
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "experiment_id": report["experiment_id"],
                "passed": passed,
                "baseline_semantic": raw["baseline_validation"][
                    "semantic_exact"
                ],
                "post_semantic": raw["post_sft_validation"]["semantic_exact"],
                "baseline_exact": raw["baseline_validation"]["exact"],
                "post_exact": raw["post_sft_validation"]["exact"],
                "examples_seen": examples_seen,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
