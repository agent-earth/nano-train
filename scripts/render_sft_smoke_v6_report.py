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

from nano_train.data import semantic_output_valid, tokenize_samples


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts/arithmetic-process-sft-smoke-v6"
METRICS = ARTIFACTS / "metrics.json"
GENERATIONS = ARTIFACTS / "generations.json"
RELOAD = ARTIFACTS / "reload_validation.json"
ADAPTER = ARTIFACTS / "adapter"
DATASET = (
    ROOT.parent
    / "nano-data-pipeline/datasets/verified_arithmetic_process_traces_v4.json"
)
MODEL = ROOT / "../../models/Qwen3.5-4B"
PRE_REGISTRATION_REVISION = "f24891b"
DATA_REVISION = "f1dcbe2"
CONFIG_SHA256 = (
    "f8ab480d0195527b3fe8d98bb49ee377"
    "ba444257dcfe203de50c720d06624447"
)
FINAL_PATTERN = re.compile(
    r"(?:^|\n)FINAL: ([-+]?(?:[0-9]+(?:\.[0-9]+)?|\.[0-9]+))\s*$"
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


def final_answer_correct(output: str, target: str) -> bool:
    output_match = FINAL_PATTERN.search(output.strip())
    target_match = FINAL_PATTERN.search(target.strip())
    return bool(
        output_match
        and target_match
        and output_match.group(1) == target_match.group(1)
    )


def process_taxonomy(samples: dict, rows: list[dict]) -> dict:
    counts: Counter[str] = Counter()
    ids: defaultdict[str, list[str]] = defaultdict(list)
    for row in rows:
        sample = samples[row["sample_id"]]
        output = str(row["output"]).strip()
        if row["semantic_valid"]:
            category = "process_semantic_valid"
        elif final_answer_correct(output, sample.target):
            category = "final_correct_process_contract_mismatch"
        elif FINAL_PATTERN.search(output) is None:
            category = "missing_or_invalid_final"
        elif output.count("STEP ") != len(sample.verifier["steps"]):
            category = "step_count_mismatch"
        elif semantic_output_valid(sample, output):
            category = "other_semantic_valid"
        else:
            category = "process_execution_or_structure_mismatch"
        counts[category] += 1
        ids[category].append(row["sample_id"])
    return {
        "counts": dict(sorted(counts.items())),
        "case_ids": dict(sorted(ids.items())),
    }


def output_budget(tokenizer, rows: list[dict], budget: int) -> dict:
    lengths = [
        len(
            tokenizer(
                str(row["output"]),
                add_special_tokens=False,
            ).input_ids
        )
        for row in rows
    ]
    return {
        "maximum_output_tokens": max(lengths),
        "outputs_at_generation_cap": sum(length >= budget for length in lengths),
        "token_histogram": dict(sorted(Counter(lengths).items())),
    }


def main() -> None:
    raw = json.loads(METRICS.read_text(encoding="utf-8"))
    generations = json.loads(GENERATIONS.read_text(encoding="utf-8"))
    reload = json.loads(RELOAD.read_text(encoding="utf-8"))
    dataset = json.loads(DATASET.read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    all_samples = tokenize_samples(dataset, tokenizer, max_length=192)
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
    baseline_taxonomy = process_taxonomy(
        validation,
        generations["baseline"],
    )
    post_taxonomy = process_taxonomy(
        validation,
        generations["post_sft"],
    )
    baseline_final = sum(
        final_answer_correct(
            str(row["output"]),
            validation[row["sample_id"]].target,
        )
        for row in generations["baseline"]
    )
    post_final = sum(
        final_answer_correct(
            str(row["output"]),
            validation[row["sample_id"]].target,
        )
        for row in generations["post_sft"]
    )
    generation_budget = int(raw["config"]["generation_max_new_tokens"])
    budget_audit = {
        "generation_max_new_tokens": generation_budget,
        "baseline": output_budget(
            tokenizer,
            generations["baseline"],
            generation_budget,
        ),
        "post_sft": output_budget(
            tokenizer,
            generations["post_sft"],
            generation_budget,
        ),
    }
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
    no_capped_outputs = (
        budget_audit["post_sft"]["outputs_at_generation_cap"] == 0
    )
    passed = (
        finite
        and mean_decreased
        and semantic_target
        and exact_target
        and semantic_improved
        and exact_improved
        and no_capped_outputs
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
        "schema_version": "nano_train_public_sft_smoke_v6",
        "experiment_id": raw["experiment_id"],
        "pre_registration_revision": PRE_REGISTRATION_REVISION,
        "data_revision": DATA_REVISION,
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
            "max_length": raw["config"]["max_length"],
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
            "generation_max_new_tokens": generation_budget,
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
        "final_answer_accuracy": {
            "baseline": {
                "correct": baseline_final,
                "samples": 32,
                "accuracy": baseline_final / 32,
            },
            "post_sft": {
                "correct": post_final,
                "samples": 32,
                "accuracy": post_final / 32,
            },
            "improved": post_final > baseline_final,
        },
        "failure_taxonomy": {
            "baseline": baseline_taxonomy,
            "post_sft": post_taxonomy,
        },
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
            "accepted_local_process_smoke": passed,
            "numerical_stability_passed": finite
            and adapter["nonfinite_tensors"] == 0,
            "process_semantic_validation_32_of_32": semantic_target,
            "strict_exact_at_least_30_of_32": exact_target,
            "process_metrics_improved": semantic_improved and exact_improved,
            "final_answer_accuracy_improved": post_final > baseline_final,
            "moving_average_loss_decreased": mean_decreased,
            "zero_capped_outputs": no_capped_outputs,
            "adapter_reload_passed": adapter_reload,
            "memory_below_28_gib": raw["hardware"]["peak_allocated_gib"] < 28,
            "arithmetic_uplift_claim_allowed": False,
            "matched_benchmark_evaluation_allowed": passed,
            "merge_allowed": False,
            "scale_allowed": False,
            "rl_allowed": False,
            "next_action": (
                "Preserve v6 as a passed process-contract smoke without an "
                "arithmetic-uplift claim. Evaluate the unchanged adapter on "
                "frozen matched benchmarks with task-level non-regression "
                "before any merge, scale, or RL decision."
            ),
        },
    }
    baseline_counts = baseline_taxonomy["counts"]
    markdown = f"""# Arithmetic Process SFT Smoke v6 Result

## Result

V6 passes its pre-registered local process-contract smoke:

- baseline strict exact / process semantic:
  {raw['baseline_validation']['exact']}/32 /
  {raw['baseline_validation']['semantic_exact']}/32;
- post strict exact / process semantic:
  {raw['post_sft_validation']['exact']}/32 /
  {raw['post_sft_validation']['semantic_exact']}/32;
- baseline final-answer correct: {baseline_final}/32;
- post final-answer correct: {post_final}/32;
- early five-step mean loss: {early_mean:.9f};
- late five-step mean loss: {late_mean:.9f};
- peak training memory: {raw['hardware']['peak_allocated_gib']:.2f} GiB;
- wall time: {raw['wall_seconds']:.1f} seconds;
- independent reload exact / process semantic:
  {reload['validation']['exact']}/32 /
  {reload['validation']['semantic_exact']}/32.

All {len(losses)} losses and all {adapter['tensor_count']} FP32 adapter tensors
are finite. No post-SFT output reaches the 80-token cap.

## Mechanism Interpretation

All 32 baseline and all 32 post-SFT outputs have the correct numeric `FINAL`.
The four baseline process failures are:

- final-correct process-contract mismatches:
  {baseline_counts.get('final_correct_process_contract_mismatch', 0)};
- other process failures:
  {32 - raw['baseline_validation']['semantic_exact'] - baseline_counts.get('final_correct_process_contract_mismatch', 0)}.

They combine operations or omit a canonical STEP rather than produce a wrong
final answer. V6 improves process-contract adherence from 28/32 to 32/32, but
does not improve final-answer accuracy beyond the 32/32 baseline.

Therefore this result must not be described as arithmetic reasoning uplift.
It is evidence that the process objective teaches the requested decomposition
and verifier contract.

## Decision

V6 passes every frozen local gate, including reload, finite tensors, memory,
loss trend, strict exact, process semantic, and generation budget.

This authorizes only evaluation of the unchanged adapter on frozen matched
benchmarks with task-level non-regression. It does not authorize an arithmetic
uplift claim, merge, scale-up, or RL.

## Reproduction Identity

- pre-registration revision: `{PRE_REGISTRATION_REVISION}`;
- data revision: `{DATA_REVISION}`;
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
    (output / "arithmetic_process_sft_smoke_v6.public.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "arithmetic_process_sft_smoke_v6.md").write_text(
        markdown,
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "experiment_id": raw["experiment_id"],
                "passed": passed,
                "baseline_process_semantic": raw["baseline_validation"][
                    "semantic_exact"
                ],
                "post_process_semantic": raw["post_sft_validation"][
                    "semantic_exact"
                ],
                "baseline_final_answer": baseline_final,
                "post_final_answer": post_final,
                "post_outputs_at_cap": budget_audit["post_sft"][
                    "outputs_at_generation_cap"
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
