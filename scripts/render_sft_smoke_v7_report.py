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


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts/hard-preservation-sft-smoke-v7"
METRICS = ARTIFACTS / "metrics.json"
GENERATIONS = ARTIFACTS / "generations.json"
RELOAD = ARTIFACTS / "reload_validation.json"
ADAPTER = ARTIFACTS / "adapter"
DATASET = (
    ROOT.parent
    / "nano-data-pipeline/datasets/hard_preservation_mix_v5.json"
)
MODEL = ROOT / "../../models/Qwen3.5-4B"
PRE_REGISTRATION_REVISION = "f5344de"
DATA_REVISION = "204b053"
CONFIG_SHA256 = (
    "787649d577e3978311c968b9d886ae21"
    "88a2a6ff9fcc7f6c00e79f5bfb896c08"
)
FINAL_NUMERIC = re.compile(
    r"FINAL\s*:?\s*\n?\s*([-+]?(?:\d[\d,]*\.?\d*|\.\d+))",
    re.IGNORECASE,
)
FINAL_CHOICE = re.compile(r"FINAL\s*:?\s*([A-D])", re.IGNORECASE)


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


def loose_final_diagnostic(dataset: dict, rows: list[dict]) -> dict:
    by_id = {row["sample_id"]: row for row in dataset["samples"]}
    result = {}
    for family in sorted({str(row["task_family"]) for row in rows}):
        subset = [row for row in rows if row["task_family"] == family]
        pattern = (
            FINAL_CHOICE
            if family == "capability_preservation_choice"
            else FINAL_NUMERIC
        )
        correct = []
        wrong = []
        missing = []
        for row in subset:
            target = by_id[row["sample_id"]]["messages"][-1]["content"]
            output_match = pattern.search(str(row["output"]))
            target_match = pattern.search(target)
            if output_match is None or target_match is None:
                missing.append(row["sample_id"])
                continue
            output_value = output_match.group(1).replace(",", "").upper()
            target_value = target_match.group(1).replace(",", "").upper()
            if output_value == target_value:
                correct.append(row["sample_id"])
            else:
                wrong.append(row["sample_id"])
        result[family] = {
            "samples": len(subset),
            "official_semantic": sum(
                row["semantic_valid"] for row in subset
            ),
            "loose_final_correct": len(correct),
            "loose_final_wrong": len(wrong),
            "missing_final": len(missing),
            "case_ids": {
                "loose_final_correct": sorted(correct),
                "loose_final_wrong": sorted(wrong),
                "missing_final": sorted(missing),
            },
        }
    return result


def main() -> None:
    raw = json.loads(METRICS.read_text(encoding="utf-8"))
    generations = json.loads(GENERATIONS.read_text(encoding="utf-8"))
    reload = json.loads(RELOAD.read_text(encoding="utf-8"))
    dataset = json.loads(DATASET.read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    losses = [float(row["loss"]) for row in raw["loss_curve"]]
    finite = all(math.isfinite(loss) for loss in losses)
    early_mean = statistics.mean(losses[:5])
    late_mean = statistics.mean(losses[-5:])
    mean_decreased = late_mean < early_mean
    adapter = adapter_summary()
    budget = int(raw["config"]["generation_max_new_tokens"])
    budget_audit = {
        "generation_max_new_tokens": budget,
        "baseline": output_budget(
            tokenizer,
            generations["baseline"],
            budget,
        ),
        "post_sft": output_budget(
            tokenizer,
            generations["post_sft"],
            budget,
        ),
    }
    diagnostic = loose_final_diagnostic(
        dataset,
        generations["post_sft"],
    )
    baseline = raw["baseline_validation"]
    post = raw["post_sft_validation"]
    family_targets = {
        "capability_preservation_numeric": 10,
        "capability_preservation_choice": 5,
        "semantic_arithmetic_process": 7,
    }
    family_target_checks = {
        family: post["by_family"][family]["semantic_exact"] >= target
        for family, target in family_targets.items()
    }
    family_improved = {
        family: (
            post["by_family"][family]["semantic_exact"]
            > baseline["by_family"][family]["semantic_exact"]
        )
        for family in family_targets
    }
    aggregate_target = post["semantic_exact"] >= 24
    strict_target = post["exact"] >= 22
    no_capped_outputs = (
        budget_audit["post_sft"]["outputs_at_generation_cap"] == 0
    )
    adapter_reload = (
        reload["reload_success"]
        and reload["adapter_sha256"] == raw["adapter_sha256"]
        and reload["validation"] == post
    )
    passed = (
        finite
        and mean_decreased
        and aggregate_target
        and strict_target
        and all(family_target_checks.values())
        and all(family_improved.values())
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
        "schema_version": "nano_train_public_sft_smoke_v7",
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
            "unique_examples_seen": examples_seen,
            "train_samples": raw["dataset"]["train_samples"],
            "training_coverage_equivalents": (
                examples_seen / raw["dataset"]["train_samples"]
            ),
            "generation_max_new_tokens": budget,
            "learning_rate": raw["config"]["learning_rate"],
            "warmup_steps": raw["config"]["warmup_steps"],
            "lora_r": raw["config"]["lora_r"],
            "lora_alpha": raw["config"]["lora_alpha"],
            "lora_targets": raw["config"]["lora_targets"],
        },
        "dependencies": raw["dependencies"],
        "hardware": raw["hardware"],
        "wall_seconds": raw["wall_seconds"],
        "baseline_validation": baseline,
        "post_sft_validation": post,
        "loose_final_diagnostic": diagnostic,
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
            "accepted_local_smoke": passed,
            "numerical_stability_passed": finite
            and adapter["nonfinite_tensors"] == 0,
            "aggregate_semantic_at_least_24": aggregate_target,
            "strict_exact_at_least_22": strict_target,
            "family_targets": family_target_checks,
            "every_family_improved": all(family_improved.values()),
            "family_improvement_checks": family_improved,
            "moving_average_loss_decreased": mean_decreased,
            "zero_capped_outputs": no_capped_outputs,
            "adapter_reload_passed": adapter_reload,
            "memory_below_28_gib": raw["hardware"]["peak_allocated_gib"] < 28,
            "sealed_canary_allowed": passed,
            "full_benchmark_allowed": False,
            "merge_allowed": False,
            "scale_allowed": False,
            "rl_allowed": False,
            "next_action": (
                "Preserve v7 as stable directional evidence. Do not run the "
                "sealed canary because aggregate, strict, and numeric family "
                "gates fail. Pre-register the next numeric-focused data-dose "
                "ablation without changing choice/process strata."
            ),
        },
    }
    numeric = post["by_family"]["capability_preservation_numeric"]
    choice = post["by_family"]["capability_preservation_choice"]
    process = post["by_family"]["semantic_arithmetic_process"]
    markdown = f"""# Hard Preservation SFT Smoke v7 Result

## Result

V7 is numerically stable and improves every family, but fails its frozen local
promotion thresholds.

- aggregate exact / semantic:
  {baseline['exact']}/32 / {baseline['semantic_exact']}/32 to
  {post['exact']}/32 / {post['semantic_exact']}/32;
- numeric exact / semantic:
  {baseline['by_family']['capability_preservation_numeric']['exact']}/16 /
  {baseline['by_family']['capability_preservation_numeric']['semantic_exact']}/16
  to {numeric['exact']}/16 / {numeric['semantic_exact']}/16;
- choice exact / semantic:
  {baseline['by_family']['capability_preservation_choice']['exact']}/8 /
  {baseline['by_family']['capability_preservation_choice']['semantic_exact']}/8
  to {choice['exact']}/8 / {choice['semantic_exact']}/8;
- process exact / semantic:
  {baseline['by_family']['semantic_arithmetic_process']['exact']}/8 /
  {baseline['by_family']['semantic_arithmetic_process']['semantic_exact']}/8
  to {process['exact']}/8 / {process['semantic_exact']}/8;
- early / late five-step mean loss:
  {early_mean:.6f} / {late_mean:.6f};
- peak training memory: {raw['hardware']['peak_allocated_gib']:.2f} GiB;
- independent reload aggregate exact / semantic:
  {reload['validation']['exact']}/32 /
  {reload['validation']['semantic_exact']}/32;
- post outputs at the 128-token cap:
  {budget_audit['post_sft']['outputs_at_generation_cap']}/32.

All {len(losses)} losses and all {adapter['tensor_count']} FP32 adapter tensors
are finite.

## Failure Analysis

V7 reaches the choice threshold 5/8 and process threshold 8/8. Numeric reaches
only 6/16 rather than the required 10/16. The post-SFT loose-final diagnostic
also reports 6 numeric final answers correct and 10 wrong, so the remaining
numeric gap is semantic, not a format or generation-budget confounder.

Aggregate semantic is {post['semantic_exact']}/32 rather than at least 24/32,
and strict exact is {post['exact']}/32 rather than at least 22/32.

## Decision

V7 fails the local gate. Do not run the sealed 40-case canary. Do not run the
full suite, merge, scale up, or start RL.

The next experiment must be separately pre-registered and target numeric hard
examples or dose while keeping the successful choice and process strata
frozen. V7 is a combined data-plus-dose intervention and cannot support causal
attribution without later ablation.

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
    (output / "hard_preservation_sft_smoke_v7.public.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "hard_preservation_sft_smoke_v7.md").write_text(
        markdown,
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "experiment_id": raw["experiment_id"],
                "passed": passed,
                "baseline_semantic": baseline["semantic_exact"],
                "post_semantic": post["semantic_exact"],
                "numeric_semantic": numeric["semantic_exact"],
                "choice_semantic": choice["semantic_exact"],
                "process_semantic": process["semantic_exact"],
                "sealed_canary_allowed": passed,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
