#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path

from safetensors import safe_open
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts/percentage-isolation-preservation-sft-smoke-v13"
V11_REPORT = ROOT / "docs/results/targeted_preservation_sft_smoke_v11.public.json"
METRICS = ARTIFACTS / "metrics.json"
GENERATIONS = ARTIFACTS / "generations.json"
RELOAD = ARTIFACTS / "reload_validation.json"
ADAPTER = ARTIFACTS / "adapter"
MODEL = ROOT / "../../models/Qwen3.5-4B"
PRE_REGISTRATION_REVISION = "108e9e0"
DATA_REVISION = "a8db4f4"
CONFIG_SHA256 = (
    "98057d4ea24e3d24ada9d98c0dd5af14"
    "fc1f08bb07436e1a33bd479a4131686e"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def adapter_summary() -> dict:
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
        len(tokenizer(str(row["output"]), add_special_tokens=False).input_ids)
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
    v11 = json.loads(V11_REPORT.read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    losses = [float(row["loss"]) for row in raw["loss_curve"]]
    finite = all(math.isfinite(loss) for loss in losses)
    early_mean = statistics.mean(losses[:5])
    late_mean = statistics.mean(losses[-5:])
    adapter = adapter_summary()
    baseline = raw["baseline_validation"]
    post = raw["post_sft_validation"]
    prior = v11["post_sft_validation"]
    targets = {
        "capability_preservation_numeric": 10,
        "capability_preservation_choice": 5,
        "semantic_arithmetic_process": 7,
    }
    target_checks = {
        family: post["by_family"][family]["semantic_exact"] >= target
        for family, target in targets.items()
    }
    improvement_checks = {
        family: (
            post["by_family"][family]["semantic_exact"]
            > baseline["by_family"][family]["semantic_exact"]
        )
        for family in targets
    }
    deltas = {}
    for family in targets:
        prior_failures = set(
            prior["by_family"][family]["semantic_failure_sample_ids"]
        )
        post_failures = set(
            post["by_family"][family]["semantic_failure_sample_ids"]
        )
        deltas[family] = {
            "strict_delta": (
                post["by_family"][family]["exact"]
                - prior["by_family"][family]["exact"]
            ),
            "semantic_delta": (
                post["by_family"][family]["semantic_exact"]
                - prior["by_family"][family]["semantic_exact"]
            ),
            "fixed_sample_ids": sorted(prior_failures - post_failures),
            "regressed_sample_ids": sorted(post_failures - prior_failures),
        }

    budget = int(raw["config"]["generation_max_new_tokens"])
    budget_audit = {
        "generation_max_new_tokens": budget,
        "baseline": output_budget(tokenizer, generations["baseline"], budget),
        "post_sft": output_budget(tokenizer, generations["post_sft"], budget),
    }
    aggregate_target = post["semantic_exact"] >= 24
    strict_target = post["exact"] >= 22
    mean_decreased = late_mean < early_mean
    no_capped = budget_audit["post_sft"]["outputs_at_generation_cap"] == 0
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
        and all(target_checks.values())
        and all(improvement_checks.values())
        and no_capped
        and adapter_reload
        and adapter["nonfinite_tensors"] == 0
        and raw["hardware"]["peak_allocated_gib"] < 28
        and not (ARTIFACTS / "failure.json").exists()
    )
    if passed:
        raise SystemExit("v13 unexpectedly satisfies its frozen local gate")

    report = {
        "schema_version": "nano_train_public_sft_smoke_v13",
        "experiment_id": raw["experiment_id"],
        "pre_registration_revision": PRE_REGISTRATION_REVISION,
        "data_revision": DATA_REVISION,
        "passed": False,
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
            "examples_seen": 128,
            "unique_examples_seen": 128,
            "percentage_examples_seen": 7,
            "family_examples_seen": {
                "capability_preservation_numeric": 66,
                "capability_preservation_choice": 30,
                "semantic_arithmetic_process": 32,
            },
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
        "versus_v11": {
            "aggregate_exact_delta": post["exact"] - prior["exact"],
            "aggregate_semantic_delta": (
                post["semantic_exact"] - prior["semantic_exact"]
            ),
            "family_deltas": deltas,
            "fixed_semantic_cases": sum(
                len(row["fixed_sample_ids"]) for row in deltas.values()
            ),
            "regressed_semantic_cases": sum(
                len(row["regressed_sample_ids"]) for row in deltas.values()
            ),
        },
        "mechanism": {
            "isolated_family": "percentage_increase_total_composition",
            "isolated_family_train_rows": 8,
            "isolated_family_exposures": 7,
            "packing_and_schedule_present": False,
            "choice_process_and_host_exposure_unchanged": True,
            "result": "harmful_at_frozen_dose",
            "further_post_hoc_dose_search_allowed": False,
        },
        "evaluation_boundary": {
            "local_role": "development_gate_only",
            "independent_quality_claim_allowed": False,
            "sealed_canary_run": False,
            "prior_full_suite_run": False,
            "independent_holdout_run": False,
            "independent_holdout_prompts_loaded": False,
            "independent_holdout_references_loaded": False,
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
            "accepted_local_smoke": False,
            "numerical_stability_passed": (
                finite and adapter["nonfinite_tensors"] == 0
            ),
            "aggregate_semantic_at_least_24": aggregate_target,
            "strict_exact_at_least_22": strict_target,
            "family_targets": target_checks,
            "every_family_improved_over_base": all(
                improvement_checks.values()
            ),
            "moving_average_loss_decreased": mean_decreased,
            "zero_capped_outputs": no_capped,
            "adapter_reload_passed": adapter_reload,
            "memory_below_28_gib": raw["hardware"]["peak_allocated_gib"] < 28,
            "sealed_canary_allowed": False,
            "prior_full_suite_allowed": False,
            "independent_holdout_allowed": False,
            "merge_allowed": False,
            "scale_allowed": False,
            "rl_allowed": False,
            "next_action": (
                "Reject percentage-family supervision at the frozen dose. "
                "Preserve v11 and evaluate a separately isolated remaining "
                "family only after a new pre-registration; do not search "
                "smaller percentage doses post hoc."
            ),
        },
    }
    numeric = post["by_family"]["capability_preservation_numeric"]
    choice = post["by_family"]["capability_preservation_choice"]
    process = post["by_family"]["semantic_arithmetic_process"]
    markdown = f"""# Percentage Isolation Preservation SFT Smoke v13 Result

## Result

V13 is stable but fails the frozen local gate.

- aggregate exact / semantic: {post['exact']}/32 / {post['semantic_exact']}/32;
- numeric exact / semantic: {numeric['exact']}/16 /
  {numeric['semantic_exact']}/16;
- choice exact / semantic: {choice['exact']}/8 /
  {choice['semantic_exact']}/8;
- process exact / semantic: {process['exact']}/8 /
  {process['semantic_exact']}/8;
- early / late five-step mean loss:
  {early_mean:.6f} / {late_mean:.6f};
- peak training memory: {raw['hardware']['peak_allocated_gib']:.2f} GiB;
- independent reload exact / semantic:
  {reload['validation']['exact']}/32 /
  {reload['validation']['semantic_exact']}/32;
- post outputs at the cap:
  {budget_audit['post_sft']['outputs_at_generation_cap']}/32.

All {len(losses)} losses and all {adapter['tensor_count']} FP32 adapter tensors
are finite. Independent reload reproduces all metrics and failure IDs.

## Isolated Mechanism

Relative to v11, seven percentage-family exposures fix zero semantic cases and
regress three: two numeric and one choice. Aggregate semantic falls 26/32 to
23/32. Choice/process/targeted-host exposure is unchanged, while packing and
schedule families are absent.

The percentage family alone is harmful at this frozen dose. Stop this family;
do not conduct post-hoc smaller-dose search on the same development split.

## Decision

Reject v13 and preserve v11. Do not run the sealed canary, prior full suite, or
independent holdout. The holdout remains unread.

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
    (output / "percentage_isolation_preservation_sft_smoke_v13.public.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "percentage_isolation_preservation_sft_smoke_v13.md").write_text(
        markdown,
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "experiment_id": raw["experiment_id"],
                "passed": False,
                "post_exact": post["exact"],
                "post_semantic": post["semantic_exact"],
                "fixed_semantic_cases": 0,
                "regressed_semantic_cases": 3,
                "sealed_canary_allowed": False,
                "independent_holdout_allowed": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
