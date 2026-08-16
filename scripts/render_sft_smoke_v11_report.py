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
ARTIFACTS = ROOT / "artifacts/targeted-preservation-sft-smoke-v11"
V10_REPORT = ROOT / "docs/results/hard_preservation_sft_smoke_v10.public.json"
METRICS = ARTIFACTS / "metrics.json"
GENERATIONS = ARTIFACTS / "generations.json"
RELOAD = ARTIFACTS / "reload_validation.json"
ADAPTER = ARTIFACTS / "adapter"
MODEL = ROOT / "../../models/Qwen3.5-4B"
PRE_REGISTRATION_REVISION = "dfdba60"
DATA_REVISION = "ba11804"
CONFIG_SHA256 = (
    "9a971cb46a1f5c21164d6117bef40aed"
    "fcb7170e9e82604bb7400c942a2be593"
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
    v10 = json.loads(V10_REPORT.read_text(encoding="utf-8"))
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
    prior = v10["post_sft_validation"]
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
    family_deltas = {}
    for family in family_targets:
        prior_failures = set(
            prior["by_family"][family]["semantic_failure_sample_ids"]
        )
        post_failures = set(
            post["by_family"][family]["semantic_failure_sample_ids"]
        )
        family_deltas[family] = {
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
    if not passed:
        raise SystemExit("v11 does not satisfy its frozen local gate")

    report = {
        "schema_version": "nano_train_public_sft_smoke_v11",
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
            "examples_seen": 128,
            "unique_examples_seen": 128,
            "targeted_examples_seen": 13,
            "train_samples": raw["dataset"]["train_samples"],
            "training_coverage_equivalents": 0.8,
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
        "versus_v10": {
            "aggregate_exact_delta": post["exact"] - prior["exact"],
            "aggregate_semantic_delta": (
                post["semantic_exact"] - prior["semantic_exact"]
            ),
            "family_deltas": family_deltas,
            "any_semantic_regression": any(
                row["regressed_sample_ids"] for row in family_deltas.values()
            ),
        },
        "development_boundary": {
            "role": "development_gate_only",
            "independent_quality_claim_allowed": False,
            "observed_validation_reused": True,
            "sealed_canary_read_during_training": False,
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
            "accepted_local_smoke": passed,
            "numerical_stability_passed": (
                finite and adapter["nonfinite_tensors"] == 0
            ),
            "aggregate_semantic_at_least_24": aggregate_target,
            "strict_exact_at_least_22": strict_target,
            "family_targets": family_target_checks,
            "every_family_improved_over_base": all(family_improved.values()),
            "family_improvement_checks": family_improved,
            "moving_average_loss_decreased": mean_decreased,
            "zero_capped_outputs": no_capped_outputs,
            "adapter_reload_passed": adapter_reload,
            "memory_below_28_gib": raw["hardware"]["peak_allocated_gib"] < 28,
            "sealed_canary_allowed": True,
            "full_benchmark_allowed": False,
            "merge_allowed": False,
            "scale_allowed": False,
            "rl_allowed": False,
            "next_action": (
                "Run only the sealed 40-case regression canary on the exact "
                "adapter. Passing permits the unchanged adapter to run the "
                "full frozen suite but does not establish quality uplift."
            ),
        },
    }

    numeric = post["by_family"]["capability_preservation_numeric"]
    choice = post["by_family"]["capability_preservation_choice"]
    process = post["by_family"]["semantic_arithmetic_process"]
    markdown = f"""# Targeted Preservation SFT Smoke v11 Result

## Result

V11 passes every frozen local gate.

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

## Data Effect

Relative to v10, aggregate semantic improves 22/32 to 26/32 and strict exact
improves 22/32 to 23/32. Numeric semantic improves 9/16 to 12/16, choice
improves 5/8 to 6/8, and process remains 8/8.

Three prior numeric failures and one prior choice failure are fixed, with zero
new semantic failures. The result supports the diagnosed covariate mechanism,
but the split informed the data intervention and remains development evidence
only.

## Decision

The exact adapter may run the sealed 40-case regression canary. Passing that
canary permits the unchanged adapter to run the full frozen suite but does not
establish quality uplift. Merge, scale-up, and RL remain forbidden.

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
    (output / "targeted_preservation_sft_smoke_v11.public.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "targeted_preservation_sft_smoke_v11.md").write_text(
        markdown,
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "experiment_id": raw["experiment_id"],
                "passed": passed,
                "post_exact": post["exact"],
                "post_semantic": post["semantic_exact"],
                "numeric_semantic": numeric["semantic_exact"],
                "choice_semantic": choice["semantic_exact"],
                "process_semantic": process["semantic_exact"],
                "sealed_canary_allowed": True,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
