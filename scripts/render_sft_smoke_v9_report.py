#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts/hard-preservation-sft-smoke-v9"
V7_REPORT = ROOT / "docs/results/hard_preservation_sft_smoke_v7.public.json"
V8_REPORT = ROOT / "docs/results/hard_preservation_sft_smoke_v8.public.json"
METRICS = ARTIFACTS / "metrics.json"
GENERATIONS = ARTIFACTS / "generations.json"
RELOAD = ARTIFACTS / "reload_validation.json"
ADAPTER = ARTIFACTS / "adapter"
MODEL = ROOT / "../../models/Qwen3.5-4B"
PRE_REGISTRATION_REVISION = "6004a3f"
DATA_REVISION = "204b053"
CONFIG_SHA256 = (
    "b83a98b345a96971207c4883db5070242"
    "83a1c7ea073b853bc7e30c6f28ff7f1"
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
    v7 = json.loads(V7_REPORT.read_text(encoding="utf-8"))
    v8 = json.loads(V8_REPORT.read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    losses = [float(row["loss"]) for row in raw["loss_curve"]]
    finite = all(math.isfinite(loss) for loss in losses)
    early_mean = statistics.mean(losses[:5])
    late_mean = statistics.mean(losses[-5:])
    mean_decreased = late_mean < early_mean
    adapter = adapter_summary()
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
    dose_curve = {}
    for label, steps, public in (
        ("v7", 20, v7),
        ("v9", 30, {"post_sft_validation": post}),
        ("v8", 40, v8),
    ):
        metrics = public["post_sft_validation"]
        dose_curve[label] = {
            "max_steps": steps,
            "aggregate_exact": metrics["exact"],
            "aggregate_semantic": metrics["semantic_exact"],
            "family_semantic": {
                family: metrics["by_family"][family]["semantic_exact"]
                for family in family_targets
            },
        }
    dose_curve["only_max_steps_changed"] = True
    report = {
        "schema_version": "nano_train_public_sft_smoke_v9",
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
            "examples_seen": 120,
            "unique_examples_seen": 120,
            "train_samples": raw["dataset"]["train_samples"],
            "training_coverage_equivalents": 0.75,
            "family_examples_seen": {
                "capability_preservation_numeric": 63,
                "capability_preservation_choice": 27,
                "semantic_arithmetic_process": 30,
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
        "dose_curve": dose_curve,
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
                "Preserve v9 as the best observed max-steps dose. Do not run "
                "the sealed canary because aggregate and numeric gates fail. "
                "Pre-register a 32-step interpolation with all data and gates "
                "frozen."
            ),
        },
    }
    numeric = post["by_family"]["capability_preservation_numeric"]
    choice = post["by_family"]["capability_preservation_choice"]
    process = post["by_family"]["semantic_arithmetic_process"]
    markdown = f"""# Hard Preservation SFT Smoke v9 Result

## Result

V9 is stable and is the best aggregate dose tested, but fails the unchanged
local gate.

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
are finite.

## Dose Ablation

| Run | Steps | Aggregate | Numeric | Choice | Process |
| --- | ---: | ---: | ---: | ---: | ---: |
| v7 | 20 | {dose_curve['v7']['aggregate_semantic']}/32 | {dose_curve['v7']['family_semantic']['capability_preservation_numeric']}/16 | {dose_curve['v7']['family_semantic']['capability_preservation_choice']}/8 | {dose_curve['v7']['family_semantic']['semantic_arithmetic_process']}/8 |
| v9 | 30 | {dose_curve['v9']['aggregate_semantic']}/32 | {dose_curve['v9']['family_semantic']['capability_preservation_numeric']}/16 | {dose_curve['v9']['family_semantic']['capability_preservation_choice']}/8 | {dose_curve['v9']['family_semantic']['semantic_arithmetic_process']}/8 |
| v8 | 40 | {dose_curve['v8']['aggregate_semantic']}/32 | {dose_curve['v8']['family_semantic']['capability_preservation_numeric']}/16 | {dose_curve['v8']['family_semantic']['capability_preservation_choice']}/8 | {dose_curve['v8']['family_semantic']['semantic_arithmetic_process']}/8 |

Thirty steps preserve choice 5/8 while matching v8 numeric 9/16. It is Pareto
better than both endpoints, but remains below promotion.

## Decision

V9 reaches strict 22/32, choice 5/8, and process 8/8, but fails aggregate
24/32 and numeric 10/16. Do not run the sealed canary. Do not run the full
suite, merge, scale up, or start RL.

The next separately pre-registered interpolation may use 32 steps while
freezing data, generation budget, model, LoRA, and all gates.

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
    (output / "hard_preservation_sft_smoke_v9.public.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "hard_preservation_sft_smoke_v9.md").write_text(
        markdown,
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "experiment_id": raw["experiment_id"],
                "passed": passed,
                "post_semantic": post["semantic_exact"],
                "numeric_semantic": numeric["semantic_exact"],
                "choice_semantic": choice["semantic_exact"],
                "process_semantic": process["semantic_exact"],
                "dose_curve": dose_curve,
                "sealed_canary_allowed": passed,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
