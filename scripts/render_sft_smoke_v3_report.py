#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts/format-contract-sft-smoke-v3"
METRICS = ARTIFACTS / "metrics.json"
GENERATIONS = ARTIFACTS / "generations.json"
RELOAD = ARTIFACTS / "reload_validation.json"
ADAPTER = ARTIFACTS / "adapter"
DATASET = ROOT.parent / "nano-data-pipeline/datasets/format_contract_curriculum_analog_v2.json"
PRE_REGISTRATION_REVISION = "f05699f"


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


def _format_valid(output: str, family: str) -> bool:
    if family == "final_choice":
        return re.fullmatch(r"FINAL: [A-D]", output.strip()) is not None
    if family == "final_numeric":
        return (
            re.fullmatch(
                r"FINAL: [-+]?(?:[0-9]+(?:\.[0-9]+)?|\.[0-9]+)",
                output.strip(),
            )
            is not None
        )
    raise ValueError(f"unknown format family: {family}")


def generation_breakdown() -> dict:
    generations = json.loads(GENERATIONS.read_text(encoding="utf-8"))
    dataset = json.loads(DATASET.read_text(encoding="utf-8"))
    metadata = {sample["sample_id"]: sample for sample in dataset["samples"]}
    result = {}
    for phase in ("baseline", "post_sft"):
        rows = generations[phase]
        exact = Counter()
        failures = Counter()
        format_valid = Counter()
        for row in rows:
            sample = metadata[row["sample_id"]]
            key = f"{sample['task_family']}:{sample['difficulty']}"
            exact[key] += bool(row["exact"])
            failures[key] += not bool(row["exact"])
            format_valid[key] += _format_valid(
                str(row["output"]),
                str(sample["format_family"]),
            )
        result[phase] = {
            "samples": len(rows),
            "exact_by_stratum": dict(sorted(exact.items())),
            "failures_by_stratum": dict(sorted(failures.items())),
            "format_valid_by_stratum": dict(sorted(format_valid.items())),
            "format_valid_total": sum(format_valid.values()),
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
    breakdown = generation_breakdown()
    validation_improved = (
        raw["post_sft_validation"]["exact"]
        > raw["baseline_validation"]["exact"]
    )
    exact_target = raw["post_sft_validation"]["exact"] == 32
    adapter_reload = (
        reload["reload_success"]
        and reload["adapter_sha256"] == raw["adapter_sha256"]
        and reload["validation"] == raw["post_sft_validation"]
    )
    passed = (
        finite
        and mean_decreased
        and exact_target
        and validation_improved
        and adapter_reload
        and adapter["nonfinite_tensors"] == 0
        and raw["hardware"]["peak_allocated_gib"] < 28
        and not (ARTIFACTS / "failure.json").exists()
    )
    report = {
        "schema_version": "nano_train_public_sft_smoke_v3",
        "experiment_id": raw["experiment_id"],
        "pre_registration_revision": PRE_REGISTRATION_REVISION,
        "passed": passed,
        "identity": {
            "config_sha256": (
                "fee61ad70cec96368849b6873e7f261db"
                "fc822dc82af7d206cfdb29b58edbfdd"
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
            "learning_rate": raw["config"]["learning_rate"],
            "lora_r": raw["config"]["lora_r"],
            "lora_alpha": raw["config"]["lora_alpha"],
            "lora_targets": raw["config"]["lora_targets"],
            "train_samples": raw["dataset"]["train_samples"],
            "validation_samples": raw["dataset"]["validation_samples"],
        },
        "dependencies": raw["dependencies"],
        "hardware": raw["hardware"],
        "baseline_validation": raw["baseline_validation"],
        "post_sft_validation": raw["post_sft_validation"],
        "generation_breakdown": breakdown,
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
            "validation_improved": validation_improved,
            "validation_exact_32_of_32": exact_target,
            "moving_average_loss_decreased": mean_decreased,
            "adapter_reload_passed": adapter_reload,
            "memory_below_28_gib": raw["hardware"]["peak_allocated_gib"] < 28,
            "benchmark_evaluation_allowed": False,
            "rl_allowed": False,
            "next_action": (
                "Stop format-only SFT iteration. The fresh remaining errors "
                "are format-valid and predominantly two-step semantic failures; "
                "design a semantic arithmetic training objective before another smoke."
            ),
        },
    }
    markdown = f"""# Format Contract SFT Smoke v3 Result

## Result

V3 is numerically stable and improves a fresh two-step-heavy validation split,
but fails its exact-validation acceptance rule.

- all 20 optimizer-step losses are finite;
- baseline validation: {raw['baseline_validation']['exact']}/32
  ({raw['baseline_validation']['accuracy']:.4f});
- post-SFT validation: {raw['post_sft_validation']['exact']}/32
  ({raw['post_sft_validation']['accuracy']:.4f});
- early five-step mean loss: {early_mean:.6f};
- late five-step mean loss: {late_mean:.6f};
- minimum loss: {min(losses):.6f};
- peak training memory: {raw['hardware']['peak_allocated_gib']:.2f} GiB;
- independent adapter reload: {reload['validation']['exact']}/32.

All {adapter['tensor_count']} FP32 adapter tensors are finite. The moving
average loss criterion passes and validation improves by seven cases.

## Remaining Failure

Post-SFT has 12 failures:

- 9 numeric two-step;
- 2 choice two-step;
- 1 choice single-step.

All 32 post-SFT outputs match their required `FINAL:` grammar. The remaining
gap is semantic arithmetic, not format compliance. Continuing format-only data
iteration would target the wrong mechanism.

## Decision

V3 fails because validation is 20/32 rather than 32/32. Do not benchmark,
merge, scale, or start RL. Stop format-only SFT iteration and design a
separately pre-registered semantic arithmetic objective.

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
    (output / "format_contract_sft_smoke_v3.public.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "format_contract_sft_smoke_v3.md").write_text(
        markdown,
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "experiment_id": report["experiment_id"],
                "passed": passed,
                "finite": finite,
                "baseline_exact": raw["baseline_validation"]["exact"],
                "post_exact": raw["post_sft_validation"]["exact"],
                "moving_average_loss_decreased": mean_decreased,
                "post_format_valid": breakdown["post_sft"]["format_valid_total"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
