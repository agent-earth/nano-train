#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts/format-contract-sft-smoke-v2"
METRICS = ARTIFACTS / "metrics.json"
GENERATIONS = ARTIFACTS / "generations.json"
RELOAD = ARTIFACTS / "reload_validation.json"
ADAPTER = ARTIFACTS / "adapter"
PRE_REGISTRATION_REVISION = "4468606"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def adapter_finite_summary() -> dict:
    from safetensors import safe_open

    path = ADAPTER / "adapter_model.safetensors"
    total = 0
    nonfinite = 0
    dtypes: dict[str, int] = {}
    with safe_open(path, framework="pt", device="cpu") as handle:
        for key in handle.keys():
            tensor = handle.get_tensor(key)
            total += 1
            dtype = str(tensor.dtype)
            dtypes[dtype] = dtypes.get(dtype, 0) + 1
            if not bool(tensor.isfinite().all()):
                nonfinite += 1
    return {
        "tensor_count": total,
        "nonfinite_tensors": nonfinite,
        "dtype_counts": dict(sorted(dtypes.items())),
    }


def main() -> None:
    raw = json.loads(METRICS.read_text(encoding="utf-8"))
    reload = json.loads(RELOAD.read_text(encoding="utf-8"))
    losses = [float(row["loss"]) for row in raw["loss_curve"]]
    finite = all(math.isfinite(loss) for loss in losses)
    adapter = adapter_finite_summary()
    exact_target = raw["post_sft_validation"]["exact"] == 26
    validation_improved = (
        raw["post_sft_validation"]["exact"]
        > raw["baseline_validation"]["exact"]
    )
    final_loss_below_initial = losses[-1] < losses[0]
    adapter_reload = (
        reload["reload_success"]
        and reload["adapter_sha256"] == raw["adapter_sha256"]
        and reload["validation"] == raw["post_sft_validation"]
    )
    passed = (
        finite
        and final_loss_below_initial
        and exact_target
        and validation_improved
        and adapter_reload
        and adapter["nonfinite_tensors"] == 0
        and raw["hardware"]["peak_allocated_gib"] < 28
        and not (ARTIFACTS / "failure.json").exists()
    )
    report = {
        "schema_version": "nano_train_public_sft_smoke_v2",
        "experiment_id": raw["experiment_id"],
        "pre_registration_revision": PRE_REGISTRATION_REVISION,
        "passed": passed,
        "identity": {
            "config_sha256": (
                "62cc5189cb048fd1a2b4070ffdd27b0a1"
                "8c3363df1ae8dfa244a381401646207"
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
        "optimization": {
            "steps": len(losses),
            "all_losses_finite": finite,
            "initial_loss": losses[0],
            "final_loss": losses[-1],
            "minimum_loss": min(losses),
            "final_loss_below_initial": final_loss_below_initial,
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
            "validation_exact_26_of_26": exact_target,
            "final_loss_below_initial": final_loss_below_initial,
            "adapter_reload_passed": adapter_reload,
            "memory_below_28_gib": raw["hardware"]["peak_allocated_gib"] < 28,
            "benchmark_evaluation_allowed": False,
            "rl_allowed": False,
            "next_action": (
                "Preserve v2 as a stable but incomplete smoke. Diagnose the "
                "remaining two-step semantic error and pre-register a small "
                "data/curriculum ablation before any benchmark evaluation."
            ),
        },
    }
    failed_id = raw["post_sft_validation"]["failure_sample_ids"][0]
    markdown = f"""# Format Contract SFT Smoke v2 Result

## Result

The FP32 numerical repair succeeds, but the pre-registered SFT smoke still
fails its full acceptance rule.

- all 20 optimizer-step losses are finite;
- baseline exact validation: {raw['baseline_validation']['exact']}/26
  ({raw['baseline_validation']['accuracy']:.4f});
- post-SFT exact validation: {raw['post_sft_validation']['exact']}/26
  ({raw['post_sft_validation']['accuracy']:.4f});
- initial loss: {losses[0]:.6f};
- minimum observed loss: {min(losses):.6f};
- final loss: {losses[-1]:.6f};
- peak training memory: {raw['hardware']['peak_allocated_gib']:.2f} GiB;
- adapter reload: 25/26 exact, matching in-process validation.

The adapter contains {adapter['tensor_count']} FP32 LoRA tensors and zero
non-finite tensors. FP32 fixes v1's first-backward instability and improves
validation from 23/26 to 25/26.

## Remaining Failure

Validation does not reach 26/26. The sole failed synthetic sample is
`{failed_id}`, a two-step arithmetic-precedence numeric example. The generated
answer obeys the exact `FINAL:` format but is semantically wrong.

The final sampled batch loss is also above the initial sampled batch loss, so
the frozen loss-decrease condition does not pass. Do not treat minimum loss as
a substitute for the pre-registered final-loss rule.

## Decision

v2 is a numerically stable, reloadable adapter with directional format
improvement, but it is not accepted for benchmark evaluation, merge, scale-up,
or RL. Preserve it as evidence. The next experiment must be a separately
pre-registered data/curriculum ablation targeting the remaining two-step
semantic weakness.

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
    (output / "format_contract_sft_smoke_v2.public.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "format_contract_sft_smoke_v2.md").write_text(
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
                "adapter_reload": adapter_reload,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
