#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts/format-contract-sft-smoke-v1"
METRICS = ARTIFACTS / "metrics.json"
GENERATIONS = ARTIFACTS / "generations.json"
PRE_REGISTRATION_REVISION = "990b695"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    raw = json.loads(METRICS.read_text(encoding="utf-8"))
    losses = raw["loss_curve"]
    first_nonfinite = next(
        (
            row["step"]
            for row in losses
            if row["loss"] is None or not math.isfinite(float(row["loss"]))
        ),
        None,
    )
    finite_losses = [
        {
            "step": row["step"],
            "loss": row["loss"],
            "learning_rate": row["learning_rate"],
        }
        for row in losses
        if row["loss"] is not None and math.isfinite(float(row["loss"]))
    ]
    passed = (
        first_nonfinite is None
        and len(losses) == raw["config"]["max_steps"]
        and finite_losses[-1]["loss"] < finite_losses[0]["loss"]
        and raw["post_sft_validation"]["exact"] == 26
        and (
            raw["post_sft_validation"]["exact"]
            > raw["baseline_validation"]["exact"]
            or raw["baseline_validation"]["exact"] == 26
        )
        and raw["hardware"]["peak_allocated_gib"] < 28
    )
    report = {
        "schema_version": "nano_train_public_sft_smoke_v1",
        "experiment_id": raw["experiment_id"],
        "pre_registration_revision": PRE_REGISTRATION_REVISION,
        "passed": passed,
        "identity": {
            "dataset_id": raw["dataset"]["dataset_id"],
            "dataset_sha256": raw["dataset"]["sha256"],
            "model_config_sha256": raw["model"]["config_sha256"],
            "config_sha256": (
                "09bbf842ea2a335e283385eeea18d352f"
                "9311dc5747e86da9be9b58bfdae2d93"
            ),
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
            "attempted_steps": len(losses),
            "finite_steps": len(finite_losses),
            "first_nonfinite_step": first_nonfinite,
            "finite_loss_curve": finite_losses,
            "runner_fail_fast_enforced": False,
        },
        "artifacts": {
            "metrics_sha256": sha256_file(METRICS),
            "generations_sha256": sha256_file(GENERATIONS),
            "adapter_sha256": raw["adapter_sha256"],
        },
        "decision": {
            "accepted": passed,
            "finite_loss": first_nonfinite is None,
            "loss_decreased": (
                len(finite_losses) >= 2
                and finite_losses[-1]["loss"] < finite_losses[0]["loss"]
            ),
            "validation_exact_26_of_26": (
                raw["post_sft_validation"]["exact"] == 26
            ),
            "memory_below_28_gib": (
                raw["hardware"]["peak_allocated_gib"] < 28
            ),
            "benchmark_evaluation_allowed": False,
            "rl_allowed": False,
            "next_action": (
                "Preserve v1 artifacts, add non-finite fail-fast, diagnose "
                "the first optimizer update, and pre-register a separate v2."
            ),
        },
    }
    markdown = f"""# Format Contract SFT Smoke v1 Result

## Result

The pre-registered SFT smoke fails.

- baseline exact validation: {raw['baseline_validation']['exact']}/26
  ({raw['baseline_validation']['accuracy']:.4f});
- post-SFT exact validation: {raw['post_sft_validation']['exact']}/26
  ({raw['post_sft_validation']['accuracy']:.4f});
- first finite loss: {finite_losses[0]['loss']:.6f};
- first non-finite loss step: {first_nonfinite};
- finite optimizer steps: {len(finite_losses)}/{len(losses)};
- peak allocated memory: {raw['hardware']['peak_allocated_gib']:.2f} GiB.

Loss becomes non-finite at step 2 and remains non-finite. Validation collapses
from 23/26 to 0/26. The saved adapter is invalid and must not be evaluated,
merged, published, or used to start RL.

## Runner Defect

The v1 runner did not stop on non-finite loss. It continued through 20 steps
and saved an invalid adapter. Preserve these artifacts as failure evidence, add
fail-fast before optimizer continuation and artifact acceptance, then diagnose
the first update under a separately pre-registered v2.

The adapter files contain FP32 LoRA tensors, but all saved tensors are
non-finite. This rules out the simple claim that saved adapter weights remained
FP16. The exact instability source remains unresolved.

## Reproduction Identity

- pre-registration revision: `{PRE_REGISTRATION_REVISION}`;
- config SHA256: `{report['identity']['config_sha256']}`;
- dataset SHA256: `{report['identity']['dataset_sha256']}`;
- model config SHA256: `{report['identity']['model_config_sha256']}`;
- metrics SHA256: `{report['artifacts']['metrics_sha256']}`;
- generations SHA256: `{report['artifacts']['generations_sha256']}`;
- adapter tree SHA256: `{report['artifacts']['adapter_sha256']}`.
"""
    output = ROOT / "docs/results"
    output.mkdir(parents=True, exist_ok=True)
    (output / "format_contract_sft_smoke_v1.public.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "format_contract_sft_smoke_v1.md").write_text(
        markdown,
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "experiment_id": report["experiment_id"],
                "passed": passed,
                "first_nonfinite_step": first_nonfinite,
                "baseline_exact": raw["baseline_validation"]["exact"],
                "post_exact": raw["post_sft_validation"]["exact"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
