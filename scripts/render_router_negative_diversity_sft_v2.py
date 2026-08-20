#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from nano_train.router_negative_diversity import (
    AUDIT_SHA256,
    CONTRACT_SHA256,
    DATASET_CANONICAL_SHA256,
    DATASET_SHA256,
    RELEASE_SHA256,
    load_config,
    verify_data_release,
)
from nano_train.sft import sha256_file, sha256_tree


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/sft/router_negative_diversity_v2.json"
PREREG = (
    ROOT
    / "docs/experiments/qwen35_router_negative_diversity_sft_v2.preregister.json"
)
ARTIFACT = ROOT / "artifacts/qwen35-router-negative-diversity-sft-v2"
METRICS = ARTIFACT / "metrics.json"
GENERATIONS = ARTIFACT / "generations.json"
RELOAD = ARTIFACT / "reload_validation.json"
RELOAD_GENERATIONS = ARTIFACT / "reload_generations.json"
ADAPTER = ARTIFACT / "adapter"
PUBLIC_JSON = (
    ROOT / "docs/results/qwen35_router_negative_diversity_sft_v2.public.json"
)
MARKDOWN = ROOT / "docs/results/qwen35_router_negative_diversity_sft_v2.md"


def git_revision() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()


def committed_preregister_sha256() -> str:
    content = subprocess.check_output(
        [
            "git",
            "show",
            "HEAD:docs/experiments/"
            "qwen35_router_negative_diversity_sft_v2.preregister.json",
        ],
        cwd=ROOT,
    )
    return hashlib.sha256(content).hexdigest()


def subtype_metrics(
    rows: list[dict],
    sample_by_id: dict[str, dict],
) -> dict[str, dict]:
    result = {}
    subtypes = sorted(
        {
            str(row["negative_subtype"])
            for row in sample_by_id.values()
            if row["split"] == "validation"
            and row["route_label"] == "C"
        }
    )
    for subtype in subtypes:
        selected = [
            row
            for row in rows
            if sample_by_id[row["sample_id"]]["negative_subtype"] == subtype
        ]
        result[subtype] = {
            "samples": len(selected),
            "exact": sum(row["exact"] for row in selected),
            "failure_sample_ids": [
                row["sample_id"] for row in selected if not row["exact"]
            ],
        }
    return result


def result_gates(
    baseline: dict,
    post: dict,
    baseline_subtypes: dict[str, dict],
    post_subtypes: dict[str, dict],
    *,
    metrics: dict,
    reload: dict,
    release: dict,
    exposure_ids_exact: bool,
) -> dict[str, bool]:
    return {
        "no_failure_receipt": not (ARTIFACT / "failure.json").exists(),
        "finite_loss_curve": (
            len(metrics["loss_curve"]) == 40
            and all(
                row["loss"] >= 0 and row["loss"] < float("inf")
                for row in metrics["loss_curve"]
            )
        ),
        "adapter_identity_matches": (
            metrics["adapter_sha256"] == sha256_tree(ADAPTER)
        ),
        "actual_exposure_ids_exact": exposure_ids_exact,
        "reload_success": reload["reload_success"] is True,
        "reload_metrics_exact": reload["metrics_exact"] is True,
        "reload_generations_exact": reload["generations_exact"] is True,
        "aggregate_post_exact_gt_baseline": post["exact"] > baseline["exact"],
        "router_a_post_exact_at_least_480_of_512": (
            post["by_family"]["router_a"]["exact"] >= 480
        ),
        "router_b_post_exact_at_least_480_of_512": (
            post["by_family"]["router_b"]["exact"] >= 480
        ),
        "router_c_post_exact_at_least_496_of_512": (
            post["by_family"]["router_c"]["exact"] >= 496
        ),
        "every_c_subtype_post_exact_at_least_60_of_64": all(
            row["exact"] >= 60 and row["samples"] == 64
            for row in post_subtypes.values()
        ),
        "every_label_non_regression": all(
            post["by_family"][family]["exact"]
            >= baseline["by_family"][family]["exact"]
            for family in ("router_a", "router_b", "router_c")
        ),
        "every_c_subtype_non_regression": all(
            post_subtypes[subtype]["exact"]
            >= baseline_subtypes[subtype]["exact"]
            for subtype in post_subtypes
        ),
        "data_release_identity_matches": (
            release["dataset_file_sha256"] == DATASET_SHA256
            and release["dataset_canonical_sha256"]
            == DATASET_CANONICAL_SHA256
            and release["audit_sha256"] == AUDIT_SHA256
            and release["contract_sha256"] == CONTRACT_SHA256
        ),
    }


def build_report() -> dict:
    config = load_config(CONFIG)
    release = verify_data_release(config)
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    metrics = json.loads(METRICS.read_text(encoding="utf-8"))
    generations = json.loads(GENERATIONS.read_text(encoding="utf-8"))
    reload = json.loads(RELOAD.read_text(encoding="utf-8"))
    reload_generations = json.loads(
        RELOAD_GENERATIONS.read_text(encoding="utf-8")
    )
    dataset = json.loads(Path(config.dataset_path).read_text(encoding="utf-8"))
    sample_by_id = {row["sample_id"]: row for row in dataset["samples"]}
    if (
        prereg.get("schema_version")
        != "nano_train_router_negative_diversity_preregister_v2"
        or prereg.get("identity", {}).get("config_sha256")
        != sha256_file(CONFIG)
        or committed_preregister_sha256() != sha256_file(PREREG)
        or metrics.get("schema_version") != "nano_train_sft_smoke_result_v1"
        or metrics.get("experiment_id") != config.experiment_id
        or metrics.get("dataset", {}).get("sha256") != DATASET_SHA256
        or metrics.get("router_negative_diversity_release", {}).get("sha256")
        != RELEASE_SHA256
        or metrics.get("adapter_sha256") != sha256_tree(ADAPTER)
        or metrics.get("generations_sha256") != sha256_file(GENERATIONS)
        or reload.get("adapter_sha256") != metrics.get("adapter_sha256")
        or reload.get("source_generations_sha256") != sha256_file(GENERATIONS)
        or reload.get("reload_generations_sha256")
        != sha256_file(RELOAD_GENERATIONS)
        or reload_generations != generations["post_sft"]
        or reload.get("validation") != metrics.get("post_sft_validation")
    ):
        raise ValueError("router negative diversity result identity differs")
    baseline = metrics["baseline_validation"]
    post = metrics["post_sft_validation"]
    baseline_subtypes = subtype_metrics(
        generations["baseline"], sample_by_id
    )
    post_subtypes = subtype_metrics(generations["post_sft"], sample_by_id)
    exposure_ids = [
        sample_id
        for row in metrics["train_exposure"]
        for sample_id in row["sample_ids"]
    ]
    exposure_ids_exact = (
        len(metrics["train_exposure"]) == 40
        and all(
            len(row["sample_ids"]) == 4
            for row in metrics["train_exposure"]
        )
        and hashlib.sha256(
            "\n".join(exposure_ids).encode()
        ).hexdigest()
        == prereg["data"]["scheduled_exposure_ids_sha256"]
    )
    gates = result_gates(
        baseline,
        post,
        baseline_subtypes,
        post_subtypes,
        metrics=metrics,
        reload=reload,
        release=release,
        exposure_ids_exact=exposure_ids_exact,
    )
    admitted = all(gates.values())
    return {
        "schema_version": "nano_train_router_negative_diversity_public_v2",
        "experiment_id": config.experiment_id,
        "identity": {
            "result_revision": git_revision(),
            "config_sha256": sha256_file(CONFIG),
            "preregister_sha256": sha256_file(PREREG),
            "metrics_sha256": sha256_file(METRICS),
            "generations_sha256": sha256_file(GENERATIONS),
            "reload_sha256": sha256_file(RELOAD),
            "reload_generations_sha256": sha256_file(RELOAD_GENERATIONS),
            "adapter_sha256": metrics["adapter_sha256"],
            "dataset_file_sha256": DATASET_SHA256,
            "dataset_canonical_sha256": DATASET_CANONICAL_SHA256,
            "release_sha256": RELEASE_SHA256,
            "audit_sha256": AUDIT_SHA256,
            "contract_sha256": CONTRACT_SHA256,
        },
        "training": {
            "steps": config.max_steps,
            "effective_batch_size": (
                config.batch_size * config.gradient_accumulation_steps
            ),
            "learning_rate": config.learning_rate,
            "seed": config.seed,
            "dtype": config.dtype,
            "lora_targets": list(config.lora_targets),
            "lora_r": config.lora_r,
            "lora_alpha": config.lora_alpha,
            "trainable_parameters": metrics["model"]["trainable_parameters"],
            "peak_allocated_gib": metrics["hardware"]["peak_allocated_gib"],
            "reload_peak_allocated_gib": reload["peak_allocated_gib"],
            "wall_seconds": metrics["wall_seconds"],
            "first_loss": metrics["loss_curve"][0]["loss"],
            "last_loss": metrics["loss_curve"][-1]["loss"],
        },
        "validation": {
            "samples": baseline["samples"],
            "baseline_exact": baseline["exact"],
            "post_exact": post["exact"],
            "delta": post["exact"] - baseline["exact"],
            "baseline_accuracy": baseline["accuracy"],
            "post_accuracy": post["accuracy"],
            "by_label": {
                family: {
                    "samples": baseline["by_family"][family]["samples"],
                    "baseline_exact": baseline["by_family"][family]["exact"],
                    "post_exact": post["by_family"][family]["exact"],
                    "delta": (
                        post["by_family"][family]["exact"]
                        - baseline["by_family"][family]["exact"]
                    ),
                    "reload_exact": reload["validation"]["by_family"][family][
                        "exact"
                    ],
                }
                for family in ("router_a", "router_b", "router_c")
            },
            "c_by_subtype": {
                subtype: {
                    "samples": baseline_subtypes[subtype]["samples"],
                    "baseline_exact": baseline_subtypes[subtype]["exact"],
                    "post_exact": post_subtypes[subtype]["exact"],
                    "delta": (
                        post_subtypes[subtype]["exact"]
                        - baseline_subtypes[subtype]["exact"]
                    ),
                }
                for subtype in post_subtypes
            },
        },
        "gates": gates,
        "decision": {
            "router_negative_diversity_sft_admitted": admitted,
            "serving_namespace_remap_required": admitted,
            "serving_parity_preregistration_allowed": admitted,
            "fresh_router_integration_allowed": False,
            "benchmark_allowed": False,
            "canary_allowed": False,
            "holdout_allowed": False,
            "rl_allowed": False,
            "further_tuning_or_second_training_run_allowed": False,
            "next_action": (
                "Build a content-identical vLLM namespace remap and "
                "pre-register 1,536-row serving parity before any fresh "
                "integration."
                if admitted
                else
                "Reject this SFT recipe. Do not tune or rerun on the observed "
                "validation set."
            ),
        },
        "artifact_boundary": {
            "adapter_committed": False,
            "raw_generations_committed": False,
            "metrics_committed": False,
            "training_data_contains_benchmark_content": False,
            "integration_rows_or_outputs_used": False,
            "benchmark_accessed": False,
            "canary_accessed": False,
            "holdout_accessed": False,
        },
        "claim_boundary": (
            "This result establishes only local synthetic classification and "
            "subtype behavior. It is not serving, fresh-transfer, benchmark, "
            "canary, holdout, or final model-superiority evidence."
        ),
    }


def render_markdown(report: dict) -> str:
    validation = report["validation"]
    return f"""# Qwen3.5 Router Negative-Diversity SFT v2 Result

## Verdict

- admitted:
  `{str(report['decision']['router_negative_diversity_sft_admitted']).lower()}`;
- baseline: {validation['baseline_exact']}/{validation['samples']};
- post: {validation['post_exact']}/{validation['samples']};
- delta: {validation['delta']:+d};
- reload metrics and all 1,536 outputs: exact.

## Per Label

```json
{json.dumps(validation['by_label'], indent=2, sort_keys=True)}
```

## C Subtypes

```json
{json.dumps(validation['c_by_subtype'], indent=2, sort_keys=True)}
```

## Training

- 40 steps, effective batch 4;
- FP32 expanded LoRA, r=8, alpha=16;
- loss: {report['training']['first_loss']:.6f} ->
  {report['training']['last_loss']:.6f};
- train/reload peak:
  {report['training']['peak_allocated_gib']:.2f}/
  {report['training']['reload_peak_allocated_gib']:.2f} GiB;
- wall: {report['training']['wall_seconds']:.1f}s.

## Gates

```json
{json.dumps(report['gates'], indent=2, sort_keys=True)}
```

## Boundary

Passing only permits a separately pre-registered namespace-remapped serving
parity run. Fresh integration, benchmark, canary, holdout, and RL remain closed.
"""


def main() -> None:
    report = build_report()
    PUBLIC_JSON.parent.mkdir(parents=True, exist_ok=True)
    PUBLIC_JSON.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    MARKDOWN.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
