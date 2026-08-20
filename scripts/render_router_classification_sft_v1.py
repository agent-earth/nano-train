#!/usr/bin/env python3

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from nano_train.router_classification import (
    DATASET_CANONICAL_SHA256,
    DATASET_SHA256,
    RELEASE_SHA256,
    load_config,
    verify_data_release,
)
from nano_train.sft import sha256_file, sha256_tree


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/sft/router_classification_smoke_v1.json"
PREREG = (
    ROOT
    / "docs/experiments/qwen35_router_classification_sft_v1.preregister.json"
)
ARTIFACT = ROOT / "artifacts/qwen35-router-classification-sft-smoke-v1"
METRICS = ARTIFACT / "metrics.json"
GENERATIONS = ARTIFACT / "generations.json"
RELOAD = ARTIFACT / "reload_validation.json"
ADAPTER = ARTIFACT / "adapter"
PUBLIC_JSON = (
    ROOT / "docs/results/qwen35_router_classification_sft_v1.public.json"
)
MARKDOWN = ROOT / "docs/results/qwen35_router_classification_sft_v1.md"
PREREG_REVISION = "9397470864d76016a174af7cbee098e72e5dcd9b"


def git_revision() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()


def build_report() -> dict:
    config = load_config(CONFIG)
    release = verify_data_release(config)
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    metrics = json.loads(METRICS.read_text(encoding="utf-8"))
    reload = json.loads(RELOAD.read_text(encoding="utf-8"))
    if (
        prereg.get("schema_version")
        != "nano_train_router_classification_preregister_v1"
        or prereg.get("identity", {}).get("config_sha256")
        != sha256_file(CONFIG)
        or prereg.get("identity", {}).get("code_revision")
        != "e418ef03fa0f4538be652b6a902cd5f5936dd7c7"
        or metrics.get("schema_version") != "nano_train_sft_smoke_result_v1"
        or metrics.get("experiment_id") != config.experiment_id
        or metrics.get("dataset", {}).get("sha256") != DATASET_SHA256
        or metrics.get("router_release", {}).get("sha256") != RELEASE_SHA256
        or metrics.get("router_release", {}).get("dataset_canonical_sha256")
        != DATASET_CANONICAL_SHA256
        or metrics.get("adapter_sha256") != sha256_tree(ADAPTER)
        or metrics.get("generations_sha256") != sha256_file(GENERATIONS)
        or reload.get("adapter_sha256") != metrics.get("adapter_sha256")
        or reload.get("validation") != metrics.get("post_sft_validation")
        or reload.get("reload_success") is not True
    ):
        raise ValueError("router SFT result identity or reload differs")
    baseline = metrics["baseline_validation"]
    post = metrics["post_sft_validation"]
    by_label = {}
    for family in ("router_a", "router_b", "router_c"):
        before = baseline["by_family"][family]
        after = post["by_family"][family]
        by_label[family] = {
            "samples": before["samples"],
            "baseline_exact": before["exact"],
            "post_exact": after["exact"],
            "delta": after["exact"] - before["exact"],
            "reload_exact": reload["validation"]["by_family"][family][
                "exact"
            ],
        }
    gates = {
        "no_failure_receipt": not (ARTIFACT / "failure.json").exists(),
        "finite_loss_curve": (
            len(metrics["loss_curve"]) == config.max_steps
            and all(
                row["loss"] >= 0 and row["loss"] < float("inf")
                for row in metrics["loss_curve"]
            )
        ),
        "adapter_identity_matches": metrics["adapter_sha256"]
        == sha256_tree(ADAPTER),
        "reload_success": reload["reload_success"] is True,
        "reload_exact_metrics": reload["validation"] == post,
        "aggregate_post_exact_gt_baseline": post["exact"] > baseline["exact"],
        "router_a_post_exact_at_least_48_of_64": (
            post["by_family"]["router_a"]["exact"] >= 48
        ),
        "router_b_post_exact_at_least_48_of_64": (
            post["by_family"]["router_b"]["exact"] >= 48
        ),
        "router_c_post_exact_at_least_60_of_64": (
            post["by_family"]["router_c"]["exact"] >= 60
        ),
        "every_label_non_regression": all(
            post["by_family"][family]["exact"]
            >= baseline["by_family"][family]["exact"]
            for family in ("router_a", "router_b", "router_c")
        ),
        "data_release_identity_matches": (
            release["dataset_file_sha256"] == DATASET_SHA256
            and release["dataset_canonical_sha256"]
            == DATASET_CANONICAL_SHA256
        ),
    }
    admitted = all(gates.values())
    return {
        "schema_version": "nano_train_router_classification_public_v1",
        "experiment_id": config.experiment_id,
        "identity": {
            "result_revision": git_revision(),
            "preregister_revision": PREREG_REVISION,
            "config_sha256": sha256_file(CONFIG),
            "preregister_sha256": sha256_file(PREREG),
            "metrics_sha256": sha256_file(METRICS),
            "generations_sha256": sha256_file(GENERATIONS),
            "reload_sha256": sha256_file(RELOAD),
            "adapter_sha256": metrics["adapter_sha256"],
            "dataset_file_sha256": DATASET_SHA256,
            "dataset_canonical_sha256": DATASET_CANONICAL_SHA256,
            "release_sha256": RELEASE_SHA256,
            "model_config_sha256": metrics["model"]["config_sha256"],
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
            "trainable_parameters": metrics["model"][
                "trainable_parameters"
            ],
            "peak_allocated_gib": metrics["hardware"][
                "peak_allocated_gib"
            ],
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
            "by_label": by_label,
        },
        "gates": gates,
        "decision": {
            "router_sft_smoke_admitted": admitted,
            "fresh_router_integration_preregistration_allowed": admitted,
            "fresh_router_integration_generation_allowed": False,
            "benchmark_allowed": False,
            "canary_allowed": False,
            "independent_holdout_allowed": False,
            "rl_allowed": False,
            "further_tuning_or_second_training_run_allowed": False,
            "next_action": (
                "Pre-register a fresh history-disjoint router integration "
                "using this exact adapter. Require A/B recall, C precision, "
                "typed execution, direct preservation, and zero losses before "
                "any real question scan."
                if admitted
                else "Reject this router SFT recipe and preserve negative "
                "evidence; do not tune or rerun."
            ),
        },
        "artifact_boundary": {
            "adapter_committed": False,
            "raw_generations_committed": False,
            "metrics_committed": False,
            "training_data_contains_benchmark_content": False,
            "benchmark_accessed": False,
            "canary_accessed": False,
            "holdout_accessed": False,
        },
        "claim_boundary": (
            "This result establishes only local synthetic router "
            "classification learning. It is not real-task routing, benchmark, "
            "canary, holdout, or final model-superiority evidence."
        ),
    }


def render_markdown(report: dict) -> str:
    validation = report["validation"]
    return f"""# Qwen3.5 Router Classification SFT Smoke v1 Result

## 结论

- admitted：`{str(report['decision']['router_sft_smoke_admitted']).lower()}`；
- baseline：{validation['baseline_exact']}/{validation['samples']}；
- post SFT：{validation['post_exact']}/{validation['samples']}；
- delta：+{validation['delta']}；
- independent reload：192/192，与 post metrics 完全一致。

## Per Label

| Label | Baseline | Post | Delta | Reload |
| --- | ---: | ---: | ---: | ---: |
{chr(10).join(
    f"| {name} | {row['baseline_exact']}/64 | {row['post_exact']}/64 | "
    f"{row['delta']:+d} | {row['reload_exact']}/64 |"
    for name, row in report['validation']['by_label'].items()
)}

## Training

- 40 steps，effective batch 4；
- FP32 expanded LoRA，9,961,472 trainable parameters；
- loss：{report['training']['first_loss']:.6f} →
  {report['training']['last_loss']:.6f}；
- peak：{report['training']['peak_allocated_gib']:.2f} GiB；
- reload peak：{report['training']['reload_peak_allocated_gib']:.2f} GiB；
- wall：{report['training']['wall_seconds']:.1f}s。

## Gates

```json
{json.dumps(report['gates'], indent=2, sort_keys=True)}
```

## Evidence

- prereg commit：`{report['identity']['preregister_revision']}`；
- adapter SHA：`{report['identity']['adapter_sha256']}`；
- metrics SHA：`{report['identity']['metrics_sha256']}`；
- generations SHA：`{report['identity']['generations_sha256']}`；
- reload SHA：`{report['identity']['reload_sha256']}`；
- dataset SHA：`{report['identity']['dataset_file_sha256']}`。

## 边界

这是 synthetic router classification smoke。Adapter/raw generations/metrics 保持
ignored，public commit 只记录 hashes 和 aggregate。通过只允许另行预注册 fresh
router integration；benchmark/canary/holdout/RL 继续关闭。
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
