from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from nano_train.config import SFTSmokeConfig, load_sft_smoke_config
from nano_train.sft import run_sft_smoke, sha256_file


EXPERIMENT_ID = "qwen35-router-negative-diversity-sft-v2"
DATASET_SHA256 = (
    "8c5975e3ceed494e20d0de54eb5654ab1af71163ed58489d42d98c8b54d0bad9"
)
RELEASE_SHA256 = (
    "5edd89701ff33db6eaef74475946abf79176c5c5a7c854a7eea4dd907e69c3f1"
)
DATASET_CANONICAL_SHA256 = (
    "f63c58b54ef4747f274599784bad9ffe4143117482c22b33005a2dbf725b1f2f"
)
AUDIT_SHA256 = (
    "9aaa69de746dbdc5cefbb52fb271c8f9ec86716d10ada70704c7e346dc2f7c17"
)
CONTRACT_SHA256 = (
    "c195a7373ea283546dde1866f70593f0912833d987ff5f1a8cb424c2bc340335"
)


def load_config(path: str | Path) -> SFTSmokeConfig:
    config = load_sft_smoke_config(path)
    expected: dict[str, Any] = {
        "schema_version": "nano_train_sft_smoke_v1",
        "experiment_id": EXPERIMENT_ID,
        "model_path": "../../../models/Qwen3.5-4B",
        "dataset_path": (
            "../nano-data-pipeline-fullstack-traex-03/datasets/"
            "qwen35_router_negative_diversity_v2.json"
        ),
        "output_dir": "artifacts/qwen35-router-negative-diversity-sft-v2",
        "seed": 20260827,
        "dtype": "float32",
        "max_length": 256,
        "max_steps": 40,
        "batch_size": 1,
        "gradient_accumulation_steps": 4,
        "learning_rate": 0.0002,
        "weight_decay": 0.0,
        "warmup_steps": 2,
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.0,
        "lora_targets": (
            "q_proj",
            "v_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ),
        "generation_max_new_tokens": 8,
        "dataset_schema": "analog_v1",
        "release_manifest_path": None,
        "train_samples_per_family": None,
        "validation_samples_per_family": None,
        "gradient_checkpointing": False,
        "validation_start_per_family": 0,
        "train_family_schedule": (),
        "train_sample_schedule": (),
    }
    for field, expected_value in expected.items():
        if getattr(config, field) != expected_value:
            raise ValueError(
                f"router negative diversity SFT freezes {field}={expected_value}"
            )
    return config


def verify_data_release(config: SFTSmokeConfig) -> dict[str, Any]:
    dataset_path = Path(config.dataset_path)
    release_path = (
        dataset_path.parent.parent
        / "manifests/qwen35_router_negative_diversity_v2.release.json"
    )
    if (
        sha256_file(dataset_path) != DATASET_SHA256
        or sha256_file(release_path) != RELEASE_SHA256
    ):
        raise ValueError("router negative diversity SFT data identity differs")
    release = json.loads(release_path.read_text(encoding="utf-8"))
    if (
        release.get("schema_version")
        != "nano_router_negative_diversity_release_v2"
        or release.get("release_id")
        != "qwen35-router-negative-diversity-v2"
        or release.get("training_unblocked") is not True
        or not all(release.get("checks", {}).values())
        or release.get("source", {}).get("dataset_canonical_sha256")
        != DATASET_CANONICAL_SHA256
        or release.get("source", {}).get("audit_sha256") != AUDIT_SHA256
        or release.get("source", {}).get("contract_sha256")
        != CONTRACT_SHA256
        or release.get("accepted", {}).get("train_tokens") != 766_519
    ):
        raise ValueError("router negative diversity release is not admitted")
    return {
        "path": str(release_path),
        "sha256": RELEASE_SHA256,
        "dataset_file_sha256": DATASET_SHA256,
        "dataset_canonical_sha256": DATASET_CANONICAL_SHA256,
        "audit_sha256": AUDIT_SHA256,
        "contract_sha256": CONTRACT_SHA256,
        "release_id": release["release_id"],
        "accepted": release["accepted"],
    }


def run(config: SFTSmokeConfig) -> dict[str, Any]:
    release = verify_data_release(config)
    metrics = run_sft_smoke(config)
    metrics["router_negative_diversity_release"] = release
    metrics_path = Path(config.output_dir) / "metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metrics
