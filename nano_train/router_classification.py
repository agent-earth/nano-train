from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from nano_train.config import SFTSmokeConfig, load_sft_smoke_config
from nano_train.sft import run_sft_smoke, sha256_file


CONFIG_SCHEMA = "nano_train_sft_smoke_v1"
EXPERIMENT_ID = "qwen35-router-classification-sft-smoke-v1"
DATASET_SHA256 = (
    "dacd3663639fe9ddc054865b87afdd0c918f0fddb12c8c9355819d4bbce95d65"
)
RELEASE_SHA256 = (
    "fb265e125e181056856a196322cf5da3b1d7d890d60ad653839d2707ebe3781d"
)
DATASET_CANONICAL_SHA256 = (
    "b9f4ef24f16c680f6c5d5999e3ca86cd7c044b83e093d18b39f7e220da70bfad"
)


def load_config(path: str | Path) -> SFTSmokeConfig:
    config = load_sft_smoke_config(path)
    expected: dict[str, Any] = {
        "schema_version": CONFIG_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "model_path": "../../../models/Qwen3.5-4B",
        "dataset_path": (
            "../nano-data-pipeline-fullstack-traex-03/datasets/"
            "qwen35_router_classification_v1.json"
        ),
        "output_dir": "artifacts/qwen35-router-classification-sft-smoke-v1",
        "seed": 20260824,
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
                f"router classification SFT freezes {field}={expected_value}"
            )
    return config


def verify_data_release(config: SFTSmokeConfig) -> dict[str, Any]:
    dataset_path = Path(config.dataset_path)
    release_path = (
        dataset_path.parent.parent
        / "manifests/qwen35_router_classification_v1.release.json"
    )
    if (
        sha256_file(dataset_path) != DATASET_SHA256
        or sha256_file(release_path) != RELEASE_SHA256
    ):
        raise ValueError("router classification SFT data identity differs")
    release = json.loads(release_path.read_text(encoding="utf-8"))
    if (
        release.get("schema_version")
        != "nano_router_classification_release_v1"
        or release.get("release_id") != "qwen35-router-classification-v1"
        or release.get("training_unblocked") is not True
        or not all(release.get("checks", {}).values())
        or release.get("source", {}).get("dataset_canonical_sha256")
        != DATASET_CANONICAL_SHA256
    ):
        raise ValueError("router classification release is not admitted")
    return {
        "path": str(release_path),
        "sha256": RELEASE_SHA256,
        "dataset_file_sha256": DATASET_SHA256,
        "dataset_canonical_sha256": DATASET_CANONICAL_SHA256,
        "release_id": release["release_id"],
        "accepted": release["accepted"],
    }


def run(config: SFTSmokeConfig) -> dict[str, Any]:
    release = verify_data_release(config)
    metrics = run_sft_smoke(config)
    metrics["router_release"] = release
    metrics_path = Path(config.output_dir) / "metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metrics
