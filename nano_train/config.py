from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SFTSmokeConfig:
    schema_version: str
    experiment_id: str
    model_path: str
    dataset_path: str
    output_dir: str
    seed: int
    dtype: str
    max_length: int
    max_steps: int
    batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    weight_decay: float
    warmup_steps: int
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    lora_targets: tuple[str, ...]
    generation_max_new_tokens: int
    dataset_schema: str = "analog_v1"
    release_manifest_path: str | None = None
    train_samples_per_family: int | None = None
    validation_samples_per_family: int | None = None
    gradient_checkpointing: bool = False


def load_sft_smoke_config(path: str | Path) -> SFTSmokeConfig:
    raw: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = set(SFTSmokeConfig.__dataclass_fields__)
    unknown = set(raw) - expected
    optional = {
        "dataset_schema",
        "release_manifest_path",
        "train_samples_per_family",
        "validation_samples_per_family",
        "gradient_checkpointing",
    }
    missing = expected - set(raw) - optional
    if unknown or missing:
        raise ValueError(
            f"invalid SFT config fields: unknown={sorted(unknown)}, "
            f"missing={sorted(missing)}"
        )
    raw["lora_targets"] = tuple(raw["lora_targets"])
    config = SFTSmokeConfig(**raw)
    validate_sft_smoke_config(config)
    return config


def validate_sft_smoke_config(config: SFTSmokeConfig) -> None:
    if config.schema_version not in {
        "nano_train_sft_smoke_v1",
        "nano_train_sft_smoke_v2",
    }:
        raise ValueError("unsupported SFT smoke schema")
    if config.dtype not in {"float16", "float32"}:
        raise ValueError("V100 SFT smoke dtype must be float16 or float32")
    for name in (
        "max_length",
        "max_steps",
        "batch_size",
        "gradient_accumulation_steps",
        "lora_r",
        "lora_alpha",
        "generation_max_new_tokens",
    ):
        if getattr(config, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if config.max_steps > 40:
        raise ValueError("smoke run may not exceed 40 optimizer steps")
    max_length_limit = (
        1152
        if config.schema_version == "nano_train_sft_smoke_v2"
        else 256
    )
    if config.max_length > max_length_limit:
        raise ValueError(
            f"smoke max_length may not exceed {max_length_limit}"
        )
    if config.learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if config.weight_decay < 0:
        raise ValueError("weight_decay must be non-negative")
    if config.warmup_steps < 0 or config.warmup_steps >= config.max_steps:
        raise ValueError("warmup_steps must be in [0, max_steps)")
    if not config.lora_targets:
        raise ValueError("lora_targets must not be empty")
    if config.schema_version == "nano_train_sft_smoke_v1":
        if config.dataset_schema != "analog_v1":
            raise ValueError("v1 smoke requires analog_v1 data")
        if any(
            value is not None
            for value in (
                config.release_manifest_path,
                config.train_samples_per_family,
                config.validation_samples_per_family,
            )
        ):
            raise ValueError("v1 smoke does not accept release data fields")
        if config.gradient_checkpointing:
            raise ValueError("v1 smoke does not enable gradient checkpointing")
    else:
        if config.dataset_schema != "skill_release_jsonl_v1":
            raise ValueError(
                "v2 smoke requires skill_release_jsonl_v1 data"
            )
        if not config.release_manifest_path:
            raise ValueError("v2 smoke requires release_manifest_path")
        for name in (
            "train_samples_per_family",
            "validation_samples_per_family",
        ):
            value = getattr(config, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not config.gradient_checkpointing:
            raise ValueError(
                "v2 long-sequence smoke requires gradient checkpointing"
            )
