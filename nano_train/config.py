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


def load_sft_smoke_config(path: str | Path) -> SFTSmokeConfig:
    raw: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = set(SFTSmokeConfig.__dataclass_fields__)
    unknown = set(raw) - expected
    missing = expected - set(raw)
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
    if config.schema_version != "nano_train_sft_smoke_v1":
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
    if config.max_steps > 20:
        raise ValueError("smoke run may not exceed 20 optimizer steps")
    if config.max_length > 256:
        raise ValueError("smoke max_length may not exceed 256")
    if config.learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if config.weight_decay < 0:
        raise ValueError("weight_decay must be non-negative")
    if config.warmup_steps < 0 or config.warmup_steps >= config.max_steps:
        raise ValueError("warmup_steps must be in [0, max_steps)")
    if not config.lora_targets:
        raise ValueError("lora_targets must not be empty")
