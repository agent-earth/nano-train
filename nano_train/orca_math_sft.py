from __future__ import annotations

import hashlib
import json
import math
import random
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path
from typing import Any

from nano_train.sft import sha256_file
from nano_train.synthetic_quality import paired_comparison


CONFIG_SCHEMA = "nano_train_orca_math_sft_v1"
PREREGISTER_SCHEMA = "nano_train_orca_math_sft_preregister_v1"
SAMPLE_SCHEMA = "nano_orca_math_sft_sample_v1"
RELEASE_SCHEMA = "nano_orca_math_sft_release_v1"
FINAL_PATTERN = re.compile(
    r"^FINAL: ([-+]?(?:[0-9]+(?:\.[0-9]+)?|[0-9]+/[0-9]+))$"
)


@dataclass(frozen=True)
class OrcaMathSFTConfig:
    schema_version: str
    experiment_id: str
    model_path: str
    model_config_sha256: str
    dataset_path: str
    dataset_file_sha256: str
    release_manifest_path: str
    release_manifest_sha256: str
    output_dir: str
    seed: int
    selection_seed: str
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
    gradient_checkpointing: bool
    generation_max_new_tokens: int
    generation_batch_size: int
    train_rows_by_stratum: dict[str, int]
    dev_rows_by_stratum: dict[str, int]
    bootstrap_samples: int
    bootstrap_seed: int
    alpha: float
    minimum_candidate_only_wins: int


def load_config(path: str | Path) -> OrcaMathSFTConfig:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = set(OrcaMathSFTConfig.__dataclass_fields__)
    if set(raw) != expected:
        raise ValueError("Orca Math SFT config fields differ")
    raw["lora_targets"] = tuple(raw["lora_targets"])
    config = OrcaMathSFTConfig(**raw)
    validate_config(config)
    return config


def validate_config(config: OrcaMathSFTConfig) -> None:
    if config.schema_version != CONFIG_SCHEMA:
        raise ValueError("unsupported Orca Math SFT schema")
    expected: dict[str, Any] = {
        "experiment_id": "orca-math-sft-smoke-v1",
        "model_path": "../../../models/Qwen3.5-4B",
        "dataset_path": (
            "../../../datasets/ultimate-distill/"
            "orca-math-sft-v1/dataset.jsonl"
        ),
        "release_manifest_path": (
            "../nano-data-pipeline-fullstack-traex-03/"
            "manifests/orca_math_sft_v1.release.json"
        ),
        "output_dir": "artifacts/orca-math-sft-smoke-v1",
        "seed": 20260821,
        "selection_seed": "orca-math-sft-smoke-v1:20260821",
        "dtype": "float32",
        "max_length": 1024,
        "max_steps": 40,
        "batch_size": 1,
        "gradient_accumulation_steps": 4,
        "learning_rate": 0.00005,
        "weight_decay": 0.0,
        "warmup_steps": 1,
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.0,
        "lora_targets": ("q_proj", "v_proj"),
        "gradient_checkpointing": True,
        "generation_max_new_tokens": 384,
        "generation_batch_size": 4,
        "train_rows_by_stratum": {
            "short": 40,
            "medium": 80,
            "long": 40,
        },
        "dev_rows_by_stratum": {
            "short": 48,
            "medium": 96,
            "long": 48,
        },
        "bootstrap_samples": 10_000,
        "bootstrap_seed": 20260821,
        "alpha": 0.05,
        "minimum_candidate_only_wins": 6,
    }
    for field, expected_value in expected.items():
        if getattr(config, field) != expected_value:
            raise ValueError(
                f"Orca Math SFT freezes {field}={expected_value}"
            )
    for field in (
        "model_config_sha256",
        "dataset_file_sha256",
        "release_manifest_sha256",
    ):
        value = getattr(config, field)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)
        ):
            raise ValueError(f"Orca Math SFT {field} is not SHA256")
    if (
        sum(config.train_rows_by_stratum.values())
        != config.max_steps
        * config.batch_size
        * config.gradient_accumulation_steps
    ):
        raise ValueError("Orca Math SFT train exposure count differs")
    if sum(config.dev_rows_by_stratum.values()) != 192:
        raise ValueError("Orca Math SFT dev count differs")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _sha256_lines(values: list[str]) -> str:
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def _rank(seed: str, sample_id: str) -> str:
    return hashlib.sha256(f"{seed}\n{sample_id}".encode("utf-8")).hexdigest()


def build_selection_contract(
    config: OrcaMathSFTConfig,
) -> dict[str, Any]:
    dataset_path = Path(config.dataset_path)
    release_path = Path(config.release_manifest_path)
    model_config_path = Path(config.model_path) / "config.json"
    if sha256_file(dataset_path) != config.dataset_file_sha256:
        raise ValueError("Orca Math SFT dataset identity differs")
    if sha256_file(release_path) != config.release_manifest_sha256:
        raise ValueError("Orca Math SFT release identity differs")
    if sha256_file(model_config_path) != config.model_config_sha256:
        raise ValueError("Orca Math SFT model identity differs")
    release = json.loads(release_path.read_text(encoding="utf-8"))
    if (
        release.get("schema_version") != RELEASE_SCHEMA
        or release.get("release_id") != "orca-math-sft-v1"
        or release.get("training_unblocked") is not True
        or release.get("rl_or_opd_unlocked") is not False
        or not all(release.get("checks", {}).values())
        or release.get("source", {}).get("dataset_file_sha256")
        != config.dataset_file_sha256
    ):
        raise ValueError("Orca Math SFT release is not admitted")

    rows = load_jsonl(dataset_path)
    by_split_stratum: dict[tuple[str, str], list[dict[str, Any]]] = {}
    seen_ids = set()
    for row in rows:
        sample_id = str(row.get("sample_id", ""))
        split = row.get("split")
        stratum = row.get("stratum")
        if (
            row.get("schema_version") != SAMPLE_SCHEMA
            or not sample_id
            or sample_id in seen_ids
            or split not in {"train", "dev"}
            or stratum not in {"short", "medium", "long"}
            or row.get("training_eligible") is not (split == "train")
        ):
            raise ValueError("Orca Math SFT dataset row differs")
        seen_ids.add(sample_id)
        by_split_stratum.setdefault((split, stratum), []).append(row)
    selected_train = []
    selected_dev = []
    for stratum in ("short", "medium", "long"):
        train = sorted(
            by_split_stratum[("train", stratum)],
            key=lambda row: (
                _rank(config.selection_seed + ":train", row["sample_id"]),
                row["sample_id"],
            ),
        )[: config.train_rows_by_stratum[stratum]]
        dev = sorted(
            by_split_stratum[("dev", stratum)],
            key=lambda row: (
                _rank(config.selection_seed + ":dev", row["sample_id"]),
                row["sample_id"],
            ),
        )[: config.dev_rows_by_stratum[stratum]]
        if (
            len(train) != config.train_rows_by_stratum[stratum]
            or len(dev) != config.dev_rows_by_stratum[stratum]
        ):
            raise ValueError("Orca Math SFT selection count differs")
        selected_train.extend(train)
        selected_dev.extend(dev)
    selected_train.sort(
        key=lambda row: (
            _rank(config.selection_seed + ":schedule", row["sample_id"]),
            row["sample_id"],
        )
    )
    selected_dev.sort(key=lambda row: row["sample_id"])
    train_ids = [row["sample_id"] for row in selected_train]
    dev_ids = [row["sample_id"] for row in selected_dev]
    if set(train_ids) & set(dev_ids):
        raise ValueError("Orca Math SFT train/dev selection overlaps")
    return {
        "release": release,
        "selected_train": selected_train,
        "selected_dev": selected_dev,
        "train_sample_ids": train_ids,
        "dev_sample_ids": dev_ids,
        "train_sample_ids_sha256": _sha256_lines(train_ids),
        "dev_sample_ids_sha256": _sha256_lines(dev_ids),
    }


def numeric_value(value: str) -> Fraction | None:
    try:
        if "/" in value:
            numerator, denominator = value.split("/", 1)
            return Fraction(Decimal(numerator)) / Fraction(
                Decimal(denominator)
            )
        return Fraction(Decimal(value))
    except (
        InvalidOperation,
        ValueError,
        ZeroDivisionError,
    ):
        return None


def parse_final(output: str) -> str | None:
    lines = [line.strip() for line in output.strip().splitlines() if line.strip()]
    if not lines:
        return None
    match = FINAL_PATTERN.fullmatch(lines[-1])
    return match.group(1) if match else None


def score_output(output: str, expected: str) -> bool:
    prediction = parse_final(output)
    predicted_value = numeric_value(prediction) if prediction else None
    expected_value = numeric_value(expected)
    return (
        predicted_value is not None
        and expected_value is not None
        and predicted_value == expected_value
    )


def compare_rows(
    candidate_rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    return paired_comparison(
        candidate_rows,
        baseline_rows,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )


def admission_gates(
    comparison: dict[str, Any],
    *,
    candidate_by_stratum: dict[str, int],
    baseline_by_stratum: dict[str, int],
    alpha: float,
    minimum_candidate_only_wins: int,
) -> dict[str, bool]:
    return {
        "point_delta_positive": comparison["delta"] > 0,
        "bootstrap_ci_lower_positive": (
            comparison["paired_bootstrap_95_ci"][0] > 0
        ),
        "mcnemar_below_alpha": comparison["mcnemar_exact_p"] < alpha,
        "minimum_candidate_only_wins": (
            comparison["paired_counts"]["candidate_only"]
            >= minimum_candidate_only_wins
        ),
        "candidate_only_exceeds_baseline_only": (
            comparison["paired_counts"]["candidate_only"]
            > comparison["paired_counts"]["baseline_only"]
        ),
        "every_stratum_non_regression": all(
            candidate_by_stratum[stratum]
            >= baseline_by_stratum[stratum]
            for stratum in ("short", "medium", "long")
        ),
    }
