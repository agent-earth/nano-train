from __future__ import annotations

import hashlib
import json
import math
import random
import re
import time
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from torch.optim import AdamW
from transformers import AutoTokenizer, Qwen3_5ForCausalLM

from nano_train.data import TokenizedSample, collate_samples, tokenize_samples
from nano_train.sft import sha256_file
from nano_train.sft import (
    _assert_finite_gradients,
    _assert_finite_loss,
    _assert_finite_parameters,
    _scheduler_scale,
    _trainable_parameters,
    _write_failure,
    dependency_versions,
    set_seed,
    sha256_tree,
)
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


def _training_dataset(selection: dict[str, Any]) -> dict[str, Any]:
    samples = []
    for row in selection["selected_train"]:
        samples.append(
            {
                "sample_id": row["sample_id"],
                "split": "train",
                "task_family": row["stratum"],
                "format_family": "orca_math_numeric",
                "messages": row["messages"],
                "verifier": row["verifier"],
            }
        )
    for row in selection["selected_dev"]:
        samples.append(
            {
                "sample_id": row["sample_id"],
                "split": "validation",
                "task_family": row["stratum"],
                "format_family": "orca_math_numeric",
                "messages": row["messages"],
                "verifier": row["verifier"],
            }
        )
    return {
        "dataset_id": "orca-math-sft-v1",
        "samples": samples,
    }


def _evaluation_rows(
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    *,
    device: torch.device,
    max_new_tokens: int,
    batch_size: int,
) -> list[dict[str, Any]]:
    model.eval()
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    results = []
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            batch_rows = rows[start : start + batch_size]
            prompts = [
                tokenizer.apply_chat_template(
                    row["messages"][:-1],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                for row in batch_rows
            ]
            encoded = tokenizer(
                prompts,
                add_special_tokens=False,
                padding=True,
                return_tensors="pt",
            )
            prompt_width = encoded["input_ids"].shape[1]
            generated = model.generate(
                input_ids=encoded["input_ids"].to(device),
                attention_mask=encoded["attention_mask"].to(device),
                do_sample=False,
                max_new_tokens=max_new_tokens,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
            )
            outputs = tokenizer.batch_decode(
                generated[:, prompt_width:],
                skip_special_tokens=True,
            )
            for row, output in zip(batch_rows, outputs):
                output = output.strip()
                expected = str(row["numeric_answer"])
                prediction = parse_final(output)
                results.append(
                    {
                        "case_id": row["sample_id"],
                        "stratum": row["stratum"],
                        "expected": expected,
                        "output": output,
                        "prediction": prediction,
                        "correct": score_output(output, expected),
                        "parse_failure": prediction is None,
                    }
                )
    tokenizer.padding_side = original_padding_side
    return results


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_stratum = {}
    for stratum in ("short", "medium", "long"):
        selected = [row for row in rows if row["stratum"] == stratum]
        by_stratum[stratum] = {
            "samples": len(selected),
            "correct": sum(bool(row["correct"]) for row in selected),
            "parse_failures": sum(
                bool(row["parse_failure"]) for row in selected
            ),
        }
    return {
        "samples": len(rows),
        "correct": sum(bool(row["correct"]) for row in rows),
        "accuracy": sum(bool(row["correct"]) for row in rows) / len(rows),
        "parse_failures": sum(bool(row["parse_failure"]) for row in rows),
        "by_stratum": by_stratum,
    }


def run(config: OrcaMathSFTConfig) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("Orca Math SFT requires one CUDA GPU")
    selection = build_selection_contract(config)
    dataset = _training_dataset(selection)
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_path,
        local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenized = tokenize_samples(
        dataset,
        tokenizer,
        max_length=config.max_length,
    )
    by_id = {sample.sample_id: sample for sample in tokenized}
    train = [by_id[sample_id] for sample_id in selection["train_sample_ids"]]
    if len(train) != 160 or len({sample.sample_id for sample in train}) != 160:
        raise ValueError("Orca Math SFT train schedule differs")

    set_seed(config.seed)
    device = torch.device("cuda")
    output_root = Path(config.output_dir)
    adapter_dir = output_root / "adapter"
    output_root.mkdir(parents=True, exist_ok=True)
    started = time.time()
    model = Qwen3_5ForCausalLM.from_pretrained(
        config.model_path,
        local_files_only=True,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
    ).to(device)
    model.config.use_cache = False
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    model = get_peft_model(
        model,
        LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=list(config.lora_targets),
            task_type="CAUSAL_LM",
        ),
    )
    trainable = _trainable_parameters(model)
    baseline_rows = _evaluation_rows(
        model,
        tokenizer,
        selection["selected_dev"],
        device=device,
        max_new_tokens=config.generation_max_new_tokens,
        batch_size=config.generation_batch_size,
    )
    baseline_summary = summarize_rows(baseline_rows)

    optimizer = AdamW(
        trainable,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        eps=1e-6,
    )
    loss_curve = []
    train_exposure = []
    optimizer.zero_grad(set_to_none=True)
    model.train()
    for step in range(config.max_steps):
        step_number = step + 1
        selected = train[
            step
            * config.gradient_accumulation_steps : (
                (step + 1) * config.gradient_accumulation_steps
            )
        ]
        if len(selected) != config.gradient_accumulation_steps:
            raise ValueError("Orca Math SFT step exposure differs")
        step_losses = []
        train_exposure.append(
            {
                "step": step_number,
                "sample_ids": [sample.sample_id for sample in selected],
                "strata": [sample.task_family for sample in selected],
            }
        )
        for sample in selected:
            batch = {
                key: value.to(device)
                for key, value in collate_samples(
                    [sample],
                    pad_token_id=tokenizer.pad_token_id,
                ).items()
            }
            outputs = model(**batch, use_cache=False)
            try:
                _assert_finite_loss(outputs.loss, step=step_number)
            except FloatingPointError as error:
                _write_failure(
                    output_root,
                    step=step_number,
                    stage="forward_loss",
                    error=error,
                )
                raise
            (outputs.loss / config.gradient_accumulation_steps).backward()
            step_losses.append(float(outputs.loss.detach().cpu()))
        scale = _scheduler_scale(
            step,
            config.warmup_steps,
            config.max_steps,
        )
        for group in optimizer.param_groups:
            group["lr"] = config.learning_rate * scale
        try:
            _assert_finite_gradients(trainable, step=step_number)
            torch.nn.utils.clip_grad_norm_(
                trainable,
                max_norm=1.0,
                error_if_nonfinite=True,
            )
            optimizer.step()
            _assert_finite_parameters(trainable, step=step_number)
        except (FloatingPointError, RuntimeError) as error:
            _write_failure(
                output_root,
                step=step_number,
                stage="optimizer_step",
                error=error,
            )
            raise
        optimizer.zero_grad(set_to_none=True)
        loss_curve.append(
            {
                "step": step_number,
                "loss": sum(step_losses) / len(step_losses),
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )

    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir, safe_serialization=True)
    tokenizer.save_pretrained(adapter_dir)
    model.config.use_cache = True
    post_rows = _evaluation_rows(
        model,
        tokenizer,
        selection["selected_dev"],
        device=device,
        max_new_tokens=config.generation_max_new_tokens,
        batch_size=config.generation_batch_size,
    )
    post_summary = summarize_rows(post_rows)
    generations_path = output_root / "generations.json"
    generations_path.write_text(
        json.dumps(
            {"baseline": baseline_rows, "post_sft": post_rows},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    comparison = compare_rows(
        post_rows,
        baseline_rows,
        bootstrap_samples=config.bootstrap_samples,
        bootstrap_seed=config.bootstrap_seed,
    )
    gates = admission_gates(
        comparison,
        candidate_by_stratum={
            key: value["correct"]
            for key, value in post_summary["by_stratum"].items()
        },
        baseline_by_stratum={
            key: value["correct"]
            for key, value in baseline_summary["by_stratum"].items()
        },
        alpha=config.alpha,
        minimum_candidate_only_wins=config.minimum_candidate_only_wins,
    )
    result = {
        "schema_version": "nano_train_orca_math_sft_result_v1",
        "experiment_id": config.experiment_id,
        "config": {
            **config.__dict__,
            "lora_targets": list(config.lora_targets),
        },
        "selection": {
            "train_sample_ids_sha256": selection[
                "train_sample_ids_sha256"
            ],
            "dev_sample_ids_sha256": selection["dev_sample_ids_sha256"],
            "train_samples": len(train),
            "dev_samples": len(selection["selected_dev"]),
        },
        "model_config_sha256": config.model_config_sha256,
        "dataset_file_sha256": config.dataset_file_sha256,
        "release_manifest_sha256": config.release_manifest_sha256,
        "adapter_sha256": sha256_tree(adapter_dir),
        "trainable_parameters": sum(
            parameter.numel() for parameter in trainable
        ),
        "baseline_validation": baseline_summary,
        "post_validation": post_summary,
        "comparison": comparison,
        "gates": gates,
        "candidate_admitted": all(gates.values()),
        "loss_curve": loss_curve,
        "train_exposure": train_exposure,
        "dependencies": dependency_versions(),
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "wall_seconds": time.time() - started,
        "generations_sha256": sha256_file(generations_path),
        "failure_receipt_exists": (output_root / "failure.json").exists(),
    }
    (output_root / "metrics.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def validate_reload(config: OrcaMathSFTConfig) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("Orca Math SFT reload requires CUDA")
    selection = build_selection_contract(config)
    output_root = Path(config.output_dir)
    adapter_dir = output_root / "adapter"
    metrics_path = output_root / "metrics.json"
    generations_path = output_root / "generations.json"
    if (
        not adapter_dir.is_dir()
        or not metrics_path.is_file()
        or not generations_path.is_file()
        or (output_root / "failure.json").exists()
    ):
        raise ValueError("Orca Math SFT reload artifacts are incomplete")
    source_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    source_generations = json.loads(
        generations_path.read_text(encoding="utf-8")
    )
    tokenizer = AutoTokenizer.from_pretrained(
        adapter_dir,
        local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = torch.device("cuda")
    model = Qwen3_5ForCausalLM.from_pretrained(
        config.model_path,
        local_files_only=True,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
    ).to(device)
    model = PeftModel.from_pretrained(
        model,
        adapter_dir,
        is_trainable=False,
    ).to(device)
    rows = _evaluation_rows(
        model,
        tokenizer,
        selection["selected_dev"],
        device=device,
        max_new_tokens=config.generation_max_new_tokens,
        batch_size=config.generation_batch_size,
    )
    summary = summarize_rows(rows)
    generations_exact = rows == source_generations["post_sft"]
    metrics_exact = summary == source_metrics["post_validation"]
    if not generations_exact or not metrics_exact:
        raise ValueError("Orca Math SFT reload differs")
    reload_path = output_root / "reload_generations.json"
    reload_path.write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    receipt = {
        "schema_version": "nano_train_orca_math_sft_reload_v1",
        "experiment_id": config.experiment_id,
        "adapter_sha256": sha256_tree(adapter_dir),
        "reload_success": True,
        "metrics_exact": True,
        "generations_exact": True,
        "post_validation": summary,
        "source_generations_sha256": sha256_file(generations_path),
        "reload_generations_sha256": sha256_file(reload_path),
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
    }
    (output_root / "reload_validation.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return receipt
