from __future__ import annotations

import hashlib
import json
import math
import random
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from torch.optim import AdamW
from transformers import AutoTokenizer, Qwen3_5ForCausalLM

from nano_train.data import TokenizedSample, collate_samples, tokenize_samples
from nano_train.sft import (
    _assert_finite_gradients,
    _assert_finite_loss,
    _assert_finite_parameters,
    _scheduler_scale,
    _trainable_parameters,
    _write_failure,
    dependency_versions,
    set_seed,
    sha256_file,
    sha256_tree,
)


CONFIG_SCHEMA = "nano_train_swe_code_repair_sft_v1"
PREREGISTER_SCHEMA = "nano_train_swe_code_repair_sft_preregister_v1"
RELEASE_SCHEMA = "nano_skill_sft_release_v1"
SAMPLE_SCHEMA = "nano_skill_sft_sample_v1"
FAMILY_ID = "swe-code-repair"


@dataclass(frozen=True)
class SWECodeRepairSFTConfig:
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
    train_rows_by_band: dict[str, int]
    dev_rows_by_band: dict[str, int]
    minimum_dev_loss_relative_improvement: float
    minimum_json_action_valid_rate: float


def load_config(path: str | Path) -> SWECodeRepairSFTConfig:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = set(SWECodeRepairSFTConfig.__dataclass_fields__)
    if set(raw) != expected:
        raise ValueError("SWE code-repair SFT config fields differ")
    raw["lora_targets"] = tuple(raw["lora_targets"])
    config = SWECodeRepairSFTConfig(**raw)
    validate_config(config)
    return config


def validate_config(config: SWECodeRepairSFTConfig) -> None:
    expected: dict[str, Any] = {
        "schema_version": CONFIG_SCHEMA,
        "experiment_id": "evoloop-swe-code-repair-sft-smoke-v1",
        "model_path": "../../../models/Qwen3.5-4B",
        "dataset_path": (
            "../../../datasets/ultimate-distill/"
            "swe-code-repair-sft-v1/accepted.jsonl"
        ),
        "release_manifest_path": (
            "../nano-data-pipeline-tinker-swe-sft-v1/"
            "manifests/swe_code_repair_sft_v1.release.json"
        ),
        "output_dir": "artifacts/evoloop-swe-code-repair-sft-smoke-v1",
        "seed": 20260911,
        "selection_seed": "evoloop-swe-code-repair-sft-smoke-v1:20260911",
        "dtype": "float32",
        "max_length": 2048,
        "max_steps": 40,
        "batch_size": 1,
        "gradient_accumulation_steps": 4,
        "learning_rate": 0.00005,
        "weight_decay": 0.0,
        "warmup_steps": 2,
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.0,
        "lora_targets": ("q_proj", "v_proj"),
        "gradient_checkpointing": True,
        "generation_max_new_tokens": 768,
        "train_rows_by_band": {"short": 40, "medium": 80, "long": 40},
        "dev_rows_by_band": {"short": 4, "medium": 4, "long": 4},
        "minimum_dev_loss_relative_improvement": 0.01,
        "minimum_json_action_valid_rate": 0.5,
    }
    for field, expected_value in expected.items():
        if getattr(config, field) != expected_value:
            raise ValueError(
                f"SWE code-repair SFT freezes {field}={expected_value}"
            )
    for field in (
        "model_config_sha256",
        "dataset_file_sha256",
        "release_manifest_sha256",
    ):
        value = getattr(config, field)
        if (
            len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"SWE code-repair SFT {field} is not SHA256")
    if (
        sum(config.train_rows_by_band.values())
        != config.max_steps
        * config.batch_size
        * config.gradient_accumulation_steps
    ):
        raise ValueError("SWE code-repair train exposure count differs")
    if sum(config.dev_rows_by_band.values()) != 12:
        raise ValueError("SWE code-repair dev count differs")


def validate_release(
    release: dict[str, Any],
    config: SWECodeRepairSFTConfig,
) -> None:
    accepted = release.get("accepted", {})
    checks = release.get("checks", {})
    boundary = release.get("training_boundary", {})
    if (
        release.get("schema_version") != RELEASE_SCHEMA
        or release.get("release_id") != "evoloop-swe-code-repair-sft-v1"
        or release.get("training_unblocked") is not True
        or len(checks) != 17
        or not all(checks.values())
        or accepted.get("train_samples") != 11_000
        or accepted.get("dev_samples") != 512
        or accepted.get("train_tokens") != 11_861_953
        or accepted.get("repositories") != 35
        or release.get("artifacts", {}).get("accepted_jsonl_sha256")
        != config.dataset_file_sha256
        or boundary.get("contains_swe_bench_verified_content") is not False
        or boundary.get("source_train_rows_training_eligible") is not True
        or boundary.get("verified_rows_training_eligible") is not False
    ):
        raise ValueError("SWE code-repair release is not admitted")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _rank(seed: str, sample_id: str) -> str:
    return hashlib.sha256(f"{seed}:{sample_id}".encode()).hexdigest()


def token_band(token_count: int) -> str:
    if token_count <= 768:
        return "short"
    if token_count <= 1_280:
        return "medium"
    return "long"


def build_selection_contract(config: SWECodeRepairSFTConfig) -> dict[str, Any]:
    validate_config(config)
    dataset_path = Path(config.dataset_path)
    release_path = Path(config.release_manifest_path)
    if sha256_file(dataset_path) != config.dataset_file_sha256:
        raise ValueError("SWE code-repair dataset identity differs")
    if sha256_file(release_path) != config.release_manifest_sha256:
        raise ValueError("SWE code-repair release identity differs")
    if sha256_file(Path(config.model_path) / "config.json") != config.model_config_sha256:
        raise ValueError("Qwen3.5-4B model identity differs")
    release = json.loads(release_path.read_text(encoding="utf-8"))
    validate_release(release, config)

    by_split_band: dict[tuple[str, str], list[dict[str, Any]]] = {}
    seen_ids = set()
    for row in load_jsonl(dataset_path):
        sample_id = str(row.get("sample_id", ""))
        split = row.get("split")
        if (
            row.get("schema_version") != SAMPLE_SCHEMA
            or row.get("family_id") != FAMILY_ID
            or row.get("training_eligible") is not True
            or not sample_id
            or sample_id in seen_ids
            or split not in {"train", "dev"}
        ):
            raise ValueError("SWE code-repair dataset row differs")
        seen_ids.add(sample_id)
        band = token_band(int(row["token_count"]))
        by_split_band.setdefault((split, band), []).append(row)

    selected_train = []
    selected_dev = []
    for band in ("short", "medium", "long"):
        train = sorted(
            by_split_band.get(("train", band), []),
            key=lambda row: (
                _rank(config.selection_seed + ":train", row["sample_id"]),
                row["sample_id"],
            ),
        )[: config.train_rows_by_band[band]]
        dev = sorted(
            by_split_band.get(("dev", band), []),
            key=lambda row: (
                _rank(config.selection_seed + ":dev", row["sample_id"]),
                row["sample_id"],
            ),
        )[: config.dev_rows_by_band[band]]
        if (
            len(train) != config.train_rows_by_band[band]
            or len(dev) != config.dev_rows_by_band[band]
        ):
            raise ValueError(f"insufficient SWE code-repair rows for {band}")
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
        raise ValueError("SWE code-repair train/dev selection overlaps")
    return {
        "release": release,
        "selected_train": selected_train,
        "selected_dev": selected_dev,
        "train_sample_ids": train_ids,
        "dev_sample_ids": dev_ids,
        "train_sample_ids_sha256": hashlib.sha256(
            "\n".join(train_ids).encode()
        ).hexdigest(),
        "dev_sample_ids_sha256": hashlib.sha256(
            "\n".join(dev_ids).encode()
        ).hexdigest(),
    }


def training_dataset(selection: dict[str, Any]) -> dict[str, Any]:
    samples = []
    for split, rows in (
        ("train", selection["selected_train"]),
        ("validation", selection["selected_dev"]),
    ):
        for row in rows:
            samples.append(
                {
                    "sample_id": row["sample_id"],
                    "split": split,
                    "task_family": token_band(int(row["token_count"])),
                    "format_family": "swe_code_repair_action",
                    "messages": row["messages"],
                    "verifier": row["verifier"],
                    "task_spec": row["task_spec"],
                }
            )
    return {"dataset_id": "evoloop-swe-code-repair-sft-v1", "samples": samples}


def valid_json_action(output: str) -> bool:
    try:
        value = json.loads(output)
    except json.JSONDecodeError:
        return False
    if not isinstance(value, dict) or set(value) != {
        "analysis",
        "plan",
        "commands",
    }:
        return False
    if (
        not isinstance(value.get("analysis"), str)
        or not value["analysis"].strip()
        or not isinstance(value.get("plan"), str)
        or not value["plan"].strip()
    ):
        return False
    commands = value.get("commands")
    return bool(
        isinstance(commands, list)
        and commands
        and all(
            isinstance(command, dict)
            and set(command) == {"keystrokes", "duration"}
            and isinstance(command.get("keystrokes"), str)
            and command["keystrokes"].strip()
            and isinstance(command.get("duration"), (int, float))
            and not isinstance(command["duration"], bool)
            and math.isfinite(command["duration"])
            and command["duration"] > 0
            for command in commands
        )
    )


def diagnose_json_action_output(output: str) -> dict[str, Any]:
    stripped = output.strip()
    exact_json = None
    parse_status = "empty"
    if stripped:
        try:
            exact_json = json.loads(stripped)
        except json.JSONDecodeError:
            parse_status = "json_decode_error"
        else:
            parse_status = (
                "exact_valid"
                if valid_json_action(stripped)
                else "json_wrong_schema"
            )
    first_brace = stripped.find("{")
    last_brace = stripped.rfind("}")
    recoverable = False
    if first_brace >= 0 and last_brace > first_brace:
        recoverable = valid_json_action(stripped[first_brace : last_brace + 1])
    if not stripped:
        prefix_class = "empty"
    elif stripped.startswith("{"):
        prefix_class = "json_object"
    elif stripped.startswith("```"):
        prefix_class = "markdown_fence"
    elif stripped.startswith("<think>") or stripped.startswith("<analysis>"):
        prefix_class = "reasoning_tag"
    else:
        prefix_class = "prose_or_other"
    return {
        "characters": len(output),
        "parse_status": parse_status,
        "prefix_class": prefix_class,
        "starts_with_open_brace": stripped.startswith("{"),
        "ends_with_close_brace": stripped.endswith("}"),
        "contains_open_brace": "{" in stripped,
        "contains_close_brace": "}" in stripped,
        "recoverable_json_action_substring": recoverable,
        "exact_json_type": (
            type(exact_json).__name__ if exact_json is not None else None
        ),
    }


@torch.inference_mode()
def mean_teacher_forced_loss(
    model: Any,
    samples: list[TokenizedSample],
    *,
    device: torch.device,
    pad_token_id: int,
) -> float:
    model.eval()
    losses = []
    for sample in samples:
        batch = {
            key: value.to(device)
            for key, value in collate_samples(
                [sample],
                pad_token_id=pad_token_id,
            ).items()
        }
        outputs = model(**batch, use_cache=False)
        losses.append(float(outputs.loss.detach().cpu()))
    return sum(losses) / len(losses)


@torch.inference_mode()
def generate_validation(
    model: Any,
    tokenizer: Any,
    samples: list[TokenizedSample],
    *,
    device: torch.device,
    max_new_tokens: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    rows = []
    for sample in samples:
        prompt = torch.tensor([sample.prompt_ids], dtype=torch.long, device=device)
        generated = model.generate(
            input_ids=prompt,
            attention_mask=torch.ones_like(prompt),
            do_sample=False,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
        )
        output = tokenizer.decode(
            generated[0, prompt.shape[1] :],
            skip_special_tokens=True,
        ).strip()
        rows.append(
            {
                "sample_id": sample.sample_id,
                "task_family": sample.task_family,
                "json_action_valid": valid_json_action(output),
                "output_sha256": hashlib.sha256(output.encode()).hexdigest(),
            }
        )
    valid = sum(row["json_action_valid"] for row in rows)
    return (
        {
            "samples": len(rows),
            "json_action_valid": valid,
            "json_action_valid_rate": valid / len(rows),
            "by_band": {
                band: {
                    "samples": sum(row["task_family"] == band for row in rows),
                    "json_action_valid": sum(
                        row["task_family"] == band and row["json_action_valid"]
                        for row in rows
                    ),
                }
                for band in ("short", "medium", "long")
            },
        },
        rows,
    )


def run(config: SWECodeRepairSFTConfig) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("SWE code-repair SFT requires one CUDA GPU")
    selection = build_selection_contract(config)
    tokenizer = AutoTokenizer.from_pretrained(config.model_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenized = tokenize_samples(
        training_dataset(selection),
        tokenizer,
        max_length=config.max_length,
    )
    by_id = {sample.sample_id: sample for sample in tokenized}
    train = [by_id[sample_id] for sample_id in selection["train_sample_ids"]]
    validation = [by_id[sample_id] for sample_id in selection["dev_sample_ids"]]
    if len(train) != 160 or len(validation) != 12:
        raise ValueError("SWE code-repair selected sample count differs")

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
    baseline_loss = mean_teacher_forced_loss(
        model,
        validation,
        device=device,
        pad_token_id=tokenizer.pad_token_id,
    )

    optimizer = AdamW(
        trainable,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        eps=1e-6,
    )
    optimizer.zero_grad(set_to_none=True)
    losses = []
    train_exposure = []
    model.train()
    for step in range(config.max_steps):
        step_number = step + 1
        selected = train[
            step * config.gradient_accumulation_steps : (
                (step + 1) * config.gradient_accumulation_steps
            )
        ]
        step_losses = []
        train_exposure.append(
            {
                "step": step_number,
                "sample_ids": [sample.sample_id for sample in selected],
                "bands": [sample.task_family for sample in selected],
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
        scale = _scheduler_scale(step, config.warmup_steps, config.max_steps)
        for group in optimizer.param_groups:
            group["lr"] = config.learning_rate * scale
        try:
            _assert_finite_gradients(trainable, step=step_number)
            gradient_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    trainable,
                    max_norm=1.0,
                    error_if_nonfinite=True,
                ).detach()
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
        losses.append(
            {
                "step": step_number,
                "loss": sum(step_losses) / len(step_losses),
                "gradient_norm": gradient_norm,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )

    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir, safe_serialization=True)
    tokenizer.save_pretrained(adapter_dir)
    post_loss = mean_teacher_forced_loss(
        model,
        validation,
        device=device,
        pad_token_id=tokenizer.pad_token_id,
    )
    model.config.use_cache = True
    generation_summary, generation_rows = generate_validation(
        model,
        tokenizer,
        validation,
        device=device,
        max_new_tokens=config.generation_max_new_tokens,
    )
    generation_path = output_root / "validation_generations.json"
    generation_path.write_text(
        json.dumps(generation_rows, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    relative_loss_improvement = (baseline_loss - post_loss) / baseline_loss
    gates = {
        "all_losses_finite": all(math.isfinite(row["loss"]) for row in losses),
        "all_gradient_norms_finite": all(
            math.isfinite(row["gradient_norm"]) for row in losses
        ),
        "dev_loss_improved": post_loss < baseline_loss,
        "minimum_dev_loss_relative_improvement": (
            relative_loss_improvement
            >= config.minimum_dev_loss_relative_improvement
        ),
        "minimum_json_action_valid_rate": (
            generation_summary["json_action_valid_rate"]
            >= config.minimum_json_action_valid_rate
        ),
        "adapter_saved": adapter_dir.is_dir(),
        "failure_receipt_absent": not (output_root / "failure.json").exists(),
    }
    result = {
        "schema_version": "nano_train_swe_code_repair_sft_result_v1",
        "experiment_id": config.experiment_id,
        "config": {
            **config.__dict__,
            "lora_targets": list(config.lora_targets),
        },
        "selection": {
            "train_samples": len(train),
            "dev_samples": len(validation),
            "train_sample_ids_sha256": selection["train_sample_ids_sha256"],
            "dev_sample_ids_sha256": selection["dev_sample_ids_sha256"],
            "train_by_band": dict(Counter(sample.task_family for sample in train)),
            "dev_by_band": dict(Counter(sample.task_family for sample in validation)),
        },
        "dataset_file_sha256": config.dataset_file_sha256,
        "release_manifest_sha256": config.release_manifest_sha256,
        "model_config_sha256": config.model_config_sha256,
        "adapter_sha256": sha256_tree(adapter_dir),
        "trainable_parameters": sum(
            parameter.numel() for parameter in trainable
        ),
        "baseline_dev_loss": baseline_loss,
        "post_dev_loss": post_loss,
        "relative_dev_loss_improvement": relative_loss_improvement,
        "generation_validation": generation_summary,
        "loss_curve": losses,
        "train_exposure": train_exposure,
        "gates": gates,
        "candidate_admitted_for_fresh8": all(gates.values()),
        "dependencies": dependency_versions(),
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "wall_seconds": time.time() - started,
        "validation_generations_sha256": sha256_file(generation_path),
    }
    (output_root / "metrics.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def validate_reload(config: SWECodeRepairSFTConfig) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("SWE code-repair SFT reload requires CUDA")
    selection = build_selection_contract(config)
    output_root = Path(config.output_dir)
    adapter_dir = output_root / "adapter"
    metrics_path = output_root / "metrics.json"
    generations_path = output_root / "validation_generations.json"
    if (
        not adapter_dir.is_dir()
        or not metrics_path.is_file()
        or not generations_path.is_file()
        or (output_root / "failure.json").exists()
    ):
        raise ValueError("SWE code-repair reload artifacts are incomplete")
    source = json.loads(metrics_path.read_text(encoding="utf-8"))
    source_generations = json.loads(
        generations_path.read_text(encoding="utf-8")
    )
    if (
        source.get("adapter_sha256") != sha256_tree(adapter_dir)
        or source.get("experiment_id") != config.experiment_id
        or source.get("dataset_file_sha256") != config.dataset_file_sha256
        or source.get("release_manifest_sha256")
        != config.release_manifest_sha256
        or source.get("model_config_sha256") != config.model_config_sha256
    ):
        raise ValueError("SWE code-repair source artifact identity differs")
    tokenizer = AutoTokenizer.from_pretrained(
        adapter_dir,
        local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenized = tokenize_samples(
        training_dataset(selection),
        tokenizer,
        max_length=config.max_length,
    )
    by_id = {sample.sample_id: sample for sample in tokenized}
    validation = [by_id[sample_id] for sample_id in selection["dev_sample_ids"]]
    device = torch.device("cuda")
    set_seed(config.seed)
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
    loss = mean_teacher_forced_loss(
        model,
        validation,
        device=device,
        pad_token_id=tokenizer.pad_token_id,
    )
    summary, rows = generate_validation(
        model,
        tokenizer,
        validation,
        device=device,
        max_new_tokens=config.generation_max_new_tokens,
    )
    if not math.isclose(loss, source["post_dev_loss"], rel_tol=0, abs_tol=1e-6):
        raise ValueError("SWE code-repair reload dev loss differs")
    if rows != source_generations:
        raise ValueError("SWE code-repair reload generations differ")
    if summary != source["generation_validation"]:
        raise ValueError("SWE code-repair reload generation metrics differ")
    reload_generations_path = output_root / "reload_generations.json"
    reload_generations_path.write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    receipt = {
        "schema_version": "nano_train_swe_code_repair_sft_reload_v1",
        "experiment_id": config.experiment_id,
        "adapter_sha256": sha256_tree(adapter_dir),
        "reload_success": True,
        "post_dev_loss": loss,
        "source_post_dev_loss": source["post_dev_loss"],
        "generation_metrics_exact": True,
        "generations_exact": True,
        "source_generations_sha256": sha256_file(generations_path),
        "reload_generations_sha256": sha256_file(reload_generations_path),
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
    }
    (output_root / "reload_validation.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return receipt
