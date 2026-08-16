from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from torch.optim import AdamW
from transformers import AutoTokenizer, Qwen3_5ForCausalLM

from nano_train.data import collate_samples, load_analog_dataset, tokenize_samples
from nano_train.sft import (
    _assert_finite_gradients,
    _assert_finite_loss,
    _assert_finite_parameters,
    _batch_order,
    _scheduler_scale,
    _write_failure,
    dependency_versions,
    evaluate_exact,
    set_seed,
    sha256_file,
    sha256_tree,
)


@dataclass(frozen=True)
class AnchoredContinuationConfig:
    schema_version: str
    experiment_id: str
    model_path: str
    model_config_sha256: str
    anchor_adapter: str
    anchor_adapter_tree_sha256: str
    dataset_path: str
    dataset_sha256: str
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
    generation_max_new_tokens: int
    train_lora_a: bool
    train_lora_b: bool
    anchor_penalty_coefficient: float


def load_config(path: str | Path) -> AnchoredContinuationConfig:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = set(AnchoredContinuationConfig.__dataclass_fields__)
    if set(raw) != expected:
        raise ValueError("anchored continuation config fields differ")
    config = AnchoredContinuationConfig(**raw)
    if config.schema_version not in {
        "nano_train_anchored_continuation_v1",
        "nano_train_anchored_continuation_v2",
    }:
        raise ValueError("unsupported anchored continuation schema")
    if config.dtype != "float32":
        raise ValueError("anchored continuation is frozen to FP32")
    if config.train_lora_a or not config.train_lora_b:
        raise ValueError("anchored continuation must train only LoRA B")
    if config.anchor_penalty_coefficient != 1.0:
        raise ValueError("anchor penalty coefficient is frozen to 1.0")
    if config.schema_version == "nano_train_anchored_continuation_v1":
        if config.max_steps != 8 or config.warmup_steps != 1:
            raise ValueError(
                "v1 anchored continuation is frozen to 8 steps / warmup 1"
            )
    else:
        expected_v2 = {
            "experiment_id": "anchored-v1-choice-replay-continuation-v2",
            "max_steps": 4,
            "batch_size": 1,
            "gradient_accumulation_steps": 4,
            "learning_rate": 0.000025,
            "weight_decay": 0.0,
            "warmup_steps": 1,
        }
        for field, expected_value in expected_v2.items():
            if getattr(config, field) != expected_value:
                raise ValueError(
                    f"v2 anchored continuation freezes {field}="
                    f"{expected_value}"
                )
    return config


def validate_choice_replay_contract(
    dataset: dict[str, Any],
    *,
    seed: int,
    examples_seen: int,
) -> dict[str, Any]:
    if dataset.get("dataset_id") != "generic-choice-replay-v11":
        raise ValueError("v2 continuation requires generic choice replay v11")
    policy = dataset.get("policy", {})
    if (
        policy.get("contains_benchmark_content") is not False
        or policy.get("contains_model_outputs") is not False
        or policy.get("contains_teacher_outputs") is not False
        or policy.get("sealed_canary_used_for_training") is not False
        or policy.get("independent_holdout_used_for_training") is not False
        or policy.get("benchmark_feedback_used_for_training") is not False
    ):
        raise ValueError("choice replay evidence boundary differs")
    train = [
        sample for sample in dataset["samples"] if sample["split"] == "train"
    ]
    validation = [
        sample
        for sample in dataset["samples"]
        if sample["split"] == "validation"
    ]
    if len(train) != 40 or len(validation) != 32:
        raise ValueError("choice replay split counts differ")
    if any(
        sample.get("task_family") != "capability_preservation_choice"
        or sample.get("format_family") != "final_choice"
        for sample in train
    ):
        raise ValueError("choice replay train rows are not choice-only")
    order = _batch_order(train, seed)
    exposed = [train[order[index]] for index in range(examples_seen)]
    rule_counts: dict[str, int] = {}
    for sample in exposed:
        rule = str(sample.get("generation_rule", ""))
        rule_counts[rule] = rule_counts.get(rule, 0) + 1
    expected_rules = {
        "preservation_host_count_choice_v5",
        "preservation_sequential_fraction_choice_v5",
        "preservation_participant_average_choice_v5",
    }
    if set(rule_counts) != expected_rules:
        raise ValueError("choice replay exposure does not cover all rules")
    return {
        "examples_seen": len(exposed),
        "sample_ids": [str(sample["sample_id"]) for sample in exposed],
        "generation_rule_counts": rule_counts,
    }


def normalized_anchor_penalty(
    parameters: list[torch.nn.Parameter],
    anchors: list[torch.Tensor],
    anchor_norm_squared: torch.Tensor,
) -> torch.Tensor:
    if len(parameters) != len(anchors) or not parameters:
        raise ValueError("anchor parameter lists differ or are empty")
    drift = sum(
        ((parameter - anchor) ** 2).sum()
        for parameter, anchor in zip(parameters, anchors)
    )
    return 0.5 * drift / anchor_norm_squared


def run(config: AnchoredContinuationConfig) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("anchored continuation requires CUDA")
    model_path = Path(config.model_path)
    anchor_path = Path(config.anchor_adapter)
    dataset_path = Path(config.dataset_path)
    if sha256_file(model_path / "config.json") != config.model_config_sha256:
        raise ValueError("model identity mismatch")
    if sha256_tree(anchor_path) != config.anchor_adapter_tree_sha256:
        raise ValueError("anchor adapter identity mismatch")
    if sha256_file(dataset_path) != config.dataset_sha256:
        raise ValueError("dataset identity mismatch")
    set_seed(config.seed)
    output_root = Path(config.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    adapter_dir = output_root / "adapter"

    tokenizer = AutoTokenizer.from_pretrained(anchor_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset = load_analog_dataset(dataset_path)
    training_exposure = None
    if config.schema_version == "nano_train_anchored_continuation_v2":
        training_exposure = validate_choice_replay_contract(
            dataset,
            seed=config.seed,
            examples_seen=(
                config.max_steps
                * config.batch_size
                * config.gradient_accumulation_steps
            ),
        )
    samples = tokenize_samples(dataset, tokenizer, max_length=config.max_length)
    train = [sample for sample in samples if sample.split == "train"]
    validation = [sample for sample in samples if sample.split == "validation"]

    device = torch.device("cuda")
    started = time.time()
    model = Qwen3_5ForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
    ).to(device)
    model.config.use_cache = False
    model = PeftModel.from_pretrained(
        model,
        anchor_path,
        is_trainable=True,
    ).to(device)
    for name, parameter in model.named_parameters():
        if "lora_A" in name:
            parameter.requires_grad_(False)
        elif "lora_B" in name:
            parameter.requires_grad_(True)
    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    trainable_names = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    if not trainable_names or any("lora_B" not in name for name in trainable_names):
        raise ValueError("trainable parameters are not exclusively LoRA B")
    anchors = [parameter.detach().clone() for parameter in trainable]
    anchor_norm_squared = sum((anchor**2).sum() for anchor in anchors)
    if not bool(torch.isfinite(anchor_norm_squared)) or anchor_norm_squared <= 0:
        raise ValueError("anchor norm is invalid")

    baseline_metrics, baseline_rows = evaluate_exact(
        model,
        tokenizer,
        validation,
        device=device,
        max_new_tokens=config.generation_max_new_tokens,
    )
    optimizer = AdamW(
        trainable,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        eps=1e-6,
    )
    order = _batch_order(train, config.seed)
    losses = []
    optimizer.zero_grad(set_to_none=True)
    model.train()
    for step in range(config.max_steps):
        step_number = step + 1
        ce_losses = []
        penalties = []
        total_losses = []
        for micro_step in range(config.gradient_accumulation_steps):
            index = order[
                (
                    step * config.gradient_accumulation_steps
                    + micro_step
                )
                % len(order)
            ]
            batch = {
                key: value.to(device)
                for key, value in collate_samples(
                    [train[index]],
                    pad_token_id=tokenizer.pad_token_id,
                ).items()
            }
            outputs = model(**batch, use_cache=False)
            penalty = normalized_anchor_penalty(
                trainable,
                anchors,
                anchor_norm_squared,
            )
            total = outputs.loss + config.anchor_penalty_coefficient * penalty
            _assert_finite_loss(total, step=step_number)
            (total / config.gradient_accumulation_steps).backward()
            ce_losses.append(float(outputs.loss.detach().cpu()))
            penalties.append(float(penalty.detach().cpu()))
            total_losses.append(float(total.detach().cpu()))
        scale = _scheduler_scale(step, config.warmup_steps, config.max_steps)
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
                stage="anchored_continuation",
                error=error,
            )
            raise
        optimizer.zero_grad(set_to_none=True)
        losses.append(
            {
                "step": step_number,
                "ce_loss": sum(ce_losses) / len(ce_losses),
                "anchor_penalty": sum(penalties) / len(penalties),
                "total_loss": sum(total_losses) / len(total_losses),
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )

    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir, safe_serialization=True)
    tokenizer.save_pretrained(adapter_dir)
    post_metrics, post_rows = evaluate_exact(
        model,
        tokenizer,
        validation,
        device=device,
        max_new_tokens=config.generation_max_new_tokens,
    )
    drift_norm_squared = sum(
        ((parameter.detach() - anchor) ** 2).sum()
        for parameter, anchor in zip(trainable, anchors)
    )
    result = {
        "schema_version": "nano_train_anchored_continuation_result_v1",
        "experiment_id": config.experiment_id,
        "config": config.__dict__,
        "dataset": {
            "dataset_id": dataset["dataset_id"],
            "sha256": sha256_file(dataset_path),
            "train_samples": len(train),
            "validation_samples": len(validation),
        },
        "model_config_sha256": sha256_file(model_path / "config.json"),
        "anchor_adapter_tree_sha256": sha256_tree(anchor_path),
        "adapter_sha256": sha256_tree(adapter_dir),
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "trainable_tensors": len(trainable),
        "frozen_lora_a": True,
        "trainable_lora_b_only": True,
        "anchor_norm_l2": math.sqrt(float(anchor_norm_squared.detach().cpu())),
        "drift_norm_l2": math.sqrt(float(drift_norm_squared.detach().cpu())),
        "relative_drift_l2": math.sqrt(
            float((drift_norm_squared / anchor_norm_squared).detach().cpu())
        ),
        "baseline_validation": baseline_metrics,
        "post_validation": post_metrics,
        "loss_curve": losses,
        "dependencies": dependency_versions(),
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "wall_seconds": time.time() - started,
        "failure_receipt_exists": (output_root / "failure.json").exists(),
    }
    if training_exposure is not None:
        result["training_exposure"] = training_exposure
    (output_root / "generations.json").write_text(
        json.dumps(
            {"baseline": baseline_rows, "post": post_rows},
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    (output_root / "metrics.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return result
