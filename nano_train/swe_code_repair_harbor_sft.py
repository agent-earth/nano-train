from __future__ import annotations

import hashlib
import json
import math
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from torch.optim import AdamW
from transformers import AutoTokenizer, Qwen3_5ForCausalLM

from nano_train.data import collate_samples, tokenize_samples
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
from nano_train.swe_code_repair_qualification import (
    build_harbor_prompts,
    build_qualification_contract,
    generate_arm,
    load_config as load_qualification_config,
    load_terminus_parser,
    summarize_rows,
)
from nano_train.swe_code_repair_sft import (
    build_selection_contract,
    load_config as load_sft_config,
    mean_teacher_forced_loss,
    token_band,
)


CONFIG_SCHEMA = "nano_train_swe_code_repair_harbor_sft_v2"
PREREGISTER_SCHEMA = "nano_train_swe_code_repair_harbor_sft_preregister_v2"


@dataclass(frozen=True)
class SWECodeRepairHarborSFTConfig:
    schema_version: str
    experiment_id: str
    source_sft_config_path: str
    source_sft_config_sha256: str
    source_v1_metrics_path: str
    source_v1_metrics_sha256: str
    source_v1_reload_path: str
    source_v1_reload_sha256: str
    source_v1_adapter_path: str
    source_v1_adapter_sha256: str
    qualification_v1_config_path: str
    qualification_v1_config_sha256: str
    qualification_v1_preregister_path: str
    qualification_v1_preregister_sha256: str
    qualification_v1_metrics_path: str
    qualification_v1_metrics_sha256: str
    terminus_parser_path: str
    terminus_parser_sha256: str
    terminus_prompt_template_path: str
    terminus_prompt_template_sha256: str
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
    minimum_v2_loss_relative_improvement_vs_base: float
    minimum_v2_loss_relative_improvement_vs_v1: float
    minimum_v2_parser_valid: int
    maximum_v1_only_losses: int


def load_config(path: str | Path) -> SWECodeRepairHarborSFTConfig:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    expected_fields = set(SWECodeRepairHarborSFTConfig.__dataclass_fields__)
    if set(raw) != expected_fields:
        raise ValueError("Harbor-prompt SFT config fields differ")
    raw["lora_targets"] = tuple(raw["lora_targets"])
    config = SWECodeRepairHarborSFTConfig(**raw)
    validate_config(config)
    return config


def validate_config(config: SWECodeRepairHarborSFTConfig) -> None:
    expected: dict[str, Any] = {
        "schema_version": CONFIG_SCHEMA,
        "experiment_id": "evoloop-swe-code-repair-harbor-sft-v2",
        "source_sft_config_path": "configs/sft/swe_code_repair_smoke_v1.json",
        "source_v1_metrics_path": (
            "artifacts/evoloop-swe-code-repair-sft-smoke-v1/metrics.json"
        ),
        "source_v1_reload_path": (
            "artifacts/evoloop-swe-code-repair-sft-smoke-v1/"
            "reload_validation.json"
        ),
        "source_v1_adapter_path": (
            "artifacts/evoloop-swe-code-repair-sft-smoke-v1/adapter"
        ),
        "qualification_v1_config_path": (
            "configs/eval/swe_code_repair_harbor_qualification_v1.json"
        ),
        "qualification_v1_preregister_path": (
            "docs/experiments/"
            "evoloop_swe_code_repair_harbor_qualification_v1.preregister.json"
        ),
        "qualification_v1_metrics_path": (
            "artifacts/evoloop-swe-code-repair-harbor-qualification-v1/"
            "metrics.json"
        ),
        "terminus_parser_path": (
            "../harbor-swebench-v3-traex-04/src/harbor/agents/"
            "terminus_2/terminus_json_plain_parser.py"
        ),
        "terminus_prompt_template_path": (
            "../harbor-swebench-v3-traex-04/src/harbor/agents/"
            "terminus_2/templates/terminus-json-plain.txt"
        ),
        "output_dir": "artifacts/evoloop-swe-code-repair-harbor-sft-v2",
        "seed": 20260911,
        "selection_seed": "evoloop-swe-code-repair-harbor-sft-v2:20260911",
        "dtype": "float32",
        "max_length": 2816,
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
        "dev_rows_by_band": {"short": 8, "medium": 8, "long": 8},
        "minimum_v2_loss_relative_improvement_vs_base": 0.05,
        "minimum_v2_loss_relative_improvement_vs_v1": 0.02,
        "minimum_v2_parser_valid": 24,
        "maximum_v1_only_losses": 0,
    }
    for field, expected_value in expected.items():
        if getattr(config, field) != expected_value:
            raise ValueError(
                f"Harbor-prompt SFT freezes {field}={expected_value}"
            )
    for field in (
        "source_sft_config_sha256",
        "source_v1_metrics_sha256",
        "source_v1_reload_sha256",
        "source_v1_adapter_sha256",
        "qualification_v1_config_sha256",
        "qualification_v1_preregister_sha256",
        "qualification_v1_metrics_sha256",
        "terminus_parser_sha256",
        "terminus_prompt_template_sha256",
    ):
        value = getattr(config, field)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"Harbor-prompt SFT {field} is not SHA256")
    if (
        sum(config.train_rows_by_band.values())
        != config.max_steps
        * config.batch_size
        * config.gradient_accumulation_steps
    ):
        raise ValueError("Harbor-prompt SFT train exposure count differs")
    if sum(config.dev_rows_by_band.values()) != 24:
        raise ValueError("Harbor-prompt SFT dev count differs")


def _rank(seed: str, sample_id: str) -> str:
    return hashlib.sha256(f"{seed}:{sample_id}".encode()).hexdigest()


def _sha256_lines(values: list[str]) -> str:
    return hashlib.sha256("\n".join(values).encode()).hexdigest()


def _check_file(path: str, expected_sha256: str, label: str) -> None:
    if sha256_file(Path(path)) != expected_sha256:
        raise ValueError(f"Harbor-prompt SFT {label} differs")


def build_training_contract(
    config: SWECodeRepairHarborSFTConfig,
) -> dict[str, Any]:
    validate_config(config)
    for label, path, digest in (
        (
            "source config",
            config.source_sft_config_path,
            config.source_sft_config_sha256,
        ),
        (
            "source metrics",
            config.source_v1_metrics_path,
            config.source_v1_metrics_sha256,
        ),
        (
            "source reload",
            config.source_v1_reload_path,
            config.source_v1_reload_sha256,
        ),
        (
            "qualification config",
            config.qualification_v1_config_path,
            config.qualification_v1_config_sha256,
        ),
        (
            "qualification preregistration",
            config.qualification_v1_preregister_path,
            config.qualification_v1_preregister_sha256,
        ),
        (
            "qualification metrics",
            config.qualification_v1_metrics_path,
            config.qualification_v1_metrics_sha256,
        ),
        (
            "Terminus parser",
            config.terminus_parser_path,
            config.terminus_parser_sha256,
        ),
        (
            "Terminus prompt",
            config.terminus_prompt_template_path,
            config.terminus_prompt_template_sha256,
        ),
    ):
        _check_file(path, digest, label)
    if sha256_tree(Path(config.source_v1_adapter_path)) != (
        config.source_v1_adapter_sha256
    ):
        raise ValueError("Harbor-prompt SFT source v1 adapter differs")

    source_config = load_sft_config(config.source_sft_config_path)
    source_selection = build_selection_contract(source_config)
    if (
        source_config.seed != config.seed
        or source_config.max_steps != config.max_steps
        or source_config.batch_size != config.batch_size
        or source_config.gradient_accumulation_steps
        != config.gradient_accumulation_steps
        or source_config.learning_rate != config.learning_rate
        or source_config.weight_decay != config.weight_decay
        or source_config.warmup_steps != config.warmup_steps
        or source_config.lora_r != config.lora_r
        or source_config.lora_alpha != config.lora_alpha
        or source_config.lora_dropout != config.lora_dropout
        or source_config.lora_targets != config.lora_targets
        or source_config.gradient_checkpointing
        != config.gradient_checkpointing
        or source_config.train_rows_by_band != config.train_rows_by_band
    ):
        raise ValueError("Harbor-prompt SFT single-variable contract differs")

    qualification_config = load_qualification_config(
        config.qualification_v1_config_path
    )
    qualification = build_qualification_contract(qualification_config)
    qualification_metrics = json.loads(
        Path(config.qualification_v1_metrics_path).read_text(encoding="utf-8")
    )
    if (
        qualification_metrics.get("candidate_admitted_for_fresh8") is not False
        or qualification_metrics.get("selection", {}).get(
            "dev_sample_ids_sha256"
        )
        != qualification["selected_dev_ids_sha256"]
    ):
        raise ValueError("Harbor-prompt SFT qualification evidence differs")

    excluded_ids = set(source_selection["dev_sample_ids"]) | set(
        qualification["selected_dev_ids"]
    )
    candidates = []
    with Path(source_config.dataset_path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if (
                    row.get("split") == "dev"
                    and row.get("sample_id") not in excluded_ids
                ):
                    candidates.append(row)
    selected_dev = []
    for band in ("short", "medium", "long"):
        band_rows = [
            row
            for row in candidates
            if token_band(int(row["token_count"])) == band
        ]
        band_rows.sort(
            key=lambda row: (
                _rank(config.selection_seed, row["sample_id"]),
                row["sample_id"],
            )
        )
        chosen = band_rows[: config.dev_rows_by_band[band]]
        if len(chosen) != config.dev_rows_by_band[band]:
            raise ValueError(f"insufficient third fresh dev rows for {band}")
        selected_dev.extend(chosen)
    selected_dev.sort(key=lambda row: row["sample_id"])
    selected_dev_ids = [row["sample_id"] for row in selected_dev]
    if set(selected_dev_ids) & excluded_ids:
        raise ValueError("Harbor-prompt SFT fresh dev overlaps prior dev")
    return {
        "source_config": source_config,
        "selected_train": source_selection["selected_train"],
        "train_sample_ids": source_selection["train_sample_ids"],
        "train_sample_ids_sha256": source_selection[
            "train_sample_ids_sha256"
        ],
        "selected_dev": selected_dev,
        "dev_sample_ids": selected_dev_ids,
        "dev_sample_ids_sha256": _sha256_lines(selected_dev_ids),
        "excluded_dev_ids_sha256": _sha256_lines(sorted(excluded_ids)),
    }


def harbor_training_dataset(
    rows: list[dict[str, Any]],
    *,
    prompt_template: str,
    split: str,
) -> dict[str, Any]:
    samples = []
    for row in rows:
        messages = row["messages"]
        rendered = prompt_template.format(
            instruction=messages[1]["content"],
            terminal_state="",
        )
        samples.append(
            {
                "sample_id": row["sample_id"],
                "split": split,
                "task_family": token_band(int(row["token_count"])),
                "format_family": "swe_code_repair_harbor_action",
                "messages": [
                    {"role": "user", "content": rendered},
                    {"role": "assistant", "content": messages[-1]["content"]},
                ],
                "verifier": row["verifier"],
                "task_spec": row["task_spec"],
            }
        )
    return {"dataset_id": "evoloop-swe-code-repair-harbor-sft-v2", "samples": samples}


def compare_actionable(
    baseline_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
) -> dict[str, int]:
    baseline = {
        row["sample_id_sha256"]: row["terminus_actionable_or_complete"]
        for row in baseline_rows
    }
    candidate = {
        row["sample_id_sha256"]: row["terminus_actionable_or_complete"]
        for row in candidate_rows
    }
    if set(baseline) != set(candidate) or len(baseline) != 24:
        raise ValueError("Harbor-prompt SFT comparison cases differ")
    return {
        "candidate_only": sum(
            not baseline[case_id] and candidate[case_id]
            for case_id in baseline
        ),
        "baseline_only": sum(
            baseline[case_id] and not candidate[case_id]
            for case_id in baseline
        ),
        "both": sum(
            baseline[case_id] and candidate[case_id]
            for case_id in baseline
        ),
        "neither": sum(
            not baseline[case_id] and not candidate[case_id]
            for case_id in baseline
        ),
    }


def admission_gates(
    *,
    base_loss: float,
    v1_loss: float,
    v2_loss: float,
    v1_rows: list[dict[str, Any]],
    v2_rows: list[dict[str, Any]],
    config: SWECodeRepairHarborSFTConfig,
) -> dict[str, bool]:
    v1_v2 = compare_actionable(v1_rows, v2_rows)
    base_improvement = (base_loss - v2_loss) / base_loss
    v1_improvement = (v1_loss - v2_loss) / v1_loss
    return {
        "finite_losses": all(
            math.isfinite(value) for value in (base_loss, v1_loss, v2_loss)
        ),
        "minimum_v2_loss_relative_improvement_vs_base": (
            base_improvement
            >= config.minimum_v2_loss_relative_improvement_vs_base
        ),
        "minimum_v2_loss_relative_improvement_vs_v1": (
            v1_improvement
            >= config.minimum_v2_loss_relative_improvement_vs_v1
        ),
        "minimum_v2_parser_valid": (
            sum(row["terminus_parser_valid"] for row in v2_rows)
            >= config.minimum_v2_parser_valid
        ),
        "maximum_v1_only_losses": (
            v1_v2["baseline_only"] <= config.maximum_v1_only_losses
        ),
        "per_band_non_regression_vs_v1": all(
            sum(
                row["task_family"] == band
                and row["terminus_actionable_or_complete"]
                for row in v2_rows
            )
            >= sum(
                row["task_family"] == band
                and row["terminus_actionable_or_complete"]
                for row in v1_rows
            )
            for band in ("short", "medium", "long")
        ),
        "v2_generation_budget_not_exhausted": not any(
            row["generation_budget_exhausted"] for row in v2_rows
        ),
        "source_v1_reload_exact": True,
        "private_content_absent": True,
    }


def run(config: SWECodeRepairHarborSFTConfig) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("Harbor-prompt SFT requires one CUDA GPU")
    contract = build_training_contract(config)
    source_config = contract["source_config"]
    prompt_template = Path(config.terminus_prompt_template_path).read_text(
        encoding="utf-8"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        source_config.model_path,
        local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset = {
        "dataset_id": "evoloop-swe-code-repair-harbor-sft-v2",
        "samples": (
            harbor_training_dataset(
                contract["selected_train"],
                prompt_template=prompt_template,
                split="train",
            )["samples"]
            + harbor_training_dataset(
                contract["selected_dev"],
                prompt_template=prompt_template,
                split="validation",
            )["samples"]
        ),
    }
    tokenized = tokenize_samples(dataset, tokenizer, max_length=config.max_length)
    by_id = {sample.sample_id: sample for sample in tokenized}
    train = [by_id[sample_id] for sample_id in contract["train_sample_ids"]]
    validation = [by_id[sample_id] for sample_id in contract["dev_sample_ids"]]
    if len(train) != 160 or len(validation) != 24:
        raise ValueError("Harbor-prompt SFT selected sample count differs")

    set_seed(config.seed)
    device = torch.device("cuda")
    output_root = Path(config.output_dir)
    adapter_dir = output_root / "adapter"
    output_root.mkdir(parents=True, exist_ok=True)
    started = time.time()
    model = Qwen3_5ForCausalLM.from_pretrained(
        source_config.model_path,
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
    optimizer = AdamW(
        trainable,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        eps=1e-6,
    )
    optimizer.zero_grad(set_to_none=True)
    loss_curve = []
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
        loss_curve.append(
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
    model.config.use_cache = True
    terminus_parser = load_terminus_parser(Path(config.terminus_parser_path))
    prompts = build_harbor_prompts(
        tokenizer,
        contract["selected_dev"],
        prompt_template,
    )
    with model.disable_adapter():
        base_loss = mean_teacher_forced_loss(
            model,
            validation,
            device=device,
            pad_token_id=tokenizer.pad_token_id,
        )
    base_rows = generate_arm(
        model,
        tokenizer,
        prompts,
        arm="base",
        max_new_tokens=config.generation_max_new_tokens,
        terminus_parser=terminus_parser,
    )
    v2_loss = mean_teacher_forced_loss(
        model,
        validation,
        device=device,
        pad_token_id=tokenizer.pad_token_id,
    )
    v2_rows = generate_arm(
        model,
        tokenizer,
        prompts,
        arm="adapter",
        max_new_tokens=config.generation_max_new_tokens,
        terminus_parser=terminus_parser,
    )
    model.load_adapter(
        config.source_v1_adapter_path,
        adapter_name="standard_sft_v1",
        is_trainable=False,
    )
    model.set_adapter("standard_sft_v1")
    v1_loss = mean_teacher_forced_loss(
        model,
        validation,
        device=device,
        pad_token_id=tokenizer.pad_token_id,
    )
    v1_rows = generate_arm(
        model,
        tokenizer,
        prompts,
        arm="adapter",
        max_new_tokens=config.generation_max_new_tokens,
        terminus_parser=terminus_parser,
    )
    gates = admission_gates(
        base_loss=base_loss,
        v1_loss=v1_loss,
        v2_loss=v2_loss,
        v1_rows=v1_rows,
        v2_rows=v2_rows,
        config=config,
    )
    rows_path = output_root / "structural_rows.json"
    rows_path.write_text(
        json.dumps(
            {"base": base_rows, "standard_sft_v1": v1_rows, "harbor_sft_v2": v2_rows},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    result = {
        "schema_version": "nano_train_swe_code_repair_harbor_sft_result_v2",
        "experiment_id": config.experiment_id,
        "config": {
            **config.__dict__,
            "lora_targets": list(config.lora_targets),
        },
        "selection": {
            "train_samples": len(train),
            "dev_samples": len(validation),
            "train_sample_ids_sha256": contract["train_sample_ids_sha256"],
            "dev_sample_ids_sha256": contract["dev_sample_ids_sha256"],
            "excluded_prior_dev_ids_sha256": contract[
                "excluded_dev_ids_sha256"
            ],
            "train_by_band": dict(Counter(sample.task_family for sample in train)),
            "dev_by_band": dict(
                Counter(sample.task_family for sample in validation)
            ),
        },
        "loss": {
            "base": base_loss,
            "standard_sft_v1": v1_loss,
            "harbor_sft_v2": v2_loss,
            "v2_relative_improvement_vs_base": (
                (base_loss - v2_loss) / base_loss
            ),
            "v2_relative_improvement_vs_v1": (
                (v1_loss - v2_loss) / v1_loss
            ),
        },
        "structure": {
            "base": summarize_rows(base_rows),
            "standard_sft_v1": summarize_rows(v1_rows),
            "harbor_sft_v2": summarize_rows(v2_rows),
            "v2_vs_base": compare_actionable(base_rows, v2_rows),
            "v2_vs_v1": compare_actionable(v1_rows, v2_rows),
        },
        "training": {
            "optimizer_steps": config.max_steps,
            "trainable_parameters": sum(
                parameter.numel() for parameter in trainable
            ),
            "loss_curve": loss_curve,
            "train_exposure": train_exposure,
            "all_losses_finite": all(
                math.isfinite(row["loss"]) for row in loss_curve
            ),
            "all_gradient_norms_finite": all(
                math.isfinite(row["gradient_norm"]) for row in loss_curve
            ),
        },
        "gates": gates,
        "candidate_admitted_for_fresh8": all(gates.values()),
        "adapter_sha256": sha256_tree(adapter_dir),
        "structural_rows_sha256": sha256_file(rows_path),
        "dependencies": dependency_versions(),
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "wall_seconds": time.time() - started,
        "failure_receipt_exists": (output_root / "failure.json").exists(),
        "private_content_recorded": False,
    }
    (output_root / "metrics.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def validate_reload(config: SWECodeRepairHarborSFTConfig) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("Harbor-prompt SFT reload requires CUDA")
    contract = build_training_contract(config)
    source_config = contract["source_config"]
    output_root = Path(config.output_dir)
    adapter_dir = output_root / "adapter"
    metrics_path = output_root / "metrics.json"
    rows_path = output_root / "structural_rows.json"
    if (
        not adapter_dir.is_dir()
        or not metrics_path.is_file()
        or not rows_path.is_file()
        or (output_root / "failure.json").exists()
    ):
        raise ValueError("Harbor-prompt SFT reload artifacts are incomplete")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    source_rows = json.loads(rows_path.read_text(encoding="utf-8"))
    if (
        metrics.get("adapter_sha256") != sha256_tree(adapter_dir)
        or metrics.get("experiment_id") != config.experiment_id
        or metrics.get("structural_rows_sha256") != sha256_file(rows_path)
    ):
        raise ValueError("Harbor-prompt SFT source artifact identity differs")

    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset = harbor_training_dataset(
        contract["selected_dev"],
        prompt_template=Path(config.terminus_prompt_template_path).read_text(
            encoding="utf-8"
        ),
        split="validation",
    )
    tokenized = tokenize_samples(dataset, tokenizer, max_length=config.max_length)
    by_id = {sample.sample_id: sample for sample in tokenized}
    validation = [by_id[sample_id] for sample_id in contract["dev_sample_ids"]]
    prompts = build_harbor_prompts(
        tokenizer,
        contract["selected_dev"],
        Path(config.terminus_prompt_template_path).read_text(encoding="utf-8"),
    )
    set_seed(config.seed)
    model = Qwen3_5ForCausalLM.from_pretrained(
        source_config.model_path,
        local_files_only=True,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
    ).cuda()
    model = PeftModel.from_pretrained(
        model,
        adapter_dir,
        is_trainable=False,
    ).cuda()
    loss = mean_teacher_forced_loss(
        model,
        validation,
        device=torch.device("cuda"),
        pad_token_id=tokenizer.pad_token_id,
    )
    rows = generate_arm(
        model,
        tokenizer,
        prompts,
        arm="adapter",
        max_new_tokens=config.generation_max_new_tokens,
        terminus_parser=load_terminus_parser(Path(config.terminus_parser_path)),
    )
    loss_exact = math.isclose(
        loss,
        metrics["loss"]["harbor_sft_v2"],
        rel_tol=0,
        abs_tol=1e-6,
    )
    generations_exact = rows == source_rows["harbor_sft_v2"]
    if not loss_exact or not generations_exact:
        raise ValueError("Harbor-prompt SFT reload differs")
    reload_rows_path = output_root / "reload_structural_rows.json"
    reload_rows_path.write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    receipt = {
        "schema_version": "nano_train_swe_code_repair_harbor_sft_reload_v2",
        "experiment_id": config.experiment_id,
        "adapter_sha256": sha256_tree(adapter_dir),
        "reload_success": True,
        "loss_exact": True,
        "generations_exact": True,
        "source_structural_rows_sha256": sha256_file(rows_path),
        "reload_structural_rows_sha256": sha256_file(reload_rows_path),
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
    }
    (output_root / "reload_validation.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return receipt
