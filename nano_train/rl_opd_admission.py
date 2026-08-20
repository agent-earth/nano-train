from __future__ import annotations

import hashlib
import json
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as functional
from peft import LoraConfig, PeftModel, get_peft_model
from safetensors import safe_open
from torch.optim import AdamW
from transformers import AutoTokenizer, Qwen3_5ForCausalLM

from nano_train.data import evaluate_arithmetic, format_number
from nano_train.sft import (
    _assert_finite_gradients,
    _assert_finite_loss,
    _assert_finite_parameters,
    _trainable_parameters,
    _write_failure,
    dependency_versions,
    set_seed,
    sha256_file,
    sha256_tree,
)


CONFIG_SCHEMA = "nano_train_rl_opd_admission_v1"
RESULT_SCHEMA = "nano_train_rl_opd_admission_result_v1"
RELOAD_SCHEMA = "nano_train_rl_opd_admission_reload_v1"
FINAL_PATTERN = re.compile(r"^FINAL: ([-+]?[0-9]+)$")


@dataclass(frozen=True)
class AdmissionConfig:
    schema_version: str
    experiment_id: str
    mode: str
    student_model_path: str
    student_model_config_sha256: str
    student_model_index_sha256: str
    student_weight_shards: tuple[dict[str, Any], ...]
    teacher_model_path: str | None
    teacher_model_config_sha256: str | None
    teacher_model_index_sha256: str | None
    teacher_weight_shards: tuple[dict[str, Any], ...]
    output_dir: str
    seed: int
    student_dtype: str
    teacher_dtype: str | None
    max_steps: int
    learning_rate: float
    weight_decay: float
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    lora_targets: tuple[str, ...]
    gradient_checkpointing: bool
    rollout_temperature: float
    rollout_top_p: float
    rollout_max_new_tokens: int
    reference_kl_weight: float
    prompt_template: str
    system_prompt: str
    train_tasks: tuple[dict[str, Any], ...]
    probe_tasks: tuple[dict[str, Any], ...]
    benchmark_sources: tuple[dict[str, str], ...]
    policy: dict[str, bool]


def load_config(path: str | Path) -> AdmissionConfig:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = set(AdmissionConfig.__dataclass_fields__)
    if set(raw) != expected:
        raise ValueError("RL/OPD admission config fields differ")
    raw["lora_targets"] = tuple(raw["lora_targets"])
    raw["train_tasks"] = tuple(raw["train_tasks"])
    raw["probe_tasks"] = tuple(raw["probe_tasks"])
    raw["benchmark_sources"] = tuple(raw["benchmark_sources"])
    raw["student_weight_shards"] = tuple(raw["student_weight_shards"])
    raw["teacher_weight_shards"] = tuple(raw["teacher_weight_shards"])
    config = AdmissionConfig(**raw)
    validate_config(config)
    return config


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _validate_tasks(tasks: tuple[dict[str, Any], ...], label: str) -> None:
    if not tasks:
        raise ValueError(f"{label} tasks are empty")
    ids = []
    expressions = []
    for task in tasks:
        if set(task) != {"task_id", "expression", "expected"}:
            raise ValueError(f"{label} task fields differ")
        task_id = str(task["task_id"])
        expression = str(task["expression"])
        expected = str(task["expected"])
        actual = format_number(evaluate_arithmetic(expression))
        if not task_id or actual != expected or not re.fullmatch(
            r"[-+]?[0-9]+", expected
        ):
            raise ValueError(f"{label} task is invalid: {task_id}")
        ids.append(task_id)
        expressions.append(expression)
    if len(ids) != len(set(ids)) or len(expressions) != len(set(expressions)):
        raise ValueError(f"{label} tasks are not unique")


def validate_config(config: AdmissionConfig) -> None:
    if config.schema_version != CONFIG_SCHEMA:
        raise ValueError("unsupported RL/OPD admission schema")
    if config.mode not in {"rl", "opd"}:
        raise ValueError("admission mode must be rl or opd")
    expected = {
        "seed": 20260820,
        "student_dtype": "float32",
        "max_steps": 2,
        "learning_rate": 0.00001,
        "weight_decay": 0.0,
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.0,
        "lora_targets": ("q_proj", "v_proj"),
        "gradient_checkpointing": True,
        "rollout_temperature": 0.8,
        "rollout_top_p": 0.95,
        "rollout_max_new_tokens": 12,
        "prompt_template": (
            "Compute this synthetic expression exactly: {expression}. "
            "Return only FINAL: <integer>."
        ),
        "system_prompt": (
            "Follow the exact output contract. Do not use tools or external "
            "information."
        ),
    }
    for field, value in expected.items():
        if getattr(config, field) != value:
            raise ValueError(f"RL/OPD admission freezes {field}={value}")
    if config.mode == "rl":
        if (
            config.teacher_model_path is not None
            or config.teacher_model_config_sha256 is not None
            or config.teacher_model_index_sha256 is not None
            or config.teacher_weight_shards
            or config.teacher_dtype is not None
            or config.reference_kl_weight != 0.02
        ):
            raise ValueError("RL admission teacher/reference contract differs")
    else:
        if (
            not config.teacher_model_path
            or not _is_sha256(config.teacher_model_config_sha256)
            or not _is_sha256(config.teacher_model_index_sha256)
            or len(config.teacher_weight_shards) != 4
            or config.teacher_dtype != "float16"
            or config.reference_kl_weight != 0.0
        ):
            raise ValueError("OPD admission teacher contract differs")
    if (
        not _is_sha256(config.student_model_config_sha256)
        or not _is_sha256(config.student_model_index_sha256)
        or len(config.student_weight_shards) != 2
    ):
        raise ValueError("student model identity differs")
    for label, shards in (
        ("student", config.student_weight_shards),
        ("teacher", config.teacher_weight_shards),
    ):
        for shard in shards:
            if set(shard) != {"name", "bytes", "sha256"}:
                raise ValueError(f"{label} model shard fields differ")
            if (
                not str(shard["name"]).endswith(".safetensors")
                or int(shard["bytes"]) <= 0
                or not _is_sha256(shard["sha256"])
            ):
                raise ValueError(f"{label} model shard identity differs")
    if len(config.train_tasks) != config.max_steps or len(config.probe_tasks) != 2:
        raise ValueError("RL/OPD admission task counts differ")
    _validate_tasks(config.train_tasks, "train")
    _validate_tasks(config.probe_tasks, "probe")
    if set(task["task_id"] for task in config.train_tasks) & set(
        task["task_id"] for task in config.probe_tasks
    ):
        raise ValueError("train and probe task IDs overlap")
    if len(config.benchmark_sources) != 3:
        raise ValueError("three benchmark sources are required")
    for source in config.benchmark_sources:
        if set(source) != {"name", "path", "sha256", "prompt_column"}:
            raise ValueError("benchmark source fields differ")
        if not _is_sha256(source["sha256"]):
            raise ValueError("benchmark source SHA256 differs")
    required_policy = {
        "contains_benchmark_rows": False,
        "contains_benchmark_outputs": False,
        "contains_canary_rows": False,
        "contains_holdout_rows": False,
        "uses_model_generated_rollouts": True,
        "training_allowed": True,
        "quality_claim_allowed": False,
        "benchmark_access_allowed_after_smoke": False,
    }
    if config.policy != required_policy:
        raise ValueError("RL/OPD admission policy differs")


def task_prompt(config: AdmissionConfig, task: dict[str, Any]) -> str:
    return config.prompt_template.format(expression=task["expression"])


def _normalize_text(value: str) -> str:
    return " ".join(value.casefold().split())


def _sha256_lines(values: list[str]) -> str:
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def build_contamination_audit(
    config: AdmissionConfig,
) -> dict[str, Any]:
    import pyarrow.parquet as parquet

    synthetic = [
        _normalize_text(task_prompt(config, task))
        for task in (*config.train_tasks, *config.probe_tasks)
    ]
    synthetic_hashes = {
        hashlib.sha256(value.encode("utf-8")).hexdigest()
        for value in synthetic
    }
    benchmark_hashes: set[str] = set()
    benchmark_rows = {}
    for source in config.benchmark_sources:
        path = Path(source["path"])
        if sha256_file(path) != source["sha256"]:
            raise ValueError(f"benchmark identity mismatch: {source['name']}")
        table = parquet.read_table(path, columns=[source["prompt_column"]])
        prompts = [
            _normalize_text(str(value))
            for value in table[source["prompt_column"]].to_pylist()
        ]
        benchmark_rows[source["name"]] = len(prompts)
        benchmark_hashes.update(
            hashlib.sha256(value.encode("utf-8")).hexdigest()
            for value in prompts
        )
    overlap = synthetic_hashes & benchmark_hashes
    if overlap:
        raise ValueError("synthetic admission prompts overlap benchmark rows")
    train_hashes = {
        hashlib.sha256(
            _normalize_text(task_prompt(config, task)).encode("utf-8")
        ).hexdigest()
        for task in config.train_tasks
    }
    probe_hashes = synthetic_hashes - train_hashes
    if train_hashes & probe_hashes:
        raise ValueError("train and probe prompt hashes overlap")
    return {
        "synthetic_prompts": len(synthetic),
        "synthetic_prompt_sha256": _sha256_lines(sorted(synthetic_hashes)),
        "train_prompt_sha256": _sha256_lines(sorted(train_hashes)),
        "probe_prompt_sha256": _sha256_lines(sorted(probe_hashes)),
        "benchmark_rows_hashed": benchmark_rows,
        "exact_normalized_prompt_overlap": 0,
        "benchmark_labels_loaded": False,
        "benchmark_outputs_loaded": False,
        "canary_or_holdout_loaded": False,
        "passed": True,
    }


def _chat_prompt(
    tokenizer: Any,
    config: AdmissionConfig,
    task: dict[str, Any],
) -> list[int]:
    text = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": config.system_prompt},
            {"role": "user", "content": task_prompt(config, task)},
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return tokenizer(text, add_special_tokens=False).input_ids


def verifier_reward(output: str, expected: str) -> float:
    match = FINAL_PATTERN.fullmatch(output.strip())
    if match is None:
        return -1.0
    return 1.0 if match.group(1) == expected else -0.25


def rollout_prediction_logits(
    logits: torch.Tensor,
    prompt_length: int,
    rollout_length: int,
) -> torch.Tensor:
    if (
        logits.ndim != 3
        or prompt_length < 1
        or rollout_length < 1
        or logits.shape[1] < prompt_length + rollout_length
    ):
        raise ValueError("rollout prediction logits shape differs")
    return logits[:, prompt_length - 1 : prompt_length + rollout_length - 1]


def reinforce_loss(
    rollout_logits: torch.Tensor,
    rollout_ids: torch.Tensor,
    *,
    reward: float,
) -> torch.Tensor:
    if rollout_logits.shape[:2] != rollout_ids.shape:
        raise ValueError("REINFORCE rollout shape differs")
    log_probability = functional.log_softmax(
        rollout_logits,
        dim=-1,
    ).gather(-1, rollout_ids.unsqueeze(-1)).squeeze(-1)
    return -float(reward) * log_probability.mean()


def distillation_kl(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
) -> torch.Tensor:
    if teacher_logits.shape != student_logits.shape:
        raise ValueError("OPD teacher/student logits shape differs")
    teacher_probability = functional.softmax(
        teacher_logits.detach().float(),
        dim=-1,
    )
    student_log_probability = functional.log_softmax(
        student_logits.float(),
        dim=-1,
    )
    return functional.kl_div(
        student_log_probability,
        teacher_probability,
        reduction="batchmean",
    )


@torch.inference_mode()
def logits_fingerprint(
    model: Any,
    tokenizer: Any,
    config: AdmissionConfig,
) -> str:
    model.eval()
    task = config.probe_tasks[0]
    prompt_ids = _chat_prompt(tokenizer, config, task)
    input_ids = torch.tensor(
        [prompt_ids],
        dtype=torch.long,
        device=next(model.parameters()).device,
    )
    logits = model(input_ids=input_ids, use_cache=False).logits[:, -1, :]
    values = logits.detach().float().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(values).hexdigest()


def _load_student(
    config: AdmissionConfig,
    device: torch.device,
) -> tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(
        config.student_model_path,
        local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = Qwen3_5ForCausalLM.from_pretrained(
        config.student_model_path,
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
    return model, tokenizer


def _rollout(
    model: Any,
    tokenizer: Any,
    config: AdmissionConfig,
    task: dict[str, Any],
    *,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, str]:
    model.eval()
    torch.manual_seed(seed)
    prompt_ids = _chat_prompt(tokenizer, config, task)
    device = next(model.parameters()).device
    prompt = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    mask = torch.ones_like(prompt)
    with torch.inference_mode():
        sequence = model.generate(
            input_ids=prompt,
            attention_mask=mask,
            do_sample=True,
            temperature=config.rollout_temperature,
            top_p=config.rollout_top_p,
            max_new_tokens=config.rollout_max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
        )
    rollout = sequence[:, prompt.shape[1] :]
    if rollout.shape[1] < 1:
        raise ValueError("student rollout is empty")
    output = tokenizer.decode(
        rollout[0],
        skip_special_tokens=True,
    ).strip()
    return prompt, rollout, output


def _load_teacher(config: AdmissionConfig) -> Any:
    if config.mode != "opd":
        return None
    if torch.cuda.device_count() < 2:
        raise RuntimeError("OPD admission requires two CUDA GPUs")
    teacher = Qwen3_5ForCausalLM.from_pretrained(
        config.teacher_model_path,
        local_files_only=True,
        dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to(torch.device("cuda:1"))
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    return teacher


def _verify_model_identity(
    path: Path,
    *,
    config_sha256: str,
    index_sha256: str,
    shards: tuple[dict[str, Any], ...],
    label: str,
) -> dict[str, Any]:
    if (
        sha256_file(path / "config.json") != config_sha256
        or sha256_file(path / "model.safetensors.index.json") != index_sha256
    ):
        raise ValueError(f"{label} model metadata identity mismatch")
    verified_shards = []
    for shard in shards:
        shard_path = path / shard["name"]
        if (
            shard_path.stat().st_size != shard["bytes"]
            or sha256_file(shard_path) != shard["sha256"]
        ):
            raise ValueError(f"{label} model shard identity mismatch")
        verified_shards.append({**shard, "verified": True})
    return {
        "config_sha256": config_sha256,
        "index_sha256": index_sha256,
        "shards": verified_shards,
    }


def model_identity_checks(config: AdmissionConfig) -> dict[str, Any]:
    student = _verify_model_identity(
        Path(config.student_model_path),
        config_sha256=config.student_model_config_sha256,
        index_sha256=config.student_model_index_sha256,
        shards=config.student_weight_shards,
        label="student",
    )
    teacher = None
    if config.mode == "opd":
        teacher = _verify_model_identity(
            Path(config.teacher_model_path or ""),
            config_sha256=str(config.teacher_model_config_sha256),
            index_sha256=str(config.teacher_model_index_sha256),
            shards=config.teacher_weight_shards,
            label="teacher",
        )
    return {"student": student, "teacher": teacher}


def run(config: AdmissionConfig) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("RL/OPD admission requires CUDA")
    model_identities = model_identity_checks(config)
    contamination = build_contamination_audit(config)
    set_seed(config.seed)
    output_root = Path(config.output_dir)
    adapter_dir = output_root / "adapter"
    output_root.mkdir(parents=True, exist_ok=True)
    student_device = torch.device("cuda:0")
    torch.cuda.set_device(student_device)
    started = time.time()
    student, tokenizer = _load_student(config, student_device)
    teacher = _load_teacher(config)
    trainable = _trainable_parameters(student)
    optimizer = AdamW(
        trainable,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        eps=1e-6,
    )
    before_fingerprint = logits_fingerprint(
        student,
        tokenizer,
        config,
    )
    trajectory_rows = []
    loss_curve = []
    optimizer.zero_grad(set_to_none=True)
    for step_index in range(config.max_steps):
        step = step_index + 1
        task = config.train_tasks[step_index]
        prompt, rollout, output = _rollout(
            student,
            tokenizer,
            config,
            task,
            seed=config.seed + step,
        )
        full_ids = torch.cat([prompt, rollout], dim=1)
        attention_mask = torch.ones_like(full_ids)
        student.train()
        student_outputs = student(
            input_ids=full_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        student_rollout_logits = rollout_prediction_logits(
            student_outputs.logits,
            prompt.shape[1],
            rollout.shape[1],
        )
        reward = verifier_reward(output, str(task["expected"]))
        if config.mode == "rl":
            policy_loss = reinforce_loss(
                student_rollout_logits,
                rollout,
                reward=reward,
            )
            with student.disable_adapter(), torch.inference_mode():
                reference_outputs = student(
                    input_ids=full_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                )
            reference_logits = rollout_prediction_logits(
                reference_outputs.logits,
                prompt.shape[1],
                rollout.shape[1],
            )
            regularizer = distillation_kl(
                reference_logits,
                student_rollout_logits,
            )
            total_loss = (
                policy_loss + config.reference_kl_weight * regularizer
            )
            components = {
                "policy_loss": float(policy_loss.detach().cpu()),
                "reference_kl": float(regularizer.detach().cpu()),
            }
        else:
            teacher_ids = full_ids.to(torch.device("cuda:1"))
            with torch.inference_mode():
                teacher_outputs = teacher(
                    input_ids=teacher_ids,
                    attention_mask=torch.ones_like(teacher_ids),
                    use_cache=False,
                )
            teacher_logits = rollout_prediction_logits(
                teacher_outputs.logits,
                prompt.shape[1],
                rollout.shape[1],
            ).to(student_device)
            total_loss = distillation_kl(
                teacher_logits,
                student_rollout_logits,
            )
            components = {
                "teacher_student_kl": float(
                    total_loss.detach().cpu()
                ),
            }
        try:
            _assert_finite_loss(total_loss, step=step)
            total_loss.backward()
            _assert_finite_gradients(trainable, step=step)
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                trainable,
                max_norm=1.0,
                error_if_nonfinite=True,
            )
            optimizer.step()
            _assert_finite_parameters(trainable, step=step)
        except (FloatingPointError, RuntimeError) as error:
            _write_failure(
                output_root,
                step=step,
                stage=f"{config.mode}_admission",
                error=error,
            )
            raise
        optimizer.zero_grad(set_to_none=True)
        trajectory_rows.append(
            {
                "step": step,
                "task_id": task["task_id"],
                "prompt": task_prompt(config, task),
                "output": output,
                "expected": task["expected"],
                "reward": reward,
                "rollout_tokens": int(rollout.shape[1]),
            }
        )
        loss_curve.append(
            {
                "step": step,
                "total_loss": float(total_loss.detach().cpu()),
                "gradient_norm": float(gradient_norm.detach().cpu()),
                "learning_rate": optimizer.param_groups[0]["lr"],
                **components,
            }
        )
    adapter_dir.mkdir(parents=True, exist_ok=True)
    student.save_pretrained(adapter_dir, safe_serialization=True)
    tokenizer.save_pretrained(adapter_dir)
    after_fingerprint = logits_fingerprint(
        student,
        tokenizer,
        config,
    )
    trajectories_path = output_root / "trajectories.json"
    trajectories_path.write_text(
        json.dumps(trajectory_rows, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    result = {
        "schema_version": RESULT_SCHEMA,
        "experiment_id": config.experiment_id,
        "mode": config.mode,
        "config": {
            **config.__dict__,
            "lora_targets": list(config.lora_targets),
            "train_tasks": list(config.train_tasks),
            "probe_tasks": list(config.probe_tasks),
            "benchmark_sources": list(config.benchmark_sources),
        },
        "identity": {
            "student_model_config_sha256": (
                config.student_model_config_sha256
            ),
            "student_model_index_sha256": (
                config.student_model_index_sha256
            ),
            "teacher_model_config_sha256": (
                config.teacher_model_config_sha256
            ),
            "teacher_model_index_sha256": (
                config.teacher_model_index_sha256
            ),
            "adapter_sha256": sha256_tree(adapter_dir),
            "trajectories_sha256": sha256_file(trajectories_path),
        },
        "model_identities": model_identities,
        "contamination_audit": contamination,
        "training": {
            "optimizer_steps": config.max_steps,
            "trainable_parameters": sum(
                parameter.numel() for parameter in trainable
            ),
            "loss_curve": loss_curve,
            "all_losses_finite": all(
                math.isfinite(row["total_loss"]) for row in loss_curve
            ),
            "all_gradient_norms_finite": all(
                math.isfinite(row["gradient_norm"]) for row in loss_curve
            ),
        },
        "adapter_effect": {
            "before_probe_logits_sha256": before_fingerprint,
            "after_probe_logits_sha256": after_fingerprint,
            "logits_changed": before_fingerprint != after_fingerprint,
        },
        "raw": {
            "trajectory_rows": len(trajectory_rows),
            "trajectory_path": str(trajectories_path),
            "rewards": [row["reward"] for row in trajectory_rows],
        },
        "dependencies": dependency_versions(),
        "hardware": {
            "student_gpu": torch.cuda.get_device_name(student_device),
            "teacher_gpu": (
                torch.cuda.get_device_name(torch.device("cuda:1"))
                if config.mode == "opd"
                else None
            ),
            "student_peak_allocated_gib": (
                torch.cuda.max_memory_allocated(student_device) / 2**30
            ),
            "teacher_peak_allocated_gib": (
                torch.cuda.max_memory_allocated(torch.device("cuda:1"))
                / 2**30
                if config.mode == "opd"
                else None
            ),
        },
        "failure_receipt_exists": (output_root / "failure.json").exists(),
        "wall_seconds": time.time() - started,
    }
    (output_root / "metrics.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def validate_reload(config: AdmissionConfig) -> dict[str, Any]:
    model_identity_checks(config)
    output_root = Path(config.output_dir)
    adapter_dir = output_root / "adapter"
    metrics_path = output_root / "metrics.json"
    if (
        not adapter_dir.is_dir()
        or not metrics_path.is_file()
        or (output_root / "failure.json").exists()
    ):
        raise ValueError("admission artifacts are incomplete or failed")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if metrics["schema_version"] != RESULT_SCHEMA:
        raise ValueError("admission result schema differs")
    finite_tensors = 0
    nonfinite_tensors = 0
    weights = adapter_dir / "adapter_model.safetensors"
    with safe_open(weights, framework="pt", device="cpu") as handle:
        for key in handle.keys():
            tensor = handle.get_tensor(key)
            finite_tensors += 1
            nonfinite_tensors += int(not bool(torch.isfinite(tensor).all()))
    device = torch.device("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(
        adapter_dir,
        local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = Qwen3_5ForCausalLM.from_pretrained(
        config.student_model_path,
        local_files_only=True,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
    ).to(device)
    model = PeftModel.from_pretrained(
        model,
        adapter_dir,
        is_trainable=False,
    ).to(device)
    fingerprint = logits_fingerprint(model, tokenizer, config)
    expected = metrics["adapter_effect"]["after_probe_logits_sha256"]
    receipt = {
        "schema_version": RELOAD_SCHEMA,
        "experiment_id": config.experiment_id,
        "mode": config.mode,
        "adapter_sha256": sha256_tree(adapter_dir),
        "finite_adapter_tensors": finite_tensors,
        "nonfinite_adapter_tensors": nonfinite_tensors,
        "probe_logits_sha256": fingerprint,
        "expected_probe_logits_sha256": expected,
        "probe_logits_exact": fingerprint == expected,
        "reload_success": (
            finite_tensors > 0
            and nonfinite_tensors == 0
            and fingerprint == expected
            and sha256_tree(adapter_dir)
            == metrics["identity"]["adapter_sha256"]
        ),
        "peak_allocated_gib": (
            torch.cuda.max_memory_allocated(device) / 2**30
        ),
    }
    (output_root / "reload_validation.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return receipt
