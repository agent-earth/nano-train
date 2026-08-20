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
import torch.nn.functional as functional
from peft import LoraConfig, PeftModel, get_peft_model
from torch.optim import AdamW
from transformers import AutoTokenizer, Qwen3_5ForCausalLM

from nano_train.data import TokenizedSample, collate_samples
from nano_train.orca_math_sft import (
    _evaluation_rows,
    admission_gates,
    compare_rows,
    summarize_rows,
)
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


CONFIG_SCHEMA = "nano_train_orca_math_dpo_v1"
RELEASE_SCHEMA = "nano_orca_math_preference_release_v1"
SAMPLE_SCHEMA = "nano_orca_math_preference_sample_v1"


@dataclass(frozen=True)
class OrcaMathDPOConfig:
    schema_version: str
    experiment_id: str
    model_path: str
    model_config_sha256: str
    dataset_path: str
    dataset_file_sha256: str
    release_manifest_path: str
    release_manifest_sha256: str
    prior_sft_result_path: str
    prior_sft_result_sha256: str
    output_dir: str
    seed: int
    selection_seed: str
    dtype: str
    max_length: int
    max_steps: int
    learning_rate: float
    weight_decay: float
    warmup_steps: int
    beta: float
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    lora_targets: tuple[str, ...]
    gradient_checkpointing: bool
    generation_max_new_tokens: int
    generation_batch_size: int
    train_pairs_by_stratum: dict[str, int]
    dev_rows_by_stratum: dict[str, int]
    bootstrap_samples: int
    bootstrap_seed: int
    alpha: float
    minimum_candidate_only_wins: int


def load_config(path: str | Path) -> OrcaMathDPOConfig:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if set(raw) != set(OrcaMathDPOConfig.__dataclass_fields__):
        raise ValueError("Orca Math DPO config fields differ")
    raw["lora_targets"] = tuple(raw["lora_targets"])
    config = OrcaMathDPOConfig(**raw)
    validate_config(config)
    return config


def validate_config(config: OrcaMathDPOConfig) -> None:
    if config.schema_version != CONFIG_SCHEMA:
        raise ValueError("unsupported Orca Math DPO schema")
    expected: dict[str, Any] = {
        "experiment_id": "orca-math-verifier-dpo-v1",
        "model_path": "../../../models/Qwen3.5-4B",
        "dataset_path": (
            "../../../datasets/ultimate-distill/"
            "orca-math-preference-v1/dataset.jsonl"
        ),
        "release_manifest_path": (
            "../nano-data-pipeline-fullstack-traex-03/"
            "manifests/orca_math_preference_v1.release.json"
        ),
        "prior_sft_result_path": (
            "docs/results/orca_math_sft_smoke_v1.public.json"
        ),
        "output_dir": "artifacts/orca-math-verifier-dpo-v1",
        "seed": 20260822,
        "selection_seed": "orca-math-verifier-dpo-v1:20260822",
        "dtype": "float32",
        "max_length": 512,
        "max_steps": 32,
        "learning_rate": 0.000001,
        "weight_decay": 0.0,
        "warmup_steps": 1,
        "beta": 0.1,
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.0,
        "lora_targets": ("q_proj", "v_proj"),
        "gradient_checkpointing": True,
        "generation_max_new_tokens": 384,
        "generation_batch_size": 4,
        "train_pairs_by_stratum": {
            "short": 8,
            "medium": 16,
            "long": 8,
        },
        "dev_rows_by_stratum": {
            "short": 48,
            "medium": 96,
            "long": 48,
        },
        "bootstrap_samples": 10_000,
        "bootstrap_seed": 20260822,
        "alpha": 0.05,
        "minimum_candidate_only_wins": 6,
    }
    for field, value in expected.items():
        if getattr(config, field) != value:
            raise ValueError(f"Orca Math DPO freezes {field}={value}")
    for field in (
        "model_config_sha256",
        "dataset_file_sha256",
        "release_manifest_sha256",
        "prior_sft_result_sha256",
    ):
        value = getattr(config, field)
        if (
            len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)
        ):
            raise ValueError(f"Orca Math DPO {field} is not SHA256")
    if sum(config.train_pairs_by_stratum.values()) != config.max_steps:
        raise ValueError("Orca Math DPO train pair count differs")
    if sum(config.dev_rows_by_stratum.values()) != 192:
        raise ValueError("Orca Math DPO dev count differs")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _rank(seed: str, sample_id: str) -> str:
    return hashlib.sha256(f"{seed}\n{sample_id}".encode()).hexdigest()


def _sha256_lines(values: list[str]) -> str:
    return hashlib.sha256("\n".join(values).encode()).hexdigest()


def build_selection(config: OrcaMathDPOConfig) -> dict[str, Any]:
    dataset_path = Path(config.dataset_path)
    release_path = Path(config.release_manifest_path)
    prior_path = Path(config.prior_sft_result_path)
    if sha256_file(dataset_path) != config.dataset_file_sha256:
        raise ValueError("Orca Math DPO dataset identity differs")
    if sha256_file(release_path) != config.release_manifest_sha256:
        raise ValueError("Orca Math DPO release identity differs")
    if sha256_file(prior_path) != config.prior_sft_result_sha256:
        raise ValueError("Orca Math DPO prior SFT identity differs")
    if (
        sha256_file(Path(config.model_path) / "config.json")
        != config.model_config_sha256
    ):
        raise ValueError("Orca Math DPO model identity differs")
    release = json.loads(release_path.read_text(encoding="utf-8"))
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    if (
        release.get("schema_version") != RELEASE_SCHEMA
        or release.get("training_unblocked") is not True
        or release.get("benchmark_unlocked") is not False
        or not all(release.get("checks", {}).values())
        or prior.get("decision", {}).get("candidate_admitted") is not False
    ):
        raise ValueError("Orca Math DPO source boundary differs")
    rows = _read_jsonl(dataset_path)
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("schema_version") != SAMPLE_SCHEMA:
            raise ValueError("Orca Math DPO sample schema differs")
        buckets.setdefault((row["split"], row["stratum"]), []).append(row)
    train = []
    dev = []
    for stratum in ("short", "medium", "long"):
        ranked_train = sorted(
            buckets[("train", stratum)],
            key=lambda row: (
                _rank(config.selection_seed + ":train", row["sample_id"]),
                row["sample_id"],
            ),
        )
        ranked_dev = sorted(
            buckets[("dev", stratum)],
            key=lambda row: (
                _rank(config.selection_seed + ":dev", row["sample_id"]),
                row["sample_id"],
            ),
        )
        train.extend(
            ranked_train[: config.train_pairs_by_stratum[stratum]]
        )
        dev.extend(ranked_dev[: config.dev_rows_by_stratum[stratum]])
    train.sort(
        key=lambda row: (
            _rank(config.selection_seed + ":schedule", row["sample_id"]),
            row["sample_id"],
        )
    )
    dev.sort(key=lambda row: row["sample_id"])
    train_ids = [row["sample_id"] for row in train]
    dev_ids = [row["sample_id"] for row in dev]
    if (
        len(train) != config.max_steps
        or len(dev) != 192
        or set(train_ids) & set(dev_ids)
        or any(row["split"] != "train" for row in train)
        or any(row["split"] != "dev" for row in dev)
    ):
        raise ValueError("Orca Math DPO selection differs")
    return {
        "train": train,
        "dev": dev,
        "train_ids": train_ids,
        "dev_ids": dev_ids,
        "train_ids_sha256": _sha256_lines(train_ids),
        "dev_ids_sha256": _sha256_lines(dev_ids),
    }


def sequence_log_probability(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    if logits.ndim != 3 or labels.ndim != 2 or logits.shape[:2] != labels.shape:
        raise ValueError("DPO logits/labels shape differs")
    shifted_logits = logits[:, :-1, :]
    shifted_labels = labels[:, 1:]
    mask = shifted_labels != -100
    safe_labels = shifted_labels.masked_fill(~mask, 0)
    token_logps = functional.log_softmax(shifted_logits, dim=-1).gather(
        -1, safe_labels.unsqueeze(-1)
    ).squeeze(-1)
    counts = mask.sum(dim=-1)
    if bool((counts == 0).any()):
        raise ValueError("DPO sequence has no target tokens")
    return (token_logps * mask).sum(dim=-1) / counts


def dpo_loss(
    policy_chosen: torch.Tensor,
    policy_rejected: torch.Tensor,
    reference_chosen: torch.Tensor,
    reference_rejected: torch.Tensor,
    *,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if beta <= 0:
        raise ValueError("DPO beta must be positive")
    policy_margin = policy_chosen - policy_rejected
    reference_margin = reference_chosen - reference_rejected
    advantage = policy_margin - reference_margin.detach()
    return -functional.logsigmoid(beta * advantage).mean(), advantage.mean()


def _tokenize_target(
    tokenizer: Any,
    row: dict[str, Any],
    target: str,
    *,
    max_length: int,
    suffix: str,
) -> TokenizedSample:
    prompt = tokenizer.apply_chat_template(
        row["prompt_messages"],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    prompt_ids = tokenizer(
        prompt,
        add_special_tokens=False,
    ).input_ids
    input_ids = tokenizer(
        prompt + target + tokenizer.eos_token,
        add_special_tokens=False,
    ).input_ids
    if len(input_ids) > max_length:
        raise ValueError("Orca Math DPO sample exceeds max_length")
    labels = [-100] * len(prompt_ids) + input_ids[len(prompt_ids) :]
    return TokenizedSample(
        sample_id=f"{row['sample_id']}:{suffix}",
        split="train",
        input_ids=input_ids,
        labels=labels,
        prompt_ids=prompt_ids,
        target=target,
        format_family="orca_math_preference",
        verifier=row.get("verifier"),
        task_family=row["stratum"],
    )


def _dev_rows(selection: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "sample_id": row["sample_id"],
            "stratum": row["stratum"],
            "numeric_answer": row["expected"],
            "messages": row["prompt_messages"]
            + [{"role": "assistant", "content": row["chosen"]}],
        }
        for row in selection["dev"]
    ]


def run(config: OrcaMathDPOConfig) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("Orca Math DPO requires CUDA")
    selection = build_selection(config)
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_path,
        local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    train = [
        {
            "row": row,
            "chosen": _tokenize_target(
                tokenizer,
                row,
                row["chosen"],
                max_length=config.max_length,
                suffix="chosen",
            ),
            "rejected": _tokenize_target(
                tokenizer,
                row,
                row["rejected"],
                max_length=config.max_length,
                suffix="rejected",
            ),
        }
        for row in selection["train"]
    ]
    dev = _dev_rows(selection)
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
        dev,
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
    optimizer.zero_grad(set_to_none=True)
    model.train()
    for step_index, item in enumerate(train):
        step = step_index + 1
        chosen_batch = {
            key: value.to(device)
            for key, value in collate_samples(
                [item["chosen"]],
                pad_token_id=tokenizer.pad_token_id,
            ).items()
        }
        rejected_batch = {
            key: value.to(device)
            for key, value in collate_samples(
                [item["rejected"]],
                pad_token_id=tokenizer.pad_token_id,
            ).items()
        }
        with model.disable_adapter(), torch.inference_mode():
            reference_chosen = sequence_log_probability(
                model(**chosen_batch, use_cache=False).logits,
                chosen_batch["labels"],
            )
            reference_rejected = sequence_log_probability(
                model(**rejected_batch, use_cache=False).logits,
                rejected_batch["labels"],
            )
        policy_chosen = sequence_log_probability(
            model(**chosen_batch, use_cache=False).logits,
            chosen_batch["labels"],
        )
        policy_rejected = sequence_log_probability(
            model(**rejected_batch, use_cache=False).logits,
            rejected_batch["labels"],
        )
        loss, advantage = dpo_loss(
            policy_chosen,
            policy_rejected,
            reference_chosen,
            reference_rejected,
            beta=config.beta,
        )
        scale = _scheduler_scale(
            step_index,
            config.warmup_steps,
            config.max_steps,
        )
        for group in optimizer.param_groups:
            group["lr"] = config.learning_rate * scale
        try:
            _assert_finite_loss(loss, step=step)
            loss.backward()
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
                stage="orca_math_dpo",
                error=error,
            )
            raise
        optimizer.zero_grad(set_to_none=True)
        loss_curve.append(
            {
                "step": step,
                "sample_id": item["row"]["sample_id"],
                "stratum": item["row"]["stratum"],
                "loss": float(loss.detach().cpu()),
                "advantage": float(advantage.detach().cpu()),
                "policy_margin": float(
                    (policy_chosen - policy_rejected).detach().cpu()
                ),
                "reference_margin": float(
                    (reference_chosen - reference_rejected).detach().cpu()
                ),
                "gradient_norm": float(gradient_norm.detach().cpu()),
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir, safe_serialization=True)
    tokenizer.save_pretrained(adapter_dir)
    post_rows = _evaluation_rows(
        model,
        tokenizer,
        dev,
        device=device,
        max_new_tokens=config.generation_max_new_tokens,
        batch_size=config.generation_batch_size,
    )
    post_summary = summarize_rows(post_rows)
    generations_path = output_root / "generations.json"
    generations_path.write_text(
        json.dumps(
            {"baseline": baseline_rows, "post_dpo": post_rows},
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
        "schema_version": "nano_train_orca_math_dpo_result_v1",
        "experiment_id": config.experiment_id,
        "config": {
            **config.__dict__,
            "lora_targets": list(config.lora_targets),
        },
        "selection": {
            "train_ids_sha256": selection["train_ids_sha256"],
            "dev_ids_sha256": selection["dev_ids_sha256"],
            "train_pairs": len(train),
            "dev_rows": len(dev),
        },
        "adapter_sha256": sha256_tree(adapter_dir),
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "baseline_validation": baseline_summary,
        "post_validation": post_summary,
        "comparison": comparison,
        "gates": gates,
        "candidate_admitted": all(gates.values()),
        "loss_curve": loss_curve,
        "all_losses_finite": all(
            math.isfinite(row["loss"]) for row in loss_curve
        ),
        "all_gradient_norms_finite": all(
            math.isfinite(row["gradient_norm"]) for row in loss_curve
        ),
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


def validate_reload(config: OrcaMathDPOConfig) -> dict[str, Any]:
    selection = build_selection(config)
    output_root = Path(config.output_dir)
    adapter_dir = output_root / "adapter"
    metrics = json.loads(
        (output_root / "metrics.json").read_text(encoding="utf-8")
    )
    generations = json.loads(
        (output_root / "generations.json").read_text(encoding="utf-8")
    )
    if (
        not adapter_dir.is_dir()
        or (output_root / "failure.json").exists()
    ):
        raise ValueError("Orca Math DPO reload artifacts are incomplete")
    tokenizer = AutoTokenizer.from_pretrained(
        adapter_dir, local_files_only=True
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
        _dev_rows(selection),
        device=device,
        max_new_tokens=config.generation_max_new_tokens,
        batch_size=config.generation_batch_size,
    )
    summary = summarize_rows(rows)
    if (
        rows != generations["post_dpo"]
        or summary != metrics["post_validation"]
        or sha256_tree(adapter_dir) != metrics["adapter_sha256"]
    ):
        raise ValueError("Orca Math DPO reload differs")
    reload_path = output_root / "reload_generations.json"
    reload_path.write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    receipt = {
        "schema_version": "nano_train_orca_math_dpo_reload_v1",
        "experiment_id": config.experiment_id,
        "adapter_sha256": metrics["adapter_sha256"],
        "reload_success": True,
        "metrics_exact": True,
        "generations_exact": True,
        "post_validation": summary,
        "reload_generations_sha256": sha256_file(reload_path),
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
    }
    (output_root / "reload_validation.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return receipt
