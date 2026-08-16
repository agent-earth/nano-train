from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import random
import time
from pathlib import Path
from typing import Any

import accelerate
import peft
import torch
import transformers
from peft import LoraConfig, get_peft_model
from torch.optim import AdamW
from transformers import AutoTokenizer, Qwen3_5ForCausalLM

from nano_train.config import SFTSmokeConfig
from nano_train.data import (
    TokenizedSample,
    collate_samples,
    load_analog_dataset,
    tokenize_samples,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(path.rglob("*")):
        if not item.is_file():
            continue
        relative = item.relative_to(path).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(item).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def dependency_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "accelerate": accelerate.__version__,
    }


def _batch_order(samples: list[TokenizedSample], seed: int) -> list[int]:
    order = list(range(len(samples)))
    random.Random(seed).shuffle(order)
    return order


@torch.inference_mode()
def evaluate_exact(
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
        attention_mask = torch.ones_like(prompt)
        output = model.generate(
            input_ids=prompt,
            attention_mask=attention_mask,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
        )
        generated = tokenizer.decode(
            output[0, prompt.shape[1] :],
            skip_special_tokens=True,
        ).strip()
        rows.append(
            {
                "sample_id": sample.sample_id,
                "target": sample.target,
                "output": generated,
                "exact": generated == sample.target,
            }
        )
    exact = sum(row["exact"] for row in rows)
    return (
        {
            "samples": len(rows),
            "exact": exact,
            "accuracy": exact / len(rows),
            "failure_sample_ids": [
                row["sample_id"] for row in rows if not row["exact"]
            ],
        },
        rows,
    )


def _scheduler_scale(step: int, warmup_steps: int, max_steps: int) -> float:
    if warmup_steps and step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return max(0.0, 1.0 - progress)


def run_sft_smoke(config: SFTSmokeConfig) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("SFT smoke requires one CUDA GPU")
    set_seed(config.seed)
    output_root = Path(config.output_dir)
    adapter_dir = output_root / "adapter"
    output_root.mkdir(parents=True, exist_ok=True)

    dataset_path = Path(config.dataset_path)
    dataset = load_analog_dataset(dataset_path)
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_path,
        local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    samples = tokenize_samples(
        dataset,
        tokenizer,
        max_length=config.max_length,
    )
    train = [sample for sample in samples if sample.split == "train"]
    validation = [sample for sample in samples if sample.split == "validation"]
    if not train or not validation:
        raise ValueError("dataset must contain train and validation samples")

    device = torch.device("cuda")
    started = time.time()
    model = Qwen3_5ForCausalLM.from_pretrained(
        config.model_path,
        local_files_only=True,
        dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to(device)
    model.config.use_cache = False
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=list(config.lora_targets),
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    total_parameters = sum(parameter.numel() for parameter in model.parameters())

    baseline_metrics, baseline_rows = evaluate_exact(
        model,
        tokenizer,
        validation,
        device=device,
        max_new_tokens=config.generation_max_new_tokens,
    )

    optimizer = AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    order = _batch_order(train, config.seed)
    losses = []
    optimizer.zero_grad(set_to_none=True)
    model.train()
    for step in range(config.max_steps):
        batch_indices = [
            order[
                (
                    step * config.batch_size * config.gradient_accumulation_steps
                    + micro_step * config.batch_size
                    + offset
                )
                % len(order)
            ]
            for micro_step in range(config.gradient_accumulation_steps)
            for offset in range(config.batch_size)
        ]
        step_losses = []
        for micro_step in range(config.gradient_accumulation_steps):
            start = micro_step * config.batch_size
            selected = [
                train[index]
                for index in batch_indices[start : start + config.batch_size]
            ]
            batch = {
                key: value.to(device)
                for key, value in collate_samples(
                    selected,
                    pad_token_id=tokenizer.pad_token_id,
                ).items()
            }
            outputs = model(**batch, use_cache=False)
            loss = outputs.loss / config.gradient_accumulation_steps
            loss.backward()
            step_losses.append(float(outputs.loss.detach().cpu()))
        lr_scale = _scheduler_scale(
            step,
            config.warmup_steps,
            config.max_steps,
        )
        for group in optimizer.param_groups:
            group["lr"] = config.learning_rate * lr_scale
        torch.nn.utils.clip_grad_norm_(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            max_norm=1.0,
        )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        losses.append(
            {
                "step": step + 1,
                "loss": sum(step_losses) / len(step_losses),
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

    generations = {
        "baseline": baseline_rows,
        "post_sft": post_rows,
    }
    generations_path = output_root / "generations.json"
    generations_path.write_text(
        json.dumps(generations, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    metrics = {
        "schema_version": "nano_train_sft_smoke_result_v1",
        "experiment_id": config.experiment_id,
        "dataset": {
            "path": str(dataset_path),
            "sha256": sha256_file(dataset_path),
            "dataset_id": dataset["dataset_id"],
            "train_samples": len(train),
            "validation_samples": len(validation),
        },
        "model": {
            "path": config.model_path,
            "config_sha256": sha256_file(Path(config.model_path) / "config.json"),
            "dtype": config.dtype,
            "total_parameters": total_parameters,
            "trainable_parameters": trainable_parameters,
        },
        "config": {
            **config.__dict__,
            "lora_targets": list(config.lora_targets),
        },
        "dependencies": dependency_versions(),
        "hardware": {
            "gpu_name": torch.cuda.get_device_name(device),
            "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        },
        "baseline_validation": baseline_metrics,
        "post_sft_validation": post_metrics,
        "loss_curve": losses,
        "adapter_sha256": sha256_tree(adapter_dir),
        "generations_sha256": sha256_file(generations_path),
        "wall_seconds": time.time() - started,
    }
    metrics_path = output_root / "metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metrics
