from __future__ import annotations

import hashlib
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, PeftModel, get_peft_model
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
from nano_train.synthetic_quality import (
    FAMILIES,
    build_cases as build_forbidden_evaluation_cases,
    case_contract as forbidden_evaluation_case_contract,
    load_config as load_forbidden_evaluation_config,
    paired_comparison,
)


CONFIG_SCHEMA = "nano_train_scaled_quality_sft_v1"
RESULT_SCHEMA = "nano_train_scaled_quality_sft_result_v1"


@dataclass(frozen=True)
class ScaledQualityConfig:
    schema_version: str
    experiment_id: str
    model_path: str
    model_config_sha256: str
    model_index_sha256: str
    weight_shards: tuple[dict[str, Any], ...]
    output_dir: str
    seed: int
    dtype: str
    train_cases_per_family: int
    dev_cases_per_family: int
    train_range_offset: int
    dev_range_offset: int
    batch_size: int
    max_steps: int
    max_length: int
    learning_rate: float
    weight_decay: float
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    lora_targets: tuple[str, ...]
    gradient_checkpointing: bool
    generation_batch_size: int
    generation_max_new_tokens: int
    bootstrap_samples: int
    bootstrap_seed: int
    system_prompt: str
    prompt_template: str
    target_template: str
    forbidden_evaluation_config_path: str
    forbidden_evaluation_config_sha256: str
    forbidden_evaluation_contract_sha256: str
    policy: dict[str, bool]


def load_config(path: str | Path) -> ScaledQualityConfig:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if set(raw) != set(ScaledQualityConfig.__dataclass_fields__):
        raise ValueError("scaled quality config fields differ")
    raw["weight_shards"] = tuple(raw["weight_shards"])
    raw["lora_targets"] = tuple(raw["lora_targets"])
    config = ScaledQualityConfig(**raw)
    validate_config(config)
    return config


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def validate_config(config: ScaledQualityConfig) -> None:
    if config.schema_version != CONFIG_SCHEMA:
        raise ValueError("unsupported scaled quality schema")
    expected = {
        "seed": 20260820,
        "dtype": "float32",
        "train_cases_per_family": 128,
        "dev_cases_per_family": 24,
        "train_range_offset": 100,
        "dev_range_offset": 5000,
        "batch_size": 4,
        "max_steps": 128,
        "max_length": 128,
        "learning_rate": 0.00005,
        "weight_decay": 0.0,
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.0,
        "lora_targets": ("q_proj", "v_proj"),
        "gradient_checkpointing": True,
        "generation_batch_size": 8,
        "generation_max_new_tokens": 32,
        "bootstrap_samples": 10_000,
        "bootstrap_seed": 20260820,
        "system_prompt": (
            "Follow the exact output contract. Solve internally without tools "
            "or external information."
        ),
        "prompt_template": (
            "Compute the synthetic arithmetic expression exactly: "
            "{expression}. Return only one line in the form FINAL: <integer>."
        ),
        "target_template": "FINAL: {expected}",
    }
    for field, expected_value in expected.items():
        if getattr(config, field) != expected_value:
            raise ValueError(
                f"scaled quality freezes {field}={expected_value}"
            )
    if (
        not _is_sha256(config.model_config_sha256)
        or not _is_sha256(config.model_index_sha256)
        or not _is_sha256(config.forbidden_evaluation_config_sha256)
        or not _is_sha256(config.forbidden_evaluation_contract_sha256)
        or len(config.weight_shards) != 2
    ):
        raise ValueError("scaled quality identity differs")
    for shard in config.weight_shards:
        if (
            set(shard) != {"name", "bytes", "sha256"}
            or int(shard["bytes"]) <= 0
            or not _is_sha256(shard["sha256"])
        ):
            raise ValueError("scaled quality shard identity differs")
    if (
        config.train_cases_per_family * len(FAMILIES)
        != config.max_steps * config.batch_size
    ):
        raise ValueError("scaled quality exposure is not exactly one epoch")
    if config.train_range_offset + config.train_cases_per_family >= 1201:
        raise ValueError("scaled quality train range approaches observed eval")
    if config.dev_range_offset <= 3000:
        raise ValueError("scaled quality dev range is not isolated")
    required_policy = {
        "contains_benchmark_rows": False,
        "contains_benchmark_outputs": False,
        "contains_canary_rows": False,
        "contains_holdout_rows": False,
        "contains_observed_quality_rows": False,
        "uses_observed_quality_outputs": False,
        "training_allowed": True,
        "benchmark_access_after_result": False,
    }
    if config.policy != required_policy:
        raise ValueError("scaled quality policy differs")


def _case(
    config: ScaledQualityConfig,
    *,
    split: str,
    family: str,
    index: int,
    expression: str,
) -> dict[str, str]:
    expected = format_number(evaluate_arithmetic(expression))
    prompt = config.prompt_template.format(expression=expression)
    target = config.target_template.format(expected=expected)
    digest = hashlib.sha256(
        f"{split}\0{family}\0{expression}".encode()
    ).hexdigest()
    return {
        "case_id": f"scaled-quality-{split}-{family}-{digest[:16]}",
        "split": split,
        "family": family,
        "expression": expression,
        "expected": expected,
        "prompt": prompt,
        "target": target,
    }


def build_dataset(config: ScaledQualityConfig) -> dict[str, Any]:
    train = _build_split(
        config,
        split="train",
        count=config.train_cases_per_family,
        offset=config.train_range_offset,
    )
    dev = _build_split(
        config,
        split="dev",
        count=config.dev_cases_per_family,
        offset=config.dev_range_offset,
    )
    train_expressions = {row["expression"] for row in train}
    dev_expressions = {row["expression"] for row in dev}
    if train_expressions & dev_expressions:
        raise ValueError("scaled quality train/dev expressions overlap")
    all_rows = train + dev
    if len({row["case_id"] for row in all_rows}) != len(all_rows):
        raise ValueError("scaled quality case IDs are not unique")
    return {
        "schema_version": "nano_train_scaled_quality_dataset_v1",
        "dataset_id": "qwen35-scaled-quality-v1",
        "train": train,
        "dev": dev,
        "identity": {
            "train_case_ids_sha256": _sha256_lines(
                sorted(row["case_id"] for row in train)
            ),
            "dev_case_ids_sha256": _sha256_lines(
                sorted(row["case_id"] for row in dev)
            ),
            "train_prompt_sha256": _sha256_lines(
                sorted(
                    hashlib.sha256(row["prompt"].encode()).hexdigest()
                    for row in train
                )
            ),
            "dev_prompt_sha256": _sha256_lines(
                sorted(
                    hashlib.sha256(row["prompt"].encode()).hexdigest()
                    for row in dev
                )
            ),
        },
    }


def _build_split(
    config: ScaledQualityConfig,
    *,
    split: str,
    count: int,
    offset: int,
) -> list[dict[str, str]]:
    rows = []
    for index in range(count):
        value = offset + index
        left = value * 7 + 101
        repeated = value * 3 + 29
        multiplier = 2 + index % 3
        rows.append(
            _case(
                config,
                split=split,
                family="repeated_operand",
                index=index,
                expression=(
                    f"({left} + {repeated}) * {multiplier} - {repeated}"
                ),
            )
        )

        first = value * 5 + 37
        second = value * 4 + 53
        rows.append(
            _case(
                config,
                split=split,
                family="mixed_products",
                index=index,
                expression=(
                    f"{first} * {3 + index % 5} + "
                    f"{second} * {2 + index % 4} - {value + 19}"
                ),
            )
        )

        divisor = 3 + index % 5
        quotient = value * 3 + 401
        addend = value + 17
        numerator = quotient * divisor - addend
        rows.append(
            _case(
                config,
                split=split,
                family="exact_division",
                index=index,
                expression=f"({numerator} + {addend}) / {divisor}",
            )
        )

        base = value * 6 + 701
        offset_value = value * 2 + 83
        rows.append(
            _case(
                config,
                split=split,
                family="nested_offset",
                index=index,
                expression=(
                    f"({base} - {offset_value}) * {3 + index % 3} "
                    f"+ {value + 41}"
                ),
            )
        )
    random.Random(config.seed + (0 if split == "train" else 1)).shuffle(rows)
    return rows


def _sha256_lines(values: list[str]) -> str:
    return hashlib.sha256("\n".join(values).encode()).hexdigest()


def public_dataset_contract(dataset: dict[str, Any]) -> dict[str, Any]:
    def rows(split: str) -> list[dict[str, Any]]:
        return [
            {
                "case_id": row["case_id"],
                "family": row["family"],
                "prompt_sha256": hashlib.sha256(
                    row["prompt"].encode()
                ).hexdigest(),
                "target_sha256": hashlib.sha256(
                    row["target"].encode()
                ).hexdigest(),
            }
            for row in dataset[split]
        ]

    return {
        "schema_version": "nano_train_scaled_quality_contract_v1",
        "dataset_id": dataset["dataset_id"],
        "train": rows("train"),
        "dev": rows("dev"),
        "identity": dataset["identity"],
    }


def _verify_model(config: ScaledQualityConfig) -> None:
    path = Path(config.model_path)
    if (
        sha256_file(path / "config.json") != config.model_config_sha256
        or sha256_file(path / "model.safetensors.index.json")
        != config.model_index_sha256
    ):
        raise ValueError("scaled quality model metadata mismatch")
    for shard in config.weight_shards:
        shard_path = path / shard["name"]
        if (
            shard_path.stat().st_size != shard["bytes"]
            or sha256_file(shard_path) != shard["sha256"]
        ):
            raise ValueError("scaled quality model shard mismatch")


def _tokenize_rows(
    rows: list[dict[str, str]],
    tokenizer: Any,
    config: ScaledQualityConfig,
) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        prompt = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": config.system_prompt},
                {"role": "user", "content": row["prompt"]},
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        full = prompt + row["target"] + tokenizer.eos_token
        prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
        input_ids = tokenizer(full, add_special_tokens=False).input_ids
        if len(input_ids) > config.max_length:
            raise ValueError("scaled quality row exceeds max_length")
        result.append(
            {
                **row,
                "prompt_ids": prompt_ids,
                "input_ids": input_ids,
                "labels": [-100] * len(prompt_ids)
                + input_ids[len(prompt_ids) :],
            }
        )
    return result


def _collate(
    rows: list[dict[str, Any]],
    *,
    pad_token_id: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    maximum = max(len(row["input_ids"]) for row in rows)
    return {
        "input_ids": torch.tensor(
            [
                row["input_ids"]
                + [pad_token_id] * (maximum - len(row["input_ids"]))
                for row in rows
            ],
            dtype=torch.long,
            device=device,
        ),
        "labels": torch.tensor(
            [
                row["labels"]
                + [-100] * (maximum - len(row["labels"]))
                for row in rows
            ],
            dtype=torch.long,
            device=device,
        ),
        "attention_mask": torch.tensor(
            [
                [1] * len(row["input_ids"])
                + [0] * (maximum - len(row["input_ids"]))
                for row in rows
            ],
            dtype=torch.long,
            device=device,
        ),
    }


@torch.inference_mode()
def evaluate_rows(
    model: Any,
    tokenizer: Any,
    config: ScaledQualityConfig,
    rows: list[dict[str, str]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    device = next(model.parameters()).device
    tokenizer.padding_side = "left"
    model.eval()
    results = []
    for offset in range(0, len(rows), config.generation_batch_size):
        selected = rows[offset : offset + config.generation_batch_size]
        texts = [
            tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": config.system_prompt},
                    {"role": "user", "content": row["prompt"]},
                ],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for row in selected
        ]
        batch = tokenizer(
            texts,
            add_special_tokens=False,
            padding=True,
            return_tensors="pt",
        ).to(device)
        generated = model.generate(
            **batch,
            do_sample=False,
            max_new_tokens=config.generation_max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
        )
        prompt_length = batch["input_ids"].shape[1]
        for row, sequence in zip(selected, generated):
            output = tokenizer.decode(
                sequence[prompt_length:],
                skip_special_tokens=True,
            ).strip()
            results.append(
                {
                    "case_id": row["case_id"],
                    "family": row["family"],
                    "expected": row["expected"],
                    "output": output,
                    "correct": output == row["target"],
                    "parse_failure": not output.startswith("FINAL: "),
                    "output_sha256": hashlib.sha256(
                        output.encode()
                    ).hexdigest(),
                }
            )
    by_family = {}
    for family in FAMILIES:
        subset = [row for row in results if row["family"] == family]
        by_family[family] = {
            "cases": len(subset),
            "correct": sum(row["correct"] for row in subset),
            "parse_failures": sum(
                row["parse_failure"] for row in subset
            ),
        }
    correct = sum(row["correct"] for row in results)
    return (
        {
            "cases": len(results),
            "correct": correct,
            "accuracy": correct / len(results),
            "parse_failures": sum(
                row["parse_failure"] for row in results
            ),
            "by_family": by_family,
        },
        results,
    )


def run(config: ScaledQualityConfig) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("scaled quality SFT requires CUDA")
    _verify_model(config)
    dataset = build_dataset(config)
    public_contract = public_dataset_contract(dataset)
    forbidden_config_path = Path(config.forbidden_evaluation_config_path)
    if (
        sha256_file(forbidden_config_path)
        != config.forbidden_evaluation_config_sha256
    ):
        raise ValueError("forbidden evaluation config identity mismatch")
    forbidden_config = load_forbidden_evaluation_config(
        forbidden_config_path
    )
    forbidden_cases = build_forbidden_evaluation_cases(forbidden_config)
    forbidden_contract = forbidden_evaluation_case_contract(forbidden_cases)
    if (
        forbidden_contract["case_contract_sha256"]
        != config.forbidden_evaluation_contract_sha256
    ):
        raise ValueError("forbidden evaluation contract identity mismatch")
    forbidden_hashes = {
        hashlib.sha256(row["prompt"].encode()).hexdigest()
        for row in forbidden_cases
    }
    all_hashes = {
        hashlib.sha256(row["prompt"].encode()).hexdigest()
        for row in (*dataset["train"], *dataset["dev"])
    }
    if all_hashes & forbidden_hashes:
        raise ValueError("scaled quality data overlaps forbidden evaluation")
    set_seed(config.seed)
    output_root = Path(config.output_dir)
    adapter_dir = output_root / "adapter"
    output_root.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_path,
        local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenized_train = _tokenize_rows(dataset["train"], tokenizer, config)
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
    baseline_metrics, baseline_rows = evaluate_rows(
        model,
        tokenizer,
        config,
        dataset["dev"],
    )
    model.train()
    optimizer = AdamW(
        _trainable_parameters(model),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        eps=1e-6,
    )
    trainable = _trainable_parameters(model)
    loss_curve = []
    exposure = []
    for step in range(config.max_steps):
        selected = tokenized_train[
            step * config.batch_size : (step + 1) * config.batch_size
        ]
        if len(selected) != config.batch_size:
            raise ValueError("scaled quality step exposure differs")
        batch = _collate(
            selected,
            pad_token_id=tokenizer.pad_token_id,
            device=device,
        )
        outputs = model(**batch, use_cache=False)
        try:
            _assert_finite_loss(outputs.loss, step=step + 1)
            outputs.loss.backward()
            _assert_finite_gradients(trainable, step=step + 1)
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                trainable,
                max_norm=1.0,
                error_if_nonfinite=True,
            )
            optimizer.step()
            _assert_finite_parameters(trainable, step=step + 1)
        except (FloatingPointError, RuntimeError) as error:
            _write_failure(
                output_root,
                step=step + 1,
                stage="scaled_quality_sft",
                error=error,
            )
            raise
        optimizer.zero_grad(set_to_none=True)
        loss_curve.append(
            {
                "step": step + 1,
                "loss": float(outputs.loss.detach().cpu()),
                "gradient_norm": float(gradient_norm.detach().cpu()),
            }
        )
        exposure.extend(row["case_id"] for row in selected)
    if (
        len(exposure) != len(dataset["train"])
        or len(set(exposure)) != len(dataset["train"])
    ):
        raise ValueError("scaled quality exposure is not exactly once")
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir, safe_serialization=True)
    tokenizer.save_pretrained(adapter_dir)
    post_metrics, post_rows = evaluate_rows(
        model,
        tokenizer,
        config,
        dataset["dev"],
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
    comparison = paired_comparison(
        post_rows,
        baseline_rows,
        bootstrap_samples=config.bootstrap_samples,
        bootstrap_seed=config.bootstrap_seed,
    )
    result = {
        "schema_version": RESULT_SCHEMA,
        "experiment_id": config.experiment_id,
        "dataset_contract": public_contract,
        "identity": {
            "model_config_sha256": config.model_config_sha256,
            "model_index_sha256": config.model_index_sha256,
            "adapter_sha256": sha256_tree(adapter_dir),
            "generations_sha256": sha256_file(generations_path),
        },
        "training": {
            "optimizer_steps": config.max_steps,
            "train_rows": len(dataset["train"]),
            "unique_exposures": len(set(exposure)),
            "trainable_parameters": sum(
                parameter.numel() for parameter in trainable
            ),
            "all_losses_finite": all(
                math.isfinite(row["loss"]) for row in loss_curve
            ),
            "all_gradient_norms_finite": all(
                math.isfinite(row["gradient_norm"])
                for row in loss_curve
            ),
            "loss_curve": loss_curve,
        },
        "baseline_dev": baseline_metrics,
        "post_sft_dev": post_metrics,
        "comparison": comparison,
        "dependencies": dependency_versions(),
        "hardware": {
            "gpu_name": torch.cuda.get_device_name(device),
            "peak_allocated_gib": (
                torch.cuda.max_memory_allocated(device) / 2**30
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


def validate_reload(config: ScaledQualityConfig) -> dict[str, Any]:
    output_root = Path(config.output_dir)
    adapter_dir = output_root / "adapter"
    metrics = json.loads(
        (output_root / "metrics.json").read_text(encoding="utf-8")
    )
    dataset = build_dataset(config)
    tokenizer = AutoTokenizer.from_pretrained(
        adapter_dir,
        local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = Qwen3_5ForCausalLM.from_pretrained(
        config.model_path,
        local_files_only=True,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
    ).cuda()
    model = PeftModel.from_pretrained(
        model,
        adapter_dir,
        is_trainable=False,
    ).cuda()
    post_metrics, post_rows = evaluate_rows(
        model,
        tokenizer,
        config,
        dataset["dev"],
    )
    generations = json.loads(
        (output_root / "generations.json").read_text(encoding="utf-8")
    )
    receipt = {
        "schema_version": "nano_train_scaled_quality_reload_v1",
        "experiment_id": config.experiment_id,
        "adapter_sha256": sha256_tree(adapter_dir),
        "metrics_exact": post_metrics == metrics["post_sft_dev"],
        "generations_exact": post_rows == generations["post_sft"],
        "reload_success": (
            post_metrics == metrics["post_sft_dev"]
            and post_rows == generations["post_sft"]
            and sha256_tree(adapter_dir)
            == metrics["identity"]["adapter_sha256"]
        ),
        "peak_allocated_gib": (
            torch.cuda.max_memory_allocated() / 2**30
        ),
    }
    (output_root / "reload_validation.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return receipt
