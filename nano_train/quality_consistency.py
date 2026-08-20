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
from peft import PeftModel
from torch.optim import AdamW
from transformers import AutoTokenizer, Qwen3_5ForCausalLM

from nano_train.data import evaluate_arithmetic, format_number
from nano_train.paired_consistency import (
    align_teacher_logits,
    paired_consistency_kl,
    supervised_target_labels,
    target_prediction_logits,
)
from nano_train.scaled_quality import evaluate_rows
from nano_train.scaled_quality import build_dataset as build_scaled_quality_dataset
from nano_train.scaled_quality import load_config as load_scaled_quality_config
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
    build_cases as build_synthetic_quality_cases,
    load_config as load_synthetic_quality_config,
    paired_comparison,
)


CONFIG_SCHEMA = "nano_train_quality_consistency_v1"
RESULT_SCHEMA = "nano_train_quality_consistency_result_v1"


@dataclass(frozen=True)
class QualityConsistencyConfig:
    schema_version: str
    experiment_id: str
    model_path: str
    model_config_sha256: str
    model_index_sha256: str
    weight_shards: tuple[dict[str, Any], ...]
    anchor_adapter_path: str
    anchor_adapter_sha256: str
    output_dir: str
    seed: int
    dtype: str
    train_pairs_per_family: int
    dev_cases_per_family: int
    train_range_offset: int
    dev_range_offset: int
    max_steps: int
    max_length: int
    learning_rate: float
    weight_decay: float
    process_ce_weight: float
    final_ce_weight: float
    consistency_weight: float
    consistency_temperature: float
    teacher_detach: bool
    generation_batch_size: int
    generation_max_new_tokens: int
    bootstrap_samples: int
    bootstrap_seed: int
    system_prompt: str
    process_prompt_template: str
    final_prompt_template: str
    process_target_template: str
    final_target_template: str
    forbidden_config_paths: tuple[dict[str, str], ...]
    benchmark_sources: tuple[dict[str, str], ...]
    policy: dict[str, bool]


def load_config(path: str | Path) -> QualityConsistencyConfig:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if set(raw) != set(QualityConsistencyConfig.__dataclass_fields__):
        raise ValueError("quality consistency config fields differ")
    raw["weight_shards"] = tuple(raw["weight_shards"])
    raw["forbidden_config_paths"] = tuple(raw["forbidden_config_paths"])
    raw["benchmark_sources"] = tuple(raw["benchmark_sources"])
    config = QualityConsistencyConfig(**raw)
    validate_config(config)
    return config


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def validate_config(config: QualityConsistencyConfig) -> None:
    if config.schema_version != CONFIG_SCHEMA:
        raise ValueError("unsupported quality consistency schema")
    expected = {
        "seed": 20260820,
        "dtype": "float32",
        "train_pairs_per_family": 64,
        "dev_cases_per_family": 48,
        "train_range_offset": 1000,
        "dev_range_offset": 10000,
        "max_steps": 256,
        "max_length": 160,
        "learning_rate": 0.00005,
        "weight_decay": 0.0,
        "process_ce_weight": 0.5,
        "final_ce_weight": 0.5,
        "consistency_weight": 1.0,
        "consistency_temperature": 1.0,
        "teacher_detach": True,
        "generation_batch_size": 8,
        "generation_max_new_tokens": 32,
        "bootstrap_samples": 10_000,
        "bootstrap_seed": 20260820,
        "system_prompt": (
            "Follow the exact output contract. Solve internally without tools "
            "or external information."
        ),
        "process_prompt_template": (
            "Compute the synthetic arithmetic expression exactly: "
            "{expression}. Show one verified WORK line, then FINAL: <integer>."
        ),
        "final_prompt_template": (
            "Compute the synthetic arithmetic expression exactly: "
            "{expression}. Return only FINAL: <integer>."
        ),
        "process_target_template": (
            "WORK: {expression} = {expected}\nFINAL: {expected}"
        ),
        "final_target_template": "FINAL: {expected}",
    }
    for field, expected_value in expected.items():
        if getattr(config, field) != expected_value:
            raise ValueError(
                f"quality consistency freezes {field}={expected_value}"
            )
    if (
        not _is_sha256(config.model_config_sha256)
        or not _is_sha256(config.model_index_sha256)
        or not _is_sha256(config.anchor_adapter_sha256)
        or len(config.weight_shards) != 2
    ):
        raise ValueError("quality consistency identity differs")
    if config.max_steps != config.train_pairs_per_family * len(FAMILIES):
        raise ValueError("quality consistency step count differs")
    for source in config.forbidden_config_paths:
        if set(source) != {"kind", "path", "sha256"} or not _is_sha256(
            source["sha256"]
        ):
            raise ValueError("quality consistency forbidden identity differs")
        if source["kind"] not in {"synthetic_quality", "scaled_quality"}:
            raise ValueError("quality consistency forbidden kind differs")
    if len(config.benchmark_sources) != 3:
        raise ValueError("quality consistency benchmark sources differ")
    for source in config.benchmark_sources:
        if (
            set(source) != {"name", "path", "sha256", "prompt_column"}
            or not _is_sha256(source["sha256"])
        ):
            raise ValueError("quality consistency benchmark identity differs")
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
        raise ValueError("quality consistency policy differs")


def _expression(family: str, value: int, index: int) -> str:
    if family == "repeated_operand":
        left = value * 7 + 101
        repeated = value * 3 + 29
        return (
            f"({left} + {repeated}) * {2 + index % 3} - {repeated}"
        )
    if family == "mixed_products":
        first = value * 5 + 37
        second = value * 4 + 53
        return (
            f"{first} * {3 + index % 5} + "
            f"{second} * {2 + index % 4} - {value + 19}"
        )
    if family == "exact_division":
        divisor = 3 + index % 5
        quotient = value * 3 + 401
        addend = value + 17
        numerator = quotient * divisor - addend
        return f"({numerator} + {addend}) / {divisor}"
    if family == "nested_offset":
        base = value * 6 + 701
        offset = value * 2 + 83
        return f"({base} - {offset}) * {3 + index % 3} + {value + 41}"
    raise ValueError(f"unsupported consistency family: {family}")


def _pair(
    config: QualityConsistencyConfig,
    *,
    family: str,
    index: int,
    offset: int,
) -> dict[str, str]:
    expression = _expression(family, offset + index, index)
    expected = format_number(evaluate_arithmetic(expression))
    pair_digest = hashlib.sha256(
        f"{family}\0{expression}".encode()
    ).hexdigest()
    return {
        "pair_id": f"quality-consistency-{family}-{pair_digest[:16]}",
        "family": family,
        "expression": expression,
        "expected": expected,
        "process_prompt": config.process_prompt_template.format(
            expression=expression
        ),
        "final_prompt": config.final_prompt_template.format(
            expression=expression
        ),
        "process_target": config.process_target_template.format(
            expression=expression,
            expected=expected,
        ),
        "final_target": config.final_target_template.format(
            expected=expected
        ),
    }


def build_dataset(config: QualityConsistencyConfig) -> dict[str, Any]:
    train = [
        _pair(
            config,
            family=family,
            index=index,
            offset=config.train_range_offset,
        )
        for family in FAMILIES
        for index in range(config.train_pairs_per_family)
    ]
    dev_pairs = [
        _pair(
            config,
            family=family,
            index=index,
            offset=config.dev_range_offset,
        )
        for family in FAMILIES
        for index in range(config.dev_cases_per_family)
    ]
    random.Random(config.seed).shuffle(train)
    random.Random(config.seed + 1).shuffle(dev_pairs)
    if (
        {row["expression"] for row in train}
        & {row["expression"] for row in dev_pairs}
    ):
        raise ValueError("quality consistency train/dev overlap")
    return {
        "schema_version": "nano_train_quality_consistency_dataset_v1",
        "dataset_id": "qwen35-quality-consistency-v1",
        "train_pairs": train,
        "dev_pairs": dev_pairs,
        "identity": {
            "train_pair_ids_sha256": _sha256_lines(
                sorted(row["pair_id"] for row in train)
            ),
            "dev_pair_ids_sha256": _sha256_lines(
                sorted(row["pair_id"] for row in dev_pairs)
            ),
        },
    }


def _sha256_lines(values: list[str]) -> str:
    return hashlib.sha256("\n".join(values).encode()).hexdigest()


def public_contract(dataset: dict[str, Any]) -> dict[str, Any]:
    def rows(key: str) -> list[dict[str, str]]:
        return [
            {
                "pair_id": row["pair_id"],
                "family": row["family"],
                "process_prompt_sha256": hashlib.sha256(
                    row["process_prompt"].encode()
                ).hexdigest(),
                "final_prompt_sha256": hashlib.sha256(
                    row["final_prompt"].encode()
                ).hexdigest(),
                "process_target_sha256": hashlib.sha256(
                    row["process_target"].encode()
                ).hexdigest(),
                "final_target_sha256": hashlib.sha256(
                    row["final_target"].encode()
                ).hexdigest(),
            }
            for row in dataset[key]
        ]

    return {
        "schema_version": "nano_train_quality_consistency_contract_v1",
        "dataset_id": dataset["dataset_id"],
        "train_pairs": rows("train_pairs"),
        "dev_pairs": rows("dev_pairs"),
        "identity": dataset["identity"],
    }


def forbidden_prompt_hashes(
    config: QualityConsistencyConfig,
) -> set[str]:
    result: set[str] = set()
    for source in config.forbidden_config_paths:
        path = Path(source["path"])
        if sha256_file(path) != source["sha256"]:
            raise ValueError("quality consistency forbidden config mismatch")
        if source["kind"] == "synthetic_quality":
            forbidden_config = load_synthetic_quality_config(path)
            prompts = [
                row["prompt"]
                for row in build_synthetic_quality_cases(forbidden_config)
            ]
        else:
            forbidden_config = load_scaled_quality_config(path)
            forbidden_dataset = build_scaled_quality_dataset(forbidden_config)
            prompts = [
                row["prompt"]
                for row in (
                    *forbidden_dataset["train"],
                    *forbidden_dataset["dev"],
                )
            ]
        result.update(
            hashlib.sha256(prompt.encode()).hexdigest()
            for prompt in prompts
        )
    return result


def dataset_prompt_hashes(dataset: dict[str, Any]) -> set[str]:
    return {
        hashlib.sha256(prompt.encode()).hexdigest()
        for pair in (
            *dataset["train_pairs"],
            *dataset["dev_pairs"],
        )
        for prompt in (pair["process_prompt"], pair["final_prompt"])
    }


def benchmark_prompt_hashes(
    config: QualityConsistencyConfig,
) -> tuple[set[str], dict[str, int]]:
    import pyarrow.parquet as parquet

    result: set[str] = set()
    counts = {}
    normalize = lambda value: " ".join(str(value).casefold().split())
    for source in config.benchmark_sources:
        path = Path(source["path"])
        if sha256_file(path) != source["sha256"]:
            raise ValueError(
                f"quality consistency benchmark mismatch: {source['name']}"
            )
        values = parquet.read_table(
            path,
            columns=[source["prompt_column"]],
        )[source["prompt_column"]].to_pylist()
        counts[source["name"]] = len(values)
        result.update(
            hashlib.sha256(normalize(value).encode()).hexdigest()
            for value in values
        )
    return result, counts


def normalized_dataset_prompt_hashes(
    dataset: dict[str, Any],
) -> set[str]:
    normalize = lambda value: " ".join(value.casefold().split())
    return {
        hashlib.sha256(normalize(prompt).encode()).hexdigest()
        for pair in (
            *dataset["train_pairs"],
            *dataset["dev_pairs"],
        )
        for prompt in (pair["process_prompt"], pair["final_prompt"])
    }


def _verify_identity(config: QualityConsistencyConfig) -> None:
    model = Path(config.model_path)
    if (
        sha256_file(model / "config.json") != config.model_config_sha256
        or sha256_file(model / "model.safetensors.index.json")
        != config.model_index_sha256
        or sha256_tree(Path(config.anchor_adapter_path))
        != config.anchor_adapter_sha256
    ):
        raise ValueError("quality consistency model/anchor identity mismatch")
    for shard in config.weight_shards:
        path = model / shard["name"]
        if (
            path.stat().st_size != shard["bytes"]
            or sha256_file(path) != shard["sha256"]
        ):
            raise ValueError("quality consistency model shard mismatch")


def _chat_prompt(
    tokenizer: Any,
    config: QualityConsistencyConfig,
    content: str,
) -> str:
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": config.system_prompt},
            {"role": "user", "content": content},
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def _tokenize_pair(
    tokenizer: Any,
    config: QualityConsistencyConfig,
    pair: dict[str, str],
) -> dict[str, Any]:
    result = {"pair_id": pair["pair_id"], "family": pair["family"]}
    for view in ("process", "final"):
        prompt = _chat_prompt(tokenizer, config, pair[f"{view}_prompt"])
        target = pair[f"{view}_target"]
        prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
        input_ids = tokenizer(
            prompt + target + tokenizer.eos_token,
            add_special_tokens=False,
        ).input_ids
        if len(input_ids) > config.max_length:
            raise ValueError("quality consistency sequence exceeds max_length")
        result[view] = {
            "input_ids": input_ids,
            "labels": [-100] * len(prompt_ids)
            + input_ids[len(prompt_ids) :],
        }
    process_labels = [
        label for label in result["process"]["labels"] if label != -100
    ]
    final_labels = [
        label for label in result["final"]["labels"] if label != -100
    ]
    if process_labels[-len(final_labels) :] != final_labels:
        raise ValueError("quality consistency final suffix is not aligned")
    return result


def _batch(
    value: dict[str, list[int]],
    *,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    input_ids = torch.tensor(
        [value["input_ids"]],
        dtype=torch.long,
        device=device,
    )
    return {
        "input_ids": input_ids,
        "labels": torch.tensor(
            [value["labels"]],
            dtype=torch.long,
            device=device,
        ),
        "attention_mask": torch.ones_like(input_ids),
    }


def _dev_rows(dataset: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {
            "case_id": pair["pair_id"],
            "family": pair["family"],
            "prompt": pair["final_prompt"],
            "target": pair["final_target"],
            "expected": pair["expected"],
        }
        for pair in dataset["dev_pairs"]
    ]


@torch.inference_mode()
def evaluate_final_only(
    model: Any,
    tokenizer: Any,
    config: QualityConsistencyConfig,
    rows: list[dict[str, str]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    device = next(model.parameters()).device
    tokenizer.padding_side = "left"
    model.eval()
    results = []
    for offset in range(0, len(rows), config.generation_batch_size):
        selected = rows[offset : offset + config.generation_batch_size]
        texts = [
            _chat_prompt(tokenizer, config, row["prompt"])
            for row in selected
        ]
        batch = tokenizer(
            texts,
            add_special_tokens=False,
            padding=True,
            return_tensors="pt",
        ).to(device)
        output = model.generate(
            **batch,
            do_sample=False,
            max_new_tokens=config.generation_max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
        )
        prompt_length = batch["input_ids"].shape[1]
        for row, sequence in zip(selected, output):
            generated = tokenizer.decode(
                sequence[prompt_length:],
                skip_special_tokens=True,
            ).strip()
            results.append(
                {
                    "case_id": row["case_id"],
                    "family": row["family"],
                    "output": generated,
                    "correct": generated == row["target"],
                    "parse_failure": not generated.startswith("FINAL: "),
                    "output_sha256": hashlib.sha256(
                        generated.encode()
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


def run(config: QualityConsistencyConfig) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("quality consistency requires CUDA")
    _verify_identity(config)
    dataset = build_dataset(config)
    contract = public_contract(dataset)
    if dataset_prompt_hashes(dataset) & forbidden_prompt_hashes(config):
        raise ValueError("quality consistency overlaps observed quality data")
    benchmark_hashes, _ = benchmark_prompt_hashes(config)
    if normalized_dataset_prompt_hashes(dataset) & benchmark_hashes:
        raise ValueError("quality consistency overlaps benchmark prompts")
    set_seed(config.seed)
    output_root = Path(config.output_dir)
    adapter_dir = output_root / "adapter"
    output_root.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(
        config.anchor_adapter_path,
        local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenized = [
        _tokenize_pair(tokenizer, config, pair)
        for pair in dataset["train_pairs"]
    ]
    dev = _dev_rows(dataset)
    started = time.time()
    model = Qwen3_5ForCausalLM.from_pretrained(
        config.model_path,
        local_files_only=True,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
    ).to(device)
    model.config.use_cache = False
    model = PeftModel.from_pretrained(
        model,
        config.anchor_adapter_path,
        is_trainable=True,
    ).to(device)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    baseline_metrics, baseline_rows = evaluate_final_only(
        model, tokenizer, config, dev
    )
    trainable = _trainable_parameters(model)
    optimizer = AdamW(
        trainable,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        eps=1e-6,
    )
    curve = []
    model.train()
    for index, pair in enumerate(tokenized):
        step = index + 1
        process_batch = _batch(pair["process"], device=device)
        final_batch = _batch(pair["final"], device=device)
        process_outputs = model(**process_batch, use_cache=False)
        process_component = config.process_ce_weight * process_outputs.loss
        _assert_finite_loss(process_component, step=step)
        process_component.backward()
        teacher_logits, teacher_labels = target_prediction_logits(
            process_outputs.logits,
            process_batch["labels"],
        )
        final_labels = supervised_target_labels(final_batch["labels"])
        teacher_logits = align_teacher_logits(
            teacher_logits,
            teacher_labels,
            final_labels,
        ).detach()
        process_ce = float(process_outputs.loss.detach().cpu())
        del process_outputs

        final_outputs = model(**final_batch, use_cache=False)
        student_logits, student_labels = target_prediction_logits(
            final_outputs.logits,
            final_batch["labels"],
        )
        if not torch.equal(student_labels, final_labels):
            raise ValueError("quality consistency final labels changed")
        consistency = paired_consistency_kl(
            teacher_logits,
            student_logits,
            temperature=config.consistency_temperature,
            teacher_detach=config.teacher_detach,
        )
        final_component = (
            config.final_ce_weight * final_outputs.loss
            + config.consistency_weight * consistency
        )
        _assert_finite_loss(final_component, step=step)
        final_component.backward()
        try:
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
                stage="quality_consistency",
                error=error,
            )
            raise
        optimizer.zero_grad(set_to_none=True)
        curve.append(
            {
                "step": step,
                "process_ce": process_ce,
                "final_ce": float(final_outputs.loss.detach().cpu()),
                "consistency_kl": float(consistency.detach().cpu()),
                "gradient_norm": float(gradient_norm.detach().cpu()),
            }
        )
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir, safe_serialization=True)
    tokenizer.save_pretrained(adapter_dir)
    post_metrics, post_rows = evaluate_final_only(
        model, tokenizer, config, dev
    )
    generations = {"baseline": baseline_rows, "post": post_rows}
    generations_path = output_root / "generations.json"
    generations_path.write_text(
        json.dumps(generations, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    result = {
        "schema_version": RESULT_SCHEMA,
        "experiment_id": config.experiment_id,
        "dataset_contract": contract,
        "identity": {
            "anchor_adapter_sha256": config.anchor_adapter_sha256,
            "adapter_sha256": sha256_tree(adapter_dir),
            "generations_sha256": sha256_file(generations_path),
        },
        "training": {
            "optimizer_steps": len(curve),
            "train_pairs": len(tokenized),
            "all_components_finite": all(
                all(
                    math.isfinite(row[key])
                    for key in (
                        "process_ce",
                        "final_ce",
                        "consistency_kl",
                        "gradient_norm",
                    )
                )
                for row in curve
            ),
            "loss_curve": curve,
        },
        "baseline_dev": baseline_metrics,
        "post_dev": post_metrics,
        "comparison": paired_comparison(
            post_rows,
            baseline_rows,
            bootstrap_samples=config.bootstrap_samples,
            bootstrap_seed=config.bootstrap_seed,
        ),
        "hardware": {
            "gpu_name": torch.cuda.get_device_name(device),
            "peak_allocated_gib": (
                torch.cuda.max_memory_allocated(device) / 2**30
            ),
        },
        "dependencies": dependency_versions(),
        "failure_receipt_exists": (output_root / "failure.json").exists(),
        "wall_seconds": time.time() - started,
    }
    (output_root / "metrics.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def validate_reload(config: QualityConsistencyConfig) -> dict[str, Any]:
    output_root = Path(config.output_dir)
    adapter = output_root / "adapter"
    metrics = json.loads(
        (output_root / "metrics.json").read_text(encoding="utf-8")
    )
    generations = json.loads(
        (output_root / "generations.json").read_text(encoding="utf-8")
    )
    dataset = build_dataset(config)
    tokenizer = AutoTokenizer.from_pretrained(adapter, local_files_only=True)
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
        adapter,
        is_trainable=False,
    ).cuda()
    post_metrics, post_rows = evaluate_final_only(
        model,
        tokenizer,
        config,
        _dev_rows(dataset),
    )
    receipt = {
        "schema_version": "nano_train_quality_consistency_reload_v1",
        "experiment_id": config.experiment_id,
        "adapter_sha256": sha256_tree(adapter),
        "metrics_exact": post_metrics == metrics["post_dev"],
        "generations_exact": post_rows == generations["post"],
        "reload_success": (
            post_metrics == metrics["post_dev"]
            and post_rows == generations["post"]
            and sha256_tree(adapter)
            == metrics["identity"]["adapter_sha256"]
        ),
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
    }
    (output_root / "reload_validation.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return receipt
