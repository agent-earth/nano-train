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
from nano_train.quality_consistency import (
    FAMILIES,
    benchmark_prompt_hashes,
    evaluate_final_only,
)
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
from nano_train.synthetic_quality import paired_comparison


CONFIG_SCHEMA = "nano_train_preservation_dual_view_v1"
RESULT_SCHEMA = "nano_train_preservation_dual_view_result_v1"
ARMS = ("control", "treatment")


@dataclass(frozen=True)
class PreservationDualViewConfig:
    schema_version: str
    experiment_id: str
    model_path: str
    model_config_sha256: str
    model_index_sha256: str
    weight_shards: tuple[dict[str, Any], ...]
    anchor_adapter_path: str
    anchor_adapter_sha256: str
    output_dir: str
    arms: tuple[str, ...]
    seed: int
    dtype: str
    train_pairs_per_family: int
    dev_cases_per_family: int
    train_range_offset: int
    dev_range_offset: int
    steps_per_pair: int
    max_steps_per_arm: int
    max_length: int
    learning_rate: float
    weight_decay: float
    process_ce_weight: float
    final_ce_weight: float
    consistency_weight: float
    consistency_temperature: float
    teacher_detach: bool
    replay_final_ce_weight: float
    control_second_step: str
    treatment_second_step: str
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


def load_config(path: str | Path) -> PreservationDualViewConfig:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if set(raw) != set(PreservationDualViewConfig.__dataclass_fields__):
        raise ValueError("preservation dual-view config fields differ")
    raw["weight_shards"] = tuple(raw["weight_shards"])
    raw["arms"] = tuple(raw["arms"])
    raw["forbidden_config_paths"] = tuple(raw["forbidden_config_paths"])
    raw["benchmark_sources"] = tuple(raw["benchmark_sources"])
    config = PreservationDualViewConfig(**raw)
    validate_config(config)
    return config


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def validate_config(config: PreservationDualViewConfig) -> None:
    if config.schema_version != CONFIG_SCHEMA:
        raise ValueError("unsupported preservation dual-view schema")
    expected = {
        "arms": ARMS,
        "seed": 20260820,
        "dtype": "float32",
        "train_pairs_per_family": 64,
        "dev_cases_per_family": 64,
        "train_range_offset": 40000,
        "dev_range_offset": 50000,
        "steps_per_pair": 2,
        "max_steps_per_arm": 512,
        "max_length": 160,
        "learning_rate": 0.00005,
        "weight_decay": 0.0,
        "process_ce_weight": 0.5,
        "final_ce_weight": 0.5,
        "consistency_weight": 1.0,
        "consistency_temperature": 1.0,
        "teacher_detach": True,
        "replay_final_ce_weight": 0.5,
        "control_second_step": "repeat_full_consistency",
        "treatment_second_step": "final_ce_only",
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
                f"preservation dual-view freezes {field}={expected_value}"
            )
    if (
        not _is_sha256(config.model_config_sha256)
        or not _is_sha256(config.model_index_sha256)
        or not _is_sha256(config.anchor_adapter_sha256)
        or len(config.weight_shards) != 2
    ):
        raise ValueError("preservation dual-view identity differs")
    expected_steps = (
        config.train_pairs_per_family
        * len(FAMILIES)
        * config.steps_per_pair
    )
    if config.max_steps_per_arm != expected_steps:
        raise ValueError("preservation dual-view step count differs")
    allowed_kinds = {
        "synthetic_quality",
        "scaled_quality",
        "quality_consistency",
        "consistency_route",
        "confidence_route",
    }
    if {source["kind"] for source in config.forbidden_config_paths} != allowed_kinds:
        raise ValueError("preservation dual-view forbidden kinds differ")
    for source in (*config.forbidden_config_paths, *config.benchmark_sources):
        if not _is_sha256(source["sha256"]):
            raise ValueError("preservation dual-view source identity differs")
    required_policy = {
        "contains_benchmark_rows": False,
        "contains_benchmark_outputs": False,
        "contains_canary_rows": False,
        "contains_holdout_rows": False,
        "contains_observed_quality_rows": False,
        "uses_observed_quality_outputs": False,
        "training_allowed": True,
        "benchmark_access_after_result": False,
        "canary_access_after_result": False,
    }
    if config.policy != required_policy:
        raise ValueError("preservation dual-view policy differs")


def _expression(family: str, value: int, index: int) -> str:
    if family == "repeated_operand":
        left = value * 7 + 101
        repeated = value * 3 + 29
        return f"({left} + {repeated}) * {2 + index % 3} - {repeated}"
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
    raise ValueError(f"unsupported preservation family: {family}")


def _pair(
    config: PreservationDualViewConfig,
    *,
    split: str,
    family: str,
    index: int,
    offset: int,
) -> dict[str, str]:
    expression = _expression(family, offset + index, index)
    expected = format_number(evaluate_arithmetic(expression))
    digest = hashlib.sha256(
        f"{split}\0{family}\0{expression}".encode()
    ).hexdigest()
    return {
        "pair_id": f"dual-view-{split}-{family}-{digest[:16]}",
        "split": split,
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
        "final_target": config.final_target_template.format(expected=expected),
    }


def build_dataset(config: PreservationDualViewConfig) -> dict[str, Any]:
    train_pairs = [
        _pair(
            config,
            split="train",
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
            split="dev",
            family=family,
            index=index,
            offset=config.dev_range_offset,
        )
        for family in FAMILIES
        for index in range(config.dev_cases_per_family)
    ]
    random.Random(config.seed).shuffle(train_pairs)
    random.Random(config.seed + 1).shuffle(dev_pairs)
    if (
        {row["expression"] for row in train_pairs}
        & {row["expression"] for row in dev_pairs}
    ):
        raise ValueError("preservation dual-view train/dev overlap")
    return {
        "schema_version": "nano_train_preservation_dual_view_dataset_v1",
        "dataset_id": config.experiment_id,
        "train_pairs": train_pairs,
        "dev_pairs": dev_pairs,
        "identity": {
            "train_pair_ids_sha256": _sha256_lines(
                sorted(row["pair_id"] for row in train_pairs)
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
        "schema_version": "nano_train_preservation_dual_view_contract_v1",
        "dataset_id": dataset["dataset_id"],
        "train_pairs": rows("train_pairs"),
        "dev_pairs": rows("dev_pairs"),
        "identity": dataset["identity"],
    }


def build_step_schedule(
    config: PreservationDualViewConfig,
    dataset: dict[str, Any],
    arm_id: str,
) -> list[dict[str, str]]:
    if arm_id not in ARMS:
        raise ValueError("preservation dual-view arm differs")
    second_kind = (
        config.control_second_step
        if arm_id == "control"
        else config.treatment_second_step
    )
    schedule = []
    for pair in dataset["train_pairs"]:
        schedule.append(
            {"pair_id": pair["pair_id"], "kind": "full_consistency"}
        )
        schedule.append({"pair_id": pair["pair_id"], "kind": second_kind})
    if len(schedule) != config.max_steps_per_arm:
        raise ValueError("preservation dual-view schedule differs")
    return schedule


def dataset_prompt_hashes(dataset: dict[str, Any]) -> set[str]:
    return {
        hashlib.sha256(prompt.encode()).hexdigest()
        for pair in (*dataset["train_pairs"], *dataset["dev_pairs"])
        for prompt in (pair["process_prompt"], pair["final_prompt"])
    }


def normalized_dataset_prompt_hashes(
    dataset: dict[str, Any],
) -> set[str]:
    normalize = lambda value: " ".join(value.casefold().split())
    return {
        hashlib.sha256(normalize(prompt).encode()).hexdigest()
        for pair in (*dataset["train_pairs"], *dataset["dev_pairs"])
        for prompt in (pair["process_prompt"], pair["final_prompt"])
    }


def forbidden_prompt_hashes(
    config: PreservationDualViewConfig,
) -> set[str]:
    from nano_train.confidence_route import (
        build_cases as build_confidence_cases,
    )
    from nano_train.confidence_route import (
        load_config as load_confidence_config,
    )
    from nano_train.consistency_route import (
        build_cases as build_route_cases,
    )
    from nano_train.consistency_route import (
        load_config as load_route_config,
    )
    from nano_train.quality_consistency import (
        build_dataset as build_consistency_dataset,
    )
    from nano_train.quality_consistency import (
        load_config as load_consistency_config,
    )
    from nano_train.scaled_quality import (
        build_dataset as build_scaled_dataset,
    )
    from nano_train.scaled_quality import load_config as load_scaled_config
    from nano_train.synthetic_quality import (
        build_cases as build_quality_cases,
    )
    from nano_train.synthetic_quality import load_config as load_quality_config

    result: set[str] = set()
    for source in config.forbidden_config_paths:
        path = Path(source["path"])
        if sha256_file(path) != source["sha256"]:
            raise ValueError("preservation dual-view forbidden config mismatch")
        kind = source["kind"]
        if kind == "synthetic_quality":
            prompts = [
                row["prompt"]
                for row in build_quality_cases(load_quality_config(path))
            ]
        elif kind == "scaled_quality":
            dataset = build_scaled_dataset(load_scaled_config(path))
            prompts = [
                row["prompt"] for row in (*dataset["train"], *dataset["dev"])
            ]
        elif kind == "quality_consistency":
            dataset = build_consistency_dataset(load_consistency_config(path))
            prompts = [
                prompt
                for row in (*dataset["train_pairs"], *dataset["dev_pairs"])
                for prompt in (row["process_prompt"], row["final_prompt"])
            ]
        elif kind == "consistency_route":
            prompts = [
                row["prompt"]
                for row in build_route_cases(load_route_config(path))
            ]
        elif kind == "confidence_route":
            prompts = [
                row["prompt"]
                for row in build_confidence_cases(
                    load_confidence_config(path)
                )
            ]
        else:
            raise ValueError("unsupported forbidden config kind")
        result.update(
            hashlib.sha256(prompt.encode()).hexdigest() for prompt in prompts
        )
    return result


def contamination_audit(
    config: PreservationDualViewConfig,
    dataset: dict[str, Any],
) -> dict[str, Any]:
    observed_overlap = dataset_prompt_hashes(dataset) & forbidden_prompt_hashes(
        config
    )
    benchmark_hashes, benchmark_counts = benchmark_prompt_hashes(config)
    benchmark_overlap = (
        normalized_dataset_prompt_hashes(dataset) & benchmark_hashes
    )
    return {
        "observed_quality_prompt_overlap": len(observed_overlap),
        "benchmark_prompt_overlap": len(benchmark_overlap),
        "benchmark_rows_hashed": benchmark_counts,
        "benchmark_outputs_loaded": False,
        "canary_or_holdout_loaded": False,
        "passed": not observed_overlap and not benchmark_overlap,
    }


def verify_identity(config: PreservationDualViewConfig) -> dict[str, Any]:
    model = Path(config.model_path)
    if (
        sha256_file(model / "config.json") != config.model_config_sha256
        or sha256_file(model / "model.safetensors.index.json")
        != config.model_index_sha256
        or sha256_tree(Path(config.anchor_adapter_path))
        != config.anchor_adapter_sha256
    ):
        raise ValueError("preservation dual-view model identity mismatch")
    shards = []
    for shard in config.weight_shards:
        path = model / shard["name"]
        if (
            path.stat().st_size != shard["bytes"]
            or sha256_file(path) != shard["sha256"]
        ):
            raise ValueError("preservation dual-view shard mismatch")
        shards.append({**shard, "verified": True})
    return {
        "model_config_sha256": config.model_config_sha256,
        "model_index_sha256": config.model_index_sha256,
        "anchor_adapter_sha256": config.anchor_adapter_sha256,
        "weight_shards": shards,
    }


def _chat_prompt(
    tokenizer: Any,
    config: PreservationDualViewConfig,
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
    config: PreservationDualViewConfig,
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
            raise ValueError("preservation dual-view sequence exceeds max_length")
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
        raise ValueError("preservation dual-view suffix is not aligned")
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


def _full_consistency_step(
    model: Any,
    pair: dict[str, Any],
    config: PreservationDualViewConfig,
    *,
    device: torch.device,
    step: int,
) -> dict[str, float]:
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
        raise ValueError("preservation dual-view final labels changed")
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
    return {
        "process_ce": process_ce,
        "final_ce": float(final_outputs.loss.detach().cpu()),
        "consistency_kl": float(consistency.detach().cpu()),
        "objective": float(
            (process_component.detach() + final_component.detach()).cpu()
        ),
    }


def _final_replay_step(
    model: Any,
    pair: dict[str, Any],
    config: PreservationDualViewConfig,
    *,
    device: torch.device,
    step: int,
) -> dict[str, float]:
    final_outputs = model(
        **_batch(pair["final"], device=device),
        use_cache=False,
    )
    component = config.replay_final_ce_weight * final_outputs.loss
    _assert_finite_loss(component, step=step)
    component.backward()
    return {
        "final_ce": float(final_outputs.loss.detach().cpu()),
        "objective": float(component.detach().cpu()),
    }


def run_arm(
    config: PreservationDualViewConfig,
    *,
    arm_id: str,
) -> dict[str, Any]:
    if arm_id not in ARMS:
        raise ValueError("preservation dual-view arm differs")
    if not torch.cuda.is_available():
        raise RuntimeError("preservation dual-view requires CUDA")
    identity = verify_identity(config)
    dataset = build_dataset(config)
    audit = contamination_audit(config, dataset)
    if not audit["passed"]:
        raise ValueError("preservation dual-view contamination detected")
    schedule = build_step_schedule(config, dataset, arm_id)
    set_seed(config.seed)
    output_root = Path(config.output_dir) / arm_id
    adapter_dir = output_root / "adapter"
    output_root.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(
        config.anchor_adapter_path,
        local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenized = {
        row["pair_id"]: _tokenize_pair(tokenizer, config, row)
        for row in dataset["train_pairs"]
    }
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
        model,
        tokenizer,
        config,
        _dev_rows(dataset),
    )
    model.train()
    trainable = _trainable_parameters(model)
    optimizer = AdamW(
        trainable,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        eps=1e-6,
    )
    curve = []
    for index, scheduled in enumerate(schedule):
        step = index + 1
        pair = tokenized[scheduled["pair_id"]]
        if scheduled["kind"] in {
            "full_consistency",
            "repeat_full_consistency",
        }:
            components = _full_consistency_step(
                model,
                pair,
                config,
                device=device,
                step=step,
            )
        elif scheduled["kind"] == "final_ce_only":
            components = _final_replay_step(
                model,
                pair,
                config,
                device=device,
                step=step,
            )
        else:
            raise ValueError("preservation dual-view step kind differs")
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
                stage=f"preservation_dual_view_{arm_id}",
                error=error,
            )
            raise
        optimizer.zero_grad(set_to_none=True)
        curve.append(
            {
                "step": step,
                "pair_id": scheduled["pair_id"],
                "kind": scheduled["kind"],
                **components,
                "gradient_norm": float(gradient_norm.detach().cpu()),
            }
        )
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir, safe_serialization=True)
    tokenizer.save_pretrained(adapter_dir)
    post_metrics, post_rows = evaluate_final_only(
        model,
        tokenizer,
        config,
        _dev_rows(dataset),
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
        "arm_id": arm_id,
        "dataset_contract": public_contract(dataset),
        "contamination_audit": audit,
        "identity": {
            **identity,
            "adapter_sha256": sha256_tree(adapter_dir),
            "generations_sha256": sha256_file(generations_path),
            "schedule_sha256": hashlib.sha256(
                json.dumps(
                    schedule,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest(),
        },
        "training": {
            "optimizer_steps": len(curve),
            "train_pairs": len(tokenized),
            "full_consistency_steps": sum(
                row["kind"] in {
                    "full_consistency",
                    "repeat_full_consistency",
                }
                for row in curve
            ),
            "final_replay_steps": sum(
                row["kind"] == "final_ce_only" for row in curve
            ),
            "all_components_finite": all(
                all(
                    math.isfinite(value)
                    for key, value in row.items()
                    if key
                    in {
                        "process_ce",
                        "final_ce",
                        "consistency_kl",
                        "objective",
                        "gradient_norm",
                    }
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


def validate_reload(
    config: PreservationDualViewConfig,
    *,
    arm_id: str,
) -> dict[str, Any]:
    if arm_id not in ARMS:
        raise ValueError("preservation dual-view arm differs")
    output_root = Path(config.output_dir) / arm_id
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
        "schema_version": "nano_train_preservation_dual_view_reload_v1",
        "experiment_id": config.experiment_id,
        "arm_id": arm_id,
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
