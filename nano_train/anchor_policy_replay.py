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
import torch.nn.functional as functional
from peft import PeftModel
from torch.optim import AdamW
from transformers import AutoTokenizer, Qwen3_5ForCausalLM

from nano_train.data import evaluate_arithmetic, format_number
from nano_train.preservation_dual_view import (
    FAMILIES,
    _batch,
    _expression,
    _full_consistency_step,
    _tokenize_pair,
    public_contract,
)
from nano_train.quality_consistency import (
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


CONFIG_SCHEMA = "nano_train_anchor_policy_replay_v1"
RESULT_SCHEMA = "nano_train_anchor_policy_replay_result_v1"
CACHE_SCHEMA = "nano_train_anchor_policy_cache_v1"
CACHE_RECEIPT_SCHEMA = "nano_train_anchor_policy_cache_public_v1"
ARMS = ("control", "treatment")


@dataclass(frozen=True)
class AnchorPolicyReplayConfig:
    schema_version: str
    experiment_id: str
    model_path: str
    model_config_sha256: str
    model_index_sha256: str
    weight_shards: tuple[dict[str, Any], ...]
    anchor_adapter_path: str
    anchor_adapter_sha256: str
    teacher_cache_path: str
    teacher_cache_receipt_path: str
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
    anchor_policy_top_k: int
    anchor_policy_temperature: float
    control_anchor_policy_kl_weight: float
    treatment_anchor_policy_kl_weight: float
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


def load_config(path: str | Path) -> AnchorPolicyReplayConfig:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if set(raw) != set(AnchorPolicyReplayConfig.__dataclass_fields__):
        raise ValueError("anchor policy replay config fields differ")
    raw["weight_shards"] = tuple(raw["weight_shards"])
    raw["arms"] = tuple(raw["arms"])
    raw["forbidden_config_paths"] = tuple(raw["forbidden_config_paths"])
    raw["benchmark_sources"] = tuple(raw["benchmark_sources"])
    config = AnchorPolicyReplayConfig(**raw)
    validate_config(config)
    return config


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def validate_config(config: AnchorPolicyReplayConfig) -> None:
    if config.schema_version != CONFIG_SCHEMA:
        raise ValueError("unsupported anchor policy replay schema")
    expected = {
        "arms": ARMS,
        "seed": 20260820,
        "dtype": "float32",
        "train_pairs_per_family": 64,
        "dev_cases_per_family": 64,
        "train_range_offset": 60000,
        "dev_range_offset": 70000,
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
        "anchor_policy_top_k": 64,
        "anchor_policy_temperature": 1.0,
        "control_anchor_policy_kl_weight": 0.0,
        "treatment_anchor_policy_kl_weight": 1.0,
        "control_second_step": "final_ce_only",
        "treatment_second_step": "final_ce_plus_anchor_policy_kl",
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
        "teacher_cache_path": (
            "artifacts/qwen35-anchor-policy-replay-v1/teacher_cache.json"
        ),
        "teacher_cache_receipt_path": (
            "docs/experiments/"
            "qwen35_anchor_policy_teacher_cache_v1.public.json"
        ),
    }
    for field, expected_value in expected.items():
        if getattr(config, field) != expected_value:
            raise ValueError(
                f"anchor policy replay freezes {field}={expected_value}"
            )
    if (
        not _is_sha256(config.model_config_sha256)
        or not _is_sha256(config.model_index_sha256)
        or not _is_sha256(config.anchor_adapter_sha256)
        or len(config.weight_shards) != 2
    ):
        raise ValueError("anchor policy replay identity differs")
    if (
        config.max_steps_per_arm
        != config.train_pairs_per_family
        * len(FAMILIES)
        * config.steps_per_pair
    ):
        raise ValueError("anchor policy replay step count differs")
    allowed_kinds = {
        "synthetic_quality",
        "scaled_quality",
        "quality_consistency",
        "consistency_route",
        "confidence_route",
        "preservation_dual_view",
    }
    if {row["kind"] for row in config.forbidden_config_paths} != allowed_kinds:
        raise ValueError("anchor policy replay forbidden kinds differ")
    for source in (*config.forbidden_config_paths, *config.benchmark_sources):
        if not _is_sha256(source["sha256"]):
            raise ValueError("anchor policy replay source identity differs")
    required_policy = {
        "contains_benchmark_rows": False,
        "contains_benchmark_outputs": False,
        "contains_canary_rows": False,
        "contains_holdout_rows": False,
        "contains_observed_quality_rows": False,
        "uses_observed_quality_outputs": False,
        "teacher_uses_evaluation_expected_answer": False,
        "teacher_uses_case_correctness": False,
        "teacher_uses_training_target_prefix": True,
        "training_allowed": True,
        "benchmark_access_after_result": False,
        "canary_access_after_result": False,
    }
    if config.policy != required_policy:
        raise ValueError("anchor policy replay policy differs")


def _pair(
    config: AnchorPolicyReplayConfig,
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
        "pair_id": f"anchor-policy-{split}-{family}-{digest[:16]}",
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


def build_dataset(config: AnchorPolicyReplayConfig) -> dict[str, Any]:
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
        raise ValueError("anchor policy replay train/dev overlap")
    return {
        "schema_version": "nano_train_anchor_policy_replay_dataset_v1",
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


def build_step_schedule(
    config: AnchorPolicyReplayConfig,
    dataset: dict[str, Any],
    arm_id: str,
) -> list[dict[str, str]]:
    if arm_id not in ARMS:
        raise ValueError("anchor policy replay arm differs")
    second = (
        config.control_second_step
        if arm_id == "control"
        else config.treatment_second_step
    )
    schedule = []
    for pair in dataset["train_pairs"]:
        schedule.append(
            {"pair_id": pair["pair_id"], "kind": "full_consistency"}
        )
        schedule.append({"pair_id": pair["pair_id"], "kind": second})
    if len(schedule) != config.max_steps_per_arm:
        raise ValueError("anchor policy replay schedule differs")
    return schedule


def dataset_prompt_hashes(dataset: dict[str, Any]) -> set[str]:
    return {
        hashlib.sha256(prompt.encode()).hexdigest()
        for pair in (*dataset["train_pairs"], *dataset["dev_pairs"])
        for prompt in (pair["process_prompt"], pair["final_prompt"])
    }


def normalized_dataset_prompt_hashes(dataset: dict[str, Any]) -> set[str]:
    normalize = lambda value: " ".join(value.casefold().split())
    return {
        hashlib.sha256(normalize(prompt).encode()).hexdigest()
        for pair in (*dataset["train_pairs"], *dataset["dev_pairs"])
        for prompt in (pair["process_prompt"], pair["final_prompt"])
    }


def forbidden_prompt_hashes(config: AnchorPolicyReplayConfig) -> set[str]:
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
    from nano_train.preservation_dual_view import (
        build_dataset as build_dual_view_dataset,
    )
    from nano_train.preservation_dual_view import (
        load_config as load_dual_view_config,
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
            raise ValueError("anchor policy replay forbidden config mismatch")
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
        elif kind == "preservation_dual_view":
            dataset = build_dual_view_dataset(load_dual_view_config(path))
            prompts = [
                prompt
                for row in (*dataset["train_pairs"], *dataset["dev_pairs"])
                for prompt in (row["process_prompt"], row["final_prompt"])
            ]
        else:
            raise ValueError("unsupported anchor policy forbidden kind")
        result.update(
            hashlib.sha256(prompt.encode()).hexdigest() for prompt in prompts
        )
    return result


def contamination_audit(
    config: AnchorPolicyReplayConfig,
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


def verify_identity(config: AnchorPolicyReplayConfig) -> dict[str, Any]:
    model = Path(config.model_path)
    if (
        sha256_file(model / "config.json") != config.model_config_sha256
        or sha256_file(model / "model.safetensors.index.json")
        != config.model_index_sha256
        or sha256_tree(Path(config.anchor_adapter_path))
        != config.anchor_adapter_sha256
    ):
        raise ValueError("anchor policy replay model identity mismatch")
    shards = []
    for shard in config.weight_shards:
        path = model / shard["name"]
        if (
            path.stat().st_size != shard["bytes"]
            or sha256_file(path) != shard["sha256"]
        ):
            raise ValueError("anchor policy replay shard mismatch")
        shards.append({**shard, "verified": True})
    return {
        "model_config_sha256": config.model_config_sha256,
        "model_index_sha256": config.model_index_sha256,
        "anchor_adapter_sha256": config.anchor_adapter_sha256,
        "weight_shards": shards,
    }


def final_input_sha256(value: dict[str, list[int]]) -> str:
    canonical = json.dumps(
        value["input_ids"],
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def compress_policy(
    logits: torch.Tensor,
    *,
    top_k: int,
    temperature: float,
) -> list[dict[str, Any]]:
    if logits.ndim != 2 or top_k <= 0 or top_k >= logits.shape[-1]:
        raise ValueError("anchor policy compression shape differs")
    if temperature <= 0:
        raise ValueError("anchor policy temperature must be positive")
    logprobs = functional.log_softmax(
        logits.double() / temperature,
        dim=-1,
    )
    top_logprobs, top_ids = torch.topk(logprobs, k=top_k, dim=-1)
    top_probabilities = top_logprobs.exp()
    other_probabilities = (
        1.0 - top_probabilities.sum(dim=-1)
    ).clamp_min(torch.finfo(torch.float64).tiny)
    categories = torch.cat(
        [top_probabilities, other_probabilities.unsqueeze(-1)],
        dim=-1,
    )
    categories = categories / categories.sum(dim=-1, keepdim=True)
    category_logprobs = categories.log()
    return [
        {
            "top_token_ids": top_ids[index].cpu().tolist(),
            "top_logprobs": category_logprobs[index, :-1].cpu().tolist(),
            "other_logprob": float(category_logprobs[index, -1].cpu()),
        }
        for index in range(logits.shape[0])
    ]


def aggregated_policy_kl(
    student_logits: torch.Tensor,
    teacher_positions: list[dict[str, Any]],
    *,
    temperature: float,
) -> torch.Tensor:
    if (
        student_logits.ndim != 2
        or student_logits.shape[0] != len(teacher_positions)
        or temperature <= 0
    ):
        raise ValueError("anchor policy KL shape differs")
    device = student_logits.device
    top_ids = torch.tensor(
        [row["top_token_ids"] for row in teacher_positions],
        dtype=torch.long,
        device=device,
    )
    teacher_top_logprobs = torch.tensor(
        [row["top_logprobs"] for row in teacher_positions],
        dtype=torch.float64,
        device=device,
    )
    teacher_other_logprobs = torch.tensor(
        [row["other_logprob"] for row in teacher_positions],
        dtype=torch.float64,
        device=device,
    )
    if (
        top_ids.ndim != 2
        or teacher_top_logprobs.shape != top_ids.shape
        or teacher_other_logprobs.shape != (student_logits.shape[0],)
    ):
        raise ValueError("anchor policy teacher cache shape differs")
    student_logprobs = functional.log_softmax(
        student_logits.double() / temperature,
        dim=-1,
    )
    student_top_logprobs = student_logprobs.gather(-1, top_ids)
    student_top_mass = student_top_logprobs.exp().sum(dim=-1)
    student_other_logprobs = torch.log(
        (1.0 - student_top_mass).clamp_min(
            torch.finfo(torch.float64).tiny
        )
    )
    teacher_top_probabilities = teacher_top_logprobs.exp()
    teacher_other_probabilities = teacher_other_logprobs.exp()
    per_position = (
        teacher_top_probabilities
        * (teacher_top_logprobs - student_top_logprobs)
    ).sum(dim=-1) + teacher_other_probabilities * (
        teacher_other_logprobs - student_other_logprobs
    )
    return (
        per_position.mean() * (temperature**2)
    ).to(student_logits.dtype)


def _teacher_model(config: AnchorPolicyReplayConfig) -> tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(
        config.anchor_adapter_path,
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
        config.anchor_adapter_path,
        is_trainable=False,
    ).cuda()
    model.eval()
    return model, tokenizer


def generate_teacher_cache(
    config: AnchorPolicyReplayConfig,
) -> dict[str, Any]:
    identity = verify_identity(config)
    dataset = build_dataset(config)
    audit = contamination_audit(config, dataset)
    if not audit["passed"]:
        raise ValueError("anchor policy teacher cache contamination detected")
    cache_path = Path(config.teacher_cache_path)
    if cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
        summary = inspect_teacher_cache(config, dataset, cache)
        summary["cache_sha256"] = sha256_file(cache_path)
        summary["reused_existing_cache"] = True
        return summary
    set_seed(config.seed)
    model, tokenizer = _teacher_model(config)
    tokenized = {
        pair["pair_id"]: _tokenize_pair(tokenizer, config, pair)
        for pair in dataset["train_pairs"]
    }
    rows = []
    started = time.time()
    with torch.inference_mode():
        for pair in dataset["train_pairs"]:
            value = tokenized[pair["pair_id"]]["final"]
            batch = _batch(value, device=next(model.parameters()).device)
            logits = model(**batch, use_cache=False).logits
            shifted_logits = logits[:, :-1, :]
            shifted_labels = batch["labels"][:, 1:]
            mask = shifted_labels != -100
            target_logits = shifted_logits[mask]
            positions = compress_policy(
                target_logits,
                top_k=config.anchor_policy_top_k,
                temperature=config.anchor_policy_temperature,
            )
            rows.append(
                {
                    "pair_id": pair["pair_id"],
                    "final_input_sha256": final_input_sha256(value),
                    "target_token_count": len(positions),
                    "positions": positions,
                }
            )
    cache = {
        "schema_version": CACHE_SCHEMA,
        "experiment_id": config.experiment_id,
        "identity": identity,
        "dataset_identity": dataset["identity"],
        "top_k": config.anchor_policy_top_k,
        "temperature": config.anchor_policy_temperature,
        "teacher_uses_evaluation_expected_answer": False,
        "teacher_uses_case_correctness": False,
        "teacher_uses_training_target_prefix": True,
        "rows": rows,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(cache, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = inspect_teacher_cache(config, dataset, cache)
    summary.update(
        {
            "cache_sha256": sha256_file(cache_path),
            "identity": identity,
            "contamination_audit": audit,
            "hardware": {
                "gpu_name": torch.cuda.get_device_name(),
                "peak_allocated_gib": (
                    torch.cuda.max_memory_allocated() / 2**30
                ),
            },
            "wall_seconds": time.time() - started,
            "reused_existing_cache": False,
        }
    )
    summary_path = cache_path.with_name("teacher_cache_summary.json")
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def inspect_teacher_cache(
    config: AnchorPolicyReplayConfig,
    dataset: dict[str, Any],
    cache: dict[str, Any],
) -> dict[str, Any]:
    if (
        cache.get("schema_version") != CACHE_SCHEMA
        or cache.get("experiment_id") != config.experiment_id
        or cache.get("dataset_identity") != dataset["identity"]
        or cache.get("top_k") != config.anchor_policy_top_k
        or cache.get("temperature") != config.anchor_policy_temperature
        or cache.get("teacher_uses_evaluation_expected_answer") is not False
        or cache.get("teacher_uses_case_correctness") is not False
        or cache.get("teacher_uses_training_target_prefix") is not True
        or cache.get("identity", {}).get("anchor_adapter_sha256")
        != config.anchor_adapter_sha256
    ):
        raise ValueError("anchor policy teacher cache identity differs")
    rows = cache.get("rows", [])
    by_id = {row["pair_id"]: row for row in rows}
    expected = {row["pair_id"] for row in dataset["train_pairs"]}
    if set(by_id) != expected or len(by_id) != len(rows):
        raise ValueError("anchor policy teacher cache pair set differs")
    top_masses = []
    target_tokens = 0
    for row in rows:
        positions = row["positions"]
        if row["target_token_count"] != len(positions) or not positions:
            raise ValueError("anchor policy cache token count differs")
        target_tokens += len(positions)
        for position in positions:
            ids = position["top_token_ids"]
            logs = position["top_logprobs"]
            other = position["other_logprob"]
            if (
                len(ids) != config.anchor_policy_top_k
                or len(set(ids)) != len(ids)
                or len(logs) != len(ids)
                or not all(math.isfinite(value) for value in logs)
                or not math.isfinite(other)
            ):
                raise ValueError("anchor policy cache position differs")
            top_mass = sum(math.exp(value) for value in logs)
            total = top_mass + math.exp(other)
            if abs(total - 1.0) > 1e-5 or top_mass <= 0 or top_mass >= 1:
                raise ValueError("anchor policy cache probability differs")
            top_masses.append(top_mass)
    return {
        "schema_version": "nano_train_anchor_policy_cache_summary_v1",
        "experiment_id": config.experiment_id,
        "train_pairs": len(rows),
        "target_token_positions": target_tokens,
        "top_k": config.anchor_policy_top_k,
        "temperature": config.anchor_policy_temperature,
        "minimum_top_k_mass": min(top_masses),
        "mean_top_k_mass": sum(top_masses) / len(top_masses),
        "maximum_top_k_mass": max(top_masses),
        "all_probabilities_finite": True,
        "all_probability_sums_within_1e_5": True,
    }


def load_teacher_cache(
    config: AnchorPolicyReplayConfig,
    dataset: dict[str, Any],
    tokenized: dict[str, dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], str]:
    cache_path = Path(config.teacher_cache_path)
    receipt_path = Path(config.teacher_cache_receipt_path)
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    cache_sha256 = sha256_file(cache_path)
    summary = inspect_teacher_cache(config, dataset, cache)
    if (
        receipt.get("schema_version") != CACHE_RECEIPT_SCHEMA
        or receipt.get("experiment_id") != config.experiment_id
        or receipt.get("identity", {}).get("teacher_cache_sha256")
        != cache_sha256
        or receipt.get("identity", {}).get("anchor_adapter_sha256")
        != config.anchor_adapter_sha256
        or receipt.get("dataset_identity") != dataset["identity"]
        or receipt.get("summary") != summary
        or receipt.get("execution_boundary", {}).get("training_started")
        is not False
    ):
        raise ValueError("anchor policy cache receipt differs")
    by_id = {row["pair_id"]: row for row in cache["rows"]}
    for pair_id, pair in tokenized.items():
        row = by_id[pair_id]
        if (
            row["final_input_sha256"]
            != final_input_sha256(pair["final"])
            or row["target_token_count"]
            != sum(label != -100 for label in pair["final"]["labels"])
        ):
            raise ValueError("anchor policy cache tokenization differs")
    return by_id, cache_sha256


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


def _replay_step(
    model: Any,
    pair: dict[str, Any],
    teacher_row: dict[str, Any],
    config: AnchorPolicyReplayConfig,
    *,
    policy_kl_weight: float,
    device: torch.device,
    step: int,
) -> dict[str, float]:
    batch = _batch(pair["final"], device=device)
    outputs = model(**batch, use_cache=False)
    shifted_logits = outputs.logits[:, :-1, :]
    shifted_labels = batch["labels"][:, 1:]
    student_logits = shifted_logits[shifted_labels != -100]
    policy_kl = aggregated_policy_kl(
        student_logits,
        teacher_row["positions"],
        temperature=config.anchor_policy_temperature,
    )
    objective = (
        config.replay_final_ce_weight * outputs.loss
        + policy_kl_weight * policy_kl
    )
    _assert_finite_loss(objective, step=step)
    objective.backward()
    return {
        "final_ce": float(outputs.loss.detach().cpu()),
        "anchor_policy_kl": float(policy_kl.detach().cpu()),
        "objective": float(objective.detach().cpu()),
    }


def run_arm(
    config: AnchorPolicyReplayConfig,
    *,
    arm_id: str,
) -> dict[str, Any]:
    if arm_id not in ARMS:
        raise ValueError("anchor policy replay arm differs")
    if not torch.cuda.is_available():
        raise RuntimeError("anchor policy replay requires CUDA")
    identity = verify_identity(config)
    dataset = build_dataset(config)
    audit = contamination_audit(config, dataset)
    if not audit["passed"]:
        raise ValueError("anchor policy replay contamination detected")
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
    teacher_cache, teacher_cache_sha256 = load_teacher_cache(
        config,
        dataset,
        tokenized,
    )
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
    policy_kl_weight = (
        config.control_anchor_policy_kl_weight
        if arm_id == "control"
        else config.treatment_anchor_policy_kl_weight
    )
    curve = []
    for index, scheduled in enumerate(schedule):
        step = index + 1
        pair = tokenized[scheduled["pair_id"]]
        if scheduled["kind"] == "full_consistency":
            components = _full_consistency_step(
                model,
                pair,
                config,
                device=device,
                step=step,
            )
            components["anchor_policy_kl"] = 0.0
        elif scheduled["kind"] in {
            "final_ce_only",
            "final_ce_plus_anchor_policy_kl",
        }:
            components = _replay_step(
                model,
                pair,
                teacher_cache[scheduled["pair_id"]],
                config,
                policy_kl_weight=policy_kl_weight,
                device=device,
                step=step,
            )
        else:
            raise ValueError("anchor policy replay step kind differs")
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
                stage=f"anchor_policy_replay_{arm_id}",
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
            "teacher_cache_sha256": teacher_cache_sha256,
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
                row["kind"] == "full_consistency" for row in curve
            ),
            "final_replay_steps": sum(
                row["kind"]
                in {"final_ce_only", "final_ce_plus_anchor_policy_kl"}
                for row in curve
            ),
            "anchor_policy_kl_weight": policy_kl_weight,
            "all_components_finite": all(
                all(
                    math.isfinite(value)
                    for key, value in row.items()
                    if key
                    in {
                        "process_ce",
                        "final_ce",
                        "consistency_kl",
                        "anchor_policy_kl",
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
    config: AnchorPolicyReplayConfig,
    *,
    arm_id: str,
) -> dict[str, Any]:
    if arm_id not in ARMS:
        raise ValueError("anchor policy replay arm differs")
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
        "schema_version": "nano_train_anchor_policy_replay_reload_v1",
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
