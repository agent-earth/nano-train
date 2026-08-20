from __future__ import annotations

import hashlib
import json
import math
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoTokenizer, Qwen3_5ForCausalLM

from nano_train.data import evaluate_arithmetic, format_number
from nano_train.sft import dependency_versions, set_seed, sha256_file, sha256_tree


CONFIG_SCHEMA = "nano_train_synthetic_quality_v1"
RESULT_SCHEMA = "nano_train_synthetic_quality_result_v1"
FINAL_PATTERN = re.compile(r"^FINAL: ([-+]?[0-9]+)$")
FAMILIES = (
    "repeated_operand",
    "mixed_products",
    "exact_division",
    "nested_offset",
)


@dataclass(frozen=True)
class SyntheticQualityConfig:
    schema_version: str
    experiment_id: str
    output_dir: str
    case_seed: int
    cases_per_family: int
    bootstrap_samples: int
    bootstrap_seed: int
    batch_size: int
    max_new_tokens: int
    temperature: float
    system_prompt: str
    prompt_template: str
    model_arms: tuple[dict[str, Any], ...]
    benchmark_sources: tuple[dict[str, str], ...]
    policy: dict[str, bool]


def load_config(path: str | Path) -> SyntheticQualityConfig:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = set(SyntheticQualityConfig.__dataclass_fields__)
    if set(raw) != expected:
        raise ValueError("synthetic quality config fields differ")
    raw["model_arms"] = tuple(raw["model_arms"])
    raw["benchmark_sources"] = tuple(raw["benchmark_sources"])
    config = SyntheticQualityConfig(**raw)
    validate_config(config)
    return config


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def validate_config(config: SyntheticQualityConfig) -> None:
    if config.schema_version != CONFIG_SCHEMA:
        raise ValueError("unsupported synthetic quality schema")
    expected = {
        "case_seed": 20260820,
        "cases_per_family": 24,
        "bootstrap_samples": 10_000,
        "bootstrap_seed": 20260820,
        "batch_size": 8,
        "max_new_tokens": 32,
        "temperature": 0.0,
        "system_prompt": (
            "Follow the exact output contract. Solve internally without tools "
            "or external information."
        ),
        "prompt_template": (
            "Compute the synthetic arithmetic expression exactly: "
            "{expression}. Return only one line in the form FINAL: <integer>."
        ),
    }
    for field, expected_value in expected.items():
        if getattr(config, field) != expected_value:
            raise ValueError(
                f"synthetic quality freezes {field}={expected_value}"
            )
    if [arm.get("arm_id") for arm in config.model_arms] != [
        "base4",
        "rl4",
        "opd4",
        "base9",
    ]:
        raise ValueError("synthetic quality arm order differs")
    for arm in config.model_arms:
        if set(arm) != {
            "arm_id",
            "model_path",
            "model_config_sha256",
            "model_index_sha256",
            "weight_shards",
            "adapter_path",
            "adapter_sha256",
            "dtype",
        }:
            raise ValueError("synthetic quality arm fields differ")
        if (
            not _is_sha256(arm["model_config_sha256"])
            or not _is_sha256(arm["model_index_sha256"])
            or arm["dtype"] not in {"float16", "float32"}
        ):
            raise ValueError("synthetic quality model identity differs")
        expected_shards = 2 if arm["arm_id"] != "base9" else 4
        if len(arm["weight_shards"]) != expected_shards:
            raise ValueError("synthetic quality shard count differs")
        for shard in arm["weight_shards"]:
            if (
                set(shard) != {"name", "bytes", "sha256"}
                or int(shard["bytes"]) <= 0
                or not _is_sha256(shard["sha256"])
            ):
                raise ValueError("synthetic quality shard identity differs")
        needs_adapter = arm["arm_id"] in {"rl4", "opd4"}
        if needs_adapter != bool(arm["adapter_path"]):
            raise ValueError("synthetic quality adapter policy differs")
        if needs_adapter != _is_sha256(arm["adapter_sha256"]):
            raise ValueError("synthetic quality adapter identity differs")
    if len(config.benchmark_sources) != 3:
        raise ValueError("synthetic quality needs three benchmark sources")
    required_policy = {
        "evaluation_only": True,
        "training_eligible": False,
        "contains_benchmark_rows": False,
        "contains_benchmark_outputs": False,
        "contains_canary_rows": False,
        "contains_holdout_rows": False,
        "quality_claim_scope_synthetic_only": True,
    }
    if config.policy != required_policy:
        raise ValueError("synthetic quality policy differs")


def _case_id(family: str, expression: str) -> str:
    digest = hashlib.sha256(f"{family}\0{expression}".encode()).hexdigest()
    return f"synthetic-quality-{family}-{digest[:16]}"


def build_cases(config: SyntheticQualityConfig) -> list[dict[str, str]]:
    cases = []
    for index in range(config.cases_per_family):
        left = 1201 + 17 * index
        repeated = 211 + 11 * index
        multiplier = 2 + index % 3
        expression = f"({left} + {repeated}) * {multiplier} - {repeated}"
        cases.append(
            _case(config, "repeated_operand", index, expression)
        )

        first = 37 + 7 * index
        first_multiplier = 11 + index % 5
        second = 53 + 9 * index
        second_multiplier = 3 + index % 4
        subtract = 19 + 5 * index
        expression = (
            f"{first} * {first_multiplier} + "
            f"{second} * {second_multiplier} - {subtract}"
        )
        cases.append(_case(config, "mixed_products", index, expression))

        divisor = 3 + index % 5
        quotient = 401 + 13 * index
        addend = 17 + 2 * index
        numerator = quotient * divisor - addend
        expression = f"({numerator} + {addend}) / {divisor}"
        cases.append(_case(config, "exact_division", index, expression))

        base = 701 + 19 * index
        offset = 83 + 7 * index
        multiplier = 3 + index % 3
        final_offset = 41 + 5 * index
        expression = (
            f"({base} - {offset}) * {multiplier} + {final_offset}"
        )
        cases.append(_case(config, "nested_offset", index, expression))
    if len(cases) != config.cases_per_family * len(FAMILIES):
        raise ValueError("synthetic quality case count differs")
    ids = [case["case_id"] for case in cases]
    prompts = [case["prompt"] for case in cases]
    if len(ids) != len(set(ids)) or len(prompts) != len(set(prompts)):
        raise ValueError("synthetic quality cases are not unique")
    random.Random(config.case_seed).shuffle(cases)
    return cases


def _case(
    config: SyntheticQualityConfig,
    family: str,
    index: int,
    expression: str,
) -> dict[str, str]:
    expected = format_number(evaluate_arithmetic(expression))
    if not re.fullmatch(r"[-+]?[0-9]+", expected):
        raise ValueError("synthetic quality expected value is not integer")
    return {
        "case_id": _case_id(family, expression),
        "family": family,
        "family_index": str(index),
        "expression": expression,
        "expected": expected,
        "prompt": config.prompt_template.format(expression=expression),
    }


def case_contract(cases: list[dict[str, str]]) -> dict[str, Any]:
    public = [
        {
            "case_id": case["case_id"],
            "family": case["family"],
            "family_index": int(case["family_index"]),
            "prompt_sha256": hashlib.sha256(
                case["prompt"].encode()
            ).hexdigest(),
            "expected_sha256": hashlib.sha256(
                case["expected"].encode()
            ).hexdigest(),
        }
        for case in cases
    ]
    canonical = json.dumps(public, sort_keys=True, separators=(",", ":"))
    return {
        "schema_version": "nano_train_synthetic_quality_cases_v1",
        "cases": public,
        "case_count": len(public),
        "case_contract_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
        "case_ids_sha256": hashlib.sha256(
            "\n".join(sorted(row["case_id"] for row in public)).encode()
        ).hexdigest(),
    }


def contamination_audit(
    config: SyntheticQualityConfig,
    cases: list[dict[str, str]],
) -> dict[str, Any]:
    import pyarrow.parquet as parquet

    normalized = lambda value: " ".join(value.casefold().split())
    synthetic_hashes = {
        hashlib.sha256(normalized(case["prompt"]).encode()).hexdigest()
        for case in cases
    }
    benchmark_hashes = set()
    counts = {}
    for source in config.benchmark_sources:
        path = Path(source["path"])
        if sha256_file(path) != source["sha256"]:
            raise ValueError(f"benchmark identity mismatch: {source['name']}")
        values = parquet.read_table(
            path,
            columns=[source["prompt_column"]],
        )[source["prompt_column"]].to_pylist()
        counts[source["name"]] = len(values)
        benchmark_hashes.update(
            hashlib.sha256(normalized(str(value)).encode()).hexdigest()
            for value in values
        )
    overlap = synthetic_hashes & benchmark_hashes
    if overlap:
        raise ValueError("synthetic quality overlaps benchmark prompts")
    return {
        "benchmark_rows_hashed": counts,
        "exact_normalized_prompt_overlap": 0,
        "benchmark_labels_loaded": False,
        "benchmark_outputs_loaded": False,
        "canary_or_holdout_loaded": False,
        "passed": True,
    }


def verify_arm_identity(arm: dict[str, Any]) -> dict[str, Any]:
    model_path = Path(arm["model_path"])
    if (
        sha256_file(model_path / "config.json")
        != arm["model_config_sha256"]
        or sha256_file(model_path / "model.safetensors.index.json")
        != arm["model_index_sha256"]
    ):
        raise ValueError(f"model metadata mismatch: {arm['arm_id']}")
    shards = []
    for shard in arm["weight_shards"]:
        path = model_path / shard["name"]
        if (
            path.stat().st_size != shard["bytes"]
            or sha256_file(path) != shard["sha256"]
        ):
            raise ValueError(f"model shard mismatch: {arm['arm_id']}")
        shards.append({**shard, "verified": True})
    adapter = None
    if arm["adapter_path"]:
        adapter_path = Path(arm["adapter_path"])
        actual = sha256_tree(adapter_path)
        if actual != arm["adapter_sha256"]:
            raise ValueError(f"adapter identity mismatch: {arm['arm_id']}")
        adapter = {"sha256": actual, "verified": True}
    return {
        "model_config_sha256": arm["model_config_sha256"],
        "model_index_sha256": arm["model_index_sha256"],
        "weight_shards": shards,
        "adapter": adapter,
    }


def _prediction(output: str) -> str | None:
    match = FINAL_PATTERN.fullmatch(output.strip())
    return match.group(1) if match else None


def run_arm(
    config: SyntheticQualityConfig,
    *,
    arm_id: str,
) -> dict[str, Any]:
    arm = next(
        (row for row in config.model_arms if row["arm_id"] == arm_id),
        None,
    )
    if arm is None:
        raise ValueError(f"unknown synthetic quality arm: {arm_id}")
    identity = verify_arm_identity(arm)
    cases = build_cases(config)
    contract = case_contract(cases)
    audit = contamination_audit(config, cases)
    set_seed(config.case_seed)
    dtype = {
        "float16": torch.float16,
        "float32": torch.float32,
    }[arm["dtype"]]
    device = torch.device("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(
        arm["model_path"],
        local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = Qwen3_5ForCausalLM.from_pretrained(
        arm["model_path"],
        local_files_only=True,
        dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    if arm["adapter_path"]:
        model = PeftModel.from_pretrained(
            model,
            arm["adapter_path"],
            is_trainable=False,
        ).to(device)
    model.eval()
    output_root = Path(config.output_dir) / arm_id
    output_root.mkdir(parents=True, exist_ok=True)
    started = time.time()
    rows = []
    with torch.inference_mode():
        for offset in range(0, len(cases), config.batch_size):
            batch_cases = cases[offset : offset + config.batch_size]
            texts = [
                tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": config.system_prompt},
                        {"role": "user", "content": case["prompt"]},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                for case in batch_cases
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
                max_new_tokens=config.max_new_tokens,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
            )
            prompt_length = batch["input_ids"].shape[1]
            for case, sequence in zip(batch_cases, generated):
                output = tokenizer.decode(
                    sequence[prompt_length:],
                    skip_special_tokens=True,
                ).strip()
                prediction = _prediction(output)
                rows.append(
                    {
                        "case_id": case["case_id"],
                        "family": case["family"],
                        "expected": case["expected"],
                        "output": output,
                        "output_sha256": hashlib.sha256(
                            output.encode()
                        ).hexdigest(),
                        "prediction": prediction,
                        "correct": prediction == case["expected"],
                        "parse_failure": prediction is None,
                    }
                )
    path = output_root / "cases.jsonl"
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    by_family = {}
    for family in FAMILIES:
        subset = [row for row in rows if row["family"] == family]
        by_family[family] = {
            "cases": len(subset),
            "correct": sum(row["correct"] for row in subset),
            "parse_failures": sum(
                row["parse_failure"] for row in subset
            ),
        }
    result = {
        "schema_version": RESULT_SCHEMA,
        "experiment_id": config.experiment_id,
        "arm_id": arm_id,
        "identity": identity,
        "case_contract": contract,
        "contamination_audit": audit,
        "metrics": {
            "cases": len(rows),
            "correct": sum(row["correct"] for row in rows),
            "accuracy": sum(row["correct"] for row in rows) / len(rows),
            "parse_failures": sum(row["parse_failure"] for row in rows),
            "by_family": by_family,
        },
        "raw": {
            "cases_sha256": sha256_file(path),
            "cases_path": str(path),
        },
        "dependencies": dependency_versions(),
        "hardware": {
            "gpu_name": torch.cuda.get_device_name(device),
            "peak_allocated_gib": (
                torch.cuda.max_memory_allocated(device) / 2**30
            ),
        },
        "wall_seconds": time.time() - started,
    }
    (output_root / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def paired_comparison(
    candidate_rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    candidate = {row["case_id"]: row for row in candidate_rows}
    baseline = {row["case_id"]: row for row in baseline_rows}
    if set(candidate) != set(baseline):
        raise ValueError("synthetic quality case sets differ")
    case_ids = sorted(candidate)
    deltas = [
        int(candidate[case_id]["correct"])
        - int(baseline[case_id]["correct"])
        for case_id in case_ids
    ]
    randomizer = random.Random(bootstrap_seed)
    samples = []
    for _ in range(bootstrap_samples):
        samples.append(
            sum(
                deltas[randomizer.randrange(len(deltas))]
                for _ in deltas
            )
            / len(deltas)
        )
    samples.sort()
    lower = samples[int(0.025 * bootstrap_samples)]
    upper = samples[int(0.975 * bootstrap_samples) - 1]
    candidate_only = sum(delta == 1 for delta in deltas)
    baseline_only = sum(delta == -1 for delta in deltas)
    return {
        "cases": len(case_ids),
        "candidate_accuracy": sum(
            candidate[case_id]["correct"] for case_id in case_ids
        )
        / len(case_ids),
        "baseline_accuracy": sum(
            baseline[case_id]["correct"] for case_id in case_ids
        )
        / len(case_ids),
        "delta": sum(deltas) / len(deltas),
        "paired_bootstrap_95_ci": [lower, upper],
        "mcnemar_exact_p": _mcnemar_exact(candidate_only, baseline_only),
        "paired_counts": {
            "candidate_only": candidate_only,
            "baseline_only": baseline_only,
            "both_correct": sum(
                candidate[case_id]["correct"]
                and baseline[case_id]["correct"]
                for case_id in case_ids
            ),
            "both_wrong": sum(
                not candidate[case_id]["correct"]
                and not baseline[case_id]["correct"]
                for case_id in case_ids
            ),
        },
    }


def _mcnemar_exact(candidate_only: int, baseline_only: int) -> float:
    discordant = candidate_only + baseline_only
    if discordant == 0:
        return 1.0
    smaller = min(candidate_only, baseline_only)
    probability = sum(
        math.comb(discordant, value) for value in range(smaller + 1)
    ) / (2**discordant)
    return min(1.0, 2.0 * probability)


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def candidate_admission_gates(
    comparison: dict[str, Any],
    candidate_metrics: dict[str, Any],
    baseline_metrics: dict[str, Any],
) -> dict[str, bool]:
    return {
        "point_delta_positive": comparison["delta"] > 0,
        "bootstrap_ci_lower_positive": (
            comparison["paired_bootstrap_95_ci"][0] > 0
        ),
        "mcnemar_below_005": comparison["mcnemar_exact_p"] < 0.05,
        "every_family_non_regression": all(
            candidate_metrics["by_family"][family]["correct"]
            >= baseline_metrics["by_family"][family]["correct"]
            for family in FAMILIES
        ),
        "parse_failures_non_regression": (
            candidate_metrics["parse_failures"]
            <= baseline_metrics["parse_failures"]
        ),
    }
