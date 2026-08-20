from __future__ import annotations

import hashlib
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoTokenizer, Qwen3_5ForCausalLM

from nano_train.quality_consistency import (
    FAMILIES,
    _expression,
)
from nano_train.sft import (
    dependency_versions,
    set_seed,
    sha256_file,
    sha256_tree,
)
from nano_train.synthetic_quality import paired_comparison


CONFIG_SCHEMA = "nano_train_consistency_route_v1"
RESULT_SCHEMA = "nano_train_consistency_route_result_v1"


@dataclass(frozen=True)
class ConsistencyRouteConfig:
    schema_version: str
    experiment_id: str
    model_path: str
    model_config_sha256: str
    model_index_sha256: str
    weight_shards: tuple[dict[str, Any], ...]
    anchor_adapter_path: str
    anchor_adapter_sha256: str
    consistency_adapter_path: str
    consistency_adapter_sha256: str
    output_dir: str
    case_seed: int
    cases_per_family: int
    range_offset: int
    batch_size: int
    max_new_tokens: int
    temperature: float
    system_prompt: str
    prompt_template: str
    routed_family: str
    bootstrap_samples: int
    bootstrap_seed: int
    forbidden_config_paths: tuple[dict[str, str], ...]
    benchmark_sources: tuple[dict[str, str], ...]
    policy: dict[str, bool]


def load_config(path: str | Path) -> ConsistencyRouteConfig:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if set(raw) != set(ConsistencyRouteConfig.__dataclass_fields__):
        raise ValueError("consistency route config fields differ")
    raw["weight_shards"] = tuple(raw["weight_shards"])
    raw["forbidden_config_paths"] = tuple(raw["forbidden_config_paths"])
    raw["benchmark_sources"] = tuple(raw["benchmark_sources"])
    config = ConsistencyRouteConfig(**raw)
    validate_config(config)
    return config


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def validate_config(config: ConsistencyRouteConfig) -> None:
    if config.schema_version != CONFIG_SCHEMA:
        raise ValueError("unsupported consistency route schema")
    expected = {
        "case_seed": 20260820,
        "cases_per_family": 64,
        "range_offset": 20000,
        "batch_size": 8,
        "max_new_tokens": 32,
        "temperature": 0.0,
        "system_prompt": (
            "Follow the exact output contract. Solve internally without tools "
            "or external information."
        ),
        "prompt_template": (
            "Compute the synthetic arithmetic expression exactly: "
            "{expression}. Return only FINAL: <integer>."
        ),
        "routed_family": "exact_division",
        "bootstrap_samples": 10_000,
        "bootstrap_seed": 20260820,
    }
    for field, expected_value in expected.items():
        if getattr(config, field) != expected_value:
            raise ValueError(
                f"consistency route freezes {field}={expected_value}"
            )
    for value in (
        config.model_config_sha256,
        config.model_index_sha256,
        config.anchor_adapter_sha256,
        config.consistency_adapter_sha256,
    ):
        if not _is_sha256(value):
            raise ValueError("consistency route identity differs")
    if len(config.weight_shards) != 2:
        raise ValueError("consistency route shard count differs")
    for source in (*config.forbidden_config_paths, *config.benchmark_sources):
        if not _is_sha256(source["sha256"]):
            raise ValueError("consistency route source identity differs")
    required_policy = {
        "evaluation_only": True,
        "training_eligible": False,
        "contains_benchmark_rows": False,
        "contains_benchmark_outputs": False,
        "contains_canary_rows": False,
        "contains_holdout_rows": False,
        "contains_observed_quality_rows": False,
        "uses_observed_quality_outputs": False,
    }
    if config.policy != required_policy:
        raise ValueError("consistency route policy differs")


def build_cases(config: ConsistencyRouteConfig) -> list[dict[str, str]]:
    rows = []
    for family in FAMILIES:
        for index in range(config.cases_per_family):
            expression = _expression(
                family,
                config.range_offset + index,
                index,
            )
            from nano_train.data import evaluate_arithmetic, format_number

            expected = format_number(evaluate_arithmetic(expression))
            digest = hashlib.sha256(
                f"{family}\0{expression}".encode()
            ).hexdigest()
            rows.append(
                {
                    "case_id": f"consistency-route-{family}-{digest[:16]}",
                    "family": family,
                    "expression": expression,
                    "expected": expected,
                    "prompt": config.prompt_template.format(
                        expression=expression
                    ),
                    "target": f"FINAL: {expected}",
                }
            )
    random.Random(config.case_seed).shuffle(rows)
    if len(rows) != config.cases_per_family * len(FAMILIES):
        raise ValueError("consistency route case count differs")
    return rows


def public_contract(cases: list[dict[str, str]]) -> dict[str, Any]:
    rows = [
        {
            "case_id": row["case_id"],
            "family": row["family"],
            "prompt_sha256": hashlib.sha256(row["prompt"].encode()).hexdigest(),
            "target_sha256": hashlib.sha256(row["target"].encode()).hexdigest(),
        }
        for row in cases
    ]
    canonical = json.dumps(rows, sort_keys=True, separators=(",", ":"))
    return {
        "schema_version": "nano_train_consistency_route_contract_v1",
        "cases": rows,
        "case_count": len(rows),
        "case_contract_sha256": hashlib.sha256(
            canonical.encode()
        ).hexdigest(),
    }


def contamination_audit(
    config: ConsistencyRouteConfig,
    cases: list[dict[str, str]],
) -> dict[str, Any]:
    import pyarrow.parquet as parquet

    hashes = {
        hashlib.sha256(row["prompt"].encode()).hexdigest() for row in cases
    }
    observed: set[str] = set()
    for source in config.forbidden_config_paths:
        path = Path(source["path"])
        if sha256_file(path) != source["sha256"]:
            raise ValueError("consistency route forbidden identity mismatch")
        raw = json.loads(path.read_text(encoding="utf-8"))
        if source["kind"] == "synthetic_quality":
            from nano_train.synthetic_quality import (
                build_cases as build_synthetic_cases,
            )
            from nano_train.synthetic_quality import (
                load_config as load_synthetic_config,
            )

            observed.update(
                hashlib.sha256(row["prompt"].encode()).hexdigest()
                for row in build_synthetic_cases(
                    load_synthetic_config(path)
                )
            )
        elif source["kind"] == "scaled_quality":
            from nano_train.scaled_quality import (
                build_dataset as build_scaled_dataset,
            )
            from nano_train.scaled_quality import (
                load_config as load_scaled_config,
            )

            dataset = build_scaled_dataset(load_scaled_config(path))
            observed.update(
                hashlib.sha256(row["prompt"].encode()).hexdigest()
                for row in (*dataset["train"], *dataset["dev"])
            )
        elif source["kind"] == "quality_consistency":
            from nano_train.quality_consistency import (
                build_dataset as build_consistency_dataset,
            )
            from nano_train.quality_consistency import (
                load_config as load_consistency_config,
            )

            dataset = build_consistency_dataset(
                load_consistency_config(path)
            )
            observed.update(
                hashlib.sha256(prompt.encode()).hexdigest()
                for pair in (*dataset["train_pairs"], *dataset["dev_pairs"])
                for prompt in (pair["process_prompt"], pair["final_prompt"])
            )
        elif source["kind"] == "consistency_route":
            route_config = load_config(path)
            observed.update(
                hashlib.sha256(row["prompt"].encode()).hexdigest()
                for row in build_cases(route_config)
            )
        else:
            raise ValueError("consistency route forbidden kind differs")
    if hashes & observed:
        raise ValueError("consistency route overlaps observed quality prompts")

    normalize = lambda value: " ".join(str(value).casefold().split())
    benchmark_hashes = set()
    counts = {}
    for source in config.benchmark_sources:
        path = Path(source["path"])
        if sha256_file(path) != source["sha256"]:
            raise ValueError("consistency route benchmark identity mismatch")
        values = parquet.read_table(
            path,
            columns=[source["prompt_column"]],
        )[source["prompt_column"]].to_pylist()
        counts[source["name"]] = len(values)
        benchmark_hashes.update(
            hashlib.sha256(normalize(value).encode()).hexdigest()
            for value in values
        )
    normalized = {
        hashlib.sha256(normalize(row["prompt"]).encode()).hexdigest()
        for row in cases
    }
    if normalized & benchmark_hashes:
        raise ValueError("consistency route overlaps benchmark prompts")
    return {
        "observed_quality_prompt_overlap": 0,
        "benchmark_prompt_overlap": 0,
        "benchmark_rows_hashed": counts,
        "benchmark_outputs_loaded": False,
        "canary_or_holdout_loaded": False,
        "passed": True,
    }


def verify_identity(config: ConsistencyRouteConfig) -> dict[str, Any]:
    model = Path(config.model_path)
    if (
        sha256_file(model / "config.json") != config.model_config_sha256
        or sha256_file(model / "model.safetensors.index.json")
        != config.model_index_sha256
        or sha256_tree(Path(config.anchor_adapter_path))
        != config.anchor_adapter_sha256
        or sha256_tree(Path(config.consistency_adapter_path))
        != config.consistency_adapter_sha256
    ):
        raise ValueError("consistency route model/adapter identity mismatch")
    shards = []
    for shard in config.weight_shards:
        path = model / shard["name"]
        if (
            path.stat().st_size != shard["bytes"]
            or sha256_file(path) != shard["sha256"]
        ):
            raise ValueError("consistency route model shard mismatch")
        shards.append({**shard, "verified": True})
    return {
        "model_config_sha256": config.model_config_sha256,
        "model_index_sha256": config.model_index_sha256,
        "weight_shards": shards,
        "anchor_adapter_sha256": config.anchor_adapter_sha256,
        "consistency_adapter_sha256": config.consistency_adapter_sha256,
    }


def _chat_prompt(
    tokenizer: Any,
    config: ConsistencyRouteConfig,
    prompt: str,
) -> str:
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": config.system_prompt},
            {"role": "user", "content": prompt},
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def run_arm(
    config: ConsistencyRouteConfig,
    *,
    arm_id: str,
) -> dict[str, Any]:
    if arm_id not in {"anchor", "consistency"}:
        raise ValueError("consistency route arm differs")
    identity = verify_identity(config)
    cases = build_cases(config)
    audit = contamination_audit(config, cases)
    set_seed(config.case_seed)
    adapter_path = (
        config.anchor_adapter_path
        if arm_id == "anchor"
        else config.consistency_adapter_path
    )
    device = torch.device("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(
        adapter_path,
        local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = Qwen3_5ForCausalLM.from_pretrained(
        config.model_path,
        local_files_only=True,
        dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to(device)
    model = PeftModel.from_pretrained(
        model,
        adapter_path,
        is_trainable=False,
    ).to(device)
    model.eval()
    started = time.time()
    rows = []
    with torch.inference_mode():
        for offset in range(0, len(cases), config.batch_size):
            selected = cases[offset : offset + config.batch_size]
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
            generated = model.generate(
                **batch,
                do_sample=False,
                max_new_tokens=config.max_new_tokens,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
            )
            prompt_length = batch["input_ids"].shape[1]
            for case, sequence in zip(selected, generated):
                output = tokenizer.decode(
                    sequence[prompt_length:],
                    skip_special_tokens=True,
                ).strip()
                rows.append(
                    {
                        "case_id": case["case_id"],
                        "family": case["family"],
                        "output": output,
                        "correct": output == case["target"],
                        "parse_failure": not output.startswith("FINAL: "),
                        "output_sha256": hashlib.sha256(
                            output.encode()
                        ).hexdigest(),
                    }
                )
    output_root = Path(config.output_dir) / arm_id
    output_root.mkdir(parents=True, exist_ok=True)
    cases_path = output_root / "cases.jsonl"
    cases_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    result = {
        "schema_version": RESULT_SCHEMA,
        "experiment_id": config.experiment_id,
        "arm_id": arm_id,
        "case_contract": public_contract(cases),
        "contamination_audit": audit,
        "identity": {
            **identity,
            "adapter_sha256": sha256_tree(Path(adapter_path)),
            "raw_cases_sha256": sha256_file(cases_path),
        },
        "metrics": _metrics(rows),
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


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_family = {}
    for family in FAMILIES:
        selected = [row for row in rows if row["family"] == family]
        by_family[family] = {
            "cases": len(selected),
            "correct": sum(row["correct"] for row in selected),
            "parse_failures": sum(
                row["parse_failure"] for row in selected
            ),
        }
    correct = sum(row["correct"] for row in rows)
    return {
        "cases": len(rows),
        "correct": correct,
        "accuracy": correct / len(rows),
        "parse_failures": sum(row["parse_failure"] for row in rows),
        "by_family": by_family,
    }


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def routed_rows(
    config: ConsistencyRouteConfig,
    anchor: list[dict[str, Any]],
    consistency: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    anchor_by_id = {row["case_id"]: row for row in anchor}
    consistency_by_id = {row["case_id"]: row for row in consistency}
    if set(anchor_by_id) != set(consistency_by_id):
        raise ValueError("consistency route case sets differ")
    result = []
    for case_id in sorted(anchor_by_id):
        source = (
            consistency_by_id[case_id]
            if anchor_by_id[case_id]["family"] == config.routed_family
            else anchor_by_id[case_id]
        )
        result.append(
            {
                **source,
                "route": (
                    "consistency"
                    if source is consistency_by_id[case_id]
                    else "anchor"
                ),
            }
        )
    return result


def compare_routed(
    config: ConsistencyRouteConfig,
    routed: list[dict[str, Any]],
    anchor: list[dict[str, Any]],
) -> dict[str, Any]:
    return paired_comparison(
        routed,
        anchor,
        bootstrap_samples=config.bootstrap_samples,
        bootstrap_seed=config.bootstrap_seed,
    )
