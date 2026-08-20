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
from transformers import AutoTokenizer, Qwen3_5ForCausalLM

from nano_train.consistency_route import (
    _chat_prompt,
    contamination_audit as route_contamination_audit,
)
from nano_train.quality_consistency import FAMILIES, _expression
from nano_train.sft import (
    dependency_versions,
    set_seed,
    sha256_file,
    sha256_tree,
)
from nano_train.synthetic_quality import paired_comparison


CONFIG_SCHEMA = "nano_train_confidence_route_v1"


@dataclass(frozen=True)
class ConfidenceRouteConfig:
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
    generation_batch_size: int
    scoring_batch_size: int
    max_new_tokens: int
    temperature: float
    system_prompt: str
    prompt_template: str
    selector: str
    tie_policy: str
    bootstrap_samples: int
    bootstrap_seed: int
    forbidden_config_paths: tuple[dict[str, str], ...]
    benchmark_sources: tuple[dict[str, str], ...]
    policy: dict[str, bool]


def load_config(path: str | Path) -> ConfidenceRouteConfig:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if set(raw) != set(ConfidenceRouteConfig.__dataclass_fields__):
        raise ValueError("confidence route config fields differ")
    raw["weight_shards"] = tuple(raw["weight_shards"])
    raw["forbidden_config_paths"] = tuple(raw["forbidden_config_paths"])
    raw["benchmark_sources"] = tuple(raw["benchmark_sources"])
    config = ConfidenceRouteConfig(**raw)
    validate_config(config)
    return config


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def validate_config(config: ConfidenceRouteConfig) -> None:
    if config.schema_version != CONFIG_SCHEMA:
        raise ValueError("unsupported confidence route schema")
    expected = {
        "case_seed": 20260820,
        "cases_per_family": 64,
        "range_offset": 30000,
        "generation_batch_size": 8,
        "scoring_batch_size": 1,
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
        "selector": "cross_model_normalized_logprob_ratio_v1",
        "tie_policy": "anchor",
        "bootstrap_samples": 10_000,
        "bootstrap_seed": 20260820,
    }
    for field, expected_value in expected.items():
        if getattr(config, field) != expected_value:
            raise ValueError(
                f"confidence route freezes {field}={expected_value}"
            )
    for value in (
        config.model_config_sha256,
        config.model_index_sha256,
        config.anchor_adapter_sha256,
        config.consistency_adapter_sha256,
    ):
        if not _is_sha256(value):
            raise ValueError("confidence route identity differs")
    if len(config.weight_shards) != 2:
        raise ValueError("confidence route shard count differs")
    for source in (*config.forbidden_config_paths, *config.benchmark_sources):
        if not _is_sha256(source["sha256"]):
            raise ValueError("confidence route source identity differs")
    required_policy = {
        "evaluation_only": True,
        "training_eligible": False,
        "contains_benchmark_rows": False,
        "contains_benchmark_outputs": False,
        "contains_canary_rows": False,
        "contains_holdout_rows": False,
        "contains_observed_quality_rows": False,
        "uses_observed_quality_outputs": False,
        "selector_uses_expected_answer": False,
    }
    if config.policy != required_policy:
        raise ValueError("confidence route policy differs")


def build_cases(config: ConfidenceRouteConfig) -> list[dict[str, str]]:
    from nano_train.data import evaluate_arithmetic, format_number

    rows = []
    for family in FAMILIES:
        for index in range(config.cases_per_family):
            expression = _expression(
                family,
                config.range_offset + index,
                index,
            )
            expected = format_number(evaluate_arithmetic(expression))
            digest = hashlib.sha256(
                f"{family}\0{expression}".encode()
            ).hexdigest()
            rows.append(
                {
                    "case_id": f"confidence-route-{family}-{digest[:16]}",
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
        "schema_version": "nano_train_confidence_route_contract_v1",
        "cases": rows,
        "case_count": len(rows),
        "case_contract_sha256": hashlib.sha256(
            canonical.encode()
        ).hexdigest(),
    }


def _route_compatible_config(config: ConfidenceRouteConfig) -> Any:
    from types import SimpleNamespace

    return SimpleNamespace(
        forbidden_config_paths=config.forbidden_config_paths,
        benchmark_sources=config.benchmark_sources,
    )


def contamination_audit(
    config: ConfidenceRouteConfig,
    cases: list[dict[str, str]],
) -> dict[str, Any]:
    return route_contamination_audit(_route_compatible_config(config), cases)


def verify_identity(config: ConfidenceRouteConfig) -> dict[str, Any]:
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
        raise ValueError("confidence route identity mismatch")
    shards = []
    for shard in config.weight_shards:
        path = model / shard["name"]
        if (
            path.stat().st_size != shard["bytes"]
            or sha256_file(path) != shard["sha256"]
        ):
            raise ValueError("confidence route model shard mismatch")
        shards.append({**shard, "verified": True})
    return {
        "model_config_sha256": config.model_config_sha256,
        "model_index_sha256": config.model_index_sha256,
        "weight_shards": shards,
        "anchor_adapter_sha256": config.anchor_adapter_sha256,
        "consistency_adapter_sha256": config.consistency_adapter_sha256,
    }


def _load_model(
    config: ConfidenceRouteConfig,
    arm_id: str,
) -> tuple[Any, Any, Path]:
    if arm_id == "anchor":
        adapter = Path(config.anchor_adapter_path)
    elif arm_id == "consistency":
        adapter = Path(config.consistency_adapter_path)
    else:
        raise ValueError("confidence route arm differs")
    tokenizer = AutoTokenizer.from_pretrained(adapter, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = Qwen3_5ForCausalLM.from_pretrained(
        config.model_path,
        local_files_only=True,
        dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).cuda()
    model = PeftModel.from_pretrained(
        model,
        adapter,
        is_trainable=False,
    ).cuda()
    model.eval()
    return model, tokenizer, adapter


def generate_arm(
    config: ConfidenceRouteConfig,
    *,
    arm_id: str,
) -> dict[str, Any]:
    identity = verify_identity(config)
    cases = build_cases(config)
    audit = contamination_audit(config, cases)
    set_seed(config.case_seed)
    model, tokenizer, adapter = _load_model(config, arm_id)
    started = time.time()
    rows = []
    with torch.inference_mode():
        for offset in range(0, len(cases), config.generation_batch_size):
            selected = cases[offset : offset + config.generation_batch_size]
            texts = [
                _chat_prompt(tokenizer, config, row["prompt"])
                for row in selected
            ]
            batch = tokenizer(
                texts,
                add_special_tokens=False,
                padding=True,
                return_tensors="pt",
            ).cuda()
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
    output_root = Path(config.output_dir) / "generation" / arm_id
    output_root.mkdir(parents=True, exist_ok=True)
    cases_path = output_root / "cases.jsonl"
    cases_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    result = {
        "schema_version": "nano_train_confidence_route_generation_v1",
        "experiment_id": config.experiment_id,
        "arm_id": arm_id,
        "case_contract": public_contract(cases),
        "contamination_audit": audit,
        "identity": {
            **identity,
            "adapter_sha256": sha256_tree(adapter),
            "raw_cases_sha256": sha256_file(cases_path),
        },
        "metrics": _metrics(rows),
        "dependencies": dependency_versions(),
        "hardware": {
            "gpu_name": torch.cuda.get_device_name(),
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        },
        "wall_seconds": time.time() - started,
    }
    (output_root / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def score_arm(
    config: ConfidenceRouteConfig,
    *,
    scorer_arm: str,
) -> dict[str, Any]:
    verify_identity(config)
    cases = {row["case_id"]: row for row in build_cases(config)}
    anchor = _load_rows(
        Path(config.output_dir) / "generation/anchor/cases.jsonl"
    )
    consistency = _load_rows(
        Path(config.output_dir) / "generation/consistency/cases.jsonl"
    )
    anchor_by_id = {row["case_id"]: row for row in anchor}
    consistency_by_id = {row["case_id"]: row for row in consistency}
    if set(cases) != set(anchor_by_id) or set(cases) != set(consistency_by_id):
        raise ValueError("confidence route generated cases differ")
    model, tokenizer, adapter = _load_model(config, scorer_arm)
    started = time.time()
    rows = []
    with torch.inference_mode():
        for case_id in sorted(cases):
            prompt = _chat_prompt(
                tokenizer,
                config,
                cases[case_id]["prompt"],
            )
            rows.append(
                {
                    "case_id": case_id,
                    "anchor_candidate_mean_logprob": candidate_mean_logprob(
                        model,
                        tokenizer,
                        prompt,
                        anchor_by_id[case_id]["output"],
                    ),
                    "consistency_candidate_mean_logprob": candidate_mean_logprob(
                        model,
                        tokenizer,
                        prompt,
                        consistency_by_id[case_id]["output"],
                    ),
                }
            )
    output_root = Path(config.output_dir) / "scores" / scorer_arm
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / "scores.jsonl"
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    result = {
        "schema_version": "nano_train_confidence_route_scores_v1",
        "experiment_id": config.experiment_id,
        "scorer_arm": scorer_arm,
        "adapter_sha256": sha256_tree(adapter),
        "rows": len(rows),
        "all_scores_finite": all(
            math.isfinite(row["anchor_candidate_mean_logprob"])
            and math.isfinite(row["consistency_candidate_mean_logprob"])
            for row in rows
        ),
        "scores_sha256": sha256_file(path),
        "wall_seconds": time.time() - started,
    }
    (output_root / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def candidate_mean_logprob(
    model: Any,
    tokenizer: Any,
    prompt: str,
    candidate: str,
) -> float:
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    candidate_ids = tokenizer(
        candidate + tokenizer.eos_token,
        add_special_tokens=False,
    ).input_ids
    full_ids = prompt_ids + candidate_ids
    input_ids = torch.tensor(
        [full_ids],
        dtype=torch.long,
        device=next(model.parameters()).device,
    )
    logits = model(input_ids=input_ids, use_cache=False).logits
    start = len(prompt_ids) - 1
    prediction_logits = logits[:, start : start + len(candidate_ids), :]
    labels = torch.tensor(
        [candidate_ids],
        dtype=torch.long,
        device=input_ids.device,
    )
    token_logprobs = functional.log_softmax(
        prediction_logits.float(),
        dim=-1,
    ).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    return float(token_logprobs.mean().cpu())


def combine(
    config: ConfidenceRouteConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    anchor = _load_rows(
        Path(config.output_dir) / "generation/anchor/cases.jsonl"
    )
    consistency = _load_rows(
        Path(config.output_dir) / "generation/consistency/cases.jsonl"
    )
    anchor_scores = _load_rows(
        Path(config.output_dir) / "scores/anchor/scores.jsonl"
    )
    consistency_scores = _load_rows(
        Path(config.output_dir) / "scores/consistency/scores.jsonl"
    )
    by_id = lambda rows: {row["case_id"]: row for row in rows}
    anchor_by_id = by_id(anchor)
    consistency_by_id = by_id(consistency)
    anchor_score_by_id = by_id(anchor_scores)
    consistency_score_by_id = by_id(consistency_scores)
    if not (
        set(anchor_by_id)
        == set(consistency_by_id)
        == set(anchor_score_by_id)
        == set(consistency_score_by_id)
    ):
        raise ValueError("confidence route combination case sets differ")
    result = []
    consistency_count = 0
    for case_id in sorted(anchor_by_id):
        anchor_relative = (
            consistency_score_by_id[case_id][
                "anchor_candidate_mean_logprob"
            ]
            - anchor_score_by_id[case_id]["anchor_candidate_mean_logprob"]
        )
        consistency_relative = (
            consistency_score_by_id[case_id][
                "consistency_candidate_mean_logprob"
            ]
            - anchor_score_by_id[case_id][
                "consistency_candidate_mean_logprob"
            ]
        )
        select_consistency = consistency_relative > anchor_relative
        source = (
            consistency_by_id[case_id]
            if select_consistency
            else anchor_by_id[case_id]
        )
        consistency_count += int(select_consistency)
        result.append(
            {
                **source,
                "route": "consistency" if select_consistency else "anchor",
                "anchor_relative_logprob": anchor_relative,
                "consistency_relative_logprob": consistency_relative,
                "relative_margin": consistency_relative - anchor_relative,
            }
        )
    summary = {
        "selector": config.selector,
        "tie_policy": config.tie_policy,
        "consistency_routes": consistency_count,
        "anchor_routes": len(result) - consistency_count,
        "uses_expected_answer": False,
    }
    return result, summary


def _chat_prompt(
    tokenizer: Any,
    config: ConfidenceRouteConfig,
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


def _load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


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


def comparison(
    config: ConfidenceRouteConfig,
    routed: list[dict[str, Any]],
    anchor: list[dict[str, Any]],
) -> dict[str, Any]:
    return paired_comparison(
        routed,
        anchor,
        bootstrap_samples=config.bootstrap_samples,
        bootstrap_seed=config.bootstrap_seed,
    )


def routed_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return _metrics(rows)
