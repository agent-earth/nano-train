from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
import time
from collections import Counter
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoTokenizer, Qwen3_5ForCausalLM

from nano_train.data import TokenizedSample, tokenize_samples
from nano_train.sft import set_seed, sha256_file, sha256_tree
from nano_train.swe_code_repair_sft import (
    build_selection_contract,
    diagnose_json_action_output,
    load_config as load_sft_config,
    mean_teacher_forced_loss,
    token_band,
)


CONFIG_SCHEMA = "nano_train_swe_code_repair_harbor_qualification_v1"
PREREGISTER_SCHEMA = (
    "nano_train_swe_code_repair_harbor_qualification_preregister_v1"
)


@dataclass(frozen=True)
class SWECodeRepairQualificationConfig:
    schema_version: str
    experiment_id: str
    source_sft_config_path: str
    source_sft_config_sha256: str
    source_metrics_path: str
    source_metrics_sha256: str
    source_reload_path: str
    source_reload_sha256: str
    adapter_path: str
    adapter_sha256: str
    prior_dev_ids_sha256: str
    terminus_parser_path: str
    terminus_parser_sha256: str
    terminus_prompt_template_path: str
    terminus_prompt_template_sha256: str
    output_dir: str
    seed: int
    selection_seed: str
    dev_rows_by_band: dict[str, int]
    max_input_tokens: int
    max_output_tokens: int
    minimum_relative_loss_improvement: float
    minimum_adapter_parser_valid: int
    minimum_adapter_only_wins: int
    maximum_base_only_losses: int


def load_config(path: str | Path) -> SWECodeRepairQualificationConfig:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    expected_fields = set(SWECodeRepairQualificationConfig.__dataclass_fields__)
    if set(raw) != expected_fields:
        raise ValueError("SWE code-repair qualification config fields differ")
    config = SWECodeRepairQualificationConfig(**raw)
    validate_config(config)
    return config


def validate_config(config: SWECodeRepairQualificationConfig) -> None:
    expected: dict[str, Any] = {
        "schema_version": CONFIG_SCHEMA,
        "experiment_id": "evoloop-swe-code-repair-harbor-qualification-v1",
        "source_sft_config_path": "configs/sft/swe_code_repair_smoke_v1.json",
        "source_metrics_path": (
            "artifacts/evoloop-swe-code-repair-sft-smoke-v1/metrics.json"
        ),
        "source_reload_path": (
            "artifacts/evoloop-swe-code-repair-sft-smoke-v1/"
            "reload_validation.json"
        ),
        "adapter_path": (
            "artifacts/evoloop-swe-code-repair-sft-smoke-v1/adapter"
        ),
        "prior_dev_ids_sha256": (
            "ffaba4db8bef0006410141fbfd75f4f4c31e0eaf41438741e8c085e6a455969e"
        ),
        "terminus_parser_path": (
            "../harbor-swebench-v3-traex-04/src/harbor/agents/"
            "terminus_2/terminus_json_plain_parser.py"
        ),
        "terminus_prompt_template_path": (
            "../harbor-swebench-v3-traex-04/src/harbor/agents/"
            "terminus_2/templates/terminus-json-plain.txt"
        ),
        "output_dir": (
            "artifacts/evoloop-swe-code-repair-harbor-qualification-v1"
        ),
        "seed": 20260911,
        "selection_seed": (
            "evoloop-swe-code-repair-harbor-qualification-v1:20260911"
        ),
        "dev_rows_by_band": {"short": 8, "medium": 8, "long": 8},
        "max_input_tokens": 7168,
        "max_output_tokens": 768,
        "minimum_relative_loss_improvement": 0.05,
        "minimum_adapter_parser_valid": 23,
        "minimum_adapter_only_wins": 1,
        "maximum_base_only_losses": 0,
    }
    for field, expected_value in expected.items():
        if getattr(config, field) != expected_value:
            raise ValueError(
                f"SWE code-repair qualification freezes "
                f"{field}={expected_value}"
            )
    for field in (
        "source_sft_config_sha256",
        "source_metrics_sha256",
        "source_reload_sha256",
        "adapter_sha256",
        "prior_dev_ids_sha256",
        "terminus_parser_sha256",
        "terminus_prompt_template_sha256",
    ):
        value = getattr(config, field)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"SWE code-repair qualification {field} is not SHA256")
    if sum(config.dev_rows_by_band.values()) != 24:
        raise ValueError("SWE code-repair qualification dev count differs")


def _rank(seed: str, sample_id: str) -> str:
    return hashlib.sha256(f"{seed}:{sample_id}".encode()).hexdigest()


def _sha256_lines(values: list[str]) -> str:
    return hashlib.sha256("\n".join(values).encode()).hexdigest()


def load_terminus_parser(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(
        "evoloop_qualification_terminus_json_parser",
        path,
    )
    if spec is None or spec.loader is None:
        raise ValueError("cannot load Terminus JSON parser")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.TerminusJSONPlainParser()


def classify_terminus_error(error: str) -> str:
    if not error:
        return "none"
    if error.startswith("No valid JSON found"):
        return "no_valid_json"
    if error.startswith("Invalid JSON"):
        return "invalid_json"
    if error.startswith("Missing required fields"):
        return "missing_required_fields"
    if error.startswith("Response must be a JSON object"):
        return "not_json_object"
    if error.startswith("Field 'commands' must be an array"):
        return "commands_not_array"
    if error.startswith("Command "):
        return "invalid_command"
    return "other"


def build_qualification_contract(
    config: SWECodeRepairQualificationConfig,
) -> dict[str, Any]:
    validate_config(config)
    identity_paths = {
        "source_sft_config": (
            Path(config.source_sft_config_path),
            config.source_sft_config_sha256,
        ),
        "source_metrics": (
            Path(config.source_metrics_path),
            config.source_metrics_sha256,
        ),
        "source_reload": (
            Path(config.source_reload_path),
            config.source_reload_sha256,
        ),
        "terminus_parser": (
            Path(config.terminus_parser_path),
            config.terminus_parser_sha256,
        ),
        "terminus_prompt_template": (
            Path(config.terminus_prompt_template_path),
            config.terminus_prompt_template_sha256,
        ),
    }
    for label, (path, expected_sha256) in identity_paths.items():
        if sha256_file(path) != expected_sha256:
            raise ValueError(f"SWE code-repair qualification {label} differs")
    if sha256_tree(Path(config.adapter_path)) != config.adapter_sha256:
        raise ValueError("SWE code-repair qualification adapter differs")

    source_config = load_sft_config(config.source_sft_config_path)
    source_selection = build_selection_contract(source_config)
    if (
        source_selection["dev_sample_ids_sha256"]
        != config.prior_dev_ids_sha256
    ):
        raise ValueError("SWE code-repair prior dev identity differs")
    source_metrics = json.loads(
        Path(config.source_metrics_path).read_text(encoding="utf-8")
    )
    source_reload = json.loads(
        Path(config.source_reload_path).read_text(encoding="utf-8")
    )
    if (
        source_metrics.get("experiment_id") != source_config.experiment_id
        or source_metrics.get("adapter_sha256") != config.adapter_sha256
        or source_reload.get("adapter_sha256") != config.adapter_sha256
        or source_reload.get("reload_success") is not True
        or source_reload.get("generations_exact") is not True
    ):
        raise ValueError("SWE code-repair source run is not reproducible")

    prior_dev_ids = set(source_selection["dev_sample_ids"])
    candidates = [
        row
        for row in source_selection["selected_dev"]
        if row["sample_id"] not in prior_dev_ids
    ]
    if candidates:
        raise ValueError("SWE code-repair source selection contract drifted")
    dataset_rows = []
    with Path(source_config.dataset_path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if (
                    row.get("split") == "dev"
                    and row.get("sample_id") not in prior_dev_ids
                ):
                    dataset_rows.append(row)

    selected = []
    for band in ("short", "medium", "long"):
        band_rows = [
            row
            for row in dataset_rows
            if token_band(int(row["token_count"])) == band
        ]
        band_rows.sort(
            key=lambda row: (
                _rank(config.selection_seed, row["sample_id"]),
                row["sample_id"],
            )
        )
        chosen = band_rows[: config.dev_rows_by_band[band]]
        if len(chosen) != config.dev_rows_by_band[band]:
            raise ValueError(
                f"insufficient fresh SWE code-repair rows for {band}"
            )
        selected.extend(chosen)
    selected.sort(key=lambda row: row["sample_id"])
    selected_ids = [row["sample_id"] for row in selected]
    if set(selected_ids) & prior_dev_ids:
        raise ValueError("fresh SWE code-repair qualification overlaps v1 dev")
    if len(selected_ids) != len(set(selected_ids)):
        raise ValueError("fresh SWE code-repair qualification contains duplicates")
    return {
        "source_config": source_config,
        "source_metrics": source_metrics,
        "source_reload": source_reload,
        "selected_dev": selected,
        "selected_dev_ids": selected_ids,
        "selected_dev_ids_sha256": _sha256_lines(selected_ids),
    }


def qualification_dataset(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "dataset_id": "evoloop-swe-code-repair-harbor-qualification-v1",
        "samples": [
            {
                "sample_id": row["sample_id"],
                "split": "validation",
                "task_family": token_band(int(row["token_count"])),
                "format_family": "swe_code_repair_action",
                "messages": row["messages"],
                "verifier": row["verifier"],
                "task_spec": row["task_spec"],
            }
            for row in rows
        ],
    }


def build_harbor_prompts(
    tokenizer: Any,
    rows: list[dict[str, Any]],
    prompt_template: str,
) -> list[dict[str, Any]]:
    prompts = []
    for row in rows:
        messages = row["messages"]
        if [message["role"] for message in messages] != [
            "system",
            "user",
            "assistant",
        ]:
            raise ValueError("SWE code-repair qualification message roles differ")
        rendered = prompt_template.format(
            instruction=messages[1]["content"],
            terminal_state="",
        )
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": rendered}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prompts.append(
            {
                "sample_id": row["sample_id"],
                "task_family": token_band(int(row["token_count"])),
                "prompt_ids": tokenizer(
                    prompt,
                    add_special_tokens=False,
                ).input_ids,
            }
        )
    return prompts


@torch.inference_mode()
def generate_arm(
    model: PeftModel,
    tokenizer: Any,
    prompts: list[dict[str, Any]],
    *,
    arm: str,
    max_new_tokens: int,
    terminus_parser: Any,
) -> list[dict[str, Any]]:
    model.eval()
    rows = []
    adapter_context = (
        model.disable_adapter() if arm == "base" else nullcontext()
    )
    with adapter_context:
        for prompt_row in prompts:
            prompt = torch.tensor(
                [prompt_row["prompt_ids"]],
                dtype=torch.long,
                device="cuda",
            )
            generated = model.generate(
                input_ids=prompt,
                attention_mask=torch.ones_like(prompt),
                do_sample=False,
                max_new_tokens=max_new_tokens,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
            )
            generated_ids = generated[0, prompt.shape[1] :]
            output = tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
            ).strip()
            parsed = terminus_parser.parse_response(output)
            rows.append(
                {
                    "sample_id_sha256": hashlib.sha256(
                        prompt_row["sample_id"].encode()
                    ).hexdigest(),
                    "task_family": prompt_row["task_family"],
                    "prompt_tokens": len(prompt_row["prompt_ids"]),
                    "output_sha256": hashlib.sha256(output.encode()).hexdigest(),
                    "generation_tokens": int(generated_ids.numel()),
                    "generation_budget_exhausted": (
                        generated_ids.numel() >= max_new_tokens
                        and int(generated_ids[-1]) != tokenizer.eos_token_id
                    ),
                    "terminus_parser_valid": parsed.error == "",
                    "terminus_command_count": len(parsed.commands),
                    "terminus_task_complete": bool(parsed.is_task_complete),
                    "terminus_actionable_or_complete": (
                        parsed.error == ""
                        and (
                            bool(parsed.commands)
                            or bool(parsed.is_task_complete)
                        )
                    ),
                    "terminus_error_class": classify_terminus_error(
                        parsed.error
                    ),
                    **diagnose_json_action_output(output),
                }
            )
    return rows


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "samples": len(rows),
        "terminus_parser_valid": sum(
            row["terminus_parser_valid"] for row in rows
        ),
        "terminus_actionable_or_complete": sum(
            row["terminus_actionable_or_complete"] for row in rows
        ),
        "terminus_commands": sum(row["terminus_command_count"] for row in rows),
        "terminus_error_class": dict(
            Counter(row["terminus_error_class"] for row in rows)
        ),
        "prefix_class": dict(Counter(row["prefix_class"] for row in rows)),
        "generation_budget_exhausted": sum(
            row["generation_budget_exhausted"] for row in rows
        ),
        "by_band": {
            band: {
                "samples": sum(row["task_family"] == band for row in rows),
                "terminus_parser_valid": sum(
                    row["task_family"] == band
                    and row["terminus_parser_valid"]
                    for row in rows
                ),
                "terminus_actionable_or_complete": sum(
                    row["task_family"] == band
                    and row["terminus_actionable_or_complete"]
                    for row in rows
                ),
            }
            for band in ("short", "medium", "long")
        },
    }


def admission_gates(
    *,
    base_loss: float,
    adapter_loss: float,
    base_rows: list[dict[str, Any]],
    adapter_rows: list[dict[str, Any]],
    config: SWECodeRepairQualificationConfig,
) -> dict[str, bool]:
    base_by_id = {row["sample_id_sha256"]: row for row in base_rows}
    adapter_by_id = {row["sample_id_sha256"]: row for row in adapter_rows}
    if set(base_by_id) != set(adapter_by_id) or len(base_by_id) != 24:
        raise ValueError("SWE code-repair qualification case sets differ")
    adapter_only = sum(
        not base_by_id[case_id]["terminus_actionable_or_complete"]
        and adapter_by_id[case_id]["terminus_actionable_or_complete"]
        for case_id in base_by_id
    )
    base_only = sum(
        base_by_id[case_id]["terminus_actionable_or_complete"]
        and not adapter_by_id[case_id]["terminus_actionable_or_complete"]
        for case_id in base_by_id
    )
    per_band_non_regression = all(
        sum(
            row["task_family"] == band
            and row["terminus_actionable_or_complete"]
            for row in adapter_rows
        )
        >= sum(
            row["task_family"] == band
            and row["terminus_actionable_or_complete"]
            for row in base_rows
        )
        for band in ("short", "medium", "long")
    )
    relative_improvement = (base_loss - adapter_loss) / base_loss
    return {
        "finite_losses": math.isfinite(base_loss) and math.isfinite(adapter_loss),
        "minimum_relative_loss_improvement": (
            relative_improvement
            >= config.minimum_relative_loss_improvement
        ),
        "minimum_adapter_parser_valid": (
            sum(row["terminus_parser_valid"] for row in adapter_rows)
            >= config.minimum_adapter_parser_valid
        ),
        "minimum_adapter_only_wins": (
            adapter_only >= config.minimum_adapter_only_wins
        ),
        "maximum_base_only_losses": (
            base_only <= config.maximum_base_only_losses
        ),
        "per_band_non_regression": per_band_non_regression,
        "adapter_generation_budget_not_exhausted": not any(
            row["generation_budget_exhausted"] for row in adapter_rows
        ),
        "source_reload_exact": True,
        "private_content_absent": True,
    }


def run(config: SWECodeRepairQualificationConfig) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("SWE code-repair qualification requires CUDA")
    contract = build_qualification_contract(config)
    source_config = contract["source_config"]
    adapter_dir = Path(config.adapter_path)
    tokenizer = AutoTokenizer.from_pretrained(
        adapter_dir,
        local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenized = tokenize_samples(
        qualification_dataset(contract["selected_dev"]),
        tokenizer,
        max_length=source_config.max_length,
    )
    source_by_id = {sample.sample_id: sample for sample in tokenized}
    loss_samples: list[TokenizedSample] = [
        source_by_id[sample_id] for sample_id in contract["selected_dev_ids"]
    ]
    prompt_template = Path(config.terminus_prompt_template_path).read_text(
        encoding="utf-8"
    )
    prompts = build_harbor_prompts(
        tokenizer,
        contract["selected_dev"],
        prompt_template,
    )
    if max(len(row["prompt_ids"]) for row in prompts) > config.max_input_tokens:
        raise ValueError("SWE code-repair qualification prompt exceeds budget")

    set_seed(config.seed)
    started = time.time()
    base_model = Qwen3_5ForCausalLM.from_pretrained(
        source_config.model_path,
        local_files_only=True,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
    ).cuda()
    model = PeftModel.from_pretrained(
        base_model,
        adapter_dir,
        is_trainable=False,
    ).cuda()
    with model.disable_adapter():
        base_loss = mean_teacher_forced_loss(
            model,
            loss_samples,
            device=torch.device("cuda"),
            pad_token_id=tokenizer.pad_token_id,
        )
    adapter_loss = mean_teacher_forced_loss(
        model,
        loss_samples,
        device=torch.device("cuda"),
        pad_token_id=tokenizer.pad_token_id,
    )
    terminus_parser = load_terminus_parser(Path(config.terminus_parser_path))
    base_rows = generate_arm(
        model,
        tokenizer,
        prompts,
        arm="base",
        max_new_tokens=config.max_output_tokens,
        terminus_parser=terminus_parser,
    )
    adapter_rows = generate_arm(
        model,
        tokenizer,
        prompts,
        arm="adapter",
        max_new_tokens=config.max_output_tokens,
        terminus_parser=terminus_parser,
    )
    gates = admission_gates(
        base_loss=base_loss,
        adapter_loss=adapter_loss,
        base_rows=base_rows,
        adapter_rows=adapter_rows,
        config=config,
    )
    base_by_id = {row["sample_id_sha256"]: row for row in base_rows}
    adapter_by_id = {row["sample_id_sha256"]: row for row in adapter_rows}
    comparison = {
        "adapter_only_wins": sum(
            not base_by_id[case_id]["terminus_actionable_or_complete"]
            and adapter_by_id[case_id]["terminus_actionable_or_complete"]
            for case_id in base_by_id
        ),
        "base_only_losses": sum(
            base_by_id[case_id]["terminus_actionable_or_complete"]
            and not adapter_by_id[case_id]["terminus_actionable_or_complete"]
            for case_id in base_by_id
        ),
        "both_actionable_or_complete": sum(
            base_by_id[case_id]["terminus_actionable_or_complete"]
            and adapter_by_id[case_id]["terminus_actionable_or_complete"]
            for case_id in base_by_id
        ),
        "neither_actionable_or_complete": sum(
            not base_by_id[case_id]["terminus_actionable_or_complete"]
            and not adapter_by_id[case_id]["terminus_actionable_or_complete"]
            for case_id in base_by_id
        ),
    }
    output_root = Path(config.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    rows_path = output_root / "structural_rows.json"
    rows_path.write_text(
        json.dumps(
            {"base": base_rows, "adapter": adapter_rows},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    relative_loss_improvement = (base_loss - adapter_loss) / base_loss
    result = {
        "schema_version": "nano_train_swe_code_repair_harbor_qualification_result_v1",
        "experiment_id": config.experiment_id,
        "config": config.__dict__,
        "selection": {
            "dev_samples": len(contract["selected_dev_ids"]),
            "dev_by_band": dict(
                Counter(
                    token_band(int(row["token_count"]))
                    for row in contract["selected_dev"]
                )
            ),
            "dev_sample_ids_sha256": contract["selected_dev_ids_sha256"],
            "prior_v1_dev_sample_ids_sha256": config.prior_dev_ids_sha256,
            "overlap_with_prior_v1_dev": 0,
        },
        "base_loss": base_loss,
        "adapter_loss": adapter_loss,
        "relative_loss_improvement": relative_loss_improvement,
        "base": summarize_rows(base_rows),
        "adapter": summarize_rows(adapter_rows),
        "comparison": comparison,
        "gates": gates,
        "candidate_admitted_for_fresh8": all(gates.values()),
        "structural_rows_sha256": sha256_file(rows_path),
        "adapter_sha256": config.adapter_sha256,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "wall_seconds": time.time() - started,
        "private_content_recorded": False,
    }
    (output_root / "metrics.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result
