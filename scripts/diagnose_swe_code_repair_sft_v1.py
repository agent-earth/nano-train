#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoTokenizer, Qwen3_5ForCausalLM

from nano_train.data import tokenize_samples
from nano_train.sft import set_seed, sha256_file, sha256_tree
from nano_train.swe_code_repair_sft import (
    build_selection_contract,
    diagnose_json_action_output,
    load_config,
    training_dataset,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/sft/swe_code_repair_smoke_v1.json"


def load_terminus_parser(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(
        "evoloop_terminus_json_plain_parser",
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


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "samples": len(rows),
        "parse_status": dict(Counter(row["parse_status"] for row in rows)),
        "prefix_class": dict(Counter(row["prefix_class"] for row in rows)),
        "exact_json_valid": sum(
            row["parse_status"] == "exact_valid" for row in rows
        ),
        "recoverable_json_action_substring": sum(
            row["recoverable_json_action_substring"] for row in rows
        ),
        "terminus_parser_valid": sum(
            row["terminus_parser_valid"] for row in rows
        ),
        "terminus_actionable_or_complete": sum(
            row["terminus_actionable_or_complete"] for row in rows
        ),
        "terminus_error_class": dict(
            Counter(row["terminus_error_class"] for row in rows)
        ),
        "generation_budget_exhausted": sum(
            row["generation_budget_exhausted"] for row in rows
        ),
        "generation_tokens": {
            "min": min(row["generation_tokens"] for row in rows),
            "max": max(row["generation_tokens"] for row in rows),
            "mean": (
                sum(row["generation_tokens"] for row in rows) / len(rows)
            ),
        },
    }


@torch.inference_mode()
def generate_arm(
    model: PeftModel,
    tokenizer: Any,
    samples: list[Any],
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
        for sample in samples:
            prompt = torch.tensor(
                [sample.prompt_ids],
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
            row = {
                "sample_id_sha256": hashlib.sha256(
                    sample.sample_id.encode()
                ).hexdigest(),
                "task_family": sample.task_family,
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
                    and (bool(parsed.commands) or bool(parsed.is_task_complete))
                ),
                "terminus_error_class": classify_terminus_error(parsed.error),
                **diagnose_json_action_output(output),
            }
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--terminus-parser-path", required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("SWE code-repair diagnostics require CUDA")
    terminus_parser_path = Path(args.terminus_parser_path).resolve()
    terminus_parser = load_terminus_parser(terminus_parser_path)
    config = load_config(CONFIG)
    selection = build_selection_contract(config)
    output_root = ROOT / config.output_dir
    adapter_dir = output_root / "adapter"
    source_rows = json.loads(
        (output_root / "validation_generations.json").read_text(
            encoding="utf-8"
        )
    )
    source_hashes = {
        hashlib.sha256(row["sample_id"].encode()).hexdigest(): row[
            "output_sha256"
        ]
        for row in source_rows
    }
    tokenizer = AutoTokenizer.from_pretrained(
        adapter_dir,
        local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenized = tokenize_samples(
        training_dataset(selection),
        tokenizer,
        max_length=config.max_length,
    )
    by_id = {sample.sample_id: sample for sample in tokenized}
    validation = [
        by_id[sample_id] for sample_id in selection["dev_sample_ids"]
    ]
    set_seed(config.seed)
    base_model = Qwen3_5ForCausalLM.from_pretrained(
        config.model_path,
        local_files_only=True,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
    ).cuda()
    model = PeftModel.from_pretrained(
        base_model,
        adapter_dir,
        is_trainable=False,
    ).cuda()
    base_rows = generate_arm(
        model,
        tokenizer,
        validation,
        arm="base",
        max_new_tokens=config.generation_max_new_tokens,
        terminus_parser=terminus_parser,
    )
    adapter_rows = generate_arm(
        model,
        tokenizer,
        validation,
        arm="adapter",
        max_new_tokens=config.generation_max_new_tokens,
        terminus_parser=terminus_parser,
    )
    adapter_reproduction_exact = all(
        source_hashes[row["sample_id_sha256"]] == row["output_sha256"]
        for row in adapter_rows
    )
    if not adapter_reproduction_exact:
        raise ValueError("adapter diagnostic generations differ")
    receipt = {
        "schema_version": "nano_train_swe_code_repair_sft_diagnostic_v1",
        "experiment_id": config.experiment_id,
        "identity": {
            "config_sha256": sha256_file(CONFIG),
            "adapter_sha256": sha256_tree(adapter_dir),
            "source_generation_sha256": sha256_file(
                output_root / "validation_generations.json"
            ),
            "terminus_parser_sha256": sha256_file(terminus_parser_path),
        },
        "arms": {
            "base": {"summary": summarize(base_rows), "rows": base_rows},
            "adapter": {
                "summary": summarize(adapter_rows),
                "rows": adapter_rows,
            },
        },
        "adapter_reproduction_exact": True,
        "private_content_recorded": False,
        "claim_boundary": (
            "Post-run structural diagnosis only. It records hashes, token "
            "counts, and parse categories, not prompts or model outputs."
        ),
    }
    output = output_root / "generation_diagnostics.json"
    output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "schema_version": receipt["schema_version"],
                "experiment_id": receipt["experiment_id"],
                "identity": receipt["identity"],
                "base": receipt["arms"]["base"]["summary"],
                "adapter": receipt["arms"]["adapter"]["summary"],
                "adapter_reproduction_exact": True,
                "private_content_recorded": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
