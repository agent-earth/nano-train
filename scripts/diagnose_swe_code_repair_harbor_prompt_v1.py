#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoTokenizer, Qwen3_5ForCausalLM

from nano_train.sft import set_seed, sha256_file, sha256_tree
from nano_train.swe_code_repair_sft import (
    build_selection_contract,
    diagnose_json_action_output,
    load_config,
)
from scripts.diagnose_swe_code_repair_sft_v1 import (
    classify_terminus_error,
    load_terminus_parser,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/sft/swe_code_repair_smoke_v1.json"


def harbor_prompts(
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
            raise ValueError("SWE code-repair message roles differ")
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
        prompt_ids = tokenizer(
            prompt,
            add_special_tokens=False,
        ).input_ids
        prompts.append(
            {
                "sample_id": row["sample_id"],
                "task_family": (
                    "short"
                    if int(row["token_count"]) <= 768
                    else "medium"
                    if int(row["token_count"]) <= 1_280
                    else "long"
                ),
                "prompt_ids": prompt_ids,
            }
        )
    return prompts


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
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
        "generation_tokens": {
            "min": min(row["generation_tokens"] for row in rows),
            "max": max(row["generation_tokens"] for row in rows),
            "mean": (
                sum(row["generation_tokens"] for row in rows) / len(rows)
            ),
        },
        "prompt_tokens": {
            "min": min(row["prompt_tokens"] for row in rows),
            "max": max(row["prompt_tokens"] for row in rows),
            "mean": sum(row["prompt_tokens"] for row in rows) / len(rows),
        },
    }


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
                    "output_sha256": hashlib.sha256(
                        output.encode()
                    ).hexdigest(),
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--terminus-parser-path", required=True)
    parser.add_argument("--prompt-template-path", required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Harbor prompt diagnostics require CUDA")

    parser_path = Path(args.terminus_parser_path).resolve()
    template_path = Path(args.prompt_template_path).resolve()
    terminus_parser = load_terminus_parser(parser_path)
    prompt_template = template_path.read_text(encoding="utf-8")
    config = load_config(CONFIG)
    selection = build_selection_contract(config)
    output_root = ROOT / config.output_dir
    adapter_dir = output_root / "adapter"
    tokenizer = AutoTokenizer.from_pretrained(
        adapter_dir,
        local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    prompts = harbor_prompts(
        tokenizer,
        selection["selected_dev"],
        prompt_template,
    )
    if max(len(row["prompt_ids"]) for row in prompts) > 7_168:
        raise ValueError("Harbor prompt exceeds frozen input budget")

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
        prompts,
        arm="base",
        max_new_tokens=config.generation_max_new_tokens,
        terminus_parser=terminus_parser,
    )
    adapter_rows = generate_arm(
        model,
        tokenizer,
        prompts,
        arm="adapter",
        max_new_tokens=config.generation_max_new_tokens,
        terminus_parser=terminus_parser,
    )
    receipt = {
        "schema_version": (
            "nano_train_swe_code_repair_harbor_prompt_diagnostic_v1"
        ),
        "experiment_id": config.experiment_id,
        "ablation_id": "evoloop-swe-code-repair-harbor-prompt-v1",
        "identity": {
            "config_sha256": sha256_file(CONFIG),
            "adapter_sha256": sha256_tree(adapter_dir),
            "terminus_parser_sha256": sha256_file(parser_path),
            "terminus_prompt_template_sha256": sha256_file(template_path),
        },
        "protocol": {
            "prompt_shape": (
                "one user message containing the exact Terminus2 template, "
                "public SWE-bench train issue text, and empty initial terminal "
                "state"
            ),
            "temperature": 0,
            "thinking": False,
            "max_input_tokens": 7_168,
            "max_output_tokens": config.generation_max_new_tokens,
            "case_set": "the same 12 already observed v1 local dev rows",
            "claim_scope": "diagnostic_only",
        },
        "arms": {
            "base": {"summary": summarize(base_rows), "rows": base_rows},
            "adapter": {
                "summary": summarize(adapter_rows),
                "rows": adapter_rows,
            },
        },
        "private_content_recorded": False,
        "claim_boundary": (
            "Prompt-alignment diagnosis on already observed local dev rows. "
            "It can explain the v1 proxy mismatch but cannot admit a new "
            "training candidate or support a SWE-bench claim."
        ),
    }
    output_path = output_root / "harbor_prompt_diagnostics.json"
    output_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "schema_version": receipt["schema_version"],
                "ablation_id": receipt["ablation_id"],
                "identity": receipt["identity"],
                "base": receipt["arms"]["base"]["summary"],
                "adapter": receipt["arms"]["adapter"]["summary"],
                "private_content_recorded": False,
                "claim_boundary": receipt["claim_boundary"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
