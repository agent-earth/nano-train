#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from transformers import AutoTokenizer

from nano_train.data import (
    evaluate_arithmetic,
    format_number,
    tokenize_samples,
)


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "artifacts/semantic-arithmetic-v5-budget-audit-v1"
RESULT = AUDIT / "result.json"
GENERATIONS = AUDIT / "generations.json"
SOURCE_GENERATIONS = (
    ROOT / "artifacts/semantic-arithmetic-sft-smoke-v5/generations.json"
)
DATASET = (
    ROOT.parent
    / "nano-data-pipeline/datasets/verified_semantic_arithmetic_traces_v3.json"
)
MODEL = ROOT / "../../models/Qwen3.5-4B"
PRE_REGISTRATION_REVISION = "d15f12c"
TRACE_PATTERN = re.compile(
    (
        r"CALC: (.+) = "
        r"([-+]?(?:[0-9]+(?:\.[0-9]+)?|\.[0-9]+))\n"
        r"FINAL: ([-+]?(?:[0-9]+(?:\.[0-9]+)?|\.[0-9]+))"
    )
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def classify(sample, row: dict) -> str:
    output = str(row["output"]).strip()
    match = TRACE_PATTERN.fullmatch(output)
    if row["semantic_valid"]:
        return "semantic_valid"
    if match is None:
        return "invalid_trace_grammar"
    expression, calc_result, final_result = match.groups()
    try:
        verified = format_number(evaluate_arithmetic(expression))
    except (SyntaxError, ValueError, ZeroDivisionError, OverflowError):
        return "unsafe_or_invalid_expression"
    expected = str(sample.verifier["expected_result"])
    if calc_result != final_result:
        return "calc_final_mismatch"
    if verified != calc_result:
        return "calc_execution_mismatch"
    if calc_result != expected:
        return "verified_but_wrong_result"
    return "other"


def taxonomy(samples: dict, rows: list[dict]) -> dict:
    counts: Counter[str] = Counter()
    ids: defaultdict[str, list[str]] = defaultdict(list)
    for row in rows:
        category = classify(samples[row["sample_id"]], row)
        counts[category] += 1
        ids[category].append(row["sample_id"])
    return {
        "counts": dict(sorted(counts.items())),
        "case_ids": dict(sorted(ids.items())),
    }


def transitions(
    samples: dict,
    source_rows: list[dict],
    audit_rows: list[dict],
) -> dict:
    source = {row["sample_id"]: row for row in source_rows}
    audit = {row["sample_id"]: row for row in audit_rows}
    counts: Counter[str] = Counter()
    ids: defaultdict[str, list[str]] = defaultdict(list)
    for sample_id in sorted(samples):
        before = classify(samples[sample_id], source[sample_id])
        after = classify(samples[sample_id], audit[sample_id])
        key = f"{before}->{after}"
        counts[key] += 1
        ids[key].append(sample_id)
    return {
        "counts": dict(sorted(counts.items())),
        "case_ids": dict(sorted(ids.items())),
    }


def output_budget_summary(tokenizer, rows: list[dict], budget: int) -> dict:
    lengths = [
        len(
            tokenizer(
                str(row["output"]),
                add_special_tokens=False,
            ).input_ids
        )
        for row in rows
    ]
    return {
        "maximum_output_tokens": max(lengths),
        "outputs_at_generation_cap": sum(
            length >= budget for length in lengths
        ),
        "outputs_above_target_with_eos_max": sum(
            length > 38 for length in lengths
        ),
    }


def main() -> None:
    raw = json.loads(RESULT.read_text(encoding="utf-8"))
    generations = json.loads(GENERATIONS.read_text(encoding="utf-8"))
    source = json.loads(SOURCE_GENERATIONS.read_text(encoding="utf-8"))
    dataset = json.loads(DATASET.read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    samples = {
        sample.sample_id: sample
        for sample in tokenize_samples(dataset, tokenizer, max_length=128)
        if sample.split == "validation"
    }
    base_taxonomy = taxonomy(samples, generations["base"])
    adapter_taxonomy = taxonomy(
        samples,
        generations["unchanged_v5_adapter"],
    )
    migration = transitions(
        samples,
        source["post_sft"],
        generations["unchanged_v5_adapter"],
    )
    budget = int(raw["contract"]["generation_max_new_tokens"])
    output_budget = {
        "base": output_budget_summary(
            tokenizer,
            generations["base"],
            budget,
        ),
        "unchanged_v5_adapter": output_budget_summary(
            tokenizer,
            generations["unchanged_v5_adapter"],
            budget,
        ),
    }
    report = {
        "schema_version": "nano_train_public_generation_budget_audit_v1",
        "audit_id": raw["audit_id"],
        "pre_registration_revision": PRE_REGISTRATION_REVISION,
        "identity": raw["identity"],
        "config_sha256": raw["config_sha256"],
        "contract": raw["contract"],
        "source_official_result": raw["source_official_result"],
        "audit_validation": raw["audit_validation"],
        "failure_taxonomy": {
            "base": base_taxonomy,
            "unchanged_v5_adapter": adapter_taxonomy,
        },
        "official_to_audit_transitions": migration,
        "output_budget": output_budget,
        "hardware": {
            "peak_allocated_gib": raw["peak_allocated_gib"],
        },
        "artifacts": {
            "local_result_sha256": sha256_file(RESULT),
            "local_generations_sha256": sha256_file(GENERATIONS),
        },
        "decision": {
            "training_performed": raw["training_performed"],
            "adapter_modified": raw["adapter_modified"],
            "official_score_changed": raw["official_score_changed"],
            "truncation_is_material_but_not_primary": True,
            "complete_adapter_trace_grammar": (
                adapter_taxonomy["counts"].get("invalid_trace_grammar", 0)
                == 0
                and adapter_taxonomy["counts"].get(
                    "calc_final_mismatch",
                    0,
                )
                == 0
            ),
            "remaining_execution_mismatches": adapter_taxonomy[
                "counts"
            ].get("calc_execution_mismatch", 0),
            "benchmark_evaluation_allowed": False,
            "rl_allowed": False,
            "next_action": (
                "Preserve the official v5 result and audit. Build fresh "
                "non-evaluation process traces with verifier-checked "
                "intermediate arithmetic steps; stop step-count, format, "
                "and generation-budget iteration."
            ),
        },
    }
    adapter = raw["audit_validation"]["unchanged_v5_adapter"]
    base = raw["audit_validation"]["base"]
    adapter_counts = adapter_taxonomy["counts"]
    migration_counts = migration["counts"]
    adapter_budget = output_budget["unchanged_v5_adapter"]
    markdown = f"""# Semantic Arithmetic v5 Budget Audit v1 Result

## Result

At 48 generated tokens:

- base strict exact / semantic valid:
  {base['exact']}/32 / {base['semantic_exact']}/32;
- unchanged v5 adapter strict exact / semantic valid:
  {adapter['exact']}/32 / {adapter['semantic_exact']}/32;
- official v5 strict exact / semantic valid remains:
  {raw['source_official_result']['post_sft_validation']['exact']}/32 /
  {raw['source_official_result']['post_sft_validation']['semantic_exact']}/32;
- peak audit memory: {raw['peak_allocated_gib']:.2f} GiB.

The audit performs no training and does not modify the adapter or official
v5 score.

## Contract

- generation budget: {budget};
- maximum validation target content:
  {raw['contract']['target_content_token_max']} tokens;
- maximum target plus EOS:
  {raw['contract']['target_with_eos_token_max']} tokens;
- unchanged adapter maximum output:
  {adapter_budget['maximum_output_tokens']} tokens;
- unchanged adapter outputs at cap:
  {adapter_budget['outputs_at_generation_cap']}/32.

The unchanged adapter has no capped output under the audit budget.

## Failure Migration

Official v5 to 48-token audit:

- CALC/FINAL mismatch to semantic valid:
  {migration_counts.get('calc_final_mismatch->semantic_valid', 0)};
- CALC/FINAL mismatch to execution mismatch:
  {migration_counts.get('calc_final_mismatch->calc_execution_mismatch', 0)};
- invalid grammar to execution mismatch:
  {migration_counts.get('invalid_trace_grammar->calc_execution_mismatch', 0)};
- execution mismatch unchanged:
  {migration_counts.get('calc_execution_mismatch->calc_execution_mismatch', 0)};
- semantic valid unchanged:
  {migration_counts.get('semantic_valid->semantic_valid', 0)}.

The final 48-token adapter taxonomy is:

- semantic valid: {adapter_counts.get('semantic_valid', 0)};
- arithmetic execution mismatch:
  {adapter_counts.get('calc_execution_mismatch', 0)};
- CALC/FINAL mismatch:
  {adapter_counts.get('calc_final_mismatch', 0)};
- invalid trace grammar:
  {adapter_counts.get('invalid_trace_grammar', 0)}.

Longer generation restores two correct cases and converts the other truncated
outputs into complete but arithmetically wrong traces. Truncation is material
but not the primary bottleneck.

## Decision

Keep official v5 at 12/32. The audit is descriptive and does not pass the
training gate. Benchmark evaluation, merge, scale-up, and RL remain forbidden.

Stop increasing optimizer steps, format supervision, and output budget. The
next data intervention must use fresh non-evaluation process traces with
verifier-checked intermediate operations so arithmetic execution, rather than
only expression copying and final agreement, receives supervision.

## Reproduction Identity

- pre-registration revision: `{PRE_REGISTRATION_REVISION}`;
- audit config SHA256: `{raw['config_sha256']}`;
- source adapter tree SHA256:
  `{raw['identity']['adapter_tree_sha256']}`;
- source metrics SHA256: `{raw['identity']['source_metrics_sha256']}`;
- source generations SHA256:
  `{raw['identity']['source_generations_sha256']}`;
- local audit result SHA256:
  `{report['artifacts']['local_result_sha256']}`;
- local audit generations SHA256:
  `{report['artifacts']['local_generations_sha256']}`.
"""
    output = ROOT / "docs/results"
    output.mkdir(parents=True, exist_ok=True)
    (output / "semantic_arithmetic_v5_budget_audit_v1.public.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "semantic_arithmetic_v5_budget_audit_v1.md").write_text(
        markdown,
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "audit_id": raw["audit_id"],
                "official_semantic": raw["source_official_result"][
                    "post_sft_validation"
                ]["semantic_exact"],
                "audit_semantic": adapter["semantic_exact"],
                "remaining_execution_mismatches": adapter_counts.get(
                    "calc_execution_mismatch",
                    0,
                ),
                "adapter_outputs_at_cap": adapter_budget[
                    "outputs_at_generation_cap"
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
