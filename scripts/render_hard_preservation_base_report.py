#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / "artifacts/hard-preservation-base-validation-v1"
V2 = ROOT / "artifacts/hard-preservation-base-validation-v2"
V1_RESULT = V1 / "result.json"
V2_RESULT = V2 / "result.json"
V2_GENERATIONS = V2 / "generations.json"
DATASET = (
    ROOT.parent
    / "nano-data-pipeline/datasets/hard_preservation_mix_v5.json"
)
PRE_REGISTRATION_REVISION = "a58b2be"
DATA_REVISION = "204b053"
FINAL_NUMERIC = re.compile(
    r"FINAL\s*:?\s*\n?\s*([-+]?(?:\d[\d,]*\.?\d*|\.\d+))",
    re.IGNORECASE,
)
FINAL_CHOICE = re.compile(r"FINAL\s*:?\s*([A-D])", re.IGNORECASE)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def loose_final_diagnostic(dataset: dict, rows: list[dict]) -> dict:
    by_id = {row["sample_id"]: row for row in dataset["samples"]}
    result = {}
    for family in sorted(
        {by_id[row["sample_id"]]["task_family"] for row in rows}
    ):
        subset = [
            row
            for row in rows
            if by_id[row["sample_id"]]["task_family"] == family
        ]
        correct = []
        wrong = []
        missing = []
        pattern = (
            FINAL_CHOICE
            if family == "capability_preservation_choice"
            else FINAL_NUMERIC
        )
        for row in subset:
            target = by_id[row["sample_id"]]["messages"][-1]["content"]
            output_match = pattern.search(str(row["output"]))
            target_match = pattern.search(target)
            if output_match is None or target_match is None:
                missing.append(row["sample_id"])
                continue
            output_value = output_match.group(1).replace(",", "").upper()
            target_value = target_match.group(1).replace(",", "").upper()
            if output_value == target_value:
                correct.append(row["sample_id"])
            else:
                wrong.append(row["sample_id"])
        result[family] = {
            "samples": len(subset),
            "official_semantic": sum(
                row["semantic_valid"] for row in subset
            ),
            "loose_final_correct": len(correct),
            "loose_final_wrong": len(wrong),
            "missing_final": len(missing),
            "case_ids": {
                "loose_final_correct": sorted(correct),
                "loose_final_wrong": sorted(wrong),
                "missing_final": sorted(missing),
            },
        }
    return result


def main() -> None:
    v1 = json.loads(V1_RESULT.read_text(encoding="utf-8"))
    v2 = json.loads(V2_RESULT.read_text(encoding="utf-8"))
    rows = json.loads(V2_GENERATIONS.read_text(encoding="utf-8"))
    dataset = json.loads(DATASET.read_text(encoding="utf-8"))
    if v1["identity"] != v2["identity"]:
        raise SystemExit("base audit identities differ")
    if v1["validation"] != v2["validation"]:
        raise SystemExit("80- and 128-token official metrics differ")
    diagnostic = loose_final_diagnostic(dataset, rows)
    report = {
        "schema_version": "nano_train_public_preservation_base_audit_v2",
        "audit_id": v2["audit_id"],
        "pre_registration_revision": PRE_REGISTRATION_REVISION,
        "data_revision": DATA_REVISION,
        "identity": v2["identity"],
        "config_sha256": v2["config_sha256"],
        "validation": v2["validation"],
        "by_family": v2["by_family"],
        "contract": v2["contract"],
        "insufficient_budget_v1": {
            "config_sha256": v1["config_sha256"],
            "generation_max_new_tokens": v1["contract"][
                "generation_max_new_tokens"
            ],
            "outputs_at_generation_cap": v1["contract"][
                "outputs_at_generation_cap"
            ],
            "official_metrics_changed_in_v2": False,
            "result_sha256": sha256_file(V1_RESULT),
        },
        "loose_final_diagnostic": diagnostic,
        "hardware": v2["hardware"],
        "artifacts": {
            "local_result_sha256": sha256_file(V2_RESULT),
            "local_generations_sha256": sha256_file(V2_GENERATIONS),
        },
        "decision": {
            "training_performed": False,
            "valid_pretraining_baseline": True,
            "zero_capped_outputs": (
                v2["contract"]["outputs_at_generation_cap"] == 0
            ),
            "material_learning_headroom": (
                v2["validation"]["semantic_exact"] < 24
            ),
            "training_automatically_authorized": False,
            "next_action": (
                "Pre-register a conservative mixed-preservation SFT smoke. "
                "Require family-level validation improvement and then the "
                "sealed 40-case canary before any full benchmark."
            ),
        },
    }
    numeric = diagnostic["capability_preservation_numeric"]
    choice = diagnostic["capability_preservation_choice"]
    process = diagnostic["semantic_arithmetic_process"]
    markdown = f"""# Hard Preservation Base Validation v2

## Result

At a sufficient 128-token generation budget:

- aggregate exact / semantic: {v2['validation']['exact']}/32 /
  {v2['validation']['semantic_exact']}/32;
- numeric exact / semantic:
  {v2['by_family']['capability_preservation_numeric']['exact']}/16 /
  {v2['by_family']['capability_preservation_numeric']['semantic_exact']}/16;
- choice exact / semantic:
  {v2['by_family']['capability_preservation_choice']['exact']}/8 /
  {v2['by_family']['capability_preservation_choice']['semantic_exact']}/8;
- process exact / semantic:
  {v2['by_family']['semantic_arithmetic_process']['exact']}/8 /
  {v2['by_family']['semantic_arithmetic_process']['semantic_exact']}/8;
- outputs at the 128-token cap:
  {v2['contract']['outputs_at_generation_cap']}/32.

No training is performed.

## Failure Diagnostic

The non-scoring loose-final diagnostic does not change official metrics:

- numeric: {numeric['loose_final_correct']}/16 have the correct final number,
  while {numeric['loose_final_wrong']}/16 have a wrong final number;
- choice: {choice['loose_final_correct']}/8 have the correct letter and
  {choice['loose_final_wrong']}/8 are wrong;
- process: {process['loose_final_correct']}/8 have the correct final number,
  while only {process['official_semantic']}/8 satisfy the process verifier.

The data has both output-contract and genuine capability headroom.

## Budget Audit

The earlier 80-token audit has
{v1['contract']['outputs_at_generation_cap']}/32 capped outputs. Raising only
the generation budget to 128 removes all caps and leaves official exact and
semantic metrics unchanged at 8/32. V2 is therefore the valid pre-training
baseline.

## Decision

The audit does not automatically authorize training. A separately
pre-registered conservative mixed-preservation SFT may proceed only with
family-level gates, followed by the sealed 40-case regression canary before
any full benchmark. Merge, scale-up, and RL remain forbidden.

## Reproduction Identity

- pre-registration revision: `{PRE_REGISTRATION_REVISION}`;
- data revision: `{DATA_REVISION}`;
- dataset SHA256: `{v2['identity']['dataset_sha256']}`;
- model config SHA256: `{v2['identity']['model_config_sha256']}`;
- v2 config SHA256: `{v2['config_sha256']}`;
- local result SHA256: `{report['artifacts']['local_result_sha256']}`;
- local generations SHA256:
  `{report['artifacts']['local_generations_sha256']}`.
"""
    output = ROOT / "docs/results"
    output.mkdir(parents=True, exist_ok=True)
    (output / "hard_preservation_base_validation_v2.public.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "hard_preservation_base_validation_v2.md").write_text(
        markdown,
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "audit_id": v2["audit_id"],
                "semantic_exact": v2["validation"]["semantic_exact"],
                "numeric_loose_final_correct": numeric[
                    "loose_final_correct"
                ],
                "numeric_loose_final_wrong": numeric["loose_final_wrong"],
                "outputs_at_cap": v2["contract"][
                    "outputs_at_generation_cap"
                ],
                "valid_pretraining_baseline": True,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
