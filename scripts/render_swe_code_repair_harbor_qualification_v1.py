#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path

from nano_train.sft import sha256_file
from nano_train.swe_code_repair_qualification import load_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT / "configs/eval/swe_code_repair_harbor_qualification_v1.json"
)
PREREGISTER = (
    ROOT
    / "docs/experiments/"
    "evoloop_swe_code_repair_harbor_qualification_v1.preregister.json"
)
ARTIFACTS = (
    ROOT / "artifacts/evoloop-swe-code-repair-harbor-qualification-v1"
)
METRICS = ARTIFACTS / "metrics.json"
ROWS = ARTIFACTS / "structural_rows.json"
PUBLIC = (
    ROOT
    / "docs/results/"
    "evoloop_swe_code_repair_harbor_qualification_v1.public.json"
)
MARKDOWN = (
    ROOT / "docs/results/evoloop_swe_code_repair_harbor_qualification_v1.md"
)


def build_report() -> dict:
    config = load_config(CONFIG)
    preregister = json.loads(PREREGISTER.read_text(encoding="utf-8"))
    metrics = json.loads(METRICS.read_text(encoding="utf-8"))
    rows = json.loads(ROWS.read_text(encoding="utf-8"))
    if (
        preregister.get("schema_version")
        != "nano_train_swe_code_repair_harbor_qualification_preregister_v1"
        or preregister.get("identity", {}).get("config_sha256")
        != sha256_file(CONFIG)
        or metrics.get("schema_version")
        != "nano_train_swe_code_repair_harbor_qualification_result_v1"
        or metrics.get("experiment_id") != config.experiment_id
        or metrics.get("adapter_sha256") != config.adapter_sha256
        or metrics.get("structural_rows_sha256") != sha256_file(ROWS)
        or metrics.get("private_content_recorded") is not False
        or set(rows) != {"base", "adapter"}
        or len(rows["base"]) != 24
        or len(rows["adapter"]) != 24
    ):
        raise ValueError("SWE code-repair qualification result identity differs")
    gates = metrics["gates"]
    if (
        gates.get("minimum_adapter_only_wins") is not False
        or any(
            value is not True
            for key, value in gates.items()
            if key != "minimum_adapter_only_wins"
        )
        or metrics.get("candidate_admitted_for_fresh8") is not False
    ):
        raise ValueError("SWE code-repair qualification decision differs")
    return {
        "schema_version": (
            "nano_train_swe_code_repair_harbor_qualification_public_v1"
        ),
        "experiment_id": config.experiment_id,
        "project_name": "EvoLoop",
        "identity": {
            "evaluation_revision": (
                "9404604ab26aab605df7212cf65b994889cadae4"
            ),
            "config_sha256": sha256_file(CONFIG),
            "preregister_sha256": sha256_file(PREREGISTER),
            "metrics_sha256": sha256_file(METRICS),
            "structural_rows_sha256": sha256_file(ROWS),
            "adapter_sha256": config.adapter_sha256,
            "source_metrics_sha256": config.source_metrics_sha256,
            "source_reload_sha256": config.source_reload_sha256,
            "terminus_parser_sha256": config.terminus_parser_sha256,
            "terminus_prompt_template_sha256": (
                config.terminus_prompt_template_sha256
            ),
        },
        "selection": metrics["selection"],
        "loss": {
            "base": metrics["base_loss"],
            "adapter": metrics["adapter_loss"],
            "relative_improvement": metrics["relative_loss_improvement"],
        },
        "structure": {
            "base": metrics["base"],
            "adapter": metrics["adapter"],
            "comparison": metrics["comparison"],
        },
        "runtime": {
            "peak_allocated_gib": metrics["peak_allocated_gib"],
            "wall_seconds": metrics["wall_seconds"],
        },
        "decision": {
            "gates": gates,
            "candidate_admitted_for_fresh8": False,
            "verdict": "reject_on_preregistered_paired_gain_gate",
            "mechanism_conclusion": (
                "On 24 fresh held-out cases, the unchanged adapter reduced "
                "teacher-forced loss by 12.25% and preserved perfect 24/24 "
                "Harbor-parser validity with zero base-only losses. The base "
                "model also scored 24/24, so the pre-registered requirement "
                "for at least one adapter-only parser repair could not pass. "
                "The structural proxy has saturated and cannot establish "
                "incremental agent quality."
            ),
            "next_action": (
                "Train one separately pre-registered adapter on the exact "
                "Harbor first-turn prompt while freezing optimizer, LoRA, "
                "step count, and source release. Compare base, standard-SFT "
                "v1, and Harbor-prompt SFT v2 on a third fresh held-out set "
                "before benchmark access."
            ),
            "fresh8_allowed": False,
            "complete_500_allowed": False,
            "rl_or_opd_allowed": False,
            "posthoc_gate_change_allowed": False,
        },
        "claim_boundary": (
            "This is a local parser/loss qualification, not a SWE-bench "
            "score. It rejects admission under the frozen gate despite "
            "directional loss improvement and perfect structural validity."
        ),
    }


def render_markdown(report: dict) -> str:
    loss = report["loss"]
    structure = report["structure"]
    decision = report["decision"]
    return f"""# EvoLoop SWE Code-Repair Harbor Qualification v1

## Verdict

**REJECT UNDER THE FROZEN GATE.**

- Fresh held-out set: {report['selection']['dev_samples']} rows, zero overlap
  with the 12 rows observed in SFT smoke v1.
- Held-out loss: {loss['base']:.6f} -> {loss['adapter']:.6f}
  ({loss['relative_improvement']:+.2%} relative).
- Harbor parser validity: base
  {structure['base']['terminus_parser_valid']}/24, adapter
  {structure['adapter']['terminus_parser_valid']}/24.
- Paired actionable outcomes: adapter-only
  {structure['comparison']['adapter_only_wins']}, base-only
  {structure['comparison']['base_only_losses']}, both
  {structure['comparison']['both_actionable_or_complete']}.
- No adapter output exhausted the 768-token budget.

## What This Proves

{decision['mechanism_conclusion']}

The adapter does not regress the deployed response contract, but this proxy is
at ceiling. It cannot justify opening fresh-8 by changing the rule after
seeing the result.

## Next Experiment

{decision['next_action']}

## Boundary

{report['claim_boundary']}
"""


def main() -> None:
    report = build_report()
    PUBLIC.parent.mkdir(parents=True, exist_ok=True)
    PUBLIC.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    MARKDOWN.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
