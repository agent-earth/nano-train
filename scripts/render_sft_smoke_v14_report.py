#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path

from safetensors import safe_open
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts/packing-isolation-preservation-sft-smoke-v14"
V11_REPORT = ROOT / "docs/results/targeted_preservation_sft_smoke_v11.public.json"
METRICS = ARTIFACTS / "metrics.json"
GENERATIONS = ARTIFACTS / "generations.json"
RELOAD = ARTIFACTS / "reload_validation.json"
ADAPTER = ARTIFACTS / "adapter"
PRE_REGISTRATION_REVISION = "dfe69d9"
DATA_REVISION = "25451af"
CONFIG_SHA256 = (
    "7206a76fa6d8307e4c1a42ce753bce35"
    "8990e65bd4a77bf8881f86c5b55bd773"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def adapter_summary() -> dict:
    total = 0
    nonfinite = 0
    dtypes: Counter[str] = Counter()
    with safe_open(
        ADAPTER / "adapter_model.safetensors",
        framework="pt",
        device="cpu",
    ) as handle:
        for key in handle.keys():
            tensor = handle.get_tensor(key)
            total += 1
            dtypes[str(tensor.dtype)] += 1
            nonfinite += not bool(tensor.isfinite().all())
    return {
        "tensor_count": total,
        "nonfinite_tensors": nonfinite,
        "dtype_counts": dict(sorted(dtypes.items())),
    }


def main() -> None:
    raw = json.loads(METRICS.read_text(encoding="utf-8"))
    generations = json.loads(GENERATIONS.read_text(encoding="utf-8"))
    reload = json.loads(RELOAD.read_text(encoding="utf-8"))
    v11 = json.loads(V11_REPORT.read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER, local_files_only=True)
    losses = [float(row["loss"]) for row in raw["loss_curve"]]
    early_mean = statistics.mean(losses[:5])
    late_mean = statistics.mean(losses[-5:])
    finite = all(math.isfinite(loss) for loss in losses)
    adapter = adapter_summary()
    baseline = raw["baseline_validation"]
    post = raw["post_sft_validation"]
    prior = v11["post_sft_validation"]
    targets = {
        "capability_preservation_numeric": 10,
        "capability_preservation_choice": 5,
        "semantic_arithmetic_process": 7,
    }
    target_checks = {
        family: post["by_family"][family]["semantic_exact"] >= target
        for family, target in targets.items()
    }
    improvement_checks = {
        family: (
            post["by_family"][family]["semantic_exact"]
            > baseline["by_family"][family]["semantic_exact"]
        )
        for family in targets
    }
    deltas = {}
    for family in targets:
        before = set(
            prior["by_family"][family]["semantic_failure_sample_ids"]
        )
        after = set(
            post["by_family"][family]["semantic_failure_sample_ids"]
        )
        deltas[family] = {
            "strict_delta": (
                post["by_family"][family]["exact"]
                - prior["by_family"][family]["exact"]
            ),
            "semantic_delta": (
                post["by_family"][family]["semantic_exact"]
                - prior["by_family"][family]["semantic_exact"]
            ),
            "fixed_sample_ids": sorted(before - after),
            "regressed_sample_ids": sorted(after - before),
        }
    budget = int(raw["config"]["generation_max_new_tokens"])
    lengths = [
        len(
            tokenizer(
                str(row["output"]),
                add_special_tokens=False,
            ).input_ids
        )
        for row in generations["post_sft"]
    ]
    no_capped = not any(length >= budget for length in lengths)
    reload_passed = (
        reload["reload_success"]
        and reload["adapter_sha256"] == raw["adapter_sha256"]
        and reload["validation"] == post
    )
    aggregate_target = post["semantic_exact"] >= 24
    strict_target = post["exact"] >= 22
    passed = (
        finite
        and late_mean < early_mean
        and aggregate_target
        and strict_target
        and all(target_checks.values())
        and all(improvement_checks.values())
        and no_capped
        and reload_passed
        and adapter["nonfinite_tensors"] == 0
        and raw["hardware"]["peak_allocated_gib"] < 28
        and not (ARTIFACTS / "failure.json").exists()
    )
    if passed:
        raise SystemExit("v14 unexpectedly satisfies its frozen local gate")

    report = {
        "schema_version": "nano_train_public_sft_smoke_v14",
        "experiment_id": raw["experiment_id"],
        "pre_registration_revision": PRE_REGISTRATION_REVISION,
        "data_revision": DATA_REVISION,
        "passed": False,
        "identity": {
            "config_sha256": CONFIG_SHA256,
            "dataset_id": raw["dataset"]["dataset_id"],
            "dataset_sha256": raw["dataset"]["sha256"],
            "model_config_sha256": raw["model"]["config_sha256"],
        },
        "configuration": {
            "seed": raw["config"]["seed"],
            "dtype": raw["config"]["dtype"],
            "max_steps": raw["config"]["max_steps"],
            "max_length": raw["config"]["max_length"],
            "effective_batch_size": (
                raw["config"]["batch_size"]
                * raw["config"]["gradient_accumulation_steps"]
            ),
            "examples_seen": 128,
            "unique_examples_seen": 128,
            "packing_examples_seen": 5,
            "generation_max_new_tokens": budget,
            "learning_rate": raw["config"]["learning_rate"],
            "lora_r": raw["config"]["lora_r"],
            "lora_alpha": raw["config"]["lora_alpha"],
            "lora_targets": raw["config"]["lora_targets"],
        },
        "baseline_validation": baseline,
        "post_sft_validation": post,
        "versus_v11": {
            "aggregate_exact_delta": post["exact"] - prior["exact"],
            "aggregate_semantic_delta": (
                post["semantic_exact"] - prior["semantic_exact"]
            ),
            "family_deltas": deltas,
            "fixed_semantic_cases": sum(
                len(row["fixed_sample_ids"]) for row in deltas.values()
            ),
            "regressed_semantic_cases": sum(
                len(row["regressed_sample_ids"]) for row in deltas.values()
            ),
        },
        "mechanism": {
            "isolated_family": "packing_efficiency_effective_volume",
            "isolated_family_train_rows": 8,
            "isolated_family_exposures": 5,
            "percentage_and_schedule_present": False,
            "choice_process_and_host_exposure_unchanged": True,
            "result": "non_pareto_at_frozen_dose",
            "further_post_hoc_dose_search_allowed": False,
        },
        "evaluation_boundary": {
            "local_role": "development_gate_only",
            "sealed_canary_run": False,
            "prior_full_suite_run": False,
            "independent_holdout_run": False,
            "independent_holdout_prompts_loaded": False,
            "independent_holdout_references_loaded": False,
        },
        "optimization": {
            "steps": len(losses),
            "all_losses_finite": finite,
            "early_five_step_mean": early_mean,
            "late_five_step_mean": late_mean,
            "failure_receipt_exists": (ARTIFACTS / "failure.json").exists(),
        },
        "generation_budget_audit": {
            "generation_max_new_tokens": budget,
            "maximum_output_tokens": max(lengths),
            "outputs_at_generation_cap": sum(
                length >= budget for length in lengths
            ),
        },
        "adapter_validation": {
            **adapter,
            "reload_success": reload["reload_success"],
            "reload_validation": reload["validation"],
            "reload_peak_allocated_gib": reload["peak_allocated_gib"],
        },
        "artifacts": {
            "metrics_sha256": sha256_file(METRICS),
            "generations_sha256": sha256_file(GENERATIONS),
            "reload_validation_sha256": sha256_file(RELOAD),
            "adapter_sha256": raw["adapter_sha256"],
        },
        "decision": {
            "accepted_local_smoke": False,
            "aggregate_semantic_at_least_24": aggregate_target,
            "strict_exact_at_least_22": strict_target,
            "family_targets": target_checks,
            "every_family_improved_over_base": all(
                improvement_checks.values()
            ),
            "numerical_stability_passed": (
                finite and adapter["nonfinite_tensors"] == 0
            ),
            "adapter_reload_passed": reload_passed,
            "zero_capped_outputs": no_capped,
            "sealed_canary_allowed": False,
            "prior_full_suite_allowed": False,
            "independent_holdout_allowed": False,
            "merge_allowed": False,
            "scale_allowed": False,
            "rl_allowed": False,
            "next_action": (
                "Reject packing-family supervision at this dose. Preserve "
                "v11 and stop packing dose search; only schedule isolation "
                "remains eligible for a separately frozen ablation."
            ),
        },
    }
    markdown = f"""# Packing Isolation Preservation SFT Smoke v14 Result

V14 is stable but fails the local gate:

- aggregate exact / semantic: {post['exact']}/32 /
  {post['semantic_exact']}/32;
- numeric exact / semantic:
  {post['by_family']['capability_preservation_numeric']['exact']}/16 /
  {post['by_family']['capability_preservation_numeric']['semantic_exact']}/16;
- choice: {post['by_family']['capability_preservation_choice']['semantic_exact']}/8;
- process: {post['by_family']['semantic_arithmetic_process']['semantic_exact']}/8;
- early / late loss: {early_mean:.6f} / {late_mean:.6f};
- finite tensors: {adapter['tensor_count'] - adapter['nonfinite_tensors']}/
  {adapter['tensor_count']};
- maximum output: {max(lengths)}/{budget} tokens.

Relative to v11, five packing exposures fix one numeric semantic case, regress
one numeric case, and regress one choice case. Numeric semantic remains 12/16,
but strict falls 23/32 to 21/32 and choice falls 6/8 to 5/8.

This is not a Pareto improvement. Reject v14, stop packing-family dose search,
and preserve v11. Canary, prior full suite, and independent holdout remain
unrun; the holdout is unread.

Identity:

- pre-registration: `{PRE_REGISTRATION_REVISION}`;
- data revision: `{DATA_REVISION}`;
- config SHA256: `{CONFIG_SHA256}`;
- dataset SHA256: `{raw['dataset']['sha256']}`;
- metrics SHA256: `{report['artifacts']['metrics_sha256']}`;
- generations SHA256: `{report['artifacts']['generations_sha256']}`;
- reload SHA256: `{report['artifacts']['reload_validation_sha256']}`;
- adapter SHA256: `{report['artifacts']['adapter_sha256']}`.
"""
    output = ROOT / "docs/results"
    output.mkdir(parents=True, exist_ok=True)
    (output / "packing_isolation_preservation_sft_smoke_v14.public.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "packing_isolation_preservation_sft_smoke_v14.md").write_text(
        markdown,
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "passed": False,
                "post_exact": post["exact"],
                "post_semantic": post["semantic_exact"],
                "fixed_semantic_cases": report["versus_v11"][
                    "fixed_semantic_cases"
                ],
                "regressed_semantic_cases": report["versus_v11"][
                    "regressed_semantic_cases"
                ],
                "sealed_canary_allowed": False,
                "independent_holdout_allowed": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
