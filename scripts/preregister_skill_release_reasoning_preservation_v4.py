#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from nano_train.config import load_sft_smoke_config
from nano_train.data import load_skill_release_dataset
from nano_train.sft import _scheduled_batch_order, sha256_file


ROOT = Path(__file__).resolve().parents[1]
CONTROL_CONFIG = ROOT / "configs/sft/skill_release_bounded_dose_v2.json"
TREATMENT_CONFIG = (
    ROOT / "configs/sft/skill_release_reasoning_preservation_v4.json"
)
OUTPUT = (
    ROOT
    / "docs/experiments/skill_release_reasoning_preservation_sft_v4.preregister.json"
)


def _load_rows(path: Path) -> dict[str, dict]:
    rows = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                rows[row["sample_id"]] = row
    return rows


def _selected(config_path: Path) -> tuple[object, dict, list, list]:
    config = load_sft_smoke_config(config_path)
    dataset = load_skill_release_dataset(
        config.dataset_path,
        config.release_manifest_path or "",
        train_samples_per_family=config.train_samples_per_family or 0,
        validation_samples_per_family=(
            config.validation_samples_per_family or 0
        ),
        validation_start_per_family=config.validation_start_per_family,
    )
    train = [
        sample for sample in dataset["samples"] if sample["split"] == "train"
    ]
    validation = [
        sample
        for sample in dataset["samples"]
        if sample["split"] == "validation"
    ]
    return config, dataset, train, validation


def main() -> None:
    control, _, control_train, control_validation = _selected(CONTROL_CONFIG)
    treatment, dataset, treatment_train, treatment_validation = _selected(
        TREATMENT_CONFIG
    )
    if [row["sample_id"] for row in control_train] != [
        row["sample_id"] for row in treatment_train
    ]:
        raise ValueError("treatment changes the frozen train subset")

    raw_rows = _load_rows(Path(treatment.dataset_path))
    train_ids = {row["sample_id"] for row in treatment_train}
    control_ids = {row["sample_id"] for row in control_validation}
    treatment_ids = {row["sample_id"] for row in treatment_validation}
    if train_ids & treatment_ids:
        raise ValueError("fresh validation overlaps train IDs")
    if control_ids & treatment_ids:
        raise ValueError("fresh validation overlaps observed validation IDs")

    train_semantic = {raw_rows[sample_id]["semantic_hash"] for sample_id in train_ids}
    control_semantic = {
        raw_rows[sample_id]["semantic_hash"] for sample_id in control_ids
    }
    treatment_semantic = {
        raw_rows[sample_id]["semantic_hash"] for sample_id in treatment_ids
    }
    if train_semantic & treatment_semantic:
        raise ValueError("fresh validation overlaps train semantics")
    if control_semantic & treatment_semantic:
        raise ValueError("fresh validation overlaps observed validation semantics")

    order = _scheduled_batch_order(
        [
            _Sample(row["sample_id"], row["task_family"])
            for row in treatment_train
        ],
        treatment.seed,
        treatment.train_family_schedule,
    )
    exposure = [
        {
            "step": step,
            "sample_id": treatment_train[index]["sample_id"],
            "task_family": treatment_train[index]["task_family"],
        }
        for step, index in enumerate(order, start=1)
    ]
    report = {
        "schema_version": "nano_train_reasoning_preservation_preregister_v1",
        "experiment_id": treatment.experiment_id,
        "identity": {
            "control_config_sha256": sha256_file(CONTROL_CONFIG),
            "treatment_config_sha256": sha256_file(TREATMENT_CONFIG),
            "accepted_jsonl_sha256": dataset["release"][
                "accepted_jsonl_sha256"
            ],
            "release_manifest_sha256": dataset["release"]["sha256"],
        },
        "method": {
            "control": control.experiment_id,
            "frozen": [
                "model",
                "release",
                "train_subset",
                "qv_lora",
                "steps",
                "learning_rate",
                "seed",
                "precision",
                "sequence_length",
                "generation_budget",
                "family_verifier",
            ],
            "changed": [
                "train_family_schedule",
                "validation_start_per_family",
                "experiment_identity",
            ],
            "train_family_schedule": list(treatment.train_family_schedule),
            "train_exposure": exposure,
            "train_exposure_by_family": dict(
                sorted(Counter(row["task_family"] for row in exposure).items())
            ),
        },
        "fresh_validation": {
            "selection_rule": "release_order_per_family_indices_4_to_7",
            "samples": len(treatment_validation),
            "sample_ids": sorted(treatment_ids),
            "sample_id_sha256": hashlib.sha256(
                "\n".join(sorted(treatment_ids)).encode("utf-8")
            ).hexdigest(),
            "by_family": dict(
                sorted(
                    Counter(
                        row["task_family"] for row in treatment_validation
                    ).items()
                )
            ),
            "train_id_overlap": len(train_ids & treatment_ids),
            "observed_dev_id_overlap": len(control_ids & treatment_ids),
            "train_semantic_overlap": len(
                train_semantic & treatment_semantic
            ),
            "observed_dev_semantic_overlap": len(
                control_semantic & treatment_semantic
            ),
        },
        "decision_rule": {
            "method_accepted": (
                "finite_and_reloadable AND corrected_verified_delta > 0 "
                "AND every JSON family post_verified >= baseline_verified"
            ),
            "forbidden_after_observation": [
                "dose_search",
                "learning_rate_search",
                "seed_search",
                "schedule_search",
                "validation_offset_search",
                "prompt_search",
                "parser_search",
                "adapter_weight_search",
                "route_search",
            ],
            "benchmark_allowed": False,
            "independent_holdout_allowed": False,
            "rl_allowed": False,
        },
        "claim_boundary": (
            "This receipt freezes one local synthetic method test before any "
            "generation on the fresh validation rows. It is not benchmark or "
            "independent-holdout evidence."
        ),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


class _Sample:
    def __init__(self, sample_id: str, task_family: str) -> None:
        self.sample_id = sample_id
        self.task_family = task_family


if __name__ == "__main__":
    main()
