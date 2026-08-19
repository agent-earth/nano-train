#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nano_train.config import load_sft_smoke_config
from nano_train.data import (
    load_skill_release_dataset,
    skill_release_output_valid,
)
from nano_train.sft import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--generations", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    config = load_sft_smoke_config(args.config)
    if config.dataset_schema != "skill_release_jsonl_v1":
        raise SystemExit("rescorer requires skill_release_jsonl_v1")
    dataset = load_skill_release_dataset(
        config.dataset_path,
        config.release_manifest_path or "",
        train_samples_per_family=config.train_samples_per_family or 0,
        validation_samples_per_family=(
            config.validation_samples_per_family or 0
        ),
    )
    validation = {
        row["sample_id"]: row
        for row in dataset["samples"]
        if row["split"] == "validation"
    }
    generations_path = Path(args.generations)
    generations = json.loads(generations_path.read_text(encoding="utf-8"))
    result = {
        "schema_version": "nano_train_skill_release_rescore_v1",
        "experiment_id": config.experiment_id,
        "config_sha256": sha256_file(Path(args.config)),
        "generations_sha256": sha256_file(generations_path),
        "dataset_sha256": dataset["release"]["accepted_jsonl_sha256"],
        "release_manifest_sha256": dataset["release"]["sha256"],
        "scorer": "skill_release_family_verifier_v1",
        "arms": {},
    }
    for arm in ("baseline", "post_sft"):
        rows = generations.get(arm)
        if not isinstance(rows, list):
            raise ValueError(f"generations lacks {arm}")
        if {row["sample_id"] for row in rows} != set(validation):
            raise ValueError(f"{arm} case IDs do not match frozen validation")
        by_family = {}
        verified = 0
        exact = 0
        changed_ids = []
        for row in rows:
            source = validation[row["sample_id"]]
            passed = skill_release_output_valid(
                source["task_family"],
                source.get("task_spec"),
                source.get("verifier"),
                row["output"],
            )
            family = source["task_family"]
            summary = by_family.setdefault(
                family,
                {"samples": 0, "exact": 0, "verified": 0},
            )
            summary["samples"] += 1
            summary["exact"] += int(row["exact"])
            summary["verified"] += int(passed)
            exact += int(row["exact"])
            verified += int(passed)
            if arm == "post_sft":
                baseline = next(
                    item
                    for item in generations["baseline"]
                    if item["sample_id"] == row["sample_id"]
                )
                if baseline["output"] != row["output"]:
                    changed_ids.append(row["sample_id"])
        result["arms"][arm] = {
            "samples": len(rows),
            "exact": exact,
            "verified": verified,
            "by_family": dict(sorted(by_family.items())),
            "changed_output_count": len(changed_ids) if arm == "post_sft" else 0,
            "changed_output_ids": sorted(changed_ids) if arm == "post_sft" else [],
        }
    result["verified_delta"] = (
        result["arms"]["post_sft"]["verified"]
        - result["arms"]["baseline"]["verified"]
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
