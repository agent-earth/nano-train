#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nano_train.config import load_sft_smoke_config
from nano_train.data import (
    execution_target_output_valid,
    load_execution_target_dataset,
)
from nano_train.sft import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--generations", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    config = load_sft_smoke_config(args.config)
    if config.dataset_schema != "execution_target_json_v1":
        raise SystemExit("rescorer requires execution_target_json_v1")
    dataset = load_execution_target_dataset(
        config.dataset_path,
        config.release_manifest_path or "",
    )
    raw_dataset = json.loads(
        Path(config.dataset_path).read_text(encoding="utf-8")
    )
    raw_validation = {
        row["sample_id"]: row
        for row in raw_dataset["samples"]
        if row["split"] == "dev"
    }
    validation_ids = {
        row["sample_id"]
        for row in dataset["samples"]
        if row["split"] == "validation"
    }
    if set(raw_validation) != validation_ids:
        raise ValueError("raw and loaded validation IDs differ")

    generations_path = Path(args.generations)
    generations = json.loads(generations_path.read_text(encoding="utf-8"))
    result = {
        "schema_version": "nano_train_execution_target_rescore_v1",
        "experiment_id": config.experiment_id,
        "config_sha256": sha256_file(Path(args.config)),
        "generations_sha256": sha256_file(generations_path),
        "dataset_file_sha256": dataset["release"]["dataset_file_sha256"],
        "dataset_canonical_sha256": dataset["release"][
            "dataset_canonical_sha256"
        ],
        "release_manifest_sha256": dataset["release"]["sha256"],
        "scorer": "execution_target_family_and_pair_verifier_v1",
    }
    result.update(
        score_execution_target_generations(
            generations=generations,
            raw_validation=raw_validation,
        )
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


def score_execution_target_generations(
    *,
    generations: dict,
    raw_validation: dict[str, dict],
) -> dict:
    validation_ids = set(raw_validation)
    result = {"arms": {}}
    for arm in ("baseline", "post_sft"):
        rows = generations.get(arm)
        if not isinstance(rows, list):
            raise ValueError(f"generations lacks {arm}")
        if {row["sample_id"] for row in rows} != validation_ids:
            raise ValueError(f"{arm} case IDs do not match frozen validation")
        baseline_rows = {
            row["sample_id"]: row for row in generations["baseline"]
        }
        by_family = {}
        by_view = {}
        by_pair: dict[str, dict[str, bool]] = {}
        verified = 0
        exact = 0
        changed_ids = []
        for row in rows:
            source = raw_validation[row["sample_id"]]
            passed = execution_target_output_valid(
                source["task_family"],
                source["view"],
                source.get("task_spec"),
                source.get("verifier"),
                row["output"],
            )
            family = source["task_family"]
            family_summary = by_family.setdefault(
                family,
                {"samples": 0, "exact": 0, "verified": 0},
            )
            family_summary["samples"] += 1
            family_summary["exact"] += int(row["exact"])
            family_summary["verified"] += int(passed)
            view = source["view"]
            view_summary = by_view.setdefault(
                view,
                {"samples": 0, "exact": 0, "verified": 0},
            )
            view_summary["samples"] += 1
            view_summary["exact"] += int(row["exact"])
            view_summary["verified"] += int(passed)
            if source.get("pair_id"):
                by_pair.setdefault(source["pair_id"], {})[view] = passed
            exact += int(row["exact"])
            verified += int(passed)
            if (
                arm == "post_sft"
                and baseline_rows[row["sample_id"]]["output"] != row["output"]
            ):
                changed_ids.append(row["sample_id"])
        pair_both_verified = sum(
            values == {"process": True, "final": True}
            for values in by_pair.values()
        )
        pair_process_only = sum(
            values.get("process") is True and values.get("final") is False
            for values in by_pair.values()
        )
        pair_final_only = sum(
            values.get("process") is False and values.get("final") is True
            for values in by_pair.values()
        )
        result["arms"][arm] = {
            "samples": len(rows),
            "exact": exact,
            "verified": verified,
            "by_family": dict(sorted(by_family.items())),
            "by_view": dict(sorted(by_view.items())),
            "pair_summary": {
                "pairs": len(by_pair),
                "both_verified": pair_both_verified,
                "process_only_verified": pair_process_only,
                "final_only_verified": pair_final_only,
                "neither_verified": len(by_pair)
                - pair_both_verified
                - pair_process_only
                - pair_final_only,
            },
            "changed_output_count": (
                len(changed_ids) if arm == "post_sft" else 0
            ),
            "changed_output_ids": (
                sorted(changed_ids) if arm == "post_sft" else []
            ),
        }
    result["verified_delta"] = (
        result["arms"]["post_sft"]["verified"]
        - result["arms"]["baseline"]["verified"]
    )
    result["pair_both_verified_delta"] = (
        result["arms"]["post_sft"]["pair_summary"]["both_verified"]
        - result["arms"]["baseline"]["pair_summary"]["both_verified"]
    )
    return result


if __name__ == "__main__":
    main()
