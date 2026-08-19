#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nano_train.paired_consistency import build_selection_contract, load_config
from nano_train.sft import sha256_file
from scripts.rescore_execution_target_generations import (
    score_execution_target_generations,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--generations", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config_path = Path(args.config)
    config = load_config(config_path)
    selection = build_selection_contract(config)
    heldout_ids = set(selection["heldout_sample_ids"])
    raw_validation = {
        sample_id: selection["raw_by_id"][sample_id]
        for sample_id in selection["heldout_sample_ids"]
    }
    if set(raw_validation) != heldout_ids:
        raise ValueError("paired consistency heldout IDs differ")
    generations_path = Path(args.generations)
    generations = json.loads(generations_path.read_text(encoding="utf-8"))
    result = {
        "schema_version": "nano_train_paired_consistency_rescore_v1",
        "experiment_id": config.experiment_id,
        "config_sha256": sha256_file(config_path),
        "generations_sha256": sha256_file(generations_path),
        "dataset_file_sha256": config.dataset_file_sha256,
        "dataset_canonical_sha256": config.dataset_canonical_sha256,
        "release_manifest_sha256": config.release_manifest_sha256,
        "heldout_sample_id_sha256": selection["hashes"][
            "heldout_sample_id_sha256"
        ],
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


if __name__ == "__main__":
    main()
