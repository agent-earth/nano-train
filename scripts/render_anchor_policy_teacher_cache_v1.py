#!/usr/bin/env python3

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from nano_train.anchor_policy_replay import (
    CACHE_RECEIPT_SCHEMA,
    build_dataset,
    inspect_teacher_cache,
    load_config,
)
from nano_train.sft import sha256_file


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs/anchor_policy_replay/"
    "qwen35_anchor_policy_replay_v1.json"
)
PREREG = (
    ROOT
    / "docs/experiments/"
    "qwen35_anchor_policy_replay_v1.preregister.json"
)


def git_revision() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()


def build_receipt() -> dict:
    config = load_config(CONFIG)
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    cache_path = Path(config.teacher_cache_path)
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    dataset = build_dataset(config)
    summary = inspect_teacher_cache(config, dataset, cache)
    if (
        prereg["schema_version"]
        != "nano_train_anchor_policy_replay_preregister_v1"
        or prereg["identity"]["config_sha256"] != sha256_file(CONFIG)
        or prereg["execution_boundary"]["teacher_cache_started"] is not False
        or cache["identity"]["anchor_adapter_sha256"]
        != config.anchor_adapter_sha256
        or cache["dataset_identity"] != dataset["identity"]
    ):
        raise ValueError("anchor policy teacher cache receipt differs")
    return {
        "schema_version": CACHE_RECEIPT_SCHEMA,
        "experiment_id": config.experiment_id,
        "identity": {
            "config_sha256": sha256_file(CONFIG),
            "preregister_sha256": sha256_file(PREREG),
            "preregister_revision": prereg["identity"]["code_revision"],
            "cache_generation_revision": git_revision(),
            "teacher_cache_sha256": sha256_file(cache_path),
            "anchor_adapter_sha256": config.anchor_adapter_sha256,
            "model_config_sha256": config.model_config_sha256,
            "model_index_sha256": config.model_index_sha256,
        },
        "dataset_identity": dataset["identity"],
        "summary": summary,
        "teacher_contract": {
            "frozen_anchor_adapter": True,
            "final_target_positions_only": True,
            "uses_training_target_prefix": True,
            "uses_evaluation_expected_answer": False,
            "uses_case_correctness": False,
            "uses_observed_quality_outputs": False,
            "uses_benchmark_rows_or_outputs": False,
            "top_k_plus_other_bucket": config.anchor_policy_top_k,
            "temperature": config.anchor_policy_temperature,
        },
        "execution_boundary": {
            "teacher_cache_completed": True,
            "arm_training_started": False,
            "model_generation_started": False,
            "dev_observed": False,
            "benchmark_accessed": False,
            "canary_accessed": False,
        },
        "claim_boundary": (
            "This receipt establishes only a finite, identity-verified frozen "
            "anchor policy cache on synthetic train rows. It is not model "
            "quality, canary, benchmark, or holdout evidence."
        ),
    }


def main() -> None:
    config = load_config(CONFIG)
    output = Path(config.teacher_cache_receipt_path)
    receipt = build_receipt()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "experiment_id": receipt["experiment_id"],
                "identity": receipt["identity"],
                "summary": receipt["summary"],
                "execution_boundary": receipt["execution_boundary"],
                "output": str(output),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
