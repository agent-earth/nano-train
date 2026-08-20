from __future__ import annotations

import copy
import unittest
from pathlib import Path

from nano_train.anchor_policy_replay import (
    build_dataset,
    load_config,
    validate_teacher_cache_receipt,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs/anchor_policy_replay/"
    "qwen35_anchor_policy_replay_v1.json"
)


class AnchorPolicyCacheReceiptTests(unittest.TestCase):
    def test_receipt_uses_canonical_arm_training_field(self):
        config = load_config(CONFIG)
        dataset = build_dataset(config)
        summary = {"all_probabilities_finite": True}
        receipt = {
            "schema_version": "nano_train_anchor_policy_cache_public_v1",
            "experiment_id": config.experiment_id,
            "identity": {
                "teacher_cache_sha256": "c" * 64,
                "anchor_adapter_sha256": config.anchor_adapter_sha256,
            },
            "dataset_identity": dataset["identity"],
            "summary": summary,
            "execution_boundary": {
                "arm_training_started": False,
                "dev_observed": False,
                "benchmark_accessed": False,
                "canary_accessed": False,
            },
        }
        validate_teacher_cache_receipt(
            config,
            dataset,
            receipt,
            summary,
            "c" * 64,
        )
        altered = copy.deepcopy(receipt)
        del altered["execution_boundary"]["arm_training_started"]
        altered["execution_boundary"]["training_started"] = False
        with self.assertRaisesRegex(ValueError, "receipt differs"):
            validate_teacher_cache_receipt(
                config,
                dataset,
                altered,
                summary,
                "c" * 64,
            )


if __name__ == "__main__":
    unittest.main()
