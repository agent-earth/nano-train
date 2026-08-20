from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from nano_train.quality_consistency import (
    benchmark_prompt_hashes,
    build_dataset,
    dataset_prompt_hashes,
    forbidden_prompt_hashes,
    load_config,
    normalized_dataset_prompt_hashes,
    public_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT / "configs/quality_consistency/qwen35_quality_consistency_v1.json"
)


class QualityConsistencyTests(unittest.TestCase):
    def test_dataset_is_balanced_disjoint_and_deterministic(self):
        config = load_config(CONFIG)
        first = build_dataset(config)
        second = build_dataset(config)
        self.assertEqual(first, second)
        self.assertEqual(len(first["train_pairs"]), 256)
        self.assertEqual(len(first["dev_pairs"]), 192)
        for key, expected in (("train_pairs", 64), ("dev_pairs", 48)):
            counts = {}
            for row in first[key]:
                counts[row["family"]] = counts.get(row["family"], 0) + 1
            self.assertEqual(set(counts.values()), {expected})
        self.assertEqual(
            {row["expression"] for row in first["train_pairs"]}
            & {row["expression"] for row in first["dev_pairs"]},
            set(),
        )

    def test_dataset_excludes_all_observed_quality_prompts(self):
        config = load_config(CONFIG)
        self.assertEqual(
            dataset_prompt_hashes(build_dataset(config))
            & forbidden_prompt_hashes(config),
            set(),
        )
        benchmark, counts = benchmark_prompt_hashes(config)
        self.assertEqual(
            counts,
            {"gsm8k": 1319, "mmlu": 14042, "gpqa_diamond": 198},
        )
        self.assertEqual(
            normalized_dataset_prompt_hashes(build_dataset(config))
            & benchmark,
            set(),
        )

    def test_public_contract_excludes_raw_content(self):
        contract = public_contract(build_dataset(load_config(CONFIG)))
        for key in ("train_pairs", "dev_pairs"):
            self.assertTrue(contract[key])
            self.assertEqual(
                set(contract[key][0]),
                {
                    "pair_id",
                    "family",
                    "process_prompt_sha256",
                    "final_prompt_sha256",
                    "process_target_sha256",
                    "final_target_sha256",
                },
            )

    def test_config_freezes_changed_mechanism(self):
        config = load_config(CONFIG)
        self.assertEqual(config.max_steps, 256)
        self.assertEqual(config.train_pairs_per_family, 64)
        self.assertEqual(config.process_ce_weight, 0.5)
        self.assertEqual(config.final_ce_weight, 0.5)
        self.assertEqual(config.consistency_weight, 1.0)
        self.assertTrue(config.teacher_detach)
        self.assertFalse(config.policy["benchmark_access_after_result"])

    def test_config_rejects_posthoc_method_change(self):
        raw = json.loads(CONFIG.read_text(encoding="utf-8"))
        for key, value, error in (
            ("max_steps", 128, "max_steps"),
            ("learning_rate", 0.0001, "learning_rate"),
            ("consistency_weight", 0.5, "consistency_weight"),
            ("teacher_detach", False, "teacher_detach"),
        ):
            with self.subTest(key=key):
                altered = copy.deepcopy(raw)
                altered[key] = value
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "config.json"
                    path.write_text(json.dumps(altered), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, error):
                        load_config(path)


if __name__ == "__main__":
    unittest.main()
