from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from nano_train.preservation_dual_view import (
    build_dataset,
    build_step_schedule,
    contamination_audit,
    load_config,
    public_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs/preservation_dual_view/"
    "qwen35_preservation_dual_view_v1.json"
)


class PreservationDualViewTests(unittest.TestCase):
    def test_dataset_is_balanced_disjoint_and_deterministic(self):
        config = load_config(CONFIG)
        first = build_dataset(config)
        second = build_dataset(config)
        self.assertEqual(first, second)
        self.assertEqual(len(first["train_pairs"]), 256)
        self.assertEqual(len(first["dev_pairs"]), 256)
        for key in ("train_pairs", "dev_pairs"):
            counts = {}
            for row in first[key]:
                counts[row["family"]] = counts.get(row["family"], 0) + 1
            self.assertEqual(set(counts.values()), {64})
        self.assertEqual(
            {row["expression"] for row in first["train_pairs"]}
            & {row["expression"] for row in first["dev_pairs"]},
            set(),
        )

    def test_schedules_match_steps_and_isolate_replay_factor(self):
        config = load_config(CONFIG)
        dataset = build_dataset(config)
        control = build_step_schedule(config, dataset, "control")
        treatment = build_step_schedule(config, dataset, "treatment")
        self.assertEqual(len(control), 512)
        self.assertEqual(len(treatment), 512)
        self.assertEqual(
            [row["pair_id"] for row in control],
            [row["pair_id"] for row in treatment],
        )
        self.assertEqual(
            [row["kind"] for row in control[::2]],
            ["full_consistency"] * 256,
        )
        self.assertEqual(
            [row["kind"] for row in treatment[::2]],
            ["full_consistency"] * 256,
        )
        self.assertEqual(
            [row["kind"] for row in control[1::2]],
            ["repeat_full_consistency"] * 256,
        )
        self.assertEqual(
            [row["kind"] for row in treatment[1::2]],
            ["final_ce_only"] * 256,
        )

    def test_contamination_audit_passes_all_prior_surfaces(self):
        config = load_config(CONFIG)
        audit = contamination_audit(config, build_dataset(config))
        self.assertTrue(audit["passed"])
        self.assertEqual(audit["observed_quality_prompt_overlap"], 0)
        self.assertEqual(audit["benchmark_prompt_overlap"], 0)
        self.assertEqual(
            audit["benchmark_rows_hashed"],
            {"gsm8k": 1319, "mmlu": 14042, "gpqa_diamond": 198},
        )

    def test_public_contract_excludes_raw_content(self):
        contract = public_contract(build_dataset(load_config(CONFIG)))
        for key in ("train_pairs", "dev_pairs"):
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

    def test_config_rejects_posthoc_method_changes(self):
        raw = json.loads(CONFIG.read_text(encoding="utf-8"))
        for key, value, error in (
            ("train_range_offset", 41000, "train_range_offset"),
            ("max_steps_per_arm", 256, "max_steps_per_arm"),
            ("learning_rate", 0.0001, "learning_rate"),
            ("replay_final_ce_weight", 1.0, "replay_final_ce_weight"),
            ("treatment_second_step", "repeat_full_consistency", "treatment"),
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
