from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from nano_train.orca_math_sft import (
    _training_dataset,
    admission_gates,
    build_selection_contract,
    load_config,
    parse_final,
    score_output,
)
from scripts.preregister_orca_math_sft_v1 import build_receipt
from scripts.render_orca_math_sft_v1 import build_report


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/sft/orca_math_smoke_v1.json"


class OrcaMathSFTTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config(CONFIG)
        cls.selection = build_selection_contract(cls.config)

    def test_config_freezes_smoke_and_closed_boundary(self):
        self.assertEqual(self.config.max_steps, 40)
        self.assertEqual(self.config.gradient_accumulation_steps, 4)
        self.assertEqual(sum(self.config.train_rows_by_stratum.values()), 160)
        self.assertEqual(sum(self.config.dev_rows_by_stratum.values()), 192)
        self.assertEqual(self.config.lora_targets, ("q_proj", "v_proj"))

    def test_selection_is_deterministic_and_disjoint(self):
        second = build_selection_contract(self.config)
        self.assertEqual(self.selection["train_sample_ids"], second["train_sample_ids"])
        self.assertEqual(self.selection["dev_sample_ids"], second["dev_sample_ids"])
        self.assertEqual(len(self.selection["train_sample_ids"]), 160)
        self.assertEqual(len(self.selection["dev_sample_ids"]), 192)
        self.assertFalse(
            set(self.selection["train_sample_ids"])
            & set(self.selection["dev_sample_ids"])
        )
        dataset = _training_dataset(self.selection)
        self.assertEqual(len(dataset["samples"]), 352)
        self.assertEqual(
            sum(row["split"] == "train" for row in dataset["samples"]),
            160,
        )

    def test_strict_final_numeric_scorer(self):
        self.assertEqual(parse_final("reasoning\nFINAL: 0.75"), "0.75")
        self.assertTrue(score_output("reasoning\nFINAL: 0.75", "3/4"))
        self.assertTrue(score_output("FINAL: -3", "-3.0"))
        self.assertFalse(score_output("Answer: 3", "3"))
        self.assertFalse(score_output("FINAL: 3\nextra", "3"))
        self.assertFalse(score_output("FINAL: 4", "3"))

    def test_summary_and_comparison_rows_use_frozen_case_ids(self):
        baseline = [
            {
                "case_id": "a",
                "stratum": "short",
                "correct": False,
                "parse_failure": False,
            },
            {
                "case_id": "b",
                "stratum": "medium",
                "correct": True,
                "parse_failure": False,
            },
        ]
        candidate = [
            {**baseline[0], "correct": True},
            baseline[1],
        ]
        from nano_train.orca_math_sft import compare_rows, summarize_rows

        summary = summarize_rows(candidate)
        comparison = compare_rows(
            candidate,
            baseline,
            bootstrap_samples=100,
            bootstrap_seed=1,
        )
        self.assertEqual(summary["correct"], 2)
        self.assertEqual(comparison["paired_counts"]["candidate_only"], 1)
        self.assertEqual(comparison["paired_counts"]["baseline_only"], 0)

    def test_admission_requires_significance_and_non_regression(self):
        comparison = {
            "delta": 0.05,
            "paired_bootstrap_95_ci": [0.01, 0.09],
            "mcnemar_exact_p": 0.03,
            "paired_counts": {
                "candidate_only": 8,
                "baseline_only": 1,
            },
        }
        gates = admission_gates(
            comparison,
            candidate_by_stratum={"short": 10, "medium": 20, "long": 10},
            baseline_by_stratum={"short": 9, "medium": 20, "long": 10},
            alpha=0.05,
            minimum_candidate_only_wins=6,
        )
        self.assertTrue(all(gates.values()))
        regressed = admission_gates(
            comparison,
            candidate_by_stratum={"short": 8, "medium": 20, "long": 10},
            baseline_by_stratum={"short": 9, "medium": 20, "long": 10},
            alpha=0.05,
            minimum_candidate_only_wins=6,
        )
        self.assertFalse(regressed["every_stratum_non_regression"])

    def test_config_rejects_mutation(self):
        raw = json.loads(CONFIG.read_text(encoding="utf-8"))
        raw["max_steps"] = 41
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "freezes max_steps"):
                load_config(path)

    def test_preregister_is_deterministic(self):
        first = build_receipt()
        second = build_receipt()
        self.assertEqual(first, second)
        self.assertTrue(
            first["execution_boundary"]["this_commit_only_preregisters"]
        )
        self.assertFalse(
            first["decision_boundary"]["benchmark_allowed"]
        )

    def test_public_report_recomputes_rejection_without_raw_prompts(self):
        report = build_report()
        self.assertFalse(report["decision"]["candidate_admitted"])
        self.assertEqual(
            report["evaluation"]["transitions"]["repaired"],
            8,
        )
        self.assertEqual(
            report["evaluation"]["transitions"]["regressed"],
            50,
        )
        self.assertTrue(report["reload"]["generations_exact"])
        serialized = json.dumps(report).lower()
        self.assertNotIn("messages", serialized)
        self.assertNotIn("question", serialized)


if __name__ == "__main__":
    unittest.main()
