from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from nano_train.confidence_route import (
    build_cases,
    combine,
    contamination_audit,
    load_config,
    public_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/confidence_route/qwen35_confidence_route_v1.json"


class ConfidenceRouteTests(unittest.TestCase):
    def test_cases_are_balanced_deterministic_and_uncontaminated(self):
        config = load_config(CONFIG)
        first = build_cases(config)
        second = build_cases(config)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 256)
        counts = {}
        for row in first:
            counts[row["family"]] = counts.get(row["family"], 0) + 1
        self.assertEqual(set(counts.values()), {64})
        audit = contamination_audit(config, first)
        self.assertTrue(audit["passed"])

    def test_public_contract_excludes_raw_content(self):
        contract = public_contract(build_cases(load_config(CONFIG)))
        self.assertEqual(
            set(contract["cases"][0]),
            {"case_id", "family", "prompt_sha256", "target_sha256"},
        )

    def test_selector_uses_strict_relative_margin_and_anchor_tie(self):
        config = load_config(CONFIG)
        anchor = [
            {
                "case_id": "case-a",
                "family": "exact_division",
                "correct": False,
                "parse_failure": False,
                "output": "FINAL: 1",
            },
            {
                "case_id": "case-b",
                "family": "repeated_operand",
                "correct": True,
                "parse_failure": False,
                "output": "FINAL: 2",
            },
        ]
        consistency = [
            {
                "case_id": "case-a",
                "family": "exact_division",
                "correct": True,
                "parse_failure": False,
                "output": "FINAL: 3",
            },
            {
                "case_id": "case-b",
                "family": "repeated_operand",
                "correct": False,
                "parse_failure": False,
                "output": "FINAL: 4",
            },
        ]
        anchor_scores = [
            {
                "case_id": "case-a",
                "anchor_candidate_mean_logprob": -1.0,
                "consistency_candidate_mean_logprob": -2.0,
            },
            {
                "case_id": "case-b",
                "anchor_candidate_mean_logprob": -1.0,
                "consistency_candidate_mean_logprob": -2.0,
            },
        ]
        consistency_scores = [
            {
                "case_id": "case-a",
                "anchor_candidate_mean_logprob": -1.2,
                "consistency_candidate_mean_logprob": -1.5,
            },
            {
                "case_id": "case-b",
                "anchor_candidate_mean_logprob": -1.0,
                "consistency_candidate_mean_logprob": -2.0,
            },
        ]
        values = {
            "generation/anchor/cases.jsonl": anchor,
            "generation/consistency/cases.jsonl": consistency,
            "scores/anchor/scores.jsonl": anchor_scores,
            "scores/consistency/scores.jsonl": consistency_scores,
        }

        def fake_load(path):
            suffix = str(path).split("qwen35-confidence-route-v1/")[-1]
            return values[suffix]

        with mock.patch(
            "nano_train.confidence_route._load_rows",
            side_effect=fake_load,
        ):
            routed, summary = combine(config)
        by_id = {row["case_id"]: row for row in routed}
        self.assertEqual(by_id["case-a"]["route"], "consistency")
        self.assertTrue(by_id["case-a"]["correct"])
        self.assertGreater(by_id["case-a"]["relative_margin"], 0)
        self.assertEqual(by_id["case-b"]["route"], "anchor")
        self.assertTrue(by_id["case-b"]["correct"])
        self.assertEqual(by_id["case-b"]["relative_margin"], 0)
        self.assertEqual(summary["consistency_routes"], 1)
        self.assertEqual(summary["anchor_routes"], 1)

    def test_config_rejects_selector_or_tie_change(self):
        raw = json.loads(CONFIG.read_text(encoding="utf-8"))
        for key, value, error in (
            ("selector", "raw_logprob", "selector"),
            ("tie_policy", "consistency", "tie_policy"),
            ("cases_per_family", 32, "cases_per_family"),
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
