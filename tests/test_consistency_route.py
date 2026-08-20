from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from nano_train.consistency_route import (
    build_cases,
    contamination_audit,
    load_config,
    public_contract,
    routed_rows,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/consistency_route/qwen35_consistency_route_v1.json"


class ConsistencyRouteTests(unittest.TestCase):
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
        self.assertEqual(audit["observed_quality_prompt_overlap"], 0)
        self.assertEqual(audit["benchmark_prompt_overlap"], 0)

    def test_public_contract_excludes_raw_content(self):
        contract = public_contract(build_cases(load_config(CONFIG)))
        self.assertEqual(contract["case_count"], 256)
        self.assertEqual(
            set(contract["cases"][0]),
            {"case_id", "family", "prompt_sha256", "target_sha256"},
        )

    def test_route_changes_only_exact_division(self):
        config = load_config(CONFIG)
        anchor = []
        consistency = []
        for family in (
            "repeated_operand",
            "mixed_products",
            "exact_division",
            "nested_offset",
        ):
            case_id = f"case-{family}"
            anchor.append(
                {
                    "case_id": case_id,
                    "family": family,
                    "correct": False,
                    "parse_failure": False,
                }
            )
            consistency.append(
                {
                    "case_id": case_id,
                    "family": family,
                    "correct": True,
                    "parse_failure": False,
                }
            )
        routed = {row["case_id"]: row for row in routed_rows(config, anchor, consistency)}
        self.assertEqual(
            routed["case-exact_division"]["route"],
            "consistency",
        )
        for family in ("repeated_operand", "mixed_products", "nested_offset"):
            self.assertEqual(routed[f"case-{family}"]["route"], "anchor")
            self.assertFalse(routed[f"case-{family}"]["correct"])

    def test_config_rejects_route_or_case_change(self):
        raw = json.loads(CONFIG.read_text(encoding="utf-8"))
        for key, value, error in (
            ("routed_family", "repeated_operand", "routed_family"),
            ("cases_per_family", 32, "cases_per_family"),
            ("max_new_tokens", 64, "max_new_tokens"),
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
