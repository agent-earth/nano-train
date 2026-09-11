from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from nano_train.swe_code_repair_qualification import (
    admission_gates,
    build_qualification_contract,
    load_config,
)
from scripts.preregister_swe_code_repair_harbor_qualification_v1 import (
    build_receipt,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/eval/swe_code_repair_harbor_qualification_v1.json"


def row(
    case_id: str,
    *,
    band: str,
    actionable: bool,
    parser_valid: bool | None = None,
    exhausted: bool = False,
) -> dict:
    return {
        "sample_id_sha256": case_id,
        "task_family": band,
        "terminus_parser_valid": (
            actionable if parser_valid is None else parser_valid
        ),
        "terminus_actionable_or_complete": actionable,
        "generation_budget_exhausted": exhausted,
    }


class SWECodeRepairQualificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config(CONFIG)
        cls.contract = build_qualification_contract(cls.config)

    def test_config_freezes_unchanged_adapter_qualification(self):
        self.assertEqual(
            self.config.experiment_id,
            "evoloop-swe-code-repair-harbor-qualification-v1",
        )
        self.assertEqual(sum(self.config.dev_rows_by_band.values()), 24)
        self.assertEqual(self.config.minimum_adapter_parser_valid, 23)
        self.assertEqual(self.config.minimum_adapter_only_wins, 1)
        self.assertEqual(self.config.maximum_base_only_losses, 0)

    def test_selection_is_deterministic_fresh_and_balanced(self):
        second = build_qualification_contract(self.config)
        self.assertEqual(
            self.contract["selected_dev_ids"],
            second["selected_dev_ids"],
        )
        self.assertEqual(len(self.contract["selected_dev_ids"]), 24)
        selected = self.contract["selected_dev"]
        counts = {
            band: sum(
                (
                    "short"
                    if int(sample["token_count"]) <= 768
                    else "medium"
                    if int(sample["token_count"]) <= 1_280
                    else "long"
                )
                == band
                for sample in selected
            )
            for band in ("short", "medium", "long")
        }
        self.assertEqual(counts, self.config.dev_rows_by_band)
        source = build_qualification_contract(self.config)
        source_sft_selection = __import__(
            "nano_train.swe_code_repair_sft",
            fromlist=["build_selection_contract"],
        ).build_selection_contract(source["source_config"])
        self.assertFalse(
            set(self.contract["selected_dev_ids"])
            & set(source_sft_selection["dev_sample_ids"])
        )

    def test_config_rejects_threshold_mutation(self):
        raw = json.loads(CONFIG.read_text(encoding="utf-8"))
        raw["minimum_adapter_parser_valid"] = 22
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaisesRegex(
                ValueError,
                "freezes minimum_adapter_parser_valid",
            ):
                load_config(path)

    def test_admission_requires_paired_gain_zero_loss_and_band_safety(self):
        bands = ["short"] * 8 + ["medium"] * 8 + ["long"] * 8
        base = [
            row(str(index), band=band, actionable=index != 0)
            for index, band in enumerate(bands)
        ]
        adapter = [
            row(str(index), band=band, actionable=True)
            for index, band in enumerate(bands)
        ]
        gates = admission_gates(
            base_loss=1.0,
            adapter_loss=0.9,
            base_rows=base,
            adapter_rows=adapter,
            config=self.config,
        )
        self.assertTrue(all(gates.values()))

        regressed = copy.deepcopy(adapter)
        regressed[1]["terminus_actionable_or_complete"] = False
        regressed[2]["terminus_actionable_or_complete"] = False
        regression_gates = admission_gates(
            base_loss=1.0,
            adapter_loss=0.9,
            base_rows=base,
            adapter_rows=regressed,
            config=self.config,
        )
        self.assertFalse(regression_gates["maximum_base_only_losses"])
        self.assertFalse(regression_gates["per_band_non_regression"])

    def test_preregister_is_deterministic_and_closed(self):
        first = build_receipt()
        second = build_receipt()
        self.assertEqual(first, second)
        self.assertEqual(first["selection"]["dev_samples"], 24)
        self.assertEqual(first["selection"]["overlap_with_prior_v1_dev"], 0)
        self.assertTrue(
            first["decision_boundary"][
                "passing_unlocks_only_fresh8_sft_screening"
            ]
        )
        self.assertFalse(first["decision_boundary"]["complete_500_allowed"])
        self.assertTrue(
            first["execution_boundary"]["this_commit_only_preregisters"]
        )


if __name__ == "__main__":
    unittest.main()
