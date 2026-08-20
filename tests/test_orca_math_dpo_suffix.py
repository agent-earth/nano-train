from __future__ import annotations

import unittest
from pathlib import Path

from transformers import AutoTokenizer

from nano_train.orca_math_dpo_suffix import (
    build_selection,
    load_config,
    tokenize_suffix_pair,
)
from scripts.preregister_orca_math_dpo_v2 import build_receipt


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/preference_orca_math_dpo_v2.json"


class SuffixDPOTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config(CONFIG)
        cls.selection = build_selection(cls.config)

    def test_selection_is_disjoint_from_v1(self):
        self.assertEqual(len(self.selection["train"]), 32)
        self.assertEqual(len(self.selection["dev"]), 192)
        prior = set()
        import json

        value = json.loads(
            Path(self.config.prior_dpo_preregister_path).read_text(
                encoding="utf-8"
            )
        )
        prior.update(value["selection"]["train_ids"])
        prior.update(value["selection"]["dev_ids"])
        self.assertFalse(
            prior
            & (
                set(self.selection["train_ids"])
                | set(self.selection["dev_ids"])
            )
        )

    def test_suffix_mask_supervises_only_differing_tail(self):
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_path, local_files_only=True
        )
        row = self.selection["train"][0]
        chosen, rejected, common = tokenize_suffix_pair(
            tokenizer, row, max_length=self.config.max_length
        )
        self.assertGreater(common, 0)
        self.assertEqual(chosen.prompt_ids, rejected.prompt_ids)
        self.assertLess(
            sum(label != -100 for label in chosen.labels),
            len(tokenizer(row["chosen"], add_special_tokens=False).input_ids),
        )
        self.assertGreater(
            sum(label != -100 for label in chosen.labels), 0
        )

    def test_preregister_is_deterministic_and_closed(self):
        first = build_receipt()
        second = build_receipt()
        self.assertEqual(first, second)
        self.assertFalse(first["decision_boundary"]["benchmark_allowed"])
        self.assertTrue(
            first["execution_boundary"]["this_commit_only_preregisters"]
        )


if __name__ == "__main__":
    unittest.main()
