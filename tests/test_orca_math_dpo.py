from __future__ import annotations

import json
import unittest
from pathlib import Path

import torch

from nano_train.orca_math_dpo import (
    _dev_rows,
    _tokenize_target,
    build_selection,
    dpo_loss,
    load_config,
    sequence_log_probability,
)
from scripts.preregister_orca_math_dpo_v1 import build_receipt
from scripts.render_orca_math_dpo_v1 import build_report


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/preference_orca_math_dpo_v1.json"


class OrcaMathDPOTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config(CONFIG)
        cls.selection = build_selection(cls.config)

    def test_selection_is_fresh_and_frozen(self):
        self.assertEqual(len(self.selection["train"]), 32)
        self.assertEqual(len(self.selection["dev"]), 192)
        self.assertFalse(
            set(self.selection["train_ids"]) & set(self.selection["dev_ids"])
        )
        self.assertEqual(
            build_selection(self.config)["train_ids"],
            self.selection["train_ids"],
        )

    def test_sequence_log_probability_masks_prompt(self):
        logits = torch.zeros(1, 4, 5)
        labels = torch.tensor([[-100, -100, 2, 3]])
        value = sequence_log_probability(logits, labels)
        self.assertEqual(value.shape, (1,))
        self.assertAlmostEqual(float(value.item()), -torch.log(torch.tensor(5.0)).item())

    def test_dpo_loss_rewards_larger_policy_margin(self):
        good, advantage = dpo_loss(
            torch.tensor([2.0]),
            torch.tensor([0.0]),
            torch.tensor([0.0]),
            torch.tensor([0.0]),
            beta=0.1,
        )
        bad, _ = dpo_loss(
            torch.tensor([0.0]),
            torch.tensor([2.0]),
            torch.tensor([0.0]),
            torch.tensor([0.0]),
            beta=0.1,
        )
        self.assertGreater(float(advantage.item()), 0)
        self.assertLess(float(good.item()), float(bad.item()))

    def test_preregister_is_deterministic_and_closed(self):
        first = build_receipt()
        second = build_receipt()
        self.assertEqual(first, second)
        self.assertFalse(first["decision_boundary"]["benchmark_allowed"])
        self.assertTrue(
            first["execution_boundary"]["this_commit_only_preregisters"]
        )

    def test_tokenization_and_dev_views_preserve_frozen_contract(self):
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_path,
            local_files_only=True,
        )
        row = self.selection["train"][0]
        chosen = _tokenize_target(
            tokenizer,
            row,
            row["chosen"],
            max_length=self.config.max_length,
            suffix="chosen",
        )
        rejected = _tokenize_target(
            tokenizer,
            row,
            row["rejected"],
            max_length=self.config.max_length,
            suffix="rejected",
        )
        self.assertEqual(chosen.prompt_ids, rejected.prompt_ids)
        self.assertNotEqual(chosen.target, rejected.target)
        dev = _dev_rows(self.selection)
        self.assertEqual(len(dev), 192)
        self.assertTrue(all(row["numeric_answer"] for row in dev))

    def test_public_report_recomputes_stable_noop(self):
        report = build_report()
        self.assertFalse(report["decision"]["candidate_admitted"])
        self.assertEqual(report["evaluation"]["changed_outputs"], 0)
        self.assertEqual(report["evaluation"]["comparison"]["delta"], 0.0)
        self.assertTrue(report["reload"]["generations_exact"])
        serialized = json.dumps(report).lower()
        self.assertNotIn("prompt_messages", serialized)
        self.assertNotIn("expected", serialized)


if __name__ == "__main__":
    unittest.main()
