from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from nano_train.router_classification import load_config, verify_data_release
from scripts.preregister_router_classification_sft_v1 import (
    build_receipt,
    render_markdown,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/sft/router_classification_smoke_v1.json"


class RouterClassificationSFTTests(unittest.TestCase):
    def test_config_freezes_recipe(self):
        config = load_config(CONFIG)
        self.assertEqual(config.schema_version, "nano_train_sft_smoke_v1")
        self.assertEqual(config.max_steps, 40)
        self.assertEqual(config.gradient_accumulation_steps, 4)
        self.assertEqual(config.learning_rate, 0.0002)
        self.assertEqual(config.dtype, "float32")
        self.assertEqual(
            config.lora_targets,
            ("q_proj", "v_proj", "gate_proj", "up_proj", "down_proj"),
        )
        self.assertEqual(config.generation_max_new_tokens, 8)

    def test_preregister_binds_data_exposure_and_closed_boundaries(self):
        first = build_receipt()
        second = build_receipt()
        self.assertEqual(first, second)
        self.assertEqual(first["data"]["train_rows"], 768)
        self.assertEqual(first["data"]["validation_rows"], 192)
        self.assertEqual(first["data"]["scheduled_exposures"], 160)
        self.assertGreaterEqual(
            min(first["data"]["scheduled_exposure_by_label"].values()),
            40,
        )
        self.assertEqual(
            first["data"]["validation_by_label"],
            {"router_a": 64, "router_b": 64, "router_c": 64},
        )
        boundary = first["execution_boundary"]
        self.assertFalse(boundary["training_started"])
        self.assertFalse(boundary["model_generation_started"])
        self.assertFalse(boundary["adapter_exists"])
        self.assertFalse(boundary["metrics_exist"])
        markdown = render_markdown(first)
        self.assertIn("40 steps", markdown)
        self.assertIn("A/B/C 各64", markdown)

    def test_data_release_is_bound(self):
        config = load_config(CONFIG)
        release = verify_data_release(config)
        self.assertEqual(
            release["dataset_file_sha256"],
            "dacd3663639fe9ddc054865b87afdd0c918f0fddb12c8c9355819d4bbce95d65",
        )
        self.assertEqual(release["accepted"]["train_tokens"], 84_160)

    def test_config_rejects_recipe_search(self):
        raw = json.loads(CONFIG.read_text(encoding="utf-8"))
        mutations = (
            ("max_steps", 20, "max_steps"),
            ("learning_rate", 0.00005, "learning_rate"),
            ("seed", 20260825, "seed"),
            ("generation_max_new_tokens", 16, "generation_max_new_tokens"),
        )
        for key, value, error in mutations:
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
