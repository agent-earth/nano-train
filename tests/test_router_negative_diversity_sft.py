from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from nano_train.router_negative_diversity import (
    load_config,
    verify_data_release,
)
from scripts.preregister_router_negative_diversity_sft_v2 import (
    build_receipt,
    render_markdown,
)
from scripts.render_router_negative_diversity_sft_v2 import (
    result_gates,
    subtype_metrics,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/sft/router_negative_diversity_v2.json"


class RouterNegativeDiversitySFTTests(unittest.TestCase):
    def test_config_freezes_original_bounded_recipe(self):
        config = load_config(CONFIG)
        self.assertEqual(config.max_steps, 40)
        self.assertEqual(config.gradient_accumulation_steps, 4)
        self.assertEqual(config.learning_rate, 0.0002)
        self.assertEqual(config.dtype, "float32")
        self.assertEqual(config.seed, 20260827)
        self.assertEqual(
            config.lora_targets,
            ("q_proj", "v_proj", "gate_proj", "up_proj", "down_proj"),
        )
        self.assertEqual(config.generation_max_new_tokens, 8)

    def test_data_release_is_bound_and_training_unblocked(self):
        release = verify_data_release(load_config(CONFIG))
        self.assertEqual(
            release["dataset_file_sha256"],
            "8c5975e3ceed494e20d0de54eb5654ab1af71163ed58489d42d98c8b54d0bad9",
        )
        self.assertEqual(
            release["sha256"],
            "5edd89701ff33db6eaef74475946abf79176c5c5a7c854a7eea4dd907e69c3f1",
        )
        self.assertEqual(release["accepted"]["train_tokens"], 766_519)

    def test_preregister_binds_exposure_full_dev_and_closed_boundary(self):
        first = build_receipt()
        second = build_receipt()
        self.assertEqual(first, second)
        data = first["data"]
        self.assertEqual(data["train_rows"], 6144)
        self.assertEqual(data["validation_rows"], 1536)
        self.assertEqual(data["scheduled_exposures"], 160)
        self.assertEqual(
            data["scheduled_exposure_by_label"],
            {"router_a": 52, "router_b": 48, "router_c": 60},
        )
        self.assertEqual(set(data["scheduled_c_exposure_by_subtype"]), {
            "box_total",
            "remaining_stock",
            "paired_average",
            "single_operation",
            "weighted_total",
            "quotient_remainder",
            "time_conversion",
            "percentage_change",
        })
        self.assertGreaterEqual(
            min(data["scheduled_c_exposure_by_subtype"].values()), 3
        )
        self.assertEqual(
            data["validation_by_label"],
            {"router_a": 512, "router_b": 512, "router_c": 512},
        )
        self.assertEqual(
            data["validation_c_by_subtype"],
            {
                "box_total": 64,
                "remaining_stock": 64,
                "paired_average": 64,
                "single_operation": 64,
                "weighted_total": 64,
                "quotient_remainder": 64,
                "time_conversion": 64,
                "percentage_change": 64,
            },
        )
        boundary = first["execution_boundary"]
        self.assertFalse(boundary["training_started"])
        self.assertFalse(boundary["model_generation_started"])
        self.assertFalse(boundary["adapter_exists"])
        self.assertFalse(boundary["metrics_exist"])
        self.assertFalse(boundary["integration_rows_or_outputs_loaded"])
        acceptance = first["acceptance"]
        self.assertTrue(
            acceptance["every_c_subtype_post_exact_at_least_60_of_64"]
        )
        self.assertTrue(acceptance["serving_namespace_remap_required"])
        markdown = render_markdown(first)
        self.assertIn("40 steps", markdown)
        self.assertIn("8 subtypes 各64", markdown)
        self.assertIn("namespace remap", markdown)

    def test_config_rejects_recipe_or_data_search(self):
        raw = json.loads(CONFIG.read_text(encoding="utf-8"))
        mutations = (
            ("max_steps", 20, "max_steps"),
            ("learning_rate", 0.00005, "learning_rate"),
            ("seed", 20260828, "seed"),
            ("generation_max_new_tokens", 16, "generation_max_new_tokens"),
            (
                "dataset_path",
                "../other.json",
                "dataset_path",
            ),
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

    def test_subtype_scorer_and_gates_require_every_subtype(self):
        subtypes = (
            "box_total",
            "remaining_stock",
            "paired_average",
            "single_operation",
            "weighted_total",
            "quotient_remainder",
            "time_conversion",
            "percentage_change",
        )
        sample_by_id = {}
        rows = []
        for subtype_index, subtype in enumerate(subtypes):
            for index in range(64):
                sample_id = f"{subtype}-{index}"
                sample_by_id[sample_id] = {
                    "sample_id": sample_id,
                    "split": "validation",
                    "route_label": "C",
                    "negative_subtype": subtype,
                }
                rows.append(
                    {
                        "sample_id": sample_id,
                        "exact": True,
                    }
                )
        scored = subtype_metrics(rows, sample_by_id)
        self.assertEqual(
            scored,
            {
                subtype: {
                    "samples": 64,
                    "exact": 64,
                    "failure_sample_ids": [],
                }
                for subtype in subtypes
            },
        )
        family = {
            "router_a": {"samples": 512, "exact": 500},
            "router_b": {"samples": 512, "exact": 500},
            "router_c": {"samples": 512, "exact": 512},
        }
        baseline = {
            "samples": 1536,
            "exact": 1200,
            "by_family": family,
        }
        post = {
            "samples": 1536,
            "exact": 1512,
            "by_family": family,
        }
        metrics = {
            "loss_curve": [{"step": step, "loss": 0.1} for step in range(40)],
            "adapter_sha256": "adapter",
            "train_exposure": [
                {"sample_ids": ["a", "b", "c", "d"]} for _ in range(40)
            ],
        }
        reload = {
            "reload_success": True,
            "metrics_exact": True,
            "generations_exact": True,
        }
        release = {
            "dataset_file_sha256": (
                "8c5975e3ceed494e20d0de54eb5654ab1af71163ed58489d42d98c8b54d0bad9"
            ),
            "dataset_canonical_sha256": (
                "f63c58b54ef4747f274599784bad9ffe4143117482c22b33005a2dbf725b1f2f"
            ),
            "audit_sha256": (
                "9aaa69de746dbdc5cefbb52fb271c8f9ec86716d10ada70704c7e346dc2f7c17"
            ),
            "contract_sha256": (
                "c195a7373ea283546dde1866f70593f0912833d987ff5f1a8cb424c2bc340335"
            ),
        }
        with patch(
            "scripts.render_router_negative_diversity_sft_v2.sha256_tree",
            return_value="adapter",
        ), patch(
            "scripts.render_router_negative_diversity_sft_v2.ARTIFACT",
            Path("/does-not-exist"),
        ):
            gates = result_gates(
                baseline,
                post,
                scored,
                scored,
                metrics=metrics,
                reload=reload,
                release=release,
                exposure_ids_exact=True,
            )
        self.assertTrue(all(gates.values()))
        failing = copy.deepcopy(scored)
        failing["box_total"]["exact"] = 59
        with patch(
            "scripts.render_router_negative_diversity_sft_v2.sha256_tree",
            return_value="adapter",
        ), patch(
            "scripts.render_router_negative_diversity_sft_v2.ARTIFACT",
            Path("/does-not-exist"),
        ):
            gates = result_gates(
                baseline,
                post,
                scored,
                failing,
                metrics=metrics,
                reload=reload,
                release=release,
                exposure_ids_exact=True,
            )
        self.assertFalse(
            gates["every_c_subtype_post_exact_at_least_60_of_64"]
        )


if __name__ == "__main__":
    unittest.main()
