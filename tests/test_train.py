from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from nano_train.config import load_sft_smoke_config
from nano_train.continuation import (
    load_config as load_continuation_config,
    normalized_anchor_penalty,
    validate_choice_replay_contract,
)
from nano_train.paired_consistency import (
    align_teacher_logits,
    build_selection_contract,
    build_step_schedule,
    load_config as load_paired_consistency_config,
    paired_consistency_kl,
    target_prediction_logits,
)
from nano_train.data import (
    TokenizedSample,
    collate_samples,
    execution_target_output_valid,
    load_execution_target_dataset,
    load_skill_release_dataset,
    semantic_output_valid,
    skill_release_output_valid,
    tokenize_samples,
)
from nano_train.sft import (
    _assert_finite_gradients,
    _assert_finite_loss,
    _assert_finite_parameters,
    _batch_order,
    _sample_scheduled_batch_order,
    _scheduled_batch_order,
    _scheduler_scale,
    _write_failure,
    evaluate_exact,
)
from scripts.run_generation_budget_audit import (
    load_config as load_audit_config,
    validate_contract as validate_audit_contract,
    verify_identity as verify_audit_identity,
)
from scripts.build_lora_delta_composition import compose_pair


class FakeTokenizer:
    eos_token = "<eos>"
    pad_token_id = 0

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
        enable_thinking,
    ):
        self.assertions = (tokenize, add_generation_prompt, enable_thinking)
        return "|".join(message["content"] for message in messages) + "|assistant:"

    def __call__(self, text, *, add_special_tokens=False):
        return SimpleNamespace(input_ids=[ord(char) % 251 + 1 for char in text])


class TrainTests(unittest.TestCase):
    def test_paired_consistency_config_and_selection_are_frozen(self):
        config = load_paired_consistency_config(
            "configs/paired_consistency/execution_target_consistency_v1.json"
        )
        selection = build_selection_contract(config)
        steps = build_step_schedule(selection)
        self.assertEqual(config.max_steps, 40)
        self.assertEqual(config.consistency_weight, 1.0)
        self.assertTrue(config.teacher_detach)
        self.assertEqual(len(selection["heldout_sample_ids"]), 80)
        self.assertEqual(len(selection["pair_schedule"]), 20)
        self.assertEqual(len(selection["json_schedule"]), 20)
        self.assertEqual(len(steps), 40)
        self.assertEqual(
            [step["kind"] for step in steps],
            ["pair", "json"] * 20,
        )
        prior = load_sft_smoke_config(
            "configs/sft/execution_target_paired_smoke_v1.json"
        )
        self.assertFalse(
            set(selection["heldout_sample_ids"])
            & set(prior.train_sample_schedule)
        )
        self.assertFalse(
            {
                sample_id
                for pair in selection["pair_schedule"]
                for sample_id in (
                    pair["process_sample_id"],
                    pair["final_sample_id"],
                )
            }
            & set(prior.train_sample_schedule)
        )

    def test_paired_consistency_kl_detaches_teacher(self):
        teacher = torch.randn(3, 7, requires_grad=True)
        student = torch.randn(3, 7, requires_grad=True)
        loss = paired_consistency_kl(
            teacher,
            student,
            temperature=1.0,
            teacher_detach=True,
        )
        loss.backward()
        self.assertIsNone(teacher.grad)
        self.assertIsNotNone(student.grad)
        self.assertTrue(torch.isfinite(student.grad).all())

    def test_paired_consistency_kl_is_zero_for_equal_logits(self):
        logits = torch.randn(4, 9)
        loss = paired_consistency_kl(
            logits,
            logits.clone(),
            temperature=1.0,
            teacher_detach=True,
        )
        self.assertAlmostEqual(float(loss), 0.0, places=6)

    def test_target_logits_align_final_suffix_and_reject_mismatch(self):
        process_logits = torch.randn(1, 6, 11)
        process_labels = torch.tensor([[-100, 1, 2, 3, 4, 5]])
        final_logits = torch.randn(1, 3, 11)
        final_labels = torch.tensor([[-100, 4, 5]])
        teacher_logits, teacher_labels = target_prediction_logits(
            process_logits,
            process_labels,
        )
        student_logits, student_labels = target_prediction_logits(
            final_logits,
            final_labels,
        )
        aligned = align_teacher_logits(
            teacher_logits,
            teacher_labels,
            student_labels,
        )
        self.assertEqual(aligned.shape, student_logits.shape)
        bad_labels = torch.tensor([8, 5])
        with self.assertRaisesRegex(ValueError, "suffix labels differ"):
            align_teacher_logits(
                teacher_logits,
                teacher_labels,
                bad_labels,
            )

    def test_normalized_anchor_penalty(self):
        parameter = torch.nn.Parameter(torch.tensor([3.0, 5.0]))
        anchor = torch.tensor([1.0, 1.0])
        anchor_norm_squared = (anchor**2).sum()
        penalty = normalized_anchor_penalty(
            [parameter],
            [anchor],
            anchor_norm_squared,
        )
        self.assertEqual(float(penalty.detach()), 5.0)

    def test_choice_replay_contract_records_all_rule_exposures(self):
        rules = [
            "preservation_host_count_choice_v5",
            "preservation_sequential_fraction_choice_v5",
            "preservation_participant_average_choice_v5",
        ]
        samples = [
            {
                "sample_id": f"train-{index}",
                "split": "train",
                "task_family": "capability_preservation_choice",
                "format_family": "final_choice",
                "generation_rule": rules[index % len(rules)],
            }
            for index in range(40)
        ]
        samples.extend(
            {
                "sample_id": f"validation-{index}",
                "split": "validation",
                "task_family": "capability_preservation_numeric",
                "format_family": "final_numeric",
            }
            for index in range(32)
        )
        dataset = {
            "dataset_id": "generic-choice-replay-v11",
            "policy": {
                "contains_benchmark_content": False,
                "contains_model_outputs": False,
                "contains_teacher_outputs": False,
                "sealed_canary_used_for_training": False,
                "independent_holdout_used_for_training": False,
                "benchmark_feedback_used_for_training": False,
            },
            "samples": samples,
        }
        exposure = validate_choice_replay_contract(
            dataset,
            seed=20260816,
            examples_seen=16,
        )
        self.assertEqual(exposure["examples_seen"], 16)
        self.assertEqual(set(exposure["generation_rule_counts"]), set(rules))

    def test_choice_replay_config_is_frozen(self):
        source = Path(
            "configs/continuation/anchored_v1_choice_replay_v2.json"
        )
        raw = json.loads(source.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(raw), encoding="utf-8")
            config = load_continuation_config(path)
            self.assertEqual(config.max_steps, 4)
            raw["learning_rate"] = 0.00005
            path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "learning_rate"):
                load_continuation_config(path)

    def test_exact_lora_delta_composition(self):
        a_left = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        b_left = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
        a_right = torch.tensor([[2.0, 1.0], [0.0, 3.0]])
        b_right = torch.tensor([[0.5, 1.0], [2.0, 0.0]])
        a, b = compose_pair(
            a_left,
            b_left,
            a_right,
            b_right,
            preservation_weight=0.75,
            capability_weight=0.25,
        )
        actual = b @ a
        expected = 0.75 * (2 * b_left @ a_left) + 0.25 * (
            2 * b_right @ a_right
        )
        self.assertTrue(torch.equal(actual, expected))

    def test_config_rejects_non_smoke_step_count(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": "nano_train_sft_smoke_v1",
                        "experiment_id": "x",
                        "model_path": "model",
                        "dataset_path": "data",
                        "output_dir": "out",
                        "seed": 1,
                        "dtype": "float16",
                        "max_length": 128,
                        "max_steps": 41,
                        "batch_size": 1,
                        "gradient_accumulation_steps": 1,
                        "learning_rate": 0.001,
                        "weight_decay": 0.0,
                        "warmup_steps": 1,
                        "lora_r": 4,
                        "lora_alpha": 8,
                        "lora_dropout": 0.0,
                        "lora_targets": ["q_proj"],
                        "generation_max_new_tokens": 8,
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "40 optimizer steps"):
                load_sft_smoke_config(path)

    def test_config_accepts_float32_smoke(self):
        config = load_sft_smoke_config(
            "configs/sft/format_contract_smoke_v2.json"
        )
        self.assertEqual(config.dtype, "float32")
        self.assertEqual(config.max_steps, 20)

    def test_v2_config_accepts_long_sequence_release_smoke(self):
        config = load_sft_smoke_config(
            "configs/sft/skill_release_long_sequence_smoke_v1.json"
        )

        self.assertEqual(config.schema_version, "nano_train_sft_smoke_v2")
        self.assertEqual(config.dataset_schema, "skill_release_jsonl_v1")
        self.assertEqual(config.max_length, 1088)
        self.assertEqual(config.max_steps, 4)
        self.assertTrue(config.gradient_checkpointing)
        self.assertEqual(config.train_samples_per_family, 2)
        self.assertEqual(config.validation_samples_per_family, 1)

    def test_v1_config_still_rejects_long_sequence(self):
        raw = json.loads(
            Path("configs/sft/format_contract_smoke_v2.json").read_text()
        )
        raw["max_length"] = 257
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(raw), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "256"):
                load_sft_smoke_config(path)

    def test_v2_config_requires_gradient_checkpointing(self):
        raw = json.loads(
            Path(
                "configs/sft/skill_release_long_sequence_smoke_v1.json"
            ).read_text()
        )
        raw["gradient_checkpointing"] = False
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(raw), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "gradient checkpointing"):
                load_sft_smoke_config(path)

    def test_bounded_dose_changes_only_preregistered_dose_fields(self):
        smoke = load_sft_smoke_config(
            "configs/sft/skill_release_long_sequence_smoke_v1.json"
        )
        dose = load_sft_smoke_config(
            "configs/sft/skill_release_bounded_dose_v2.json"
        )
        changed = {
            "experiment_id",
            "output_dir",
            "max_steps",
            "train_samples_per_family",
            "validation_samples_per_family",
        }
        for field in smoke.__dataclass_fields__:
            if field in changed:
                continue
            self.assertEqual(getattr(smoke, field), getattr(dose, field), field)
        self.assertEqual(dose.max_steps, 20)
        self.assertEqual(dose.train_samples_per_family, 16)
        self.assertEqual(dose.validation_samples_per_family, 4)

    def test_expanded_lora_changes_only_method_and_identity(self):
        dose = load_sft_smoke_config(
            "configs/sft/skill_release_bounded_dose_v2.json"
        )
        expanded = load_sft_smoke_config(
            "configs/sft/skill_release_expanded_lora_v3.json"
        )
        changed = {"experiment_id", "output_dir", "lora_targets"}
        for field in dose.__dataclass_fields__:
            if field in changed:
                continue
            self.assertEqual(
                getattr(dose, field),
                getattr(expanded, field),
                field,
            )
        self.assertEqual(
            expanded.lora_targets,
            ("q_proj", "v_proj", "gate_proj", "up_proj", "down_proj"),
        )

    def test_reasoning_preservation_preregistration(self):
        control = load_sft_smoke_config(
            "configs/sft/skill_release_bounded_dose_v2.json"
        )
        treatment = load_sft_smoke_config(
            "configs/sft/skill_release_reasoning_preservation_v4.json"
        )
        changed = {
            "experiment_id",
            "output_dir",
            "validation_start_per_family",
            "train_family_schedule",
        }
        for field in control.__dataclass_fields__:
            if field in changed:
                continue
            self.assertEqual(
                getattr(control, field),
                getattr(treatment, field),
                field,
            )
        self.assertEqual(treatment.validation_start_per_family, 4)
        self.assertEqual(len(treatment.train_family_schedule), 20)
        self.assertEqual(
            treatment.train_family_schedule.count("verified-reasoning"),
            10,
        )
        for family in (
            "coding-and-validation",
            "planning-and-state",
            "skill-routing-and-reflection",
            "tool-use-and-recovery",
        ):
            self.assertGreaterEqual(
                treatment.train_family_schedule.count(family),
                2,
            )

    def test_execution_target_preregistration_is_frozen(self):
        config = load_sft_smoke_config(
            "configs/sft/execution_target_paired_smoke_v1.json"
        )
        self.assertEqual(config.schema_version, "nano_train_sft_smoke_v3")
        self.assertEqual(config.dataset_schema, "execution_target_json_v1")
        self.assertEqual(config.max_steps, 40)
        self.assertEqual(config.max_length, 704)
        self.assertEqual(config.generation_max_new_tokens, 160)
        self.assertEqual(config.lora_targets, ("q_proj", "v_proj"))
        self.assertEqual(len(config.train_sample_schedule), 40)
        self.assertEqual(len(set(config.train_sample_schedule)), 40)

    def test_execution_target_loader_matches_release(self):
        dataset = load_execution_target_dataset(
            "../../../datasets/ultimate-distill/"
            "skill-sft-execution-target-paired-v1/dataset.json",
            "../../../datasets/ultimate-distill/"
            "skill-sft-execution-target-paired-v1/release.json",
        )
        self.assertEqual(dataset["dataset_id"], "skill-sft-execution-target-paired-v1")
        self.assertEqual(
            sum(row["split"] == "train" for row in dataset["samples"]),
            512,
        )
        self.assertEqual(
            sum(row["split"] == "validation" for row in dataset["samples"]),
            80,
        )
        self.assertEqual(
            {
                row["format_family"]
                for row in dataset["samples"]
            },
            {
                "execution_target_final",
                "process_trace_numeric",
                "skill_release_exact",
            },
        )

    def test_execution_target_loader_rejects_tampered_dataset(self):
        source = Path(
            "../../../datasets/ultimate-distill/"
            "skill-sft-execution-target-paired-v1/dataset.json"
        )
        release = Path(
            "../../../datasets/ultimate-distill/"
            "skill-sft-execution-target-paired-v1/release.json"
        )
        raw = json.loads(source.read_text(encoding="utf-8"))
        raw["samples"][0]["messages"][-1]["content"] += " "
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dataset.json"
            path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                load_execution_target_dataset(path, release)

    def test_v3_changes_only_dataset_identity_fields(self):
        v2 = load_sft_smoke_config(
            "configs/sft/format_contract_smoke_v2.json"
        )
        v3 = load_sft_smoke_config(
            "configs/sft/format_contract_smoke_v3.json"
        )
        excluded = {"experiment_id", "dataset_path", "output_dir"}
        for field in v2.__dataclass_fields__:
            if field in excluded:
                continue
            self.assertEqual(getattr(v2, field), getattr(v3, field), field)

    def test_v4_changes_only_semantic_objective_fields(self):
        v3 = load_sft_smoke_config(
            "configs/sft/format_contract_smoke_v3.json"
        )
        v4 = load_sft_smoke_config(
            "configs/sft/semantic_arithmetic_smoke_v4.json"
        )
        excluded = {
            "experiment_id",
            "dataset_path",
            "output_dir",
            "generation_max_new_tokens",
        }
        for field in v3.__dataclass_fields__:
            if field in excluded:
                continue
            self.assertEqual(getattr(v3, field), getattr(v4, field), field)
        self.assertEqual(v4.generation_max_new_tokens, 32)

    def test_v5_changes_only_step_count_and_identity(self):
        v4 = load_sft_smoke_config(
            "configs/sft/semantic_arithmetic_smoke_v4.json"
        )
        v5 = load_sft_smoke_config(
            "configs/sft/semantic_arithmetic_smoke_v5.json"
        )
        excluded = {"experiment_id", "output_dir", "max_steps"}
        for field in v4.__dataclass_fields__:
            if field in excluded:
                continue
            self.assertEqual(getattr(v4, field), getattr(v5, field), field)
        self.assertEqual(v4.max_steps, 20)
        self.assertEqual(v5.max_steps, 40)

    def test_v6_changes_only_process_objective_fields(self):
        v5 = load_sft_smoke_config(
            "configs/sft/semantic_arithmetic_smoke_v5.json"
        )
        v6 = load_sft_smoke_config(
            "configs/sft/arithmetic_process_smoke_v6.json"
        )
        excluded = {
            "experiment_id",
            "dataset_path",
            "output_dir",
            "max_length",
            "generation_max_new_tokens",
        }
        for field in v5.__dataclass_fields__:
            if field in excluded:
                continue
            self.assertEqual(getattr(v5, field), getattr(v6, field), field)
        self.assertEqual(v6.max_length, 192)
        self.assertEqual(v6.generation_max_new_tokens, 80)

    def test_v7_changes_only_mixed_safety_intervention_fields(self):
        v6 = load_sft_smoke_config(
            "configs/sft/arithmetic_process_smoke_v6.json"
        )
        v7 = load_sft_smoke_config(
            "configs/sft/hard_preservation_smoke_v7.json"
        )
        excluded = {
            "experiment_id",
            "dataset_path",
            "output_dir",
            "generation_max_new_tokens",
            "max_steps",
        }
        for field in v6.__dataclass_fields__:
            if field in excluded:
                continue
            self.assertEqual(getattr(v6, field), getattr(v7, field), field)
        self.assertEqual(v7.max_steps, 20)
        self.assertEqual(v7.generation_max_new_tokens, 128)

    def test_v8_changes_only_step_count_and_identity(self):
        v7 = load_sft_smoke_config(
            "configs/sft/hard_preservation_smoke_v7.json"
        )
        v8 = load_sft_smoke_config(
            "configs/sft/hard_preservation_smoke_v8.json"
        )
        excluded = {"experiment_id", "output_dir", "max_steps"}
        for field in v7.__dataclass_fields__:
            if field in excluded:
                continue
            self.assertEqual(getattr(v7, field), getattr(v8, field), field)
        self.assertEqual(v7.max_steps, 20)
        self.assertEqual(v8.max_steps, 40)

    def test_v9_changes_only_step_count_and_identity(self):
        v8 = load_sft_smoke_config(
            "configs/sft/hard_preservation_smoke_v8.json"
        )
        v9 = load_sft_smoke_config(
            "configs/sft/hard_preservation_smoke_v9.json"
        )
        excluded = {"experiment_id", "output_dir", "max_steps"}
        for field in v8.__dataclass_fields__:
            if field in excluded:
                continue
            self.assertEqual(getattr(v8, field), getattr(v9, field), field)
        self.assertEqual(v8.max_steps, 40)
        self.assertEqual(v9.max_steps, 30)

    def test_v10_changes_only_step_count_and_identity(self):
        v9 = load_sft_smoke_config(
            "configs/sft/hard_preservation_smoke_v9.json"
        )
        v10 = load_sft_smoke_config(
            "configs/sft/hard_preservation_smoke_v10.json"
        )
        excluded = {"experiment_id", "output_dir", "max_steps"}
        for field in v9.__dataclass_fields__:
            if field in excluded:
                continue
            self.assertEqual(getattr(v9, field), getattr(v10, field), field)
        self.assertEqual(v9.max_steps, 30)
        self.assertEqual(v10.max_steps, 32)

    def test_v11_changes_only_dataset_identity_fields(self):
        v10 = load_sft_smoke_config(
            "configs/sft/hard_preservation_smoke_v10.json"
        )
        v11 = load_sft_smoke_config(
            "configs/sft/targeted_preservation_smoke_v11.json"
        )
        excluded = {"experiment_id", "dataset_path", "output_dir"}
        for field in v10.__dataclass_fields__:
            if field in excluded:
                continue
            self.assertEqual(getattr(v10, field), getattr(v11, field), field)

    def test_v12_changes_only_dataset_identity_fields(self):
        v11 = load_sft_smoke_config(
            "configs/sft/targeted_preservation_smoke_v11.json"
        )
        v12 = load_sft_smoke_config(
            "configs/sft/failure_targeted_preservation_smoke_v12.json"
        )
        excluded = {"experiment_id", "dataset_path", "output_dir"}
        for field in v11.__dataclass_fields__:
            if field in excluded:
                continue
            self.assertEqual(getattr(v11, field), getattr(v12, field), field)

    def test_v13_changes_only_dataset_identity_fields(self):
        v11 = load_sft_smoke_config(
            "configs/sft/targeted_preservation_smoke_v11.json"
        )
        v13 = load_sft_smoke_config(
            "configs/sft/percentage_isolation_preservation_smoke_v13.json"
        )
        excluded = {"experiment_id", "dataset_path", "output_dir"}
        for field in v11.__dataclass_fields__:
            if field in excluded:
                continue
            self.assertEqual(getattr(v11, field), getattr(v13, field), field)

    def test_v14_changes_only_dataset_identity_fields(self):
        v11 = load_sft_smoke_config(
            "configs/sft/targeted_preservation_smoke_v11.json"
        )
        v14 = load_sft_smoke_config(
            "configs/sft/packing_isolation_preservation_smoke_v14.json"
        )
        excluded = {"experiment_id", "dataset_path", "output_dir"}
        for field in v11.__dataclass_fields__:
            if field not in excluded:
                self.assertEqual(getattr(v11, field), getattr(v14, field), field)

    def test_v15_changes_only_dataset_identity_fields(self):
        v11 = load_sft_smoke_config(
            "configs/sft/targeted_preservation_smoke_v11.json"
        )
        v15 = load_sft_smoke_config(
            "configs/sft/schedule_isolation_preservation_smoke_v15.json"
        )
        excluded = {"experiment_id", "dataset_path", "output_dir"}
        for field in v11.__dataclass_fields__:
            if field not in excluded:
                self.assertEqual(getattr(v11, field), getattr(v15, field), field)

    def test_tokenize_masks_prompt_and_keeps_assistant(self):
        dataset = {
            "samples": [
                {
                    "sample_id": "synthetic-a",
                    "split": "train",
                    "format_family": "final_choice",
                    "messages": [
                        {"role": "system", "content": "system"},
                        {"role": "user", "content": "question"},
                        {"role": "assistant", "content": "FINAL: A"},
                    ],
                }
            ]
        }
        tokenizer = FakeTokenizer()
        sample = tokenize_samples(dataset, tokenizer, max_length=256)[0]
        self.assertTrue(all(value == -100 for value in sample.labels[: len(sample.prompt_ids)]))
        self.assertTrue(all(value != -100 for value in sample.labels[len(sample.prompt_ids) :]))
        self.assertEqual(tokenizer.assertions, (False, True, False))
        self.assertEqual(sample.task_family, "")

    def test_release_loader_selects_stratified_train_and_dev(self):
        families = ["alpha", "beta"]
        rows = []
        for family in families:
            for split, count in (("train", 3), ("dev", 2)):
                for index in range(count):
                    sample_id = f"{family}-{split}-{index}"
                    messages = [
                        {"role": "system", "content": "system"},
                        {"role": "user", "content": sample_id},
                        {"role": "assistant", "content": "FINAL: ok"},
                    ]
                    rows.append(
                        {
                            "schema_version": "nano_skill_sft_sample_v1",
                            "sample_id": sample_id,
                            "split": split,
                            "family_id": family,
                            "messages": messages,
                            "verifier": {"kind": "exact"},
                            "exact_hash": f"exact-{sample_id}",
                            "semantic_hash": f"semantic-{sample_id}",
                        }
                    )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_path = root / "accepted.jsonl"
            dataset_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            import hashlib

            digest = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
            release_path = root / "release.json"
            release_path.write_text(
                json.dumps(
                    {
                        "schema_version": "nano_skill_sft_release_v1",
                        "release_id": "test-release",
                        "training_unblocked": True,
                        "checks": {"all_pass": True},
                        "artifacts": {
                            "accepted_jsonl_sha256": digest,
                        },
                    }
                ),
                encoding="utf-8",
            )

            dataset = load_skill_release_dataset(
                dataset_path,
                release_path,
                train_samples_per_family=2,
                validation_samples_per_family=1,
                validation_start_per_family=1,
            )

        self.assertEqual(dataset["dataset_id"], "test-release")
        self.assertEqual(
            [row["split"] for row in dataset["samples"]].count("train"),
            4,
        )
        self.assertEqual(
            [row["split"] for row in dataset["samples"]].count("validation"),
            2,
        )
        self.assertEqual(
            {row["task_family"] for row in dataset["samples"]},
            set(families),
        )
        self.assertEqual(
            {
                row["sample_id"]
                for row in dataset["samples"]
                if row["split"] == "validation"
            },
            {"alpha-dev-1", "beta-dev-1"},
        )

    def test_release_loader_rejects_blocked_or_tampered_release(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_path = root / "accepted.jsonl"
            dataset_path.write_text("{}\n", encoding="utf-8")
            release_path = root / "release.json"
            release = {
                "schema_version": "nano_skill_sft_release_v1",
                "release_id": "blocked",
                "training_unblocked": False,
                "checks": {"all_pass": False},
                "artifacts": {"accepted_jsonl_sha256": "0" * 64},
            }
            release_path.write_text(json.dumps(release), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not training-unblocked"):
                load_skill_release_dataset(
                    dataset_path,
                    release_path,
                    train_samples_per_family=1,
                    validation_samples_per_family=1,
                )
            release["training_unblocked"] = True
            release["checks"] = {"all_pass": True}
            release_path.write_text(json.dumps(release), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                load_skill_release_dataset(
                    dataset_path,
                    release_path,
                    train_samples_per_family=1,
                    validation_samples_per_family=1,
                )

    def test_skill_release_semantic_scorers_accept_equivalent_outputs(self):
        cases = [
            (
                "verified-reasoning",
                {"expression": "(3 + 4) * 2"},
                {"kind": "safe_execution_receipt_v1"},
                "FINAL: 14",
            ),
            (
                "tool-use-and-recovery",
                {
                    "required_calls": [
                        {
                            "name": "lookup",
                            "arguments": {"key": "x"},
                            "status": "error",
                        }
                    ]
                },
                {"kind": "tool_trace_contract_v1"},
                json.dumps(
                    {
                        "final_status": "verified",
                        "tool_calls": [
                            {
                                "status": "error",
                                "arguments": {"key": "x"},
                                "name": "lookup",
                            }
                        ],
                    }
                ),
            ),
            (
                "planning-and-state",
                {
                    "constraints": ["a"],
                    "evidence": ["b"],
                    "pending": ["c"],
                    "stop": False,
                },
                {"kind": "state_plan_consistency_v1"},
                json.dumps(
                    {
                        "stop": False,
                        "pending": ["c"],
                        "evidence": ["b"],
                        "constraints": ["a"],
                    }
                ),
            ),
            (
                "coding-and-validation",
                {
                    "file": "x.py",
                    "original_content": "x = 1\n",
                    "expected_content": "x = 2\n",
                    "test_command": "python -m unittest x",
                },
                {"kind": "patch_test_receipt_v1"},
                json.dumps(
                    {
                        "test_status": "passed",
                        "test_command": "python -m unittest x",
                        "after_content": "x = 2\n",
                        "before_sha256": __import__("hashlib")
                        .sha256(b"x = 1\n")
                        .hexdigest(),
                        "file": "x.py",
                    }
                ),
            ),
            (
                "skill-routing-and-reflection",
                {
                    "request_tags": ["data"],
                    "skills": [
                        {"skill_id": "broad", "tags": ["data", "train"]},
                        {"skill_id": "minimal", "tags": ["data"]},
                    ],
                },
                {"kind": "skill_route_receipt_v1"},
                json.dumps(
                    {
                        "steps": ["validate"],
                        "selected_skill": "minimal",
                    }
                ),
            ),
        ]
        for family, task_spec, verifier, output in cases:
            self.assertTrue(
                skill_release_output_valid(
                    family,
                    task_spec,
                    verifier,
                    output,
                ),
                family,
            )

    def test_skill_release_semantic_scorers_reject_wrong_outputs(self):
        self.assertFalse(
            skill_release_output_valid(
                "verified-reasoning",
                {"expression": "3 + 4"},
                {"kind": "safe_execution_receipt_v1"},
                "FINAL: 8",
            )
        )
        self.assertFalse(
            skill_release_output_valid(
                "planning-and-state",
                {
                    "constraints": ["a"],
                    "evidence": ["b"],
                    "pending": ["c"],
                    "stop": False,
                },
                {"kind": "state_plan_consistency_v1"},
                json.dumps(
                    {
                        "constraints": ["a"],
                        "evidence": ["b"],
                        "pending": [],
                        "stop": False,
                    }
                ),
            )
        )

    def test_evaluate_exact_reports_family_metrics(self):
        class FakeModel:
            def eval(self):
                return None

            def generate(self, *, input_ids, **kwargs):
                return torch.cat(
                    [input_ids, torch.tensor([[9]], device=input_ids.device)],
                    dim=1,
                )

        class DecodeTokenizer:
            eos_token_id = 2
            pad_token_id = 0

            def decode(self, token_ids, *, skip_special_tokens):
                return "ok"

        samples = [
            TokenizedSample(
                "a",
                "validation",
                [1, 9],
                [-100, 9],
                [1],
                "ok",
                "final_numeric",
                None,
                "family_a",
            ),
            TokenizedSample(
                "b",
                "validation",
                [1, 9],
                [-100, 9],
                [1],
                "ok",
                "final_numeric",
                None,
                "family_b",
            ),
        ]
        metrics, rows = evaluate_exact(
            FakeModel(),
            DecodeTokenizer(),
            samples,
            device=torch.device("cpu"),
            max_new_tokens=1,
        )
        self.assertEqual(metrics["exact"], 2)
        self.assertEqual(metrics["semantic_exact"], 2)
        self.assertEqual(
            metrics["by_family"],
            {
                "family_a": {
                    "samples": 1,
                    "exact": 1,
                    "semantic_exact": 1,
                    "exact_failure_sample_ids": [],
                    "semantic_failure_sample_ids": [],
                },
                "family_b": {
                    "samples": 1,
                    "exact": 1,
                    "semantic_exact": 1,
                    "exact_failure_sample_ids": [],
                    "semantic_failure_sample_ids": [],
                },
            },
        )
        self.assertEqual(
            [row["task_family"] for row in rows],
            ["family_a", "family_b"],
        )

    def test_collator_masks_padding(self):
        first = TokenizedSample(
            "a", "train", [1, 2], [-100, 2], [1], "x", "final_numeric", None
        )
        second = TokenizedSample(
            "b", "train", [3], [3], [], "y", "final_numeric", None
        )
        batch = collate_samples([first, second], pad_token_id=0)
        self.assertEqual(batch["input_ids"].tolist(), [[1, 2], [3, 0]])
        self.assertEqual(batch["labels"].tolist(), [[-100, 2], [3, -100]])
        self.assertEqual(batch["attention_mask"].tolist(), [[1, 1], [1, 0]])

    def test_batch_order_and_scheduler_are_deterministic(self):
        samples = [
            TokenizedSample(
                str(index),
                "train",
                [index],
                [index],
                [],
                str(index),
                "final_numeric",
                None,
            )
            for index in range(8)
        ]
        self.assertEqual(_batch_order(samples, 7), _batch_order(samples, 7))
        self.assertNotEqual(_batch_order(samples, 7), _batch_order(samples, 8))
        self.assertEqual(_scheduler_scale(0, 2, 20), 0.5)
        self.assertEqual(_scheduler_scale(1, 2, 20), 1.0)
        self.assertGreater(_scheduler_scale(2, 2, 20), _scheduler_scale(19, 2, 20))

    def test_family_schedule_is_deterministic_and_exact(self):
        samples = [
            TokenizedSample(
                f"{family}-{index}",
                "train",
                [index],
                [index],
                [],
                str(index),
                "final_numeric",
                None,
                family,
            )
            for family in ("reasoning", "json")
            for index in range(3)
        ]
        schedule = ("reasoning", "json", "reasoning", "json")
        first = _scheduled_batch_order(samples, 7, schedule)
        second = _scheduled_batch_order(samples, 7, schedule)
        self.assertEqual(first, second)
        self.assertEqual(
            [samples[index].task_family for index in first],
            list(schedule),
        )
        with self.assertRaisesRegex(ValueError, "missing"):
            _scheduled_batch_order(samples, 7, ("missing",))

    def test_sample_schedule_is_exact_and_rejects_missing(self):
        samples = [
            TokenizedSample(
                f"sample-{index}",
                "train",
                [index],
                [index],
                [],
                str(index),
                "final_numeric",
                None,
            )
            for index in range(4)
        ]
        schedule = ("sample-2", "sample-0", "sample-3")
        order = _sample_scheduled_batch_order(samples, schedule)
        self.assertEqual(order, [2, 0, 3])
        with self.assertRaisesRegex(ValueError, "missing"):
            _sample_scheduled_batch_order(samples, ("missing",))

    def test_execution_target_scorer_checks_process_final_and_json(self):
        self.assertTrue(
            execution_target_output_valid(
                "execution-target-final",
                "final",
                {"expression": "(20 + 4) * 2 - 4", "view": "final"},
                {"kind": "safe_execution_receipt_v1"},
                "FINAL: 44",
            )
        )
        self.assertFalse(
            execution_target_output_valid(
                "execution-target-final",
                "final",
                {"expression": "(20 + 4) * 2 - 4", "view": "final"},
                {"kind": "safe_execution_receipt_v1"},
                "FINAL: 48",
            )
        )
        self.assertTrue(
            execution_target_output_valid(
                "execution-target-process",
                "process",
                {
                    "expression": "(20 + 4) * 2 - 4",
                    "view": "process",
                },
                {
                    "kind": "safe_ast_arithmetic_process_v2",
                    "source_expression": "(20 + 4) * 2 - 4",
                    "steps": [
                        {"expression": "20 + 4", "expected_result": "24"},
                        {"expression": "24 * 2", "expected_result": "48"},
                        {"expression": "48 - 4", "expected_result": "44"},
                    ],
                    "expected_result": "44",
                },
                (
                    "STEP 1: 20 + 4 = 24\n"
                    "STEP 2: 24 * 2 = 48\n"
                    "STEP 3: 48 - 4 = 44\n"
                    "FINAL: 44"
                ),
            )
        )

    def test_nonfinite_guards_and_failure_receipt(self):
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        parameter.grad = torch.tensor([float("nan")])
        with self.assertRaisesRegex(FloatingPointError, "gradients"):
            _assert_finite_gradients([parameter], step=2)
        with self.assertRaisesRegex(FloatingPointError, "loss"):
            _assert_finite_loss(torch.tensor(float("nan")), step=2)
        parameter.data.fill_(float("inf"))
        with self.assertRaisesRegex(FloatingPointError, "parameters"):
            _assert_finite_parameters([parameter], step=2)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_failure(
                root,
                step=2,
                stage="gradient",
                error=FloatingPointError("bad gradient"),
            )
            receipt = json.loads((root / "failure.json").read_text())
        self.assertEqual(receipt["optimizer_step"], 2)
        self.assertEqual(receipt["stage"], "gradient")
        self.assertFalse(receipt["adapter_saved"])
        self.assertFalse(receipt["post_validation_run"])

    def test_semantic_trace_accepts_equivalent_safe_expression(self):
        sample = TokenizedSample(
            "trace",
            "validation",
            [1],
            [1],
            [],
            "CALC: 7 + 5 * 3 = 22\nFINAL: 22",
            "trace_numeric",
            {
                "kind": "safe_ast_arithmetic_v1",
                "expression": "7 + 5 * 3",
                "expected_result": "22",
            },
        )
        self.assertTrue(
            semantic_output_valid(
                sample,
                "CALC: (5 * 3) + 7 = 22\nFINAL: 22",
            )
        )
        self.assertFalse(
            semantic_output_valid(
                sample,
                "CALC: (5 * 3) + 7 = 21\nFINAL: 21",
            )
        )
        self.assertFalse(
            semantic_output_valid(
                sample,
                "CALC: __import__('os').system('id') = 22\nFINAL: 22",
            )
        )

    def test_process_trace_verifies_every_intermediate_step(self):
        sample = TokenizedSample(
            "process",
            "validation",
            [1],
            [1],
            [],
            (
                "STEP 1: 5 * 3 = 15\n"
                "STEP 2: 7 + 15 = 22\n"
                "FINAL: 22"
            ),
            "process_trace_numeric",
            {
                "kind": "safe_ast_arithmetic_process_v2",
                "source_expression": "7 + 5 * 3",
                "steps": [
                    {"expression": "5 * 3", "expected_result": "15"},
                    {"expression": "7 + 15", "expected_result": "22"},
                ],
                "expected_result": "22",
            },
        )
        self.assertTrue(
            semantic_output_valid(
                sample,
                (
                    "STEP 1: (5 * 3) = 15\n"
                    "STEP 2: (7 + 15) = 22\n"
                    "FINAL: 22"
                ),
            )
        )
        self.assertFalse(
            semantic_output_valid(
                sample,
                (
                    "STEP 1: 5 * 3 = 14\n"
                    "STEP 2: 7 + 14 = 21\n"
                    "FINAL: 21"
                ),
            )
        )
        self.assertFalse(
            semantic_output_valid(
                sample,
                (
                    "STEP 1: 5 * 3 = 15\n"
                    "STEP 2: 8 + 14 = 22\n"
                    "FINAL: 22"
                ),
            )
        )
        self.assertFalse(
            semantic_output_valid(
                sample,
                (
                    "STEP 1: __import__('os').system('id') = 15\n"
                    "STEP 2: 7 + 15 = 22\n"
                    "FINAL: 22"
                ),
            )
        )

    def test_reasoning_numeric_accepts_safe_equivalent_work(self):
        sample = TokenizedSample(
            "reasoning",
            "validation",
            [1],
            [1],
            [],
            "WORK: 1 + 12 + 12 * 2 = 37\nFINAL: 37",
            "reasoning_numeric",
            {
                "kind": "safe_ast_reasoning_numeric_v1",
                "expression": "1 + 12 + 12 * 2",
                "expected_result": "37",
            },
        )
        self.assertTrue(
            semantic_output_valid(
                sample,
                "WORK: 12 * 2 + 12 + 1 = 37\nFINAL: 37",
            )
        )
        self.assertFalse(
            semantic_output_valid(
                sample,
                "WORK: 12 * 2 + 12 = 36\nFINAL: 36",
            )
        )
        self.assertFalse(
            semantic_output_valid(
                sample,
                "WORK: __import__('os').system('id') = 37\nFINAL: 37",
            )
        )
        self.assertFalse(
            semantic_output_valid(
                sample,
                "WORK: 12 * 2 + 12 + 1 = 37 FINAL: 37",
            )
        )

    def test_v5_budget_audit_identity_and_contract(self):
        config = load_audit_config(
            Path(
                "configs/audits/"
                "semantic_arithmetic_v5_budget_audit_v1.json"
            )
        )
        identity = verify_audit_identity(config)
        self.assertEqual(identity, config["expected"])
        validation = [
            TokenizedSample(
                "case-a",
                "validation",
                [1],
                [-100, 1, 2, 3],
                [1],
                "target",
                "trace_numeric",
                {
                    "kind": "safe_ast_arithmetic_v1",
                    "expression": "1 + 1",
                    "expected_result": "2",
                },
            )
        ]
        tokenizer = mock.Mock()
        tokenizer.return_value = SimpleNamespace(input_ids=[1, 2, 3])
        contract = validate_audit_contract(
            config,
            tokenizer,
            validation,
            {"post_sft": [{"sample_id": "case-a"}]},
        )
        self.assertTrue(contract["source_case_set_matches"])
        self.assertTrue(contract["budget_above_target_with_eos_max"])

        short_budget = {**config, "generation_max_new_tokens": 3}
        with self.assertRaisesRegex(ValueError, "must exceed"):
            validate_audit_contract(
                short_budget,
                tokenizer,
                validation,
                {"post_sft": [{"sample_id": "case-a"}]},
            )

        with self.assertRaisesRegex(ValueError, "case sets differ"):
            validate_audit_contract(
                config,
                tokenizer,
                validation,
                {"post_sft": [{"sample_id": "different-case"}]},
            )


if __name__ == "__main__":
    unittest.main()
