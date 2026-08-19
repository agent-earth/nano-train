from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as functional
from peft import LoraConfig, get_peft_model
from torch.optim import AdamW
from transformers import AutoTokenizer, Qwen3_5ForCausalLM

from nano_train.data import (
    TokenizedSample,
    collate_samples,
    load_execution_target_dataset,
    tokenize_samples,
)
from nano_train.sft import (
    _assert_finite_gradients,
    _assert_finite_loss,
    _assert_finite_parameters,
    _scheduler_scale,
    _trainable_parameters,
    _write_failure,
    dependency_versions,
    evaluate_exact,
    set_seed,
    sha256_file,
    sha256_tree,
)


JSON_FAMILIES = (
    "coding-and-validation",
    "planning-and-state",
    "skill-routing-and-reflection",
    "tool-use-and-recovery",
)

FROZEN_METHOD_FIELDS = (
    "model_config_sha256",
    "seed",
    "dtype",
    "max_length",
    "max_steps",
    "learning_rate",
    "weight_decay",
    "warmup_steps",
    "lora_r",
    "lora_alpha",
    "lora_dropout",
    "lora_targets",
    "generation_max_new_tokens",
    "gradient_checkpointing",
    "process_ce_weight",
    "final_ce_weight",
    "consistency_weight",
    "consistency_temperature",
    "teacher_detach",
    "train_pair_count",
    "train_json_per_family",
)


@dataclass(frozen=True)
class PairedConsistencyConfig:
    schema_version: str
    experiment_id: str
    model_path: str
    model_config_sha256: str
    dataset_path: str
    dataset_file_sha256: str
    dataset_canonical_sha256: str
    release_manifest_path: str
    release_manifest_sha256: str
    prior_standard_config_path: str
    prior_standard_config_sha256: str
    output_dir: str
    seed: int
    dtype: str
    max_length: int
    max_steps: int
    learning_rate: float
    weight_decay: float
    warmup_steps: int
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    lora_targets: tuple[str, ...]
    generation_max_new_tokens: int
    gradient_checkpointing: bool
    process_ce_weight: float
    final_ce_weight: float
    consistency_weight: float
    consistency_temperature: float
    teacher_detach: bool
    heldout_pair_count: int
    train_pair_offset: int
    train_pair_count: int
    heldout_json_per_family: int
    train_json_offset: int
    train_json_per_family: int
    expected_heldout_sample_id_sha256: str
    expected_pair_ids_sha256: str
    expected_json_ids_sha256: str
    expected_prior_train_ids_sha256: str
    expected_source_dev_ids_sha256: str
    expected_train_pair_count: int = 20
    expected_heldout_pair_count: int = 24
    expected_train_json_per_family: int = 5
    expected_heldout_json_per_family: int = 8
    bootstrap_samples: int = 10_000
    aggregate_bootstrap_seed: str = "paired-consistency-v1-all"
    final_bootstrap_seed: str = "paired-consistency-v1-final"
    pair_bootstrap_seed: str = "paired-consistency-v1-pair"
    json_bootstrap_seed: str = "paired-consistency-v1-json"
    mcnemar_alpha: float = 0.05
    require_ci_lower_positive: bool = False
    minimum_final_only_wins: int = 1
    maximum_final_only_losses: int = 0


def load_config(path: str | Path) -> PairedConsistencyConfig:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = set(PairedConsistencyConfig.__dataclass_fields__)
    optional = {
        "expected_train_pair_count",
        "expected_heldout_pair_count",
        "expected_train_json_per_family",
        "expected_heldout_json_per_family",
        "bootstrap_samples",
        "aggregate_bootstrap_seed",
        "final_bootstrap_seed",
        "pair_bootstrap_seed",
        "json_bootstrap_seed",
        "mcnemar_alpha",
        "require_ci_lower_positive",
        "minimum_final_only_wins",
        "maximum_final_only_losses",
    }
    if set(raw) - expected or expected - set(raw) - optional:
        raise ValueError("paired consistency config fields differ")
    raw["lora_targets"] = tuple(raw["lora_targets"])
    config = PairedConsistencyConfig(**raw)
    validate_config(config)
    return config


def validate_config(config: PairedConsistencyConfig) -> None:
    if config.schema_version not in {
        "nano_train_paired_consistency_v1",
        "nano_train_paired_consistency_v2",
    }:
        raise ValueError("unsupported paired consistency schema")
    expected = {
        "dtype": "float32",
        "max_length": 704,
        "max_steps": 40,
        "learning_rate": 0.00005,
        "weight_decay": 0.0,
        "warmup_steps": 1,
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.0,
        "lora_targets": ("q_proj", "v_proj"),
        "generation_max_new_tokens": 160,
        "gradient_checkpointing": True,
        "process_ce_weight": 0.5,
        "final_ce_weight": 0.5,
        "consistency_weight": 1.0,
        "consistency_temperature": 1.0,
        "teacher_detach": True,
    }
    for field, expected_value in expected.items():
        if getattr(config, field) != expected_value:
            raise ValueError(
                f"paired consistency freezes {field}={expected_value}"
            )
    for field in (
        "model_config_sha256",
        "dataset_file_sha256",
        "dataset_canonical_sha256",
        "release_manifest_sha256",
        "prior_standard_config_sha256",
        "expected_heldout_sample_id_sha256",
        "expected_pair_ids_sha256",
        "expected_json_ids_sha256",
        "expected_prior_train_ids_sha256",
        "expected_source_dev_ids_sha256",
    ):
        value = getattr(config, field)
        if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise ValueError(f"paired consistency {field} is not SHA256")
    if config.max_steps != config.train_pair_count + (
        len(JSON_FAMILIES) * config.train_json_per_family
    ):
        raise ValueError("paired consistency step composition differs")
    if config.train_pair_count != config.expected_train_pair_count:
        raise ValueError("paired consistency train pair count differs")
    if config.heldout_pair_count != config.expected_heldout_pair_count:
        raise ValueError("paired consistency heldout pair count differs")
    if (
        config.train_json_per_family
        != config.expected_train_json_per_family
    ):
        raise ValueError("paired consistency train JSON count differs")
    if (
        config.heldout_json_per_family
        != config.expected_heldout_json_per_family
    ):
        raise ValueError("paired consistency heldout JSON count differs")
    if config.schema_version == "nano_train_paired_consistency_v1":
        expected_v1 = {
            "heldout_pair_count": 24,
            "train_pair_offset": 24,
            "train_pair_count": 20,
            "heldout_json_per_family": 8,
            "train_json_offset": 8,
            "train_json_per_family": 5,
        }
        for field, expected_value in expected_v1.items():
            if getattr(config, field) != expected_value:
                raise ValueError(
                    f"paired consistency v1 freezes {field}={expected_value}"
                )
    else:
        expected_v2 = {
            "heldout_pair_count": 192,
            "train_pair_offset": 0,
            "train_pair_count": 20,
            "heldout_json_per_family": 32,
            "train_json_offset": 0,
            "train_json_per_family": 5,
            "bootstrap_samples": 10_000,
            "aggregate_bootstrap_seed": (
                "paired-consistency-replication-v1-all"
            ),
            "final_bootstrap_seed": (
                "paired-consistency-replication-v1-final"
            ),
            "pair_bootstrap_seed": (
                "paired-consistency-replication-v1-pair"
            ),
            "json_bootstrap_seed": (
                "paired-consistency-replication-v1-json"
            ),
            "mcnemar_alpha": 0.05,
            "require_ci_lower_positive": True,
            "minimum_final_only_wins": 6,
            "maximum_final_only_losses": 0,
        }
        for field, expected_value in expected_v2.items():
            if getattr(config, field) != expected_value:
                raise ValueError(
                    f"paired consistency v2 freezes {field}={expected_value}"
                )


def _sha256_lines(values: list[str]) -> str:
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def frozen_method_contract(
    config: PairedConsistencyConfig,
) -> dict[str, Any]:
    result = {
        field: getattr(config, field)
        for field in FROZEN_METHOD_FIELDS
    }
    result["lora_targets"] = list(config.lora_targets)
    return result


def load_replication_dataset(
    dataset_path: Path,
    release_path: Path,
) -> dict[str, Any]:
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    release = json.loads(release_path.read_text(encoding="utf-8"))
    if (
        dataset.get("schema_version")
        != "nano_consistency_replication_dataset_v1"
        or release.get("schema_version")
        != "nano_consistency_replication_release_v1"
    ):
        raise ValueError("unsupported consistency replication release")
    if release.get("training_unblocked") is not True:
        raise ValueError("consistency replication release is blocked")
    checks = release.get("checks")
    if not isinstance(checks, dict) or not checks or not all(checks.values()):
        raise ValueError("consistency replication release checks differ")
    policy = dataset.get("policy", {})
    if (
        policy.get("contains_benchmark_content") is not False
        or policy.get("contains_independent_holdout") is not False
        or policy.get("dev_training_allowed") is not False
        or policy.get("training_allowed_after_release_gate") is not True
    ):
        raise ValueError("consistency replication dataset policy differs")
    canonical_sha = hashlib.sha256(
        json.dumps(
            dataset,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    if canonical_sha != release.get("source", {}).get(
        "dataset_canonical_sha256"
    ):
        raise ValueError("consistency replication dataset identity mismatch")
    samples = []
    for row in dataset.get("samples", []):
        split = row.get("split")
        if split not in {"train", "dev"}:
            raise ValueError("consistency replication split differs")
        if row.get("training_eligible") is not (split == "train"):
            raise ValueError(
                "consistency replication training eligibility differs"
            )
        view = row.get("view")
        format_family = {
            "process": "process_trace_numeric",
            "final": "execution_target_final",
            "json_preservation": "skill_release_exact",
        }.get(view)
        if format_family is None:
            raise ValueError("consistency replication view differs")
        samples.append(
            {
                "sample_id": row["sample_id"],
                "split": "validation" if split == "dev" else "train",
                "task_family": row["task_family"],
                "format_family": format_family,
                "generation_rule": "paired_consistency_replication_v1",
                "training_eligible": split == "train",
                "messages": row["messages"],
                "verifier": row["verifier"],
                "task_spec": row["task_spec"],
            }
        )
    if (
        sum(row["split"] == "train" for row in samples)
        != release["accepted"]["train_rows"]
        or sum(row["split"] == "validation" for row in samples)
        != release["accepted"]["dev_rows"]
    ):
        raise ValueError("consistency replication release counts differ")
    return {
        "schema_version": "nano_analog_dataset_v1",
        "dataset_id": release["release_id"],
        "policy": {
            "source_split": "non_eval_analog_only",
            "training_allowed": True,
            "contains_benchmark_content": False,
        },
        "release": {
            "path": str(release_path),
            "sha256": sha256_file(release_path),
            "dataset_canonical_sha256": canonical_sha,
            "dataset_file_sha256": sha256_file(dataset_path),
        },
        "samples": samples,
    }


def build_selection_contract(
    config: PairedConsistencyConfig,
) -> dict[str, Any]:
    dataset_path = Path(config.dataset_path)
    release_path = Path(config.release_manifest_path)
    prior_config_path = Path(config.prior_standard_config_path)
    if sha256_file(dataset_path) != config.dataset_file_sha256:
        raise ValueError("paired consistency dataset file identity mismatch")
    if sha256_file(release_path) != config.release_manifest_sha256:
        raise ValueError("paired consistency release identity mismatch")
    if sha256_file(prior_config_path) != config.prior_standard_config_sha256:
        raise ValueError("paired consistency prior config identity mismatch")
    if (
        sha256_file(Path(config.model_path) / "config.json")
        != config.model_config_sha256
    ):
        raise ValueError("paired consistency model identity mismatch")

    if config.schema_version == "nano_train_paired_consistency_v1":
        dataset = load_execution_target_dataset(dataset_path, release_path)
    else:
        dataset = load_replication_dataset(dataset_path, release_path)
    if (
        dataset["release"]["dataset_file_sha256"]
        != config.dataset_file_sha256
        or dataset["release"]["dataset_canonical_sha256"]
        != config.dataset_canonical_sha256
        or dataset["release"]["sha256"]
        != config.release_manifest_sha256
    ):
        raise ValueError("paired consistency loaded dataset identity mismatch")
    raw = json.loads(dataset_path.read_text(encoding="utf-8"))
    raw_by_id = {row["sample_id"]: row for row in raw["samples"]}
    if len(raw_by_id) != len(raw["samples"]):
        raise ValueError("paired consistency sample IDs are not unique")
    prior_config = json.loads(prior_config_path.read_text(encoding="utf-8"))
    prior_train_ids = sorted(prior_config.get("train_sample_schedule", []))
    prior_train = set(prior_train_ids)
    source_dev_ids = sorted(
        row["sample_id"] for row in raw["samples"] if row["split"] == "dev"
    )
    source_dev = set(source_dev_ids)

    train_pairs: dict[str, dict[str, str]] = {}
    dev_pairs: dict[str, dict[str, str]] = {}
    train_json_rows = {family: [] for family in JSON_FAMILIES}
    dev_json_rows = {family: [] for family in JSON_FAMILIES}
    for row in raw["samples"]:
        pair_buckets = train_pairs if row["split"] == "train" else dev_pairs
        json_buckets = (
            train_json_rows if row["split"] == "train" else dev_json_rows
        )
        if row.get("pair_id"):
            pair_buckets.setdefault(row["pair_id"], {})[row["view"]] = row[
                "sample_id"
            ]
        elif row["task_family"] in json_buckets:
            json_buckets[row["task_family"]].append(row["sample_id"])
    for sample_ids in [
        *train_json_rows.values(),
        *dev_json_rows.values(),
    ]:
        sample_ids.sort()
    for split_name, pairs in (("train", train_pairs), ("dev", dev_pairs)):
        incomplete = [
            pair_id
            for pair_id, views in pairs.items()
            if set(views) != {"process", "final"}
        ]
        if incomplete:
            raise ValueError(
                f"paired consistency {split_name} pairs are incomplete"
            )
    if config.schema_version == "nano_train_paired_consistency_v1":
        available_pairs = [
            pair_id
            for pair_id in sorted(train_pairs)
            if not (set(train_pairs[pair_id].values()) & prior_train)
        ]
        heldout_pair_ids = available_pairs[: config.heldout_pair_count]
        selected_train_pair_ids = available_pairs[
            config.train_pair_offset : (
                config.train_pair_offset + config.train_pair_count
            )
        ]
        heldout_json = {
            family: [
                sample_id
                for sample_id in train_json_rows[family]
                if sample_id not in prior_train
            ][: config.heldout_json_per_family]
            for family in JSON_FAMILIES
        }
        train_json = {
            family: [
                sample_id
                for sample_id in train_json_rows[family]
                if sample_id not in prior_train
            ][
                config.train_json_offset : (
                    config.train_json_offset + config.train_json_per_family
                )
            ]
            for family in JSON_FAMILIES
        }
        heldout_pair_source = train_pairs
    else:
        heldout_pair_ids = sorted(dev_pairs)
        selected_train_pair_ids = sorted(train_pairs)[
            config.train_pair_offset : (
                config.train_pair_offset + config.train_pair_count
            )
        ]
        heldout_json = {
            family: list(dev_json_rows[family])
            for family in JSON_FAMILIES
        }
        train_json = {
            family: train_json_rows[family][
                config.train_json_offset : (
                    config.train_json_offset + config.train_json_per_family
                )
            ]
            for family in JSON_FAMILIES
        }
        heldout_pair_source = dev_pairs
    pair_schedule = [
        {
            "pair_id": pair_id,
            "process_sample_id": train_pairs[pair_id]["process"],
            "final_sample_id": train_pairs[pair_id]["final"],
        }
        for pair_id in selected_train_pair_ids
    ]
    heldout_pair_schedule = [
        {
            "pair_id": pair_id,
            "process_sample_id": heldout_pair_source[pair_id]["process"],
            "final_sample_id": heldout_pair_source[pair_id]["final"],
        }
        for pair_id in heldout_pair_ids
    ]
    json_schedule = [
        sample_id
        for family in JSON_FAMILIES
        for sample_id in train_json[family]
    ]
    heldout_sample_ids = [
        heldout_pair_source[pair_id][view]
        for pair_id in heldout_pair_ids
        for view in ("process", "final")
    ] + [
        sample_id
        for family in JSON_FAMILIES
        for sample_id in heldout_json[family]
    ]
    if (
        len(pair_schedule) != config.train_pair_count
        or len(json_schedule)
        != len(JSON_FAMILIES) * config.train_json_per_family
        or len(heldout_sample_ids)
        != config.heldout_pair_count * 2
        + len(JSON_FAMILIES) * config.heldout_json_per_family
    ):
        raise ValueError("paired consistency selection counts differ")
    selected_train_ids = {
        row[key]
        for row in pair_schedule
        for key in ("process_sample_id", "final_sample_id")
    } | set(json_schedule)
    heldout_ids = set(heldout_sample_ids)
    if (
        selected_train_ids & prior_train
        or heldout_ids & prior_train
        or heldout_ids & selected_train_ids
        or (
            config.schema_version == "nano_train_paired_consistency_v1"
            and heldout_ids & source_dev
        )
    ):
        raise ValueError("paired consistency selection overlaps prior evidence")
    if any(
        raw_by_id[sample_id]["split"] != "train"
        for sample_id in selected_train_ids
    ):
        raise ValueError("paired consistency train selection is not train")
    expected_heldout_split = (
        "train"
        if config.schema_version == "nano_train_paired_consistency_v1"
        else "dev"
    )
    if any(
        raw_by_id[sample_id]["split"] != expected_heldout_split
        for sample_id in heldout_ids
    ):
        raise ValueError("paired consistency heldout split differs")

    hashes = {
        "heldout_sample_id_sha256": _sha256_lines(
            sorted(heldout_sample_ids)
        ),
        "pair_ids_sha256": _sha256_lines(selected_train_pair_ids),
        "json_ids_sha256": _sha256_lines(json_schedule),
        "prior_train_ids_sha256": _sha256_lines(prior_train_ids),
        "source_dev_ids_sha256": _sha256_lines(source_dev_ids),
    }
    expected_hashes = {
        "heldout_sample_id_sha256": config.expected_heldout_sample_id_sha256,
        "pair_ids_sha256": config.expected_pair_ids_sha256,
        "json_ids_sha256": config.expected_json_ids_sha256,
        "prior_train_ids_sha256": config.expected_prior_train_ids_sha256,
        "source_dev_ids_sha256": config.expected_source_dev_ids_sha256,
    }
    if hashes != expected_hashes:
        raise ValueError("paired consistency selection hashes differ")
    return {
        "dataset": dataset,
        "raw_by_id": raw_by_id,
        "heldout_sample_ids": heldout_sample_ids,
        "heldout_pair_schedule": heldout_pair_schedule,
        "pair_schedule": pair_schedule,
        "json_schedule": json_schedule,
        "hashes": hashes,
    }


def target_prediction_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if logits.ndim != 3 or labels.ndim != 2 or logits.shape[:2] != labels.shape:
        raise ValueError("target prediction logits shape mismatch")
    shifted_logits = logits[:, :-1, :]
    shifted_labels = labels[:, 1:]
    mask = shifted_labels != -100
    if mask.sum().item() == 0:
        raise ValueError("target prediction contains no supervised tokens")
    return shifted_logits[mask], shifted_labels[mask]


def supervised_target_labels(labels: torch.Tensor) -> torch.Tensor:
    if labels.ndim != 2:
        raise ValueError("supervised target labels shape mismatch")
    shifted_labels = labels[:, 1:]
    target = shifted_labels[shifted_labels != -100]
    if target.numel() == 0:
        raise ValueError("supervised target contains no labels")
    return target


def align_teacher_logits(
    teacher_logits: torch.Tensor,
    teacher_labels: torch.Tensor,
    student_labels: torch.Tensor,
) -> torch.Tensor:
    if teacher_logits.ndim != 2 or teacher_labels.ndim != 1:
        raise ValueError("teacher target logits shape mismatch")
    if teacher_logits.shape[0] < student_labels.shape[0]:
        raise ValueError("process target is shorter than final target")
    aligned_logits = teacher_logits[-student_labels.shape[0] :]
    aligned_labels = teacher_labels[-student_labels.shape[0] :]
    if not torch.equal(aligned_labels, student_labels):
        raise ValueError("paired consistency target suffix labels differ")
    return aligned_logits


def aligned_pair_logits(
    process_logits: torch.Tensor,
    process_labels: torch.Tensor,
    final_logits: torch.Tensor,
    final_labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    teacher_logits, teacher_labels = target_prediction_logits(
        process_logits,
        process_labels,
    )
    student_logits, student_labels = target_prediction_logits(
        final_logits,
        final_labels,
    )
    teacher_logits = align_teacher_logits(
        teacher_logits,
        teacher_labels,
        student_labels,
    )
    return teacher_logits, student_logits, student_labels


def paired_consistency_kl(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    *,
    temperature: float,
    teacher_detach: bool,
) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("consistency temperature must be positive")
    if teacher_logits.shape != student_logits.shape:
        raise ValueError("paired consistency logits shape mismatch")
    teacher = teacher_logits.detach() if teacher_detach else teacher_logits
    teacher_probability = functional.softmax(
        teacher / temperature,
        dim=-1,
    )
    student_log_probability = functional.log_softmax(
        student_logits / temperature,
        dim=-1,
    )
    return functional.kl_div(
        student_log_probability,
        teacher_probability,
        reduction="batchmean",
    ) * (temperature**2)


def build_step_schedule(selection: dict[str, Any]) -> list[dict[str, str]]:
    pair_schedule = selection["pair_schedule"]
    json_schedule = selection["json_schedule"]
    if len(pair_schedule) != len(json_schedule):
        raise ValueError("paired consistency pair and JSON step counts differ")
    result = []
    for pair, json_sample_id in zip(pair_schedule, json_schedule):
        result.append({"kind": "pair", **pair})
        result.append(
            {"kind": "json", "sample_id": json_sample_id}
        )
    return result


def _sample_batch(
    sample: TokenizedSample,
    *,
    pad_token_id: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device)
        for key, value in collate_samples(
            [sample],
            pad_token_id=pad_token_id,
        ).items()
    }


def run(config: PairedConsistencyConfig) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("paired consistency requires CUDA")
    selection = build_selection_contract(config)
    dataset = selection["dataset"]
    raw_by_id = selection["raw_by_id"]
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_path,
        local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenized = tokenize_samples(dataset, tokenizer, max_length=config.max_length)
    by_id = {sample.sample_id: sample for sample in tokenized}
    heldout = [by_id[sample_id] for sample_id in selection["heldout_sample_ids"]]
    steps = build_step_schedule(selection)
    if len(steps) != config.max_steps:
        raise ValueError("paired consistency resolved step count differs")

    set_seed(config.seed)
    device = torch.device("cuda")
    output_root = Path(config.output_dir)
    adapter_dir = output_root / "adapter"
    output_root.mkdir(parents=True, exist_ok=True)
    started = time.time()
    model = Qwen3_5ForCausalLM.from_pretrained(
        config.model_path,
        local_files_only=True,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
    ).to(device)
    model.config.use_cache = False
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    model = get_peft_model(
        model,
        LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=list(config.lora_targets),
            task_type="CAUSAL_LM",
        ),
    )
    trainable = _trainable_parameters(model)
    baseline_metrics, baseline_rows = evaluate_exact(
        model,
        tokenizer,
        heldout,
        device=device,
        max_new_tokens=config.generation_max_new_tokens,
    )
    optimizer = AdamW(
        trainable,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        eps=1e-6,
    )
    loss_curve = []
    optimizer.zero_grad(set_to_none=True)
    model.train()
    for step_index, step in enumerate(steps):
        step_number = step_index + 1
        if step["kind"] == "pair":
            process = by_id[step["process_sample_id"]]
            final = by_id[step["final_sample_id"]]
            if (
                raw_by_id[process.sample_id]["pair_id"]
                != raw_by_id[final.sample_id]["pair_id"]
            ):
                raise ValueError("paired consistency step pair identity differs")
            process_batch = _sample_batch(
                process,
                pad_token_id=tokenizer.pad_token_id,
                device=device,
            )
            final_batch = _sample_batch(
                final,
                pad_token_id=tokenizer.pad_token_id,
                device=device,
            )
            process_outputs = model(**process_batch, use_cache=False)
            process_component = (
                config.process_ce_weight * process_outputs.loss
            )
            _assert_finite_loss(process_component, step=step_number)
            process_component.backward()
            teacher_logits, teacher_labels = target_prediction_logits(
                process_outputs.logits,
                process_batch["labels"],
            )
            final_labels = supervised_target_labels(final_batch["labels"])
            teacher_logits = align_teacher_logits(
                teacher_logits,
                teacher_labels,
                final_labels,
            ).detach()
            process_ce_value = float(process_outputs.loss.detach().cpu())
            del process_outputs

            final_outputs = model(**final_batch, use_cache=False)
            student_logits, student_labels = target_prediction_logits(
                final_outputs.logits,
                final_batch["labels"],
            )
            if not torch.equal(student_labels, final_labels):
                raise ValueError("final target labels changed during forward")
            consistency = paired_consistency_kl(
                teacher_logits,
                student_logits,
                temperature=config.consistency_temperature,
                teacher_detach=config.teacher_detach,
            )
            final_component = (
                config.final_ce_weight * final_outputs.loss
                + config.consistency_weight * consistency
            )
            _assert_finite_loss(final_component, step=step_number)
            final_component.backward()
            total_value = (
                config.process_ce_weight * process_ce_value
                + config.final_ce_weight
                * float(final_outputs.loss.detach().cpu())
                + config.consistency_weight
                * float(consistency.detach().cpu())
            )
            components = {
                "kind": "pair",
                "process_ce": process_ce_value,
                "final_ce": float(final_outputs.loss.detach().cpu()),
                "consistency_kl": float(consistency.detach().cpu()),
            }
        else:
            sample = by_id[step["sample_id"]]
            outputs = model(
                **_sample_batch(
                    sample,
                    pad_token_id=tokenizer.pad_token_id,
                    device=device,
                ),
                use_cache=False,
            )
            total = outputs.loss
            _assert_finite_loss(total, step=step_number)
            total.backward()
            total_value = float(total.detach().cpu())
            components = {
                "kind": "json",
                "json_ce": float(outputs.loss.detach().cpu()),
            }
        scale = _scheduler_scale(
            step_index,
            config.warmup_steps,
            config.max_steps,
        )
        for group in optimizer.param_groups:
            group["lr"] = config.learning_rate * scale
        try:
            _assert_finite_gradients(trainable, step=step_number)
            torch.nn.utils.clip_grad_norm_(
                trainable,
                max_norm=1.0,
                error_if_nonfinite=True,
            )
            optimizer.step()
            _assert_finite_parameters(trainable, step=step_number)
        except (FloatingPointError, RuntimeError) as error:
            _write_failure(
                output_root,
                step=step_number,
                stage="paired_consistency",
                error=error,
            )
            raise
        optimizer.zero_grad(set_to_none=True)
        loss_curve.append(
            {
                "step": step_number,
                "total_loss": total_value,
                "learning_rate": optimizer.param_groups[0]["lr"],
                **components,
            }
        )

    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir, safe_serialization=True)
    tokenizer.save_pretrained(adapter_dir)
    post_metrics, post_rows = evaluate_exact(
        model,
        tokenizer,
        heldout,
        device=device,
        max_new_tokens=config.generation_max_new_tokens,
    )
    generations = {"baseline": baseline_rows, "post_sft": post_rows}
    generations_path = output_root / "generations.json"
    generations_path.write_text(
        json.dumps(generations, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    result = {
        "schema_version": "nano_train_paired_consistency_result_v1",
        "experiment_id": config.experiment_id,
        "config": {
            **config.__dict__,
            "lora_targets": list(config.lora_targets),
        },
        "selection": {
            "heldout_sample_ids": selection["heldout_sample_ids"],
            "pair_schedule": selection["pair_schedule"],
            "json_schedule": selection["json_schedule"],
            "step_schedule": steps,
            "hashes": selection["hashes"],
        },
        "dataset": {
            "dataset_id": dataset["dataset_id"],
            "dataset_file_sha256": sha256_file(Path(config.dataset_path)),
            "release_manifest_sha256": sha256_file(
                Path(config.release_manifest_path)
            ),
            "heldout_samples": len(heldout),
        },
        "model_config_sha256": sha256_file(
            Path(config.model_path) / "config.json"
        ),
        "adapter_sha256": sha256_tree(adapter_dir),
        "trainable_parameters": sum(
            parameter.numel() for parameter in trainable
        ),
        "baseline_validation": baseline_metrics,
        "post_validation": post_metrics,
        "loss_curve": loss_curve,
        "dependencies": dependency_versions(),
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "wall_seconds": time.time() - started,
        "generations_sha256": sha256_file(generations_path),
        "failure_receipt_exists": (output_root / "failure.json").exists(),
    }
    (output_root / "metrics.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result
