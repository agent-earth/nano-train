from __future__ import annotations

import json
import ast
import hashlib
import math
import operator
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass(frozen=True)
class TokenizedSample:
    sample_id: str
    split: str
    input_ids: list[int]
    labels: list[int]
    prompt_ids: list[int]
    target: str
    format_family: str
    verifier: dict[str, Any] | None
    task_family: str = ""
    task_spec: dict[str, Any] | None = None


OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def evaluate_arithmetic(expression: str) -> int | float:
    tree = ast.parse(expression, mode="eval")

    def evaluate(node: ast.AST) -> int | float:
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if (
            isinstance(node, ast.Constant)
            and type(node.value) in {int, float}
        ):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in OPERATORS:
            return OPERATORS[type(node.op)](
                evaluate(node.left),
                evaluate(node.right),
            )
        if isinstance(node, ast.UnaryOp) and type(node.op) in OPERATORS:
            return OPERATORS[type(node.op)](evaluate(node.operand))
        raise ValueError(f"unsafe arithmetic node: {type(node).__name__}")

    result = evaluate(tree)
    if not isinstance(result, (int, float)) or not math.isfinite(float(result)):
        raise ValueError("arithmetic result is not finite")
    return result


def format_number(value: int | float) -> str:
    number = float(value)
    return str(int(number)) if number.is_integer() else format(number, ".12g")


def _arithmetic_structure(expression: str) -> str:
    tree = ast.parse(expression, mode="eval")
    evaluate_arithmetic(expression)
    return ast.dump(tree, annotate_fields=True, include_attributes=False)


def _arithmetic_constants(expression: str) -> set[str]:
    tree = ast.parse(expression, mode="eval")
    evaluate_arithmetic(expression)
    return {
        format_number(node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and type(node.value) in {int, float}
    }


def semantic_output_valid(sample: TokenizedSample, output: str) -> bool:
    if sample.format_family == "skill_release_exact":
        return skill_release_output_valid(
            sample.task_family,
            sample.task_spec,
            sample.verifier,
            output,
        )
    if sample.format_family == "reasoning_numeric":
        verifier = sample.verifier or {}
        if verifier.get("kind") != "safe_ast_reasoning_numeric_v1":
            return False
        match = re.fullmatch(
            (
                r"WORK: (.+) = "
                r"([-+]?(?:[0-9]+(?:\.[0-9]+)?|\.[0-9]+))\n"
                r"FINAL: ([-+]?(?:[0-9]+(?:\.[0-9]+)?|\.[0-9]+))"
            ),
            output.strip(),
        )
        if match is None:
            return False
        expression, work_result, final_result = match.groups()
        try:
            verified = format_number(evaluate_arithmetic(expression))
        except (SyntaxError, ValueError, ZeroDivisionError, OverflowError):
            return False
        return (
            work_result == final_result
            and work_result == verifier.get("expected_result")
            and verified == verifier.get("expected_result")
        )
    if sample.format_family == "process_trace_numeric":
        verifier = sample.verifier or {}
        steps = verifier.get("steps")
        if (
            verifier.get("kind") != "safe_ast_arithmetic_process_v2"
            or not isinstance(steps, list)
            or len(steps) not in {2, 3}
        ):
            return False
        lines = output.strip().splitlines()
        if len(lines) != len(steps) + 1:
            return False
        previous_result = None
        try:
            for index, (line, expected_step) in enumerate(
                zip(lines[:-1], steps),
                start=1,
            ):
                match = re.fullmatch(
                    (
                        rf"STEP {index}: (.+) = "
                        r"([-+]?(?:[0-9]+(?:\.[0-9]+)?|\.[0-9]+))"
                    ),
                    line,
                )
                if match is None:
                    return False
                expression, result = match.groups()
                verified = format_number(evaluate_arithmetic(expression))
                if (
                    _arithmetic_structure(expression)
                    != _arithmetic_structure(expected_step["expression"])
                    or result != expected_step["expected_result"]
                    or verified != expected_step["expected_result"]
                ):
                    return False
                if (
                    previous_result is not None
                    and previous_result not in _arithmetic_constants(expression)
                ):
                    return False
                previous_result = result
            final_match = re.fullmatch(
                (
                    r"FINAL: "
                    r"([-+]?(?:[0-9]+(?:\.[0-9]+)?|\.[0-9]+))"
                ),
                lines[-1],
            )
            source_result = format_number(
                evaluate_arithmetic(verifier["source_expression"])
            )
        except (
            KeyError,
            SyntaxError,
            TypeError,
            ValueError,
            ZeroDivisionError,
            OverflowError,
        ):
            return False
        return (
            final_match is not None
            and previous_result == verifier.get("expected_result")
            and final_match.group(1) == verifier.get("expected_result")
            and source_result == verifier.get("expected_result")
        )
    if sample.format_family != "trace_numeric":
        return output.strip() == sample.target
    verifier = sample.verifier or {}
    match = re.fullmatch(
        (
            r"CALC: (.+) = "
            r"([-+]?(?:[0-9]+(?:\.[0-9]+)?|\.[0-9]+))\n"
            r"FINAL: ([-+]?(?:[0-9]+(?:\.[0-9]+)?|\.[0-9]+))"
        ),
        output.strip(),
    )
    if match is None:
        return False
    expression, calc_result, final_result = match.groups()
    try:
        verified = format_number(evaluate_arithmetic(expression))
    except (SyntaxError, ValueError, ZeroDivisionError, OverflowError):
        return False
    return (
        verifier.get("kind") == "safe_ast_arithmetic_v1"
        and calc_result == final_result
        and calc_result == verifier.get("expected_result")
        and verified == verifier.get("expected_result")
    )


def skill_release_output_valid(
    family_id: str,
    task_spec: dict[str, Any] | None,
    verifier: dict[str, Any] | None,
    output: str,
) -> bool:
    task_spec = task_spec or {}
    verifier = verifier or {}
    kind = verifier.get("kind")
    if family_id == "verified-reasoning":
        if kind != "safe_execution_receipt_v1":
            return False
        expression = task_spec.get("expression")
        match = re.fullmatch(
            r"FINAL: ([-+]?(?:[0-9]+(?:\.[0-9]+)?|\.[0-9]+))",
            output.strip(),
        )
        if not isinstance(expression, str) or match is None:
            return False
        try:
            expected = format_number(evaluate_arithmetic(expression))
        except (SyntaxError, ValueError, ZeroDivisionError, OverflowError):
            return False
        return match.group(1) == expected
    value = _json_object(output)
    if value is None:
        return False
    if family_id == "tool-use-and-recovery":
        return (
            kind == "tool_trace_contract_v1"
            and value.get("tool_calls") == task_spec.get("required_calls")
            and value.get("final_status") == "verified"
        )
    if family_id == "planning-and-state":
        if kind != "state_plan_consistency_v1":
            return False
        return all(
            value.get(key) == task_spec.get(key)
            for key in ("constraints", "evidence", "pending", "stop")
        )
    if family_id == "coding-and-validation":
        original = task_spec.get("original_content")
        if (
            kind != "patch_test_receipt_v1"
            or not isinstance(original, str)
        ):
            return False
        return (
            value.get("file") == task_spec.get("file")
            and value.get("before_sha256")
            == hashlib.sha256(original.encode("utf-8")).hexdigest()
            and value.get("after_content") == task_spec.get("expected_content")
            and value.get("test_command") == task_spec.get("test_command")
            and value.get("test_status") == "passed"
        )
    if family_id == "skill-routing-and-reflection":
        if kind != "skill_route_receipt_v1":
            return False
        request_tags = task_spec.get("request_tags")
        skills = task_spec.get("skills")
        if not isinstance(request_tags, list) or not isinstance(skills, list):
            return False
        required = set(request_tags)
        eligible = [
            (len(skill["tags"]), skill["skill_id"])
            for skill in skills
            if (
                isinstance(skill, dict)
                and isinstance(skill.get("tags"), list)
                and required <= set(skill["tags"])
                and skill.get("skill_id")
            )
        ]
        steps = value.get("steps")
        return (
            bool(eligible)
            and value.get("selected_skill") == sorted(eligible)[0][1]
            and isinstance(steps, list)
            and bool(steps)
        )
    return False


def _json_object(output: str) -> dict[str, Any] | None:
    try:
        value = json.loads(output)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def load_analog_dataset(path: str | Path) -> dict[str, Any]:
    dataset = json.loads(Path(path).read_text(encoding="utf-8"))
    if dataset.get("schema_version") != "nano_analog_dataset_v1":
        raise ValueError("unsupported analog dataset schema")
    policy = dataset.get("policy", {})
    if (
        policy.get("source_split") != "non_eval_analog_only"
        or policy.get("training_allowed") is not True
        or policy.get("contains_benchmark_content") is not False
    ):
        raise ValueError("analog dataset is not eligible for SFT smoke")
    samples = dataset.get("samples", [])
    if not samples:
        raise ValueError("analog dataset contains no samples")
    if any(sample.get("training_eligible") is not True for sample in samples):
        raise ValueError("analog dataset contains an ineligible sample")
    return dataset


def load_skill_release_dataset(
    path: str | Path,
    release_manifest_path: str | Path,
    *,
    train_samples_per_family: int,
    validation_samples_per_family: int,
) -> dict[str, Any]:
    dataset_path = Path(path)
    release_path = Path(release_manifest_path)
    release = json.loads(release_path.read_text(encoding="utf-8"))
    if release.get("schema_version") != "nano_skill_sft_release_v1":
        raise ValueError("unsupported skill SFT release schema")
    if release.get("training_unblocked") is not True:
        raise ValueError("skill SFT release is not training-unblocked")
    expected_sha256 = (
        release.get("artifacts", {}).get("accepted_jsonl_sha256")
    )
    if _sha256_file(dataset_path) != expected_sha256:
        raise ValueError("skill SFT release dataset SHA256 mismatch")
    checks = release.get("checks")
    if not isinstance(checks, dict) or not checks or not all(checks.values()):
        raise ValueError("skill SFT release checks are incomplete")

    selected: dict[tuple[str, str], list[dict[str, Any]]] = {}
    seen_ids = set()
    seen_exact = set()
    seen_semantic = set()
    with dataset_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("schema_version") != "nano_skill_sft_sample_v1":
                raise ValueError(
                    f"unsupported skill sample at line {line_number}"
                )
            split = row.get("split")
            normalized_split = (
                "validation" if split == "dev" else split
            )
            if normalized_split not in {"train", "validation"}:
                raise ValueError("skill sample has unsupported split")
            if row.get("sample_id") in seen_ids:
                raise ValueError("skill release contains duplicate sample IDs")
            if row.get("exact_hash") in seen_exact:
                raise ValueError("skill release contains duplicate exact hashes")
            if row.get("semantic_hash") in seen_semantic:
                raise ValueError(
                    "skill release contains duplicate semantic hashes"
                )
            seen_ids.add(row["sample_id"])
            seen_exact.add(row["exact_hash"])
            seen_semantic.add(row["semantic_hash"])
            family = str(row.get("family_id", ""))
            if not family:
                raise ValueError("skill sample lacks family_id")
            key = (normalized_split, family)
            limit = (
                train_samples_per_family
                if normalized_split == "train"
                else validation_samples_per_family
            )
            bucket = selected.setdefault(key, [])
            if len(bucket) < limit:
                messages = row.get("messages")
                if (
                    not isinstance(messages, list)
                    or len(messages) < 2
                    or messages[-1].get("role") != "assistant"
                ):
                    raise ValueError("skill sample has invalid messages")
                bucket.append(
                    {
                        "sample_id": row["sample_id"],
                        "split": normalized_split,
                        "task_family": family,
                        "format_family": "skill_release_exact",
                        "generation_rule": "skill_release_v2",
                        "training_eligible": True,
                        "messages": messages,
                        "verifier": row.get("verifier"),
                        "task_spec": row.get("task_spec"),
                    }
                )

    families = sorted(
        {
            family
            for split, family in selected
            if split == "train"
        }
    )
    if not families:
        raise ValueError("skill release contains no train families")
    samples = []
    for family in families:
        train = selected.get(("train", family), [])
        validation = selected.get(("validation", family), [])
        if len(train) != train_samples_per_family:
            raise ValueError(f"insufficient train rows for {family}")
        if len(validation) != validation_samples_per_family:
            raise ValueError(f"insufficient validation rows for {family}")
        samples.extend(train)
        samples.extend(validation)
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
            "sha256": _sha256_file(release_path),
            "accepted_jsonl_sha256": expected_sha256,
        },
        "samples": samples,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tokenize_samples(
    dataset: dict[str, Any],
    tokenizer: Any,
    *,
    max_length: int,
) -> list[TokenizedSample]:
    result = []
    for sample in dataset["samples"]:
        messages = sample["messages"]
        prompt = tokenizer.apply_chat_template(
            messages[:-1],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        target = str(messages[-1]["content"])
        full = prompt + target + tokenizer.eos_token
        prompt_ids = tokenizer(
            prompt,
            add_special_tokens=False,
        ).input_ids
        input_ids = tokenizer(
            full,
            add_special_tokens=False,
        ).input_ids
        if len(input_ids) > max_length:
            raise ValueError(
                f"sample exceeds max_length: {sample['sample_id']} "
                f"{len(input_ids)}>{max_length}"
            )
        labels = [-100] * len(prompt_ids) + input_ids[len(prompt_ids) :]
        if len(labels) != len(input_ids) or not any(label != -100 for label in labels):
            raise ValueError(f"invalid assistant mask: {sample['sample_id']}")
        result.append(
            TokenizedSample(
                sample_id=str(sample["sample_id"]),
                split=str(sample["split"]),
                input_ids=input_ids,
                labels=labels,
                prompt_ids=prompt_ids,
                target=target,
                format_family=str(sample["format_family"]),
                verifier=sample.get("verifier"),
                task_family=str(sample.get("task_family", "")),
                task_spec=sample.get("task_spec"),
            )
        )
    return result


def collate_samples(
    samples: list[TokenizedSample],
    *,
    pad_token_id: int,
) -> dict[str, torch.Tensor]:
    max_length = max(len(sample.input_ids) for sample in samples)
    input_ids = []
    labels = []
    attention_mask = []
    for sample in samples:
        padding = max_length - len(sample.input_ids)
        input_ids.append(sample.input_ids + [pad_token_id] * padding)
        labels.append(sample.labels + [-100] * padding)
        attention_mask.append([1] * len(sample.input_ids) + [0] * padding)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
    }
