#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(path.rglob("*")):
        if item.is_file():
            digest.update(item.relative_to(path).as_posix().encode())
            digest.update(b"\0")
            digest.update(sha256_file(item).encode())
            digest.update(b"\0")
    return digest.hexdigest()


def load(path: Path) -> tuple[dict[str, torch.Tensor], dict[str, str] | None]:
    tensors = {}
    with safe_open(path, framework="pt", device="cpu") as handle:
        metadata = handle.metadata()
        for key in handle.keys():
            tensors[key] = handle.get_tensor(key)
    return tensors, metadata


def compose_pair(
    a_preservation: torch.Tensor,
    b_preservation: torch.Tensor,
    a_capability: torch.Tensor,
    b_capability: torch.Tensor,
    *,
    preservation_weight: float,
    capability_weight: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if (
        a_preservation.shape[0] != b_preservation.shape[1]
        or a_capability.shape[0] != b_capability.shape[1]
        or a_preservation.shape != a_capability.shape
        or b_preservation.shape != b_capability.shape
    ):
        raise ValueError("source LoRA pair shapes differ")
    a_composed = torch.cat([a_preservation, a_capability], dim=0)
    b_composed = torch.cat(
        [
            b_preservation * (2 * preservation_weight),
            b_capability * (2 * capability_weight),
        ],
        dim=1,
    )
    return a_composed, b_composed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preservation", required=True)
    parser.add_argument("--capability", required=True)
    parser.add_argument("--preservation-weight", type=float, required=True)
    parser.add_argument("--capability-weight", type=float, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--receipt", required=True)
    args = parser.parse_args()
    if args.preservation_weight != 0.75 or args.capability_weight != 0.25:
        raise SystemExit("only the pre-registered 0.75/0.25 blend is allowed")

    preservation = Path(args.preservation).resolve()
    capability = Path(args.capability).resolve()
    output = Path(args.output).resolve()
    receipt_path = Path(args.receipt).resolve()
    config_a = json.loads((preservation / "adapter_config.json").read_text())
    config_b = json.loads((capability / "adapter_config.json").read_text())
    for field in (
        "base_model_name_or_path",
        "bias",
        "fan_in_fan_out",
        "lora_alpha",
        "r",
        "target_modules",
        "task_type",
        "use_dora",
        "use_rslora",
    ):
        left = config_a[field]
        right = config_b[field]
        if field == "target_modules":
            left, right = sorted(left), sorted(right)
        if left != right:
            raise SystemExit(f"source adapter config differs for {field}")
    if config_a["r"] != 8 or config_a["lora_alpha"] != 16:
        raise SystemExit("source adapters must be rank 8 / alpha 16")

    weights_a = preservation / "adapter_model.safetensors"
    weights_b = capability / "adapter_model.safetensors"
    tensors_a, metadata_a = load(weights_a)
    tensors_b, metadata_b = load(weights_b)
    if set(tensors_a) != set(tensors_b) or metadata_a != metadata_b:
        raise SystemExit("source adapter keys or metadata differ")

    combined: dict[str, torch.Tensor] = {}
    pair_count = 0
    for key in sorted(tensors_a):
        if not key.endswith(".lora_A.weight"):
            continue
        b_key = key.removesuffix(".lora_A.weight") + ".lora_B.weight"
        if b_key not in tensors_a:
            raise SystemExit(f"missing LoRA B tensor for {key}")
        a_tensor, b_tensor = compose_pair(
            tensors_a[key],
            tensors_a[b_key],
            tensors_b[key],
            tensors_b[b_key],
            preservation_weight=args.preservation_weight,
            capability_weight=args.capability_weight,
        )
        combined[key] = a_tensor
        combined[b_key] = b_tensor
        pair_count += 1
    if set(combined) != set(tensors_a):
        raise SystemExit("composed tensor key set differs from sources")
    if pair_count != 112:
        raise SystemExit(f"expected 112 LoRA module pairs, got {pair_count}")

    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    save_file(
        combined,
        output / "adapter_model.safetensors",
        metadata=metadata_a,
    )
    config = dict(config_a)
    config["r"] = 16
    config["lora_alpha"] = 16
    config["target_modules"] = sorted(config["target_modules"])
    (output / "adapter_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n"
    )
    for name in (
        "README.md",
        "chat_template.jinja",
        "tokenizer.json",
        "tokenizer_config.json",
    ):
        source = preservation / name
        if source.is_file():
            shutil.copy2(source, output / name)

    actual, _ = load(output / "adapter_model.safetensors")
    max_block_error = 0.0
    for key, tensor in actual.items():
        if key.endswith(".lora_A.weight"):
            expected = torch.cat([tensors_a[key], tensors_b[key]], dim=0)
        else:
            expected = torch.cat(
                [
                    tensors_a[key] * 1.5,
                    tensors_b[key] * 0.5,
                ],
                dim=1,
            )
        max_block_error = max(
            max_block_error,
            float((tensor - expected).abs().max()),
        )
    if max_block_error != 0.0:
        raise SystemExit("composed adapter block parity failed")

    receipt = {
        "schema_version": "nano_train_exact_lora_delta_composition_v1",
        "formula": "0.75 * delta_v11 + 0.25 * delta_v15",
        "representation": "rank16_block_concatenation",
        "preservation_weight": args.preservation_weight,
        "capability_weight": args.capability_weight,
        "source_rank": 8,
        "source_alpha": 16,
        "target_rank": 16,
        "target_alpha": 16,
        "module_pairs": pair_count,
        "tensor_count": len(actual),
        "max_block_error": max_block_error,
        "source": {
            "preservation_adapter_tree_sha256": sha256_tree(preservation),
            "preservation_weights_sha256": sha256_file(weights_a),
            "capability_adapter_tree_sha256": sha256_tree(capability),
            "capability_weights_sha256": sha256_file(weights_b),
        },
        "output": {
            "adapter_tree_sha256": sha256_tree(output),
            "weights_sha256": sha256_file(
                output / "adapter_model.safetensors"
            ),
        },
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
