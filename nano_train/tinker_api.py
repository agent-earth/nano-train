from __future__ import annotations

import asyncio
import importlib
import importlib.metadata
import inspect
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


TINKER_REQUIRED_METHODS = {
    "ServiceClient": (
        "create_lora_training_client",
        "create_lora_training_client_async",
        "create_sampling_client",
        "create_rest_client",
        "get_server_capabilities",
    ),
    "TrainingClient": (
        "forward",
        "forward_async",
        "forward_backward",
        "forward_backward_async",
        "optim_step",
        "optim_step_async",
        "save_state",
        "save_state_async",
        "save_weights_and_get_sampling_client",
        "save_weights_and_get_sampling_client_async",
    ),
    "SamplingClient": (
        "sample",
        "sample_async",
        "compute_logprobs",
        "compute_logprobs_async",
    ),
}


@dataclass(frozen=True)
class TinkerBackendConfig:
    schema_version: str
    experiment_id: str
    provider: str
    model_name: str
    renderer_name: str
    lora_rank: int
    learning_rate: float
    submit_ahead: int
    api_key_env: str
    base_url_env: str | None
    train_mlp: bool
    train_attn: bool
    train_unembed: bool
    source_revisions: dict[str, str]


@dataclass(frozen=True)
class TinkerRuntime:
    config: TinkerBackendConfig
    module: Any
    service_client: Any


def load_tinker_backend_config(path: str | Path) -> TinkerBackendConfig:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = set(TinkerBackendConfig.__dataclass_fields__)
    if set(raw) != expected:
        raise ValueError(
            "invalid Tinker backend config fields: "
            f"unknown={sorted(set(raw) - expected)}, "
            f"missing={sorted(expected - set(raw))}"
        )
    config = TinkerBackendConfig(**raw)
    validate_tinker_backend_config(config)
    return config


def validate_tinker_backend_config(config: TinkerBackendConfig) -> None:
    if config.schema_version != "nano_train_tinker_backend_v1":
        raise ValueError("unsupported Tinker backend schema")
    if config.provider not in {"tinker", "twinkle"}:
        raise ValueError("provider must be tinker or twinkle")
    if not config.experiment_id:
        raise ValueError("experiment_id must not be empty")
    if not config.model_name:
        raise ValueError("model_name must not be empty")
    if not config.renderer_name:
        raise ValueError("renderer_name must not be empty")
    if config.lora_rank <= 0:
        raise ValueError("lora_rank must be positive")
    if config.learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if config.submit_ahead < 1:
        raise ValueError("submit_ahead must be at least one")
    if not config.api_key_env:
        raise ValueError("api_key_env must not be empty")
    if config.provider == "twinkle" and not config.base_url_env:
        raise ValueError("Twinkle provider requires base_url_env")
    if not config.source_revisions:
        raise ValueError("source_revisions must not be empty")
    invalid_revisions = {
        name: revision
        for name, revision in config.source_revisions.items()
        if not name
        or not re.fullmatch(r"[0-9a-f]{7,40}", revision)
    }
    if invalid_revisions:
        raise ValueError(
            f"source_revisions must contain git hashes: {invalid_revisions}"
        )


def _version_tuple(value: str) -> tuple[int, int, int]:
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)", value)
    if match is None:
        raise ValueError(f"unsupported package version format: {value}")
    return tuple(int(part) for part in match.groups())


def _prepare_provider(config: TinkerBackendConfig) -> Any:
    if config.provider == "twinkle":
        twinkle_client = importlib.import_module("twinkle_client")
        twinkle_client.init_tinker_client()
    tinker_module = importlib.import_module("tinker")
    version = importlib.metadata.version("tinker")
    version_tuple = _version_tuple(version)
    if config.provider == "twinkle" and version_tuple[:2] != (0, 16):
        raise RuntimeError(
            "pinned Twinkle source requires the isolated tinker 0.16.x "
            f"client environment, found {version}"
        )
    if config.provider == "tinker" and version_tuple < (0, 23, 0):
        raise RuntimeError(
            "Tinker Cookbook compatibility requires tinker>=0.23.0, "
            f"found {version}"
        )
    return tinker_module


def inspect_tinker_runtime(config: TinkerBackendConfig) -> dict[str, Any]:
    tinker_module = _prepare_provider(config)
    classes = {
        class_name: getattr(tinker_module, class_name)
        for class_name in TINKER_REQUIRED_METHODS
    }
    methods: dict[str, dict[str, str]] = {}
    missing: list[str] = []
    for class_name, required_methods in TINKER_REQUIRED_METHODS.items():
        methods[class_name] = {}
        runtime_class = classes[class_name]
        for method_name in required_methods:
            method = getattr(runtime_class, method_name, None)
            if method is None:
                missing.append(f"{class_name}.{method_name}")
                continue
            methods[class_name][method_name] = str(inspect.signature(method))
    if missing:
        raise RuntimeError(f"Tinker runtime is missing methods: {missing}")
    versions = {
        "tinker": importlib.metadata.version("tinker"),
    }
    if config.provider == "twinkle":
        versions["twinkle-kit"] = importlib.metadata.version("twinkle-kit")
    return {
        "schema_version": "nano_train_tinker_runtime_report_v1",
        "ok": True,
        "provider": config.provider,
        "experiment_id": config.experiment_id,
        "model_name": config.model_name,
        "renderer_name": config.renderer_name,
        "versions": versions,
        "methods": methods,
        "submit_ahead": config.submit_ahead,
        "source_revisions": config.source_revisions,
        "credentials": {
            "api_key_env": config.api_key_env,
            "api_key_present": bool(os.environ.get(config.api_key_env)),
            "base_url_env": config.base_url_env,
            "base_url_present": bool(
                config.base_url_env and os.environ.get(config.base_url_env)
            ),
        },
    }


def create_tinker_runtime(config: TinkerBackendConfig) -> TinkerRuntime:
    tinker_module = _prepare_provider(config)
    api_key = os.environ.get(config.api_key_env)
    if not api_key:
        raise RuntimeError(
            f"missing API key environment variable: {config.api_key_env}"
        )
    base_url = (
        os.environ.get(config.base_url_env)
        if config.base_url_env
        else None
    )
    if config.provider == "twinkle" and not base_url:
        raise RuntimeError(
            f"missing Twinkle base URL environment variable: {config.base_url_env}"
        )
    client_kwargs: dict[str, Any] = {
        "api_key": api_key,
        "user_metadata": {
            "experiment_id": config.experiment_id,
            "provider": config.provider,
        },
    }
    if base_url:
        client_kwargs["base_url"] = base_url
    service_client = tinker_module.ServiceClient(**client_kwargs)
    return TinkerRuntime(
        config=config,
        module=tinker_module,
        service_client=service_client,
    )


async def create_lora_training_client(runtime: TinkerRuntime) -> Any:
    config = runtime.config
    return await runtime.service_client.create_lora_training_client_async(
        base_model=config.model_name,
        rank=config.lora_rank,
        train_mlp=config.train_mlp,
        train_attn=config.train_attn,
        train_unembed=config.train_unembed,
        user_metadata={
            "experiment_id": config.experiment_id,
            "provider": config.provider,
            "renderer_name": config.renderer_name,
        },
    )


async def run_pipelined_training_step(
    training_client: Any,
    data: list[Any],
    *,
    tinker_module: Any,
    learning_rate: float,
    loss_fn: str = "cross_entropy",
    loss_fn_config: dict[str, float | str] | None = None,
) -> tuple[Any, Any]:
    forward_future = await training_client.forward_backward_async(
        data,
        loss_fn=loss_fn,
        loss_fn_config=loss_fn_config,
    )
    optimizer_future = await training_client.optim_step_async(
        tinker_module.AdamParams(
            learning_rate=learning_rate,
            beta1=0.9,
            beta2=0.95,
            eps=1e-8,
        )
    )
    return tuple(
        await asyncio.gather(
            forward_future.result_async(),
            optimizer_future.result_async(),
        )
    )


async def refresh_sampling_client(training_client: Any) -> Any:
    return await training_client.save_weights_and_get_sampling_client_async()
