from __future__ import annotations

import argparse
import json

from nano_train.continuation import load_config as load_continuation_config
from nano_train.continuation import run as run_continuation
from nano_train.config import load_sft_smoke_config
from nano_train.sft import run_sft_smoke


def main() -> None:
    parser = argparse.ArgumentParser(prog="nano-train")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-config")
    validate.add_argument("--config", required=True)

    run = subparsers.add_parser("sft-smoke")
    run.add_argument("--config", required=True)
    continuation = subparsers.add_parser("anchored-continuation")
    continuation.add_argument("--config", required=True)

    args = parser.parse_args()
    if args.command == "anchored-continuation":
        result = run_continuation(load_continuation_config(args.config))
    else:
        config = load_sft_smoke_config(args.config)
    if args.command == "validate-config":
        result = {
            "ok": True,
            "experiment_id": config.experiment_id,
            "max_steps": config.max_steps,
            "max_length": config.max_length,
            "effective_batch_size": (
                config.batch_size * config.gradient_accumulation_steps
            ),
        }
    elif args.command == "sft-smoke":
        result = run_sft_smoke(config)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
