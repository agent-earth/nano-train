#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json

from nano_train.orca_math_dpo_suffix import (
    load_config,
    run,
    validate_reload,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/preference_orca_math_dpo_v2.json",
    )
    parser.add_argument(
        "--mode", choices=("train", "reload"), default="train"
    )
    args = parser.parse_args()
    config = load_config(args.config)
    result = run(config) if args.mode == "train" else validate_reload(config)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
