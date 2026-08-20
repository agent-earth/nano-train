from __future__ import annotations

import argparse
import json

from nano_train.anchor_policy_replay import (
    generate_teacher_cache as generate_anchor_policy_teacher_cache,
)
from nano_train.anchor_policy_replay import (
    load_config as load_anchor_policy_replay_config,
)
from nano_train.anchor_policy_replay import (
    run_arm as run_anchor_policy_replay_arm,
)
from nano_train.anchor_policy_replay import (
    validate_reload as validate_anchor_policy_replay_reload,
)
from nano_train.confidence_route import (
    generate_arm as generate_confidence_route_arm,
)
from nano_train.confidence_route import (
    load_config as load_confidence_route_config,
)
from nano_train.confidence_route import score_arm as score_confidence_route_arm
from nano_train.consistency_route import (
    load_config as load_consistency_route_config,
)
from nano_train.consistency_route import run_arm as run_consistency_route_arm
from nano_train.continuation import load_config as load_continuation_config
from nano_train.continuation import run as run_continuation
from nano_train.config import load_sft_smoke_config
from nano_train.paired_consistency import (
    load_config as load_paired_consistency_config,
)
from nano_train.paired_consistency import run as run_paired_consistency
from nano_train.preservation_dual_view import (
    load_config as load_preservation_dual_view_config,
)
from nano_train.preservation_dual_view import (
    run_arm as run_preservation_dual_view_arm,
)
from nano_train.preservation_dual_view import (
    validate_reload as validate_preservation_dual_view_reload,
)
from nano_train.quality_consistency import (
    load_config as load_quality_consistency_config,
)
from nano_train.quality_consistency import run as run_quality_consistency
from nano_train.quality_consistency import (
    validate_reload as validate_quality_consistency_reload,
)
from nano_train.rl_opd_admission import (
    load_config as load_admission_config,
)
from nano_train.rl_opd_admission import run as run_admission
from nano_train.rl_opd_admission import validate_reload
from nano_train.scaled_quality import load_config as load_scaled_quality_config
from nano_train.scaled_quality import run as run_scaled_quality
from nano_train.scaled_quality import validate_reload as validate_scaled_quality_reload
from nano_train.sft import run_sft_smoke
from nano_train.synthetic_quality import load_config as load_quality_config
from nano_train.synthetic_quality import run_arm as run_quality_arm


def main() -> None:
    parser = argparse.ArgumentParser(prog="nano-train")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-config")
    validate.add_argument("--config", required=True)

    run = subparsers.add_parser("sft-smoke")
    run.add_argument("--config", required=True)
    continuation = subparsers.add_parser("anchored-continuation")
    continuation.add_argument("--config", required=True)
    paired = subparsers.add_parser("paired-consistency")
    paired.add_argument("--config", required=True)
    admission = subparsers.add_parser("rl-opd-admission")
    admission.add_argument("--config", required=True)
    admission_reload = subparsers.add_parser("rl-opd-admission-reload")
    admission_reload.add_argument("--config", required=True)
    quality = subparsers.add_parser("synthetic-quality")
    quality.add_argument("--config", required=True)
    quality.add_argument("--arm", required=True)
    scaled_quality = subparsers.add_parser("scaled-quality-sft")
    scaled_quality.add_argument("--config", required=True)
    scaled_quality_reload = subparsers.add_parser("scaled-quality-sft-reload")
    scaled_quality_reload.add_argument("--config", required=True)
    quality_consistency = subparsers.add_parser("quality-consistency")
    quality_consistency.add_argument("--config", required=True)
    quality_consistency_reload = subparsers.add_parser(
        "quality-consistency-reload"
    )
    quality_consistency_reload.add_argument("--config", required=True)
    consistency_route = subparsers.add_parser("consistency-route")
    consistency_route.add_argument("--config", required=True)
    consistency_route.add_argument("--arm", required=True)
    confidence_generate = subparsers.add_parser("confidence-route-generate")
    confidence_generate.add_argument("--config", required=True)
    confidence_generate.add_argument("--arm", required=True)
    confidence_score = subparsers.add_parser("confidence-route-score")
    confidence_score.add_argument("--config", required=True)
    confidence_score.add_argument("--arm", required=True)
    dual_view = subparsers.add_parser("preservation-dual-view")
    dual_view.add_argument("--config", required=True)
    dual_view.add_argument("--arm", required=True)
    dual_view_reload = subparsers.add_parser(
        "preservation-dual-view-reload"
    )
    dual_view_reload.add_argument("--config", required=True)
    dual_view_reload.add_argument("--arm", required=True)
    anchor_policy_cache = subparsers.add_parser(
        "anchor-policy-teacher-cache"
    )
    anchor_policy_cache.add_argument("--config", required=True)
    anchor_policy = subparsers.add_parser("anchor-policy-replay")
    anchor_policy.add_argument("--config", required=True)
    anchor_policy.add_argument("--arm", required=True)
    anchor_policy_reload = subparsers.add_parser(
        "anchor-policy-replay-reload"
    )
    anchor_policy_reload.add_argument("--config", required=True)
    anchor_policy_reload.add_argument("--arm", required=True)

    args = parser.parse_args()
    if args.command == "anchored-continuation":
        result = run_continuation(load_continuation_config(args.config))
    elif args.command == "paired-consistency":
        result = run_paired_consistency(
            load_paired_consistency_config(args.config)
        )
    elif args.command == "rl-opd-admission":
        result = run_admission(load_admission_config(args.config))
    elif args.command == "rl-opd-admission-reload":
        result = validate_reload(load_admission_config(args.config))
    elif args.command == "synthetic-quality":
        result = run_quality_arm(
            load_quality_config(args.config),
            arm_id=args.arm,
        )
    elif args.command == "scaled-quality-sft":
        result = run_scaled_quality(
            load_scaled_quality_config(args.config)
        )
    elif args.command == "scaled-quality-sft-reload":
        result = validate_scaled_quality_reload(
            load_scaled_quality_config(args.config)
        )
    elif args.command == "quality-consistency":
        result = run_quality_consistency(
            load_quality_consistency_config(args.config)
        )
    elif args.command == "quality-consistency-reload":
        result = validate_quality_consistency_reload(
            load_quality_consistency_config(args.config)
        )
    elif args.command == "consistency-route":
        result = run_consistency_route_arm(
            load_consistency_route_config(args.config),
            arm_id=args.arm,
        )
    elif args.command == "confidence-route-generate":
        result = generate_confidence_route_arm(
            load_confidence_route_config(args.config),
            arm_id=args.arm,
        )
    elif args.command == "confidence-route-score":
        result = score_confidence_route_arm(
            load_confidence_route_config(args.config),
            scorer_arm=args.arm,
        )
    elif args.command == "preservation-dual-view":
        result = run_preservation_dual_view_arm(
            load_preservation_dual_view_config(args.config),
            arm_id=args.arm,
        )
    elif args.command == "preservation-dual-view-reload":
        result = validate_preservation_dual_view_reload(
            load_preservation_dual_view_config(args.config),
            arm_id=args.arm,
        )
    elif args.command == "anchor-policy-teacher-cache":
        result = generate_anchor_policy_teacher_cache(
            load_anchor_policy_replay_config(args.config)
        )
    elif args.command == "anchor-policy-replay":
        result = run_anchor_policy_replay_arm(
            load_anchor_policy_replay_config(args.config),
            arm_id=args.arm,
        )
    elif args.command == "anchor-policy-replay-reload":
        result = validate_anchor_policy_replay_reload(
            load_anchor_policy_replay_config(args.config),
            arm_id=args.arm,
        )
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
