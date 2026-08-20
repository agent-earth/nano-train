# Qwen3.5 Router Negative-Diversity SFT v2 Result

## Verdict

- admitted:
  `true`;
- baseline: 1127/1536;
- post: 1536/1536;
- delta: +409;
- reload metrics and all 1,536 outputs: exact.

## Per Label

```json
{
  "router_a": {
    "baseline_exact": 512,
    "delta": 0,
    "post_exact": 512,
    "reload_exact": 512,
    "samples": 512
  },
  "router_b": {
    "baseline_exact": 512,
    "delta": 0,
    "post_exact": 512,
    "reload_exact": 512,
    "samples": 512
  },
  "router_c": {
    "baseline_exact": 103,
    "delta": 409,
    "post_exact": 512,
    "reload_exact": 512,
    "samples": 512
  }
}
```

## C Subtypes

```json
{
  "box_total": {
    "baseline_exact": 0,
    "delta": 64,
    "post_exact": 64,
    "samples": 64
  },
  "paired_average": {
    "baseline_exact": 0,
    "delta": 64,
    "post_exact": 64,
    "samples": 64
  },
  "percentage_change": {
    "baseline_exact": 18,
    "delta": 46,
    "post_exact": 64,
    "samples": 64
  },
  "quotient_remainder": {
    "baseline_exact": 41,
    "delta": 23,
    "post_exact": 64,
    "samples": 64
  },
  "remaining_stock": {
    "baseline_exact": 0,
    "delta": 64,
    "post_exact": 64,
    "samples": 64
  },
  "single_operation": {
    "baseline_exact": 44,
    "delta": 20,
    "post_exact": 64,
    "samples": 64
  },
  "time_conversion": {
    "baseline_exact": 0,
    "delta": 64,
    "post_exact": 64,
    "samples": 64
  },
  "weighted_total": {
    "baseline_exact": 0,
    "delta": 64,
    "post_exact": 64,
    "samples": 64
  }
}
```

## Training

- 40 steps, effective batch 4;
- FP32 expanded LoRA, r=8, alpha=16;
- loss: 0.098509 ->
  0.000002;
- train/reload peak:
  19.63/
  15.83 GiB;
- wall: 1400.8s.

## Gates

```json
{
  "actual_exposure_ids_exact": true,
  "adapter_identity_matches": true,
  "aggregate_post_exact_gt_baseline": true,
  "data_release_identity_matches": true,
  "every_c_subtype_non_regression": true,
  "every_c_subtype_post_exact_at_least_60_of_64": true,
  "every_label_non_regression": true,
  "finite_loss_curve": true,
  "no_failure_receipt": true,
  "reload_generations_exact": true,
  "reload_metrics_exact": true,
  "reload_success": true,
  "router_a_post_exact_at_least_480_of_512": true,
  "router_b_post_exact_at_least_480_of_512": true,
  "router_c_post_exact_at_least_496_of_512": true
}
```

## Boundary

Passing only permits a separately pre-registered namespace-remapped serving
parity run. Fresh integration, benchmark, canary, holdout, and RL remain closed.
