# Semantic Arithmetic SFT Smoke v5

## Hypothesis

SFT v4 is numerically stable and improves semantic valid from 4/32 to 10/32,
but its 20 optimizer steps expose only 80 examples for 160 training rows. V5
tests whether exactly one train-set coverage equivalent is sufficient to bind
executable CALC results to FINAL values on the unchanged fresh validation set.

## Single-Variable Change

Relative to v4, the only training configuration field that changes is:

- `max_steps`: 20 to 40.

`experiment_id` and `output_dir` change only to isolate artifacts. Dataset,
validation split, model, FP32, seed, batch, gradient accumulation, optimizer,
LR, warmup steps, LoRA scope/rank/alpha/dropout, sequence length, generation
budget, fail-fast behavior, and reload validation remain frozen.

At effective batch size 4, 40 optimizer steps expose 160 examples for 160
training rows. The deterministic shuffled order and modulo traversal therefore
visit every training sample exactly once.

`max_steps` also defines the linear decay horizon in the existing runner.
Extending it from 20 to 40 changes the learning-rate values after warmup as a
derived consequence of the single config-field intervention. V5 can identify
the effect of this complete 40-step training schedule relative to v4, but it
cannot separately attribute any change to example coverage versus the longer
decay horizon.

## Frozen Identity

- config SHA256:
  `89e48fa387851e06a9394253e3bbdc345d7a0e84d963015be67e2ae8183fad38`;
- dataset SHA256:
  `d226f243051b7d2d2d4db4d5a596b871032fa44d71b296586f879559a8781c09`;
- model config SHA256:
  `ddc63e1c717afa86c865bb5e01313d89d72bb53b97ad4a8a03ba8510c0621670`.

## Pre-Registered Decision

V5 passes its local semantic smoke only if:

- all 40 steps have finite loss, gradients, and parameters;
- no failure receipt exists;
- mean loss over steps 36-40 is below mean loss over steps 1-5;
- post-SFT fresh semantic validation is 32/32;
- post-SFT strict exact validation is at least 30/32;
- both metrics improve over their measured pre-training baselines;
- all 32 outputs have verifier-valid trace grammar;
- saved adapter tensors are finite and reload reproduces both metrics;
- peak training memory is below 28 GiB.

These gates are unchanged from v4 except that the late loss window moves to
the final five steps. Failure preserves the adapter as local evidence but
forbids benchmark evaluation, merge, scale-up, and RL. Passing authorizes only
matched benchmark evaluation with task-level non-regression; it does not
directly authorize merge, scale-up, or RL.

## Reproduction

```bash
PYTHONPATH=. ../.venv/bin/python scripts/validate_sft_smoke.py \
  --config configs/sft/semantic_arithmetic_smoke_v5.json
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. ../.venv/bin/python -m nano_train.cli \
  sft-smoke --config configs/sft/semantic_arithmetic_smoke_v5.json
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. ../.venv/bin/python \
  scripts/validate_sft_adapter.py \
  --config configs/sft/semantic_arithmetic_smoke_v5.json
```

## Result

V5 is numerically stable but fails the frozen acceptance thresholds:

- baseline strict exact / semantic valid is 3/32 / 4/32;
- post-SFT strict exact / semantic valid is 12/32 / 12/32;
- all 40 losses and all 224 FP32 adapter tensors are finite;
- early and late five-step mean losses are 0.124293 and 0.096740;
- independent reload reproduces strict exact and semantic valid at 12/32;
- peak training memory is 18.65 GiB.

The post-SFT taxonomy is 12 semantic-valid, 13 CALC/FINAL mismatch, 6
execution mismatch, and 1 invalid trace. Relative to v4, three execution
mismatches become valid while one valid case regresses. All 13 CALC/FINAL
mismatches remain.

An audit after the frozen run found that the 32-token generation budget is
shorter than the target contract: target content reaches 37 tokens, 69/160
train and 14/32 validation targets exceed the cap, and all 13 CALC/FINAL
mismatch cases have over-cap canonical targets. This does not alter the
official v5 result. Six execution mismatches remain independent of this
truncation confounder.

The adapter is rejected. Benchmark evaluation, merge, scale-up, and RL remain
unauthorized. Preserve v5 as negative evidence and separately pre-register an
evaluation-only audit of this unchanged adapter with a sufficient generation
budget before another training intervention.

Public result:

- `docs/results/semantic_arithmetic_sft_smoke_v5.md`;
- `docs/results/semantic_arithmetic_sft_smoke_v5.public.json`.
