# Arithmetic Process SFT Smoke v6

## Hypothesis

At a sufficient 48-token budget, the unchanged v5 adapter produces complete
and CALC/FINAL-consistent traces but leaves 18/32 arithmetic execution
failures. V6 tests whether fresh supervision on every intermediate operation
improves executable multi-step arithmetic.

## Verified Data

`verified-arithmetic-process-traces-v4` contains:

- 192 fresh deterministic non-evaluation samples;
- 160 train and 32 validation;
- 96 two-step and 96 three-step expressions;
- zero sample-ID, exact-hash, semantic-hash, or source-expression overlap with
  analog v1, curriculum v2, and semantic traces v3;
- no benchmark content, sealed case ID, raw model output, or teacher output.

Every target has two or three ordered `STEP` lines followed by `FINAL`. The
restricted verifier executes every step, requires later steps to consume the
preceding result, executes the source expression independently, and requires
the final step, source expression, and `FINAL` to agree.

## Frozen Training Configuration

Relative to v5, only process-objective fields change:

- dataset identity changes to process traces v4;
- `max_length` changes from 128 to 192 to fit the longest 187-token sample;
- generation budget changes from 32 to 80, above the 72-token target-plus-EOS
  maximum.

FP32, model identity, seed, LoRA scope/rank/alpha/dropout, effective batch 4,
optimizer, LR, warmup, 40-step full coverage, fail-fast, and reload checks
remain frozen.

Identity:

- data revision: `f1dcbe2`;
- config SHA256:
  `f8ab480d0195527b3fe8d98bb49ee377ba444257dcfe203de50c720d06624447`;
- dataset SHA256:
  `0e53fb3d05fb60569a4109da05b66d93c1158f734495e0126a55cf195c41653a`;
- model config SHA256:
  `ddc63e1c717afa86c865bb5e01313d89d72bb53b97ad4a8a03ba8510c0621670`.

## Evaluation

- strict exact requires the canonical target text;
- process semantic requires contiguous STEP lines, expected arithmetic AST
  structure, restricted execution of every step, consumption of the preceding
  result, and correct FINAL;
- capped output count reports any response reaching 80 tokens.

Equivalent redundant parentheses may pass process semantic without strict
exact equality. Unrelated expressions that happen to reach the final answer
do not pass.

## Pre-Registered Decision

V6 passes its local process smoke only if:

- all 40 steps have finite loss, gradients, and parameters;
- no failure receipt exists;
- mean loss over steps 36-40 is below mean loss over steps 1-5;
- post-SFT fresh process semantic validation is 32/32;
- post-SFT strict exact validation is at least 30/32;
- both metrics improve over their measured pre-training baselines;
- zero validation output reaches the 80-token cap;
- saved adapter tensors are finite and reload reproduces both metrics;
- peak training memory is below 28 GiB.

Failure preserves local evidence and forbids benchmark evaluation, merge,
scale-up, and RL. Passing authorizes only matched benchmark evaluation with
task-level non-regression; it does not directly authorize merge, scale-up, or
RL.

## Reproduction

```bash
PYTHONPATH=. ../.venv/bin/python scripts/validate_sft_smoke.py \
  --config configs/sft/arithmetic_process_smoke_v6.json
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. ../.venv/bin/python -m nano_train.cli \
  sft-smoke --config configs/sft/arithmetic_process_smoke_v6.json
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. ../.venv/bin/python \
  scripts/validate_sft_adapter.py \
  --config configs/sft/arithmetic_process_smoke_v6.json
```

## Result

V6 passes every frozen local process-contract gate:

- baseline strict exact / process semantic is 28/32 / 28/32;
- post-SFT strict exact / process semantic is 32/32 / 32/32;
- all 40 losses and all 224 FP32 adapter tensors are finite;
- early and late five-step mean losses are 0.001741121 and 0.000000474;
- independent reload reproduces both metrics at 32/32;
- peak training memory is 20.14 GiB;
- no post-SFT output reaches the 80-token cap.

The mechanism interpretation is limited. Baseline and post-SFT final-answer
accuracy are both 32/32. All four baseline process failures already have the
correct numeric `FINAL`; they combine operations or omit a canonical STEP.
V6 improves process-contract adherence, not final arithmetic accuracy.

This passed local smoke authorizes only frozen matched benchmark evaluation
with task-level non-regression. It does not authorize an arithmetic-uplift
claim, merge, scale-up, or RL.

Public result:

- `docs/results/arithmetic_process_sft_smoke_v6.md`;
- `docs/results/arithmetic_process_sft_smoke_v6.public.json`.
