# Semantic Arithmetic SFT Smoke v4

## Hypothesis

Format SFT v3 reaches 32/32 valid output grammar but only 20/32 exact answers
on a fresh two-step-heavy split. Eleven of twelve errors are multi-step
semantic failures. V4 supervises executable arithmetic traces and finals.

## Verified Data

`verified-semantic-arithmetic-traces-v3` contains:

- 192 fresh deterministic synthetic samples;
- 160 train and 32 validation;
- 96 two-step and 96 three-step expressions;
- zero sample-ID, exact-hash, or semantic-hash overlap with analog v1/v2;
- no benchmark content or sealed case ID.

Every target has:

```text
CALC: <expression> = <result>
FINAL: <result>
```

A restricted AST verifier executes each expression and requires CALC and FINAL
to agree. Calls, names, attributes, indexing, powers, and unsafe nodes are
rejected.

## Frozen Training Configuration

FP32, Qwen3.5-4B identity, seed, LoRA scope, rank/alpha/dropout, effective
batch size, optimizer, LR, warmup, 20-step cap, fail-fast, and reload checks
remain frozen from SFT v3.

Only:

- dataset identity changes to semantic traces;
- generation budget increases from 8 to 32 tokens to permit two-line traces;
- evaluation adds deterministic semantic verification.

Identity:

- config SHA256:
  `a162cc982896b16d5f3f1bdb79ba455f24b629ec95cc149b289e90e0b6ffab04`;
- dataset SHA256:
  `d226f243051b7d2d2d4db4d5a596b871032fa44d71b296586f879559a8781c09`;
- model config SHA256:
  `ddc63e1c717afa86c865bb5e01313d89d72bb53b97ad4a8a03ba8510c0621670`.

## Evaluation

Two metrics are reported:

- strict exact: generated two-line output exactly matches the canonical target;
- semantic valid: generated CALC expression is safe, executes to the expected
  result, and agrees with FINAL.

Equivalent safe expressions may pass semantic validation without strict exact
match. Format-only datasets continue to use exact equality for both metrics.

## Pre-Registered Decision

V4 passes its local semantic smoke only if:

- all 20 steps have finite loss, gradients, and parameters;
- no failure receipt exists;
- mean loss over steps 16-20 is below mean loss over steps 1-5;
- post-SFT fresh semantic validation is 32/32;
- post-SFT strict exact validation is at least 30/32;
- both metrics improve over their measured pre-training baselines;
- all 32 outputs have verifier-valid trace grammar;
- saved adapter tensors are finite and reload reproduces both metrics;
- peak training memory is below 28 GiB.

Passing still does not authorize merge, scale, or RL. The adapter must next be
evaluated on unchanged matched benchmarks with task-level non-regression.

## Reproduction

```bash
PYTHONPATH=. ../.venv/bin/python scripts/validate_sft_smoke.py \
  --config configs/sft/semantic_arithmetic_smoke_v4.json
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. ../.venv/bin/python -m nano_train.cli \
  sft-smoke --config configs/sft/semantic_arithmetic_smoke_v4.json
```

## Result

V4 is numerically stable and directionally improves semantic arithmetic, but
it fails the frozen acceptance thresholds:

- baseline strict exact is 3/32 and post-SFT strict exact is 10/32;
- baseline semantic valid is 4/32 and post-SFT semantic valid is 10/32;
- all 20 losses and all 224 FP32 adapter tensors are finite;
- early and late five-step mean losses are 0.124278 and 0.078391;
- independent reload reproduces strict exact and semantic valid at 10/32;
- peak training memory is 18.65 GiB.

The post-SFT failure taxonomy is 13 CALC/FINAL mismatches, 8 execution
mismatches, 1 invalid trace, and 10 semantic-valid cases. The run exposes 80
examples for 160 training samples, or 0.50 train-set equivalents.

The adapter is rejected because semantic valid is below 32/32 and strict exact
is below 30/32. Benchmark evaluation, merge, scale-up, and RL remain
unauthorized. Preserve this negative evidence and pre-register one
full-coverage run that changes only `max_steps` from 20 to 40.

Public result:

- `docs/results/semantic_arithmetic_sft_smoke_v4.md`;
- `docs/results/semantic_arithmetic_sft_smoke_v4.public.json`.
