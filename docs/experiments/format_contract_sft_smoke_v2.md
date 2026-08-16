# Format Contract SFT Smoke v2

## Root Cause

SFT smoke v1 has finite forward losses but non-finite gradients during the
first FP16 backward:

- 124/224 LoRA gradient tensors are non-finite;
- affected tensors include 108 MLP and 16 self-attention tensors;
- the failure occurs before gradient clipping and AdamW;
- v1 post-validation collapses from 23/26 to 0/26.

An FP32 control on the same first sample produces 0/224 non-finite gradients.
An FP32 four-microbatch optimizer-step control also passes:

- all four losses finite;
- pre-clip gradient norm 2.6533;
- all trainable parameters finite after AdamW;
- all 672 optimizer-state tensors finite;
- next-forward loss 0.0298 and finite;
- peak allocated memory 17.97 GiB.

The failure is therefore attributed to FP16 backward instability on this
Qwen3.5/Transformers fallback path, not AdamW epsilon or saved-adapter dtype.

## Single Change

v2 changes only training/model dtype from FP16 to FP32.

Data, seed, model identity, LoRA rank/alpha/targets, effective batch size,
learning rate, warmup, max length, 20 optimizer steps, validation generation,
and success criteria remain frozen from v1.

## Safety Changes

The runner now:

- fails immediately on non-finite forward loss;
- fails before optimizer step on non-finite gradients;
- clips gradients with `error_if_nonfinite=True`;
- fails after optimizer step on non-finite trainable parameters;
- writes `failure.json` and does not save an adapter or run post-validation
  after a numerical failure.

v1 artifacts remain in their original ignored path. v2 writes only to
`artifacts/format-contract-sft-smoke-v2`.

## Frozen Identity

- config SHA256:
  `62cc5189cb048fd1a2b4070ffdd27b0a18c3363df1ae8dfa244a381401646207`;
- dataset SHA256:
  `46f2128f219db7011d5db95b5ca3a97029b57f5ac959e194860b4c0f4ba3ad53`;
- model config SHA256:
  `ddc63e1c717afa86c865bb5e01313d89d72bb53b97ad4a8a03ba8510c0621670`.

## Pre-Registered Decision

v2 passes its local format smoke only if:

- all 20 optimizer steps complete with finite loss, gradients, and parameters;
- final loss is lower than initial loss;
- no `failure.json` exists;
- the saved adapter has a stable tree SHA256 and reloads;
- post-validation reaches 26/26 exact targets;
- post-validation is strictly above the frozen 23/26 base result;
- peak allocated GPU memory remains below 28 GiB.

Passing does not authorize RL or scale-up. The adapter must next run through
unchanged matched nano-harness evaluation and show no GSM8K/MMLU/GPQA
regression.

## Reproduction

```bash
PYTHONPATH=. ../.venv/bin/python scripts/validate_sft_smoke.py \
  --config configs/sft/format_contract_smoke_v2.json
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. ../.venv/bin/python -m nano_train.cli \
  sft-smoke --config configs/sft/format_contract_smoke_v2.json
```
