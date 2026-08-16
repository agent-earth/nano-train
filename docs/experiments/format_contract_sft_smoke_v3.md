# Format Contract SFT Smoke v3

## Hypothesis

FP32 SFT v2 is numerically stable and improves validation from 23/26 to
25/26, but its sole remaining error is a valid-format two-step precedence
example. V3 tests a fresh two-step-heavy synthetic curriculum.

## Single Data Change

All training configuration remains frozen from FP32 v2. Only dataset identity
and sample composition change.

`format-contract-curriculum-analog-v2` contains:

- 160 fresh samples;
- 128 train and 32 validation;
- 32 single-step and 128 two-step examples;
- train: 24 single-step / 104 two-step;
- validation: 8 single-step / 24 two-step;
- zero sample-ID, exact-hash, or semantic-hash overlap with analog v1;
- no reuse of the observed v1 validation split;
- no benchmark content or sealed case ID.

## Frozen Training Configuration

- Qwen3.5-4B model config SHA256:
  `ddc63e1c717afa86c865bb5e01313d89d72bb53b97ad4a8a03ba8510c0621670`;
- dataset SHA256:
  `95d8e3e8a173960fd8604f284bae0243e74f4c924c96b719252c8c9a6525f001`;
- config SHA256:
  `fee61ad70cec96368849b6873e7f261dbfc822dc82af7d206cfdb29b58edbfdd`;
- FP32, seed 20260816;
- LoRA rank/alpha/dropout 8/16/0;
- unchanged five target-module names;
- effective batch size 4;
- 20 optimizer steps;
- LR 2e-4, two warmup steps, linear decay;
- assistant-only target loss;
- greedy validation with eight new tokens;
- finite loss/gradient/parameter fail-fast.

## Pre-Registered Decision

V3 passes its local curriculum smoke only if:

- all 20 steps complete with finite loss, gradients, and parameters;
- no failure receipt exists;
- mean loss over steps 16-20 is lower than mean loss over steps 1-5;
- post-SFT fresh validation reaches 32/32 exact;
- post-SFT fresh validation is strictly above its measured pre-training
  baseline;
- all saved adapter tensors are finite;
- independent adapter reload reproduces 32/32;
- peak training memory remains below 28 GiB.

The five-step mean replaces v2's single final-batch comparison because v2
shows high batch-to-batch loss variance. This criterion is frozen before v3
training and is not computed from v3 outcomes.

Passing still does not authorize RL or scaling. V3 must next be evaluated on
the unchanged matched benchmark harness and show no task regression.

## Reproduction

```bash
PYTHONPATH=. ../.venv/bin/python scripts/validate_sft_smoke.py \
  --config configs/sft/format_contract_smoke_v3.json
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. ../.venv/bin/python -m nano_train.cli \
  sft-smoke --config configs/sft/format_contract_smoke_v3.json
```
