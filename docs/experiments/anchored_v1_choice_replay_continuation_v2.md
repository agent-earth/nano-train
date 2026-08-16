# Anchored-v1 Choice Replay Continuation v2

## Hypothesis

Anchored-v1 is the first aggregate win over the base 4B model and already
passes the old sealed canary, but its old full-development MMLU result is one
item below base. A very short generic choice-only continuation may restore
answer-choice preservation without discarding its schedule gain.

## Frozen Method

- initialize from the exact anchored-v1 adapter;
- freeze every LoRA A tensor and train only the 112 LoRA B tensors;
- train only the 40 synthetic choice rows in generic choice replay v11;
- keep the 32-row development split evaluation-only;
- run 4 optimizer steps / 16 examples, effective batch 4;
- learning rate `2.5e-5`, warmup 1, FP32;
- use normalized B-only proximal penalty coefficient `1.0`.

The 16 deterministic exposures cover all three generic choice rules:

- host-count choice: 5;
- sequential-fraction choice: 7;
- participant-average choice: 4.

This is one pre-registered dose. No step, learning-rate, penalty, seed, or
adapter-weight search is allowed after observing the result.

## Identity

- config SHA256:
  `afb70e3c2a7008bc4c6175ed1d988a5d99b14465320a2c18918848728bfaad16`;
- anchored-v1 adapter tree SHA256:
  `d29963cbd9284fe9a21de690babecbb34ed3fbbedd16d5d780daaf8abc45bc82`;
- choice replay dataset SHA256:
  `4657e96af9f9d1b81bfdb5fac6a29c31baf24b23d7753508aafdea5603ffd80d`;
- model config SHA256:
  `ddc63e1c717afa86c865bb5e01313d89d72bb53b97ad4a8a03ba8510c0621670`.

The dataset contains no benchmark, model-output, teacher-output, canary, or
independent-holdout payload. Training must remain restricted to `split=train`.

## Local Gate

Baseline validation must exactly reproduce anchored-v1:

- strict 22/32, semantic 25/32;
- numeric 11/16, choice 6/8, process 8/8.

Post-training must satisfy all of:

- strict >=22/32 and semantic >=25/32;
- numeric >=11/16 and process =8/8;
- choice >=7/8;
- relative B drift <=0.06;
- all LoRA A tensors unchanged, only LoRA B trainable;
- independent reload exactly reproduces metrics and failure IDs.

Failure ends this intervention with negative evidence. Passing authorizes only
the old sealed 40-case canary. The independent holdout remains unread until
the old 211-case suite passes per-task base non-regression.

## Reproduction

```bash
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. ../.venv/bin/python -m nano_train.cli \
  anchored-continuation \
  --config configs/continuation/anchored_v1_choice_replay_v2.json
```
