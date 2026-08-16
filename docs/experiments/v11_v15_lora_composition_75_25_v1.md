# V11/V15 Exact LoRA Composition 75/25 v1

## Hypothesis

V11 preserves strict and choice behavior. V15 adds strong schedule-driven
semantic capability but interferes with those contracts. A preservation-heavy
adapter composition may retain part of v15's gain without another training
run or data mutation.

## Frozen Method

Test exactly one composition:

```text
delta_composed = 0.75 * delta_v11 + 0.25 * delta_v15
```

Both sources are rank-8, alpha-16 LoRA adapters with identical modules. The
composition is represented exactly as rank 16, alpha 16 by concatenating A
blocks and scaling B blocks. No SVD, approximation, training, or weight search
is allowed.

The builder must prove zero block error for all 112 LoRA module pairs and bind
both source adapter trees. Local evaluation uses the unchanged v6 development
rows and v11 gates.

Passing local authorizes only the old regression canary. The old 211-case suite
must pass base-4B non-regression before the independent holdout may be read.

## Frozen Identity

- v11 adapter tree:
  `87248908918b06c2d28ff68efd4f0b1ff92ca8bf8b7588e1c7e81a85eb7da852`;
- v15 adapter tree:
  `d7ed4de2d613de424c6240148703e5656e73c5a323ad59f930d8c9cb8972b660`;
- dataset SHA256:
  `ab51a1be5f45d7f71796fbf98ef6cce83ff9cb0f0a756fed01cb1e7aea55651d`;
- weights: `0.75 / 0.25`;
- target rank / alpha: `16 / 16`.

## Gate

- aggregate semantic >=24/32;
- numeric >=10/16;
- choice >=5/8;
- process >=7/8;
- strict exact >=22/32;
- finite rank-16 tensors and exact composition receipt;
- two independent evaluations reproduce all metrics and failure IDs;
- no capped outputs, memory below 28 GiB.

Merge, scale, RL, prior suite, and independent holdout remain blocked until
their staged gates pass.
