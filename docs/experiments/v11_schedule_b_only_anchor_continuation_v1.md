# V11 Schedule B-Only Anchored Continuation v1

## Hypothesis

V15 schedule supervision raises semantic accuracy to 28/32 and numeric
semantic accuracy to 15/16, but unconstrained fresh LoRA training loses v11
strict and choice preservation. Exact adapter interpolation retains capability
but does not restore the contract.

A short continuation from v11 may preserve its learned subspace while adding
schedule capability if LoRA A is frozen, only LoRA B is trained, and B remains
explicitly anchored to v11.

## Frozen Method

- initialize from the exact v11 adapter;
- freeze every LoRA A tensor;
- train only the 112 LoRA B tensors;
- use schedule-isolation data v10 without mutation;
- run 8 optimizer steps / 32 examples;
- expose exactly 4 schedule rows in steps 2-4;
- learning rate `5e-5`, warmup 1, FP32, effective batch 4;
- add normalized proximal penalty:

```text
0.5 * ||B - B_v11||^2 / ||B_v11||^2
```

with coefficient `1.0`.

The coefficient is fixed once. At the observed unconstrained v11-to-v15 B
drift (`relative L2 = 0.659`), this penalty would be about 0.217, comparable to
the observed CE scale. No coefficient, step, LR, or schedule search is allowed.

## Identity And Exposure

- config SHA256:
  `41e9c47229f52b9ff3c97985b6a637429f17756791226eaca6b06f703d863c59`;
- v11 anchor tree:
  `87248908918b06c2d28ff68efd4f0b1ff92ca8bf8b7588e1c7e81a85eb7da852`;
- schedule dataset SHA256:
  `2bb712de519149d776b1c346466ee49d20017f1065aa3d1b44ae59eb6f5b973a`;
- model config SHA256:
  `ddc63e1c717afa86c865bb5e01313d89d72bb53b97ad4a8a03ba8510c0621670`;
- trainable B parameters: 5,963,776;
- frozen A parameters: 3,997,696.

## Required Evidence

- baseline validation exactly reproduces v11;
- all saved A tensors are byte-equal to v11 A tensors;
- only B tensors drift;
- finite CE, penalty, total loss, gradients, and parameters;
- relative B drift is recorded;
- independent reload reproduces all post metrics and failure IDs;
- no capped output and memory below 28 GiB.

## Local Gate

Use the unchanged v11 gate:

- aggregate semantic >=24/32;
- numeric >=10/16;
- choice >=5/8;
- process >=7/8;
- strict exact >=22/32;
- every family above base.

Passing authorizes only the old sealed canary. The old 211-case suite must pass
base-4B non-regression before the independent holdout may be read. Merge,
scale, and RL remain forbidden.

## Reproduction

```bash
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. ../.venv/bin/python -m nano_train.cli \
  anchored-continuation \
  --config configs/continuation/v11_schedule_b_only_anchor_v1.json
```
