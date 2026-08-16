# Semantic Arithmetic SFT Smoke v5 Result

## Result

V5 is numerically stable and reaches one complete train-set coverage
equivalent, but fails its frozen validation thresholds.

- baseline strict exact: 3/32;
- post strict exact: 12/32;
- baseline semantic valid: 4/32;
- post semantic valid: 12/32;
- early five-step mean loss: 0.124293;
- late five-step mean loss: 0.096740;
- peak training memory: 18.65 GiB;
- wall time: 311.3 seconds;
- independent reload semantic valid: 12/32.

All 40 losses and all 224 FP32 adapter tensors
are finite. Independent reload reproduces both metrics exactly.

## Failure Analysis

Post-SFT taxonomy:

- semantic valid: 12;
- CALC/FINAL mismatch: 13;
- CALC execution mismatch: 6;
- invalid trace grammar: 1.

Relative to v4, three execution mismatches become semantic-valid, one
semantic-valid case regresses to an execution mismatch, and all 13
CALC/FINAL mismatches remain:

- execution mismatch to semantic valid:
  3;
- semantic valid to execution mismatch:
  1;
- CALC/FINAL mismatch unchanged:
  13.

Full coverage therefore adds only two net semantic-valid cases over v4.

## Generation Budget Audit

The frozen 32-token generation budget is shorter than the target contract:

- train target content: max 37 tokens,
  69/160 above the cap;
- validation target content: max
  37 tokens,
  14/32 above the cap;
- validation post-SFT outputs at the cap:
  17/32;
- failed validation outputs at the cap:
  16/32;
- failures whose canonical target is above the cap:
  14/32.

All 13 CALC/FINAL mismatches are in the over-cap group. Six arithmetic
execution mismatches remain and are not explained by target truncation.
This audit exposes an evaluation-contract defect; it does not rescore v5 or
turn the failed run into a pass.

## Decision

V5 fails because semantic validation is
12/32 rather than 32/32 and strict
exact is 12/32 rather than at least 30/32.
Do not benchmark, merge, scale, or start RL.

Preserve the official 12/32 result. Before another training intervention,
pre-register an evaluation-only audit that loads the unchanged adapter and
raises only the generation budget above the 37-token target-content maximum.
The audit must be reported separately and cannot overwrite v5.

## Reproduction Identity

- pre-registration revision: `c8e331d`;
- config SHA256: `89e48fa387851e06a9394253e3bbdc345d7a0e84d963015be67e2ae8183fad38`;
- dataset SHA256: `d226f243051b7d2d2d4db4d5a596b871032fa44d71b296586f879559a8781c09`;
- model config SHA256: `ddc63e1c717afa86c865bb5e01313d89d72bb53b97ad4a8a03ba8510c0621670`;
- metrics SHA256: `696d00d59c65f861cde9c93f2a31fd106ffcfd16bfc8e0e1fbbb5c9d015f4c9e`;
- generations SHA256: `6210f9ee7cb8a03377dc6588c82f3c79012d6104465cab7bdeb739035adc68d6`;
- reload receipt SHA256: `7f3cdb37a1d1ef3d8cc8519488e83b9ac612d45b4203a4ed5b5757ec4b717fa2`;
- adapter tree SHA256: `7ecb48dad68b0a7499baefcfeb587ce72ecf85df60a8a7c338c25b3d464f3421`.
