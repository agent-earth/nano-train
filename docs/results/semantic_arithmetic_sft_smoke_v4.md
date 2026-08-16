# Semantic Arithmetic SFT Smoke v4 Result

## Result

V4 is numerically stable and improves verified semantic accuracy, but fails
its pre-registered validation thresholds.

- baseline strict exact: 3/32;
- post strict exact: 10/32;
- baseline semantic valid: 4/32;
- post semantic valid: 10/32;
- early five-step mean loss: 0.124278;
- late five-step mean loss: 0.078391;
- peak training memory: 18.65 GiB;
- independent reload semantic valid: 10/32.

All 224 FP32 adapter tensors are finite. Semantic
validation improves by six cases and strict exact improves by seven.

## Remaining Failure

Post-SFT taxonomy:

- semantic valid: 10;
- CALC/FINAL mismatch: 13;
- CALC execution mismatch: 8;
- invalid trace grammar: 1.

V4 exposes only 80 training examples for
160 train samples, or
0.50 epoch equivalents. It
does not cover one full pass through the dataset.

## Decision

V4 fails because semantic validation is
10/32 rather than 32/32 and strict
exact is 10/32 rather than at least 30/32.
Do not benchmark, merge, scale, or start RL.

The next experiment may change only optimizer-step count from 20 to 40 so the
effective batch exposes exactly 160 examples, one train-set equivalent.
Data, validation, FP32, seed, LoRA, LR, and all gates must remain frozen.

## Reproduction Identity

- pre-registration revision: `0fdec2b`;
- config SHA256: `a162cc982896b16d5f3f1bdb79ba455f24b629ec95cc149b289e90e0b6ffab04`;
- dataset SHA256: `d226f243051b7d2d2d4db4d5a596b871032fa44d71b296586f879559a8781c09`;
- model config SHA256: `ddc63e1c717afa86c865bb5e01313d89d72bb53b97ad4a8a03ba8510c0621670`;
- metrics SHA256: `d3b4fe446db0e12f5b269caa61eb956512fd24e423bfa0283c0149489c70ea5c`;
- generations SHA256: `c2ed5e07c786a6f8654936bbcef214f01441efcc047d77e63a4f2fdadf1623be`;
- reload receipt SHA256: `687a52d6cd66e6d8674686a8f3d97738cb95b8bb1b016e44061810c6e9bf436a`;
- adapter tree SHA256: `1e04e9a22ec7ae76fd891e988b9963f3d49e89a21af860750fe6074b866ae2e1`.
