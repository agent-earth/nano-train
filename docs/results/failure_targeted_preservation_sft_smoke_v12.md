# Failure-Targeted Preservation SFT Smoke v12 Result

## Result

V12 is numerically stable but fails the frozen local gate.

- aggregate exact / semantic: 21/32 / 25/32;
- numeric exact / semantic: 8/16 /
  12/16;
- choice exact / semantic: 5/8 /
  5/8;
- process exact / semantic: 8/8 /
  8/8;
- early / late five-step mean loss:
  0.232207 / 0.018025;
- peak training memory: 20.18 GiB;
- independent reload exact / semantic:
  21/32 /
  25/32;
- post outputs at the cap:
  0/32.

All 32 losses and all 224 FP32 adapter tensors
are finite. Independent reload reproduces all metrics and failure IDs.

## Comparison To V11

V12 changes only the training dataset but moves aggregate strict from 23/32 to
21/32 and aggregate semantic from 26/32 to 25/32. Numeric semantic remains
12/16 with one fixed and one regressed case. Choice falls from 6/8 to 5/8,
while process remains 8/8.

The strict 21/32 result is below the frozen 22/32 gate. This is not a Pareto
improvement over v11.

## Decision

Reject v12 and keep v11 as the current candidate. Do not run the sealed
canary, prior 211-case suite, or independent holdout. Merge, scale-up, and RL
remain forbidden.

The independent holdout remains unread: no prompts or references were loaded.

## Reproduction Identity

- pre-registration revision: `b689446`;
- data revision: `3b4f76f`;
- config SHA256: `82ad3ca17fc23e5722fead74cf9387364183db2cda8493ed02474e0ef60d2d02`;
- dataset SHA256: `b9dcbec512831a3f2c96e7db5abf4a0750420f26a28cc0f2a27699661f79aa23`;
- model config SHA256: `ddc63e1c717afa86c865bb5e01313d89d72bb53b97ad4a8a03ba8510c0621670`;
- metrics SHA256: `1dd1203e02d4715849cf0245cdc00440eaafdb0b83dc6ac897adba6ffb2e8171`;
- generations SHA256: `99766340864d896118147ef9b289f5b890227355df193c0048a2e3e4d8e0385b`;
- reload receipt SHA256: `43f0506fe61fa9cf59ba40247033226322b9221eead2bb958e612f71591d9e2e`;
- adapter tree SHA256: `70da5f6069b4cb17cad1df989229b0f1de614f24915119b8c37e4fa521bcb492`.
