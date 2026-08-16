# Semantic Arithmetic v5 Budget Audit v1 Result

## Result

At 48 generated tokens:

- base strict exact / semantic valid:
  3/32 / 4/32;
- unchanged v5 adapter strict exact / semantic valid:
  14/32 / 14/32;
- official v5 strict exact / semantic valid remains:
  12/32 /
  12/32;
- peak audit memory: 15.81 GiB.

The audit performs no training and does not modify the adapter or official
v5 score.

## Contract

- generation budget: 48;
- maximum validation target content:
  37 tokens;
- maximum target plus EOS:
  38 tokens;
- unchanged adapter maximum output:
  37 tokens;
- unchanged adapter outputs at cap:
  0/32.

The unchanged adapter has no capped output under the audit budget.

## Failure Migration

Official v5 to 48-token audit:

- CALC/FINAL mismatch to semantic valid:
  2;
- CALC/FINAL mismatch to execution mismatch:
  11;
- invalid grammar to execution mismatch:
  1;
- execution mismatch unchanged:
  6;
- semantic valid unchanged:
  12.

The final 48-token adapter taxonomy is:

- semantic valid: 14;
- arithmetic execution mismatch:
  18;
- CALC/FINAL mismatch:
  0;
- invalid trace grammar:
  0.

Longer generation restores two correct cases and converts the other truncated
outputs into complete but arithmetically wrong traces. Truncation is material
but not the primary bottleneck.

## Decision

Keep official v5 at 12/32. The audit is descriptive and does not pass the
training gate. Benchmark evaluation, merge, scale-up, and RL remain forbidden.

Stop increasing optimizer steps, format supervision, and output budget. The
next data intervention must use fresh non-evaluation process traces with
verifier-checked intermediate operations so arithmetic execution, rather than
only expression copying and final agreement, receives supervision.

## Reproduction Identity

- pre-registration revision: `d15f12c`;
- audit config SHA256: `e1cb92f5b6a1f70b70caf806318280c24489aa1ac8622f5c1c459ee77957d465`;
- source adapter tree SHA256:
  `7ecb48dad68b0a7499baefcfeb587ce72ecf85df60a8a7c338c25b3d464f3421`;
- source metrics SHA256: `696d00d59c65f861cde9c93f2a31fd106ffcfd16bfc8e0e1fbbb5c9d015f4c9e`;
- source generations SHA256:
  `6210f9ee7cb8a03377dc6588c82f3c79012d6104465cab7bdeb739035adc68d6`;
- local audit result SHA256:
  `96ec926f969b5c39234d622f6753e5e713f375cb7b13adc360a9172c7697899c`;
- local audit generations SHA256:
  `d31480478eca783193c70ee4a2bdfdeb04c12638c95266b20cd80079d44802b3`.
