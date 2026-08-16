# Hard Preservation SFT Smoke v10 Result

## Result

V10 is stable but exactly matches v9 validation and fails the unchanged local
gate.

- aggregate exact / semantic: 22/32 / 22/32;
- numeric exact / semantic: 9/16 /
  9/16;
- choice exact / semantic: 5/8 /
  5/8;
- process exact / semantic: 8/8 /
  8/8;
- early / late five-step mean loss:
  0.197263 / 0.024580;
- peak training memory: 20.17 GiB;
- independent reload exact / semantic:
  22/32 /
  22/32;
- post outputs at the cap:
  0/32.

All 32 losses and all 224 FP32 adapter tensors
are finite. The independent reload reproduces every family metric and failure
sample ID.

## Dose Ablation

| Run | Steps | Aggregate | Numeric | Choice | Process |
| --- | ---: | ---: | ---: | ---: | ---: |
| v7 | 20 | 19/32 | 6/16 | 5/8 | 8/8 |
| v9 | 30 | 22/32 | 9/16 | 5/8 | 8/8 |
| v10 | 32 | 22/32 | 9/16 | 5/8 | 8/8 |
| v8 | 40 | 21/32 | 9/16 | 4/8 | 8/8 |

The two additional steps and eight additional examples produce no validation
change relative to v9. More max-step interpolation is not justified.

## Decision

V10 reaches strict 22/32, choice 5/8, and process 8/8, but fails aggregate
24/32 and numeric 10/16. Do not run the sealed canary. Do not run the full
suite, merge, scale up, or start RL.

Stop this dose-search line. Preserve the choice and process strata, analyze
the seven persistent numeric failures, and pre-register a genuinely new
non-evaluation numeric-data intervention before another SFT smoke.

## Reproduction Identity

- pre-registration revision: `528f7aa`;
- data revision: `204b053`;
- config SHA256: `49c5d50572bb568235fd25e4ad5882b381facc795e6131196423f829985c8910`;
- dataset SHA256: `ac07e10f1e04ca8ddec74aefc8df9b00475f7fc9c57345ff98182dd8e4c8bae9`;
- model config SHA256: `ddc63e1c717afa86c865bb5e01313d89d72bb53b97ad4a8a03ba8510c0621670`;
- metrics SHA256: `902f4982ba3e44e90486c7c36b14f9326f544d7bae55c386c13e123cfc40dcad`;
- generations SHA256: `404969c874612ac04321c3c502be530846974e2a63cfe20e29928899ba4e3b8a`;
- reload receipt SHA256: `3b6774babd02256141bb5983d1cc41fea4f878460c46b49a6a7fed5d3c3f7b48`;
- adapter tree SHA256: `a0b13668ba5034b155a8aecdc8f45df039f7580773c6b9ff86f2f3727917495d`.
