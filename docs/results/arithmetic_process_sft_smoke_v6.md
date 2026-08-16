# Arithmetic Process SFT Smoke v6 Result

## Result

V6 passes its pre-registered local process-contract smoke:

- baseline strict exact / process semantic:
  28/32 /
  28/32;
- post strict exact / process semantic:
  32/32 /
  32/32;
- baseline final-answer correct: 32/32;
- post final-answer correct: 32/32;
- early five-step mean loss: 0.001741121;
- late five-step mean loss: 0.000000474;
- peak training memory: 20.14 GiB;
- wall time: 415.6 seconds;
- independent reload exact / process semantic:
  32/32 /
  32/32.

All 40 losses and all 224 FP32 adapter tensors
are finite. No post-SFT output reaches the 80-token cap.

## Mechanism Interpretation

All 32 baseline and all 32 post-SFT outputs have the correct numeric `FINAL`.
The four baseline process failures are:

- final-correct process-contract mismatches:
  4;
- other process failures:
  0.

They combine operations or omit a canonical STEP rather than produce a wrong
final answer. V6 improves process-contract adherence from 28/32 to 32/32, but
does not improve final-answer accuracy beyond the 32/32 baseline.

Therefore this result must not be described as arithmetic reasoning uplift.
It is evidence that the process objective teaches the requested decomposition
and verifier contract.

## Decision

V6 passes every frozen local gate, including reload, finite tensors, memory,
loss trend, strict exact, process semantic, and generation budget.

This authorizes only evaluation of the unchanged adapter on frozen matched
benchmarks with task-level non-regression. It does not authorize an arithmetic
uplift claim, merge, scale-up, or RL.

## Reproduction Identity

- pre-registration revision: `f24891b`;
- data revision: `f1dcbe2`;
- config SHA256: `f8ab480d0195527b3fe8d98bb49ee377ba444257dcfe203de50c720d06624447`;
- dataset SHA256: `0e53fb3d05fb60569a4109da05b66d93c1158f734495e0126a55cf195c41653a`;
- model config SHA256: `ddc63e1c717afa86c865bb5e01313d89d72bb53b97ad4a8a03ba8510c0621670`;
- metrics SHA256: `4c6e6cc07984a5fe1fa260066ef00701eaea116df96113fdded9f670a6ac021c`;
- generations SHA256: `fee4c3dded8441b3d5a00571ba55e2f821ae91e8b3b911ef60b92ec3b0a448b9`;
- reload receipt SHA256: `2909330dbfddd50feeef24ef3b99b78a4db580c134ced22188f50e4dd5b61050`;
- adapter tree SHA256: `49f08829e06aa75c1cf6e5f16891bf79378011b8fe874fde4e392f5fcb5aa083`.
