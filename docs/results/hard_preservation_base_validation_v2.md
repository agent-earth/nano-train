# Hard Preservation Base Validation v2

## Result

At a sufficient 128-token generation budget:

- aggregate exact / semantic: 8/32 /
  8/32;
- numeric exact / semantic:
  0/16 /
  0/16;
- choice exact / semantic:
  2/8 /
  2/8;
- process exact / semantic:
  6/8 /
  6/8;
- outputs at the 128-token cap:
  0/32.

No training is performed.

## Failure Diagnostic

The non-scoring loose-final diagnostic does not change official metrics:

- numeric: 11/16 have the correct final number,
  while 5/16 have a wrong final number;
- choice: 2/8 have the correct letter and
  6/8 are wrong;
- process: 8/8 have the correct final number,
  while only 6/8 satisfy the process verifier.

The data has both output-contract and genuine capability headroom.

## Budget Audit

The earlier 80-token audit has
8/32 capped outputs. Raising only
the generation budget to 128 removes all caps and leaves official exact and
semantic metrics unchanged at 8/32. V2 is therefore the valid pre-training
baseline.

## Decision

The audit does not automatically authorize training. A separately
pre-registered conservative mixed-preservation SFT may proceed only with
family-level gates, followed by the sealed 40-case regression canary before
any full benchmark. Merge, scale-up, and RL remain forbidden.

## Reproduction Identity

- pre-registration revision: `a58b2be`;
- data revision: `204b053`;
- dataset SHA256: `ac07e10f1e04ca8ddec74aefc8df9b00475f7fc9c57345ff98182dd8e4c8bae9`;
- model config SHA256: `ddc63e1c717afa86c865bb5e01313d89d72bb53b97ad4a8a03ba8510c0621670`;
- v2 config SHA256: `41389316f1979750161132c6e5c16c7da4051f4d61877188fbbb3de3d5f97905`;
- local result SHA256: `e03d63dbdb6c1465e8b2005fff0426b00850b68d0e8e5a91b7a117502c3e5293`;
- local generations SHA256:
  `614d10e61f3bbaebdbea3a2bc5a93f68a368c5969f3a7e66c619b06300a4baf9`.
