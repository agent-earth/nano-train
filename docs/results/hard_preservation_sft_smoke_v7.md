# Hard Preservation SFT Smoke v7 Result

## Result

V7 is numerically stable and improves every family, but fails its frozen local
promotion thresholds.

- aggregate exact / semantic:
  8/32 / 8/32 to
  19/32 / 19/32;
- numeric exact / semantic:
  0/16 /
  0/16
  to 6/16 / 6/16;
- choice exact / semantic:
  2/8 /
  2/8
  to 5/8 / 5/8;
- process exact / semantic:
  6/8 /
  6/8
  to 8/8 / 8/8;
- early / late five-step mean loss:
  0.197253 / 0.058721;
- peak training memory: 20.10 GiB;
- independent reload aggregate exact / semantic:
  19/32 /
  19/32;
- post outputs at the 128-token cap:
  0/32.

All 20 losses and all 224 FP32 adapter tensors
are finite.

## Failure Analysis

V7 reaches the choice threshold 5/8 and process threshold 8/8. Numeric reaches
only 6/16 rather than the required 10/16. The post-SFT loose-final diagnostic
also reports 6 numeric final answers correct and 10 wrong, so the remaining
numeric gap is semantic, not a format or generation-budget confounder.

Aggregate semantic is 19/32 rather than at least 24/32,
and strict exact is 19/32 rather than at least 22/32.

## Decision

V7 fails the local gate. Do not run the sealed 40-case canary. Do not run the
full suite, merge, scale up, or start RL.

The next experiment must be separately pre-registered and target numeric hard
examples or dose while keeping the successful choice and process strata
frozen. V7 is a combined data-plus-dose intervention and cannot support causal
attribution without later ablation.

## Reproduction Identity

- pre-registration revision: `f5344de`;
- data revision: `204b053`;
- config SHA256: `787649d577e3978311c968b9d886ae2188a2a6ff9fcc7f6c00e79f5bfb896c08`;
- dataset SHA256: `ac07e10f1e04ca8ddec74aefc8df9b00475f7fc9c57345ff98182dd8e4c8bae9`;
- model config SHA256: `ddc63e1c717afa86c865bb5e01313d89d72bb53b97ad4a8a03ba8510c0621670`;
- metrics SHA256: `156a70ff503acdf30d0b93916aee11098bd2b9a9702dcb88032d4cfc41801f85`;
- generations SHA256: `ca3a14683033e3dbf9c38a9daf6ce811aa2b2560f4f36371eb01cf13eb61216f`;
- reload receipt SHA256: `8e52f7a6ce94f775ea5509b991c23a5e9802634d2d1f82cdb5a71240c7e9c25c`;
- adapter tree SHA256: `279c33fd05e987f6312be01da86be3aae9dbefd05622fbeb6dc1d83e2cda6ccc`.
