# Anchored-v1 Choice Replay Continuation v2 Result

## Result

V2 is numerically stable but fails its frozen local choice gate.

- baseline and post strict / semantic: 22/32 /
  25/32 and 22/32 /
  25/32;
- post numeric / choice / process semantic:
  11/16, 6/8,
  8/8;
- relative LoRA B drift: 0.051039;
- training / reload peak memory:
  18.26 / 15.81 GiB;
- independent reload exactly reproduces metrics and failure IDs.

The only failed gate is choice >=7/8: the result remains 6/8.

## Mechanism Evidence

All 112 LoRA A tensors remain byte-identical, all 112 B tensors change, and
all adapter tensors are finite. Of 32 development outputs, exactly one changes:
`synthetic-cb1b63a543f33dcb72fe` moves from `FINAL: A` to
`FINAL: C` while the synthetic target is
`FINAL: B`. The update moves a choice decision boundary but does
not fix a case; strict, semantic, numeric, choice, and process scores are all
unchanged.

Stop this supervised replay path. Do not search a larger replay dose on the
same development split.

## Decision

Reject v2 and preserve anchored-v1. The sealed canary, old full-development
suite, and independent holdout were not run. The holdout remains unread.

## Reproduction Identity

- pre-registration revision / tree: `277b46f` /
  `818dcce54a6fff99d8a20bf14dded061a4e06d42`;
- data revision: `744965a`;
- config SHA256: `afb70e3c2a7008bc4c6175ed1d988a5d99b14465320a2c18918848728bfaad16`;
- dataset SHA256: `4657e96af9f9d1b81bfdb5fac6a29c31baf24b23d7753508aafdea5603ffd80d`;
- anchor adapter tree SHA256: `d29963cbd9284fe9a21de690babecbb34ed3fbbedd16d5d780daaf8abc45bc82`;
- candidate adapter tree SHA256: `a3ad49aff01a1fd05c7f357dfef587fbb25a1cc2b4cb1d35988895c060c7f4de`;
- metrics SHA256: `cc671ce1ec055709a27a683bc91097dd9d52943e332f0ad4f14288a67ab800dc`;
- generations SHA256: `0a32a565056fe24e75834b92c0ce93379b322c68ffad547621b3491aebbe462c`;
- reload receipt SHA256: `c5506c8dca5a4e13ab437f242b637bd495da4c6ff0f070df38ea907f01abf453`.
