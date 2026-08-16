# Format Contract SFT Smoke v2 Result

## Result

The FP32 numerical repair succeeds, but the pre-registered SFT smoke still
fails its full acceptance rule.

- all 20 optimizer-step losses are finite;
- baseline exact validation: 23/26
  (0.8846);
- post-SFT exact validation: 25/26
  (0.9615);
- initial loss: 0.149361;
- minimum observed loss: 0.016843;
- final loss: 0.247657;
- peak training memory: 18.14 GiB;
- adapter reload: 25/26 exact, matching in-process validation.

The adapter contains 224 FP32 LoRA tensors and zero
non-finite tensors. FP32 fixes v1's first-backward instability and improves
validation from 23/26 to 25/26.

## Remaining Failure

Validation does not reach 26/26. The sole failed synthetic sample is
`synthetic-db1039faf1f1b223cfb5`, a two-step arithmetic-precedence numeric example. The generated
answer obeys the exact `FINAL:` format but is semantically wrong.

The final sampled batch loss is also above the initial sampled batch loss, so
the frozen loss-decrease condition does not pass. Do not treat minimum loss as
a substitute for the pre-registered final-loss rule.

## Decision

v2 is a numerically stable, reloadable adapter with directional format
improvement, but it is not accepted for benchmark evaluation, merge, scale-up,
or RL. Preserve it as evidence. The next experiment must be a separately
pre-registered data/curriculum ablation targeting the remaining two-step
semantic weakness.

## Reproduction Identity

- pre-registration revision: `4468606`;
- config SHA256: `62cc5189cb048fd1a2b4070ffdd27b0a18c3363df1ae8dfa244a381401646207`;
- dataset SHA256: `46f2128f219db7011d5db95b5ca3a97029b57f5ac959e194860b4c0f4ba3ad53`;
- model config SHA256: `ddc63e1c717afa86c865bb5e01313d89d72bb53b97ad4a8a03ba8510c0621670`;
- metrics SHA256: `91a6999e60e5580fe4bfbbedc07f7b7845d4b51c33464334a2c33dd08fafa2dd`;
- generations SHA256: `907708b992238ef10ed186dd85928160fc51dcd5df733c1362ac3fcd394b0bf9`;
- reload receipt SHA256: `c8f6dfa5319e5e502196d41ee87573ec5c95e59dffeb787b3a1a833d8bc3d292`;
- adapter tree SHA256: `f02e7b9ae551649b6057e28e81a93f8b5166d0f1cf6780077ba7355ebddabf31`.
