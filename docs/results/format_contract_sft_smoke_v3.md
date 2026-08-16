# Format Contract SFT Smoke v3 Result

## Result

V3 is numerically stable and improves a fresh two-step-heavy validation split,
but fails its exact-validation acceptance rule.

- all 20 optimizer-step losses are finite;
- baseline validation: 13/32
  (0.4062);
- post-SFT validation: 20/32
  (0.6250);
- early five-step mean loss: 0.428718;
- late five-step mean loss: 0.222961;
- minimum loss: 0.053835;
- peak training memory: 18.33 GiB;
- independent adapter reload: 20/32.

All 224 FP32 adapter tensors are finite. The moving
average loss criterion passes and validation improves by seven cases.

## Remaining Failure

Post-SFT has 12 failures:

- 9 numeric two-step;
- 2 choice two-step;
- 1 choice single-step.

All 32 post-SFT outputs match their required `FINAL:` grammar. The remaining
gap is semantic arithmetic, not format compliance. Continuing format-only data
iteration would target the wrong mechanism.

## Decision

V3 fails because validation is 20/32 rather than 32/32. Do not benchmark,
merge, scale, or start RL. Stop format-only SFT iteration and design a
separately pre-registered semantic arithmetic objective.

## Reproduction Identity

- pre-registration revision: `f05699f`;
- config SHA256: `fee61ad70cec96368849b6873e7f261dbfc822dc82af7d206cfdb29b58edbfdd`;
- dataset SHA256: `95d8e3e8a173960fd8604f284bae0243e74f4c924c96b719252c8c9a6525f001`;
- model config SHA256: `ddc63e1c717afa86c865bb5e01313d89d72bb53b97ad4a8a03ba8510c0621670`;
- metrics SHA256: `9a52f5af7ded83c8198258b230c635e288ae077336b4ebcb1648335702c9075a`;
- generations SHA256: `ca075c648a2c8427589092b864046602c75a451fd807abc9ece3b288fe54a0fc`;
- reload receipt SHA256: `cd149c4e3b971b832cccee9511a74420746b30999af2d0036094075a92810034`;
- adapter tree SHA256: `5d193fcea89e0b297ef69fe2500d366508ca92bcc600fd97cdb5062d39b1e309`.
