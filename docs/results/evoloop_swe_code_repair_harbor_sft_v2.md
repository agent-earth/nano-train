# EvoLoop Harbor-Prompt SFT v2 Result

## Verdict

**ADMIT TO FROZEN FRESH-8 SCREENING.**

- Same 160 train rows as v1 and the same
  optimizer/LoRA settings; only the deployed prompt shape changed.
- Training completed 40 finite optimizer steps with
  917,504 trainable parameters.
- Fresh 24-case loss: base 1.305525, standard SFT v1
  1.243614, Harbor-prompt SFT v2
  1.116045.
- v2 loss improvement: +14.51% versus
  base and +10.26% versus v1.
- Terminus parser validity: base
  24/24, v1
  24/24, v2
  24/24.
- v2 versus v1 structural regressions:
  0.
- Independent reload reproduced v2 loss and all 24 structural generation rows.

## What This Proves

The train/deploy prompt mismatch was material. Aligning supervision to the
actual Harbor interaction reduced fresh held-out loss substantially without
breaking the command protocol. All frozen local gates pass.

## What It Does Not Prove

Parser validity and teacher-forced loss are proxies. They do not show that the
agent diagnoses repositories, writes correct patches, or passes hidden tests.
Only official frozen fresh-8 scoring can test that.

## Next

Serve adapter `d53815f016eb6d246cc74ef8fb9daeab24fbb3e88c4642f6a4d14603b54a7010` and run the preregistered
fresh-8 SFT-only arm. Do not launch the complete 500 cases from this local
result alone.
