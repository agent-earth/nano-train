# EvoLoop SWE Code-Repair SFT Smoke v1

## Frozen Run

- Model: Qwen3.5-4B, FP32 q/v-only LoRA r=8.
- Training: 160 unique rows, 40
  optimizer steps, accumulation 4.
- Development: 12 held-out rows, four per length band.
- Longest selected sequence:
  1993 / 2048 tokens.
- Dataset and release identities are pinned in the machine-readable receipt.
- Every selected target is a valid Terminus JSON action.

## Admission

The adapter reaches fresh-8 only if held-out teacher-forced loss improves by at
least 1.0%, at least
50% of held-out generations are
valid JSON actions, all numerical and artifact gates pass, and an independent
reload reproduces both held-out loss and all 12 generation receipts.

## Boundary

The source is the public SWE-bench train split. The data release separately
proved zero exact instance, problem, patch, and near-problem overlap with all
500 SWE-bench Verified cases. This smoke does not itself prove benchmark
improvement. Full-500, RL, OPD, and post-hoc tuning remain closed.
