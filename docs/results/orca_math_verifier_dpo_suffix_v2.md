# Orca Math Suffix DPO v2 Result

## Verdict

**REJECT: STRONGER MARGIN, ZERO BEHAVIOR CHANGE.**

- Baseline / post-DPO: 83/192 ->
  83/192;
- changed outputs: 4/192;
- correctness / parse-status changes:
  0 /
  0;
- paired delta: +0.0000;
- candidate-only / baseline-only:
  0 /
  0.

## Training

- 32 fresh pairs and 192 fresh dev rows, all disjoint from DPO v1;
- only 2-
  7 differing suffix tokens per arm were
  scored;
- preference advantage 0 ->
  0.00617694855, maximum
  0.00617694855;
- all losses and gradient norms finite;
- independent reload reproduced all generations exactly.

Attempt 1 failed before any optimizer step because two reference forwards were
kept live simultaneously and exhausted GPU memory. The repair used
mathematically equivalent split backward coefficients; config, selection,
objective, and thresholds were unchanged.

## Conclusion

Masking shared trajectory tokens increased preference advantage by about two orders of magnitude versus v1, but the low-dose adapter changed only four output strings and changed no score or parse status. The training signal is now targeted but not large enough to move measured behavior under the frozen smoke; post-hoc dose or LR tuning is forbidden.

Stop this preference-training family. The next experiment returns to the frozen
base model and tests full-solve self-consistency on fresh local data.

## Boundary

This is a local synthetic-development preference-optimization result. It is not a GSM8K, MMLU, GPQA, 9B, 27B, or agent-benchmark result.
