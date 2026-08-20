# Orca Math Verifier DPO v1 Result

## Verdict

**REJECT: STABLE NO-OP.**

- Baseline / post-DPO: 91/192 ->
  91/192;
- changed outputs: 0/192;
- paired delta: +0.0000;
- 95% CI:
  [+0.0000,
  +0.0000];
- candidate-only / baseline-only:
  0 /
  0.

## Training

- 32 fresh preference pairs, one optimizer step each;
- loss 0.693147182 ->
  0.693144441;
- preference advantage 0 ->
  5.4359436e-05;
- all losses and gradient norms finite;
- independent reload reproduced all 192 generations exactly.

## Conclusion

Chosen and rejected targets differ only in the FINAL suffix. Averaging log probability over the long shared trajectory diluted that signal: the final preference advantage remained near zero and no development output changed.

Do not tune beta, LR, steps, seed, selection, parser, or LoRA scope on this
observed dev. The next experiment must use disjoint pairs and dev rows, and
score only the differing FINAL suffix tokens.

## Boundary

This is a local synthetic-development preference-optimization result. It is not a GSM8K, MMLU, GPQA, 9B, 27B, or agent-benchmark result.
