# Orca Math SFT Smoke v1 Pre-Registration

## Frozen Run

- Train: 160 unique rows across 40 optimizer steps
  and four gradient-accumulation micro-batches;
- development: 192 untouched rows;
- max selected sequence:
  1012 / 1024 tokens;
- model: Qwen3.5-4B, FP32 q/v-only LoRA r=8;
- generation: greedy, batch 4,
  up to 384 new tokens;
- scorer: strict final-line numeric equivalence;
- statistics: 10,000 paired bootstrap samples
  and exact McNemar alpha 0.05.

## Admission

positive point delta, positive paired-bootstrap lower bound, exact McNemar p below alpha, at least six candidate-only wins, candidate-only > baseline-only, and every difficulty stratum non-regressing.

Passing unlocks only a separately reviewed next step. Benchmark, independent
holdout, RL, OPD, and post-hoc tuning remain closed.

## Boundary

This commit selects no benchmark rows and starts no training or generation.
