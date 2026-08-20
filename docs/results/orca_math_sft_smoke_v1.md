# Orca Math SFT Smoke v1 Result

## Verdict

**REJECT.**

- Baseline: 100/192
  (0.5208);
- post-SFT: 58/192
  (0.3021);
- paired delta: -0.2188;
- paired bootstrap 95% CI:
  [-0.2917,
  -0.1458];
- exact McNemar p: 1.57026e-08;
- candidate-only / baseline-only:
  8 /
  50.

## What Happened

- The training loss moved from 0.489840 to
  0.354532, with minimum 0.264968.
- All 160 training rows were unique and seen
  exactly once.
- Final-line parse failures improved from
  76 to
  68.
- Despite better format completion, the adapter repaired only
  8 cases and regressed
  50 previously correct cases.
- Independent reload reproduced every post-SFT generation and metric exactly.

This shows that loss reduction and fewer format failures did not transfer into
math quality. Standard SFT on 160 verbose teacher trajectories catastrophically
reduced final-answer correctness, especially in the medium stratum
(50/96 to 24/96).

## Decision

Reject the adapter and forbid rerun or hyperparameter tuning on the observed
development rows. The next method must use fresh non-benchmark data and a
verifier-guided RL/OPD objective rather than another standard SFT dose search.

## Boundary

This is a local synthetic-development SFT result. It is not a GSM8K, MMLU, GPQA, 9B, 27B, RL, OPD, or agent-benchmark result.
