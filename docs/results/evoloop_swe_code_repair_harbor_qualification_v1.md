# EvoLoop SWE Code-Repair Harbor Qualification v1

## Verdict

**REJECT UNDER THE FROZEN GATE.**

- Fresh held-out set: 24 rows, zero overlap
  with the 12 rows observed in SFT smoke v1.
- Held-out loss: 1.273458 -> 1.117471
  (+12.25% relative).
- Harbor parser validity: base
  24/24, adapter
  24/24.
- Paired actionable outcomes: adapter-only
  0, base-only
  0, both
  24.
- No adapter output exhausted the 768-token budget.

## What This Proves

On 24 fresh held-out cases, the unchanged adapter reduced teacher-forced loss by 12.25% and preserved perfect 24/24 Harbor-parser validity with zero base-only losses. The base model also scored 24/24, so the pre-registered requirement for at least one adapter-only parser repair could not pass. The structural proxy has saturated and cannot establish incremental agent quality.

The adapter does not regress the deployed response contract, but this proxy is
at ceiling. It cannot justify opening fresh-8 by changing the rule after
seeing the result.

## Next Experiment

Train one separately pre-registered adapter on the exact Harbor first-turn prompt while freezing optimizer, LoRA, step count, and source release. Compare base, standard-SFT v1, and Harbor-prompt SFT v2 on a third fresh held-out set before benchmark access.

## Boundary

This is a local parser/loss qualification, not a SWE-bench score. It rejects admission under the frozen gate despite directional loss improvement and perfect structural validity.
