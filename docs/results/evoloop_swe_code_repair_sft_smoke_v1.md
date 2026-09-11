# EvoLoop SWE Code-Repair SFT Smoke v1

## Verdict

**REJECT.** This adapter is reproducible, but it is not eligible for fresh-8.

- Data release: 11,000 train rows,
  11,861,953 Qwen3.5 tokens, and zero exact or
  near overlap with all 500 SWE-bench Verified cases.
- Training exposure: 160 unique rows across
  40 optimizer steps.
- Held-out loss: 1.356485 ->
  1.210654
  (+10.75% relative).
- Strict JSON action validity: 0/
  12.
- Harbor Terminus parser validity:
  0/12.
- Exact Harbor prompt ablation: base
  11/12
  versus LoRA
  12/12
  parser-valid responses.
- Independent reload reproduced the loss and all 12 generations exactly.

## What Failed

Under the generic local prompt, the model usually emitted a JSON-looking
response but used the wrong command schema. Replacing only that prompt with the
exact Harbor first-turn template changed parser validity from 0/12 to 11/12 for
the base model and 12/12 for the adapter. This proves the original local gate
was badly misaligned with the deployed harness. It does not retroactively pass
v1, because the prompt changed after observation.

## Conclusion

Standard q/v-only SFT lowered held-out teacher-forced loss by 10.75%. Under the generic local prompt it did not emit the executable Terminus command schema, but under the exact Harbor first-turn prompt the same frozen adapter reached 12/12 parser-valid outputs versus 11/12 for the base model. This identifies prompt-schema alignment as the dominant local proxy mismatch; because the cases were already observed, it remains diagnostic rather than admission.

Loss reduction alone is insufficient evidence for agent quality. The next run
must change the supervision objective, use a fresh held-out slice, and preserve
the current v1 artifacts as negative evidence.

## Boundary

This is a local held-out training result, not a SWE-bench score. It rejects one SFT recipe and supports only the next mechanism-level experiment.
