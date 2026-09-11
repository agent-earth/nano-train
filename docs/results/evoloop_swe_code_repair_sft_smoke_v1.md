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
- Independent reload reproduced the loss and all 12 generations exactly.

## What Failed

The model usually emitted a JSON-looking response, often inside a Markdown
fence, but the real Harbor parser rejected the command entries. The base model
mostly produced command strings. The LoRA shifted many outputs to
`command`/`output` objects, while Terminus requires `keystrokes`/`duration`
objects. Only one adapter response exhausted the 768-token budget, so simple
truncation is not the main cause.

## Conclusion

Standard q/v-only SFT lowered held-out teacher-forced loss by 10.75% but did not teach the executable Terminus command schema. The dominant failure moved from command strings to command/output objects, while Harbor requires keystrokes/duration objects.

Loss reduction alone is insufficient evidence for agent quality. The next run
must change the supervision objective, use a fresh held-out slice, and preserve
the current v1 artifacts as negative evidence.

## Boundary

This is a local held-out training result, not a SWE-bench score. It rejects one SFT recipe and supports only the next mechanism-level experiment.
