# Semantic Arithmetic v5 Budget Audit v1

## Purpose

The official v5 result remains strict exact 12/32 and semantic valid 12/32 at
the pre-registered 32-token generation budget. A post-run contract audit found
that validation target content reaches 37 tokens and target plus EOS reaches
38 tokens. Fourteen validation targets exceed the official content budget.

This separately named evaluation-only audit asks how the unchanged v5 adapter
behaves when it can finish every target-length response. It does not rescore or
replace v5.

## Frozen Intervention

The audit changes only evaluation `generation_max_new_tokens` from 32 to 48.
It performs no optimizer step, writes no adapter tensor, and evaluates:

- the unchanged Qwen3.5-4B base model;
- the unchanged v5 adapter.

The audit freezes:

- all 32 validation case IDs and targets;
- tokenizer and greedy decoding;
- model, dataset, and verifier;
- v5 adapter tree;
- v5 config, metrics, generations, and reload receipt.

Identity:

- audit config SHA256:
  `e1cb92f5b6a1f70b70caf806318280c24489aa1ac8622f5c1c459ee77957d465`;
- v5 adapter tree SHA256:
  `7ecb48dad68b0a7499baefcfeb587ce72ecf85df60a8a7c338c25b3d464f3421`;
- v5 metrics SHA256:
  `696d00d59c65f861cde9c93f2a31fd106ffcfd16bfc8e0e1fbbb5c9d015f4c9e`;
- v5 generations SHA256:
  `6210f9ee7cb8a03377dc6588c82f3c79012d6104465cab7bdeb739035adc68d6`;
- v5 reload SHA256:
  `7f3cdb37a1d1ef3d8cc8519488e83b9ac612d45b4203a4ed5b5757ec4b717fa2`;
- dataset SHA256:
  `d226f243051b7d2d2d4db4d5a596b871032fa44d71b296586f879559a8781c09`;
- model config SHA256:
  `ddc63e1c717afa86c865bb5e01313d89d72bb53b97ad4a8a03ba8510c0621670`.

## Pre-Registered Interpretation

- The official v5 result stays 12/32 regardless of audit outcome.
- The audit is descriptive and cannot pass the v5 training gate.
- If 48-token adapter accuracy materially exceeds 12/32 and truncation-shaped
  failures disappear, repair the evaluation contract before choosing another
  training intervention.
- If execution mismatches remain, the next data/training objective must target
  arithmetic execution rather than output length.
- Benchmark evaluation, merge, scale-up, and RL remain forbidden after this
  audit because v5 did not pass its original gate.

## Reproduction

```bash
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. ../.venv/bin/python \
  scripts/run_generation_budget_audit.py \
  --config configs/audits/semantic_arithmetic_v5_budget_audit_v1.json
```

## Result

At 48 generated tokens, the base model remains 3/32 strict exact and 4/32
semantic valid. The unchanged v5 adapter reaches 14/32 on both metrics, versus
the official 12/32 at 32 tokens.

The unchanged adapter emits at most 37 tokens and has zero outputs at the
48-token cap. Its failure taxonomy becomes 14 semantic-valid and 18 arithmetic
execution mismatches, with zero CALC/FINAL mismatches and zero invalid traces.
Of the 13 official CALC/FINAL mismatches, 2 become valid and 11 become complete
but arithmetically wrong traces. The one invalid official trace also becomes a
complete arithmetic execution mismatch.

Truncation is material but not the primary bottleneck. No training occurred,
the adapter was not modified, and official v5 remains 12/32. Benchmark
evaluation, merge, scale-up, and RL remain unauthorized.

Public result:

- `docs/results/semantic_arithmetic_v5_budget_audit_v1.md`;
- `docs/results/semantic_arithmetic_v5_budget_audit_v1.public.json`.
