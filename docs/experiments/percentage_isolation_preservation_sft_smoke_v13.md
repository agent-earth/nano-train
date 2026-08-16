# Percentage Isolation Preservation SFT Smoke v13

## Hypothesis

V12 mixes three failure-targeted numeric families and regresses choice and
strict behavior without improving numeric semantic accuracy. V13 isolates the
most direct arithmetic category error: percentage increase versus resulting
total composition.

If the narrower seven-example exposure preserves v11 choice/strict behavior,
the broad v12 regression is attributable to replacement dose or one of the
deferred families rather than any failure-targeted supervision.

## Single Data Intervention

Relative to v11, only dataset and output identity change:

- `targeted-preservation-mix-v6` at revision `ba11804`;
- to `percentage-isolation-preservation-mix-v8` at revision `a8db4f4`.

V8 changes exactly 8 numeric train rows and keeps the other 184 rows
byte-identical to v6. The deterministic 32-step schedule exposes 7 percentage
rows.

Packing efficiency, recurring schedule, and the choice-domain abstraction are
deferred. Model, FP32, LoRA, seed, batch, optimizer, LR, warmup, max steps,
sequence/generation budgets, gates, and reload checks remain frozen.

## Evaluation Boundary

- the 32-row local split is unchanged development evidence;
- no benchmark/canary payload enters training;
- the old canary runs only after local pass;
- the old 211-case suite runs only after canary pass;
- the new independent holdout stays unread until old-suite base non-regression;
- independent holdout prompts and references are not loaded during training.

## Frozen Local Gate

- 32 finite steps and decreasing early/late mean loss;
- aggregate semantic at least 24/32;
- numeric at least 10/16;
- choice at least 5/8;
- process at least 7/8;
- every family above base;
- strict exact at least 22/32;
- zero capped outputs;
- finite tensors, exact independent reload, memory below 28 GiB.

Passing authorizes only the old sealed regression canary. Merge, scale-up, and
RL remain forbidden.

## Reproduction

```bash
PYTHONPATH=. ../.venv/bin/python scripts/validate_sft_smoke.py \
  --config configs/sft/percentage_isolation_preservation_smoke_v13.json
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. ../.venv/bin/python -m nano_train.cli \
  sft-smoke --config configs/sft/percentage_isolation_preservation_smoke_v13.json
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. ../.venv/bin/python \
  scripts/validate_sft_adapter.py \
  --config configs/sft/percentage_isolation_preservation_smoke_v13.json
```

## Frozen Identity

- data revision: `a8db4f4`;
- dataset SHA256:
  `0ae81bb4c385703592946b5c75971b39cbb388b02a76fafa477e53e55756bc9c`;
- config SHA256:
  `98057d4ea24e3d24ada9d98c0dd5af14fc1f08bb07436e1a33bd479a4131686e`;
- model config SHA256:
  `ddc63e1c717afa86c865bb5e01313d89d72bb53b97ad4a8a03ba8510c0621670`;
- independent holdout manifest SHA256:
  `181bc26725ff55fbf929cb09ca9484207ef73fcd20780a17c5702755a40ca0bb`;
- holdout selection receipt SHA256:
  `33239ffc09965981088c7de0af2a98e072e3c02ac9219795dd325b603c990e1d`.
