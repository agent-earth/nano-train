# Failure-Targeted Preservation SFT Smoke v12

## Hypothesis

V11 improves local families and GPQA but scores 162/211 versus base 4B at
163/211. Its four base-only discordances reduce to three numerical abstractions
and one choice-domain concept.

V12 tests fresh verifier-backed supervision for the three numerical
abstractions while preserving v11's host-count, choice, process, and
development strata. The choice-domain concept is deferred to avoid narrow
benchmark memorization.

## Single Data Intervention

Relative to v11, only dataset and output identity change:

- `targeted-preservation-mix-v6` at revision `ba11804`;
- to `failure-targeted-preservation-mix-v7` at revision `3b4f76f`.

V7 replaces exactly 24 numeric training slots in place:

- 8 percentage-increase total-composition rows;
- 8 packing-efficiency effective-volume rows;
- 8 weighted recurring-schedule rows.

All 32 development rows and the other 136 train rows are byte-identical to v6.
The deterministic 32-step schedule exposes 19 new rows:

- percentage composition: 7;
- packing efficiency: 5;
- recurring schedule: 7.

Model, FP32, LoRA, seed, batch, optimizer, LR, warmup, max steps, sequence and
generation budgets, family gates, reload checks, and canary policy remain
frozen.

## Leakage And Evaluation Boundary

V7 consumes only an irreversible receipt containing four abstract labels and
source-set hashes. It contains no benchmark/canary case IDs, prompts,
references, predictions, outputs, or reversible payloads.

- benchmark and canary rows are training-ineligible;
- independent holdout rows are not loaded or used for training;
- all replacements are deterministic synthetic data;
- all numeric targets are restricted-AST verified;
- the 32-row local split is development evidence only.

A new independent holdout is frozen as 16 unseen GSM8K, 16 unseen MMLU, and 8
unseen GPQA-Diamond source indices. Selection uses only prior source-index
coverage and GPQA question length eligibility. Its prompts and references have
not been loaded.

The independent holdout must remain unread until the unchanged adapter passes:

1. this local development gate;
2. the post-v6-calibrated 40-case regression canary;
3. base-4B non-regression on the prior 211-case development suite.

## Frozen Local Gate

V12 must meet the unchanged local gate:

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
  --config configs/sft/failure_targeted_preservation_smoke_v12.json
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. ../.venv/bin/python -m nano_train.cli \
  sft-smoke --config configs/sft/failure_targeted_preservation_smoke_v12.json
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. ../.venv/bin/python \
  scripts/validate_sft_adapter.py \
  --config configs/sft/failure_targeted_preservation_smoke_v12.json
```

## Frozen Identity

- data revision: `3b4f76f`;
- dataset SHA256:
  `b9dcbec512831a3f2c96e7db5abf4a0750420f26a28cc0f2a27699661f79aa23`;
- config SHA256:
  `82ad3ca17fc23e5722fead74cf9387364183db2cda8493ed02474e0ef60d2d02`;
- model config SHA256:
  `ddc63e1c717afa86c865bb5e01313d89d72bb53b97ad4a8a03ba8510c0621670`;
- independent holdout manifest SHA256:
  `181bc26725ff55fbf929cb09ca9484207ef73fcd20780a17c5702755a40ca0bb`;
- holdout selection receipt SHA256:
  `33239ffc09965981088c7de0af2a98e072e3c02ac9219795dd325b603c990e1d`.
