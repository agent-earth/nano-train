# Hard Preservation SFT Smoke v9

## Hypothesis

V7 at 20 steps reaches numeric 6/16 and choice 5/8. V8 at 40 steps reaches
numeric 9/16 but choice falls to 4/8. V9 tests a 30-step interpolation while
freezing all data and gates.

## Single-Variable Intervention

Relative to v8, only:

- `max_steps`: 40 to 30.

Experiment identity and output path change only to isolate artifacts. All
other fields are byte-identical to v8.

The deterministic schedule exposes 120 unique training samples:

- numeric: 63/80;
- choice: 27/40;
- process: 30/40.

As in prior dose ablations, `max_steps` also changes the linear decay horizon.
The result cannot separate example exposure from the derived LR schedule.

## Frozen Gate

V9 uses the unchanged v7/v8 local gate:

- 30 finite steps, no failure receipt, and decreasing early/late loss means;
- aggregate semantic at least 24/32;
- numeric semantic at least 10/16;
- choice semantic at least 5/8;
- process semantic at least 7/8;
- every family improves over base;
- strict exact at least 22/32;
- zero capped outputs;
- finite adapter tensors and exact independent reload;
- peak memory below 28 GiB.

Passing authorizes only the sealed 40-case canary. Full suite, merge, scale-up,
and RL remain forbidden until later gates pass.

## Reproduction

```bash
PYTHONPATH=. ../.venv/bin/python scripts/validate_sft_smoke.py \
  --config configs/sft/hard_preservation_smoke_v9.json
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. ../.venv/bin/python -m nano_train.cli \
  sft-smoke --config configs/sft/hard_preservation_smoke_v9.json
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. ../.venv/bin/python \
  scripts/validate_sft_adapter.py \
  --config configs/sft/hard_preservation_smoke_v9.json
```
