# Hard Preservation SFT Smoke v10

## Hypothesis

V9 at 30 steps is Pareto best but remains one numeric and two aggregate cases
below the local gate. V10 tests the smallest practical dose increase: two
optimizer steps and eight deterministic training examples.

## Single-Variable Intervention

Relative to v9, only:

- `max_steps`: 30 to 32.

All data, model, FP32, LoRA, seed, batch, optimizer, LR, warmup, sequence and
generation budgets, family validation, reload checks, and canary rules remain
frozen.

The deterministic schedule exposes 128 unique examples:

- numeric: 66;
- choice: 30;
- process: 32.

As with prior dose experiments, `max_steps` also changes the linear LR decay
horizon.

## Frozen Gate

V10 must meet the unchanged local gate:

- 32 finite steps and decreasing early/late mean loss;
- aggregate semantic at least 24/32;
- numeric at least 10/16;
- choice at least 5/8;
- process at least 7/8;
- every family above base;
- strict exact at least 22/32;
- no capped outputs;
- finite tensors, exact reload, memory below 28 GiB.

Passing authorizes only the sealed 40-case canary. Full suite, merge, scale-up,
and RL remain forbidden until later gates pass.

## Reproduction

```bash
PYTHONPATH=. ../.venv/bin/python scripts/validate_sft_smoke.py \
  --config configs/sft/hard_preservation_smoke_v10.json
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. ../.venv/bin/python -m nano_train.cli \
  sft-smoke --config configs/sft/hard_preservation_smoke_v10.json
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. ../.venv/bin/python \
  scripts/validate_sft_adapter.py \
  --config configs/sft/hard_preservation_smoke_v10.json
```
