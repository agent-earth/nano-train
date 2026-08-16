# Hard Preservation SFT Smoke v8

## Hypothesis

V7 improves every validation family but leaves numeric semantic at 6/16 after
only half a training-set coverage equivalent. V8 tests one complete coverage
equivalent without changing data composition or any other training field.

## Single-Variable Intervention

Relative to v7, the only training config field that changes is:

- `max_steps`: 20 to 40.

`experiment_id` and `output_dir` change only to isolate artifacts. Dataset,
FP32, model, seed, LoRA, batch, optimizer, LR, warmup, sequence length,
generation budget, fail-fast, validation, and reload behavior remain frozen.

The deterministic order exposes:

- v7: 80 unique samples, including numeric 42, choice 20, process 18;
- v8: all 160 unique train samples, including numeric 80, choice 40, process
  40.

`max_steps` also defines the linear LR decay horizon. This ablation measures
the full 40-step schedule and cannot separately attribute effects to complete
coverage versus the longer derived schedule.

## Identity

- data revision: `204b053`;
- dataset SHA256:
  `ac07e10f1e04ca8ddec74aefc8df9b00475f7fc9c57345ff98182dd8e4c8bae9`;
- model config SHA256:
  `ddc63e1c717afa86c865bb5e01313d89d72bb53b97ad4a8a03ba8510c0621670`.

## Frozen Gate

V8 uses the same local gate as v7:

- 40 finite optimizer steps and no failure receipt;
- late five-step mean loss below early five-step mean;
- aggregate semantic at least 24/32;
- numeric semantic at least 10/16;
- choice semantic at least 5/8;
- process semantic at least 7/8;
- every family improves over base;
- strict exact at least 22/32;
- zero outputs at the 128-token cap;
- finite adapter tensors and exact independent reload;
- peak memory below 28 GiB.

Passing authorizes only the sealed 40-case canary. The canary remains excluded
from training and cannot establish uplift. Full suite, merge, scale-up, and RL
remain forbidden until later gates pass.

## Reproduction

```bash
PYTHONPATH=. ../.venv/bin/python scripts/validate_sft_smoke.py \
  --config configs/sft/hard_preservation_smoke_v8.json
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. ../.venv/bin/python -m nano_train.cli \
  sft-smoke --config configs/sft/hard_preservation_smoke_v8.json
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. ../.venv/bin/python \
  scripts/validate_sft_adapter.py \
  --config configs/sft/hard_preservation_smoke_v8.json
```
