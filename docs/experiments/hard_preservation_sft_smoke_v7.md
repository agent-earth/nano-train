# Hard Preservation SFT Smoke v7

## Hypothesis

V6 overfits a process-only objective: it passes synthetic validation but
regresses the matched suite from base 4B 163/211 to 145/211. V7 tests a mixed
hard curriculum with a lower training dose before any full benchmark.

## Data

`hard-preservation-mix-v5` contains 192 fresh non-evaluation samples:

- 96 hard numeric word problems covering omission, percentage-category,
  participant-average, and sequential-remainder boundaries;
- 48 answer-only choice contract samples;
- 48 verified process traces;
- 160 train and 32 validation;
- zero ID, exact, semantic, or source-signature overlap with v1-v4;
- no benchmark, canary, raw model, or teacher content.

Data revision: `204b053`.

## Training Intervention

Relative to v6:

- dataset changes from process-only v4 to mixed preservation v5;
- generation budget changes from 80 to 128 to cover uncapped baseline outputs;
- `max_steps` changes from 40 to 20, exposing 80/160 training examples.

`max_length=192`, FP32, model identity, seed, LoRA scope/rank/alpha/dropout,
effective batch 4, optimizer, LR, warmup, fail-fast, and reload checks remain
frozen.

This is a combined safety intervention, not a single-variable causal
experiment. If successful, later ablations must separate data composition from
training dose.

Identity:

- dataset SHA256:
  `ac07e10f1e04ca8ddec74aefc8df9b00475f7fc9c57345ff98182dd8e4c8bae9`;
- model config SHA256:
  `ddc63e1c717afa86c865bb5e01313d89d72bb53b97ad4a8a03ba8510c0621670`.

## Frozen Baseline

At 128 tokens, base validation is:

- aggregate exact / semantic: 8/32 / 8/32;
- numeric: 0/16 / 0/16;
- choice: 2/8 / 2/8;
- process: 6/8 / 6/8;
- zero capped outputs.

## Pre-Registered Local Gate

V7 passes local validation only if:

- all 20 optimizer steps have finite loss, gradients, and parameters;
- no failure receipt exists;
- late five-step mean loss is below early five-step mean;
- aggregate post-SFT semantic is at least 24/32;
- numeric semantic is at least 10/16;
- choice semantic is at least 5/8;
- process semantic is at least 7/8;
- every family improves over its base semantic baseline;
- post strict exact is at least 22/32;
- zero validation output reaches 128 tokens;
- all adapter tensors are finite;
- independent reload reproduces aggregate and family metrics;
- peak training memory is below 28 GiB.

Passing authorizes only the sealed 40-case regression canary:

- GSM8K at least 14/16;
- MMLU at least 13/16;
- GPQA-Diamond at least 3/8;
- total at least 30/40;
- parse failures no worse than base 4B;
- zero API errors and complete serving parity.

The canary is excluded from training and cannot establish uplift. Full suite,
merge, scale-up, and RL remain forbidden until their later gates pass.

## Reproduction

```bash
PYTHONPATH=. ../.venv/bin/python scripts/validate_sft_smoke.py \
  --config configs/sft/hard_preservation_smoke_v7.json
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. ../.venv/bin/python -m nano_train.cli \
  sft-smoke --config configs/sft/hard_preservation_smoke_v7.json
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. ../.venv/bin/python \
  scripts/validate_sft_adapter.py \
  --config configs/sft/hard_preservation_smoke_v7.json
```

## Result

V7 is stable and improves every validation family, but fails the frozen local
gate:

- aggregate exact / semantic improves from 8/32 to 19/32;
- numeric improves from 0/16 to 6/16, below the required 10/16;
- choice improves from 2/8 to 5/8 and reaches its threshold;
- process improves from 6/8 to 8/8 and reaches its threshold;
- aggregate semantic is below 24/32 and strict exact is below 22/32;
- all 20 losses and all 224 FP32 adapter tensors are finite;
- early/late five-step loss means are 0.197253 / 0.058721;
- independent reload reproduces aggregate and family metrics;
- peak training memory is 20.10 GiB;
- no post-SFT output reaches the 128-token cap.

The 10 remaining numeric failures have wrong final numbers in a non-scoring
diagnostic, so the gap is semantic rather than a format or truncation artifact.

V7 is rejected. The sealed canary, full suite, merge, scale-up, and RL remain
forbidden.

Public result:

- `docs/results/hard_preservation_sft_smoke_v7.md`;
- `docs/results/hard_preservation_sft_smoke_v7.public.json`.
