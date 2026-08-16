# Targeted Preservation SFT Smoke v11

## Hypothesis

V9 and v10 both reach numeric 9/16, and v10 exactly reproduces every v9
failure ID. All seven persistent numeric failures belong to the same
host-count family.

The v5 data has a support gap: all 16 host-count train rows use 3 or 4
companions, while all 8 host-count development rows use 2. V11 tests whether
fresh companion-count 2 supervision closes that structural gap without
regressing choice or process behavior.

## Single Data Intervention

Relative to v10, only the dataset identity changes:

- `hard-preservation-mix-v5` at revision `204b053`;
- to `targeted-preservation-mix-v6` at revision `ba11804`.

V6 replaces exactly 16 host-count train rows in place. All 32 development
rows and the other 144 rows are byte-identical to v5. The 32-step schedule
exposes 13 targeted replacement rows.

Model, FP32, LoRA, seed, batch, optimizer, LR, warmup, max steps, sequence and
generation budgets, family gates, reload checks, and canary rules remain
frozen.

## Evidence Boundary

The v10 development failures informed v6. This 32-row split is therefore a
development gate only and cannot support an independent quality claim.

The v6 builder consumes only public-safe failure IDs and verifies their family.
It copies no development prompt, target, model output, benchmark content, or
canary content into training. The sealed 40-case canary remains unread.

## Frozen Gate

V11 must meet the unchanged local gate:

- 32 finite steps and decreasing early/late mean loss;
- aggregate semantic at least 24/32;
- numeric at least 10/16;
- choice at least 5/8;
- process at least 7/8;
- every family above base;
- strict exact at least 22/32;
- zero capped outputs;
- finite tensors, exact independent reload, memory below 28 GiB.

Passing authorizes only the sealed 40-case regression canary. The canary
cannot establish quality uplift. Full suite, merge, scale-up, and RL remain
forbidden until their later gates pass.

## Reproduction

```bash
PYTHONPATH=. ../.venv/bin/python scripts/validate_sft_smoke.py \
  --config configs/sft/targeted_preservation_smoke_v11.json
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. ../.venv/bin/python -m nano_train.cli \
  sft-smoke --config configs/sft/targeted_preservation_smoke_v11.json
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. ../.venv/bin/python \
  scripts/validate_sft_adapter.py \
  --config configs/sft/targeted_preservation_smoke_v11.json
```

## Frozen Identity

- data revision: `ba11804`;
- dataset SHA256:
  `ab51a1be5f45d7f71796fbf98ef6cce83ff9cb0f0a756fed01cb1e7aea55651d`;
- config SHA256:
  `9a971cb46a1f5c21164d6117bef40aedfcb7170e9e82604bb7400c942a2be593`;
- model config SHA256:
  `ddc63e1c717afa86c865bb5e01313d89d72bb53b97ad4a8a03ba8510c0621670`.
