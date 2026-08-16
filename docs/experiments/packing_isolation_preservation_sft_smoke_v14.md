# Packing Isolation Preservation SFT Smoke v14

V14 isolates packing-efficiency supervision after percentage isolation v13
fixes zero and regresses three development cases.

Relative to v11, only dataset/output identity changes. Data revision `25451af`
changes 8 train rows and exposes 5 packing rows under the frozen 32-step
schedule. All other 184 rows, training parameters, local gates, reload checks,
and staged evaluation boundaries remain unchanged.

The independent holdout stays unread. Passing local authorizes only the old
regression canary; old 211-case non-regression must pass before holdout access.

```bash
PYTHONPATH=. ../.venv/bin/python scripts/validate_sft_smoke.py \
  --config configs/sft/packing_isolation_preservation_smoke_v14.json
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. ../.venv/bin/python -m nano_train.cli \
  sft-smoke --config configs/sft/packing_isolation_preservation_smoke_v14.json
```

Frozen identities:

- config SHA256:
  `7206a76fa6d8307e4c1a42ce753bce358990e65bd4a77bf8881f86c5b55bd773`;
- dataset SHA256:
  `9f79b1cf5af9fa4b36c7507318b32991692f253d2210b5b6ed70a44bee940f2d`;
- model config SHA256:
  `ddc63e1c717afa86c865bb5e01313d89d72bb53b97ad4a8a03ba8510c0621670`.

## Result

V14 is stable but rejected locally:

- aggregate exact / semantic: 21/32 / 25/32;
- numeric exact / semantic: 8/16 / 12/16;
- choice: 5/8; process: 8/8;
- relative to v11: one numeric fix, one numeric regression, one choice
  regression;
- 32 finite losses, 224/224 finite FP32 tensors, exact independent reload,
  early/late loss 0.199554/0.020787, peak 20.18 GiB, zero capped outputs.

Five packing exposures are not a Pareto improvement. Reject v14, stop packing
dose search, and preserve v11. Canary, old full suite, and independent holdout
remain unrun; holdout prompts and references remain unread.
