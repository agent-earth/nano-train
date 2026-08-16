# Schedule Isolation Preservation SFT Smoke v15

V15 isolates recurring-schedule supervision after percentage and packing
isolations both fail to improve v11. Relative to v11, only dataset/output
identity changes.

Data revision `890b576` changes 8 train rows and exposes 7 schedule examples.
All other rows, FP32/LoRA/optimizer settings, local gates, reload checks, and
staged evaluation boundaries remain frozen. The independent holdout is unread.

Passing local authorizes only the old canary; old 211-case base non-regression
must pass before holdout access.

```bash
PYTHONPATH=. ../.venv/bin/python scripts/validate_sft_smoke.py \
  --config configs/sft/schedule_isolation_preservation_smoke_v15.json
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. ../.venv/bin/python -m nano_train.cli \
  sft-smoke --config configs/sft/schedule_isolation_preservation_smoke_v15.json
```

- config SHA256:
  `413cff6c370c69a9ef6ac9d4ebef32bf3f695ecd14f02c9f105da2893f63230d`;
- dataset SHA256:
  `2bb712de519149d776b1c346466ee49d20017f1065aa3d1b44ae59eb6f5b973a`.
