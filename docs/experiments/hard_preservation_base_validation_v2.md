# Hard Preservation Base Validation v2

## Purpose

Base validation v1 reports 8/32 exact and semantic valid, but 8 outputs reach
its 80-token cap. V2 removes that contract confounder before any v7 training
decision.

## Frozen Intervention

V2 changes only `generation_max_new_tokens` from 80 to 128. It retains:

- Qwen3.5-4B base model and tokenizer;
- all 32 hard-preservation validation cases;
- `max_length=192`;
- greedy generation and `enable_thinking: false`;
- exact and family-aware semantic verifiers;
- dataset and model identities.

No optimizer, adapter, benchmark, or sealed canary is used.

## Identity

- data revision: `204b053`;
- dataset SHA256:
  `ac07e10f1e04ca8ddec74aefc8df9b00475f7fc9c57345ff98182dd8e4c8bae9`;
- model config SHA256:
  `ddc63e1c717afa86c865bb5e01313d89d72bb53b97ad4a8a03ba8510c0621670`.

## Interpretation

- V2 is the only valid pre-training baseline for a possible v7 smoke.
- V1 remains evidence of an insufficient generation budget.
- A v7 experiment is allowed only if v2 has zero capped outputs and at least
  one family has material semantic failure headroom.
- No audit outcome authorizes training automatically; v7 requires a separate
  pre-registration.

## Reproduction

```bash
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. ../.venv/bin/python \
  scripts/run_base_validation_audit.py \
  --config configs/audits/hard_preservation_base_validation_v2.json
```
