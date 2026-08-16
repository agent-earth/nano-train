# Format Contract SFT Smoke v1

## Goal

Test whether a tiny LoRA SFT run can improve exact `FINAL:` compliance on
leak-free synthetic analog data. This is a training-stack and target-behavior
smoke, not a benchmark-quality claim.

## Inputs

- Base model: local Qwen3.5-4B;
- model config SHA256:
  `ddc63e1c717afa86c865bb5e01313d89d72bb53b97ad4a8a03ba8510c0621670`;
- dataset: `format-contract-analog-v1`;
- dataset SHA256:
  `46f2128f219db7011d5db95b5ca3a97029b57f5ac959e194860b4c0f4ba3ad53`;
- 102 deterministic synthetic train samples;
- 26 deterministic synthetic validation samples;
- no benchmark content, model output, teacher output, or sealed case ID.

## Frozen Configuration

- seed: `20260816`;
- precision: FP16 on one V100 32GB;
- max sequence length: 128;
- LoRA rank/alpha/dropout: 8/16/0;
- LoRA targets: `q_proj`, `v_proj`, `gate_proj`, `up_proj`, `down_proj`;
- batch size: 1;
- gradient accumulation: 4;
- effective batch size: 4;
- optimizer: AdamW;
- optimizer steps: 20;
- learning rate: 2e-4 with two warmup steps and linear decay;
- loss: assistant target plus EOS only;
- validation generation: greedy, max eight new tokens.

Dependencies are frozen in `requirements-smoke.lock`.

## Compatibility Probe

A real GPU3 forward/backward probe loads `Qwen3_5ForCausalLM`, applies the
same LoRA module family, masks the prompt, and backpropagates one analog
sample. It observes:

- 4,980,736 trainable parameters for rank 4 probe LoRA;
- 224 trainable tensors with gradients;
- finite loss 0.2421;
- peak allocated memory 9.97 GiB.

This probe validates the execution path only. It is not a training result.

## Pre-Registered Smoke Decision

The SFT smoke passes its local format objective only if:

- all 20 optimizer steps complete with finite loss;
- final loss is lower than initial loss;
- the saved adapter has a stable tree SHA256 and reloads successfully;
- baseline validation accuracy is recorded before training;
- post-SFT validation reaches 26/26 exact targets;
- post-SFT validation is strictly better than baseline unless baseline is
  already 26/26;
- no training or generation errors occur;
- peak allocated GPU memory remains below 28 GiB.

Even if these conditions pass, SFT is not accepted for the long-term goal
until the adapter is evaluated through nano-harness on unchanged matched
GSM8K/MMLU/GPQA cases and shows no task regression. Do not start RL or scale
data before that evaluation.

## Artifacts

Local ignored artifacts:

- `artifacts/format-contract-sft-smoke-v1/adapter/`;
- `artifacts/format-contract-sft-smoke-v1/generations.json`;
- `artifacts/format-contract-sft-smoke-v1/metrics.json`.

Public evidence after the run may include only config/data/model identities,
aggregate loss, aggregate format accuracy, dependency/hardware versions,
failure sample IDs, artifact hashes, and decision fields. Prompts, targets,
and generations remain local.

## Reproduction

```bash
PYTHONPATH=. ../.venv/bin/python scripts/validate_sft_smoke.py
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. ../.venv/bin/python -m nano_train.cli \
  sft-smoke --config configs/sft/format_contract_smoke_v1.json
```

## Result

The v1 smoke fails:

- baseline exact validation: 23/26;
- step 1 loss: 0.148261;
- first non-finite loss: step 2;
- finite steps: 1/20;
- post-SFT exact validation: 0/26;
- peak allocated memory: 10.11 GiB.

The v1 runner incorrectly continued after non-finite loss and saved an invalid
adapter. That adapter is quarantined in ignored local artifacts and must not
be evaluated, merged, published, or used for RL.

All 224 saved LoRA tensors are FP32 but non-finite, so the exact instability
source remains unresolved. The runner now requires finite loss, gradients, and
post-update parameters and writes a failure receipt instead of saving an
adapter. A separate v2 requires a diagnostic and new pre-registration.

- [`docs/results/format_contract_sft_smoke_v1.md`](../results/format_contract_sft_smoke_v1.md)
- [`docs/results/format_contract_sft_smoke_v1.public.json`](../results/format_contract_sft_smoke_v1.public.json)
