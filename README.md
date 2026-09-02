# nano-train

`nano-train` is the reproducible training and ablation layer for Ultimate
Distill.

## Progression

1. Validate data and configuration without allocating a GPU.
2. Run a tiny SFT overfit/smoke experiment.
3. Run a small reward/verifier and RL sanity check.
4. Scale only after loss, generation, checkpoint, and benchmark evidence agree.

## Required Evidence

- immutable config and input dataset identity;
- environment and dependency lock;
- seed, hardware, precision, sequence length, and effective batch size;
- train/eval curves and checkpoint lineage;
- sample generations and failure analysis;
- matched downstream benchmark comparison;
- ablations for SFT, RL, SFT+RL, data filters, verifier/reward, and harness
  strategies.

Training success is measured by held-out target behavior, not training loss
alone. Framework selection remains open until baseline and harness evidence
clarify the smallest sufficient implementation.

## Tinker-compatible clients

`nano-train` provides a small compatibility layer for both the native Tinker
service and a Twinkle service exposing Tinker's API:

```bash
nano-train tinker-compat \
  --config configs/tinker/native_qwen35_4b_client_v1.json

nano-train tinker-compat \
  --config configs/tinker/twinkle_qwen35_4b_client_v1.json
```

Run those commands from the matching isolated environment:

- native Tinker Cookbook: `tinker-cookbook/.venv`;
- Twinkle client: `twinkle/.venv-client`.

Do not install both clients into the existing local V100 environment. The
pinned Twinkle source uses `tinker 0.16.x`, while the pinned Cookbook uses
`tinker 0.23+`; their current Torch and Transformers resolutions also differ
from the validated local Qwen3.5 stack. The configuration files contain only
environment-variable names. Set `TINKER_API_KEY` for the native provider, or
`TWINKLE_SERVER_URL` and `TWINKLE_SERVER_TOKEN` for Twinkle, at runtime.

The shared training helper submits `forward_backward_async` and
`optim_step_async` before awaiting either result, matching the Cookbook's
pipelined request guidance. After saving weights, always request a new sampling
client so evaluation cannot silently use stale weights.

See `docs/experiments/twinkle_tinker_compatibility_v1.md` for pinned revisions,
compatibility findings, validation commands, and non-claims.
