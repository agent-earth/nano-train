# Twinkle and Tinker compatibility v1

## Scope

This experiment establishes a client-side compatibility boundary before any
remote training or benchmark claim.

- Twinkle source: `modelscope/twinkle@1040a02d08c390031800093336718589160b52af`
- Tinker Cookbook source:
  `thinking-machines-lab/tinker-cookbook@600b802d09be13e03756bfdc31e8c864fed90e7b`
- Target model: `Qwen/Qwen3.5-4B`
- Recommended renderer: `qwen3_5`
- Recommended LoRA learning rate from the pinned Cookbook:
  `0.0004905250962784578`

## Environment boundary

The two clients must stay in separate Python environments.

| Environment | Purpose | Tinker SDK | Transformers | Local GPU use |
| --- | --- | --- | --- | --- |
| `twinkle/.venv-client` | Twinkle-compatible client and mock server | `0.16.1` | `5.16.1` | No; installed Torch is CUDA 13 |
| `tinker-cookbook/.venv` | Current native Tinker Cookbook | `0.27.0` | `5.5.4` | No; installed Torch is CUDA 13 |
| `ultimate-distill-workspace/.venv` | Existing local V100 training and inference | not installed | `5.12.1` | Yes |

Twinkle pins `tinker==0.16.1`, while the pinned Tinker Cookbook requires
`tinker>=0.23.0`. The Cookbook also caps Transformers at `5.5.4`, while the
existing local Qwen3.5 stack uses `5.12.1`. Installing either client into the
existing V100 environment would therefore replace validated dependencies.

## Stable API subset

`nano_train.tinker_api` validates the common API used by both client versions:

- `ServiceClient.create_lora_training_client[_async]`
- `TrainingClient.forward[_async]`
- `TrainingClient.forward_backward[_async]`
- `TrainingClient.optim_step[_async]`
- `TrainingClient.save_state[_async]`
- `TrainingClient.save_weights_and_get_sampling_client[_async]`
- `SamplingClient.sample[_async]`
- `SamplingClient.compute_logprobs[_async]`

The provider-specific preparation is explicit:

- `provider=tinker` requires `tinker>=0.23.0`.
- `provider=twinkle` requires the isolated `tinker 0.16.x` environment and
  calls `twinkle_client.init_tinker_client()` before importing `tinker`.
- API keys and base URLs are read only from named environment variables.
  They are not stored in experiment JSON.

## Training-loop rules

The compatibility helper follows the pinned Cookbook behavior:

1. Submit `forward_backward_async`.
2. Submit `optim_step_async` immediately after it.
3. Await both API futures together.
4. Create a fresh sampling client after saving weights.
5. Do not add client-side retries or fixed request timeouts around Tinker
   calls; the SDK owns retry and stuck-request handling.

## Validation

```bash
export PLAYGROUND_ROOT=/path/to/playground
export NANO_TRAIN_ROOT="$PLAYGROUND_ROOT/ultimate-distill-workspace/worktrees/nano-train-tinker-traex-04"

PYTHONPATH="$NANO_TRAIN_ROOT" \
  "$PLAYGROUND_ROOT/twinkle/.venv-client/bin/python" -m nano_train.cli \
  tinker-compat \
  --config "$NANO_TRAIN_ROOT/configs/tinker/twinkle_qwen35_4b_client_v1.json"

PYTHONPATH="$NANO_TRAIN_ROOT" \
  "$PLAYGROUND_ROOT/tinker-cookbook/.venv/bin/python" -m nano_train.cli \
  tinker-compat \
  --config "$NANO_TRAIN_ROOT/configs/tinker/native_qwen35_4b_client_v1.json"

cd "$NANO_TRAIN_ROOT"
PYTHONPATH=. "$PLAYGROUND_ROOT/ultimate-distill-workspace/.venv/bin/python" \
  -m pytest tests/test_train.py -q -k tinker
```

The compatibility commands intentionally do not contact a service. A live
training smoke requires either `TWINKLE_SERVER_URL` plus
`TWINKLE_SERVER_TOKEN`, or `TINKER_API_KEY` and access to a native Tinker
service. No such credentials are embedded or inferred.

## Observed results

- Twinkle client contract: passed with `twinkle-kit 0.4.0.dev0` and
  `tinker 0.16.1`.
- Native client contract: passed with `tinker-cookbook 0.5.6` and
  `tinker 0.27.0`.
- Twinkle's official CPU mock integration test passed end to end with
  `TWINKLE_TEST_TINKER=1`. It exercised session creation,
  `forward_backward`, `optim_step`, base-model sampling, sampler-weight
  checkpoint creation, sampling from the checkpoint, `save_state`, and
  `load_state`.
- Three focused `nano-train` compatibility tests pass. The broader
  `tests/test_train.py` run passes 52 of 52 tests when the isolated worktree
  is given read-only access to the historical ignored
  `artifacts/semantic-arithmetic-sft-smoke-v5/metrics.json` fixture. Without
  that local artifact, the unrelated historical budget-audit test fails
  closed; no ignored artifact is committed.
- Both client environments resolved `torch 2.13.0+cu130`, and CUDA
  initialization is unavailable with the current driver. They are therefore
  client/protocol environments only. Local V100 training remains in the
  existing CUDA-12-compatible workspace environment.

## Non-claims

- Client installation and API compatibility do not prove model improvement.
- A mock-server E2E proves protocol behavior, not GPU training correctness.
- SWE-bench superiority requires matched Harbor runs for 4B, 9B, and 27B on
  the same case set, harness, budgets, and scorer.
