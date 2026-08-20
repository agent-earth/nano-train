# Qwen3.5 Router Classification SFT Smoke v1

## Data

- Train：768；
- Dev：192，A/B/C 各64；
- 40 steps × effective batch 4 = 160 exposures；
- exposure by label：
  `{"router_a": 46, "router_b": 61, "router_c": 53}`；
- max sequence：130；
- target max：3。

## Frozen Recipe

- parent recipe：format-contract-sft-smoke-v3；
- FP32；
- expanded LoRA：q/v/gate/up/down，r=8，alpha=16；
- LR 2e-4，40 steps，seed 20260824；
- max length 256；generation 8 tokens。

## Acceptance

- finite loss/gradients/adapter；
- independent reload exact metrics；
- aggregate exact improves；
- A/B each ≥48/64；
- C ≥60/64；
- every label non-regression。

通过也只允许另行预注册 fresh router integration；benchmark/canary/holdout/RL
继续关闭。

## Boundary

- config SHA：`8729bea1798e4e0d7c1299ff8f7f101db14649e11d14d0971a31af4ffa055689`；
- dataset SHA：`dacd3663639fe9ddc054865b87afdd0c918f0fddb12c8c9355819d4bbce95d65`；
- release SHA：`fb265e125e181056856a196322cf5da3b1d7d890d60ad653839d2707ebe3781d`；
- training/model generation started：false；
- adapter/metrics exists：false。
