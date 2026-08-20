# Qwen3.5 Router Negative-Diversity SFT v2

## Data

- Train：6,144，tokens：
  766,519；
- Dev：1,536，A/B/C 各512；
- C dev：8 subtypes 各64；
- 40 steps × effective batch 4 = 160 exposures；
- exposure A/B/C：
  `{"router_a": 52, "router_b": 48, "router_c": 60}`；
- C exposure：
  `{"box_total": 10, "paired_average": 3, "percentage_change": 7, "quotient_remainder": 9, "remaining_stock": 10, "single_operation": 7, "time_conversion": 9, "weighted_total": 5}`。

## Frozen Recipe

- FP32 expanded LoRA：q/v/gate/up/down，r=8，alpha=16；
- LR 2e-4，40 steps，seed 20260827；
- max length 256；generation 8 tokens；
- 保持 framework smoke 上限，不扩大到长训练。

## Acceptance

- finite loss/gradients/adapter；
- exact exposure IDs；
- independent reload metrics + 1,536 generations exact；
- A/B ≥480/512，C ≥496/512；
- 每个 C subtype ≥60/64；
- label 和 subtype 全部 non-regression；
- vLLM namespace remap + serving parity 必须另行通过。

通过也只允许另行预注册 fresh integration；benchmark/canary/holdout/RL
继续关闭。

## Boundary

- config SHA：`d8081d1d565436987e5b94ee6cd6d00a2e113eed2af9dc0d2b84429fef3f3f52`；
- dataset SHA：`8c5975e3ceed494e20d0de54eb5654ab1af71163ed58489d42d98c8b54d0bad9`；
- release SHA：`5edd89701ff33db6eaef74475946abf79176c5c5a7c854a7eea4dd907e69c3f1`；
- exposure SHA：`b6451354be276e92446a47b0b6b268af227db0b45c1300a4825800ef0264ec81`；
- training/model generation started：false；
- adapter/metrics exists：false。
