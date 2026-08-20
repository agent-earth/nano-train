# Qwen3.5 Quality Consistency v1

## 变化

上一轮 answer-only SFT 只有 2 个 fresh-dev wins。这里不增加 dose，而是更换
监督机制：每个 train task 同时提供 verified process view 和 final-only view，
并用 detached-teacher KL 把 process final logits 约束到 final-only logits。

## 数据

- train pairs：`256`，每个 family
  `64`；
- untouched final-only dev：`192`，每个 family
  `48`；
- train full-sequence tokens：`47,321`；
- maximum sequence：`124`；
- process/final suffix alignment：
  `448` pairs 全部通过；
- 与已观察 quality surfaces prompt overlap：0；
- 与 GSM8K/MMLU/GPQA 题面 overlap：0；
- benchmark/canary/holdout rows、labels、outputs：0。

## Objective

`0.5 * process CE + 0.5 * final CE + 1.0 * KL(detach(process final logits) || final logits)`

- temperature 1.0；
- 256 pair optimizer steps；
- LR 5e-5；
- anchor 固定为 scaled-SFT adapter
  `7287cfcc9372894aae2c4081f6ccf7b18a9292a9e52160579c9ab199906eda45`。

## Gate

- finite + independent reload；
- post accuracy > baseline；
- paired bootstrap CI lower > 0；
- exact McNemar p < 0.05；
- 至少 12 wins、0 losses；
- every-family 和 parse non-regression。

即使通过，benchmark/canary/holdout 仍需单独预注册。

## 执行边界

```json
{
  "benchmark_accessed": false,
  "dev_observed": false,
  "model_generation_started": false,
  "this_commit_only_preregisters": true,
  "training_started": false
}
```
