# Qwen3.5 Conservative Consistency Route v1

## 假设

Consistency 在 fresh dev 上有 15 wins / 2 losses。两个 losses 都来自
repeated-operand，而 10/15 wins 来自 exact-division。因此预注册一个最简单、
target-blind 的结构 route：

- exact-division → consistency adapter；
- 其他三个 family → anchor adapter。

规则只看程序化 family，不看 model confidence、output、expected answer 或已观察
case 结果。

## Fresh evaluation

- 4 families × 64 = 256 cases；
- case contract SHA：
  `dc3b375748993321db167fb61aa04000b70201d8d285556fc20da749ea2fe3c6`；
- 与所有已观察 synthetic quality prompts overlap=0；
- 与完整 GSM8K/MMLU/GPQA 题面 overlap=0；
- benchmark/canary/holdout rows 和 outputs=0；
- raw cases/output 仅写 ignored `artifacts/`。

## Arms

1. anchor adapter；
2. consistency adapter；
3. routed candidate 在 report 阶段按 frozen family 规则组合。

两条原始 arm 都必须先完整运行；不能根据 anchor 结果改变 route。

## Gate

- routed accuracy > anchor；
- paired bootstrap CI lower > 0；
- exact McNemar p < 0.05；
- 至少 6 wins、0 losses；
- every-family 和 parse non-regression。

通过只允许消费为已预注册 211-case canary 的本地 admission dependency；不直接
开放完整 benchmark 或 independent holdout。

## 执行边界

```json
{
  "benchmark_accessed": false,
  "canary_accessed": false,
  "evaluation_started": false,
  "model_generation_started": false,
  "this_commit_only_preregisters": true
}
```
