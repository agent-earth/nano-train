# Qwen3.5 Synthetic Quality Ablation v1

## 目的

用从未生成过输出的 96 个 synthetic arithmetic cases，比较：

1. base Qwen3.5-4B；
2. 两步 RL adapter；
3. 两步 OPD adapter；
4. base Qwen3.5-9B。

这一步回答“adapter 是否真的改变正确率”，不是 benchmark 评测。

## 冻结 case

- 4 个 family，每类 24 个，共 96 个；
- case contract SHA：`c9de0fcd02da9354aa38ac1f5c864f6bfc4bf6590bca3313d54232c1bac41e4b`；
- case IDs SHA：`b2ed535d23fcc04f743ee63e0500f482c15793e42e240d2a195ce04335185ccc`；
- public receipt 只保存 case ID、family、prompt SHA 和 expected SHA；
- raw expression、expected 和模型 output 只写 ignored `artifacts/`。

## 污染边界

- 与完整 GSM8K 1,319、MMLU 14,042、GPQA 198 题面做 normalized exact-hash；
- overlap 为 0；
- 不读取 benchmark label、output、canary 或 independent holdout；
- 所有 96 cases `training_eligible=false`。

## 生成合同

- arm order：`base4 → rl4 → opd4 → base9`；
- greedy decoding；
- thinking disabled；
- batch size 8；
- max new tokens 32；
- strict parser：整段输出必须恰好匹配 `FINAL: <integer>`。

## Candidate gate

RL/OPD 相对 base4 必须同时满足：

- overall delta > 0；
- paired bootstrap 95% CI 下界 > 0；
- exact McNemar `p < 0.05`；
- 每个 family correct 不低于 base4；
- parse failures 不多于 base4。

每个 arm 都必须运行，不能根据前一个结果跳过。观察后禁止更改 case、prompt、
parser、budget、batch、adapter 或 decoding。

## 执行边界

```json
{
  "benchmark_accessed": false,
  "evaluation_started": false,
  "model_generation_started": false,
  "this_commit_only_preregisters": true
}
```
