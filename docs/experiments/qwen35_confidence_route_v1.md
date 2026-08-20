# Qwen3.5 Normalized-Confidence Route v1

## 目的

Family route 在 fresh cases 上不稳定。这里改用 target-blind 的双模型相对
likelihood selector；规则在生成前冻结，不使用 expected answer 或 correctness。

## Fresh surface

- 4 families × 64 = 256 cases；
- case contract SHA：
  `3fce148ab84d3683d0527dc437242ef3d7714138106bfbd83a22c4641a649aad`；
- 与所有已观察 synthetic quality prompts overlap=0；
- 与完整 GSM8K/MMLU/GPQA 题面 overlap=0；
- raw candidates/scores 只写 ignored `artifacts/`。

## 五阶段

1. anchor 生成完整 256 candidates；
2. consistency 生成完整 256 candidates；
3. anchor 对两份 candidates 计算 mean token logprob；
4. consistency 对两份 candidates 计算 mean token logprob；
5. 按 frozen relative-logprob rule 组合。

## Selector

- anchor relative：`logp_consistency_model(anchor_candidate) - logp_anchor_model(anchor_candidate)`；
- consistency relative：`logp_consistency_model(consistency_candidate) - logp_anchor_model(consistency_candidate)`；
- consistency 仅在 `consistency_relative > anchor_relative` 时生效；
- 平局回退 anchor；
- score 包含 EOS，按 candidate token 数取 mean；
- 无 threshold、confidence search、expected answer、correctness 或 model-output
  feedback。

## Gate

- routed accuracy > anchor；
- CI lower > 0；
- McNemar p < 0.05；
- 至少 6 wins / 0 losses；
- family/parse non-regression；
- 所有 logprob finite。

通过只允许进入已预注册 211-case canary，不直接开放完整 benchmark。

## 执行边界

```json
{
  "benchmark_accessed": false,
  "canary_accessed": false,
  "evaluation_started": false,
  "generation_started": false,
  "scoring_started": false,
  "this_commit_only_preregisters": true
}
```
