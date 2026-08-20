# Qwen3.5 RL / OPD Admission v1

## 这次要验证什么

这不是 benchmark 实验，也不是模型质量实验。只验证两件事：

1. Qwen3.5-4B 能否在 exact verifier 奖励下完成真实 RL 更新；
2. Qwen3.5-4B 能否对自己的 on-policy rollout，接受冻结
   Qwen3.5-9B token 分布的蒸馏更新。

| Mode | Steps | Student dtype | Teacher dtype | Config SHA256 |
| --- | ---: | --- | --- | --- |
| `rl` | 2 | `float32` | `None` | `17ab5c2c9d4e58d0d178833713ca68a09db3b1b73bfbadedc2402150f3a5b7ca` |
| `opd` | 2 | `float32` | `float16` | `987c491501d28595eb0b1a812da107a4e47413be4959c6a193dad04f4f912913` |

## 数据和污染边界

- 每个实验只有 2 条 train synthetic arithmetic 和 2 条 probe。
- synthetic prompt 与完整 GSM8K 1,319、MMLU 14,042、GPQA 198 题面做
  normalized exact-hash 对比，重叠为 0。
- 污染审计只读取 benchmark 题面列，不读取标签；不读取任何 benchmark、
  canary、holdout 模型输出。
- raw rollout、adapter、metrics 写入 ignored `artifacts/`，不会提交。

## 固定机制

- RL：4B 采样 rollout，exact verifier 给 `+1/-0.25/-1` reward，
  优化 REINFORCE loss，并用 detached base-policy KL 约束。
- OPD：4B 采样 rollout；冻结 9B 在 GPU1 对同一 token sequence 输出 logits；
  4B 在 GPU0 最小化 teacher→student KL。
- 两个实验都从 fresh base 4B 开始，不串联 adapter。
- FP32 student、q/v LoRA r=8 alpha=16、2 steps、LR 1e-5、
  seed 20260820、temperature 0.8、top-p 0.95、12-token rollout 全部冻结。

## 通过条件

- 恰好 2 个 optimizer steps；
- 所有 loss、gradient norm、adapter tensor 有限；
- adapter 使固定 probe logits 发生变化；
- 独立 reload 后 probe logits SHA256 逐字一致；
- failure receipt 不存在；
- 污染审计通过。

通过只说明实现可用，不说明能力提升，不自动开放 benchmark、canary、holdout
或更大训练。

## 禁止事项

- `synthetic_task_change`
- `reward_change`
- `teacher_change`
- `rollout_temperature_change`
- `rollout_top_p_change`
- `rollout_budget_change`
- `optimizer_change`
- `learning_rate_change`
- `step_change`
- `seed_change`
- `lora_scope_change`
- `adapter_weight_change`
- `benchmark_access`
- `canary_access`
- `independent_holdout_access`

## 执行边界

```json
{
  "model_generation_started": false,
  "opd_started": false,
  "rl_started": false,
  "this_commit_only_preregisters": true,
  "training_started": false
}
```
