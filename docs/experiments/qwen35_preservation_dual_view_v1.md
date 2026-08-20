# Qwen3.5 Preservation-Aware Dual-View SFT v1

## 假设

旧 consistency adapter 在 fresh dev 上从 2/192 提升到 15/192，但覆盖掉 2
个 anchor wins。这里不使用旧 dev 调参，而是在全新数值区间做 matched two-arm
ablation，检验遗忘是否来自第二次 process/KL 压力。

## 数据

- train pairs：`256`；
- untouched final-only dev：`256`；
- 每个 family train/dev：`64` /
  `64`；
- train full-sequence tokens：`51,890`；
- train final target tokens：`2,662`；
- maximum sequence：`130`；
- suffix alignment：`512`；
- 与五个已观察 local surfaces overlap：0；
- 与完整 GSM8K/MMLU/GPQA prompts overlap：0。

## Matched arms

- control：每个 pair 连续执行两次完整
  `0.5 process CE + 0.5 final CE + 1.0 detached-teacher KL`；
- treatment：第一次相同，第二次替换为 `0.5 final CE`；
- 两臂共享 anchor、seed、data、pair order、LR、总步数 512、prompt、parser、
  generation budget；
- 唯一隔离因素：second-step objective。

## Gate

- 两臂 finite 且 independent reload exact；
- treatment vs anchor：accuracy 提升、CI lower > 0、McNemar p < 0.05、
  至少 12 wins、0 losses、family/parse non-regression；
- treatment accuracy >= control；
- treatment anchor-only losses < control anchor-only losses；
- baseline rows 在两臂间逐条完全一致。

通过只允许 treatment 进入已预注册 211-case canary；完整 benchmark 仍关闭。

## 执行边界

```json
{
  "benchmark_accessed": false,
  "canary_accessed": false,
  "dev_observed": false,
  "model_generation_started": false,
  "this_commit_only_preregisters": true,
  "training_started": false
}
```
