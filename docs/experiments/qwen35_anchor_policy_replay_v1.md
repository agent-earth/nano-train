# Qwen3.5 Anchor-Policy Replay v1

## 假设

上一 matched ablation 证明 gold final replay 显著提升净质量，但 control/treatment
都丢掉同一个 anchor win。这里在全新 60k/70k 数值区间加入 frozen anchor
policy KL，直接约束 replay step 的决策分布。

## 数据

- train pairs：`256`；
- untouched final-only dev：`256`；
- 每个 family train/dev：`64` /
  `64`；
- train full-sequence tokens：`52,361`；
- train final target tokens：`2,730`；
- maximum sequence：`130`；
- suffix alignment：`512`；
- 与六个已观察 local surfaces overlap：0；
- 与完整 GSM8K/MMLU/GPQA prompts overlap：0。

## Teacher cache

- teacher：冻结的相同 4B anchor adapter；
- 只缓存 train final-view supervised token positions；
- 每个位置 top-64 log probabilities + one residual other bucket；
- temperature 1.0；
- 采用 train target teacher-forcing 前缀，但不读取 dev/benchmark expected answer、
  correctness 或 observed outputs；
- raw cache 只写 ignored artifacts，公开收据只保存 identity/统计。

## Matched arms

- control replay：`0.5 final CE`；
- treatment replay：`0.5 final CE + 1.0 anchor-policy KL`；
- 两臂共享 anchor、seed、data、pair order、CE、LR、512 steps、prompt、parser
  和 generation budget；
- 唯一隔离因素：replay step 上的 anchor-policy KL。

## Gate

- teacher cache finite + identity verified；
- 两臂 finite + independent reload exact；
- treatment vs anchor：显著、至少 12 wins、0 losses、family/parse
  non-regression；
- treatment accuracy >= control；
- treatment anchor-only losses < control；
- baseline rows 跨两臂逐条一致。

通过只允许 treatment 进入已预注册 211-case canary；完整 benchmark 仍关闭。

## 执行边界

```json
{
  "benchmark_accessed": false,
  "canary_accessed": false,
  "dev_observed": false,
  "model_generation_started": false,
  "teacher_cache_started": false,
  "this_commit_only_preregisters": true,
  "training_started": false
}
```
