# Skill Release Long-Sequence SFT Smoke v1

## 做了什么

- 模型：Qwen3.5-4B；
- 数据：`skill-sft-10k-10m-v2` release；
- 每个 family 取 2 条 train + 1 条 dev，共 10 / 5 条；
- 最大序列长度：1088；
- LoRA：q_proj + v_proj，4 optimizer steps；
- gradient checkpointing：开启；
- 独立重新加载 adapter 并复跑同一 dev。

## 稳定性

- Loss：0.573847, 0.102379, 2.406942, 0.106042；
- Train peak：19.86 GiB；
- Reload peak：16.02 GiB；
- Adapter tensors：32；
- Non-finite tensors：0；
- Reload 与进程内结果一致：true。

## 质量

- Baseline exact / semantic：
  1/5 /
  1/5；
- Post-SFT exact / semantic：
  1/5 /
  1/5；
- Semantic delta：+0.0000。

这轮没有质量提升。它只证明长序列训练路径可运行、显存可承受、adapter 可保存并
独立重载。

## 决策

- accepted_local_smoke：true；
- quality_improved：false；
- scale_allowed：false；
- benchmark_allowed：false；
- holdout_allowed：false；
- rl_allowed：false。

下一步：Keep full training closed. Increase only the bounded local training dose and synthetic dev sample count under a new pre-registered config; require a positive dev delta before any benchmark or RL work.

## Evidence

- config SHA256: `22bf219a939c5ddfa3e28068428174dcb1008f1db8eac6e1546d3064b8912896`;
- dataset SHA256: `b5503761900cfa290dba03aff306ff511630d01544a4dab93de3a0be1e74abc1`;
- release manifest SHA256:
  `26ddb15f5c2e043d20527103a5a59216e54290aabeea2a6d228ebce7b7bb35e3`;
- adapter SHA256: `022852127de3808266ed257fe2523c9c31e0b51a665527b0f3e41808fa2b3a96`;
- metrics SHA256: `b52f6ae6702fdbd4c1da5ed5a03d881a6ad7261c50870c854e78ec3eac3daee0`;
- reload SHA256: `29c3b793ec60ff2b7f7773ccb04069da00360ed14d7972b0914e0904f3707fdc`.

## 结论边界

This smoke proves the 4B long-sequence LoRA path is finite, fits one V100, saves and reloads reproducibly, and consumes the released JSONL. It does not establish quality or benchmark uplift.
