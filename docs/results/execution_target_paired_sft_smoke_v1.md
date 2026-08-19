# Execution-Target Paired SFT Smoke v1

## 这次具体做了什么

- 数据：`skill-sft-execution-target-paired-v1`，512 train / 80 dev；
- 训练：Qwen3.5-4B，q/v-only LoRA，FP32，40 steps；
- 40 条 train exposure 包含 10 个完整 process/final pair，以及四个 JSON
  family 各 5 条；
- max sequence 704，generation budget 160；
- 配置、40 个 train ID、80 个 dev ID 和 decision rule 已在 commit
  `09f5747` 中先提交；
- 训练后独立 reload 全部 80 条 dev。

## 合同是否执行

- schedule matches：true；
- scheduled complete pairs：10；
- scheduled views：{'final': 10, 'json_preservation': 20, 'process': 10}；
- scheduled JSON：{'coding-and-validation': 5, 'planning-and-state': 5, 'skill-routing-and-reflection': 5, 'tool-use-and-recovery': 5}。

## 稳定性

- 可训练参数：917,504；
- 训练显存峰值：18.28 GiB；
- reload 显存峰值：15.93 GiB；
- 训练耗时：617.86 秒；
- adapter tensors：32；
- non-finite tensors：0；
- 独立 reload 一致：true。

## Corrected Dev

- aggregate verified：51/80 →
  54/80，delta +3；
- JSON verified：27/32 →
  30/32，delta
  +3；
- process verified：24/24 →
  24/24；
- final-only verified：0/24 →
  0/24；
- both-verified pairs：0/24 →
  0/24；
- changed outputs：28/80。

aggregate 的 +3 全部来自 `skill-routing-and-reflection` 4/8 → 7/8。
execution 的 final-only 和 paired gate 都没有提升。

## 为什么没迁移

Base 4B 的 process view 已经是 24/24，但对应 final-only view 是 0/24。
训练后仍然是 24 个 process-only pair，0 个 both-verified pair。

发生变化但仍错误的 final-only 样例：

- `(145 + 43) * 3 - 43`：expected `FINAL: 521`，base `FINAL: 344`，SFT `FINAL: 374`。
- `(225 + 43) * 2 - 43`：expected `FINAL: 493`，base `FINAL: 514`，SFT `FINAL: 264`。
- `(225 + 43) * 3 - 43`：expected `FINAL: 761`，base `FINAL: 804`，SFT `FINAL: 808`。
- `(385 + 59) * 3 - 59`：expected `FINAL: 1273`，base `FINAL: 1374`，SFT `FINAL: 1376`。

标准 SFT 同时看过两种 view，但没有把 process 中已经正确的最终值迁移到
final-only 输出。

## 决策

- paired_execution_method_accepted：
  false；
- aggregate_positive：true；
- final_view_positive：false；
- pair_both_verified_positive：
  false；
- larger training / benchmark / independent holdout / RL：全部关闭。

下一步：拒绝把标准 SFT 直接用于 paired views 的 execution transfer。不要根据这组 dev 继续搜索 steps、LR、seed、schedule、LoRA scope、prompt、parser 或 adapter weight。下一轮训练前先设计显式 consistency/distillation objective，把 process view 的正确最终值约束到 final-only logits。

## Evidence

- config SHA256: `7dddcad143d37338ab2bd572e0fe540f1e55cbaf14893fafde2da460f10851d7`;
- dataset canonical SHA256:
  `77728a0531f18e55989c172a21fb267284aa1001c17fd62de6bcd13b9d300659`;
- adapter SHA256: `a12b90ce3961edd3b8f256ea7642c174880730a1352c247342d0dedd2047ffe8`;
- metrics SHA256: `2a06d255724ef9ed12df3177eaaee72f019abf960d51d62fa1b58a5e008e9803`;
- generations SHA256: `f077e9ad9d9c42a477399753395f05bca09695fb3a53b17278a7003d300b3656`;
- reload SHA256: `726267cf1a5eb9fa9be2e30c2d5d88e3315e0df682955fd5ba73d34abd774c84`;
- rescore SHA256: `57aba19fe463ef59c749db82cbd4c1f8cc5ecdf6f06c8c524e0b3960fea50c58`.

## 结论边界

adapter 训练稳定，aggregate verified 增加 3 分，但全部来自 JSON routing 改善。final-only execution 仍是 0/24，both-verified pairs 仍是 0/24。因此本方法被拒绝，不允许扩大训练、访问 benchmark 或 independent holdout，也不允许启动 RL。
