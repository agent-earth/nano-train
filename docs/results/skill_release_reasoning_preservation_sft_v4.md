# Skill Release Reasoning-Preservation SFT v4

## 这次具体做了什么

这是一个预注册的单方法实验：

- 保持 Qwen3.5-4B、release、80 条 train、q/v-only LoRA、20 steps、LR、
  seed、FP32、max length 和 verifier 不变；
- 把 20 steps 的训练暴露固定为 reasoning 10 次，四个 JSON family 合计
  10 次保护性 replay；
- fresh dev 固定取每个 family 在 release 顺序中的第 5–8 条，共 20 条；
- fresh dev 与训练集、旧 20 题在 sample ID 和 semantic hash 上的 overlap
  都是 0；
- 配置和 decision rule 已在 commit
  `4628044` 中先提交，再运行模型。

## 合同是否真的执行

- planned exposure：coding-and-validation 3 次、planning-and-state 3 次、skill-routing-and-reflection 2 次、tool-use-and-recovery 2 次、verified-reasoning 10 次；
- actual exposure：coding-and-validation 3 次、planning-and-state 3 次、skill-routing-and-reflection 2 次、tool-use-and-recovery 2 次、verified-reasoning 10 次；
- schedule matches：true；
- fresh dev ID SHA256：
  `f2e71078acbca081229c8f0d3bf849fae1befa61b9181e7dcbae1684983b694a`。

## 训练稳定性

- 可训练参数：917,504；
- 训练耗时：191.47 秒；
- 训练显存峰值：19.85 GiB；
- reload 显存峰值：16.02 GiB；
- adapter tensors：32；
- non-finite tensors：0；
- 独立 reload 与训练后结果一致：
  true。

训练和保存流程正常，负结果不是运行故障。

## Corrected Verified Dev

- aggregate：16/20 →
  16/20，delta
  +0；
- changed outputs：10/20；
- 四个 JSON family：全部 4/4 → 4/4；
- verified-reasoning：
  0/4 →
  0/4。

## 为什么没有提升

- `(96 + 68) * 3 - 68`：expected `FINAL: 424`，base `FINAL: 324`，SFT `FINAL: 324`。
- `(89 + 57) * 3 - 57`：expected `FINAL: 381`，base `FINAL: 312`，SFT `FINAL: 312`。
- `(82 + 46) * 3 - 46`：expected `FINAL: 338`，base `FINAL: 312`，SFT `FINAL: 312`。
- `(103 + 79) * 3 - 79`：expected `FINAL: 467`，base `FINAL: 342`，SFT `FINAL: 344`。

4 个 fresh reasoning case 中只有
1 个输出发生变化，而且仍然错误。
10 次 answer-only reasoning 暴露没有教会多步算术过程；它只产生了局部 token
偏移。

## 决策

- reasoning_preservation_method_accepted：
  false；
- larger_training_allowed：
  false；
- benchmark / independent holdout / RL：全部关闭。

下一步：拒绝只增加 answer-only reasoning 样本频次的方案。不要根据已经看到的结果继续调整 schedule、offset、dose、LR、seed、prompt、parser、adapter weight 或 route。下一轮必须改变监督结构，例如使用 verifier 检查的过程轨迹，并换一组新的冻结本地评估数据。

## Evidence

- config SHA256: `a85fafb0810d995536d0620e5c11f58160ccb995474f47f15601f5efb0b42d69`;
- preregister SHA256: `e50d29dcf1c2c541476f42f2255e88eb23b94aa1d281e61f4e68ab765958b139`;
- dataset SHA256: `b5503761900cfa290dba03aff306ff511630d01544a4dab93de3a0be1e74abc1`;
- adapter SHA256: `fa572f8f03a49345697b7164091c1f03d02437b39b99a948904c8e3602bc53d0`;
- metrics SHA256: `3d7121d2b008bae198446aab37b60c835415af731bc345a7decbf0b8585db8e3`;
- generations SHA256: `a9015649c4035c4a0237a1a57837f204a41e537842245b6ee1d0080fc384f347`;
- reload SHA256: `c9c8fb56283304b9bedde528e8299ad5bd39f5b96a7018b4225e62ca8e6e2d24`;
- rescore SHA256: `15077462f77d59a2c5e3a8fef973005bc8b9347a51f5974425c7d91c33d8c636`.

## 结论边界

这次预注册实验只证明：固定 schedule 的 q/v-only SFT 可以稳定训练，并在一组 fresh synthetic dev 上保住四个 JSON family。corrected verified 指标没有提升，reasoning 也没有提升。因此不能扩大训练，不能访问 benchmark 或 independent holdout，也不能启动 RL。
