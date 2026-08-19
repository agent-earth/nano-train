# Skill Release Expanded-LoRA SFT v3

## 这次具体改了什么

这是一个单变量实验。对照组只训练 `q_proj` 和 `v_proj`；本次把 LoRA
target 扩展为 `q_proj`、`v_proj`、`gate_proj`、`up_proj`、`down_proj`。
模型、数据、80 train / 20 frozen dev、20 steps、LR、seed、FP32、max
length 和评估方式全部不变。

## 训练是否正常

- 训练参数：9,961,472；
- 训练耗时：212.88 秒；
- 显存峰值：19.95 GiB；
- Adapter tensors：224；
- Non-finite tensors：0；
- 独立进程 reload 与训练后输出一致：
  true。

所以训练和保存流程是稳定的，下面的负结果不是 adapter 损坏或 reload
失败造成的。

## 正确指标

- Family verifier：17/20 →
  16/20，delta
  -1；
- 字符串 exact：5/20 →
  12/20，delta
  +7；
- 改变输出：16/
  20；
- Family non-regression：
  false。

字符串 exact 上升，是因为 JSON 类输出更贴近固定模板；它没有变成真实
正确性提升。corrected family verifier 反而下降 1 题。

## 与 q/v-only 对照

- q/v-only：917,504 个可训练参数，
  changed 9/20，verified
  17/20；
- expanded-LoRA：9,961,472 个可训练
  参数，changed 16/20，verified
  16/20；
- 相对 q/v-only 的 verified delta：
  -1。

扩展 LoRA 让更多输出发生变化，但没有带来更高正确率。

## 失败样例

`verified-reasoning` 从 1/4 降到
0/4。代表样例：

- 表达式：`(68 + 24) * 3 - 24`；
- 正确答案：`FINAL: 252`；
- base 4B：`FINAL: 252`；
- q/v-only：`FINAL: 252`；
- expanded-LoRA：`FINAL: 276`。

`276` 是 `(68 + 24) * 3` 的中间值。expanded-LoRA 忽略了最后的 `-24`，
把原本唯一正确的 reasoning case 改错。

## 决策

- expanded_lora_method_accepted：
  false；
- larger_training_allowed：
  false；
- benchmark / holdout / RL：全部关闭。

下一步：拒绝在这版数据上使用 expanded-LoRA。不要根据已经看过的 dev 结果继续调 dose、LR、seed、adapter weight、parser 或 prompt。保留 q/v-only 对照，下一轮在新的本地评估面上预注册 reasoning-preservation objective 或 verified-execution method。

## Evidence

- config SHA256: `e4714e8c04ceba2a2b1fadf2db70fe04b911e3620750e23183546038b1c9ef62`;
- dataset SHA256: `b5503761900cfa290dba03aff306ff511630d01544a4dab93de3a0be1e74abc1`;
- adapter SHA256: `2e0123ba7d8d40708cd5ad0eba05f29a194ff04295d9e7587e3e6f1905884c0d`;
- metrics SHA256: `d3c4c627fa8e1ec41381680577aef820896b83888412e0e39e1d028d17720c77`;
- generations SHA256: `daeb51e1d1ac7e91aeff70bdccf6b6160b1ca04114f3a4e31e27446c5a213dc2`;
- reload SHA256: `0e1b177b9242f582faf4377d08667613a31b8f504619e30a5f838c30f1a06b40`;
- rescore SHA256: `802490717f9bc328bb47c4c3b77dc15ab759146b36d4afe324494f896ce0153d`.

## 结论边界

Expanded-LoRA 的训练数值正常、adapter 可以独立重载，也确实改变了更多输出；但 corrected verified dev 从 17/20 降到 16/20，verified-reasoning 从 1/4 降到 0/4。raw exact 从 5/20 升到 12/20 只说明格式和模板更接近目标，不代表质量提升。本结果拒绝该方法，不允许据此扩大训练、访问 benchmark/holdout 或启动 RL。
