# Paired Consistency v1

## 做了什么

- Qwen3.5-4B，q/v-only LoRA，FP32，40 steps；
- 20 个 pair step 与 20 个 JSON step 交替；
- pair loss：
  `0.5*process_ce + 0.5*final_ce + 1.0*KL(detach(process)||final)`；
- fresh heldout：24 个完整 pair + 四个 JSON family 各 8 条，共 80 条；
- config、loss、train schedule、heldout 和 decision rule 已在 commit
  `00aa03b` 中先冻结；
- training 前明确记录 `training_started=false`。

## 稳定性

- 可训练参数：917,504；
- 训练显存峰值：19.55 GiB；
- reload 显存峰值：15.93 GiB；
- 训练耗时：675.67 秒；
- adapter tensors：32；
- non-finite tensors：0；
- 独立 reload metrics 和 80 条 generations 逐字一致：
  true。

## Corrected Fresh Heldout

- aggregate verified：51/80 →
  55/80，delta +4；
- process：24/24 →
  24/24；
- final-only：0/24 →
  1/24；
- both-verified pairs：0/24 →
  1/24；
- JSON：27/32 →
  30/32；
- 四个 JSON family 均 non-regression：
  true。

## 真实修复

- 表达式：`(130 + 63) * 2 - 63`；
- process view：最后得到 `FINAL: 323`；
- base final-only：`FINAL: 223`；
- consistency final-only：`FINAL: 323`。

这是第一个在 fresh pair 上观察到的 process→final 正向迁移。

## 不确定性

- aggregate：5 wins / 1 loss，delta
  +0.0500，95% CI
  [+0.0000,
  +0.1125]，
  McNemar p=0.21875；
- final-only：1 win / 0 loss，delta
  +0.0417，95% CI
  [+0.0000,
  +0.1250]，
  McNemar p=1.0；
- both-verified pair：1 win / 0 loss，指标同 final-only。

方向是正的，但置信区间下界为 0，统计检验不显著。

## 决策

- accepted_local_method_smoke：
  true；
- statistically_supported：
  false；
- larger training / benchmark / independent holdout / RL：全部关闭。

下一步：保留 consistency v1，作为第一个方向正确的 execution-transfer 方法。不要在这组 heldout 上调参。下一步使用完全相同的 objective 预注册更大的 fresh local replication，样本量必须足够检验显著性；在此之前不允许扩大训练或访问 benchmark。

## Evidence

- config SHA256: `b7a87ab0360383e94124c09c4b91fcb501b1219ff304b0b4531980519ea63b18`;
- heldout SHA256:
  `ba697ea371a7e93b9d6a6928226e339ab77043c65e3ec21be580c8b83397ec62`;
- adapter SHA256: `8abf055caa7a3a70cb99d46cbc8c0275851d43a89f6512d3dd4e4a4f80d508c2`;
- metrics SHA256: `118532857caad10de6fb570e83989a211f636651275df7af656fada05fc93921`;
- generations SHA256: `ee05d12c0d01ad9c6ab070aeaaff1305f213277b0347c92e4c389d48c2d6909c`;
- reload SHA256: `ef34b20f0024995a354e26c1c4b444e3994d1733a2de4f417eef53f4a0c79eb8`;
- rescore SHA256: `aa43a2d9da42661faaaca9f0451c0627a9bc3e73ee5c56396b683a014734acb5`.

## 结论边界

Consistency v1 通过了预注册的本地方向 gate，并修复了 1 个 fresh final-only execution pair；但 aggregate 和 final 的置信区间都包含 0，McNemar 检验也不显著。这是机制证据，不是稳定提升证据，不能据此扩大训练、访问 benchmark/holdout 或启动 RL。
