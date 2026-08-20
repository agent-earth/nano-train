# Qwen3.5 Synthetic Quality Ablation v1 Result

## 结论

两步 RL 和两步 OPD 都**没有**成为质量候选：

- RL：`false`；
- OPD：`false`。

它们的实现准入仍然成立，但不能据此扩大训练或访问 benchmark。

## 四臂结果

| Arm | Correct | Accuracy | Parse failures |
| --- | ---: | ---: | ---: |
| `base4` | 3/96 | 0.0312 | 0 |
| `rl4` | 3/96 | 0.0312 | 0 |
| `opd4` | 3/96 | 0.0312 | 0 |
| `base9` | 6/96 | 0.0625 | 0 |

## Paired 比较

| Comparison | Delta | 95% bootstrap CI | McNemar p | Wins/Losses |
| --- | ---: | --- | ---: | ---: |
| `rl4_vs_base4` | +0.0000 | [+0.0000, +0.0000] | 1 | 0/0 |
| `opd4_vs_base4` | +0.0000 | [+0.0000, +0.0000] | 1 | 0/0 |
| `base9_vs_base4` | +0.0312 | [-0.0208, +0.0833] | 0.453125 | 5/2 |

## 具体观察

- base4 只有 3/96，三个正确样例都来自 exact-division family；
- rl4 同样是 3/96；raw output SHA 与 base4 不同，说明 adapter 改了输出，
  但没有改正确率；
- opd4 也是 3/96，且 96 条 raw output 与 base4 逐字相同；
- base9 提供相同 case、prompt、parser、budget 下的 reference。

因此 “probe logits changed” 只能证明 adapter 生效，不能证明质量提升。这次
fresh evaluation 正好区分了实现证据与质量证据。

## Candidate gates

```json
{
  "opd4": {
    "admitted_synthetic_quality_candidate": false,
    "gates": {
      "bootstrap_ci_lower_positive": false,
      "every_family_non_regression": true,
      "mcnemar_below_005": false,
      "parse_failures_non_regression": true,
      "point_delta_positive": false
    }
  },
  "rl4": {
    "admitted_synthetic_quality_candidate": false,
    "gates": {
      "bootstrap_ci_lower_positive": false,
      "every_family_non_regression": true,
      "mcnemar_below_005": false,
      "parse_failures_non_regression": true,
      "point_delta_positive": false
    }
  }
}
```

## Evidence

- preregistration SHA：`d00f661d480d0a99af68233e4826f3d609c592ef97d2a5e2722e8f2c21e2a31c`；
- case contract SHA：`c9de0fcd02da9354aa38ac1f5c864f6bfc4bf6590bca3313d54232c1bac41e4b`；
- base4 summary/raw SHA：`eba5dc100b9e53d0cbe66e6584c5901aba78f1d7d2c32b1e30df338a7c4b7645` / `05833b188e25fa2d8737032ca68bd017c83cd6f4098976e734352910eba011ff`；
- rl4 summary/raw SHA：`59e6bb51637110a5a3094c4cdc370a96b49ecc6f14cb41220d3365db0c5e0295` / `8e62ea2329a28dad443119601f4551f474321f07617222ac72d99d25fee569ee`；
- opd4 summary/raw SHA：`eafa7ef7cea95b41cb1e2d202890d43a41455eac155be9a261c954df73519032` / `05833b188e25fa2d8737032ca68bd017c83cd6f4098976e734352910eba011ff`；
- base9 summary/raw SHA：`6b8f7c69df71992adab8c5304d397631a791901df6090dbf638fbde09d709970` / `9359a4902586d0329ced23d6c5483fd3bddd75bede5a4c04b9644ab7c4cdfdda`；

## 下一步

保留 RL/OPD 的可运行机制，但拒绝把两步 adapter 当成质量候选。不要在这
96 cases 上改 task、prompt、reward、teacher、steps、LR、LoRA、budget 或
parser。等待 peer consistency replication，同时只允许另行预注册使用新
synthetic train/dev 数据的 scaled quality intervention。

本结果不是 benchmark、canary、independent holdout 或最终 4B/9B superiority。
