# Qwen3.5 Scaled Quality SFT v1 Result

## 结论

候选被拒绝。512 条新 train rows、128 steps 的 SFT 在 untouched dev 上从
`0/96` 提升到
`2/96`，但只有 2 wins / 0 losses：

- delta `+0.0208`；
- 95% bootstrap CI
  `[+0.0000,
  +0.0521]`；
- exact McNemar `p=0.5`。

CI 下界仍为 0，p 不显著，并且未达到预注册的 12-win gate。

## 训练

- 512 train rows，512 unique exposures；
- 128 optimizer steps，batch 4，exactly one epoch；
- trainable LoRA parameters：
  `917,504`；
- first loss：`1.218617`；
- last loss：`0.747872`；
- minimum loss：`0.228398`；
- maximum gradient norm：
  `16.472870`；
- peak GPU memory：`17.00` GiB；
- independent reload：
  `true`。

## Fresh dev

| Family | Base | Post-SFT |
| --- | ---: | ---: |
| exact_division | 0/24 | 1/24 |
| mixed_products | 0/24 | 0/24 |
| nested_offset | 0/24 | 0/24 |
| repeated_operand | 0/24 | 1/24 |

两条 win 分别来自 exact-division 和 repeated-operand。public report 只保留
case IDs，不包含 raw expression、target 或 output。

## Gate

```json
{
  "every_family_non_regression": true,
  "exact_mcnemar_p_lt_005": false,
  "finite_training": true,
  "independent_reload_exact": true,
  "maximum_baseline_only_losses": true,
  "minimum_candidate_only_wins": false,
  "paired_bootstrap_ci_lower_gt_zero": false,
  "parse_failures_non_regression": true,
  "post_accuracy_gt_baseline": true
}
```

## Evidence

- preregistration SHA：`267bcd69df6ce468e14f68326c05c92de92c1e9d859a15dd68e07fff5d097090`；
- metrics SHA：`625d86c1d9299bab3fb4e581e24438d3fbfbf01684426bce4676fbf6c724042f`；
- reload SHA：`ebd4b2f25f18d704561935e9fc25b43c71bfd6bd9236cad554a2bd4d36cb9679`；
- adapter tree SHA：`7287cfcc9372894aae2c4081f6ccf7b18a9292a9e52160579c9ab199906eda45`；
- generations SHA：`fec43aad912017179aefc029ce15dd071a6c362ffed7e950926d37b75c2f0256`。

## 下一步

保留这 2 个 directionally correct wins，但拒绝当前 adapter。不要在已观察的
96 dev rows 上增加 steps、改 LR、seed、LoRA、prompt、parser 或 adapter
weight。下一轮必须更换监督机制，或使用全新的 train/dev surface。

benchmark、canary 和 independent holdout 继续关闭。
