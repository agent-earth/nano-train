# Qwen3.5 Quality Consistency v1 Result

## 结论

这是当前第一个**统计显著**的 fresh local quality gain：

- baseline `2/192`；
- consistency `15/192`；
- delta `+0.0677`；
- 95% CI
  `[+0.0260,
  +0.1094]`；
- exact McNemar `p=0.002349853515625`；
- 15 wins / 2 losses。

但候选仍被拒绝，因为预注册要求 0 baseline-only losses。

## Family

| Family | Baseline | Post |
| --- | ---: | ---: |
| exact_division | 0/48 | 10/48 |
| mixed_products | 0/48 | 0/48 |
| nested_offset | 0/48 | 0/48 |
| repeated_operand | 2/48 | 5/48 |

## 训练

- 256 fresh train pairs / 256 optimizer steps；
- process CE、final CE、detached-teacher KL 全部 finite；
- first step：`{'consistency_kl': 0.2374447137117386, 'final_ce': 0.1830972582101822, 'gradient_norm': 46.07467269897461, 'process_ce': 0.5501426458358765, 'step': 1}`；
- last step：`{'consistency_kl': 0.7757587432861328, 'final_ce': 0.7757704854011536, 'gradient_norm': 27.808500289916992, 'process_ce': 0.18300238251686096, 'step': 256}`；
- peak GPU memory：`16.51` GiB；
- independent reload：
  `true`。

## Gate

```json
{
  "every_family_non_regression": true,
  "exact_mcnemar_p_lt_005": true,
  "finite_training": true,
  "independent_reload_exact": true,
  "maximum_baseline_only_losses": false,
  "minimum_candidate_only_wins": true,
  "paired_bootstrap_ci_lower_gt_zero": true,
  "parse_failures_non_regression": true,
  "post_accuracy_gt_baseline": true
}
```

- significant local improvement：
  `true`；
- candidate admitted：
  `false`。

## Discordant cases

- wins：`['quality-consistency-exact_division-1688d7a102567cff', 'quality-consistency-exact_division-27dbd40fd1db8120', 'quality-consistency-exact_division-6401f96ecf09d402', 'quality-consistency-exact_division-75d40e6b755c483f', 'quality-consistency-exact_division-775ceeb0889b0a87', 'quality-consistency-exact_division-9b958772456fb913', 'quality-consistency-exact_division-a951642040399b58', 'quality-consistency-exact_division-b4160d6eac7bfecb', 'quality-consistency-exact_division-f7757551dc4b9d36', 'quality-consistency-exact_division-ffb887ef583ffa13', 'quality-consistency-repeated_operand-1f6f312d5ae3b908', 'quality-consistency-repeated_operand-2d5cb0ef60990817', 'quality-consistency-repeated_operand-7a82e84070e05ccc', 'quality-consistency-repeated_operand-c4a9c11fbdd60564', 'quality-consistency-repeated_operand-c769ded3dbf8f493']`；
- losses：`['quality-consistency-repeated_operand-1d907cb4f8be03b5', 'quality-consistency-repeated_operand-3f523aff934b3c05']`。

这里只公开 case IDs，不公开 expression、target 或 model output。

## Evidence

- preregistration SHA：`648ee0ed68f4fca2714f684f405b2feccd9b97bd4df0bc6e9a7dbc49e8399cf1`；
- metrics SHA：`1745d27e4c1311d0f61c977b6535be480f6aeb1f5e691f3b882754701f0aed28`；
- reload SHA：`935240f2122dfc8dec0ca6d01a22290fae11594a1f2c23caa24cfc12c48fe129`；
- anchor adapter SHA：`7287cfcc9372894aae2c4081f6ccf7b18a9292a9e52160579c9ab199906eda45`；
- consistency adapter SHA：`23c4760e9d6693c880859c1bd9984164eb5243742ed3517bad4929364643df49`；
- generations SHA：`aee53b77b98a1d719081babdac7a3b4516f7cdbef6f812245d81e7e247a0bc2f`。

## 下一步

保留显著提升，但不访问 canary/benchmark。下一步只允许在全新 local surface
上预注册 conservative adapter routing / rollback，使 consistency 只在可安全
获益的 family/condition 生效并要求 0 losses。禁止在已观察 dev 上调整
consistency weight、steps、LR、seed、prompt、parser 或 adapter weight。
