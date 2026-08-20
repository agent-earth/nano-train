# Qwen3.5 Normalized-Confidence Route v1 Result

## 结论

- local route admitted：
  `false`；
- 211-case canary allowed：
  `false`；
- complete benchmark allowed：`false`；
- route tuning on observed cases：`false`。

## Arms

| Arm | Correct | Accuracy | Parse failures |
| --- | ---: | ---: | ---: |
| anchor | 2/256 | 0.0078 | 0 |
| consistency | 2/256 | 0.0078 | 0 |
| routed | 2/256 | 0.0078 | 0 |

## Routed vs anchor

- delta `+0.0000`；
- paired bootstrap 95% CI
  `[-0.0117,
  +0.0117]`；
- exact McNemar `p=1.0`；
- wins/losses：
  `1/
  1`。

## Frozen selector

- anchor routes：`3`；
- consistency routes：`253`；
- tie fallback：`anchor`；
- all cross-model candidate scores finite：
  `true`；
- expected answer / correctness feedback：`false`。

```json
{
  "all_scores_finite": true,
  "every_family_non_regression": true,
  "exact_mcnemar_p_lt_005": false,
  "maximum_anchor_only_losses": false,
  "minimum_candidate_only_wins": false,
  "paired_bootstrap_ci_lower_gt_zero": false,
  "parse_failures_non_regression": true,
  "routed_accuracy_gt_anchor": false
}
```

## Discordant cases

- wins：`['confidence-route-repeated_operand-aa50166e3c9ac79d']`；
- losses：`['confidence-route-repeated_operand-868de251171faef4']`。

只公开 case IDs、SHA 和聚合指标，不公开 prompt、target、output 或逐条 score。

## Evidence

- prereg SHA：`038f9688ce3b05eb45f3d7500b266b6d3a6fef3a235237af4082363c77f536c8`；
- config SHA：`07c2accc1dbfef131ce822f6b02a55d1c631788da31a4c4f48485e0810e1cba0`；
- anchor generation raw：
  `5a644de18055ea48b8b8367caf07f78a2a9263cba724bce3e26699c7f75960a7`；
- consistency generation raw：
  `b7a4cef28e482b193f931a3e7bfa05086bdcb6002da5e841588cf23fc912f9c6`；
- anchor score raw：
  `6d5e9cc36acc1c0c3b04bd89c353feac69bbabe53f1aeca64a4ead309c917818`；
- consistency score raw：
  `800fff0485eb8ef52e8b939c45b334726ccdea52097a788a05b7c766e3763403`；
- routed raw：`b9b114d6324615b259336c40b072869fb60d07dcafb88cf3ce0d1abf003222da`。

若 gate 通过，下一步只能用完全相同的 selector、candidate、score、adapter identities
进入已预注册 211-case canary；不能修改 threshold、tie、prompt、parser、budget
或 adapter。无论通过与否，完整 benchmark 和 independent holdout 都保持关闭。
