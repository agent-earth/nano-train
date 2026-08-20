# Qwen3.5 Anchor-Policy Replay v1 Result

## 结论

- treatment admitted：
  `false`；
- 211-case canary allowed：
  `false`；
- complete benchmark allowed：`false`；
- tuning/rerun on observed dev：`false`。

## Matched arms vs anchor

| Arm | Correct | Delta | Wins | Losses | McNemar p |
| --- | ---: | ---: | ---: | ---: | ---: |
| control | 2/256 | +0.0000 | 2 | 2 | 1 |
| treatment | 3/256 | +0.0039 | 3 | 2 | 1 |

两臂 baseline rows 逐条完全一致；唯一隔离因素是 treatment replay step 上
weight=1.0 的 frozen anchor top-64+other policy KL。

## Treatment vs control

- delta `+0.0039`；
- paired bootstrap 95% CI
  `[-0.0117,
  +0.0234]`；
- exact McNemar `p=1.0`；
- wins/losses：
  `3/
  2`。

## Frozen gates

```json
{
  "both_arms_finite_and_reloadable": true,
  "teacher_cache_finite_and_identity_verified": true,
  "treatment_accuracy_gt_anchor": true,
  "treatment_accuracy_gte_control": true,
  "treatment_anchor_bootstrap_ci_lower_gt_zero": false,
  "treatment_anchor_exact_mcnemar_p_lt_005": false,
  "treatment_anchor_maximum_losses": false,
  "treatment_anchor_minimum_wins": false,
  "treatment_every_family_non_regression_vs_anchor": false,
  "treatment_losses_lt_control": false,
  "treatment_parse_non_regression_vs_anchor": true
}
```

## Evidence

- prereg SHA：`109e6fecd7e813b6474f737a77d8667e6d165e58fcfe96de8d570041f53a920d`；
- config SHA：`47c2d1df78a4beca4ed41a7ef381dca5240eedb07834c349cb677a7b8249b3c6`；
- cache receipt / raw SHA：
  `6eff8e83b198579188d67c779c750d8592e93439ea1423e022294120edfb9fc6` /
  `f30308453dcd0aff603b4aafffe59db177b4bf7c3fcbad915f7de3aa5523ebe3`；
- control metrics/reload/generations：
  `bb885da107360a9591b3758317a7c049b5faa92f6a92e1760901bf67361acbf8` /
  `01d49a3d9386d3ce1da8eacb197b5b497e9d4522613b3205048d6b9e7e7293c2` /
  `909b78fd35ffc9dfd079b366817abfcd8d94140a4888d0de52ed36f8ce50596e`；
- treatment metrics/reload/generations：
  `31b14e25d0439a44ad7ab8074e8684296aa24f139de1fddf02ea1bd5fb37bda4` /
  `084783f931c5c462e069460ee13aa6b9cbddce2e6eba20b8892c9641ad16c26d` /
  `8e2c65821ac896af4631bebea785aaf79a03cdd6928e7e756725d61e7e49e9d5`。

公开报告只包含聚合指标、case IDs 和 SHA，不包含 prompt、target、output 或
teacher logits。通过只允许 treatment 进入已预注册 211-case canary；完整 benchmark
与 independent holdout 继续关闭。
