# Qwen3.5 Preservation-Aware Dual-View SFT v1 Result

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
| control | 21/256 | +0.0742 | 20 | 1 | 2.09808e-05 |
| treatment | 30/256 | +0.1094 | 29 | 1 | 5.7742e-08 |

两臂 baseline rows 逐条完全一致。control 和 treatment 都使用相同 anchor、
seed、data、pair order、LR、512 optimizer steps、prompt、parser 和 generation
budget；唯一差异是第二步 objective。

## Treatment vs control

- delta `+0.0352`；
- paired bootstrap 95% CI
  `[+0.0039,
  +0.0664]`；
- exact McNemar `p=0.049041748046875`；
- wins/losses：
  `13/
  4`。

## Frozen gates

```json
{
  "both_arms_finite_and_reloadable": true,
  "treatment_accuracy_gt_anchor": true,
  "treatment_accuracy_gte_control": true,
  "treatment_anchor_bootstrap_ci_lower_gt_zero": true,
  "treatment_anchor_exact_mcnemar_p_lt_005": true,
  "treatment_anchor_maximum_losses": false,
  "treatment_anchor_minimum_wins": true,
  "treatment_every_family_non_regression_vs_anchor": true,
  "treatment_losses_lt_control": false,
  "treatment_parse_non_regression_vs_anchor": true
}
```

## Evidence

- prereg SHA：`3acdb6b6f9d8bb779e6bfa5c326be31caacedbe508ed70c28bac29741508c86f`；
- config SHA：`f746058deb503db91d772a3011ed7b965c984dadeb08becafe8b3d32afb8cb9e`；
- control metrics/reload/generations：
  `d674ba7b6836a2bb6ea90038be4888f36e24e5eedc5fb71b70ee81a3be4a0507` /
  `99cc996c9d892b5675affada763b0b19645cf7ea075f3e9c0f9ecf92cee806e2` /
  `9e8ca9823e5b5198dfc75398868757849111a2741d1b18ecd6fe50daf3e59ac2`；
- treatment metrics/reload/generations：
  `b5db28b5663be7536718cc9c0a13954411058654c77a2fbf4f127591c5375d95` /
  `b4e1a60a55a44c4423419383fb2825be48e6fae8cd108ee6721fe877536d3ede` /
  `6dd9062f51c159ddad7979c9c49c72bee64112dbb34cb08b3630fe5cfc9eeca1`。

公开报告只包含聚合指标、case IDs 和 SHA，不包含 prompt、target 或 output。
通过只允许 treatment 进入已预注册 211-case canary；完整 benchmark 与 independent
holdout 继续关闭。
