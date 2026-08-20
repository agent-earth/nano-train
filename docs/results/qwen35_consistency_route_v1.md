# Qwen3.5 Conservative Consistency Route v1 Result

## 结论

- local route admitted：
  `false`；
- canary allowed：`false`；
- benchmark allowed：false。

## Arms

| Arm | Correct | Accuracy | Parse failures |
| --- | ---: | ---: | ---: |
| anchor | 2/256 | 0.0078 | 0 |
| consistency | 10/256 | 0.0391 | 0 |
| routed | 2/256 | 0.0078 | 0 |

## Routed vs anchor

- delta `+0.0000`；
- 95% CI
  `[-0.0117,
  +0.0117]`；
- exact McNemar `p=1.0`；
- wins/losses：
  `1/
  1`。

## Route

- exact-division：consistency；
- 其他 192 cases：anchor fallback；
- confidence/output/expected-answer routing：false。

```json
{
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

- wins：`['consistency-route-exact_division-0b097059c19d6a22']`；
- losses：`['consistency-route-exact_division-0ba6fcf9b3f0971c']`。

只公开 case IDs，不公开 prompt、target 或 output。

## Evidence

- prereg SHA：`1cd6e1377cc18f36af2bdf6dbc8647f8202127022c2c2e69c037672d96157489`；
- anchor summary/raw：
  `7303175aa55f47089cc673e676a61d651430e26c9ec6c0f20718b8e5b8c7cafd` /
  `2734c8266fc39a96b8d47f244585a29617bed5beedf122619f995f70b760462f`；
- consistency summary/raw：
  `b187d60bbcfd772f23cb49a5e3f3a4528823fb44c560a54d5c85705f75b3cd29` /
  `3f944bfb9a688e1e1faa8309a81d6019f901a092b3477378ca6bbf0e82425a4f`；
- routed raw：`00080a26a10ba4435264a500daba8bb0a919ca39f3d9f972c8bf871d8f0263ae`。

若 gate 通过，下一步只能消费完全相同的 route 和 adapter identities 进入已预注册
211-case canary；不能修改 route、threshold、prompt、parser 或 adapter。
