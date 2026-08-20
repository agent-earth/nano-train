# Qwen3.5 Router Classification SFT Smoke v1 Result

## 结论

- admitted：`true`；
- baseline：112/192；
- post SFT：192/192；
- delta：+80；
- independent reload：192/192，与 post metrics 完全一致。

## Per Label

| Label | Baseline | Post | Delta | Reload |
| --- | ---: | ---: | ---: | ---: |
| router_a | 24/64 | 64/64 | +40 | 64/64 |
| router_b | 59/64 | 64/64 | +5 | 64/64 |
| router_c | 29/64 | 64/64 | +35 | 64/64 |

## Training

- 40 steps，effective batch 4；
- FP32 expanded LoRA，9,961,472 trainable parameters；
- loss：0.248296 →
  0.000010；
- peak：18.67 GiB；
- reload peak：15.81 GiB；
- wall：324.5s。

## Gates

```json
{
  "adapter_identity_matches": true,
  "aggregate_post_exact_gt_baseline": true,
  "data_release_identity_matches": true,
  "every_label_non_regression": true,
  "finite_loss_curve": true,
  "no_failure_receipt": true,
  "reload_exact_metrics": true,
  "reload_success": true,
  "router_a_post_exact_at_least_48_of_64": true,
  "router_b_post_exact_at_least_48_of_64": true,
  "router_c_post_exact_at_least_60_of_64": true
}
```

## Evidence

- prereg commit：`9397470864d76016a174af7cbee098e72e5dcd9b`；
- adapter SHA：`48d72666cf269f64825791801c71aad5ae1d9e97cbc70574c216b12974e46e63`；
- metrics SHA：`d65f323aafc6e4b4d5b680148aa8d8d3e8d22eae43109b75d154ed6ffc36ca99`；
- generations SHA：`226d8ed88197ebb847d216364f4e84b6654e69150c4fdca75867d4297d9dcfff`；
- reload SHA：`645b61a679b213e6bdb86d9c884e47c25049ad671051bd026050833ad97fd316`；
- dataset SHA：`dacd3663639fe9ddc054865b87afdd0c918f0fddb12c8c9355819d4bbce95d65`。

## 边界

这是 synthetic router classification smoke。Adapter/raw generations/metrics 保持
ignored，public commit 只记录 hashes 和 aggregate。通过只允许另行预注册 fresh
router integration；benchmark/canary/holdout/RL 继续关闭。
