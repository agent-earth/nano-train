# Qwen3.5 Scaled Quality SFT v1

## 假设

两步 RL/OPD 只证明机制可运行，没有改变 96-case correctness。下一实验只改变
训练规模和监督覆盖：用新生成的 512 条 SFT train rows，训练 128 steps，
检查 96 条未触碰 dev 是否出现显著提升。

## 数据

- train：`512` rows，每个 family
  `128`；
- dev：`96` rows，每个 family
  `24`；
- Qwen3.5 tokenizer train tokens：
  `39,852`；
- maximum sequence：`87`；
- train/dev prompt overlap：0；
- 与已观察 96-case quality suite prompt overlap：0；
- benchmark/canary/holdout rows 和 outputs：0。

public receipt 只保存 case IDs、family、prompt SHA 和 target SHA；raw expression、
target、generation、adapter 都在 ignored `artifacts/`。

## 训练

- Qwen3.5-4B FP32；
- q/v LoRA r=8 alpha=16；
- batch=4；
- steps=128；
- exactly one epoch，每个 train row 恰好曝光一次；
- LR=5e-05；
- seed=20260820。

## 验收

- finite + independent reload；
- post accuracy > baseline；
- paired bootstrap 95% CI 下界 > 0；
- exact McNemar p < 0.05；
- 至少 12 wins、0 losses；
- 每个 family 不回退；
- parse failures 不回退。

即使通过，也只建立 fresh synthetic quality evidence；canary、benchmark 和
independent holdout 仍需单独预注册。

## 执行边界

```json
{
  "benchmark_accessed": false,
  "dev_observed": false,
  "model_generation_started": false,
  "this_commit_only_preregisters": true,
  "training_started": false
}
```
