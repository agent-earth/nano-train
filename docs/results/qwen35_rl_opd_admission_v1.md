# Qwen3.5 RL / OPD Admission v1 Result

## 结论

RL 和 OPD 都通过了**实现准入**，但没有做 benchmark，也没有证明模型能力提升。

| Mode | Steps | Loss | Gradient norm | Finite tensors | Reload exact | Admitted |
| --- | ---: | --- | --- | ---: | --- | --- |
| RL | 2 | [0.0014417454367503524, 0.012689761817455292] | [0.03451624512672424, 0.4846930205821991] | 32/32 | true | true |
| OPD | 2 | [0.006081541068851948, 0.013980121351778507] | [0.11529161036014557, 0.1695849895477295] | 32/32 | true | true |

## 具体做了什么

### RL

- Qwen3.5-4B 在 GPU0 上生成 2 条 synthetic arithmetic rollout；
- exact verifier 对两条都给 `+1`；
- 执行 2 次 REINFORCE + detached base-policy KL 更新；
- 917,504 个 q/v LoRA 参数参与训练；
- probe logits SHA 从
  `12834e2bbf65e991432ea98b4c51fdba2e4162dae9dde58d7cda085e6a0924e6` 变为
  `6d80887c4900e2f7e64bfdc0fadda0e4df5bd01517ecbfa0d7904b102fcf94ce`；
- 独立 reload 后 logits SHA 完全一致；
- peak GPU memory `15.88` GiB。

### OPD

- fresh Qwen3.5-4B 在 GPU0 生成 2 条 on-policy rollout；
- 冻结 Qwen3.5-9B 在 GPU1 对相同 token sequence 输出 teacher logits；
- 4B 执行 2 次 teacher→student KL 更新；
- 917,504 个 q/v LoRA 参数参与训练；
- probe logits SHA 从
  `412f1418364dc2d0d62cbba531469e3645701b2eb95c3eee7ced9f9a9accf63f` 变为
  `a748e76efbccc5094bf7235d8ecb1d24c5cecd5d0ce24efc17c00d296b231836`；
- 独立 reload 后 logits SHA 完全一致；
- student/teacher peak GPU memory
  `15.84` /
  `16.74` GiB。

## 污染审计

两个实验都只用 4 条新 synthetic prompt（2 train + 2 probe）。这些 prompt 与：

- GSM8K 1,319 个题面；
- MMLU 14,042 个题面；
- GPQA-Diamond 198 个题面；

做 normalized exact-hash 比较，重叠为 0。审计没有读取 benchmark label、模型
output、canary 或 independent holdout。

## 失败样例

RL attempt 1 在第一个 optimizer step 前失败：

- stage：`reinforce_gather_before_backward`；
- root cause：student rollout token IDs were inference tensors and could not be saved by autograd gather；
- optimizer steps：0；
- adapter saved：false；
- treatment changed：false。

修复仅把 inference-mode rollout token tensor clone 成普通 tensor，未改变冻结的
task、reward、teacher、seed、LR、steps、LoRA、temperature、top-p 或 budget。
attempt 2 使用原 config SHA 成功。

## Evidence

- preregistration SHA：`bd53a860686a5c3f7878b6642ebf92ab57c2130421f25a7b629dd599f0140d16`；
- RL metrics SHA：`6e3d0e0942b87a30f1147f77bbc72a2bfcccac465b7fb007ee2382309c4961d6`；
- RL reload SHA：`daf39c0b9f9805dd3b7a45cdd15b882f34e018089241e1087f91e057c31c74d0`；
- RL adapter tree SHA：`f270b009214b7508cad5bf1d404f6eaccb9980ca383845a18a6dea397dc9c2fc`；
- OPD metrics SHA：`0895f35eab14494965a900b6914081f969c00e5ec1b0d23627b5f1cd1fe627cc`；
- OPD reload SHA：`d14b888bc75d29c374f9efd5278930f140e09592f54b1047a0b5c8470628148e`；
- OPD adapter tree SHA：`4373481f3e7a90003760f4103a68014002a29c92c0cff1286d8b8947c01006b2`。

## 不代表什么

这次只证明两套机制可以真实更新、finite、产生 adapter effect 并独立 reload。
它不证明训练后回答更好，不开放 benchmark/canary/holdout，也不允许扩大训练。
