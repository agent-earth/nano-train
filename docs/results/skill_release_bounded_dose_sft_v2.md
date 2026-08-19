# Skill Release Bounded-Dose SFT v2

## 配置

- 80 train / 20 frozen dev，五个 family 均衡；
- 20 optimizer steps；
- max length 1088；
- LoRA q_proj + v_proj；
- 模型、LR、seed 与 4-step smoke 相同，只增加 dose 和 dev 数。

## 稳定性

- Loss：2.358655, 0.886619, 0.108165, 0.126122, 0.668682, 0.955835, 0.578958, 0.813879, 0.000847, 0.092479, 0.094225, 0.082798, 0.000688, 0.086199, 2.104096, 0.072032, 0.762484, 0.085006, 0.000559, 0.961193；
- Peak：19.85 GiB；
- Adapter tensors：32；
- Non-finite tensors：0；
- Independent reload 一致：true。

## Verified Dev

- Baseline：17/20；
- Post-SFT：17/20；
- Delta：+0；
- 改变输出：9/20；
- Family non-regression：
  true。

Adapter 确实改变了 9/20 个输出，但 verified 分数没有提高。

## 决策

- accepted_local_smoke：true；
- positive_dev_delta：false；
- larger_training_allowed：false；
- benchmark / holdout / RL：全部关闭。

下一步：Do not increase dose or search LR/seed. The adapter changes outputs but does not improve verified dev. Investigate the remaining verified-reasoning failures and choose one method-level intervention with protected JSON families.

## Evidence

- config SHA256: `fd6303456fb192c613b20d049219900e4977a407b47c38150a90e59e543bd059`;
- dataset SHA256: `b5503761900cfa290dba03aff306ff511630d01544a4dab93de3a0be1e74abc1`;
- adapter SHA256: `96263ed167ca989002b4dee510c1f5d894bbdb809fc1b991f7b827b9450d62dd`;
- metrics SHA256: `6e60285113e752e5f0dae8a6c4a66014c908117456efbd00044b06748e1c8c93`;
- generations SHA256: `75f0127eb80654617bc729d4c71525ae35af01dadfa1768dce5d97316eb5fce8`;
- reload SHA256: `afd4667a848ea00f5c7ccb9091b57063f98ebcef01affd8057970ada234900ce`;
- rescore SHA256: `fe6977592c90f652ac0ed0c1045f68d9fc2a5471033fbce34f1e79d31ea739fc`.

## 结论边界

The bounded dose proves stable optimization and adapter effect, but verified dev remains unchanged. It does not justify larger training, benchmark access, or quality claims.
