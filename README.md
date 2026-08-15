# nano-train

`nano-train` is the reproducible training and ablation layer for Ultimate
Distill.

## Progression

1. Validate data and configuration without allocating a GPU.
2. Run a tiny SFT overfit/smoke experiment.
3. Run a small reward/verifier and RL sanity check.
4. Scale only after loss, generation, checkpoint, and benchmark evidence agree.

## Required Evidence

- immutable config and input dataset identity;
- environment and dependency lock;
- seed, hardware, precision, sequence length, and effective batch size;
- train/eval curves and checkpoint lineage;
- sample generations and failure analysis;
- matched downstream benchmark comparison;
- ablations for SFT, RL, SFT+RL, data filters, verifier/reward, and harness
  strategies.

Training success is measured by held-out target behavior, not training loss
alone. Framework selection remains open until baseline and harness evidence
clarify the smallest sufficient implementation.
