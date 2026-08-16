# Format Contract SFT Smoke v1 Result

## Result

The pre-registered SFT smoke fails.

- baseline exact validation: 23/26
  (0.8846);
- post-SFT exact validation: 0/26
  (0.0000);
- first finite loss: 0.148261;
- first non-finite loss step: 2;
- finite optimizer steps: 1/20;
- peak allocated memory: 10.11 GiB.

Loss becomes non-finite at step 2 and remains non-finite. Validation collapses
from 23/26 to 0/26. The saved adapter is invalid and must not be evaluated,
merged, published, or used to start RL.

## Runner Defect

The v1 runner did not stop on non-finite loss. It continued through 20 steps
and saved an invalid adapter. Preserve these artifacts as failure evidence, add
fail-fast before optimizer continuation and artifact acceptance, then diagnose
the first update under a separately pre-registered v2.

The adapter files contain FP32 LoRA tensors, but all saved tensors are
non-finite. This rules out the simple claim that saved adapter weights remained
FP16. The exact instability source remains unresolved.

## Reproduction Identity

- pre-registration revision: `990b695`;
- config SHA256: `09bbf842ea2a335e283385eeea18d352f9311dc5747e86da9be9b58bfdae2d93`;
- dataset SHA256: `46f2128f219db7011d5db95b5ca3a97029b57f5ac959e194860b4c0f4ba3ad53`;
- model config SHA256: `ddc63e1c717afa86c865bb5e01313d89d72bb53b97ad4a8a03ba8510c0621670`;
- metrics SHA256: `3c99f9094a08dd980f428707e59146fea662a6dddb536d0da48ded8e7b7c41f3`;
- generations SHA256: `b7959ec7c072c6e5932acb8f9cb25e78c80dd0e1e5776bbd6e21d7125f8cc709`;
- adapter tree SHA256: `26247565058e52c0a1c0e104704d4442c719dfcea922fc604fbf3e7832c51452`.
