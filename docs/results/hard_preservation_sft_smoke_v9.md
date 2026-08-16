# Hard Preservation SFT Smoke v9 Result

## Result

V9 is stable and is the best aggregate dose tested, but fails the unchanged
local gate.

- aggregate exact / semantic: 22/32 / 22/32;
- numeric exact / semantic: 9/16 /
  9/16;
- choice exact / semantic: 5/8 /
  5/8;
- process exact / semantic: 8/8 /
  8/8;
- early / late five-step mean loss:
  0.197262 / 0.049462;
- peak training memory: 20.17 GiB;
- independent reload exact / semantic:
  22/32 /
  22/32;
- post outputs at the cap:
  0/32.

All 30 losses and all 224 FP32 adapter tensors
are finite.

## Dose Ablation

| Run | Steps | Aggregate | Numeric | Choice | Process |
| --- | ---: | ---: | ---: | ---: | ---: |
| v7 | 20 | 19/32 | 6/16 | 5/8 | 8/8 |
| v9 | 30 | 22/32 | 9/16 | 5/8 | 8/8 |
| v8 | 40 | 21/32 | 9/16 | 4/8 | 8/8 |

Thirty steps preserve choice 5/8 while matching v8 numeric 9/16. It is Pareto
better than both endpoints, but remains below promotion.

## Decision

V9 reaches strict 22/32, choice 5/8, and process 8/8, but fails aggregate
24/32 and numeric 10/16. Do not run the sealed canary. Do not run the full
suite, merge, scale up, or start RL.

The next separately pre-registered interpolation may use 32 steps while
freezing data, generation budget, model, LoRA, and all gates.

## Reproduction Identity

- pre-registration revision: `6004a3f`;
- data revision: `204b053`;
- config SHA256: `b83a98b345a96971207c4883db507024283a1c7ea073b853bc7e30c6f28ff7f1`;
- dataset SHA256: `ac07e10f1e04ca8ddec74aefc8df9b00475f7fc9c57345ff98182dd8e4c8bae9`;
- model config SHA256: `ddc63e1c717afa86c865bb5e01313d89d72bb53b97ad4a8a03ba8510c0621670`;
- metrics SHA256: `72d89ea6a43fe8ec9fbaaf961e089b97e72efafa4deb4467c5c6128bd867b679`;
- generations SHA256: `3788fd56a65fe9bbbb915df1825129df5083d4276ad6138a0d8302ae2117ff1f`;
- reload receipt SHA256: `678e1307a326b71b5047a3783f0dbed72fe5c0adc657d7624e416b01b0806984`;
- adapter tree SHA256: `40e5f4a6cd856354361b62c191248759b9bba4725cdb1196e126416ab193d100`.
