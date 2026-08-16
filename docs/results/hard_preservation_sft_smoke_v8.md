# Hard Preservation SFT Smoke v8 Result

## Result

V8 is stable and improves aggregate and numeric validation over v7, but fails
the unchanged local gate.

- aggregate exact / semantic: 21/32 / 21/32;
- numeric exact / semantic: 9/16 /
  9/16;
- choice exact / semantic: 4/8 /
  4/8;
- process exact / semantic: 8/8 /
  8/8;
- early / late five-step mean loss:
  0.197267 / 0.040788;
- peak training memory: 20.17 GiB;
- independent reload exact / semantic:
  21/32 /
  21/32;
- post outputs at the cap:
  0/32.

All 40 losses and all 224 FP32 adapter tensors
are finite.

## Dose Ablation

V8 changes only `max_steps` from v7's 20 to 40:

- aggregate semantic delta: +2;
- numeric semantic delta:
  +3;
- choice semantic delta:
  -1;
- process semantic delta:
  +0.

Full coverage improves numeric 6/16 to 9/16 but reduces choice 5/8 to 4/8.
The dose tradeoff prevents promotion.

## Decision

V8 fails aggregate 24/32, strict 22/32, numeric 10/16, and choice 5/8 gates.
Do not run the sealed canary. Do not run the full suite, merge, scale up, or
start RL.

The next separately pre-registered dose interpolation should use 30 steps
(120 unique examples: numeric 63, choice 27, process 30) while freezing data,
generation budget, model, LoRA, and all gates.

## Reproduction Identity

- pre-registration revision: `a65b74f`;
- data revision: `204b053`;
- config SHA256: `1d74ff3fb8a6bd9d87a63d73d19af6b3f21dde4831742bfe7681a9628556039e`;
- dataset SHA256: `ac07e10f1e04ca8ddec74aefc8df9b00475f7fc9c57345ff98182dd8e4c8bae9`;
- model config SHA256: `ddc63e1c717afa86c865bb5e01313d89d72bb53b97ad4a8a03ba8510c0621670`;
- metrics SHA256: `00c0c5c5f4b8378d5840088b971a3d3bd995dddc6e042f79f88ae5edb28c1cdb`;
- generations SHA256: `5dc1d139b95a783f6b06f69d46b917f255c038283455085be03d1394d1dd5ad8`;
- reload receipt SHA256: `5df008ceec0a84406483907bb8a6975d0dd0fa7ddca7e115f4e36bbaf0a7e073`;
- adapter tree SHA256: `22ad63743eb5fa1e4f8e7cea4a2ddf37c2c456e663a6f21f8c68616431644f26`.
