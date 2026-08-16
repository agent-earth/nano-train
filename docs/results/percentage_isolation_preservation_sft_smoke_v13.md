# Percentage Isolation Preservation SFT Smoke v13 Result

## Result

V13 is stable but fails the frozen local gate.

- aggregate exact / semantic: 22/32 / 23/32;
- numeric exact / semantic: 9/16 /
  10/16;
- choice exact / semantic: 5/8 /
  5/8;
- process exact / semantic: 8/8 /
  8/8;
- early / late five-step mean loss:
  0.204367 / 0.012033;
- peak training memory: 20.17 GiB;
- independent reload exact / semantic:
  22/32 /
  23/32;
- post outputs at the cap:
  0/32.

All 32 losses and all 224 FP32 adapter tensors
are finite. Independent reload reproduces all metrics and failure IDs.

## Isolated Mechanism

Relative to v11, seven percentage-family exposures fix zero semantic cases and
regress three: two numeric and one choice. Aggregate semantic falls 26/32 to
23/32. Choice/process/targeted-host exposure is unchanged, while packing and
schedule families are absent.

The percentage family alone is harmful at this frozen dose. Stop this family;
do not conduct post-hoc smaller-dose search on the same development split.

## Decision

Reject v13 and preserve v11. Do not run the sealed canary, prior full suite, or
independent holdout. The holdout remains unread.

## Reproduction Identity

- pre-registration revision: `108e9e0`;
- data revision: `a8db4f4`;
- config SHA256: `98057d4ea24e3d24ada9d98c0dd5af14fc1f08bb07436e1a33bd479a4131686e`;
- dataset SHA256: `0ae81bb4c385703592946b5c75971b39cbb388b02a76fafa477e53e55756bc9c`;
- model config SHA256: `ddc63e1c717afa86c865bb5e01313d89d72bb53b97ad4a8a03ba8510c0621670`;
- metrics SHA256: `ed2176a896e2cf87f44950ff9a883a58710598b8ed8c6a69f289c614711b28b4`;
- generations SHA256: `c75af20de42c35ef7c83c7a580cbce15bc662454e0c25fe615860af0cea7dead`;
- reload receipt SHA256: `59951eed25529f07d536d3a04f3a9fa4b228192e123f8a4795af4a8c36326bf7`;
- adapter tree SHA256: `415509176374da8f7e21b1ee85edb89f654bfe52881ca50b88e777a09a862c01`.
