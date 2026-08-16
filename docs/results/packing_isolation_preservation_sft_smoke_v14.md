# Packing Isolation Preservation SFT Smoke v14 Result

V14 is stable but fails the local gate:

- aggregate exact / semantic: 21/32 /
  25/32;
- numeric exact / semantic:
  8/16 /
  12/16;
- choice: 5/8;
- process: 8/8;
- early / late loss: 0.199554 / 0.020787;
- finite tensors: 224/
  224;
- maximum output: 64/128 tokens.

Relative to v11, five packing exposures fix one numeric semantic case, regress
one numeric case, and regress one choice case. Numeric semantic remains 12/16,
but strict falls 23/32 to 21/32 and choice falls 6/8 to 5/8.

This is not a Pareto improvement. Reject v14, stop packing-family dose search,
and preserve v11. Canary, prior full suite, and independent holdout remain
unrun; the holdout is unread.

Identity:

- pre-registration: `dfe69d9`;
- data revision: `25451af`;
- config SHA256: `7206a76fa6d8307e4c1a42ce753bce358990e65bd4a77bf8881f86c5b55bd773`;
- dataset SHA256: `9f79b1cf5af9fa4b36c7507318b32991692f253d2210b5b6ed70a44bee940f2d`;
- metrics SHA256: `cf0c8668926a35fdbc07ef172572489a9030e9c61ae5c23ec6c8a1e611af6a1c`;
- generations SHA256: `92e7fe2dd16955a19b534a03ce01dfbf31956c43d93350354df05a03801d1b49`;
- reload SHA256: `70690af97b8018d723a6b945004971f65c06d49a65b89613281dddb8a7e6a253`;
- adapter SHA256: `92f517fa18e76996f0460745f2ecdb37288e16edf404dbbd3bc7de53560e8797`.
