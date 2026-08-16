# V11 Schedule B-Only Anchored Continuation Result

The anchored continuation passes every frozen local gate:

- baseline reproduces v11 at 23/32 strict and 26/32 semantic;
- post result: 22/32 strict, 25/32 semantic;
- numeric / choice / process semantic: 11/16, 6/8, 8/8;
- 112/112 LoRA A tensors remain byte-identical; 112/112 B tensors change;
- relative B drift: 0.120085;
- independent reload exactly matches.

This is a preservation-method pass, not independent quality evidence. It
authorizes only the old sealed regression canary. Full suite, independent
holdout, merge, scale, and RL remain blocked.
