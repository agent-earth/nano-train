# V11/V15 Exact LoRA Composition 75/25 Result

The exact rank-16 composition represents:

```text
0.75 * delta_v11 + 0.25 * delta_v15
```

All 112 module pairs and 224 tensors pass block parity with zero error.

Two independent FP32 evaluations are byte-identical:

- aggregate exact / semantic: 21/32 / 27/32;
- numeric exact / semantic: 8/16 / 14/16;
- choice: 5/8;
- process: 8/8;
- peak memory: 15.85 GiB.

The composition retains most of v15's semantic gain but does not restore v11
strict/choice preservation. It fails strict exact 22/32 and is not Pareto over
v11.

Reject canary, prior full suite, independent holdout, merge, scale, and RL. Do
not perform post-hoc composition-weight search on this development split.

V11 remains current. The next method should change the training objective or
stage preservation explicitly rather than interpolate adapter weights.

