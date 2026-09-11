# EvoLoop SWE Code-Repair Harbor Qualification v1

## Frozen Evidence

- 24 previously unobserved release-dev rows: 8 short, 8 medium, 8 long;
- zero overlap with the 12 development rows used by SFT smoke v1;
- unchanged base model and unchanged adapter;
- exact Harbor Terminus2 prompt template and JSON parser pinned by SHA256;
- greedy decoding, 7168 input tokens, 768 output tokens, thinking disabled.

## Admission

The adapter reaches fresh-8 only if all conditions pass:

- held-out loss improves by at least
  5%;
- at least 23/24 outputs are accepted
  by the exact Harbor parser;
- at least 1 paired adapter-only format
  repair and at most 0 base-only loss;
- no short, medium, or long band regresses;
- no adapter output exhausts the 768-token budget;
- the source adapter has an exact independent reload receipt.

## Boundary

No training occurs in this experiment. Passing only allows fresh-8 SFT
screening; it does not support a SWE-bench performance claim or a complete-500
launch.
