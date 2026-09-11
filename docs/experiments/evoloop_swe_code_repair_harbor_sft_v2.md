# EvoLoop Harbor-Prompt SFT v2

## Single Variable

Train the same 160 SWE code-repair rows, in the same order and with the same
optimizer, seed, and q/v-only LoRA as v1. The supervision prompt now exactly
matches Harbor Terminus2. `max_length` increases from 2048 to 2816 only because
the full Harbor prompt would otherwise truncate 34 of the unchanged rows.

## Frozen Evidence

- Train rows: 160 ({'long': 40, 'medium': 80, 'short': 40});
- fresh dev rows: 24 ({'long': 8, 'medium': 8, 'short': 8});
- overlap with both prior observed dev sets: 0;
- max full train/dev sequence: 2638 /
  2816 tokens;
- max Harbor dev prompt: 2139 /
  7168 tokens.

## Admission

All gates are required: v2 loss improves at least
5% versus base and
2% versus v1,
24/24 outputs are parser-valid, there are at
most 0 v1-only structural wins, no band
regresses, no v2 output exhausts its budget, all training values are finite, and
independent reload is exact.

## Boundary

Passing allows only the frozen fresh-8 SFT screen. It does not support a
SWE-bench score claim or a complete-500 launch.
