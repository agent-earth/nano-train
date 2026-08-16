# Schedule Isolation Preservation SFT Smoke v15 Result

V15 is stable and reaches the strongest semantic result, but fails the frozen
strict/preservation gate:

- aggregate exact / semantic: 21/32 / 28/32;
- numeric exact / semantic: 8/16 / 15/16;
- choice: 5/8; process: 8/8;
- versus v11: semantic +2, numeric semantic +3, strict -2, choice -1;
- finite losses/tensors: 32/224; reload exact: True.

Seven schedule exposures produce real semantic gain but contract interference.
Reject v15 for canary/holdout/promotion. Close further family-data expansion
and move to a method-level preservation intervention. V11 remains current.

Holdout prompts/references remain unread.
