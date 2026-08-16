# Targeted Preservation SFT Smoke v11 Result

## Result

V11 passes every frozen local gate.

- aggregate exact / semantic: 23/32 / 26/32;
- numeric exact / semantic: 9/16 /
  12/16;
- choice exact / semantic: 6/8 /
  6/8;
- process exact / semantic: 8/8 /
  8/8;
- early / late five-step mean loss:
  0.185905 / 0.012403;
- peak training memory: 20.18 GiB;
- independent reload exact / semantic:
  23/32 /
  26/32;
- post outputs at the cap:
  0/32.

All 32 losses and all 224 FP32 adapter tensors
are finite. Independent reload reproduces all metrics and failure IDs.

## Data Effect

Relative to v10, aggregate semantic improves 22/32 to 26/32 and strict exact
improves 22/32 to 23/32. Numeric semantic improves 9/16 to 12/16, choice
improves 5/8 to 6/8, and process remains 8/8.

Three prior numeric failures and one prior choice failure are fixed, with zero
new semantic failures. The result supports the diagnosed covariate mechanism,
but the split informed the data intervention and remains development evidence
only.

## Decision

The exact adapter may run the sealed 40-case regression canary. Passing that
canary permits the unchanged adapter to run the full frozen suite but does not
establish quality uplift. Merge, scale-up, and RL remain forbidden.

## Reproduction Identity

- pre-registration revision: `dfdba60`;
- data revision: `ba11804`;
- config SHA256: `9a971cb46a1f5c21164d6117bef40aedfcb7170e9e82604bb7400c942a2be593`;
- dataset SHA256: `ab51a1be5f45d7f71796fbf98ef6cce83ff9cb0f0a756fed01cb1e7aea55651d`;
- model config SHA256: `ddc63e1c717afa86c865bb5e01313d89d72bb53b97ad4a8a03ba8510c0621670`;
- metrics SHA256: `c53784231ba64c80ab41baac81d0c5ddb6e1eaa2e9e458ae94ecdb70474c015f`;
- generations SHA256: `10f73faf7a2fbacc4865c99f781012b5a23c2996a2710b5aecfe82a033158ec7`;
- reload receipt SHA256: `71e00330881e3178caeff46a59a28de68fa007e1ad2fb5e03831682d080cdc51`;
- adapter tree SHA256: `87248908918b06c2d28ff68efd4f0b1ff92ca8bf8b7588e1c7e81a85eb7da852`.
