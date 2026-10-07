---
id: 8hem19ixeb0j06talwjfl4m
title: Test_models_genslm
desc: ''
updated: 1791416318611
created: 1791416318611
---

## 2026.10.07 - Coverage

Fourteen tests over the vendored tokenizer and a two-layer, 16-wide GPT-NeoX built in the test
and saved as a Lightning-style `{"state_dict": ...}` with a weights manifest under `tmp_path`:
vocabulary (64 codons + 5 specials), in-frame grouping, refusal of out-of-frame and non-ACGT
input, truncation to 2,048 codons with longest-padding, the four-model registry against the
vendored configs, the provenance pins, manifest-missing and sha256-mismatch refusals, missing
parameter and unexpected key refusals, exact weight round-trip, and pad-masked mean pooling.
See [[torchcell.models.genslm]].
