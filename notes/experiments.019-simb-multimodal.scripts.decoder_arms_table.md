---
id: 74lxvxl3vs4tnpgyx3wt758
title: Decoder_arms_table
desc: ''
updated: 1788996293433
created: 1788996293433
---

## 2026.09.09 - The decoder-family table for the expression document

Answers one question in `notes-tex/019-simb-multimodal-expression/`: was a head or decoder
family comparison (GEARS-style cross-gene readout, bilinear pair term, Perceiver mixing,
response basis, context concatenation, graph propagation, null sink, deeper perturbation
heads) ever run. It was, in waves 1 to 3 of `torchcell_019_expr_v8`, and every run
stopped between epochs 50 and 276 against a curve that has not turned by 9,900, so the
table records what was tried and at what budget and is labeled as not a comparison.

Reads `results/decoder_arms_torchcell_019_expr_v8.csv` (from `score_decoder_arms.py`) and
parses each arm's `OVERRIDES=(...)` from `gh_expr_008_arm.sh`, the launcher's own text,
so the mechanism column is a gloss beside the override rather than a retyping. Waves 4a
and 4b (null-sink stability pairs) and the learning-rate and scheduler arms are excluded.
Writes `tables/decoder_arms_v8.tex` (21 arm-wave cells, 35 runs) and
`results/decoder_arms_v8_table.csv`. Override strings get `\allowbreak` after `.`, `=`,
`,` and `[` so the 80-character graph list breaks inside its column; without that the
table ran 67 mm past the text block.
