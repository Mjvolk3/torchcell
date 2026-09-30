---
id: mhmxqr3hpqoxvxnh2gec4tq
title: Test_fungal_up_down_transformer
desc: ''
updated: 1695841843225
created: 1695841839364
---
We only really test the upstream model. This is the most sensitive case since we have to deal with padding.

## 2026.09.30 - Phase 18: the hermetic half of the wrapper

Thirteen to twenty-five functions (15 new cases); the module-level `network` mark moved onto the original class so the new tests run offline. The download-by-revision call log and a single `eval()`; the warm-cache notice; the exact species-plus-six-mer text; the downstream window with 11 and 300 accepted and 10 and 301 refused; `_pad_sequence` producing 1001 tokens with `pad_start` 2 and `pad_end` 999; short upstream input pooling to [748.25, 10]; 1003 bp unpadded and 1004 refused; `target_layer` as an int, a one-tuple and a two-tuple; batch stacking.

Findings: `VALID_MODEL_NAMES` is never checked, `upstream_agnostic_lm` is accepted with limit 1003 and the default `""` fails only at use (line 111); the message says "must be > 11" but 11 is accepted (189); `(2, 13)` passes the range check and silently averages layers 2 to 12, only 14 is refused (226). The SpeciesLM hidden-state count of 13 is taken from the existing network test's message, not re-measured.
