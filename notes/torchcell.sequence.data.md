---
id: rxzfzy6t7t8z9pz3159goqo
title: Data
desc: ''
updated: 1695168328312
created: 1695168255107
---

## 2026.09.30 - Named refusals for window, description, CDS and payload edge cases (issue #538)

- `GeneSet.__repr__` returned None for exactly three members, so `repr` raised `TypeError`; sizes up to 3 now list every member and larger sets show three plus `...`.
- `get_chr_from_description` returned None for a description with no `[chromosome=...]` and no `[location=mitochondrion]` tag, against its `-> int`; it now raises `ValueError` quoting the description. The S288C SGD FASTA carries a tag on all 17 records, so genome construction is unchanged.
- `calculate_window_undersized` raised `UnboundLocalError` on a strand other than `+`/`-`, and `calculate_window_bounds` returned a 29 bp window for a 30 bp request on such a strand (the odd base goes upstream, and `.` has no upstream). Both now raise `ValueError("Strand must be '+' or '-', got ...")` first; `calculate_window_bounds` also asserts the returned window is exactly `window_size`.
- `compute_codon_frequency("")` passed validation and divided by zero; it now raises a named `ValueError`, which the codon-frequency dataset already skips.
- `DnaSelectionResult` ran its start/end check `mode="before"`, so a payload without `start` raised a bare `TypeError`; the check now runs `mode="after"` and a missing field is pydantic's `missing` error at `start`.
- Evidence: `tests/torchcell/sequence/test_data.py` (`test_geneset_repr_size_three_and_four`, `test_get_chr_from_description_non_mito_location_and_no_tag`, `test_calculate_window_undersized_unknown_strand_refused`, `test_calculate_window_bounds_odd_flank_is_exact_and_unknown_strand_refused`, `test_compute_codon_frequency_empty_cds_is_refused`, `test_dna_selection_missing_start_is_named_validation_error`).
