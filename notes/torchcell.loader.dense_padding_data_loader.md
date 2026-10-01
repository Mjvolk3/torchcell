---
id: 836txt3abyptvi6mzp8vgnn
title: Dense_padding_data_loader
desc: ''
updated: 1790816101841
created: 1790816101841
---

## 2026.09.30 - Length masks, forwarded padding, no dead fallbacks (issue #538)

- The mask was `value != pad`, so a real id of -1 or a feature of exactly 1e-5 was masked as padding. `_dense_pad_tensor` now builds the mask from each value's length along the padded axis (the edge axis for a `[2, E]` `*edge_index*`) and always returns it; bool/uint8 pads are zeroed through that mask.
- The dict and list recursion dropped `float_padding_value`/`non_float_padding_value`; both are forwarded now.
- The worker shared-memory path read `torch_geometric.typing.WITH_PT112`, which does not exist; the pre-2.0 fallbacks are removed and the untyped storage is always used.
- The collater's TensorFrame branch called `torch_frame.cat` (an `object` placeholder without torch_frame); it now raises the same `NotImplementedError` as the attribute collate. TensorFrame support was refused rather than implemented, since nothing in the repo produces one.
- Evidence: `tests/torchcell/loader/test_dense_padding_data_loader.py` (`test_nested_mappings_and_lists_honor_a_custom_pad`, `test_a_real_value_equal_to_the_pad_stays_unmasked`, `test_worker_branch_ignores_the_pre_2_0_flags`, `test_tensor_frames_are_refused_by_collate_and_by_the_collater`).
