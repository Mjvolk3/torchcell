---
id: jwtolp4xyr76psvnij7hs3v
title: Test_dense_padding_data_loader
desc: ''
updated: 1790549272058
created: 1790549272058
---

## 2026.09.27 - The dense collate on two hand-built HeteroData objects

Eleven tests: exact padded tensor shapes and values (the originals where present, the pad value where not), the mask tensors, the per-node-type and per-edge-type handling, and the error paths. Coverage of the module from this file 93%; the rest is the torch < 2.0 storage branch, the `TensorFrame` branch (`torch_frame` is not installed, so the class is a placeholder) and the `OnDiskDataset` branch. Findings: floats pad with `1e-5`, not zero (dense_padding_data_loader.py line 19); `follow_batch` is accepted and ignored, the dense batch carries no `batch`, `ptr` or `x_batch` (line 219); the nested-tensor `NotImplementedError` is unreachable because `_dense_pad_tensor` runs first and `pad_sequence` raises torch's own `RuntimeError` (lines 124 to 134). Phase 7 of [[plan.test-suite-buildout.2026.09.25]], recorded in [[test-campaign.2026.09.25]].

## 2026.09.30 - Phase 16: nested padding, legacy branches, a real on-disk dataset

Eleven to nineteen tests, 93 to 99 percent. Nested padding, `*edge_index` keys, float64, the legacy shared-storage branches, TensorFrame handling, a real `OnDiskDataset`.

Findings: the recursion drops a custom padding value (lines 190, 204); a real value equal to the pad (-1 or 1e-5) is masked as padding (161); the fallback reads `WITH_PT112`, which torch_geometric does not define, so it raises `AttributeError` (151); the TensorFrame branch raises `AttributeError` because torch_frame is not installed (373).

## 2026.09.30 - Findings retired (issue #538)

- Retired: the recursion dropping custom padding values, the `value != pad` mask, the undefined `WITH_PT112` read, and the collater's TensorFrame `AttributeError`.
- Now asserted: -7 and 0.5 pads inside dicts and lists; a real -1 id and 1e-5 feature stay True while only the pad row of a shorter graph is False; the worker batch with `WITH_PT20` False equals the main-process batch; both TensorFrame branches raise the same `NotImplementedError`. The two legacy shared-storage parametrizations were removed with the branches they covered. The nested-tensor finding remains pinned.
