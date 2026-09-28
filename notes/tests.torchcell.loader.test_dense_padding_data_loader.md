---
id: jwtolp4xyr76psvnij7hs3v
title: Test_dense_padding_data_loader
desc: ''
updated: 1790549272058
created: 1790549272058
---

## 2026.09.27 - The dense collate on two hand-built HeteroData objects

Eleven tests: exact padded tensor shapes and values (the originals where present, the pad value where not), the mask tensors, the per-node-type and per-edge-type handling, and the error paths. Coverage of the module from this file 93%; the rest is the torch < 2.0 storage branch, the `TensorFrame` branch (`torch_frame` is not installed, so the class is a placeholder) and the `OnDiskDataset` branch. Findings: floats pad with `1e-5`, not zero (dense_padding_data_loader.py line 19); `follow_batch` is accepted and ignored, the dense batch carries no `batch`, `ptr` or `x_batch` (line 219); the nested-tensor `NotImplementedError` is unreachable because `_dense_pad_tensor` runs first and `pad_sequence` raises torch's own `RuntimeError` (lines 124 to 134). Phase 7 of [[plan.test-suite-buildout.2026.09.25]], recorded in [[test-campaign.2026.09.25]].
