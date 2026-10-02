---
id: 9los1u27s7g9y5gr0ftltnd
title: Test_conftest_no_gpu
desc: ''
updated: 1790918746610
created: 1790918746610
---

## 2026.10.02 - The hidden-GPU contract of plain pytest

Two tests for the fifth rule of [[tests.conftest]]. `test_plain_session_has_no_cuda_device` asserts `CUDA_VISIBLE_DEVICES == ""`, `torch.cuda.is_available() is False` and `torch.cuda.device_count() == 0` in the session. `test_a_subprocess_inherits_the_hidden_devices` starts a child interpreter and asserts it prints `'' False`, since several tests run code in subprocesses (the hash-seed test of the SGD gene graph is one). Both skip under `--gpu`, which leaves the devices visible. Before the conftest change both fail on GilaHyper (four cards) and pass on the CI runner (none).
