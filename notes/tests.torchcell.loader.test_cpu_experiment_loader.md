---
id: jiygcig0x0ja7ij7rsjlztb
title: Test_cpu_experiment_loader
desc: ''
updated: 1790773248249
created: 1790773248249
---

## 2026.09.30 - Phase 16: batching, ordering, the parent-death hook

Four to eleven tests, 88 to 100 percent. Batch slicing, the ceil `len`, in-order batches with one worker, an empty dataset, the parent-death hook, the 50-join then terminate path. Hermeticity fix: `test_worker_function_drops_inherited_env` used to call the real `_die_with_parent`, which armed `PR_SET_PDEATHSIG` on the pytest process; it is now stubbed and asserts through a recorder.
