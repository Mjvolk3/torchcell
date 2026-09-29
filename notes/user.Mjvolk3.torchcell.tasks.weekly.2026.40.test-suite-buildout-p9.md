---
id: 3yn2cz9nj5dkem5k58sxfz3
title: test-suite-buildout-p9
desc: ''
updated: 1790646548349
created: 1790646548349
---

## 2026.09.28

- [x] PR-9 (Phase 9, four Fable writers in parallel + four Fable audits, 304 new test functions in 31 files): the whole `sga/` package from 0% to 95 to 100% ([[tests.torchcell.sga.test_image]], [[tests.torchcell.sga.test_cellpose_seg]] and eight more), [[tests.torchcell.verification.test_runners]] (0% to 100%) with [[tests.torchcell.verification.test_rnaseq]] and [[tests.torchcell.verification.test_fitness]], six viz modules to 97 to 100% ([[tests.torchcell.viz.test_graph_recovery]], [[tests.torchcell.viz.test_transformer_diagnostics]], [[tests.torchcell.viz.test_visual_regression]], ...), [[tests.torchcell.scheduler.test_cosine_annealing_warmup]] and [[tests.torchcell.profiling.test_timing]] to 100%, the five KG build scripts to 100% on a fake BioCypher ([[tests.torchcell.knowledge_graphs.test_create_kg]], ...), the three SGA adapters to 100% ([[tests.torchcell.adapters.test_kuzmin2018_adapter]], ...), [[tests.torchcell.data.test_embedding]], [[tests.torchcell.utils.test_utils]]; TOTAL 43.4% -> 48.1% (line only 49.3%); full suite 3350 passed; findings in [[test-campaign.2026.09.25]], the sga rotation sign and the transformer-diagnostics decompression bomb first
- [ ] Decide fix PRs for the Phase 9 live-path findings: `sga/image.py` rotation sign (`_rotate(cents, -theta)` doubles the tilt; a 6-degree plate keeps 16 of 35 colonies), the watershed NaN marker, the unreachable S flag and the dropped bottom-corner colonies; `viz/transformer_diagnostics.py` with `residual_ratios=None` (what the trainer passes when its accumulator is empty) writes a 4.5-gigapixel PNG; the scheduler's explicit-epoch path and `get_last_lr`; the verification runners' backwards L4 label; `create_kg`'s recorded script path; `embedding.__add__` mutating its left operand
