---
id: m9isd5qbwp1o6mobxu2c4hs
title: Test_create_scerevisiae_kg_small
desc: ''
updated: 1790649282296
created: 1790649282296
---

## 2026.09.28 - The small and incremental KG builds on a fake BioCypher (Phase 9)

7 tests: the full build prefilters, caps, skips and writes the import call; the incremental build emits one dataset, prepares the increment, refuses a duplicate-edge risk and requires the staged LMDB; membership and mode are validated after the genome and graph are built (lines 160 to 168) but before any dataset; `_count_while_writing` counts what the writer consumes. Finding: `get_num_workers` (lines 50 to 57) says "CPUs allocated by SLURM" but returns the constant 10 when the variable is unset, the `cpu_count` fallback being commented out, unlike the sibling scripts. Coverage 0% to 100%. Phase 9 of [[plan.test-suite-buildout.2026.09.25]], recorded in [[test-campaign.2026.09.25]].
