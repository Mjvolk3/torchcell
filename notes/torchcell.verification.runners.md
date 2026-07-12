---
id: iwieqze1ae069druc5ovm1m
title: Runners
desc: ''
updated: 1783563836060
created: 1783563836060
---

## 2026.07.08 - Runners: the importable harness that runs every family's L0-L4 gate and writes reports

This module is the executable top of the framework: it knows WHICH built LMDBs exist, pins each one's `Provenance` and expected-count oracle, loads its records, dispatches to the matching per-family verifier, adds the cross-source L4 checks, and writes a `verification_report.json` beside each dataset. It was moved out of `scripts/` into `src` (`d457b5d8`) so the runners can be imported, tested, and reused -- it REPLACES the old `scripts/verify_expression_datasets.py` + `scripts/verify_morphology_datasets.py`, which could only be run, never composed.

- The per-family dataset registries (expression WS5, morphology WS6, visual-score WS7, metabolite WS8, protein/metabolite WS9, rnaseq WS10) are the single place each source's provenance + count oracle is declared -- provenance travels with the run, not the loader.
- Owns the L4 cross-source assertions that no single verifier can make: expression datasets share one platform gene universe; deletion screens are gene-contained in Ohya's morphology set; RNA-seq genes are contained in the S288C SGD reference. Per-family verifiers only expose the gene-set key; the runner joins them.
- Reads the Ohya LMDB from the (possibly read-only) KG-build tree but WRITES reports to the writable `data/torchcell/...` tree -- decoupling verification output from the build user's ownership.
- `main`/`run_all` returns a shell exit code, so the whole abstract's data can gate CI. Uses the checks in [[torchcell.verification.levels]] and the models in [[torchcell.verification.report]].
