---
id: cld90zn87nvgque2osczrk5
title: Test_dataset_registry
desc: ''
updated: 1790412608205
created: 1790412608205
---

## 2026.09.26 - Registry-wide loader contract without building a dataset

Instances come from `cls.__new__(cls)` so `__init__` never runs `process()`. For each of the 52 registered classes: the key equals `__name__`, the class is an `ExperimentDataset` under `torchcell.datasets`, `experiment_class` / `reference_class` are concrete schema types, and `raw_file_names` is a list of unique names (empty only for `GeneEssentialitySgdDataset`, listed strictly in `EMPTY_RAW`). The scerevisiae package `__all__` (40 names) is a subset of the registry.

Finding on the first run: seven loaders return `raw_file_names` as a bare `str` with a `# type: ignore[override]` (`SmfKuzmin2018`, `DmfKuzmin2018`, `TmfKuzmin2018`, `DmiKuzmin2018`, `TmiKuzmin2018`, `SmfCostanzo2016`, `SmfKuzmin2020`); PyG's `raw_paths` accepts a string, so nothing downstream had noticed. A fix (one-element lists, `[0]` at the six `osp.join(self.raw_dir, self.raw_file_names)` call sites in `preprocess_raw`) was written and reverted in the same session: those call sites run only under `--data`, so the CI diff-cover gate scored the change 71% (6 of 21 changed lines uncovered) and blocked it, which is the gate doing its job. The sweep pins the seven in a strict `STR_RAW` set; the fix belongs to a PR that adds a synthetic-TSV `preprocess_raw` test or runs the data-gated Kuzmin tests. Phase 1 of [[plan.test-suite-buildout.2026.09.25]].
