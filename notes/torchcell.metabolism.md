---
id: eujmb24rpu57qxp47zqzsw0
title: Metabolism
desc: ''
updated: 1790409486274
created: 1790409486274
---

## 2026.09.26 - Package marker

`torchcell/metabolism/__init__.py` was added in Phase 0a of [[plan.test-suite-buildout.2026.09.25]] (Decision 11). coverage.py with `source = ["torchcell"]` reports un-executed files only in directories that have an `__init__.py`, and `pkgutil.walk_packages` walks the same set, so the live, tested `flux_layer`, `yeast_GEM`, `pathway` and `media` modules were invisible to both. The marker re-exports nothing: `yeast_GEM` imports cobra and downloads on first use, so callers import the submodule they need. This is one of the three denominator changes the campaign allows (Decision 1).
