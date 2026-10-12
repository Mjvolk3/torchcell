---
id: km22wb1txt7hk1dz2ln621b
title: Verification_overlay
desc: ''
updated: 1791817915686
created: 1791817915686
---

## 2026.10.12 - Reports written outside the shared dev tree

Builds a `DATA_ROOT` mirror of symlinks in which each `data/torchcell/<store>/preprocess` is a real directory of links to the store's own files, minus every `verification_report*`. A verification rerun under unlanded code then writes its reports into the overlay, never into the shared tree, where `main`'s `LevelResult` (`extra="forbid"`) would refuse the new `finding_class` key. Linking rather than copying keeps `build_manifest.json` live, so the pre-build sweep still sees a store rebuilt during its verification. Verifiers write nothing but their reports (`runners._write_report`, the loaders' `verify*` entry points); the loaders' other `_write_*` helpers run at build time.
