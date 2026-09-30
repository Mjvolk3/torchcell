---
id: nlmb3136xp4hx0ilkl9tzl4
title: Test_schema_deps
desc: ''
updated: 1790773233105
created: 1790773233105
---

## 2026.09.30 - Phase 16: the canonical text, every Field folding, exact closures

Thirteen to twenty-nine tests, 78 to 100 percent. The exact `canonical()` text and its SHA-256 (`ed8d43cc...f641`); every `Field` folding form; assigns, validators and Config; description edits do not change the fingerprint; the exact reference graph and closures (the old subset checks are now exact).

Findings: base-class order does not change the fingerprint although the MRO differs (line 118); `Field(..., le=5)` and `Field(default=..., le=5)` get different fingerprints (163-178); a duplicate class name silently replaces the first (307-308).
