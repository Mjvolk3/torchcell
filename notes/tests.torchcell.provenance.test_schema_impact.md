---
id: iq539n5a3y53681g65i4w5i
title: Test_schema_impact
desc: ''
updated: 1790756792862
created: 1790756792862
---

## 2026.09.30 - Phase 11: every classification and the CLI on a throwaway repo

Sixteen to forty-one tests, 66 to 100 percent. The existing classification tests now assert the exact `(kind, status, reasons)` tuples; one diff hits every reason string, pinning all 15 and their order; `map_impacts` ordering (breaking first, then path, the max kind wins); `_dataset_classes` and `_loader_paths`; end to end on a throwaway git repository under `tmp_path` (its own `.git`, hooks and global config off): `_git_show`, the exact report object and text, `main` exit codes (breaking 1, ack env, stale-only 0, `--strict` 1, ack overrides strict, a breaking symbol no loader reaches exits 0 with "Impacted datasets: none").

Findings: reordering base classes is invisible because `canonical()` sorts them (`schema_deps.py` line 118); tightening a `Field` constraint is reported as a stale "default changed" (line 152) although it can invalidate stored records; `TORCHCELL_SCHEMA_ACK=0` acknowledges, the check is truthiness (line 372); `@register_dataset(...)` in call form is not recognized (lines 227-228); a new validator is classified stale.
