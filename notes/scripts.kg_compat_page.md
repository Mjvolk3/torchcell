---
id: y1nf0lts7kdbmw79m4go5s1
title: Kg_compat_page
desc: ''
updated: 1790716453916
created: 1790716453916
---

## 2026.09.29 - The generated compatibility page

Plan: [[plan.data-release-program.2026.09.29]], Decisions 1 and 2 (T4, issue #467).
Writes `docs/source/database/compatibility.md` (MyST) from the committed snapshots
([[torchcell.knowledge_graphs.release_snapshot]]) and the package tags from `v1.2.0` on
(`git tag --sort=v:refname`); linked from the README as
<https://mjvolk3.github.io/torchcell/database/compatibility.html>. The toctree entry in
`docs/source/database/index.md` belongs to the docs branch and is added after both land.

- Verdict per (tag, release): the schema surface AT THE TAG is read with `git show
  <tag>:torchcell/datamodels/schema.py` and `pydant.py` into
  `schema_deps.load_surface_from_sources` (no checkout, no temp files) and compared with
  the release's closures by `releases.closure_compatibility`. `compatible` when every
  served dataset's fingerprints match; `partial (<n> datasets drifted: A, B)` naming
  them, or `partial (all <n> datasets drifted)` when nothing survives; `unknown` when a
  surface module is absent at the tag.
- The page: a generated-file header, the releases table (`KG release | KG version |
  built | commit | package version at build | tag`), the matrix (`package tag` by one
  column per release), and the three-sentence rule. `--check` exits 1 when the file on
  disk differs; no timestamps, so the output is byte-stable.
- Measured 2026-09-29 for release `2026.09.21-ab6d8c5d`: `v1.2.0` is `partial (all 51
  datasets drifted)`, `v1.2.1` is `compatible`.
- CI caveat: `actions/checkout` with the default `fetch-depth: 1` has no tags, so a CI
  `--check` step needs `fetch-depth: 0`; no CI step is wired yet.
- Tests: `tests/scripts/test_kg_compat_page.py` builds a throwaway tagged git repo with
  two surfaces and two snapshots and pins the whole page text and the `--check` codes.
  `pytest tests/torchcell` (the CI gate) does not collect `tests/scripts/`; run it by
  path.
