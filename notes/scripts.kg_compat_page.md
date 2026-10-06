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

## 2026.10.06 - Pairs table, `incompatible` replaces `partial`, CI check

The page's lead table is now **Pairs**: one row per release with its paired package (the snapshot's `torchcell_tag`, stamped at a build from a tagged commit or set afterwards by `releases retag`) and every tag that reads it. A paired tag whose verdict is not `compatible` is a broken pair and `build_page` raises, so `--check` fails in CI rather than publishing a page that contradicts the client. The matrix keeps the per-tag evidence but the word changed: `partial (...)` is now `incompatible (<n> of <m> datasets drift: ...)`, `incompatible (all <m> datasets drift)`, or `incompatible (<n> of <m> datasets unverified)` for a release whose snapshot recorded no closures, because the client (`releases.require_paired`) refuses the whole release at connect and the named datasets are evidence, not a usable subset. `package_tags` keeps only `vX.Y.Z` tags; `legacy-pre-move-2026.10` and `abandoned/...` are skipped.

`--check` now runs in CI, in the `query-drift` job of `.github/workflows/docs.yaml`, whose checkout gained `fetch-depth: 0` and `fetch-tags: true` so `git show <tag>:...` works there.

Measured 2026-10-06 after `releases retag` paired the three releases (`2026.09.21-ab6d8c5d` with `v1.2.1`, `2026.10.02-833970cd` with `v1.6.1`, `2026.10.06-4b293d34` with `v1.6.2`, the `DB(kg)` release cut that day): `v1.6.2` is the only tag that reads KG 3.0; `v1.2.1` through `v1.6.1` read KG 1.2 and 2.0; `v1.2.0` reads nothing.
