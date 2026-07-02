---
id: ju9dqy2hf8cm70nl70uiu6o
title: '01'
desc: ''
updated: 1782942966568
created: 1782942966568
---

# Docs Build Modernization + Session Handoff (2026.07.01)

Handoff for a fresh session (Fable 5) to do the **proper docs-build modernization**.
Below: (A) what already landed this session, (B) THE task, (C) repo/env state + landmines.
Related: [[plan.ci-quality-finish-roadmap.2026.06.25]], [[plan.pytest-ci-blocking.2026.07.01]],
[[ruff-up-breaks-pyg-messagepassing]], [[semantic-release-pushes-bump-commit]].

> **STATUS 2026.07.01 - COMPLETED IN-SESSION (not handed off).** Section B was executed
> and landed on `main`: Sphinx 8 / py3.13, the `torchcell_sphinx_theme` fork's
> `sphinx==5.1.1` pin relaxed to `>=7`, local `pip install -e .`, the `sequence`
> `data_classes`/`helper_functions` doc-lists restored, and the gh-pages deploy fixed via
> the admin PAT. **Docs build + deploy are green and published to GitHub Pages.** `#16`
> (pytest blocking) is also merged and enforced as a required check. The sections below
> are kept as the execution record.

## A. Already done this session (context — do NOT redo)

### WS6 -- torch/PyG upgrade (`#9`): DONE + RELEASED `v1.2.0`

- torch 2.9.0 -> **2.11.0+cu128**, torch_geometric 2.7.0 -> **2.8.0**, torch_scatter
  **2.1.2+pt211cu128** (prebuilt wheel), `torch_sparse` dropped (was dead code).
- Chose 2.11.0 (NOT 2.12.x): cu128 wheel index caps at 2.11.0; 2.12.x needs a CUDA-toolkit
  migration (cu126/cu130/cu132). Prebuilt scatter wheel exists so no `--no-build-isolation`.
- Landed on main (`e96a78cc`); semantic-release cut **v1.2.0** (bump commit `4309a482`, admin
  PAT push validated for real). Shared conda env `~/miniconda3/envs/torchcell` reinstalled
  **in-place** (a worktree isolates code, NOT the env). See the roadmap note's WS6 EXECUTION.

### `#16` -- make pytest CI blocking: DONE, **PR #18 OPEN + GREEN, awaiting merge**

- Branch `plan/pytest-ci-blocking` (worktree exists). Plan: [[plan.pytest-ci-blocking.2026.07.01]].
- Hardened 4 CI-fragile tests (declarative guards, no fallbacks): `test_s288c.py` module-level
  DATA_ROOT skip (was in-fixture assert -> 9 CI errors); `test_masked_vs_filtered_speed`
  -> `@pytest.mark.gpu`; `test_pyg_concatenated_batch` DATA_ROOT `skipif`; `test_cleanup_lock_files`
  rewritten (filelock >=3.21.0 deletes lock-on-release on Unix). CI: CPU torch + prebuilt +cpu
  scatter wheel, removed job `continue-on-error`, Codecov step best-effort, cache key hashes 3
  req files. `filelock>=3.20.1`, `pytest>=9.0,<10`, `coverage>=7,<8`.
- **All 3 real checks GREEN**: ruff, mypy, **pytest-coverage PASSED (19m)**. Only `build`(docs) red
  (pre-existing, unrelated). Local `not gpu` suite: 217 passed / 3 skipped / 3 deselected / 0 failed.
- **PENDING (2 steps, NOT done):**
  1. **Merge #18** -> `/merge-worktree plan/pytest-ci-blocking` (rebase+ff, closes PR, cleans up).
  2. **After merge + one green `pytest-coverage` on main**, add the required check:
     `gh api -X PUT repos/Mjvolk3/torchcell/branches/main/protection/required_status_checks`
     with contexts `["ruff","mypy-check","pytest-coverage"]` (use the JOB KEY `pytest-coverage`,
     NOT the workflow name, or it hangs perpetually-pending and blocks ALL PRs). Keep
     `strict:false` + `enforce_admins:false` (semantic-release PAT bump must bypass).

## B. THE TASK -- docs build proper modernization (branch `fix/docs-build`, PR #19)

**Continue on `fix/docs-build`** (worktree `~/Documents/projects/torchcell.worktrees/fix/docs-build`,
PR #19). Its config fixes are correct and needed -- BUILD ON them, don't restart.

### Already fixed in PR #19 (`14e42ba1`) -- the workflow *config*

`.github/workflows/docs.yaml`: py3.13 (was 3.11.7); `checkout@v4`/`setup-python@v5` (was v2);
build `docs/source` (was `docs` -- `conf.py` lives in `docs/source/`); install CPU torch
(`torch==2.11.0 --index-url .../cpu`) + prebuilt `+cpu` torch-scatter wheel before
`docs/requirements.txt`. `docs/requirements.txt`: dropped `git+.../pytorch_scatter` source line.
Result: the build now reaches Sphinx (py3.13, right source dir) but STILL FAILS.

### The remaining blocker (diagnosed -- do NOT re-discover)

1. **Sphinx is pinned to 5.1.1** (2022) by the theme stack. pip grabs Sphinx 9.1.0 then
   backtracks to 5.1.1 to satisfy `pytorch_sphinx_theme 0.0.19` and/or `torchcell_sphinx_theme
   0.1.1`. Sphinx 5.1.1 imports `imghdr` (stdlib, REMOVED in py3.13 / PEP 594) ->
   `Could not import extension sphinx.builders.epub3 (No module named 'imghdr')`. Pinning
   Sphinx UP conflicts with the theme.
2. **`torchcell` resolves to PyPI `0.2.6`** (ancient; 1.x isn't on PyPI). `docs/requirements.txt`
   has bare `torchcell` -> autodoc would document 3-yr-old code. Should install the LOCAL
   checkout (`pip install -e .`), not PyPI.
3. **Ancient themes**: `pytorch_sphinx_theme 0.0.19`, `torchcell_sphinx_theme 0.1.1` (git,
   `github.com/Mjvolk3/torchcell_sphinx_theme`). ACTIVE theme (`docs/source/conf.py`
   `html_theme = "torchcell_sphinx_theme"`); `pytorch_sphinx_theme` + `sphinx_rtd_theme` may be
   vestigial. conf.py: Sphinx 5.1.1 project, extensions autodoc + nbsphinx,
   `suppress_warnings=["autodoc.import_object"]`, build runs WITHOUT `-W` (warnings non-fatal).

### Modernization approach (USER CHOSE THIS over the band-aid)

1. **Upgrade Sphinx to py3.13-compatible** (>=7.4, ideally 8.x). This requires the ACTIVE theme
   `torchcell_sphinx_theme` to tolerate modern Sphinx -- likely means updating the EXTERNAL repo
   `github.com/Mjvolk3/torchcell_sphinx_theme` (a pyg-sphinx-theme fork; inspect its Sphinx pin in
   `setup.py`/`pyproject`). If `pytorch_sphinx_theme` (0.0.19) is the forcer AND is unused, just
   REMOVE it from `docs/requirements.txt`. Pin Sphinx explicitly (e.g. `sphinx>=8,<9`) once the
   theme allows it.
2. **Document local code**: switch to `pip install -e .` in docs.yaml; remove `torchcell` from
   `docs/requirements.txt`.
3. **Verify autodoc** imports the full torchcell (needs torch 2.11 + prebuilt scatter + all deps --
   already handled in docs.yaml). Expect possible autodoc/nbsphinx follow-ons; iterate on CI.
4. **Rejected band-aid** (for reference, NOT chosen): `standard-imghdr` (v3.13.0 on PyPI) +
   `pip install -e .` -- keeps ancient Sphinx 5.1.1; user wants proper modernization.

### Verification

CI-ONLY: Sphinx is NOT in the local `torchcell` env, so the docs build can only be exercised on
the runner (each ~4 min). The gh-pages deploy step is `main`-only (`if: github.ref == ...`), so PR
runs exercise install + `sphinx-build` only. `readthedocs` job just POSTs a webhook (curl exits 0
regardless) -> not a real gate.

## C. Repo / env state + landmines

- **main** at `4309a482` (v1.2.0). Env `~/miniconda3/envs/torchcell` (py3.13, torch 2.11.0+cu128,
  pyg 2.8.0). Local python: `~/miniconda3/envs/torchcell/bin/python` (CLAUDE.local.md: user runs
  py files/tests; verification pytest/ruff/mypy is fine to run).
- **Open PRs**: #18 (`plan/pytest-ci-blocking`, `#16`, GREEN, awaiting merge), #19
  (`fix/docs-build`, docs -- THIS task).
- **Required checks on main**: only `ruff` + `mypy-check` (job keys), `strict:false`,
  `enforce_admins:false`.
- **Landmines**: torch-scatter MUST be a prebuilt wheel not a source build (CPU CI:
  `-f https://data.pyg.org/whl/torch-2.11.0+cpu.html`); markdownlint MD024 blocks NOTE commits
  (roadmap note has dup headings) -> `git commit --no-verify` for notes ONLY, never bundling code;
  semantic-release pushes a version-bump commit to main after each merge -> `git fetch` + rebase
  before the next merge; commit prefix for non-releasing CI/test work = `ci:`/`test:` (scipy parser
  inert), releasing tags are API/DEP/DEV/ENH/REV/FEAT/BLD/BUG/MAINT.
- **Stale orphaned dir** (NOT a git worktree, safe to `rm -rf`, user hasn't confirmed):
  `~/Documents/projects/torchcell.worktrees/plan/ci-foundation-ruff-mypy-pytest`.
- Other worktrees (unrelated): `feat/literature-zotero-ocr`, `fitness-interaction-n_samples_2`,
  `plan/schematization-ingestion-roadmap`, `write/supported-datasets-table-revitalized`.
