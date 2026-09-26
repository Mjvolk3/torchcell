---
id: 3wtfnzesu26otuqh5dzukbk
title: Check_paired_tests
desc: ''
updated: 1790409449457
created: 1790409449457
---

## 2026.09.26 - Paired-test gate for new torchcell modules

`scripts/check_paired_tests.py` requires `tests/torchcell/<same dir>/test_<name>.py` for every Python file added under `torchcell/` since the merge base with `origin/main` (`--strict`: added or modified, the hard mode that switches on in the legacy-move PR). `__init__.py` and `__main__.py` are exempt. The diff is index-vs-merge-base (`git diff --cached`), so the pre-commit hook sees a staged addition and CI sees the branch's commits.

Exceptions come from ONE place, `[tool.torchcell.test_exceptions]` in `pyproject.toml`: `paths` (a `/`-terminated prefix, a `*` glob, or an exact path) and `pairs` (source -> the test file that covers it when the mirrored basename would collide under pytest's prepend import mode, Gotcha 4 of [[plan.test-suite-buildout.2026.09.25]]). The table was seeded verbatim from the mypy and ruff carve-outs; after PR-0d it collapses to `torchcell/legacy/`, `torchcell/scratch/`, `torchcell/experiments/`. Wired to `make paired-tests`, the `paired-tests` pre-commit hook, and a blocking step in `.github/workflows/test.yaml` (which needs `fetch-depth: 0`, the same reason `mypy.yaml` has it).
