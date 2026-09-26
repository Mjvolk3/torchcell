---
id: ygm0tvf6arrzb6l8a7mx1hb
title: Test_check_paired_tests
desc: ''
updated: 1790416778850
created: 1790416778850
---

## 2026.09.26 - The paired-test gate against a temporary git repository

`is_excepted` is tabled (prefix with trailing slash, glob on the path or the basename, exact path), `expected_test` mirrors the directory, `load_exceptions` reads `paths` and `pairs` or returns empty. The git piece runs the real `git diff --cached <merge-base>` in `tmp_path`: `main` holds one module and its test; `feature` adds `new.py`, `scratch/junk.py` (excepted directory), `losses/dcell.py` (a `pairs` entry), `__init__.py` and `__main__.py` (exempt basenames) and modifies `old.py`. Default mode lists the three additions sorted; `--strict` adds the modification; a staged new module counts and an unstaged one does not (the pre-commit path). `main()` prints one line per missing pair and the summary, exit 1; after the two test files exist (untracked is enough) it passes in both modes with the exact summary lines. Git is hermetic through `HOME`, `GIT_CONFIG_GLOBAL=/dev/null`, the author and committer variables, and removal of the inherited `GIT_*` variables. Phase 4 of [[plan.test-suite-buildout.2026.09.25]].
