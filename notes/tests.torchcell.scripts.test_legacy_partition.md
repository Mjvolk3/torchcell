---
id: rl17b2haapefoqjq3fh2y46
title: Test_legacy_partition
desc: ''
updated: 1790416786293
created: 1790416786293
---

## 2026.09.26 - The importer partition on an eleven-module repository

`legacy_partition.REPO` is monkeypatched to a tree under `tmp_path` built so every category and every root-edge kind appears once: a relative import (`from .helper import`), a package `__init__` re-export that makes `pkg/init_only.py` init-only, a path-style reference in `scripts/run.sh`, a `[project.scripts]` entry point, the setuptools version attribute, an experiment below the live threshold (ignored) and a letter-prefixed one above it (a root), a docstring mention (not a reference), a string constant in `tests/torchcell/test_import_all.py` (exempt from the string scan; the same literal in another test file makes the module live), a carve-out module and an already-legacy package. `EXPECTED` writes out every row: category, the root or `__init__` that reached it, live-critical from the production roots only, the already-legacy flag. Also covered: `root_files` order and `production_roots`, the root and module edge sets, `check` naming the three violations (and a root that imports `torchcell.legacy`, which makes the legacy module live but the root a violation), `check` passing once the dead pair lives under `legacy/` and the init-only module is imported directly, `main --no-git --table --output` (table rows, JSON counts and lines, the summary line), the `torchcell.cell#file` collision warning, `string_references`, and `_git_last_commit` returning `untracked` outside a repository.

Pinned: a package whose only member is init-only is itself relabeled live ("package has live members"); seeding from the roots precedes the BFS, so a module named by both a test root and a production root reports the production root only if it comes first in root order (here `scripts/run.sh` for `helper`). Phase 4 of [[plan.test-suite-buildout.2026.09.25]].
