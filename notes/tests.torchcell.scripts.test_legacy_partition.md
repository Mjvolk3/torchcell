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

## 2026.10.06 - The package-data category

`scripts/legacy_partition.py` gains `package-data`: an `__init__.py` that no root reaches but whose directory holds a non-`.py` file matched by a `[tool.setuptools.package-data]` pattern (parsed with `tomllib`, globbed per package key). It is not a `--check` violation and counts as non-legacy for the package rule. Tests on the synthetic tree: `torchcell/conf/__init__.py` plus `a.yaml` with `torchcell = ["py.typed", "conf/*.yaml"]` is `package-data` with via `ships torchcell:conf/*.yaml`, and `--check` still names exactly the three base violations; the same tree without the table is `legacy` and a fourth violation; a `loader.py` beside the YAML stays `legacy` (and a violation) while the `__init__` keeps `package-data`; a pattern matching only `.py` files marks nothing; a dotted key (`torchcell.kg.conf`) globs from its own directory and makes the parent package live, a prefix-sharing key (`torchcellx`) is skipped; the first matching pattern is the one reported; the summary line ends `package-data 1 (1 lines)`. The two existing `main` assertions now include `package-data 0`.

After audit 2: a `package-data` `__init__` must be empty or docstring-only; one with code (`X = 1`, or a docstring plus `import os`) is `legacy` when unreached and a `--check` violation (parametrized test, both sides). Added: a root importing the package keeps it `live` via that root; a matched subpackage directory (`conf/sub`) marks nothing; `torchcell = ["conf/*.yaml"]` matches only the top-level `torchcell/conf`, not the nested `torchcell/kg/conf`; full `--check` output and the exact summary line (`live 5 (7 lines), init-only 1 (1 lines), legacy 4 (3 lines), carve-out 1 (1 lines), package-data 1 (1 lines)`).
