---
id: 48hbg8zozryax4g9m8zp12i
title: Legacy_partition
desc: ''
updated: 1790409442030
created: 1790409442030
---

## 2026.09.26 - Importer-graph partition of torchcell/ into live and legacy

`scripts/legacy_partition.py` computes which `torchcell/` modules a root reaches through imports and which none does. Roots are where code is launched from today: `tests/`, `scripts/`, `database/`, experiments numbered 016 and later (a leading letter is ignored, so `W019-*` counts), the Makefile, `.pre-commit-config.yaml`, the GitHub workflows, and the `[project.scripts]` entry points plus the setuptools version attr in `pyproject.toml`. Frozen experiments 015 and earlier are not roots; they rerun from the `legacy-pre-move-*` tag. Design record: Decision 22 of [[plan.test-suite-buildout.2026.09.25]].

Edges: `ast` imports (with the implicit parent packages an import executes), name-level re-exports (`from torchcell.sga import normalize_plate` edges to `sga/normalize.py` because `sga/__init__.py` imports it from there), string references in root files of every kind (hydra `_target_`, `python -m torchcell.x`, `python torchcell/x.py`), and non-docstring string constants inside package files (a sibling `runner.py` path, a `-m` subprocess target). `pyproject.toml` is not string-scanned: its mypy and ruff carve-outs name modules because they are dead, and a first version that scanned it marked 16 dead modules live.

Categories: `live`, `init-only` (executed only because a package `__init__` imports it wholesale and no root asks for a name it defines; the move drops that import), `legacy` (unreachable), `carve-out` (`scratch/`, `experiments/`). An `__init__.py` follows its package. `--check` is the hard-mode invariant used by `make legacy-check` after PR-0d: exit 1 on any legacy or init-only module outside `torchcell/legacy/`, or any root importing `torchcell.legacy`.

First run on `main` at e1db34e35 (2026.09.26, `--no-git`): live 267 modules (108,792 lines), init-only 15 (2,562), legacy 140 (69,979), carve-out 19 (4,512). The init-only rows are the models and trainers re-exported by `models/__init__.py` and `trainers/__init__.py` that no experiment >= 016 or test names (`dcell` is among them until its Phase 2 test lands), and `viz/fitness.py` reached only through `trainers/__init__`. The user confirms this list before the move PR; the script output, not the plan's nominated list, is the artifact the PR cites.

## 2026.10.06 - package-data category and the CI step

New category `package-data` for a package `__init__.py` no root imports but whose directory ships files matched by `[tool.setuptools.package-data]` (`adapters/conf/*.yaml`, `knowledge_graphs/conf/*.yaml`): their YAML is read by path, so the `__init__` exists to ship data. The table is parsed with `tomllib` and globbed, which is distinct from the string scan of pyproject the script refuses; only a matched non-`.py` file counts, and only an `__init__.py` qualifies. `--check` is otherwise unchanged. CI runs `python scripts/legacy_partition.py --check` as the blocking step "Legacy partition (live tree closed under imports, blocking)" in `.github/workflows/test.yaml`, after the paired-tests step.

After audit 2: `package-data` now also requires the `__init__.py` to be empty or docstring-only (`_is_trivial_init`), so the category cannot hide code. Known limit, documented in the module docstring: an unreached parent `__init__` above a `package-data` package is relabeled `live` by the package rule, and any code it holds is not checked.
