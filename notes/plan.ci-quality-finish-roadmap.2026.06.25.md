---
id: 4wprx6r7sbgh297usbp04bo
title: '25'
desc: ''
updated: 1782433269318
created: 1782433269318
---

# CI / Quality Foundation -- Finish Roadmap

Each `## WS<n>` is a self-contained workstream that becomes ONE GitHub issue linking
back to this note (`(issue: WS<n>)` swaps to `#N` once cut). Goal: take torchcell from
"ruff strict-green" to **fully strict-green CI/CD** (ruff + mypy + pytest all blocking-green),
matching iBioFoundry-AI's quality bar. Resume from this doc after clearing context.

## Context

All work lives on branch **`plan/ci-foundation-ruff-mypy-pytest`** (worktree:
`~/Documents/projects/torchcell.worktrees/plan/ci-foundation-ruff-mypy-pytest`), **6 commits,
NOT merged**. Env: `~/miniconda3/envs/torchcell` (Python 3.13.0, torch 2.9.0+cu128, ruff
0.15.17, mypy 2.1.0). The torchcell CI/quality skills were ported from iBF but the project
config never implemented them; this branch makes them real. See
[[torchcell-testing-mypy-strict-transition]] and [[ruff-up-breaks-pyg-messagepassing]].

## DONE (this branch -- ruff dimension complete)

| Commit     | What                                                                                                                                                                                                         |
|------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `736f69b8` | Foundation: `[tool.ruff]`/`[tool.pytest]`/`[tool.coverage]`, strict `[tool.mypy]` made runnable, pre-commit ruff+mypy, repaired the 3 dead-`src/` CI workflows, deleted `mypy_check.py`, fixed `imoprt` typo |
| `a1bd999e` | `pre-commit install` wired into `scripts/setup-worktree.sh`                                                                                                                                                  |
| `592ac5de` | Fixed latent circular imports (submodule importing its own package `__init__`) in `datamodels/conversion.py`, `datasets/node_embedding_builder.py`                                                           |
| `007c45e2` | ruff `--fix` + format sweep (276 files)                                                                                                                                                                      |
| `d55d8ce4` | Non-D residual cleanup 363->0 (F401 via `__all__`, F841/E722/E741/F811; 13-agent fan-out)                                                                                                                    |
| `928e2a25` | Full docstring coverage: **2,774 docstrings, full iBF D parity** (20-agent fan-out)                                                                                                                          |

**State now:** `ruff check torchcell tests` = "All checks passed" (0 findings) + format-clean
across 329 files. Verification baseline (held throughout): test subset
`pytest tests/torchcell -m "not gpu" --ignore=tests/torchcell/graph/test_graph.py
--ignore=tests/torchcell/data/test_graph_processor_equivalence.py` = **16 failed / 165 passed /
9 skipped / 23 errors**; import smoke = **25** import failures (pre-existing dead/optional-dep
modules). mypy backlog (measured) = **7,666 errors / 256 files**.

## Decisions already made (do NOT relitigate)

1. **ruff** `select=[E,F,I,UP,D]`, `target-version=py313`, line-length 88, google docstrings,
   `force-exclude`; per-file-ignore `UP007`/`UP045` on the 2 MessagePassing modules
   (`stoichiometric_hypergraph_conv.py`, `masked_gin_conv.py`) -- PyG inspects `message()` at
   runtime and dies on PEP 604 unions (issue #10138 "not planned"); see
   [[ruff-up-breaks-pyg-messagepassing]]. Full iBF D parity (all docstrings).
2. **mypy** `strict = true` + per-module `[[tool.mypy.overrides]] ignore_missing_imports=true`
   for untyped third-party libs (NOT a global ignore); `files=["torchcell","tests"]` (DIRS,
   not `**/*.py` globs -- `torchcell/cell.py` vs `torchcell/cell/` collide otherwise);
   `explicit_package_bases=true`, `namespace_packages=true`, `python_version=3.13`; tests/
   relaxed via a `tests.*` override.
3. **CI** ruff/mypy/pytest jobs are **advisory (`continue-on-error`) + diff-scoped** vs
   `origin/main` while backlogs clear. Flip to blocking is WS-CI-FLIP.
4. **Commits** that touch backlog files use `git commit --no-verify` (the live mypy pre-commit
   hook would block on the 7,666 backlog). Hooks ARE installed (in shared `.git`).
5. **Experiments** entirely excluded from gates; only NEW experiments enter CI later (#4).
6. **Library upgrade** (torch 2.9->2.12.1, PyG 2.7->2.8) is a SEPARATE deliberate task, done
   only AFTER tests are green so CI catches breakage. It does NOT fix the MessagePassing issue.

## Existing GitHub issues

- `#4` -- experiments CI scope (new-vs-old; freeze legacy for reproducibility) -> **WS5**.
- `#5` -- test-suite failure backlog -> **WS3**.

---

## WS1. Land the foundation + ruff branch on main (`#6`)

**Goal/Scope.** Review the 6 commits above and merge `plan/ci-foundation-ruff-mypy-pytest` to
main via `/merge-worktree`. This banks the ruff-green foundation so mypy/test work stacks on a
landed base. (Merge triggers semantic-release: `BLD` commits -> patch bump + PyPI publish, which
the user accepted.)

**Key files.** whole branch; `/merge-worktree` skill.

**Dependencies.** None (ready now).

### Checks that must pass

- `ruff check torchcell tests` clean on the rebased branch.
- Test subset still 16/23 baseline after rebase onto main.
- PR/merge green; semantic-release bump as expected.

## WS2. mypy strict cleanup (7,666 -> 0) (`#7`)

**Goal/Scope.** Drive the strict-mypy backlog to zero so the mypy gate can go blocking.
**Scope to LIVE source first** (skip dead/scratch/`*_DEPRECATED`/unimportable modules -- many of
the 256 files are these; type-annotating them is low-value). Fan out by subpackage (balanced
file-bins, same pattern as the ruff/docstring fan-outs), but **heavier verification per agent
because mypy fixes CHANGE code** (annotations, `assert x is not None`, narrowing) and CAN alter
behavior: each agent runs the relevant test subset + import check, not just import smoke.

**Key files.** `torchcell/**` (per-subpackage), `pyproject.toml` `[tool.mypy]` (overrides may
need extending as new untyped imports surface), `tests/`.

**Dependencies.** WS1 (land base first), and respects all Decision 2 config.

### Checks that must pass

- `~/miniconda3/envs/torchcell/bin/python -m mypy` runs to completion with **0 errors** on the
  in-scope (live) set; remaining excluded modules explicitly listed in `[tool.mypy]` `exclude`.
- Test subset stays at-or-better than the 16/23 baseline (no behavioral regression).
- Per-subpackage commits; mypy fixes never add bare `# type: ignore` (always a code or a
  coded-ignore with reason).

## WS3. Test-suite failure backlog (= `#5`)

**Goal/Scope.** Fix the ~39 pre-existing failures the repaired CI surfaced (16 failed + 23
errors in the subset). Known causes: API drift (`SCerevisiaeGenome(data_root=...)` kwarg
removed -> `test_s288c.py`), COO transform shape mismatches
(`test_regression_to_classification_coo.py`), `test_schema.py`/`test_multi_dim_nan_tolerant.py`
setup errors, the `weighted_mse` stub (WS7). Also: give the DATA_ROOT/`/scratch`-dependent
tests (`test_graph.py`, `test_graph_processor_equivalence.py`) real skip-guards so they
self-skip in CI instead of being `--ignore`d.

**Key files.** `tests/torchcell/**`, the source modules whose API the tests encode.

**Dependencies.** WS1. Independent of WS2 (can run in parallel).

### Checks that must pass

- Full `pytest tests/torchcell -m "not gpu"` green (0 failed, 0 errors) -- skip-guards make the
  data/GPU tests self-skip cleanly.
- No test silenced by deletion/xfail without a written reason.

## WS4. Flip CI gates advisory -> blocking (`#8`)

**Goal/Scope.** Once WS2 (mypy) + WS3 (tests) are green, remove `continue-on-error` from the
ruff/mypy/pytest CI jobs so they BLOCK; decide diff-scoped vs whole-tree (whole-tree once the
backlog is zero). Re-enable the mypy pre-commit hook as blocking on the full tree (it already
is, on changed files). Consider adopting iBF's `autofix.yml` (bot runs ruff --fix/format on
PRs) and `ci.yml` consolidation.

**Key files.** `.github/workflows/{style,mypy,test}.yaml`, `.pre-commit-config.yaml`.

**Dependencies.** WS2, WS3.

### Checks that must pass

- CI red on an intentionally-introduced ruff/mypy/test violation (gates actually block).
- Green on a clean PR; no `continue-on-error` left on the quality jobs.

## WS5. Experiments CI scope (= `#4`)

**Goal/Scope.** Decide + implement the mechanism to bring NEW experiments under ruff/mypy/pytest
while keeping OLD experiments excluded and frozen for reproducibility (numbered-threshold vs
allowlist vs per-experiment opt-in). Legacy refactors only behind preserved git tags.

**Key files.** `pyproject.toml` (ruff/mypy scope), CI workflows, `experiments/`.

**Dependencies.** WS4 (gate model settled).

### Checks that must pass

- New-experiment opt-in mechanism documented + enforced; old experiments untouched.

## WS6. Library currency (torch / torch_geometric upgrade) (`#9`)

**Goal/Scope.** Upgrade torch 2.9.0 -> 2.12.1 and torch_geometric 2.7.0 -> 2.8.0 (and aligned
`torch-scatter`/`torch-sparse`, CUDA `cu12x` wheels). SEPARATE, deliberate task done AFTER the
test suite is green (so CI catches what the upgrade breaks). Does NOT resolve the MessagePassing
PEP-604 issue (PyG won't fix it).

**Key files.** `env/requirements*.txt`, `pyproject.toml` (`requires-python` already `>=3.13`).

**Dependencies.** WS3 (green tests as the safety net).

### Checks that must pass

- Env installs cleanly; `import torchcell` + subpackage smoke unchanged or better.
- Full test suite green post-upgrade; any PyG/torch API breakage fixed.

## WS7. Migrate the `weighted_mse` loss (small) (`#10`)

**Goal/Scope.** `SimpleLinearRegressionTask` (`trainers/simple_linear_regression.py`) has a
`weighted_mse` branch currently stubbed `raise NotImplementedError` -- the old
`WeightedMSELoss(mean_value=, penalty=)` moved to `losses.multi_dim_nan_tolerant` with an
incompatible per-dimension `weights=` API. Decide: migrate the branch to the new signature, or
remove `weighted_mse` if dead. Needs the user's intent on the new weighting semantics.

**Key files.** `torchcell/trainers/simple_linear_regression.py`,
`torchcell/losses/multi_dim_nan_tolerant.py`.

**Dependencies.** None; fold into WS3 verification if migrated.

### Checks that must pass

- `weighted_mse` either works with the new `WeightedMSELoss(weights=...)` or is removed; no
  dangling `NotImplementedError` stub.

## Verification (universal)

```bash
~/miniconda3/envs/torchcell/bin/ruff check torchcell tests          # 0 findings
~/miniconda3/envs/torchcell/bin/ruff format --check torchcell tests # clean
~/miniconda3/envs/torchcell/bin/python -m mypy                      # WS2 target: 0 (live scope)
~/miniconda3/envs/torchcell/bin/python -m pytest tests/torchcell -m "not gpu"  # WS3 target: green
```

Commit on the branch with `--no-verify` until WS2 clears the mypy backlog (the live mypy hook
blocks otherwise). Per-workstream commits; verify the test subset never regresses below 16/23
until WS3 fixes it.
