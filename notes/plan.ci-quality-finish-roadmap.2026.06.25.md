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

**State now (2026.06.25 baseline — SUPERSEDED by the 2026.06.29 section: mypy=0 / tests green / landed on main `170890c3`):** `ruff check torchcell tests` = "All checks passed" (0 findings) + format-clean
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

## WS1. Land the foundation + ruff branch on main (`#6`) — ✅ DONE (in main)

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

## WS2. mypy strict cleanup (7,666 -> 0) (`#7`) — ✅ DONE (landed `170890c3`; mypy 0 / 294 files)

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

## WS3. Test-suite failure backlog (= `#5`) — ✅ DONE (suite green 218/3/0)

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

## WS4. Flip CI gates advisory -> blocking (`#8`) — ✅ DONE (ruff+mypy blocking & required; pytest -> #16, now blocking)

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

## WS5. Experiments CI scope (= `#4`) — ✅ DONE (numbered threshold >=016)

**Goal/Scope.** Decide + implement the mechanism to bring NEW experiments under ruff/mypy/pytest
while keeping OLD experiments excluded and frozen for reproducibility (numbered-threshold vs
allowlist vs per-experiment opt-in). Legacy refactors only behind preserved git tags.

**Key files.** `pyproject.toml` (ruff/mypy scope), CI workflows, `experiments/`.

**Dependencies.** WS4 (gate model settled).

### Checks that must pass

- New-experiment opt-in mechanism documented + enforced; old experiments untouched.

## WS6. Library currency (torch / torch_geometric upgrade) (`#9`) — ✅ DONE (torch 2.11.0 / PyG 2.8.0; released v1.2.0)

**Goal/Scope.** Upgrade torch 2.9.0 -> 2.12.1 and torch_geometric 2.7.0 -> 2.8.0 (and aligned
`torch-scatter`/`torch-sparse`, CUDA `cu12x` wheels). SEPARATE, deliberate task done AFTER the
test suite is green (so CI catches what the upgrade breaks). Does NOT resolve the MessagePassing
PEP-604 issue (PyG won't fix it).

**Key files.** `env/requirements*.txt`, `pyproject.toml` (`requires-python` already `>=3.13`).

**Dependencies.** WS3 (green tests as the safety net).

### Checks that must pass

- Env installs cleanly; `import torchcell` + subpackage smoke unchanged or better.
- Full test suite green post-upgrade; any PyG/torch API breakage fixed.

## WS7. Migrate the `weighted_mse` loss (small) (`#10`) — ✅ DONE (dead stub removed)

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

## 2026.06.28 - Status Refresh (mid-WS2; parallel WS2-Wave3 + WS3 launch)

**Branch topology changed since this roadmap was written.** The original
`plan/ci-foundation-ruff-mypy-pytest` branch/worktree is gone (a leftover orphan
directory remains on disk but is no longer a git worktree). WS1's foundation work —
ruff config, full docstring coverage, circular-import fixes, the repaired CI
workflows — all landed in `main` (verified: foundation commit `736f69b8` is an
ancestor of `main`, and `main` HEAD `2d1aece6` is "fully ruff-green"). Active work
moved to a fresh branch **`plan/ci-ws2-mypy-semantic`** (worktree of the same name),
rebased cleanly onto current `main`: it is `main` + 15 WS2 commits with an empty
`ws2..main`, so it fast-forward-merges when done — modulo the semantic-release
bump-commit dance (fetch + rebase `main` before the next merge; see
[[semantic-release-pushes-bump-commit]]).

### Revised status

| WS | Was (2026.06.25) | Now (2026.06.28) |
|----|------------------|------------------|
| WS1 (`#6`) land foundation + ruff | 6 commits, not merged | **DONE** — foundation is in `main`; `main` is ruff-green |
| WS2 (`#7`) mypy strict 7,666 → 0 | 7,666 / 256 files | **2,131 / 84 files (~72% cleared).** 13 subpackages typed to 0; dead/legacy modules excluded from the gate (`c322abb0`; mypy now `checked 295 source files`) |
| WS3 (`#5`) test backlog | ~16 failed / 23 errors | **DONE** — subset re-measured green (215 passed / 0 failed / 0 errors); the 2 `/scratch`-dependent tests already carry `skipif` data-guards. Final close-out = post-WS2 full `not gpu` run |
| WS4–WS7 | — | unchanged (gated on WS2 + WS3) |

### What WS2 has left (= "Wave 3")

The entire remaining 2,131 mypy errors live in the deliberately-deferred,
behavior-critical layer: `models/` (~1,076 / 43 files) + `trainers/` (~971 / 25
files). Every other subpackage (graph, sequence, data, datasets, nn, losses,
metrics, transforms, adapters, datamodels, yeastmine, metabolism, top-level) is at
0. Dominant trainer failure: `self.hparams.<field>` → `"MutableMapping" has no
attribute` (Lightning types `hparams` as a plain mapping); fix via typed local /
`cast`, behavior-preserving — never by changing how config is consumed at runtime.

### In flight this session (2026.06.28)

Parallel fan-out, **Opus 4.8** agents, shared `ci-ws2-mypy-semantic` worktree, under
the proven protocol — every agent is git-forbidden and owns a disjoint file set; the
orchestrator does all commits and runs the final whole-project mypy/pytest gate (see
[[shared-worktree-concurrent-agent-git-hazard]]):

- **WS2 Wave 3** — 4 trainer agents (T1–T4) launched first; then 6 model agents (M1–M6).
- **`models/equivariant_cell_graph_transformer.py`** (the 010/011 model, frozen at
  tag `v1.1.2`; see [[cell-data-edge-name-convention]]) is carved into its OWN agent
  and its diff is **held for user review** before commit.
- **WS3** needs NO fan-out — re-measured green (the note's 39-failure baseline was
  stale; the n_replicates refactor on `main` + WS2 typing cleared it), and the 2
  `/scratch` tests already self-skip via `skipif`. `weighted_mse` in
  `simple_linear_regression.py` stays as-is (WS7 `#10`, not WS3).

Commit gate (orchestrator, per workstream): whole-project
`~/miniconda3/envs/torchcell/bin/python -m mypy` == 0 on changed scope, and the
`not gpu` test subset at-or-better than the 16/23 baseline.

## 2026.06.29 - WS2 mypy backlog CLEARED to 0; WS3 GREEN (WS4 unblocked)

WS2 Wave 3 + residual cleanup landed. **Whole-project `mypy` = 0**
(`Success: no issues found in 294 source files`); WS2 backlog **7,666 -> 0**.
WS3: full `not gpu` suite green (**218 passed / 3 skipped / 0 failed**), run
*without* `--ignore` (the 2 `/scratch` tests self-skip/run via their `skipif`
guards).

### How

- **11 Opus 4.8 agents** typed the deferred behavior-critical layer: 4 trainer
  bins (~971 err), 6 model bins (~1076 err), and the 010/011
  `equivariant_cell_graph_transformer` carved out + hand-reviewed (19 err). All
  type-only / behavior-preserving; shared-worktree protocol (every agent
  git-forbidden + disjoint files, orchestrator commits) per
  [[shared-worktree-concurrent-agent-git-hazard]].
- **2 Opus cleanup agents** cleared the 45 the full gate surfaced beyond the
  models+trainers partition (a scoping gap I'd mis-counted as fuzz, NOT a
  regression): 27 test + 7 source errors. `torchcell/experiments/` excluded from
  the gate per Decision 5.
- **112 coded `# type: ignore[...]`** total, all reasoned (zero bare). Bulk are
  pre-existing demo-block API drift in `if __name__=="__main__"` smoke-blocks and
  untyped third-party calls (transformers/Bio) -- preserved, not silently "fixed."

### Real bug strict mypy caught (argues for keeping tests strict)

`tests/torchcell/sequence/test_data.py` had 3 `assert start, end == (...)` lines --
the comma made Python treat the tuple comparison as the assert *message*, so the
window values were **never compared** (vacuous pass under 218 green tests). mypy's
`comparison-overlap` surfaced it; fixed to `assert (start, end) == (...)`, expected
tuples verified against `sequence/data.py`.

### Commits (branch `plan/ci-ws2-mypy-semantic`)

`1efdb8be` trainers · `4b06384c` models · `6d04679f` equivariant · + source/config
cleanup · + tests · + this note. All `STY:`/`test:` -> non-releasing under
semantic-release.

### Now unblocked / remaining

- **WS4** (flip CI gates advisory -> blocking) is unblocked: mypy=0 + tests green.
- **Optional polish** (NOT done; await greenlight): (A) sweep dormant demo-block API
  drift; (C) standardize the trainer hparams idiom on T1's `Protocol` pattern (the 4
  trainer agents each invented a different variant); (D) demote 3 dead modules
  (`graph_attention`, `graph_convolution`, `trainers/utils`) from coded-ignores to
  clean `exclude` entries.
- **Merge**: branch is `main` + clean FF stack; merging to main is a separate step
  (mind the semantic-release bump dance, [[semantic-release-pushes-bump-commit]]).

## 2026.06.29 - WS4 in progress: ruff gate DONE (landed), mypy/pytest remaining

WS4 turned out to be a multi-front CI-engineering task, not a flag-flip. Two
discoveries reframed it: the 3 quality workflows were **`disabled_manually`** at the
repo level (so they never ran, advisory or not — `gh workflow list --all`), and they
had **never passed in CI**, which had masked real env/version issues.

**Decision (made after the failures surfaced):** whole-tree blocking **ruff** +
**diff-scoped** blocking **mypy** + fix the **pytest** install. Whole-tree mypy was
abandoned: its CI findings are *environment-driven* (CI's installed packages differ
from the dev conda env; mypy results depend on them), so reproducing local `0` would
require a full dependency lock.

### DONE — banked on main (`b4a01898`)

- **ruff gate**: newest ruff **0.15.20**, 28 findings fixed, pinned 0.15.20 across
  local + CI (`env/requirements_style.txt`) + pre-commit. The ruff gate passed green
  in CI on PR #15. Landed to main **decoupled** from the gate-flip so a future
  diff-scoped mypy run stays clean (a 46-file sweep in the PR diff is what broke it).

### REMAINING — WS4 branch `plan/ci-ws4-blocking-gates` (PR #15; workflows RE-DISABLED)

1. **mypy gate** (diff-scoped + `--follow-imports=silent`; edits already on the
   branch): with the ruff sweep now on main, **rebase the branch** → its diff becomes
   config-only → diff-scoped mypy sees "no changed torchcell files" → passes. Two real
   fixes still worth doing: (a) 8 genuine `self.env: lmdb.Environment = None` bugs
   (`datasets/base_cell.py` 99/139, `dataset_readers/reader.py` 25/79,
   `datasets/experiment.py` 67/98) → `lmdb.Environment | None`; (b) filter
   `torchcell/experiments/` OUT of the diff-scoped `$FILES` — **explicit `mypy $FILES`
   bypasses the pyproject `exclude` regex** (exclude only filters *discovery*).
2. **pytest gate**: torch-scatter build fixed with `--no-build-isolation` (edit on the
   branch). Remaining = **4 collection-import errors** (`test_cell_data`,
   `test_graph_processor_equivalence`, `test_hetero_cell_nsa`, `test_hetero_nsa`) —
   debug why those modules fail to import in CI.
3. **Re-enable** the 3 workflows (`gh workflow enable 66135990 66135988 66433423`)
   only once all 3 pass on the PR, then merge via rebase+ff and close PR #15.

### Orthogonal (its own issue, NOT WS4)

- **Build and Deploy Docs** (active) fails on every main push: `docs/` has no
  `conf.py` (`sphinx-build` config error). Pre-existing; deserves its own issue.

### Gotchas logged (so resume is fast)

- `disabled_manually` workflows ignore their triggers — check `gh workflow list --all`
  before debugging `on:` syntax.
- Explicit-file `mypy $FILES` bypasses the `exclude` regex (only discovery is filtered).
- ruff `I001` reflow detaches `# type: ignore` from its import line → silently breaks
  mypy; re-run mypy as an acceptance test after any ruff autofix.
- CI mypy env-divergence (redundant-cast etc.) is unwinnable without a dependency lock
  → the reason mypy is diff-scoped, not whole-tree.

## 2026.06.29 - WS4 LANDED (decouple): ruff+mypy blocking on main; pytest → #16

WS4 shipped as a **decouple** (user decision): the two gates that are genuinely green
now BLOCK; pytest stays advisory pending a focused hardening follow-up. Landed on
`main` via rebase + ff-only — commits `3d68e57b` (ci) / `ce1f8613` (fix) / `135fd609`
(test); PR #15 auto-closed. **No release**: the repo uses the **scipy** commit parser
(releasing tags `API`/`DEP`/`DEV`/`ENH`/`REV`/`FEAT`/`BLD`/`BUG`/`MAINT`), so the
`ci:`/`fix:`/`test:` prefixes are inert — latest release stayed `v1.1.4`.

- **ruff** — BLOCKING, whole-tree (`ruff check` + `format --check` on torchcell+tests).
  Green in CI.
- **mypy** — BLOCKING, diff-scoped + `--follow-imports=silent` on changed `torchcell`
  files; `torchcell/experiments/` filtered out of the `$FILES` list (explicit
  `mypy FILES` BYPASSES the pyproject `exclude`). Green in CI. Whole-tree mypy stays
  abandoned (CI packages ≠ dev conda env, e.g. `lmdb` typed in CI but `Any` locally).
- **pytest** — ADVISORY (`continue-on-error` in test.yaml). First-ever full CI run
  surfaced **167 passed / 22 skipped / 3 failed / 9 errors** (WS3's green was only ever
  validated locally). Fixed in this PR: the optional `calm` import made
  `torchcell.datasets` unimportable (deferred to TYPE_CHECKING + a runtime import in
  `initialize_model`); and 4 modules raised on ambient `DATA_ROOT` at collection
  (module-level `pytest.skip(allow_module_level=True)` guards). Remaining CI-fragile
  tests → **follow-up issue #16**: `test_s288c` DATA_ROOT guard, a flaky wall-clock
  micro-benchmark (`test_masked_vs_filtered_speed`), a filelock-version-fragile test
  (`test_cleanup_lock_files`). Removing `continue-on-error` is #16's final step.

**Not enforced yet:** `main` has NO branch protection, so red ruff/mypy checks are
*visible* but don't *prevent* a merge. True enforcement needs a required-status-checks
rule — but semantic-release pushes the version-bump commit directly to `main`, so a
naive rule with "include administrators" would jam the release flow. Decide before
enabling.

**WS4 (`#8`):** ruff+mypy blocking = done; pytest-blocking tracked in **`#16`**.

## 2026.06.30 - WS5 DONE: experiments CI scope = numbered threshold ≥016

**Decision (user):** NEW experiments (numbered **≥ 016**) are held to **ruff + format**;
everything **≤ 015**, all `DEPRECATED_*`, and non-numbered utility dirs (`database`,
`embeddings`, `figures`, `smf-dmf-tmf-001`, `tcdb-001`, `W006-*`, …) stay **frozen** for
reproducibility. Allowlist (drift risk) and per-experiment opt-in markers (machinery)
were rejected in favor of the numeric threshold — it matches the existing 002–015
convention and auto-includes new work by its id, zero bookkeeping.

**Enforcement** (`.github/workflows/style.yaml`, a step in the *blocking* ruff job):
numerically selects `experiments/NNN-*` with `NNN ≥ 16` (base-10, leading zeros handled
via `10#`) and runs `ruff check` + `ruff format --check` on them. Empty today (latest is
015) → verified no-op; the first `016-*` auto-enters the gate. The
`[tool.ruff.lint.per-file-ignores]` `experiments/**` relaxations (D100/D103/E402) still
apply, so experiment scripts get a pragmatic lint bar, not the full docstring regime.

**mypy + pytest: opt-in, not forced.** Strict mypy on research scripts is too heavy to
force; experiments stay out of the mypy gate (`files=[torchcell,tests]` + `exclude
…torchcell/experiments/`). To type-check a specific experiment, add its path to a mypy
invocation. pytest is moot unless an experiment ships tests (none do); if one does, add
its test path to the pytest gate.

**Legacy freeze** unchanged (Decision 5/6): old experiments are refactored only behind
preserved git tags.

Remaining roadmap: **WS6** (torch/PyG upgrade, `#9`), **WS7** (`weighted_mse` migration,
`#10`, needs user intent), plus **#16** (finish pytest → blocking).

## 2026.06.30 - WS7 DONE: removed dead weighted_mse stub

`SimpleLinearRegressionTask`'s `weighted_mse` branch was a `raise NotImplementedError`
stub reachable **only** from the frozen `DEPRECATED_costanzo_smf_dmf_supervised`
experiment (its config sets `loss: weighted_mse`; the trainer is instantiated nowhere
else live). The new `WeightedMSELoss(weights=)` (multi_dim_nan_tolerant.py:327) is a
per-dimension, NaN-tolerant, **tuple-returning** loss — incompatible with the trainer's
single-tensor `self.loss(y_hat, y)` flow and semantically unrelated to the old scalar
`mean_value=/penalty=`. Migrating would mean *inventing* weighting semantics for dead
code, so **removed** the branch + the stale commented import (user decision). `loss` now
supports `mse`/`mae` and `ValueError`s otherwise. ruff + mypy clean; `#10` closed.

**Roadmap status: WS1–WS5 + WS7 done.** Remaining: **WS6** (torch/PyG upgrade, `#9`) +
**#16** (finish pytest → blocking). WS6 is a large, breaking dependency upgrade best done
as its own planned effort (green tests as the safety net); #16 was deliberately deferred
by the WS4 decouple.

## 2026.06.30 - WS6 KICKOFF / PREFLIGHT (torch 2.9→2.12.1, PyG 2.7→2.8, `#9`)

Everything WS6 needs is on `main` (~`af07c437`): WS1–WS5 + WS7 + two latent bug fixes
(`#11` AUROC `~`-on-float crash, `#13` NucleotideTransformer arg misroute). All gates
green; `main` releasable, still at `v1.1.4` (nothing published since — all commits used
non-releasing prefixes).

### State WS6 inherits

- **Gates on `main`:** ruff BLOCKING whole-tree (+ a `≥016` experiments step, no-op
  today); mypy BLOCKING diff-scoped `--follow-imports=silent` on changed `torchcell`
  files (`torchcell/experiments/` filtered out of `$FILES`); pytest ADVISORY
  (`continue-on-error`).
- **Enforcement:** `main` branch protection requires the `ruff` + `mypy-check` checks,
  `enforce_admins=false`. A WS6 **PR must be ruff+mypy green to merge** (admin can
  override). Land via worktree → PR → gates → merge.
- **Release:** semantic-release runs on push to main via admin PAT
  `TORCHCELL_SEMANTIC_RELEASE_TOKEN` (bypasses protection; validated no-op, not yet a
  real release). scipy parser — releasing tags `API/DEP/DEV/ENH/REV/FEAT/BLD/BUG/MAINT`;
  a dep bump is naturally `DEP:` (minor). Non-scipy prefixes (`ci:`/`fix:`/`test:`/
  `refactor:`) DON'T release.
- **Safety net = LOCAL tests.** `not gpu` suite is locally green (~218/3/0, WS3) but
  pytest is CI-advisory + CI-fragile (`#16`). Verify WS6 with the **local** suite +
  import smoke, not the CI gate. Env: `~/miniconda3/envs/torchcell` (py3.13, currently
  torch 2.9.0+cu128, PyG 2.7.0).

### Scope + known landmines

- Bump torch 2.9.0→2.12.1, torch_geometric 2.7.0→2.8.0, aligned `torch-scatter` /
  `torch-sparse`, CUDA `cu12x` wheels. Files: `env/requirements*.txt`, `pyproject.toml`
  (`requires-python>=3.13` already set).
- `torch-scatter`/`torch-sparse` compile C++ exts that import torch at BUILD time →
  install with `--no-build-isolation`. Wheel/CUDA (cu12x) alignment is the usual pain.
- **PyG MessagePassing PEP-604 issue is NOT fixed by the upgrade** (PyG #10138,
  not-planned): keep the `UP007/UP045` per-file-ignores on
  `nn/stoichiometric_hypergraph_conv.py` + `nn/masked_gin_conv.py`. See
  [[ruff-up-breaks-pyg-messagepassing]].
- Expect PyG 2.8 API drift + torch 2.12 deprecations across models/nn/trainers, and
  possible disturbance to the local CUDA/GPU setup — worktree-isolate; be ready to
  reinstall the env.

### Recommended approach

1. `setup-worktree`; optionally scope the breaks with `/plan-4.8` first.
2. Bump requirements + pyproject; reinstall env (`--no-build-isolation` for
   scatter/sparse).
3. `import torchcell` + subpackage smoke; fix breakage iteratively.
4. Run the LOCAL `not gpu` suite as the safety net — keep it at-or-better than 218/3/0.
5. Land via PR (ruff+mypy verify); merge; decide release (`DEP:` minor vs hold).

## 2026.07.01 - WS6 EXECUTION (DONE: torch 2.11.0, PyG 2.8.0)

Landed in worktree `deps/torch-pyg-upgrade`. Preflight recon changed two of its
assumptions, both de-risking the bump:

- **Target is torch 2.11.0, NOT 2.12.1.** On our CUDA (`cu128`) the PyTorch wheel
  index caps at 2.11.0; torch 2.12.x ships only `cu126/cu130/cu132` — reaching it
  would force a CUDA-toolkit migration (the "disturb GPU setup" landmine). We chose
  2.11.0 to stay on cu128. PyG 2.8.0 is the real latest.
- **No C++ compile needed.** `torch-scatter` has a **prebuilt** cp313 wheel
  (`scatter-2.1.2+pt211cu128`) on the PyG wheel index → plain wheel install, the
  `--no-build-isolation`/nvcc path is avoided. `torch_sparse` was **dead code**
  (unreachable `import` after a `raise` in `loader/dense_padding_data_loader.py`) and
  isn't in any requirements file — dropped it + trimmed the orphaned
  `cat`/`is_torch_sparse_tensor` imports.
- **Env: in-place reinstall of the shared `~/miniconda3/envs/torchcell`** (a worktree
  isolates code, not the conda env). Stack was lean (no torchvision/torchaudio), so
  only 3 packages moved.

### Files changed (3)

- `env/requirements.txt`: `torch>=2.11.0`, `torch_geometric>=2.8.0` (floors).
- `env/requirements_dependent.txt`: documented the PyG-wheel-index install for
  `torch-scatter` (still `>=2.1.2`).
- `torchcell/loader/dense_padding_data_loader.py`: removed the dead `torch_sparse`
  branch body + its two now-unused imports.

### Install method (reproducible)

```bash
pip install --upgrade "torch==2.11.0+cu128" --index-url https://download.pytorch.org/whl/cu128
pip install --upgrade "torch_geometric==2.8.0"
pip install --force-reinstall --no-deps torch_scatter -f https://data.pyg.org/whl/torch-2.11.0+cu128.html
```

`--force-reinstall` on scatter is required: pip treats bare `2.1.2` as satisfied and
won't swap the `pt29`→`pt211` build otherwise.

### Verification (all green)

- Import smoke: torch 2.11.0+cu128, PyG 2.8.0, scatter 2.1.2+pt211cu128, `cuda.is_available()=True`
  (GPU intact); the two PEP-604 MessagePassing convs still import under PyG 2.8 — the
  `UP007/UP045` ignores stay correct (see [[ruff-up-breaks-pyg-messagepassing]]).
- Ruff whole-tree: pass. Mypy (changed file, `--follow-imports=silent`): pass.
- LOCAL `not gpu` suite: **218 passed, 3 skipped, 2 deselected** — identical to the
  218/3/0 baseline, now on the new stack.

### Deferred (out of WS6 scope; backlog)

- `env/igb_delta_match_req.txt` left frozen (exact cluster-match snapshot; separate
  Delta-sync task, not the dev floors).
- Deprecations surfaced by the suite, none first-party-blocking: PyG-internal
  `inspector.py` `typing._eval_type` (disallowed Py3.15, PyG upstream);
  `torch.jit.script` deprecated (torch-wide); Pydantic-2.12 `@model_validator`
  after-on-classmethod in `datamodels/schema.py:643` (pre-existing, Pydantic not torch).
- CI's **advisory** pytest job may go red: it source-builds `torch-scatter` against
  torch 2.11 (`--no-build-isolation`, no prebuilt CPU wheel fetched). Non-blocking
  (only ruff+mypy gate the PR). Optional follow-up: point that step at the `+cpu`
  prebuilt wheel (`-f https://data.pyg.org/whl/torch-2.11.0+cpu.html`).

### Release

Committed `DEP:` (semantic-release minor). Release fires at merge-to-main via the
admin PAT — this merge doubles as the first real release since `v1.1.4` (validates
the PAT flow) and clears all accumulated unreleased commits.

## 2026.07.01 - ROADMAP COMPLETE + post-roadmap mypy polish

> **STATUS: WS1-WS7 all DONE and enforced on `main`.** ruff + mypy + pytest are all
> BLOCKING and are the three required status checks
> (`ruff` / `mypy-check` / `pytest-coverage`); the latest `main` CI run is green on all
> five jobs. The WS3 5-source-bug fixes are on `main` (commit `65558323`, sitting
> between v1.1.2 and v1.1.3 in first-parent history); WS6 shipped `v1.2.0`. Stale
> issues `#9` (WS6) and `#16` (pytest-blocking) closed as done. The optional polish
> parked in the 2026.06.29 section is now executed (branch `plan/ci-mypy-polish`).

### Post-roadmap mypy polish (branch `plan/ci-mypy-polish`, 5 `STY:` commits)

Ran a **local whole-tree** `mypy` (CI only runs it diff-scoped) and found it was
**not** clean: **4 errors** in `scheduler/cosine_annealing_warmup.py` — a regression
the **WS6 torch 2.11 upgrade** introduced (its `LRScheduler.get_lr()` stub widened to
`list[float | Tensor]`; `list` is invariant so our `-> list[float]` override clashed),
**masked by diff-scoped CI** (unchanged file, never re-checked). Restored whole-tree
`mypy = 0` (293 files) and did the three parked items. Net **-17 `# type: ignore`, 0
added**; ruff clean; `not gpu` suite unaffected (no test imports any changed module).

- **(E)** widened the `get_lr` override annotation to match the torch-2.11 supertype
  (pure type change, no runtime effect).
- **(D)** demoted 3 dead modules from inline ignores to a pyproject `ignore_errors`
  override: `graph_attention` / `graph_convolution` (re-exported but used only by
  `DEPRECATED_costanzo`) and `trainers/utils.py` (pydantic-v1, raises on import, 0
  importers). Chose `ignore_errors` over `exclude` because the diff-scoped CI passes
  changed files to mypy explicitly, which **bypasses `exclude`** (discovery-only);
  `ignore_errors` is module-scoped and honored either way.
- **(C)** unified the last divergent trainer (`fit_int_hetero_cell`,
  `cast(AttributeDict, ...)`) onto the `_HParams(Protocol)` + `hp` idiom used by
  `int_hetero_cell` / `int_hetero_cell_nsa` (18 sites; type-only).
- **(A)** fixed **9 real latent bugs** in `__main__` demo blocks: attrs classes were
  called with the wrong kwarg (`data_root` → `genome_root` / `sgd_root`;
  `transformer_model_name` → `model_name`) — would `TypeError` if run. Verified against
  the `@define` fields + the file's own correct usage (`s288c.py:836`); mypy
  re-verifies each. Left as-is: `cell.py`'s documented-illustrative `__main__`, the
  third-party pronto/biocypher ignores, and the removed `fungal_utr_transformer` branch.

**Finding worth keeping:** CI mypy is diff-scoped by design (env-divergence makes
whole-tree unwinnable in CI), so **whole-tree type regressions from dependency
upgrades can land invisibly**. A periodic local `python -m mypy` (or a scheduled
whole-tree job) is the only thing that catches them.
