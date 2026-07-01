---
id: ysnhu3kq0a8ds1xt6hadhb4
title: '01'
desc: ''
updated: 1782929402507
created: 1782929402507
---

## Context

Roadmap item `#16` asks us to make the `pytest-coverage` CI job **blocking** so that
`pytest tests/torchcell -m "not gpu"` gates PRs into `main`. Today it is advisory:
`.github/workflows/test.yaml` job `pytest-coverage` carries `continue-on-error: true`
(line 21), so the suite runs for visibility but a red result never blocks a merge. Only
`ruff` and `mypy-check` are enforced (branch protection `contexts:["ruff","mypy-check"]`,
confirmed via `gh api` on 2026-07-01). This is the **third blocking gate**.

Why it was deferred: during the WS4 "decouple" pass (see
`[[plan.ci-quality-finish-roadmap.2026.06.25]]`, Decisions 3/5) we deliberately kept
pytest non-blocking because the full suite was not yet CI-portable -- several tests assume
ambient `DATA_ROOT` / `/scratch` data, one is a wall-clock micro-benchmark, and one asserts
filelock-version-specific behavior. WS4 blessed ruff+mypy as blocking and left pytest as a
follow-up. This plan is that follow-up.

Why now: torch 2.11.0 / PyG 2.8.0 just landed (v1.2.0). **Locally** the `not gpu` suite is
green -- 218 passed / 3 skipped / 0 failed (`~/miniconda3/envs/torchcell/bin/python -m
pytest tests/torchcell -m "not gpu"`). But that is a machine with `DATA_ROOT` populated and
`filelock 3.20.0` pinned. The **first CI run** (issue `#16` body) was 167 passed / 22
skipped / 3 failed / 9 errors -- the gap between local-green and CI-red is exactly the four
environmental/version-drift problems catalogued below. Local-green does not imply CI-green;
CI is CPU-only, has no `DATA_ROOT`, and floats unpinned `filelock >= 3.21.0`.

What "done" means: one code PR that (a) hardens the four tests with declarative skip-guards
(WS3 blesses "real skip-guards" over fallbacks), (b)
fixes the CI install to be CPU-only + wheel-based, (c) flips the blocking flag, (d)
reconciles the now-contradictory comment blocks -- **then** one post-merge admin `gh api`
call that adds `pytest-coverage` to the required checks *after* a green run on `main` is
observed. The API call cannot be atomic with the PR (the check must exist and pass once
before it can be required), so it is a sequenced closing action.

## Relevant Files

| Path | Action | Purpose | Stance |
| --- | --- | --- | --- |
| `.github/workflows/test.yaml` | MODIFY | Flip job blocking; CPU-only wheel install; Codecov step-level COE; reconcile comments; extend cache key | undocumented |
| `tests/torchcell/sequence/genome/scerevisiae/test_s288c.py` | MODIFY | Add module-level `DATA_ROOT` skip-guard (fixes 9 CI errors) | undocumented |
| `tests/torchcell/nn/test_masked_gin_performance.py` | MODIFY | `@pytest.mark.gpu` on the wall-clock benchmark; targeted `DATA_ROOT` skipif on the batch test | undocumented |
| `tests/torchcell/utils/test_file_lock.py` | MODIFY | Rewrite `test_cleanup_lock_files` to be filelock-version-independent | undocumented |
| `env/requirements.txt` | MODIFY | Bump `filelock >= 3.13.0` -> `>= 3.20.1` (CVE floor) | undocumented |
| `env/requirements_dependent.txt` | MODIFY | Document/point the `+cpu` torch-scatter wheel index for CI | undocumented |
| `env/requirements_test.txt` | MODIFY | Add reproducible ceilings `pytest>=9.0,<10`, `coverage>=7,<8` | undocumented |
| `pyproject.toml` | REFERENCE | `gpu` marker already registered (line 186); `[tool.semantic_release] version_variables` (lines 218-220) MUST stay untouched | undocumented |
| branch protection (`repos/Mjvolk3/torchcell/branches/main/protection/required_status_checks`) | REFERENCE | GitHub API object, not a file; the post-merge `gh api` PUT target | in-flux |
| `[[plan.ci-quality-finish-roadmap.2026.06.25]]` | REFERENCE | WS4 decouple rationale, Decisions 3/5 (gate scope, deferral) | in-flux |
| `[[ruff-up-breaks-pyg-messagepassing]]` | REFERENCE | Why `masked_gin_conv` imports / UP007-UP045 ignores are untouchable | in-flux |

## Key Design Decisions

**1. One code PR, then one post-merge `gh api` call -- not a single atomic change.**
Adding `pytest-coverage` to required contexts *before* the job has ever passed on `main`
creates a required check that is perpetually pending on every subsequent PR (a required
context with no matching completed run blocks the merge). So the PR must merge and produce
**one observed green `pytest-coverage` run on `main`** first; only then is the context safe
to require. GitHub offers no way to make "flip the flag" and "require the check" one atomic
operation, so we accept the two-step sequence and document the closing `gh api` call
explicitly for whoever holds admin.

**2. torch-scatter: repoint to a `+cpu` wheel, do not drop it.** The current CI builds
torch-scatter from source with `--no-build-isolation` (test.yaml lines 45-52) because the
build imports torch at build time. On investigation, **no collected `not gpu` test imports
`torch_scatter`**, so dropping it would not break collection today. We still keep it
(repoint rather than remove) to honor the install contract and avoid a latent trap the day a
future collected test imports scatter. Rejected: dropping it entirely (silent future
breakage); keeping the source build (slow, and it ABI-mismatches, see Decision 6).

**3. `test_masked_vs_filtered_speed` -> `@pytest.mark.gpu`, not a softened assertion or a
new benchmark marker.** The test (line 79) has no marker and a hard wall-clock assert
`t_masked <= t_filtering * 1.1` (line 129); on a shared CPU runner, timing noise makes that
assert flaky-red. Its two in-file GPU siblings (`test_equivalence_masked_vs_filtered`,
`test_memory_efficiency`, lines 193-282) are already `@pytest.mark.gpu`, and its own
docstring frames it as a GPU-scale benchmark -- so `gpu` is the honest, consistent marker and
`-m "not gpu"` deselects it cleanly. Rejected: introducing a `benchmark` marker (new
taxonomy for one test, and the gate filter would still need updating); softening the
tolerance (papers over genuine noise, still flaky). We do **not** touch the file's
`from torchcell.nn.masked_gin_conv import MaskedGINConv` import or its
UP007/UP045 posture -- see `[[ruff-up-breaks-pyg-messagepassing]]`.

**4. `test_pyg_concatenated_batch` -> targeted `skipif(DATA_ROOT is None)`, do NOT broaden
the except.** The test (line 134) guards only `ImportError` (lines 143-144), but on CI the
imports succeed and `load_sample_data_batch(...)` raises `TypeError` because `DATA_ROOT` is
`None` -> uncaught -> ERROR. The forbidden fix is broadening `except ImportError` to also
catch `TypeError` (a fallback / error-swallowing pattern banned by CLAUDE.md). The
declarative fix is a decorator `@pytest.mark.skipif(os.getenv("DATA_ROOT") is None,
reason=...)` on that one function, which keeps the file's pure-CPU tests
(`test_no_edge_filtering`) running. Rejected: module-level skip (would needlessly skip the
CPU-only tests in the same file).

**5. `test_cleanup_lock_files` -> rewrite version-independent, do NOT pin filelock down and
do NOT skip.** This is a real upstream behavior change, not a flaky test. filelock v3.21.0
(2026-02-12, py-filelock changelog: "delete lock file on release" on Unix) now removes the
`.lock` sentinel on release; the test asserts three `.lock` files survive writes
(`len(list(tmp_path.glob("*.lock"))) == 3`, line 146). Local `filelock 3.20.0` keeps them
(passes); CI floats unpinned and gets `>= 3.21.0` (0 survive -> fail). We cannot pin filelock
*down* to `< 3.21.0` because CVE-2025-68146 wants `>= 3.20.1`, and pinning *up* to exactly
one version is out of scope (Decision 8). So we rewrite the test to not depend on
release-time lock-file persistence at all: create explicit orphan `.lock` files by hand
(`Path(tmp_path, f"file_{i}.json.lock").touch()`), then assert `cleanup_lock_files()` returns
that count, zero `*.lock` remain, and the `*.json` data files are untouched. This tests the
*cleanup behavior we actually care about* independent of filelock internals. Rejected:
skipping (loses coverage of a utility we own); `xfail` (WS3 forbids hiding real tests).

**6. CI install becomes CPU-only + wheel-based (torch from the CPU index, scatter from the
`+cpu` PyG wheel).** On Linux, `torch >= 2.11.0` from PyPI is a **CUDA** build. Installing a
`+cpu` torch-scatter wheel against CUDA torch ABI-mismatches at import, and the CUDA
libraries are dead weight on a CPU-only GitHub runner (slow install, wasted cache). So CI
installs torch from `--index-url https://download.pytorch.org/whl/cpu` and torch-scatter from
`-f https://data.pyg.org/whl/torch-2.11.0+cpu.html` (confirmed 2026-07-01 via data.pyg.org:
`torch_scatter-2.1.2+pt211cpu-cp313-cp313-linux_x86_64.whl` is HTTP-200 present). This also
retires the fragile `--no-build-isolation` source compile.

**7. Codecov upload becomes the only step-level `continue-on-error`.** Removing the
job-level flag makes *every* step blocking, including the Codecov upload (lines 71-74). A
Codecov service outage or a token hiccup must not block a PR whose tests are green. So we add
`continue-on-error: true` to only that upload step. Coverage *computation* (`coverage run` /
`report` / `xml`) stays blocking; only the external upload is tolerant.

**8. Out of scope (stated so reviewers do not expect it).** No coverage `--fail-under`
threshold (blocking on a number is a separate policy decision). No change to any
`experiments/` gating (Decision 5: the gate is `tests/torchcell` only, never `experiments/`,
never bare repo-wide `pytest`). No hard-pinning any dependency to exact `==` (we use floors
and, where reproducibility matters for the gate, major-version ceilings). No touching the two
`gpu`-marked benchmarks' internals or the `MaskedGINConv` / MessagePassing conv imports.

## Approach

Execution order is: fix the tests, fix the install/requirements, flip the flag, verify
locally, open the PR, merge, observe green on `main`, then make the API call. Commit code
with a **non-releasing** prefix (`ci:` or `test:`; the scipy commit parser treats these as
inert so semantic-release will not cut a version bump).

**Step 1 -- Harden the four tests.**

- `test_s288c.py`: add a module-level guard right after `DATA_ROOT = os.getenv("DATA_ROOT")`
  (line 13). The heavy import at line 9 (`SCerevisiaeGenome`) does not itself need
  `DATA_ROOT` (only the `genome` fixture does), so unlike the sibling files we need **not**
  reorder imports below the guard. Use the exact sibling idiom (verbatim from
  `tests/torchcell/data/test_cell_data.py:16` and `tests/torchcell/nn/test_hetero_nsa.py:19`):

  ```python
  DATA_ROOT = os.getenv("DATA_ROOT")
  if DATA_ROOT is None:
      pytest.skip("requires DATA_ROOT data (absent in CI)", allow_module_level=True)
  ```

  After the guard, `DATA_ROOT` is narrowed to `str`, so the existing
  `assert DATA_ROOT is not None` inside the fixture (line 21) can be removed and the
  `osp.join` calls stay mypy-strict clean.
- `test_masked_gin_performance.py`: add `@pytest.mark.gpu` above
  `test_masked_vs_filtered_speed` (line 79). Separately, add `import os` (module currently
  imports only `time`) and `@pytest.mark.skipif(os.getenv("DATA_ROOT") is None, reason="...")`
  above `test_pyg_concatenated_batch` (line 134). Leave every other line -- imports,
  UP007/UP045 posture, the two already-`gpu` benchmarks -- untouched.
- `test_file_lock.py`: rewrite the body of `test_cleanup_lock_files` (lines 134-158) per
  Decision 5 -- hand-create orphan `.lock` files, assert `cleanup_lock_files()` return count,
  zero `*.lock` remaining, `*.json` untouched. No new imports beyond `Path` (already
  imported).

**Step 2 -- Requirements.** In `env/requirements.txt` bump `filelock>=3.13.0` ->
`filelock>=3.20.1` (line 83; CVE-2025-68146 floor). In `env/requirements_test.txt` change
the three bare pins to `pytest>=9.0,<10`, `pytest-cov` (floor, unchanged intent), and
`coverage>=7,<8` (reproducible gate). For the `+cpu` scatter index, keep local CUDA installs
unaffected: rather than editing the CUDA example in `env/requirements_dependent.txt` (which
local installs read), inject the `+cpu` `-f` index in the workflow install step. Optionally
leave a comment in `requirements_dependent.txt` pointing at the `+cpu` variant for CI.

**Step 3 -- test.yaml.** Remove `continue-on-error: true` at job level (line 21). Add
`continue-on-error: true` to only the Codecov step (after line 72). Rewrite the two install
steps: install torch via the CPU index and torch-scatter via the `+cpu` `-f` wheel, dropping
`--no-build-isolation`/`ninja`. Merge the contradictory comment blocks (lines 16-20
"Advisory / non-blocking" vs 62-65 "Blocking (WS4)") into a single truthful "blocking" note.
Extend the pip cache key (line 36) per Gotcha 4.

**Step 4 -- Verify locally** (see Verification). The `not gpu` suite must stay 218/3/0 (or
better) with the guards in place, and `test_cleanup_lock_files` must pass on both sides of
the filelock 3.21.0 behavior change.

**Step 5 -- PR, merge, observe.** Open the PR; watch the (still non-required) `pytest-coverage`
check go green on the PR itself. Merge. Note the semantic-release bump-commit race
(Gotcha 4). After merge, confirm **one green `pytest-coverage` run on `main`**.

**Step 6 -- Require the check (admin, post-merge).** Once that green run on `main` exists,
add the context, preserving `strict:false` and `enforce_admins:false` (so the
semantic-release PAT bump still bypasses):

```bash
gh api -X PUT repos/Mjvolk3/torchcell/branches/main/protection/required_status_checks \
  -f strict=false \
  -f 'contexts[]=ruff' -f 'contexts[]=mypy-check' -f 'contexts[]=pytest-coverage'
```

The context is the **job key** `pytest-coverage` (not the workflow display name "Pytest with
Coverage"). Using the display name yields a required check that never matches a run -> every
PR blocked (Gotcha 1).

## Gotchas

**1. Wrong context name = all PRs perpetually blocked.** The required-check string must be
the job key `pytest-coverage` (from `jobs:` in test.yaml), *not* the workflow `name:` "Pytest
with Coverage". A non-matching required context sits pending forever and blocks every merge.
Sidestep: PUT exactly `["ruff","mypy-check","pytest-coverage"]`; verify with a follow-up
`gh api GET` and a throwaway PR before walking away.

**2. MD024 blocks the note commit.** This roadmap/plan note has duplicate headings across
dated sections; markdownlint MD024 (via pre-commit) will reject the commit. Sidestep:
`git commit --no-verify` for the **note only**. Never bundle code into a `--no-verify` commit
-- code must pass ruff + strict-mypy pre-commit hooks.

**3. semantic-release pushes a bump commit to `main` after merge.** CI commits a version bump
to `main` after each merge (`[[semantic-release-pushes-bump-commit]]`). Before any subsequent
push to `main`, `git fetch` + rebase local `main`, or the push is rejected. Also: this is why
we commit with `ci:`/`test:` -- a `feat:`/`fix:` prefix would trigger its own release.

**4. Stale pip cache masks the new install.** The cache key currently hashes only
`requirements.txt` (line 36); after we change `requirements_test.txt` and
`requirements_dependent.txt` the old cache could still be restored. Sidestep: extend the key
to `hashFiles('env/requirements.txt', 'env/requirements_dependent.txt',
'env/requirements_test.txt')`.

**5. Do not touch `masked_gin_conv` imports.** `test_masked_gin_performance.py` exercises a
PyG MessagePassing module; `X|Y` union syntax (ruff UP007/UP045) breaks PyG's runtime
signature inspection (`[[ruff-up-breaks-pyg-messagepassing]]`, PyG issue #10138 not-planned).
Adding markers/skipif is safe; changing imports or the union posture is not.

## Verification

Local (on GilaHyper, the populated-`DATA_ROOT` machine):

```bash
# Full gate stays green after the guards (target: 218 passed / 3 skipped / 0 failed, or better)
~/miniconda3/envs/torchcell/bin/python -m pytest tests/torchcell -m "not gpu"

# The filelock rewrite in isolation
~/miniconda3/envs/torchcell/bin/python -m pytest \
  tests/torchcell/utils/test_file_lock.py -xvs
```

Prove the filelock rewrite is genuinely version-independent -- run the rewritten test on
**both** sides of the 3.21.0 behavior change:

```bash
# Current (3.20.0): keeps .lock on release
~/miniconda3/envs/torchcell/bin/python -m pytest \
  tests/torchcell/utils/test_file_lock.py::test_cleanup_lock_files -x
# Force the post-3.21.0 behavior, re-run, then restore
~/miniconda3/envs/torchcell/bin/pip install 'filelock>=3.21.0'
~/miniconda3/envs/torchcell/bin/python -m pytest \
  tests/torchcell/utils/test_file_lock.py::test_cleanup_lock_files -x
~/miniconda3/envs/torchcell/bin/pip install 'filelock==3.20.0'   # restore local pin
```

Lint / type the touched test files (pre-commit runs these on commit anyway):

```bash
ruff check tests/torchcell/sequence/genome/scerevisiae/test_s288c.py \
  tests/torchcell/nn/test_masked_gin_performance.py \
  tests/torchcell/utils/test_file_lock.py
mypy tests/torchcell/sequence/genome/scerevisiae/test_s288c.py \
  tests/torchcell/nn/test_masked_gin_performance.py \
  tests/torchcell/utils/test_file_lock.py
```

CI side (the part that actually validates the goal):

1. On the PR, confirm the `pytest-coverage` check runs **green** (0 failed / 0 errors; the
   `DATA_ROOT`-dependent files self-skip, the benchmark is deselected by `-m "not gpu"`).
2. Merge; confirm **one green `pytest-coverage` run on `main`**.
3. Run the branch-protection `gh api` PUT from Approach Step 6.
4. Confirm with `gh api GET .../required_status_checks` that `contexts` is exactly
   `["ruff","mypy-check","pytest-coverage"]`, `strict:false`, and that a fresh PR now shows
   `pytest-coverage` as a **required** check that must pass to merge.
