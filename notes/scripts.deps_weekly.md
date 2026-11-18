---
id: trtgvl4l4nk5vsamk3ylsh7
title: Deps_weekly
desc: ''
updated: 1794963677026
created: 1794963677026
---

## 2026.11.17 - The weekly dependency job

Owner's ask: a Sunday cron job that checks whether packages moved, runs the tests on
the new versions and does the release routine, aggressively, to stay current.

`scripts/deps_weekly.py run` (cron: [[scripts.deps_weekly_cron]], Sunday 05:00, in a
fresh detached worktree of `origin/main`):

1. Refuses a dirty tree, a cwd that is not the worktree root, a non-empty work dir
   (`/scratch/tmp/torchcell-deps-weekly/<YYYY.MM.DD>`), and an interpreter other than
   the `python` of `env/deps_policy.toml` (3.13.0; schema fingerprints depend on the
   patch release).
2. Picks torch: the newest final PyPI release, not below the torch floor of
   `env/requirements.txt`, whose `https://data.pyg.org/whl/torch-<v>+cpu.html` lists a
   cp313 linux x86_64 `torch_scatter` wheel. On 2026-11-17 PyPI's latest was 2.15.0
   and the pick was 2.12.1 (2.15.0 and 2.14.1 have no PyG page, 2.14.0 and 2.13.0
   pages list no torch_scatter).
3. Arm `capped`: `python -m venv`, `pip install uv` into it, then `uv pip install` in
   the CI order: torch from the CPU index, `-r env/requirements.txt`, torch-scatter from
   the PyG page (`--only-binary torch-scatter`, so never a source build),
   `-r env/requirements_test.txt`, `-e . --no-deps`. Steps 2 to 4 carry a constraint
   `torch==<v>`, so no package can silently swap the tested CPU torch for a PyPI build;
   a package that needs a newer torch fails the install instead. uv, pip and `TMPDIR`
   live in the work dir on /scratch.
4. `nice -n 10 pytest tests/torchcell -q -p no:cacheprovider` with a fresh empty
   `DATA_ROOT`, `PYTHONHASHSEED=0`, `PYTHONPATH` and `TC_DATA_URL` unset, thread
   variables capped at 8. After the suite the worktree must still be clean.
5. Diff of the resolution against `env/tested/requirements.lock.txt` (added, removed,
   changed). No lock is "first snapshot".
6. Arm `uncapped` (only when `capped` passed and a cap has `probe = true`): the probed
   caps stripped from a temp copy of the requirements. A cap is "lifted" when the
   uncapped version breaks its bound and the uncapped suite passed, "not binding" when
   the uncapped version still satisfies it, "kept: uncapped suite failed" otherwise.
   One arm strips every probed cap at once, so when two caps bind and the suite fails,
   neither is lifted (no per-cap bisection).
7. When `capped` passed, the new tracked files are computed from the effective
   resolution (the uncapped one when a cap was lifted, else the capped one): the lock;
   `env/torch-ci-version.txt` when the pick moved; floors in `env/requirements.txt`
   raised to the tested version (public part, so `2.12.1+cpu` gives `2.12.1`), never
   lowered, never for `[hold]`, upper bounds kept unless lifted, every comment, blank
   line and untouched requirement line kept byte for byte; a lifted cap's `[caps.<name>]`
   table removed from the policy. A lifted cap's comment block (the contiguous comment
   lines directly above its line) is deleted only when the policy entry names a
   `comment_marker` found in that block; otherwise the comment stays and the report
   flags it as stale. Prose comments are never rewritten.
8. No file changed: "no-change". Otherwise (not `--dry-run`): branch
   `deps/weekly-<date>`, commit, push, `gh pr create`. Subject
   `PATCH(deps): weekly dependency update <date>` when `env/requirements.txt` changed
   (semantic-release cuts a patch release to PyPI once it lands), else
   `build(deps): weekly tested-versions snapshot <date>` (no release). The body is the
   report as markdown.
9. `--land`: waits for the PR's checks to register, then
   `gh pr checks <n> --watch --fail-fast` bounded by `--land-timeout` (2 h), and
   enqueues with the primary checkout's `scripts/merge_queue.py add <branch> --worktree
   <wt> --by deps-weekly` only when every check is pass or skipping and
   `merge_queue.py loop-status` exits 0 (the `*/2` cron drainer then lands it). The
   primary checkout's copy is used because its `.env` places the queue database. The
   job does not start a drainer itself.
10. A red `capped` suite changes nothing, opens no PR, and (not `--dry-run`) files one
    issue titled `deps-weekly: test suite fails on the newest dependencies` (label
    `torchcell-lib`) with the moved packages and the last 60 log lines, or comments on
    the open issue with that title.
11. Always writes the JSON report to
    `${XDG_STATE_HOME:-~/.local/state}/torchcell/deps-weekly/<date>.json` plus a copy in
    the work dir, prints the markdown report and a final one-line status
    (`deps-weekly <date>: <STATUS> exit=... torch=... capped=... moved=...`). The venvs,
    caches, temp and data-root directories go to `/scratch/projects/torchcell-deprecated`
    through `scripts/deprecate.sh` (a rename on the same filesystem; the daily purge
    removes them after 14 days) unless `--keep-work`; the logs and report stay.

Exit codes: 0 done (no change, changes, dry run), 1 the suite failed on the newest
dependencies, 2 the job failed (refusal, install, git, gh, network, a malformed file; an
uncaught crash is mapped to 2 as well).

Other subcommands: `report [--path]` prints the newest report (`make deps-report`);
`write-lock [--report]` writes the lock from a report's tested resolution through the
same `render_lock` and `write_tracked` path `run` uses (how the first lock was made from
the dry run). `make deps-weekly` runs a dry run; `make deps-weekly DEPS_ARGS=` a real one.

Not exercised for real: `--land`, the commit/push/PR path and the issue path (the
decision logic is unit-tested in [[tests.scripts.test_deps_weekly]]).

## 2026.11.17 - One uncapped arm per probed cap, and the first dry runs

Step 6 above changed after the third dry run. One arm that stripped every probed cap at
once could not judge them separately: bibtexparser 2.1.0 has no
`bibtexparser.bibdatabase`, so `torchcell/literature/bib.py` failed to import, pytest
stopped at `tests/torchcell/conftest.py` (exit 4, no summary line, 14 s), and neo4j 6.4.1,
installed in the same arm, was never tested. Now each cap with `probe = true` gets its own
arm `uncapped-<name>` with only that cap removed. A binding cap whose arm passes is
lifted, but at most one per run (the first in file order): each arm tested one cap
removed, so lifting two at once would ship a combination no arm ran. The others are
reported "deferred" and next week's run tests them on top of the lifted one. The lifted
cap's arm is the resolution the rewritten files and the lock describe.

Dry runs on 2026-11-17 (`run --dry-run --keep-work`, GilaHyper, conda Python 3.13.0;
driver logs `/scratch/tmp/torchcell-deps-weekly/dryrun-2026.11.17*.log`):

- Torch pick: 2.12.1 (PyPI latest 2.15.0; 2.15.0 and 2.14.1 have no PyG page, 2.14.0
  and 2.13.0 list no torch_scatter). `torch-scatter==2.1.2+pt212cpu` installed as a
  prebuilt wheel. Capped install 47 to 51 s with a cold uv cache.
- Run 1 (work dir `2026.11.17`): capped arm `16246 passed, 1176 skipped, 10 xfailed,
  7930 warnings in 1113.11s`, then the job stopped with exit 2 because notes created in
  the worktree during the run made it dirty (the post-suite guard working as meant).
- Run 2 (`2026.11.17-rerun`, identical resolution): `1 failed, 16245 passed` in
  1077.25 s. The failure was
  `tests/torchcell/sequence/genome/scerevisiae/test_s288c_synthetic.py::test_concurrent_processes_keep_the_original_pair_over_an_earlier_one`
  (one of four racing processes exited 1). The same test passed 15 of 15 times alone in
  the job's venv and 15 of 15 in the conda env. Hypothesis (untested): a load-dependent
  race in the test, not a dependency regression; the same versions passed run 1 and
  run 3.
- Run 3 (`2026.11.17-run3`): capped arm `16246 passed, 1176 skipped, 10 xfailed, 7929
  warnings in 1087.17s`; the single uncapped arm (neo4j 6.4.1 with bibtexparser 2.1.0)
  failed at conftest import as described above. Would raise 44 floors and move the CI
  torch from 2.11.0 to 2.12.1.
