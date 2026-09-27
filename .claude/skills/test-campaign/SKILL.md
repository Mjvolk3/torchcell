---
name: test-campaign
description: Run one phase of the test-coverage campaign end to end -- pick live-critical, low-coverage modules, write tests that assert exact behavior (never shape/truthiness padding), run every gate, have an independent Fable 5.1 agent audit every test before it is committed, record the generated coverage table and the audit in the campaign note, then land through the merge queue. Use for "raise coverage on X", "next test phase", "add tests for module Y".
---

# Test Campaign

One phase of the coverage campaign, from target selection to a landed PR. The
standing record is `notes/test-campaign.2026.09.25.md` (one dated H2 per phase) and
the design is `notes/plan.test-suite-buildout.2026.09.25.md`. This skill exists so a
phase is never run from memory: the order below is what produced Phases 0 to 4.

**Usage:** `/test-campaign <phase description or module list>`

Examples:

```
/test-campaign dataset loaders on synthetic raw files: costanzo2016, kuzmin2018, kemmeren2014
/test-campaign next phase: the lowest-covered live-critical modules under torchcell/data
```

## What a test is for here

The campaign is about **coverage of behavior, not lines**. A test that raises the
number without pinning a fact the code promises is padding and costs review time twice
(once to audit, once to delete). Every test must be able to FAIL on a plausible bug:

- **Exact expectations.** Closed-form values worked in the docstring (with the
  arithmetic), exact tensors, exact strings, exact rows. Not "is not None", not
  "shape == (3, 4)" alone, not "does not raise" unless the contract is "does not raise".
- **Randomly initialized networks** carry no float contract, so pair shape and
  finiteness with a **structural identity** (Decision 12 of the plan): seeded
  determinism, equivariance to a permutation, a ReZero or zero-init identity, exact
  parameter arithmetic, a gradient reaching every parameter or a NAMED exception.
- **Findings are pinned, named and recorded.** When the code does something other than
  its docstring says, the test asserts what the code DOES, the docstring of the test says
  why, and the campaign note lists it under "Findings". The fix, if any, is its own PR;
  the campaign does not fix source unless the test cannot be written otherwise (and
  then diff-cover must pass on the changed lines).
- **Hermetic.** Plain `pytest` runs with the sentinel `DATA_ROOT`, no network, no `gh`,
  no `sbatch`, no Neo4j, no wandb (the root conftest enforces it). Anything else is
  behind `--data`, `--network`, `--neo4j`, `--gpu`, `--slow`, `--wandb`.

## Steps

### 1. Choose targets from the table, not from memory

```bash
MAIN="$HOME/Documents/projects/torchcell"
PY="$HOME/miniconda3/envs/torchcell/bin/python"
# baseline table on current main (behavioral run + import-only run + gaps)
cd "$MAIN" && make cov-gaps
```

Read the live-critical rows (`legacy_partition.py` marks them) with the largest
statement counts and lowest coverage. Prefer modules whose behavior can be exercised on
hand-built inputs (a three-gene graph, a two-row raw file, a hand-written batch).
Modules that need the genome, a served database or a multi-GB raw file stay data-gated.
Write the chosen list and the reason for each into the phase's section of the campaign
note as the first thing (it is the plan of record for the phase).

### 2. Worktree

```bash
git -C "$MAIN" fetch origin main --quiet
git -C "$MAIN" worktree add ~/Documents/projects/torchcell.worktrees/plan/<slug> -b plan/<slug> origin/main
cd ~/Documents/projects/torchcell.worktrees/plan/<slug> && bash scripts/setup-worktree.sh
```

Every edit below happens in the worktree. Wrap every shell command in a subshell
`( cd <worktree> && ... )`: parallel tool calls share one cwd.

### 3. Write the tests

- Paired location: `tests/torchcell/<same relative dir>/test_<module>.py`
  (`scripts/check_paired_tests.py` enforces it; collisions go in `pyproject.toml`
  `[tool.torchcell.test_exceptions] pairs`).
- Header: the three-line frontmatter (path, `[[dendron link]]`, GitHub URL) then a
  module docstring stating the fixture and the expected values with their derivation.
- Fixtures shared across files live in `tests/torchcell/conftest.py` (the CGT batch,
  the DCell graph, `FakeTxn`).
- Read the source before asserting; a wrong constant is the most expensive audit
  outcome. Verify each closed form with a two-line Python check before writing it in.
- A test that pins a quirk names it: `"""Finding: ... Pinned until ..."""`.

### 4. Gates, in the worktree

```bash
PY="$HOME/miniconda3/envs/torchcell/bin/python"
R="$HOME/miniconda3/envs/torchcell/bin/ruff"
F=(tests/torchcell/<new files>)
$R format "${F[@]}" && $R check --fix "${F[@]}"
$PY -m mypy "${F[@]}"
env -u DATA_ROOT PYTHONPATH=$(pwd) $PY -m pytest "${F[@]}" -q -p no:cacheprovider
$PY scripts/test_quality_check.py tests          # anti-padding lint, must be clean
$PY scripts/check_paired_tests.py --base origin/main
```

Then the full behavioral run for the table (about a minute), keeping the JSON beside
the previous phase's under the session scratchpad:

```bash
env -u DATA_ROOT PYTHONPATH=$(pwd) $PY -m coverage run --data-file=$OUT/.coverage.pN \
  -m pytest tests/torchcell --deselect tests/torchcell/test_import_all.py -q -p no:cacheprovider
$PY -m coverage json --data-file=$OUT/.coverage.pN -o $OUT/coverage-pN.json
$PY -m coverage xml  --data-file=$OUT/.coverage.pN -o $OUT/coverage-pN.xml
$PY scripts/coverage_gaps.py --before $OUT/coverage-p(N-1).json --after $OUT/coverage-pN.json \
  --import-only $OUT/coverage-import.json > $OUT/phaseN_table.md
diff-cover $OUT/coverage-pN.xml --compare-branch=origin/main --fail-under=80 \
  --exclude 'torchcell/legacy/*' 'torchcell/scratch/*' 'torchcell/experiments/*'
```

The sentinel directory `/tmp/torchcell-test-data-root` must not exist afterwards; if it
does, a test wrote to `DATA_ROOT` and that is a finding to fix before the audit.

### 5. Independent audit BEFORE committing (mandatory)

Spawn one `general-purpose` Agent with `model: "fable"`, read-only, in the background,
and keep working on notes while it runs. The prompt names the worktree, the files, the
plan's Decision 12, and asks for, per test function, one of ACCEPT / ACCEPT WITH NOTE /
REWRITE (with the stronger assertion spelled out) / REJECT (padding), with every
constant and exact string re-derived from the source and every irreproducible constant
reported; plus hermeticity (network, `gh`, `sbatch`, the real queue database, the real
`main`, the developer's `HOME`), docstring-vs-assertion agreement, and prose style (no
em-dashes, American spelling). Totals at the end.

Apply every REWRITE and every REJECT (replace, never delete to make the number look
better). Apply the cheap notes (exact string instead of substring, a docstring that
names what is pinned). Re-run step 4 for the touched files. Record the audit verbatim in
shape: reviewer, files, test functions before/after, rejected, rewritten, accepted with
note, accepted, wrong constants, "After the audit: N passed".

### 6. Notes

- One paired note per new test file (`dendron-cli note write --fname
  "tests.torchcell.<...>"`, then a dated H2 describing the fixture, the expected values
  and the findings).
- Campaign note `notes/test-campaign.2026.09.25.md`: a new dated H2 for the phase with
  the file list, the findings, the "Runs:" line (counts, seconds, diff-cover), the table
  pasted verbatim from `coverage_gaps.py` (never hand-typed; normalize the
  `Generated by:` paths to file names), and `### Quality audit`.
- Weekly child note `notes/user.Mjvolk3.torchcell.tasks.weekly.<YYYY>.<WW>.<slug>.md`:
  one `- [x] PR-N ...` bullet with the dendron links.
- Update the paired note of any SOURCE module the phase touched.

### 7. Commit, PR, land

```bash
git add -A && ( git commit -q -F msg.txt || true )
if [ -n "$(git status --short)" ]; then git add -A && git commit -q -F msg.txt; fi   # hook auto-fixes
git log --oneline -2          # exactly one new commit on top of origin/main
git push -u origin plan/<slug>
gh pr create --title "TST: ... (audited)" --body "..."
gh pr checks <n> --watch --fail-fast
```

Never `--amend` after a hook failure (it can rewrite the landed commit under HEAD).
When CI is green, `/enqueue-merge plan/<slug>` (gates, queue, watch, banner). If another
phase is waiting in a sibling worktree, `git rebase --autostash origin/main` it after the
landing and rerun its step 4 before its own audit.

## Rules

- No test without an exact expectation or a structural identity; the lint and the audit
  both enforce it, and a REJECT is replaced, not deleted.
- The audit runs on Fable 5.1, read-only, before the commit, every phase, no exceptions.
- Source fixes are separate PRs unless a test cannot be written without them.
- Tables in notes come from `scripts/coverage_gaps.py`; numbers in prose come from the
  run that produced them, with the run named.
- Everything in a worktree; land through the queue; the worktree is gone when the phase
  is done.
