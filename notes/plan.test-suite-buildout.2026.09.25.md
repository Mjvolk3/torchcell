---
id: 36ls13bdrq4hrxl9y64m9i9
title: Test Suite Buildout Plan (2026.09.25)
desc: ''
updated: 1790374849506
created: 1790374849506
---

## Context

Source coverage of `torchcell/` is low and unevenly low: the packages the paper's models
live in are almost untested while the data-model layer is well covered. Measured baseline
(codecov on `main`, run in the CI environment, which has no `DATA_ROOT`, no GPU, and runs
`coverage run -m pytest tests/torchcell -m "not gpu"`; statistic = codecov line coverage,
2026-09-25): torchcell total 24.3% (16,521 of 67,961 lines hit). Per package: models 4.4%
(16,645 lines), trainers 2.7% (6,635), data 21.3% (4,029), datasets 31.8% (12,976),
knowledge_graphs 38.0%, adapters 35.8%, graph 27.5%, sequence 35.4%, nn 25.2%, losses
41.4%, transforms 63.8%, datamodels 71.6%, provenance 76.3%; at 0%: sga (1,294),
pypy_adapters (1,219), yeastmine (901), metrics (465), cpu_benchmark_system_monitor.py
(369), profilers (330), ontology (325). A second number exists (local dev box with the real
`DATA_ROOT` and `--slow --data`) and has never been measured; it is a different statistic.

Three things produce that shape. First, there is no shared test infrastructure: no
`tests/conftest.py`, no flag gating, and `DATA_ROOT` guarding is hand-rolled per file in
three idioms (module-level skip, `pytestmark`, in-body skip), so any test that needs a
fixture copies one. Second, the model and trainer surface has one synthetic fixture,
duplicated verbatim across two files
(`tests/torchcell/models/test_equivariant_cell_graph_transformer.py:30-64` and
`test_cell_graph_transformer_metabolism.py:30-64`), and no test constructs a trainer.
Third, nothing holds the number: [[plan.pytest-ci-blocking.2026.07.01]] Decision 8
deferred `fail_under` as "a separate policy decision", and models at 4.4% is what no gate
produced. The port takes the sibling repo's ideas
(`/home/michaelvolk/Documents/projects/iBioFoundry-AI/tests/conftest.py`:
setdefault-before-import, `pytest_addoption` opt-in buckets, autouse boundary guards,
registry-wide invariant sweeps, hermetic subprocess tests) and adds what that repo lacks:
a generated before/after table and a ratcheted gate.

User review, 2026-09-25, three additions carried by Decisions 17-22 and Phase 0d: the
headline number must measure behavior, not padding (a "no cheating" evaluation between
phases); new or changed source without a test is gated, with exceptions shaped like the
diff-scoped mypy gate; and legacy code moves out of the live tree so the live tree runs in
"hard mode" with no per-module exception. Where an earlier decision changes, the text
names the supersession (Decisions 1, 2, 5, 9; Phase 0b's delta; old Open Question 1).

No open issue is about tests or coverage. Adjacent: `#321` (DCell training speed, touches
`dcell.py`/`dcell_opt.py`, the reason DCell earns a test and stays in the live tree),
`#101` (`torchcell.mplstyle` sets `savefig.bbox: tight`; six trainers apply that style at
import), `#319` (stable unique ids for genes and alleles; would change schema fields, so
the round-trip sweep introspects classes instead of naming them).

## Relevant Files

Infrastructure and config (Phase 0a-0d); the per-phase test files are named in Approach,
the eight `is None` gate files in Gotcha 2.

| Path | Action | Purpose |
|---|---|---|
| `tests/conftest.py` | NEW | sentinel `DATA_ROOT` via `setdefault` before any `torchcell` import; `pytest_addoption` flags; `pytest_collection_modifyitems`; autouse boundary guards |
| `tests/torchcell/conftest.py` | NEW | shared tiny CGT `cell_graph`/`batch` fixtures, `_FakeTxn` LMDB stub, 3-term DCell GO fixture |
| `tests/torchcell/test_import_all.py` | NEW | smoke test: parametrized import of every `torchcell` module minus `NEVER_IMPORT`, strict-xfail `KNOWN_BROKEN`; deselected from the behavioral run (Decision 17) |
| `scripts/coverage_gaps.py` | NEW | reads `coverage json` output; per-module table, before/after diff, import-only column, live-critical column from the importer graph |
| `scripts/test_quality_check.py` | NEW | stdlib + `ast` anti-padding lint on `tests/` (Decision 19); `make test-quality`, pre-commit, CI |
| `scripts/check_paired_tests.py` | NEW | stdlib paired-test gate on added (strict: changed) `torchcell/**/*.py` vs `origin/main`; exceptions from `[tool.torchcell.test_exceptions]` (Decision 22) |
| `scripts/legacy_partition.py` | NEW | computes the closed legacy cluster from the importer graph; `--check` asserts the live tree has no legacy-only module (Decision 22) |
| `torchcell/legacy/` | NEW | destination of the Phase 0d move, `<original subpath>` preserved, no shims; the one exception prefix |
| `torchcell/metabolism/__init__.py` | NEW | makes live `flux_layer`/`yeast_GEM`/`pathway` a package so coverage and `walk_packages` see it |
| `notes/test-campaign.2026.09.25.md` | NEW | campaign note; dated H2 per phase; embeds `coverage_gaps.py` output; `### Quality audit` per test PR |
| `pyproject.toml` | MODIFY | markers `data neo4j network wandb` (213-216), `timeout`, `[tool.torchcell.test_exceptions]`, `fail_under` after Phase 1; Phase 0d: `omit` in `[tool.coverage.run]` (218-220), mypy `exclude`/`ignore_errors` (103-140, 201) and ruff `extend-exclude` (55) collapse to `torchcell/legacy/` |
| `Makefile` | MODIFY | `test test-ci test-fast test-import test-quality paired-tests diff-cov legacy-check cov cov-html cov-gaps`; only `tc-onto*`, `ops*`, `paper*` today (1-15) |
| `.github/workflows/test.yaml` | MODIFY | `fetch-depth: 0` on checkout (24), `git fetch origin main`, behavioral + import-only coverage runs replacing line 67, `diff-cover`, `check_paired_tests.py`, `test_quality_check.py` steps |
| `.github/workflows/mypy.yaml` | MODIFY | extend the diff-scope glob at line 57 to `tests/**/*.py` |
| `.pre-commit-config.yaml` | MODIFY | two local hooks: `test-quality` on `^tests/`, `paired-tests` on `^torchcell/` (shape of the `mypy` hook, 31-37) |
| `.gitignore` | MODIFY | add `lightning_logs/` beside `wandb/` (26) |
| `env/requirements_test.txt` | MODIFY | add `pytest-timeout`, `diff-cover>=10.6,<11`; keeps `pytest>=9.0,<10`, `coverage>=7,<8` |
| `tests/torchcell/scripts/test_wt_cleanup.py` | REFERENCE | the bare-origin + clone fixture template (24-52) for Phase 4 |
| `tests/torchcell/datasets/scerevisiae/test_hoepfner2014.py` | REFERENCE | `_FakeTxn` (70) and the `__new__` no-process construction (83-87) |
| `tests/torchcell/models/test_equivariant_cell_graph_transformer.py` | MODIFY | in-flux (note 2026.07.31); builders move to conftest, component tests added in Phase 2 |

## Key Design Decisions

1. **No `omit` in the live tree; the denominator moves exactly three times, each its own
   PR with before/after stated: `metabolism/__init__.py` (0a), the deprecations (0c), the
   legacy move (0d).** The original form, "the denominator stays fixed for the whole
   campaign; rejected: omitting legacy models", is superseded by Decision 22 at the user's
   request for hard mode. What survives: an `omit` is honest only in `[tool.coverage.run]`
   (in `[tool.coverage.report]` it hides files from the text report while `coverage xml`
   still includes them), only for the one prefix `torchcell/legacy/*`, and only after the
   code has physically moved; [[plan.ci-foundation-ruff-mypy-pytest.2026.06.18]]
   (2026.06.21 outcome, item 2: "Excluded-from-gates != pruned; the file stays") still
   holds: legacy code is relocated, not pruned.
2. **Legacy models: test the live surface; test DCell; move the rest.** The request's
   legacy list is wrong on two of five names: `equivariant_cell_graph_transformer` and
   `trainers/int_transformer_cell` are live (experiments 019/025, September 2026 commits,
   existing tests). `dcell.py` and `DCellSubsystem` stay in the live tree and get a test:
   `#321` is open against `dcell.py`/`dcell_opt.py`, `dcell_opt.py` imports `dcell`, and
   the fixture is small (`dcell.py:104-230` reads five `HeteroData` attributes and one
   edge type). The earlier "Dango stays in the denominator untested" is superseded by
   Decision 22: Dango is a closed cluster (reached only from experiments 004-006 and
   `trainers/int_hetero_cell{,_nsa}.py`, themselves reached only from those experiments,
   grep verified) and moves in Phase 0d, as does `trainers/regression.py` (pre-existing
   `ImportError` on `WeightedMSELoss` at 35); "trainers regression" means
   `neo_regression.RegressionTask`.
3. **Deprecate `torchcell/yeastmine/` and `torchcell/cpu_benchmark_system_monitor.py`;
   keep `torchcell/ontology/`.** Both have zero code importers; the monitor's only
   reference is a slurm launcher in frozen `experiments/002-dmi-tmi`, a runtime monitor
   rather than a result-producing step, recorded in the `DEPRECATION.txt` manifest.
   `ontology` is live tooling (`Makefile:5-15` runs `tc_ontology.py` and
   `mermaid_diagram.py`; the `ontology-figure` hook runs `torchcell/paper/ontology_*`
   instead), so the durable fix is a Phase 3 test of its pure functions. `pypy_adapters`
   is classified by `legacy_partition.py` with its sole importer
   `knowledge_graphs/create_pypy_scerevisiae_kg.py` (Decision 22), not deprecated. One PR
   after a user go-ahead, per Decision 8 ("do not delete or prune") of the 06.18 plan.
   Computed, not measured: removing 1,270 zero-hit lines moves 24.3% to 16,521/66,691 =
   24.8% with no test added.
4. **A coverage gate: `fail_under` set once after Phase 1, at the CI-measured `coverage
   report` TOTAL of the behavioral run minus 2 points, ratcheted by hand at Phase 5.**
   Phase 1 lands after Phase 0d, so the floor is set once on the post-move denominator.
   Two points is roughly one large untested loader of slack: the gate blocks a trend, not
   a PR. Rejected: no gate (the iBF precedent is a smaller repo that never sank to 4%) and
   an automatic ratchet (a flaky upward move would block unrelated PRs). TOTAL under
   `branch = true` (`pyproject.toml:220`) counts branch arcs, a different statistic from
   codecov's line figure; the table records both from one run.
5. **The reported number is the CI behavioral run of Decision 17: what codecov shows and
   what `fail_under` and diff-cover block on.** The local number (real `DATA_ROOT`,
   `--slow --data`) and the import-only run are second and third labeled columns.
6. **Markers `data`, `neo4j`, `network`, `wandb` join `gpu`, `slow`; every bucket is
   opt-in by flag; CI runs plain `pytest tests/torchcell` and `-m "not gpu"` is dropped
   from `test.yaml:67`.** `pytest_collection_modifyitems` skips, with a visible reason,
   every marked item whose flag (`--gpu --slow --data --neo4j --network --wandb`) is
   absent. Existing `skipif(not torch.cuda.is_available())` decorators stay.
7. **Makefile targets** (the list is in Relevant Files): `test` is plain pytest, the CI
   contract; `test-ci` runs under `DATA_ROOT=$(mktemp -d)`; `test-fast` adds `-x -q
   --deselect tests/torchcell/test_import_all.py`. `coverage`, `pytest-cov`, `diff-cover`
   are not installed in `~/miniconda3/envs/torchcell`, so `cov*`/`diff-cov` fail with the
   instruction `pip install -r env/requirements_test.txt` and never auto-install.
8. **New tests are typed and docstringed; CI mypy diff-scope extends to `tests/`.**
   Pre-commit already runs strict mypy on `^(torchcell|tests)/` with only the untyped-def
   family relaxed (`pyproject.toml:142-147`) and ruff with D100/D101/D102 live (65-66), so
   every test and conftest carries a module docstring and annotates hooks, fixtures, and
   `monkeypatch: pytest.MonkeyPatch`. CI mypy (`mypy.yaml:57`) scopes to
   `torchcell/**/*.py`, so a type error in a test passes CI and fails only locally;
   [[plan.ci-quality-finish-roadmap.2026.06.25]] WS4 records a real vacuous-assert bug
   strict mypy caught in `tests/torchcell/sequence/`.
9. **Import-all is a parametrized smoke test with two committed lists.** `NEVER_IMPORT`
   holds side-effecting modules excluded from the parameter set (an xfail would still run
   the download or `plt.show()`); `KNOWN_BROKEN` holds importable-but-failing modules as
   strict xfails, so a fix becomes an XPASS failure and the list decays instead of
   rotting. Parametrization attributes a failure to the module; `pytest-timeout` supplies
   the budget. Its coverage feeds only the import-only column (Decision 17). In Phase 0d
   `^torchcell\.legacy\.` joins `NEVER_IMPORT`, removing `trainers.utils` and
   `trainers.regression` from `KNOWN_BROKEN`.
10. **`torchcell.graph.sgd` does not raise at import.** `sgd.py:26` calls `load_dotenv()`
    but `data_root()` (30) resolves lazily; the four test comments claiming otherwise are
    stale, and the real reason those files skip is that `torchcell/scratch/load_batch.py`
    opens real LMDBs. Corrected in Phase 0a.
11. **`torchcell/metabolism/__init__.py` is added in Phase 0a.** coverage.py with `source`
    reports un-executed files, but "only importable files (ones at the root of the tree,
    or in directories with a `__init__.py` file) will be considered"
    (<https://coverage.readthedocs.io/en/latest/source.html>, fetched 2026-09-25). Live,
    tested code invisible to both coverage and `walk_packages` is the bug.
    `torchcell/scratch/` stays without one. The denominator change is stated in the PR.
12. **Assertions by layer.** Exact values wherever a human can compute them: pydantic
    round trips, transforms, losses on a 3-element tensor with the expectation worked in a
    comment, registry invariants, script exit codes and file contents. For a randomly
    initialized network, exact floats are not a contract across torch versions: assert
    output shape, finiteness, a non-None gradient on every parameter, and seeded
    determinism (two forwards under the same `torch.manual_seed` are `torch.equal`).
13. **`pytest-timeout` with `timeout = 300` in `[tool.pytest.ini_options]`.** Import-all
    and the Lightning `fast_dev_run` tests are the first tests that can hang rather than
    fail; the per-test ceiling is what makes Decision 9 safe. `media.py:1783` runs
    `_check_library()` at import, slow but safe, and stays in the parameter set.
14. **`scripts/coverage_gaps.py` is the table generator.** A campaign is not an
    experiment, so the artifacts rule's `experiments/<id>/` location does not apply;
    precedent is the paper table generators. It reads `coverage json` output, emits the
    per-module table sorted live-critical first then ascending, and diffs JSON files into
    the before/after table; `make cov-gaps` wires it.
15. **Sentinel `DATA_ROOT` is the fixed path `/tmp/torchcell-test-data-root`, set by
    `setdefault`.** Never `basetemp`: `scripts/deprecate.sh:24-32` exits 2 for a graveyard
    under `$DATA_ROOT` by prefix match, and `scripts/merge_queue.py:52-75` resolves its DB
    under `$DATA_ROOT` at import, which on the dev box is the LIVE queue. Never
    assignment: the dev shell exports the real root, and `load_dotenv()` (override=False)
    in five modules cannot undo a prior set. A session-end autouse check, active only when
    `DATA_ROOT` equals the sentinel, asserts the directory is still empty.
16. **Goldens pin `pydantic.VERSION` in a sidecar and skip with a reason on mismatch.**
    Only pydantic JSON and transform outputs qualify; tensor goldens do not. Phase 1 at
    most, low priority.
17. **The reported statistic is behavioral coverage: the CI run with
    `tests/torchcell/test_import_all.py` deselected.** Import-all runs in a second CI step
    (`coverage run --data-file=.coverage.import -m pytest tests/torchcell/test_import_all.py`,
    then `coverage json --data-file=.coverage.import -o coverage-import.json`) and
    `coverage_gaps.py --import-only coverage-import.json` reports it as a separate
    "import-only" column, so a reader sees how much of a module's total is mere def/class
    execution. This supersedes the Phase 0b sentence that let import-all raise the
    reported number by 4 to 7 points: that gain appears only in the import-only column.
18. **A quality audit before every test PR is enqueued.** An independent reviewer pass (a
    fresh agent, or the author using the rubric) reads every new test file against the iBF
    test-campaign rules (`.claude/skills/test-campaign/SKILL.md` in iBioFoundry-AI, "Test
    quality rules" 62-74, "Quality Audit" 109-133) and torchcell's Evidence Discipline:
    assert exact values, not truthiness; expectations computed by hand in a comment; no
    test that only checks pydantic parses a field; mock only at system boundaries; test
    through the public entry point; every test names the behavior it pins; and the
    decisive question, a test that would pass against a stubbed-out function is rejected.
    Output: a dated `### Quality audit` in the campaign note listing tests rejected or
    rewritten, with counts. The mechanical checks (Decision 19) catch the cheap cheats;
    this audit catches the subtle ones (a wrong exact value, an assertion on a mock).
19. **Mechanical anti-padding lint: `scripts/test_quality_check.py` (stdlib + `ast`).**
    Fails on a `test_*` function with zero `assert` statements and no `pytest.raises`; a
    `parametrize`d function whose only assertions are `isinstance(...)` or bare truthiness
    (`assert x`, `assert x is not None`, `assert len(x) > 0`); a `test_*` function that
    calls a name imported from `torchcell` and never asserts on a value derived from that
    call (simple name binding, so conservative and cheap). Wired to `make test-quality`, a
    pre-commit hook on `^tests/`, and a CI step. It cannot judge whether an exact value is
    the right one.
20. **Live-critical column.** `coverage_gaps.py` builds one importer graph (ast-parsed
    imports over `torchcell/`, `tests/`, `scripts/`, `database/`, experiments >= 016) and
    marks a module live-critical when an experiment >= 016 or live `torchcell/` code
    reaches it, never from a hand list; the table sorts live-critical first so effort
    visibly goes to modules that matter. `legacy_partition.py` inverts the same graph.
21. **Diff-coverage gate in `test.yaml`, the diff-scoped mypy gate's shape** (`mypy.yaml:
    46-63`: `origin/main...HEAD`, changed `torchcell/**/*.py`, exceptions in pyproject).
    After `coverage xml`: `diff-cover coverage.xml --compare-branch=origin/main
    --fail-under=80 --exclude 'torchcell/legacy/*' 'torchcell/scratch/*'
    'torchcell/experiments/*'`; 80 is a starting policy, revisited at Phase 5. `diff-cover`
    10.6.0, uploaded 2026-09-22, Python >= 3.10 (<https://pypi.org/pypi/diff-cover/json>,
    fetched 2026-09-25); it consumes "Cobertura, Clover or JaCoCo XML format, or LCov" (what
    `coverage xml` writes) and `--fail-under` returns "a non zero status code if the report
    quality/coverage percentage is below a certain threshold"
    (<https://raw.githubusercontent.com/Bachmann1234/diff_cover/main/README.rst>, fetched
    2026-09-25). Pin `diff-cover>=10.6,<11`; not installed locally. The merge base:
    `test.yaml:24` checks out at depth 1, so `origin/main...HEAD` would scope to the wrong
    file set (the failure `mypy.yaml:48-52` documents); `fetch-depth: 0` on the checkout
    plus `git fetch origin main || true` before the step, copied from `mypy.yaml:18-20,52`.
22. **Paired-test gate, one exception table, and the legacy move (hard mode).**
    `scripts/check_paired_tests.py` (stdlib; shells to `git diff --name-only
    --diff-filter=A origin/main...HEAD -- 'torchcell/**/*.py'`): every ADDED module needs
    `tests/torchcell/<same relative dir>/test_<name>.py`; `--strict` uses `--diff-filter=AM`
    (every changed module); `__init__.py`/`__main__.py` are exempt. ADDED-only until PR-0d:
    strict mode would fire on any edit to a live module whose mirrored test file the
    campaign has not yet written; it flips on in PR-0d, once the modules that could never
    earn a test have moved and the exception table is three prefixes. Exceptions come from
    ONE place, `[tool.torchcell.test_exceptions]` `paths = [...]` in `pyproject.toml`, read
    with stdlib `tomllib`: pyproject already holds the mypy and ruff carve-outs, and a
    separate file would be a third exception mechanism where the user's remark ("I think
    we have this with exceptions", the diff-scoped mypy gate) asks for one. In Phase 0a the
    table is seeded verbatim from those carve-outs (pyproject 55, 103-112, 133-140, 201):
    three directory prefixes (`scratch/`, `experiments/`, `pypy_adapters/`), two globs
    (`*_DEPRECATED.py`, `*_scratch.py`), and seven named files (`models/graph_attention.py`,
    `trainers/utils.py`, `delete_subset.py`, ...). The move: `scripts/legacy_partition.py`
    computes the closed legacy cluster from the importer graph; roots are experiments
    >= 016, `tests/`, `scripts/`, `database/`, `Makefile`, `.pre-commit-config.yaml`, and
    `[project.scripts]`; a module is legacy when no root reaches it through imports. The
    user's nominated list (the dcell family except `dcell.py`/`DCellSubsystem`,
    `hetero_cell_bipartite_dango*` and the other pooling and latent-perturbation models,
    the `fit_int_*`, `int_hetero_cell*`, `int_dcell`, `int_dango` and `regression.py`-family
    trainers, `pypy_adapters`, `metrics`, `profilers`, viz modules used only by them) is the
    expected output, never the input; the script's output is the artifact the move PR
    cites. A package re-export (`models/__init__.py` re-exports
    `graph_attention`/`graph_convolution` for `experiments/DEPRECATED_costanzo_smf_dmf_supervised`
    alone) makes a module look live, so the script reports "reached only via `__init__`
    re-export" as its own category and the move PR drops those re-exports. Modules move by
    `git mv` to `torchcell/legacy/<original subpath>` with NO import shims, so an
    experiment >= 016 still importing a moved module would break; those experiments are
    roots, so a module they import is live by construction, and `--check` re-runs that
    proof after the move and also fails on any root importing `torchcell.legacy`.
    Reproducibility follows [[plan.ci-quality-finish-roadmap.2026.06.25]] WS5 ("old
    experiments are refactored only behind preserved git tags"): an annotated tag
    `legacy-pre-move-2026.09` is cut on main immediately before the move commit and
    recorded in the PR; frozen experiments <= 015 are not roots and rerun from it. Then
    every exception collapses to one prefix: coverage `omit = ["torchcell/legacy/*"]` in
    `[tool.coverage.run]`, mypy `exclude` plus one `ignore_errors` override for
    `torchcell.legacy.*`, ruff `extend-exclude`, diff-cover `--exclude`, and
    `test_exceptions.paths` becomes `torchcell/legacy/`, `torchcell/scratch/`,
    `torchcell/experiments/` (the last two pre-exist as directory carve-outs under
    Decision 11 and [[plan.pytest-ci-blocking.2026.07.01]] Decision 5). Hard mode: no
    per-module exception anywhere in the live tree; `fail_under`, diff-cover, and the
    strict paired-test check apply to all of it. Every named-file carve-out in the seeded
    table is either proven legacy (moves) or proven live (loses its exception in the same
    PR); a third outcome is not allowed.

## Approach

Denominator: 67,961 lines (codecov, 2026-09-25). Every delta below is a Hypothesis
(untested), reasoned from wc line counts of the targeted modules, which overstate coverage
statements; the campaign note replaces each with the generated table. Every test PR
(1 through 4) ends with `make test-quality` and the Decision 18 audit before
`/enqueue-merge`; the audit section lands in the campaign note in the same PR.

**Phase 0a, infrastructure and gates (PR-0a).** `tests/conftest.py` opens with `import os`
and `os.environ.setdefault("DATA_ROOT", "/tmp/torchcell-test-data-root")` before `import
pytest` (`# noqa: E402`); the ordering is the whole point. Then the Decision 6 flags and
skip hook, and autouse guards that fail loudly on `sbatch`/`gh`/`ssh` in `subprocess.run`,
`neo4j.GraphDatabase.driver`, `wandb.init`, and
`socket.getaddrinfo`/`urllib.request.urlopen`/`requests.Session.request` (never
`socket.socket`, which breaks torch.distributed and wandb offline mode), each disabled by
the matching flag; `WANDB_MODE=disabled` and `MPLBACKEND=Agg` set in the same autouse.
`tests/torchcell/conftest.py` lifts the CGT builders (`GENE_NUM=8, HIDDEN=16,
NUM_LAYERS=2, NUM_HEADS=4, BATCH_SIZE=3, NUM_REACTIONS=4, NUM_METABOLITES=3`) into
`cell_graph` and `batch` fixtures, adds `_FakeTxn`, and the DCell fixture. The eight
`is None` gates become `os.path.isdir(<mirror path>)` gates so a sentinel root skips
instead of failing (Gotcha 2); the `test_hetero_cell_nsa.py:26` env write is deleted; the
four stale `sgd` comments are corrected (Decision 10). The gates of Decisions 19, 21 and
22 land here, the paired-test check in ADDED-only mode; the remaining MODIFY rows of
Relevant Files complete it. First because five open branches touch `tests/` or
`pyproject.toml` and rebase over it once. Expected delta: about 0.

**Phase 0b, import-all and campaign tooling (PR-0b).** `tests/torchcell/test_import_all.py`
walks `torchcell` with `pkgutil.walk_packages`, runs under `monkeypatch.chdir(tmp_path)`
(Gotcha 1), and asserts each module imports. `NEVER_IMPORT`:
`^torchcell\.(scratch|experiments|pypy_adapters)\.`, `models.species_aware_lm`
(HuggingFace download at 67-74), `go.check_deprecated` (cwd-relative `GODag` at 6),
`ncbi.sequence_scratch` (cwd `read_csv` at 22), `nn.sort_adj_block_model` (26),
`nn.flex_attention_graph_adj` (22, 40-43), `nn.flex_attention_graph` (70) (dataset build
and `plt.show()` at import), `graph.metabolism` (builds and prints at 50),
`knowledge_graphs.(create_|gene_interactions_|smf_)` (SSL env mutation and cwd log files).
`KNOWN_BROKEN` (strict xfail): `datasets.scerevisiae.mechanisitc_aware` (`rpy2` absent),
`trainers.utils` (pydantic-v1 `ConstrainedStr`), `trainers.regression`
(`WeightedMSELoss`). A second test scans source for hard-coded `~/Documents/projects`
paths (`datasets/scerevisiae/spell.py:20-25` is the known hit and seeds its allowlist).
`test.yaml:67` becomes the behavioral run (`coverage run -m pytest tests/torchcell
--deselect tests/torchcell/test_import_all.py`) followed by the import-only run
(Decision 17). `scripts/coverage_gaps.py` and the campaign note land here with the Phase 0
table. Expected behavioral delta: about 0; Decision 17 moved the earlier +4 to +7 into the
import-only column, where it stays (models and trainers plus the 0% packages are mostly
never imported, and an import executes def/class/decorator/import lines, typically 15-25%
of a module's statements).

**Phase 0c, deprecations (PR-0c, after go-ahead).** `bash scripts/deprecate.sh
torchcell/yeastmine "zero importers; test-suite campaign 2026.09.25"` then the same for
`cpu_benchmark_system_monitor.py`, one target per call, `git rm -r --cached` in the same
commit, and the 9 dendron wikilinks retargeted to the DEPRECATION manifest path.
Expected delta: +0.5 points computed (Decision 3), zero from tests.

**Phase 0d, the legacy move (PR-0d, after the user confirms the computed list, Open
Question 1).** `python scripts/legacy_partition.py --output
notes/assets/test-campaign/legacy_partition.json --table` emits the cluster (module, wc
lines, last commit, reachability proof, `__init__`-re-export category) for the campaign
note and the PR. Then, in order: the annotated tag `legacy-pre-move-2026.09` on `main` at
the commit the move is rebased onto; `git mv` of every listed module to
`torchcell/legacy/<original subpath>` with `__init__.py` files; the `__init__` re-export
edits the script named; the pyproject collapse of Decision 22; the `NEVER_IMPORT` and
`KNOWN_BROKEN` edits of Decision 9; `check_paired_tests.py` to `--strict` in `test.yaml`
and the pre-commit hook. Acceptance: `python scripts/legacy_partition.py --check` exits 0
(Decision 22) and `python -m pytest tests/torchcell -x -q` passes with unchanged skip counts.
Expected delta, computed from wc counts and not measured: the nominated list is at least
the 10,237 zero-importer model and trainer lines from the old Open Question 1 plus the
Dango cluster, `pypy_adapters` 1,219, `metrics` 465, `profilers` 330, roughly 12,000 to
16,000 lines almost all at 0%, so the denominator falls from 66,691 to about 51,000 to
55,000 and the reported number rises from 24.8% to about 30 to 32% with no test added. The
script output replaces this range; the new baseline is measured once, after the move, as
the "Phase 0d after" column every later phase diffs against.

**Phase 1, contracts and invariants (PR-1).** `test_datamodels_roundtrip.py`: walk
`torchcell.datamodels.*` with `inspect.getmembers`, collect every concrete `BaseModel`
subclass, build an instance from each class's own example or a minimal valid dict kept in
the test, assert the JSON round trip is equal and `model_json_schema()` is stable; classes
without a constructible example are listed with a reason, never skipped silently.
`test_dataset_registry.py`: for each of the 38 `dataset_registry` entries, construct via
`cls.__new__(cls)` (never `__init__`, which runs `process()` and downloads on an empty
root) and assert the `ExperimentDataset` subclass, `experiment_class`/`reference_class`
types, non-empty `raw_file_names`, key equals `cls.__name__`, and
`datasets/scerevisiae/__init__.py` `__all__` equals the key set; a property that reads
`self.root` fails the sweep, which is a finding. `test_hetero_to_dense*.py`: a 3-node
graph with a hand-written adjacency, exact dense tensor and mask. Nine loss files
(`test_{mle_wasserstein,point_dist_graph_reg,dango,isomorphic_cell_loss,diffusion_loss,supcr,losses_dcell,list_mle,logcosh}.py`):
each loss on a 3-element prediction/target with the closed-form value in a comment,
NaN-mask behavior where the loss supports it, and `loss.backward()` producing finite
gradients. `nn/aggr/test_set_transformer.py`: PMA/ISAB output shapes, seeded determinism.
Expected delta: +2 to +3 points on the post-move denominator (losses: 2,027 wc lines
untested of 5,670; transforms 325; set_transformer 168).

**Phase 1b, the gate (PR-1b).** Read TOTAL from the PR-1 CI log's behavioral `coverage
report`, set `fail_under = TOTAL - 2` in `[tool.coverage.report]`, one-line PR.

**Phase 2, live model surface and DCell (PR-2).** In the existing CGT test file: units for
the tensor-only components `GraphRegularizedTransformerLayer` (34), `HyperSAGNN` (186),
`EquivariantPerturbationTransform` (360), `PerturbationGraphPropagation` (741),
`LowRankBilinear` (881), `ObservedLabelEncoder` (941), `CrossGeneMixing` (1021),
`ResponseBasisHead` (1077), `PerceiverMixing` (1145), the five heads (1255-1735),
`calculate_weight_l2_norm` (2986), `compute_smoothness` (2995); full `forward` then
backward, determinism, `return_attention=True` shapes (Decision 12).
`trainers/test_int_transformer_cell.py`: `RegressionTask(model, cell_graph, ...,
device="cpu")` with every `plot_*_every_n_epochs=0` under `L.Trainer(fast_dev_run=True,
logger=False, enable_checkpointing=False, accelerator="cpu", default_root_dir=tmp_path)`
on a two-batch `DataLoader` of the fixture; `wandb.log` monkeypatched to a recorder that
the test asserts was not called (Gotcha 3). Lightning 2.5.5 semantics: `fast_dev_run`
"runs n ... else 1 batch(es)", "disables all loggers", and checkpoint/early-stop callbacks
do not trigger
(<https://raw.githubusercontent.com/Lightning-AI/pytorch-lightning/2.5.5/docs/source-pytorch/common/trainer.rst>,
fetched 2026-09-25). `trainers/test_neo_regression.py`: same shape for
`neo_regression.RegressionTask`. `models/test_dcell.py` (today a one-line placeholder):
the conftest fixture (root term 0, two children at stratum 1, 4 genes,
`go_gene_strata_state` as `[N,4]` of go_idx/gene_idx/stratum/state, edge
`("gene_ontology","is_child_of","gene_ontology")`); assert subsystem count and the
per-term output sizes from `min_subsystem_size`/`subsystem_ratio` arithmetic (exact,
hand-computed), forward shape `[2, 1]`, backward, determinism; `DCellSubsystem` (391) as a
pure unit with exact output size. Expected delta: +5 to +7 points on the post-move
denominator (CGT 3,677 lines to roughly 70%, `int_transformer_cell` 1,576 with its step
methods executed, `neo_regression` 395, `dcell` 841 from 0).

**Phase 3, kg/adapters/graph/sequence/data (PR-3).** `graph/test_graph.py`: `GeneGraph`
(`graph.py:36`) and `GeneMultiGraph` (74) pydantic construction, key iteration, validation
errors; the `pytestmark` at 21-29 moves onto the data tests as `@pytest.mark.data`.
`graph/validation/test_locus.py`, `test_raw_structure.py`: each validator on a minimal
valid dict and one invalid dict with `pytest.raises(..., match=)`.
`data/test_graph_processor.py`: the processors on the shared fixture, exact index tensors
for a 2-perturbation batch (the equivalence test stays gated). `data/test_cell_data.py`:
pure tensor helpers above the gate. `data/test_mean_experiment_deduplicate.py`: three
duplicate records with values 1, 2, 6 reduce to mean 3 and the recorded `n`.
`ontology/test_tc_ontology.py`: schema-building functions on a two-class ontology dict,
`chdir(tmp_path)` for biocypher; fixes the false "untested" claim at `tc_ontology.py:4`.
Adapters: extend the existing per-adapter files with a `tmp_path` CSV through
`create_experiment` and the adapter's node/edge generators, asserting exact node ids and
counts; `chdir(tmp_path)` in every adapter test. Sequence stays data-gated except
`registry.py` path resolution on a tmp registry. Expected delta: +4 to +6 points on the
post-move denominator (graph.py 1,514, validation 536, graph_processor 2,700, cell_data
566, dedup 499, ontology 700, adapters 35.8% of a large package).

**Phase 4, hermetic script tests (PR-4).** Each test copies `test_wt_cleanup.py:24-52`: a
bare `origin` under `tmp_path`, a clone, `git -C`. `test_merge_queue.py`: subprocess with
`env={"DATA_ROOT": str(tmp), "PATH": <no gh>}` and `--db tmp_path/q.db` (lock and
heartbeat co-locate with the DB, `merge_queue.py:200-213`); enqueue, list, reorder, exact
row order and exit codes. `test_drain_merge_queue.py`: `--main tmp_main --db tmp_db`,
`_cleanup_remote` stubbed, `shutil.which` returning `None`, `SLACK_CLAUDE_WEBHOOK` and
the four `GIT_*` vars deleted, `chdir(tmp_main)`, and an autouse assertion that every
repo's `origin` is under `tmp_path` (Gotcha 9). `test_setup_worktree.py`:
`PATH=/usr/bin:/bin`, `HOME=tmp`, a `.env` in the tmp main; assert the files, the `data`
symlink, and the `merge.weeklynote.*` config keys. `test_deprecate_sh.py`:
`DEPRECATED_DIR=tmp/graveyard`, `DATA_ROOT` outside the graveyard's ancestry; assert the
manifest and the move, and exit 2 when the graveyard sits under `DATA_ROOT`.
`test_ops_sh.py`: `bash scripts/ops.sh bogus` prints usage and exits non-zero; everything
else is `network`. The three campaign scripts get tests here too: `test_quality_check.py`
on a fixture file with one padded and one real test (exact finding list),
`check_paired_tests.py` on a tmp repo with an added module and no test (exit 1, the path
named), `legacy_partition.py` on a tmp package with a two-module closed cluster (exact
cluster). Expected delta: 0 on the reported number (`scripts/` is outside
`source=["torchcell"]`); the value is regression protection for the landing tooling and
the gates.

**Phase 5, ratchet (PR-5).** Final `make cov-gaps` in both environments, the table into the
campaign note, `fail_under` raised to the new TOTAL minus 2, the diff-cover threshold
revisited against the campaign's observed per-PR diff coverage.

Campaign note template (`notes/test-campaign.2026.09.25.md`, created with `dendron-cli`;
one dated H2 per phase, table pasted verbatim from `scripts/coverage_gaps.py`, never
hand-typed; one `### Quality audit` per test PR):

```markdown
## 2026.09.25 - Phase 0 baseline

Generated by: scripts/coverage_gaps.py --before ci_before.json --after ci_after.json --import-only ci_import.json --local local_after.json

| Module | Live-critical | CI behavioral before | CI behavioral after | Import-only after | Local-with-data after |
|---|---|---|---|---|---|
| (command) | importer graph @ <commit> | `coverage run -m pytest tests/torchcell --deselect tests/torchcell/test_import_all.py` @ <commit> | same @ <commit> | `coverage run -m pytest tests/torchcell/test_import_all.py` @ <commit> | `... --slow --data` @ <commit> |
| torchcell/models/dcell.py | yes | 0.0% | ... | ... | ... |
| TOTAL (coverage report, line+branch) | | ... | ... | ... | ... |

### Quality audit

Reviewer: <agent or author>. Rejected: N (list). Rewritten: M (list). Test count before/after.
```

PR sequence: 0a, 0b, 0c (after go-ahead), 0d (after list confirmation), 1, 1b, 2, 3, 4,
5, each its own worktree and PR, landed in that order through `/enqueue-merge`. Out of
scope: `experiments/` (Decision 5 of [[plan.pytest-ci-blocking.2026.07.01]], never in a
gate), whole-tree mypy (WS4), `pytest-xdist` (LMDB envs must not be opened before fork;
revisit after the campaign).

## Gotchas

1. **biocypher creates `biocypher-log/` in cwd** (`biocypher/_logger.py:63-99`) for 19
   top-level importers including every adapter, and eight `knowledge_graphs/*kg*.py`
   modules `basicConfig` to `biocypher_warnings.log` in cwd. Sidestep:
   `monkeypatch.chdir(tmp_path)` in import-all and every adapter test; `.gitignore:83-85`
   already ignores the names.
2. **`setdefault` versus the `is None` gates.** A sentinel `DATA_ROOT` turns eight
   `os.getenv("DATA_ROOT") is None` gates into failures that open real data
   (`sequence/genome/scerevisiae/test_s288c.py:13-31`, `data/test_cell_data.py:15-34`,
   `models/test_hetero_cell_nsa.py:15-26`, `nn/test_hetero_nsa.py:18-19,327`,
   `nn/test_masked_gin_performance.py:136-137`,
   `datasets/scerevisiae/test_gene_name_reconcile.py:17-28`, `test_kuzmin2018.py:22-26`,
   `test_kuzmin2020.py:22-26`, whose slow builds at 67-171 also get `data`). Sidestep:
   `isdir` gates in PR-0a; the `osp.exists` gates already present in the other 17 files
   degrade to skip on their own.
3. **Trainer tests under `fast_dev_run`.** Trainers call `wandb.log` directly
   (`int_transformer_cell.py:1326,1334`, `regression.py:399,409-416`), outside the logger
   Lightning disables; unpatched it raises "You must call wandb.init()".
   `int_transformer_cell.py:55` defaults `device="cuda"`. `lightning_logs/` lands in cwd
   (`default_root_dir` defaults to cwd) and `.gitignore:26` lists `wandb/` but not it.
   Sidestep: `WANDB_MODE=disabled` plus a monkeypatched recorder, plotting frequencies 0,
   `device="cpu"` with `accelerator="cpu"`, `default_root_dir=tmp_path`, and the
   `.gitignore` line in PR-0a.
4. **Basename collisions under prepend import mode.** Only `tests/torchcell/utils/` and
   `tests/torchcell/paper/` have `__init__.py`; a second `test_dcell.py` under `losses/`
   gives "import file mismatch". Sidestep: unique basenames (`test_losses_dcell.py`), no
   new `__init__.py`; the paired-test check accepts the alternative basename through the
   exception table, never by dropping the test.
5. **`schema-impact` (`.pre-commit-config.yaml:43-49`) blocks a commit touching
   `datamodels/{schema,pydant}.py` without `TORCHCELL_SCHEMA_ACK=1`; `ontology-figure`
   (59-67) regenerates SVGs and fails the commit when those files or `utils/utils.py` are
   staged.** Sidestep: the round-trip sweep needs no edit to either file; the deprecated
   `class Config` in `pydant.py:13-28` is a separate PR with the ACK.
6. **semantic-release bumps on `TST`/`TEST` commit tags** (`pyproject.toml:262-281`),
   pushing a version commit to main after each landing. Not a stray commit;
   `drain_merge_queue.py:27-31` already re-rebases the next branch.
7. **biocypher local 0.5.43 versus pinned 0.15.2.** Adapters import the private
   `biocypher._create` (`adapters/cell_adapter.py:20`); a local pass is not a CI pass.
   Sidestep: read the CI run for adapter tests, not the local run.
8. **ruff `UP007`/`UP045` rewrite `Optional[...]` in any new `MessagePassing` subclass**
   outside the two per-file-ignored modules (`pyproject.toml:69-74`), and PyG's runtime
   signature inspection breaks on PEP 604 unions. Sidestep: no `MessagePassing` subclass
   in fixtures.
9. **`drain_merge_queue.py` reaches `gh` (`_cleanup_remote`, 252-275, on the hardcoded
   slug at 58-60), Slack (`_slack`, 124-132), and pushes `HEAD:main` in the worktree it
   resolves (`land_branch`, 206-237); a polluted test once force-pushed temp history to
   the real main, which is why `_strip_inherited_git_env` (340-353) exists.** Sidestep:
   the Phase 4 env, stubs, and origin-under-`tmp_path` assertion.

## Verification

Every `python` and `make` below runs as `~/miniconda3/envs/torchcell/bin/python` with
`PYTHONPATH` set to this worktree, so its code is imported rather than the primary
checkout's.

- CI contract: `python -m pytest tests/torchcell -x -q --deselect
  tests/torchcell/test_import_all.py` (behavioral) and `make test-import` (smoke). The full
  suite reproduced two ways with the same skip counts: `env -u DATA_ROOT python -m pytest
  tests/torchcell -x -q` (the conftest sets the sentinel) and the same under
  `DATA_ROOT=$(mktemp -d)` (an empty real root). Never `DATA_ROOT=""`: empty-but-set
  bypasses the guards (the 2026.09.19 section of [[plan.pytest-ci-blocking.2026.07.01]]).
  Afterwards `ls -A /tmp/torchcell-test-data-root` is empty.
- Local column: `python -m pytest tests/torchcell --slow --data -q`.
- Table: `make cov-gaps` (behavioral and import-only `coverage run`, `coverage json` for
  each, `scripts/coverage_gaps.py --import-only`); paste the output into the campaign note.
  `coverage report | tail -1` shows the same statement total from PR-1 through PR-5 as the
  PR-0d run (only PR-0a, PR-0c and PR-0d may move it).
- Gates, locally before enqueue: `make test-quality` exits 0 on `tests/` (and exit 1 naming
  a scratch file seeded with one padded test, then deleted); `coverage xml` then the
  Decision 21 `diff-cover` command against `origin/main`, its number into the PR
  description; `python scripts/check_paired_tests.py --base origin/main` (PR-0a through
  PR-0c) and `--strict` (PR-0d onward) exit 0; `python scripts/legacy_partition.py --table`
  equals the list the user confirmed before PR-0d and `--check` exits 0 after it.
- Import-all timing: `--durations=10` on `test_import_all.py`; the slowest modules go
  into the PR-0b description.
- Audit: a `### Quality audit` section exists in the campaign note for every test PR
  before it is enqueued; a PR without one is not ready.
- After each landing: `gh run list --branch main --limit 5` green before enqueuing the
  next PR; the runner has been red on unrelated causes before, so read the failing step.

## Open Questions

1. **Needs the user before PR-0d: confirm the computed move list.** `python
   scripts/legacy_partition.py --table` (the script lands in PR-0a) prints the closed
   legacy cluster with module, wc lines, last commit, and the reachability proof; the user
   confirms or names a module to keep live, and a kept module gets a test in its package's
   phase rather than an exception. The earlier question (deprecate the 10,237
   zero-importer model and trainer lines under Decision 8 of
   [[plan.ci-foundation-ruff-mypy-pytest.2026.06.18]], "do not delete or prune", versus
   the WS7 precedent in [[plan.ci-quality-finish-roadmap.2026.06.25]]) is subsumed by
   Decision 22: relocation without pruning, behind the tag. Two entries the script must
   settle: `pypy_adapters` moves only if `knowledge_graphs/create_pypy_scerevisiae_kg.py`
   is reached by no root (a `database/` slurm launcher would make it live), and
   `models/graph_attention.py`/`graph_convolution.py` move only with the
   `models/__init__.py` re-export edit.
