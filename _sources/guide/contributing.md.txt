# Contributing

## Branches, pull requests and landing

All work, code and notes alike, happens on a branch checked out as a git worktree under
`torchcell.worktrees/<branch>/`, created with `scripts/setup-worktree.sh`, while the
primary checkout stays on `main`. Every branch gets a pull request (`gh pr create`)
before it lands, because a pull request opened after its commits reach `main` shows no
commits. Branches land by rebasing onto `main` and fast-forwarding, never through a merge
commit or the GitHub merge button, and the pull request is then closed with a comment
saying so. Landings go one at a time through the merge queue (`scripts/merge_queue.py`),
since all worktrees share one object store. A landing is complete only when the pull
request is closed and the worktree, the local branch and the remote branch are deleted.

(contributing-tests)=

## Tests

Plain `pytest tests/torchcell` is the hermetic contract CI runs. `tests/conftest.py`
enforces it:

- It sets `DATA_ROOT` to an empty sentinel path (`/tmp/torchcell-test-data-root`) unless
  the shell already exports one, and fails the session if any test writes into it.
- Tests that need something expensive or external carry a marker and are skipped, with
  the reason printed, unless the matching flag is given:

  | Flag | Marker | Unlocks |
  | :-- | :-- | :-- |
  | `--data` | `data` | tests that read the real `$DATA_ROOT` |
  | `--neo4j` | `neo4j` | tests that open a Neo4j driver |
  | `--network` | `network` | tests that reach the network, `gh` or `ssh` |
  | `--gpu` | `gpu` | tests that need a CUDA device |
  | `--slow` | `slow` | full dataset builds from the `$DATA_ROOT` mirrors |
  | `--wandb` | `wandb` | tests that call `wandb.init` |

- An unmarked test that tries to run `sbatch`, `gh` or `ssh`, resolve a hostname, open a
  URL, send an HTTP request, open a Neo4j driver or call `wandb.init` fails with an
  error naming the marker it needs. `sbatch` is refused even with every flag set.

```bash
pytest tests/torchcell                  # the CI contract
pytest tests/torchcell --data --slow    # also run tests that read and build from $DATA_ROOT
make test-ci                            # plain pytest with DATA_ROOT pointed at an empty temp dir
```

Fixtures shared across the test tree (`tests/torchcell/conftest.py`) are built in
memory and need no `DATA_ROOT`. A new module under `torchcell/` ships a test at
`tests/torchcell/<same directory>/test_<name>.py`, unless it is listed in
`[tool.torchcell.test_exceptions]` in `pyproject.toml`.

## Pre-commit hooks

`.pre-commit-config.yaml` defines these hooks (install them with `pre-commit install`):

| Hook | Runs on | Does |
| :-- | :-- | :-- |
| `ruff-check` | `torchcell/`, `tests/torchcell/`, experiments numbered 016 and later | lint with auto-fix |
| `ruff-format` | same | format |
| `markdownlint-cli2` | `notes/*.md` | Markdown lint with auto-fix |
| `mypy` | `torchcell/`, `tests/` | strict type check of the staged files (`scripts/run-mypy.sh`) |
| `test-quality` | `tests/**/test_*.py` | rejects tests with no assertion or truthiness-only checks (`scripts/test_quality_check.py`) |
| `paired-tests` | `torchcell/**/*.py` | requires a test file for each new module (`scripts/check_paired_tests.py`) |
| `schema-impact` | `torchcell/datamodels/schema.py`, `pydant.py` | reports which dataset loaders a schema change forces to rebuild; blocks a breaking change unless `TORCHCELL_SCHEMA_ACK=1` |
| `ontology-figure` | the schema modules and the ontology renderers | regenerates the schema ontology figure and fails so the change is seen |

A hook that fails leaves the commit unmade; fix the cause and commit again rather than
bypassing the hooks.

## Commit messages and releases

Versions are cut by python-semantic-release on every push to `main`
(`.github/workflows/semantic-release.yaml`) with the repo's own parser,
`scripts/release_parser.py`, which reads the `TAG(scope): subject` form most commits
carry (`FIX(loaders): ...`) as well as `TAG: subject`. A bump is a deliberate act, not a
side effect of landing work: only three tags bump, and every other tag in
`allowed_tags` (`[tool.semantic_release.commit_parser_options]` in `pyproject.toml`)
parses with no bump, so `main` is "latest" between releases.

| Bump | Tags | When |
| :-- | :-- | :-- |
| major | `API` | a public interface changes; also any tag with `!` or a `BREAKING CHANGE:` paragraph |
| minor | `REL` | a source release, cut before a knowledge-graph build so the release names a tagged version |
| patch | `DB` | a database compatibility change: a new release snapshot under `database/releases/`, a supported-query registry change, a schema closure change |
| none | `FEAT`, `ENH`, `DEP`, `DEV`, `REV`, `FIX`, `BUG`, `BLD`, `MAINT`, `PERF`, `DOC`, `DOCS`, `NOTE`, `TST`, `TEST`, `STY`, `CI`, `BENCH` | everything else |

A message whose tag is not in `allowed_tags`, or whose tag is lowercase, does not parse
and does not contribute to a release.

When a push does bump the version, the same workflow builds the wheel and sdist
(`build_command` in `[tool.semantic_release]`), attaches them to the GitHub release
and publishes them to [PyPI](https://pypi.org/project/torchcell/) through trusted
publishing: the `pypi` environment on the job is registered on pypi.org as the
publisher for `semantic-release.yaml`, and PyPI mints a short-lived token from the job's
OIDC identity, so no PyPI secret is stored in the repository. A push with no bumping tag
publishes nothing. The wheel carries the `torchcell` package and its data files
(`[tool.setuptools.package-data]`), not `tests/`, `experiments/` or `notes/`; check a
build locally with `python -m build` followed by `python -m twine check dist/*`.

A release that was tagged but did not reach PyPI (the publish step failed, or the
trusted publisher was not yet registered) is published without a new bump by running the
same workflow by hand on the existing tag: `gh workflow run semantic-release.yaml -f
tag=vX.Y.Z`. The `publish-tag` job checks out the tag, builds it, refuses a build whose
version is not the tag's, replaces the assets on the GitHub release and publishes to
PyPI, so the two always carry the files of one build.

## Adding a dataset page

A dataset page under `docs/source/datasets/<organism>/` follows the section contract on the {doc}`../datasets/index` page: introduction and terms, the draw.io diagram, the record dumps, the data tables and figures with one exploration, the supported query and its results from a named release, the download section, sourced caveats, and the provenance table. The checklist before opening the PR:

- a generator script and a query script (with its slurm launcher) under `experiments/034-showcase-datasets/scripts/`, with paired Dendron notes;
- every fragment the page includes written by those scripts into the page's `_generated/` directory, and the query and summary JSON committed under `experiments/034-showcase-datasets/results/`;
- the diagram source under `notes/assets/drawio/` and its SVG export in `_generated/`;
- `docs_page` set on the supported query in `registry.json`, and `python -m torchcell.knowledge_graphs.supported_queries check` reporting no drift;
- the page in the organism's `index.md` toctree, and `sphinx-build -b html docs/source build/docs` adding no warnings.

