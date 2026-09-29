---
id: gm6l741bmrr9y5rp3kt7ww3
title: Check
desc: ''
updated: 1790725769580
created: 1790725769580
---

## 2026.09.29 - Drift check against committed release snapshots

Plan: [[plan.data-release-program.2026.09.29]], Decisions 3 and 4; issue #468. Lifecycle
and issue flow: [[database.supported-queries]].

`python -m torchcell.knowledge_graphs.supported_queries [--repo-root R] check [--release ID]
[--json]` reads the registry, the newest snapshot under `database/releases/` (or the named
one) and its `.closures.json`, the checkout's schema surface (`schema.py` + `pydant.py`,
parsed with `ast`), and `biocypher/config/torchcell_schema_config.yaml` (for `is_a`). The
import closure is pydantic, PyYAML and python-dotenv; no torch, no store, no manifest.

| kind | fires when |
|---|---|
| `file_missing` | the `.cql` is absent |
| `missing_node_label` | a label is neither `Pascal(class)` of a snapshot node class nor `Pascal(ancestor)` of one through the checkout's `is_a` |
| `missing_relationship_type` | a type is not `Pascal(class)` of a snapshot edge class |
| `missing_property` | `Label.prop` where the class lacks `prop`; `id` and `preferred_id` always count; an ancestor-only label (`PhenotypicFeature`) needs `prop` on every class under it |
| `missing_dataset` | a `dataset.id` literal the snapshot does not serve |
| `contract_changed` | a class in `phenotype_classes`, or a surface class it references transitively, has a checkout fingerprint different from a selected dataset's recorded closure; or no selected dataset's closure holds the class |
| `composite_changed` | the recorded `dataset_composite` differs from `composite_sha256` over the selected, served datasets (or the query was never validated) |
| `converter_missing` | the converter's module file is absent or defines no top-level class of that name (ast; nothing is imported) |

Choices worth knowing:

- The snapshot records no `is_a`, so the hierarchy comes from the checkout's schema config.
  A concrete class label wins over an ancestor label of the same name: `Genotype` is both a
  class and the `is_a` parent of `perturbation`, and a read through `(g:Genotype)` is
  checked against the `genotype` class alone.
- `contract_changed` goes beyond the listed phenotype class to its forward closure in the
  surface (restricted to symbols the dataset's closure recorded), because a change to
  `Phenotype` or `ModelStrict` changes how the records serialize as surely as a change to
  `FitnessPhenotype` does.
- `converter_missing` was added to the seven drift kinds of the task: a renamed converter
  breaks every consumer of the query, and the check can see it from source. The converter's
  own behavior has no contract fingerprint, so it is not compared.
- Exit 1 when any `supported` query drifts; `deprecated` queries report and exit 0.
- `validate <id> --release R` runs the same check for one query, refuses on any drift other
  than `composite_changed`, and records `validated_release`, `dataset_composite` and (when
  unset) `since_kg_version`.
- `--json` emits `CheckReport` with, per query, `issue_title` and `issue_body` (the markdown
  the CI job posts); plain output is the markdown report.

Wiring: pre-commit hook `supported-queries` (`scripts/run-supported-queries.sh`, exports
`PYTHONPATH` to the checkout's top level; blocks unless `TORCHCELL_QUERY_DRIFT_ACK=1`) and CI
job `query-drift` in `.github/workflows/docs.yaml`.

Run on 2026.09.29 against `2026.09.21-ab6d8c5d` from the branch checkout: 4 queries, 0
drifted, exit 0.

### Interpreter pin: contract fingerprints depend on the Python patch release

The first CI run of `query-drift` (PR #498, runner Python 3.13.15) reported
`contract_changed` on `Phenotype` for all four queries while the same commit passed
locally (torchcell env, Python 3.13.0). Measured: `ast.unparse` renders
`f'a {", ".join(x)}'`-style nested-quote f-strings as `f'a {', '.join(x)}'` on 3.13.0 and
`f"a {', '.join(x)}"` on 3.13.15, and `schema_deps` hashes the unparsed text of validator
methods, so `Phenotype` (whose `validate_fields` has such an f-string) fingerprints to
`447168409f...` on 3.13.0, the value every committed closure records, and `1d57c3d8...` on
3.13.15. The `query-drift` job and the `test.yaml` job are pinned to `python-version:
"3.13.0"`. The same dependence affects any other fingerprint comparison run on a different
patch release (`scripts/kg_compat_page.py`, `kg_manifest admit`); making fingerprints
independent of `ast.unparse` quoting would change every recorded fingerprint and is a
separate decision.
