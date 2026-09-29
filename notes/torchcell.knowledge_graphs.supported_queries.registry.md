---
id: jljlgjyckqrwtlfgkt2yyqb
title: Registry
desc: ''
updated: 1790725754343
created: 1790725754343
---

## 2026.09.29 - Supported-query registry (T5 of the data release program)

Plan: [[plan.data-release-program.2026.09.29]], Decisions 3 and 4; issue #468. Lifecycle and
issue flow: [[database.supported-queries]].

- `SupportedQuery` (pydantic, `extra="forbid"`): `id`, `title`, `cql_path` (relative to the
  `torchcell/knowledge_graphs` package, e.g. `queries/solid_growth_025.cql`), `converter`
  (dotted class path or None), `phenotype_classes` (sorted), `status`
  (`supported | deprecated`), `since_kg_version`, `deprecated_in`, `validated_release`,
  `dataset_composite`, `docs_page`.
- Validators encode the lifecycle: `deprecated_in` is set exactly when `status` is
  `deprecated`; `validated_release` and `dataset_composite` are recorded together; the
  composite is a sha256 hex digest; `cql_path` is a relative `.cql` path with no `..`.
- `QueryDependencies` is the extractor's output (sorted, unique lists). It is NOT stored in
  `registry.json`: storing it would let the file disagree with the `.cql` it describes, so
  every consumer re-extracts it ([[torchcell.knowledge_graphs.supported_queries.cypher_deps]]).
  Beyond the six fields the plan named it carries `label_properties` (`Label.prop` for each
  read through a labeled variable, which the property check needs) and `parameters`.
- `QueryRegistry` (`schema_version` 1) saves byte-stable: queries sorted by id, fields in
  model order, `indent=2`, one trailing newline. `registry.json` ships as package data.
- The test file is `test_supported_queries_registry.py`, paired in `pyproject.toml`,
  because `tests/torchcell/sequence/genome/test_registry.py` owns the basename under
  pytest's prepend import mode.

Seeded entries, each validated on `2026.09.21-ab6d8c5d` (`since_kg_version` 1.2):

| id | datasets | phenotype classes | converter | `dataset_composite` |
|---|---|---|---|---|
| `amino_acid_betaxanthin` | 3 | `MetabolitePhenotype` | none | `684edad79ca7df16bd4983cd644e2915c6110f3c5f1d6625990e0cc59ca17d2d` |
| `essentiality_smf` | 2 | `FitnessPhenotype`, `GeneEssentialityPhenotype` | `CompositeFitnessConverter` | `9704a2985f8935417ae027176c3eab73d52c65cb6a523fc894e146b02b07ea5f` |
| `expression_proteome_morphology` | 5 | `CalMorphPhenotype`, `MicroarrayExpressionPhenotype`, `ProteinAbundancePhenotype` | none | `ec6ba6c5aa29b25fae70589dc81742f791b5a3895283cd8b2fe07cbe1ea3be91` |
| `solid_growth_025` | 15 | `FitnessPhenotype`, `GeneEssentialityPhenotype`, `GeneInteractionPhenotype`, `SyntheticLethalityPhenotype` | `CompositeFitnessConverter` | `cc00363ea5f72d4f68a6d08e5592085120680ea6c78f8acc050bab947f389714` |

The composites were written by `python -m torchcell.knowledge_graphs.supported_queries
validate <id> --release 2026.09.21-ab6d8c5d`; the `essentiality_smf` value was re-derived by
hand (sha256 of the two sorted `content_sha256` values of `SmfCostanzo2016Dataset` and
`GeneEssentialitySgdDataset`, newline-joined with a trailing newline) and matched.
`essentiality_smf.cql` is owned by the showcase branch `feat/showcase-essentiality-smf`; this
branch carries a byte-identical copy so its own gate passes, and whichever lands second
reconciles the add.
