---
id: zmoytxh2spxsu13ws1ykv9x
title: Addition_family_census_correction
desc: ''
updated: 1792088211982
created: 1792088211983
---

## 2026.10.15 - The hand-written addition family was missing half the family

`experiments/036-dataset-fixes-before-kg-build/scripts/addition_family_census_correction.py`
measures what `ADDITION_PERTURBATION_TYPES` was costing the construct-placement census.
Lane R found the defect and left the fix to lane T, which owns the census. Results:
`experiments/036-dataset-fixes-before-kg-build/results/addition_family_census_correction.json`.

### The family

`torchcell/verification/construct_placement.py` named the addition leaves as the literal
pair `{"gene_addition", "heterologous_pathway"}`. The union has **38 genotype leaves** and
**4** of them subclass `GeneAdditionPerturbation`:

| leaf | `perturbation_type` | in the old literal? |
|---|---|---|
| `GeneAdditionPerturbation` | `gene_addition` | yes |
| `HeterologousPathwayPerturbation` | `heterologous_pathway` | yes |
| `PromoterDrivenGeneIntegrationPerturbation` | `promoter_driven_gene_integration` | **no** |
| `VariablePromoterReporterPerturbation` | `variable_promoter_reporter` | **no** |

Two independent properties of the class give the same four, measured
(`two_derivations_agree: true`): subclassing `GeneAdditionPerturbation`, and declaring
`placement`. The fix derives the family from the first and a test asserts it equals the
second, so a future leaf that adds DNA without a placement, or carries a placement
without adding DNA, fails the test instead of quietly changing what the row counts.

### The census, before and after, on real bytes

| store | additions the OLD family saw | additions the DERIVED family sees | skipped |
|---|---|---|---|
| `pooled_promoter_expression_vaishnav2022` | **0** | **71,158** | 71,158 |

The old row's message for that store was "no gene additions in this store", for a store
every one of whose 71,158 records is an addition. That is the worst failure mode a census
has: not a wrong number but a confident zero.

Corrected census on those same bytes: 71,158 additions, 0 `sequence_backed`, 0
`host_placed_without_coordinate`, 0 `composition_only`, **71,158 `n_untyped`**. The
untyped count is a property of THOSE BYTES, not of the loader: the store was built on
2026-10-13, before `GeneAdditionPerturbation.placement` existed, so no record carries
either a placement or a gap on it. The loader as it stands now gaps the field
(`vaishnav2022.placement_gap()`, `not_released_by_primary`, because the reporter plasmid
is an Addgene handle with no released map), so after a rebuild those 71,158 become
`composition_only`. **Expected, not measured: no rebuild has run.**

Cachera's own four Btx-cassette records are `gene_addition` and were always in the family,
so their census is unchanged by this fix: 4 additions, 4 `host_placed_without_coordinate`,
0 `sequence_backed`, 0 untyped
([[experiments.036-dataset-fixes-before-kg-build.scripts.cachera2023_construct_span_parts]]).

### Three stores cannot be read under the current closure at all

`betaxanthin_cachera2023`, `carotenoid_ozaydin2013` and `crispri_mormino2022` fail to
unpickle: `Can't get attribute 'ReleasedGeneKind'`, a class the integrated branch retired
(lane R's `released-identifier-leaf` record). That is not a defect of this measurement, it
is the measurement: those are exactly the three served datasets this branch's
schema-change record names, and it is what `rebuild_scope: full_kg_rebuild` means in
practice.

### Why the default is a shortlist

`--all-stores` reads every record of every built store, and the dev tree holds stores with
millions of records (`crispri_segregant_hale2024` alone is 8.7 million). The default is
`DEFAULT_STORES`, the four stores whose loaders construct an addition leaf, found by
grepping `torchcell/datasets/` for the four class names. `--store` takes any other.
