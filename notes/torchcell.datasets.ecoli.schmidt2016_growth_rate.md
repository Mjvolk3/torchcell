---
id: ulaznh7vxesci468ylyosjd
title: Schmidt2016_growth_rate
desc: ''
updated: 1791435718897
created: 1791435718897
---

## 2026.10.08 - Loader, sourcing and build

Schmidt et al. 2016's only gene-perturbation phenotype, Supplementary Table 24. The loader
is `torchcell/datasets/ecoli/schmidt2016_growth_rate.py`, the class is
`GrowthRateSchmidt2016Dataset` and the test module is
`tests/torchcell/datasets/ecoli/test_schmidt2016_growth_rate.py`. The paper, the citation
key `schmidtQuantitativeConditiondependentEscherichia2016`, the pinned `si2.xlsx` and the
media objects are the ones [[torchcell.datasets.ecoli.schmidt2016]] already records; this
note states only what is new.

### Why a separate module from `schmidt2016.py`

`build_manifest` decides a built store's staleness from the schema closure of the loader
MODULE's own `torchcell.datamodels` imports (`provenance/schema_deps.py:loader_closure`,
which parses `from torchcell.datamodels...` statements out of the module source). Adding
`FitnessPhenotype`, `BacterialDeletionPerturbation` and `BacterialFitnessExperiment` to
`schmidt2016.py` would therefore have marked the already-served `proteome_schmidt2016`
store stale and forced a full knowledge-graph rebuild for a change that does not touch one
of its records. Measured: `python -m torchcell.provenance.build_manifest` reports
`proteome_schmidt2016` as `fresh` both before and after this branch. The pinned artifact,
the condition table, `build_environment` and `check_environments_distinct` are imported
FROM `schmidt2016`, so each is stated once.

### What is stored

| field | value |
|---|---|
| experiment class | `BacterialFitnessExperiment` |
| reference class | `BacterialFitnessExperimentReference` |
| records | 6, one per (deletion strain, medium) |
| references | 2, the wild-type row of each medium |
| stored quantity | growth rate relative to the wild type of the SAME medium |
| genotype | one `BacterialDeletionPerturbation`, `collection="KEIO collection"` |
| `n_samples` | 3, and 2 for the two cells that release two replicates |
| `sample_unit` | `biological_replicate` |
| `fitness_uncertainty_type` | `sample_sd` |
| assembly pin | `ecoli_K12_BW25113_ASM75055v1` / `GCA_000750555.1` |
| consumed file | `$DATA_ROOT/torchcell-raw/<citation key>/data/si2.xlsx` |

The six records, from `preprocess/strains.csv`:

| medium | strain | locus tag | n | rate (h-1) | WT rate (h-1) | fitness | uncertainty | `fitness_se` |
|---|---|---|---|---|---|---|---|---|
| Glucose | ΔrimI | BW25113_4373 | 3 | 0.544500 | 0.592100 | 0.919608 | 0.011563 | 0.014793 |
| Glucose | ΔrimJ | BW25113_1066 | 3 | 0.362033 | 0.592100 | 0.611440 | 0.018011 | 0.013608 |
| Glucose | ΔrimL | BW25113_1427 | 3 | 0.531533 | 0.592100 | 0.897709 | 0.008747 | 0.013841 |
| Acetate | ΔrimI | BW25113_4373 | 3 | 0.314233 | 0.310333 | 1.012567 | 0.148946 | 0.086653 |
| Acetate | ΔrimJ | BW25113_1066 | 2 | 0.141000 | 0.310333 | 0.454350 | 0.036457 | 0.026219 |
| Acetate | ΔrimL | BW25113_1427 | 3 | 0.212133 | 0.310333 | 0.683566 | 0.038226 | 0.023215 |

The wild type's glucose rate rests on two replicates and its acetate rate on three, which
is why the two media's references carry different `n_samples`.

### `FitnessPhenotype` and not a log2 ratio, decided by measurement

`FitnessPhenotype.validate_fitness` clamps a non-positive value to 0.0, so a ratio at or
below zero would be silently destroyed and the honest home would be
`EnvironmentResponsePhenotype` as a log2 ratio instead. Measured on the pinned workbook:
the six ratios run from **0.454350** (ΔrimJ in acetate) to **1.012567** (ΔrimI in
acetate), every one strictly positive, so the clamp never fires and nothing is lost. The
log2 route is therefore not taken. Table S23's per-condition rates, by contrast, DO include
non-positive values (below), which is one reason that table is not loaded.

### The replicate design is back-solved, not assumed

The Methods never name the replicate count behind Table S24's `Stdev`; they describe the
FIT instead ("The growth rate was calculated from at least four consecutive
measurements"). The released table identifies its own design, measured over all eight
(strain, medium) cells:

| identity | result |
|---|---|
| `Average` equals the mean of that cell's non-empty `Replicate` cells | worst absolute deviation **1.11e-16** |
| `Stdev` equals their SAMPLE standard deviation (n-1 denominator) | worst absolute deviation **8.33e-17** |
| `Stdev` equals their POPULATION standard deviation | nearest absolute deviation **9.50e-04**, so it does NOT |

So `n_samples` is the number of non-empty replicate cells and the uncertainty type is
`sample_sd`, both read off the bytes. This is **rule (1) of the range-resolution
discipline, back-solve from another released statistic**, not the conservative-lower-end
fallback: no range had to be resolved, because the statistic names its own n. Two cells
release two replicates rather than three (WT in glucose, ΔrimJ in acetate). The strains
WERE grown in three replicates (Table S21's title, verbatim: "... from WT, ΔrimI, ΔrimJ and
ΔrimL strains grown in glucose medium using label-free quantification from biological
triplicates"; Table S20 lists `Replicate` 1, 2 and 3 for every strain in acetate), but
only two growth rates are released, so `n_samples` is 2 there. That is both the
back-solved denominator and the conservative count. `check_released_statistics` asserts all
three identities at build time and writes them to
`preprocess/released_statistics_check.json`.

### The two uncertainty numbers, each named

`fitness` is a ratio of two independently measured means and the released `Stdev` is a
dispersion of the NUMERATOR in h^-1, so it cannot be stored verbatim against a
dimensionless ratio. Two numbers are stored:

- **`fitness_uncertainty` = `Stdev_strain / mean(WT)`**, with
  `fitness_uncertainty_type=sample_sd` and `n_samples` the strain's released replicate
  count. Exact arithmetic: it is the sample standard deviation of the n released ratio
  observations `{rate_r / mean(WT)}`, so the released statistic's kind and n survive the
  change of units.
- **`fitness_se` = the delta-method standard error of the ratio**,
  `fitness * sqrt((SE_strain/mean_strain)^2 + (SE_WT/mean_WT)^2)` with
  `SE_x = Stdev_x / sqrt(n_x)`. The one assumption is that the strain's and the wild
  type's cultures are independent, which the cultivation protocol makes true (each is its
  own shake flask inoculated from a preculture). It is always at least as large as the
  auto-derived `fitness_uncertainty / sqrt(n)`, which conditions on the wild-type mean as a
  fixed denominator and so understates the spread. `fitness_phenotype` asserts that
  direction per record, so the stored SE can never be the optimistic one; measured, the
  propagated SE is 1.01x to 2.74x the conditioned one.

The reference carries `fitness = 1.0` (the L3 `reference_one` convention) with the wild
type's own relative spread, `Stdev_WT / mean_WT`, as its `sample_sd` over the wild type's
own replicate count.

### Header-versus-values verification: a clean negative for Table S24

This release already carries one proven header fault -- Table S6's coefficient-of-variance
block holds LB's value under the header `Glucose` and glucose's under `LB`, for all 2,058
dataset-2 rows ([[torchcell.datasets.ecoli.schmidt2016]], asserted by
`check_cv_header_swap`) -- so Table S24's `Glucose:` and `Acetate:` block labels were
checked against an independent released statistic rather than trusted. Table S23 publishes
its own BW25113 growth rate per condition, and the two agree:

| Table S24 block | its WT average (h-1) | Table S23 glucose, 0.58 | Table S23 acetate, 0.30 |
|---|---|---|---|
| `Glucose:` | 0.592100 | **0.0209** relative distance | 0.9737 |
| `Acetate:` | 0.310333 | 0.4649 | **0.0344** relative distance |

Each block is an order of magnitude nearer its own medium than the other, so **Table S24's
medium headers are correct: a clean negative, no swap**. `check_medium_headers` asserts
that each block stays nearest its own medium and within `MEDIUM_HEADER_RTOL` of it, so a
future re-export cannot swap the pair silently. The sheet's five statistic headers are
matched exactly too, so a re-export that adds or renames a replicate column stops the
build rather than reading a statistic out of the wrong column.

### Rank 13, Table S23: the block is CONFIRMED, and two more release defects

The E. coli SI audit ranks Table S23's per-condition growth rate as loadable-now-but-blocked
and warns that its `Stdev`'s replicate design is not stated
(`notes/plan.bacteria-si-phenotype-audit-ecoli.md:207`). It was read in full here and **the
block is confirmed, for three independent reasons**, none of which this branch can lift:

1. **Gap 1 stands.** The quantity is an ABSOLUTE growth rate in h^-1 and `MeasurementType`
   has no member for an absolute growth readout; the environment-response verifier's L3
   `reference_zero` rule cannot be satisfied by an absolute rate either. That is gap 1 of
   the audit's twelve, which blocks seven of its fourteen papers, and closing it is a
   schema change this branch does not make.
2. **The log2 fallback is blocked too, on its own rows.** Measured: 2 of the 26
   strain-condition rows release a NEGATIVE growth rate -- `Stationary phase 1 day` and
   `Stationary phase 3 days`, both -0.01 h^-1. A log2 ratio of a negative rate does not
   exist and `FitnessPhenotype` would clamp both to 0.0, so the route that rescues Table
   S24 does not rescue Table S23.
3. **The `Stdev`'s replicate design is NOT back-solvable.** The audit's caveat is
   confirmed by measurement, not repeated: Table S23 releases ten per-condition columns
   (`Growth rate (h-1)`, `Stdev`, `Single cell volume [fl]`, `Doubling time (h-1)`, `Time
   exp before harvest (h)`, `# of doublings ...`, three unlabelled `OD @ harvesting`
   replicates and `Number of Proteins Identified`) and **not one of them is a replicate set
   of growth rates** -- the three replicate cells are optical densities at harvest, not
   rates. The back-solve that works for Table S24 has nothing to work on here, so
   `n_samples` would be a typed gap. Loading it is therefore not merely gated on gap 1.

Two further released defects were found while reading it, and both are recorded here
because a future Table S23 or Table S9 loader would hit them:

- **The `Doubling time (h-1)` header's unit is wrong.** The column is a doubling time in
  HOURS. Measured on all 23 rows with a positive rate: the released value equals
  `ln(2) / rate` to a worst absolute deviation of **0.0496 h**, which is inside the 0.05 h
  rounding of a one-decimal column, while `rate / ln(2)` is off by factors of 5 to 10 (LB:
  released 0.4, `ln2/1.9 = 0.3648`, `1.9/ln2 = 2.7411`). A loader keying on the header's
  `h-1` would store a reciprocal.
- **Table S23 spells the strain `MG1665`.** Measured: its 26 rows are 22 BW25113, 2
  `MG1665` and 2 NCM3722, while Table S25 spells the same strain `MG1655` on 6 samples and
  the Methods say "MG1655". `MG1665` is a typo and no assembly is deposited under it, so a
  loader of Table S23 or of Table S9's MG1655 arms must not key on the released string.

### Sourcing table

Every quote is a verbatim substring of the pinned bytes. `paper.md` carries sha256
`67bedae8f934421086c23b7fa9582b0950a07206bffe7c4ffdd11e4d003a8710` and `si2.xlsx` carries
sha256 `3280a13ff67a73f25440cff6ee73fb99b5ce3ef57854213dbbf6272be241912f`; the `si2.xlsx`
quotes are row renderings of the `CONTENT_AND_ABBREVIATIONS` sheet (a row's non-empty cells
joined by " | "), the convention this family already uses.

| value | source | section | quote (abridged where long) |
|---|---|---|---|
| the three strains and their collection | paper.md | Online Methods, *Strains and plasmids* | "Mutant strains with either the rimL, rimJ or rimI gene deleted were taken from the KEIO collection19. Correctness of the deletions were checked by PCR." |
| how a growth rate was obtained | paper.md | Online Methods, *Determination of cell counts and growth rates* | "The growth rate of the cultures was determined from the cell counts over time at cell concentrations from ... The growth rate was calculated from at least four consecutive measurements." |
| the media ARE Table S24's conditions | paper.md | Online Methods, *Media* (closing sentence) | "An overview about the used growth conditions can be found in Supplementary Tables 23 and 24." |
| 37 C, shaken, aerobic | paper.md | Online Methods, *Cultivation* | "For the batch cultures, ... grown at 37 °C, orbital shaking at 300 r.p.m. and 5-cm shaking diameter (ISF-4-V, Kühner)." |
| glucose 5 g/L and sodium acetate 3.5 g/L | paper.md | Online Methods, *Media* | "The following carbon sources and concentrations were used: acetate (sodium acetate, 3.5 g/L), ... glucose (5 g/L), ..." |
| one replicate is one grown culture | si2.xlsx | Table S21 title | "List of Nα-acetylated peptides relatively quantified from WT, ΔrimI, ΔrimJ and ΔrimL strains grown in glucose medium using label-free quantification from biological triplicates" |
| the growth table | si2.xlsx | Table S24 title | "Growth rates determined for WT, ΔrimI, ΔrimJ and ΔrimL E. coli strains grown in glucose and acetate medium" |
| the cross-check table | si2.xlsx | Table S23 title | "Experimental details for all E. coli samples analzed in this study including growth rate, harvesting conditions, OD-values and number of identified proteins" |

The medium objects are `M9_SCHMIDT2016` plus the weighed carbon-source reagent, built by
`schmidt2016.build_environment` from these same quotes. `media.py` is not touched.

### What could not be sourced

- **The replacement cassette.** The paper names the KEIO collection and the PCR
  verification but never the cassette, which is a property of the collection that Baba
  2006 states and that this mirror does not hold. `cassette` is left unset with a typed
  `ProvenanceGap` in `preprocess/provenance_gaps.json`. `BacterialDeletionPerturbation` is
  not a `ProvenanceGapMixin`, so the gap is recorded in the ledger rather than on the leaf
  -- the same place `schmidt2016.py` files the BW25113 background lesions, and for the same
  reason: the field exists, the carrier for its typed absence does not.
- **The strain construction.** No plate, well or accession of the three KEIO strains is
  released, so `construction` is a second typed gap.

### Build

```
PYTHONPATH=$PWD python -m torchcell.database.build_dataset_lmdb --dataset GrowthRateSchmidt2016Dataset
```

| metric | value |
|---|---|
| records | 6 |
| gene-set size | 3 (`BW25113_4373`, `BW25113_1066`, `BW25113_1427`) |
| references | 2 |
| build time | 1 s |
| store | `$DATA_ROOT/data/torchcell/growth_rate_schmidt2016` |
| `provenance.build_manifest` | `fresh` |

Nothing is dropped: both media have a `MEDIA_LIBRARY` entry and all three deletion labels
resolve to a BW25113 locus through the gene-symbol layer, with 0 collisions and 0
ambiguities, so `MIN_RESOLVED_FRACTION` is 1.0 and a renamed strain label stops the build.

### Verification, L0 to L4

`verify_build(root)` runs the shared `verify_fitness_dataset` with the record's own pinned
BW25113 assembly as the gene universe and as the canonical-name resolver, which is how the
landed Campos 2018 loader states the same situation for the same collection. Result:
**PASS**.

```
growth_rate_schmidt2016: PASS
  [ok] L0 structural: 6 records validated
  [ok] L1 count: observed 6, expected 6
  [ok] L1 pair_uniqueness: 6 unique (strain, environment) records, one each
  [ok] L1 provenance_gaps: 0 documented provenance gaps over 0/6 records
  [ok] L1 canonical_gene_names: 3 systematic names, one canonical spelling each
  [ok] L2 value_fidelity: 6 values checked
  [ok] L2 se_nonnegative: 6 values checked
  [ok] L2 uncertainty_sanity: 6 labeled uncertainties, none a zero dispersion
  [ok] L3 reference_one: reference fitness == 1.0 for all 6 records
  [ok] L3 compound_identity: 6 compound references carry a structure identifier
  [ok] L3 media_compound_identity: 72 compound references carry a structure identifier
  [ok] L3 media_membership: 6 records on a shared MEDIA_LIBRARY medium (1 distinct media)
  [ok] L4 gene_containment_sgd: 1.000 of 3 measured genes are BW25113 genes
  [ok] L4 current_genome_genes: every one of the 3 measured systematic names is a gene
```

### BioCypher adapter and its enable-list

`GrowthRateSchmidt2016Adapter` (`torchcell/adapters/schmidt2016_growth_rate_adapter.py`)
serves this dataset, with its enable-list in
`torchcell/adapters/conf/growth_rate_schmidt2016_adapter.yaml`. It is registered in
`dataset_adapter_map`, re-exported from `torchcell/adapters/__init__.py`, named in
`torchcell/knowledge_graphs/conf/kg_bacteria.yaml`, and holds exactly one adapter class, so
`kg_manifest._CONF_RE` reads its own conf rather than a sibling's (issue #743).

This is the **one** Schmidt 2016 dataset that serves a gene perturbation, which is why
`bacterial perturbation (chunked)` and `perturbation to genotype (chunked)` are ON here and
OFF in the paper's three proteome confs. The environment-perturbation pair is on too: each
medium carries one `EnvironmentPhysicalPerturbation` for its carbon source, on every record
and on both wild-type references. `tests/torchcell/adapters/test_schmidt2016_growth_rate_adapter.py`
pins both directions on the conf (hermetic) and on the emitted graph (data-gated).

## 2026.10.09 - Correction: the "Rank 13, Table S23" section above is stale, and Table S23 is now loaded

The section "Rank 13, Table S23: the block is CONFIRMED, and two more release defects"
is superseded. Re-measured on 2026.10.09 after PR #836 (#776) landed the absolute branch:

1. Its reason 1, the absent absolute growth member, is CLOSED for this readout.
   `MeasurementType.growth_rate` is in `ABSOLUTE_MEASUREMENT_TYPES`, so
   `reference_centered=False` is admissible and L3 `reference_zero` is satisfied by the
   reference stating its own finite rate.
2. Its reason 2, the two negative rates a log2 ratio cannot hold, is MOOT on the absolute
   route, which takes no ratio. Both negative rows drop on
   `growth_phase_not_representable` instead.
3. Its reason 3, the unresolvable `Stdev` denominator, is CONFIRMED by re-test and is not
   a blocker: `n_samples` is `int | None`, and the released `Stdev` is stored with the
   conservative `n_samples = 3`.

The live reasons the record count is 15 rather than 26 are the strain vocabulary together
with the empty-genotype L1 key (4 rows), the absent medium (1 row) and the absent
growth-phase and culture-mode slots on `Environment` (6 rows). The loader, the full
measurement and the L0 to L4 table are in
[[torchcell.datasets.ecoli.schmidt2016_s23_growth_rate]].

The two release defects this note recorded for a future Table S23 loader both held and
are both acted on there: the `MG1665` spelling is one of the three strain spellings the
reader admits and is dropped by name, and the released condition labels are normalized
three ways (padding space, trailing footnote digit, lowercase chemostat label) before
they join `schmidt2016.CONDITIONS`.
