---
id: 1cj7wh38f1x166rv14q82ht
title: Campos2018
desc: ''
updated: 1791424616250
created: 1791424616250
---

## 2026.10.07 - Row 36: Campos 2018, the imaged Keio collection

Campos et al. 2018, *Mol Syst Biol* 14:e7573, doi:10.15252/msb.20177573, PMC6018989,
citation key `camposGenomewidePhenotypicAnalysis2018`. Row 36 of the fifty bacterial
datasets. Loader `torchcell/datasets/ecoli/campos2018.py`, class
`GrowthRateCampos2018Dataset`, adapter `torchcell/adapters/campos2018_adapter.py`.

### What the 26 values are, and the evidence

They are **26 features of ONE condition**, not 26 conditions. Established three ways,
each independent:

1. **The paper's own three group counts sum to 26.** 19 morphological + 2 growth + 5
   cell cycle.
2. **The paper's own cross-sum confirms the split.** It refers to "the 24 morphological
   and cell cycle features considered in our screen", which is 19 + 5.
3. **Appendix Table S1 names every symbol** under the headings "Morphological features",
   "Growth features" and "Cell cycle features", and the Dataset EV2 legend sheet gives
   the same names column by column.

There is one medium and one temperature for every strain, so no part of the 26 lies on
the environment axis.

| group | n | symbols |
| --- | --- | --- |
| morphological | 19 | mean and CV of `<L>`, `<W>`, `<A>`, `<V>`, `<SA>`, `<P>`, `<SA/V>`, `<C>`, `<Ar>` (18), plus `CV_DR` |
| growth | 2 | `alpha_max`, `ODmax` |
| cell cycle | 5 | `rho_CD`, `CDN_C0`, `Rel.timing div`, `Rel.timing nuc`, `%2N` |

The mean division ratio is deliberately absent from the 19: pole identity was unknown,
so "measurements of mean division ratio were meaningless and not included in our
analysis". The division ratio contributes only its CV, which is why the count is 19 and
not 20.

Dataset EV2 releases **30** numeric score columns, which is the 26 plus the mean and CV
of nucleoid area (Appendix Table S1 lists those two under morphological, giving 21 there
and 28 symbols in all) plus `%non-div` and `%1N`, the raw proportions the two relative
timings are computed from. So the released column count and the paper's feature count
differ for a stated reason, not through a parse error.

### The record type, and why 25 of 26 values are not served

**One record per strain, `BacterialFitnessExperiment`**, carrying `alpha_max` as a ko/wt
growth-rate ratio. One line of why: `FitnessPhenotype.fitness` is documented as
`ko_growth_rate/wt_growth_rate`, and `alpha_max` is a growth rate, so the ratio is the
schema's own definition rather than a reinterpretation.

The other 25 values have no home, and that is the finding of this row:

| values | the exact mismatch |
| --- | --- |
| 19 morphological + 5 cell cycle | `CalMorphPhenotype` is the only multi-feature morphology class. Its shape fits well (`calmorph: dict[str, float]` for named means, `calmorph_coefficient_of_variation: dict[str, float]` for named CVs, which is exactly Campos's mean/CV split), but its two `field_validator`s reject any key outside `CALMORPH_LABELS` (281 Ohya 2005 CalMorph base parameters) and `CALMORPH_STATISTICS` (220 CalMorph CV parameters). Measured: that vocabulary is **disjoint** from all 26 Campos symbols. CalMorph is a yeast image-analysis program; Campos measured with MicrobeTracker and Oufti. `EnvironmentResponsePhenotype` cannot hold them either, because it carries ONE score per (strain, environment) and has no field naming which feature a number is, so 24 features in one medium collide on the L1 key. There is also no `BacterialCalMorphExperiment`. |
| `ODmax` | a saturating optical density is a carrying capacity, not a rate, so it is not a `FitnessPhenotype` ratio; `MeasurementType` has no member for it (`growth_rate` is "absolute or normalized growth rate / doubling time", `colony_size` is an absolute colony size); and a second `FitnessPhenotype` record per strain would collide on the L1 key, since the environment is the same. |

Serving the 24 morphological and cell cycle features needs either E. coli symbols added
to a yeast program's vocabulary (which would assert a measurement that was not made) or
a new host-neutral multi-feature morphology phenotype class. Both are edits to
`torchcell/datamodels/schema.py`, out of scope for this row, so the mismatch is recorded
rather than forced. The check is a hermetic test
(`test_the_calmorph_vocabulary_rejects_every_campos_feature_name`), and the per-feature
reason is written to `preprocess/served_features.json` at build time.

### Sourcing table

Every quote below is a verbatim substring of `paper.md` in the literature mirror,
sha256 `1bf2f74bbd528f1cd88e46bf96e829b06a7c70dca9a8ad35502fa45e9c594e83`, or of the
Dataset EV2 legend sheet, sha256
`10188365b9ebcf40309c4bdcb415b13472a460c6d0966ed860ad9e59df208274`. All 14 paper-anchored
`SourcedValue`s are audited against the pinned OCR by a `--data` test, and all 4
legend-anchored ones against the pinned xlsx.

| value | source | section | verbatim quote |
| --- | --- | --- | --- |
| reference strain BW25113 | paper.md | Results, imaging and growth measurements | "To provide a reference, 240 replicates of the parental strain (BW25113, here referred to as WT) were also grown and imaged under the same conditions as the mutants." |
| 4,227 strains imaged | paper.md | Results, imaging and growth measurements | "we imaged 4,227 strains of the Keio collection" |
| coverage | paper.md | Results, imaging and growth measurements | "This set of single-gene deletion strains represents $9 8 \%$ of the non-essential genome" |
| medium and temperature | paper.md | Results, imaging and growth measurements | "The strains were grown in 96-well plates in M9 medium supplemented with $0 . 1 \%$ casamino acids and $0 . 2 \%$ glucose at $3 0 ^ { \circ } \mathrm { C }$ ." |
| one culture per row, shaking | paper.md | Methods, screening setup and microscopy | "Cultures were diluted 1:300 in $1 5 0 ~ \mu \mathrm { l }$ of fresh M9 medium supplemented with $0 . 1 \%$ casamino acids and $0 . 2 \%$ glucose and grown in 96-well plates at $3 0 ^ { \circ } \mathrm { C }$ with continuous shaking in a BioTek plate reader." |
| the 2 growth features | paper.md | Results, imaging and growth measurements | "In parallel, using a microplate reader, we recorded the growth curves of all the strains (Fig 1A) and estimated two population-growth features. We fitted the Gompertz function to estimate the maximal growth rate ... and used the last hour of" |
| ODmax definition | paper.md | Results, imaging and growth measurements | "growth to calculate the saturating density ... of each culture" |
| 19 morphological | paper.md | Results, morphological features | "In total, each strain was characterized by 19 morphological features" |
| 5 cell cycle | paper.md | Results, growth and cell cycle features | "As a result, each strain was associated with five cell cycle features (Dataset EV2), in addition to the 19 morphological features and two growth features mentioned above" |
| 24 = 19 + 5 | paper.md | Results, dependencies between dimensions and cell cycle | "the close-to-zero correlations between growth rate and any of the 24 morphological and cell cycle features considered in our screen" |
| the WT median is the normalization anchor | paper.md | Methods, data processing | "For each plate, we set the median values of each feature, $F _ { ; }$ , to the median feature value of the parental strain." |
| the score is a robust z-score | paper.md | Methods, data processing | "The $F$ values were transformed into normalized scores by a transformation akin to a $z$ -score transformation but more robust to outliers." |
| cassette | paper.md | Results, genes, functions and pathways | "Genes in the Keio collection were deleted by an in-frame replacement of a kanamycin-resistance cassette that has a constitutive promoter and no transcriptional terminator to ensure expression of downstream genes in operons (Baba et al, 2006)." |
| cells imaged per strain | paper.md | Results, imaging and growth measurements | "On average, about 360 $\left( \pm 1 6 5 \right)$ cells were imaged for each strain." |
| the label column is gene names | Dataset EV2 legend | sheet "Legend scores" | "Name of the deleted gene" |
| plate and well | Dataset EV2 legend | sheet "Legend scores" | "Number of the Keio plate" |
| the `*` marker | Dataset EV2 legend | sheet "Legend scores" | "Re-imaged from 2mL liquid cultures" |
| the degree marker | Dataset EV2 legend | sheet "Legend scores" | "Strains independently checked with a different phenotype" |

#### n_samples and the uncertainty type

`n_samples = 1`, `sample_unit = biological_replicate` for a record, and 240 for the
reference. Sourced, not guessed: the Methods state one diluted culture per well in a
96-well plate read by the plate reader, and the release carries one `(Plate nb, Well nb)`
position per row (measured: 4,467 rows, 4,467 distinct positions). The 240 wild-type
rows are the parent's replicates.

The 360 +/- 165 imaged cells per strain are the replicate design of the MORPHOLOGICAL
features, not of the growth curve, so they are deliberately NOT this phenotype's
`n_samples`. Dataset EV1 carries the exact per-strain cell count in `nb Cells`, which is
the number a future morphology phenotype would use.

**Genuinely not sourced, carried as typed gaps:**

- `fitness_uncertainty` and `fitness_se`: `not_reported_by_primary`. Dataset EV2 releases
  one corrected `alpha_max` per strain and no dispersion; with one growth curve per well
  there is no within-strain spread to report. Searched for `replicate`, `standard
  deviation`, `bootstrap`, `n =`, `error` across the Methods, the Appendix and both
  Dataset EV2 legend sheets.
- `Environment.duration_hours`: `not_reported_by_primary`. The endpoint is an
  optical-density rule, not a clock: every strain was sampled at an OD600 of 0.2 +/- 0.1.

### The fitness is derived, and the derivation is verified to machine precision

`fitness = alpha_max_strain / median(alpha_max over the 240 wild-type rows)`, reading the
`alpha_max (min-1)` column of Dataset EV2's "Normalized data" sheet (the authors' plate-,
position-, time- and OD-corrected value). The denominator is **0.0097366516 min^-1**.
The wild-type median is the right denominator because the correction is anchored on it
(the plate-median quote above).

The cross-check, measured over the 3,664 kept rows: the released `alpha_max` SCORE is an
exact affine function of the derived fitness,

```text
score = 23.777071 * (fitness - 1)
max |residual| = 1.4e-13
Pearson r = 1.000000000000
```

The intercept equals minus the slope to 1e-4, which is the algebra the paper states
(`s = 1.35 * (F_i - median(F_i^WT)) / iqr(F_i^WT)`) rearranged in terms of the ratio. So
the ratio is the released number in different units. Measured fitness range 0.121789 to
1.351417, median 1.004160; no value is non-positive, so `FitnessPhenotype`'s clamp never
fires. Both the synthetic and the real affine check are tests.

### Identifier route, with counts

Route: `gene_symbol` through `bacteria_common.reconcile_locus_tags` on the BW25113
GenBank annotation (`GCA_000750555.1_ASM75055v1`), recorded on every record as
`DerivedIdentifierMapping(source_identifier=<released label>, route="gene_symbol")`.

| quantity | count |
| --- | --- |
| distinct non-wild-type labels reaching the reconciler | 4,158 |
| resolved to one locus | 3,720 (3,586 RENAMED + 134 NON_GENE_FEATURE) |
| decided by the gene-symbol layer | 3,718 |
| decided by an ECK `gene_synonym` | 2 |
| retired (not found) | 438 |
| collisions, ambiguous | 0, 0 |
| resolved fraction | 0.8947 |

`MIN_RESOLVED_FRACTION` is 0.89, below the measured 0.8947, so a release whose
identifiers drift further stops and reports instead of dropping strains silently.

The 438 unresolved labels split into two **measured** causes, not two guesses:

- **412 Keio JW strain ids.** Campos labels a row by its JW id exactly when the strain
  has no current gene name. The BW25113 GenBank annotation DOES carry JW ids as
  `gene_synonym` (measured: 4,334 distinct ones, which is how
  `torchcell/datasets/ecoli/fuhrer2017.py` resolves its strains), and **none** of these
  412 is among them. So these are the Keio strains whose deleted locus the pinned
  annotation no longer carries, not a resolver miss. MG1655 carries no JW synonym at all
  (measured: 0), so no second assembly rescues them. The Keio strain-to-gene table is
  Baba 2006's web table, which the mirror does not hold (its mirrored `si1.pdf` carries
  no `JW####` string), so no `JW#### -> b####` arithmetic is invented. Both facts are
  `--data` tests.
- **26 gene names the 2014 BW25113 annotation predates**: `adeP`, `adeQ`, `elyC`,
  `ettA`, `ghoS`, `ghoT`, `ghxP`, `ghxQ`, `htgA`, `iceT`, `lgoD`, `lgoR`, `lgoT`,
  `opgE`, `rhoL`, `sslE`, `tiaE`, `waaO`, `yahH`, `ybbV`, `yiaI`, `ykiB`, `ymfH`,
  `yzcX`, `yzfA`, `zapE`. Measured: the MG1655 annotation resolves **22 of the 26**, so
  `bacteria_common.eck_crosswalk` is the identified route to recover them. That is a
  second, two-step derived route for 22 of 4,227 rows (0.5%), left to a follow-up rather
  than added here; the measurement is recorded so the follow-up does not have to redo it.

### Retention ledger, with the arithmetic

Released rows are the 4,471 rows of the "Normalized data" sheet (the "Scores" and
Dataset EV1 "Raw data" sheets hold 4,467, which is the same table without the four
footer rows).

| # | rule | rows | labels |
| --- | --- | --- | --- |
| 1 | `row_is_a_table_footer_summary` | 4 | the sheet's trailing blank row plus `Mean`, `Stdev`, `CV` |
| 2 | `row_is_a_wild_type_replicate` | 240 | `WT0511` (48), `WT0813` (96), `WT0815` (96) |
| 3 | `label_carries_a_different_culture_or_check_annotation` | 5 | `hfq*`, `pgm*`, `rapZ*`, `rodZ*`, `fabH` + degree |
| 4 | `label_is_not_in_the_bw25113_annotation` | 438 | 412 JW ids + 26 post-annotation names |
| 5 | `label_is_a_fragment_of_a_merged_bw25113_locus` | 0 | none on this release |
| 6 | `label_is_ambiguous_in_bw25113` | 0 | none on this release |
| 7 | `label_heads_more_than_one_row` | 120 | 56 labels, each heading 2 to 4 rows |
| | **kept** | **3,664** | 3,664 distinct BW25113 locus tags |

```text
4471 - 4 - 240 - 5 - 438 - 0 - 0 - 120 = 3664
```

4,471 minus the 4 footer rows minus the 240 wild-type rows is 4,227, the strain count the
paper states, which is the first check that the sheet was read as released.

Rule 3 is legend-sourced, not a heuristic: `*` means "Re-imaged from 2mL liquid
cultures", which is a different culture format from the 96-well screen, and the degree
sign means "Strains independently checked with a different phenotype", which is a QC
statement about the strain. Neither row is a measurement of this screen. Measured: none
of the five bare names (`hfq`, `pgm`, `rapZ`, `rodZ`, `fabH`) appears on another row, so
nothing is dropped twice.

Rule 7 keeps strains from merging. Each of the 56 labels heads several rows at distinct
Keio plate/well positions, so they are distinct strains the release names identically.
`ygaQ` heads four rows, and `torchcell/datasets/ecoli/shiver2016.py` shows why: the
current annotation merges that region's `ygaQ_1`..`ygaQ_4` fragments into one locus.
Keeping them would put several records on one locus tag; keeping one would be an
arbitrary choice between real measurements, so the whole group goes. Rules 5 and 6 carry
zero rows on this release and exist so a future annotation that merges or splits one of
these labels is dropped rather than silently merged.

The released Keio plate and well are not lost: they ride on
`BacterialDeletionPerturbation.construction` as `StrainConstruction(plate=..., well=...)`,
which the fitness L1 key does not read, so storing them cannot manufacture distinctness.

### Media

`CAMPOS2018_M9_CASAMINO_GLUCOSE` is declared in the loader module (the Shiver 2016
convention for a dataset-specific medium), deriving from the shared `M9` salts recipe
plus the two stated supplements:

- casamino acids at 0.1% (w/v), `complex_ingredient`, typed
  `ComponentDefinition.intrinsically_undefined` with a bare `Compound` and no invented
  structure identifier, because it is an acid hydrolysate of casein (the Wang 2018
  convention). This is also why `is_synthetic` is False.
- D-glucose at 0.2% (w/v), `carbon_source`. It is fixed across the screen, so it is a
  component rather than a varied physical factor.

Kanamycin at 30 ug/ml appears only in the overnight pre-culture, not in the assay
medium the Methods describe, so it is not a component here. `Environment` carries
temperature 30 C, `aerobicity="aerobic"` (continuous shaking), and no perturbations.

### Build and verification

```text
BUILT GrowthRateCampos2018Dataset: 3664 records at
  $DATA_ROOT/data/torchcell/ecoli_growth_rate_campos2018 in 4s;
  gene_set size 3664; references 1
build manifest: .../preprocess/build_manifest.json
```

`python -m torchcell.provenance.build_manifest` reads
`ecoli_growth_rate_campos2018` as **fresh**, no drift.

`verify_build` (the fitness family verifier, L0 to L4) PASSES every check:

| level | check | result |
| --- | --- | --- |
| L0 | structural | 3,664 records validated |
| L1 | count | observed 3,664, expected 3,664 |
| L1 | pair_uniqueness | 3,664 unique (strain, environment) records, one each |
| L1 | provenance_gaps | 10,992 documented gaps over 3,664/3,664 records, 0 deferred |
| L1 | canonical_gene_names | 3,664 systematic names, one canonical spelling each, each current |
| L2 | value_fidelity | 3,664 values checked |
| L2 | se_nonnegative / uncertainty_sanity | 0 values (nothing reported) |
| L3 | reference_one | reference fitness == 1.0 for all 3,664 records |
| L3 | media_compound_identity | 18,320 component references carry a structure identifier, 0 unencodable |
| L3 | media_membership | 3,664 records on a medium deriving from a shared library base |
| L4 | gene_containment_sgd | 1.000 of 3,664 measured genes are BW25113 genes |
| L4 | current_genome_genes | every one of the 3,664 names is a gene of the current genome |

The report is at `preprocess/verification_report.json`; the drop ledger, the identifier
histogram and the served/unserved feature inventory are at
`preprocess/dropped_records.json`, `preprocess/identifier_reconciliation.json` and
`preprocess/served_features.json`.

### Raw mirror

`$DATA_ROOT/torchcell-raw/camposGenomewidePhenotypicAnalysis2018/` holds exactly the
file the loader consumes, with a `Manifest`:

| path | sha256 | retrieval |
| --- | --- | --- |
| `data/MSB-14-e7573-s004.xlsx` | `10188365b9ebcf40309c4bdcb415b13472a460c6d0966ed860ad9e59df208274` | `pmc_cloud`, `torchcell.literature.retrieve.pmc_cloud_object`, key `PMC6018989.1/MSB-14-e7573-s004.xlsx` |

The recorded retrieval was re-run on 2026-10-07 and returned byte-identical bytes, so it
is scriptable and reproducible; nothing here needed a manual recipe.

`si_expected` names what is deliberately not duplicated:

- Dataset EV1 (`MSB-14-e7573-s003.xlsx`, the pre-normalization raw table with `nb Cells`,
  sampling OD and elapsed time). Already in the literature mirror as `si/si3.xlsx`,
  sha256 `82f0777cf85c3277be95e04c04e893fdf46c412e2640d61b35e70af75b312837`, and not
  consumed by this loader.
- The Appendix (`MSB-14-e7573-s001.docx`), whose Table S1 names every feature and its
  symbol. Already in the literature mirror as `si/si1.docx`, sha256
  `72cc3510fa63cbf625bb1cd17acebf2a4ed764be0b11acae312b7afbaec40c75`.
- Computer Code EV1/EV2 (`-s005.zip`, `-s006.zip`): the authors' analysis scripts, not
  data. Not mirrored.

### Follow-ups this row identified

- A host-neutral multi-feature morphology phenotype (or a bacterial morphology
  experiment family) would make 24 of these 26 features storable, and Dataset EV1's
  `nb Cells` is the per-strain replicate count it would need. The shape `CalMorphPhenotype`
  already has is the right shape; only its vocabulary is yeast-bound.
- A `MeasurementType` member for a saturating optical density / carrying capacity would
  make `ODmax` storable.
- `eck_crosswalk` through MG1655 recovers 22 of the 26 post-annotation gene names
  (measured), worth 22 rows.

## 2026.10.09 - Issue #774: the 26 morphology features get a phenotype class

Issue #774 recorded the largest single-paper representational gap in the bacterial
program: the loader stored 1 of the release's 26 per-strain values and dropped 25, not
for want of data but for want of a class. This closes 25 of the 26. The one left is
`ODmax`, a saturating optical density, which is a carrying capacity rather than a
growth-rate ratio and is not morphology either.

New: `BacterialMorphologyPhenotype` in `torchcell/datamodels/schema.py`, its feature
vocabulary in `torchcell/datamodels/bacterial_morphology_features.py`, the dataset
`MorphologyCampos2018Dataset` beside the existing `GrowthRateCampos2018Dataset`, its
adapter `torchcell/adapters/campos2018_morphology_adapter.py`, and the L0 to L4 gate
`torchcell/verification/bacterial_morphology.py`.

### The vocabulary is the assay's, not CalMorph's

`CalMorphPhenotype` has exactly the right shape, a dict of named per-strain values plus
a dict of named coefficients of variation, and exactly the wrong vocabulary: its two
validators reject any key outside the 281 base and 220 CV parameters of CalMorph, a *S.
cerevisiae* image-analysis program, and that label set is disjoint from every Campos
symbol. The shape is kept and the label set becomes a property of the ASSAY:
`BacterialMorphologyPhenotype.assay` names a `MorphologyAssay` in `MORPHOLOGY_ASSAYS`
and that assay's features are the only permitted keys, so a later bacterial imaging
screen registers its own assay and widens nobody else's set.

Each `MorphologyFeature` declares what statistic its number IS, and the split between
the two dicts is that declaration rather than a naming convention, so a CV cannot be
filed as a mean. The Campos assay is six statistics side by side in one row:

| statistic | n | symbols |
| --- | --- | --- |
| `mean` | 10 | `<L>` `<W>` `<A>` `<V>` `<SA>` `<P>` `<C>` `<Ar>` `<SA/V>` `<NA>` |
| `coefficient_of_variation` | 11 | the CV of each of those nine dimensions, plus `CV_SA/V` and `CV_DR` |
| `pearson_correlation` | 1 | `rho_CD` |
| `regression_intercept` | 1 | `CDN_C0` |
| `inferred_relative_timing` | 2 | `Rel.timing div` `Rel.timing nuc` |
| `fraction_of_cells` | 1 | `%2N` |

A consumer that averages a correlation with a relative timing is pooling quantities the
source never claimed were comparable; naming the statistic is what lets it refuse.

### Why the vocabulary is 26 and why Table S1 is the authority

The main text counts "19 morphological features", and Appendix Table S1 ("Features
considered in this study and their associated symbols", sha256
`72cc3510fa63cbf625bb1cd17acebf2a4ed764be0b11acae312b7afbaec40c75`, read as the text
runs of `word/document.xml`) names 21 morphological symbols, the difference being the
mean and variability of nucleoid area. Table S1 is the release's naming authority and the
main text says so: "The name and abbreviation for all the features can be found in
Appendix Table S1." Table S1's 28 symbols are therefore 21 morphological + 2 growth + 5
cell cycle, and the 26 the assay holds are the 28 minus the two growth symbols
(`alpha_max`, served as the `FitnessPhenotype` of the fitness dataset, and `ODmax`).

### The two released columns the vocabulary leaves out, and the measurement that justifies it

Dataset EV2's "Normalized data" sheet holds 30 numeric feature columns against Table S1's
28, plus 2 island assignments. The two extras are `%non-div` and `%1N`, and they are not
dropped for want of a class: each is an exact deterministic inverse of a feature that IS
stored. Measured over the 4,189 rows that determined both:

| relation | residual |
| --- | --- |
| `Rel.timing div == -log2(1 - %non-div / 2)` | max abs 1.3e-15 |
| `Rel.timing nuc == -log2(1 - %1N / 2)` | max abs 1.2e-15 |

That is the steady-state cell-age inversion Table S1's caption describes ("estimated as
the proportions of cells without any significant constriction (constriction degree <0.15)
or with a single nucleoid, respectively"). Storing both sides would write one measurement
twice under two names. `%2N` is NOT the complement of `%1N` and IS stored: measured,
`|%1N + %2N - 1|` reaches 0.129, so cells with more than two nucleoids exist. The two
island columns are the paper's own clustering labels over the screen, not measurements of
a strain, so they are no dataset's phenotype. Recorded in
`RELEASED_COLUMNS_OUTSIDE_THE_VOCABULARY` and `RELEASED_CLUSTER_COLUMNS`.

### `n_samples` is the segmented cell, and Dataset EV1 became a build input

A record's number is a mean (or CV, or correlation) over the strain's retained segmented
cells from ONE well, so `biological_replicate` would report 1 for a mean over hundreds.
`SampleUnit.cell` was added and `n_samples` is Dataset EV1's `nb Cells` column. That
makes Dataset EV1 (`MSB-14-e7573-s003.xlsx`) a consumed raw file rather than a referenced
supplement, so it is now deposited in the raw mirror with its own retrieval record:

| path | sha256 | retrieval |
| --- | --- | --- |
| `data/MSB-14-e7573-s004.xlsx` | `10188365b9ebcf40309c4bdcb415b13472a460c6d0966ed860ad9e59df208274` | `pmc_cloud`, key `PMC6018989.1/MSB-14-e7573-s004.xlsx` |
| `data/MSB-14-e7573-s003.xlsx` | `82f0777cf85c3277be95e04c04e893fdf46c412e2640d61b35e70af75b312837` | `pmc_cloud`, key `PMC6018989.1/MSB-14-e7573-s003.xlsx` |

The two sheets are joined on the Keio `(plate, well)`. Measured: both carry the same
4,467 positions with the same gene label in the same order, so the join is a row
alignment and a position present in one and absent from the other stops the build. The
cell counts corroborate the paper: `nb Cells` sums to 1,301,055 with mean 291.26 and SD
116.71 over 4,467 rows, against "retaining about 1,300,000 identified cells ( $2 9 1 \pm
1 1 6$ cells/strain)". Minimum 43, maximum 1,658.

The reference is a different statistic and says so: the per-feature MEDIAN over the 240
parental replicate wells, `n_samples=240`, `sample_unit=biological_replicate`, the same
convention the fitness reference already uses.

A feature that needs an absolute unit gets one from the release and nowhere else:
Appendix Table S1 has no unit column and `paper.md` states no feature unit, so the 8
units come from Dataset EV2's "Legend normalized data" sheet ("Mean cell length (µm)"),
which the "Normalized data" header repeats as `<L> (µm)`. The loader BUILDS each column
name as `symbol (unit)` from the vocabulary, so a header that moved stops the build. The
other 18 features state no unit, which is the honest record: a CV, a correlation, a shape
factor, a relative timing and a population fraction are dimensionless.

### Coverage is partial by design, and the split is measured

`BacterialMorphologyPhenotype` accepts the features a source DETERMINED for a strain, not
the whole vocabulary: Dataset EV2's legend states "NaN (Not a Number) values are
attributed to non-determined fields". Measured over the 4,227 imaged strains, 7 columns
are NaN on exactly the same 278 rows and the other 19 are never NaN, and no wild-type row
is among the 278:

| set | n | symbols |
| --- | --- | --- |
| `MORPHOLOGY_REQUIRED_FEATURES` | 19 | every mean and CV of the phase-contrast dimensions, plus `CV_DR` |
| `MORPHOLOGY_DAPI_FEATURES` | 7 | `<NA>` `CV_NA` `rho_CD` `CDN_C0` `Rel.timing div` `Rel.timing nuc` `%2N` |

The set is read off the data rather than reasoned out from the derivations, because
`Rel.timing div` is inferred from a phase-contrast proportion yet is absent on those same
rows. Demanding full coverage would drop 19 real features to save 7 absent ones.

### Measured before and after

| | before | after |
| --- | --- | --- |
| Campos records in the dev tree | 3,664 (fitness) | 3,664 fitness + 3,664 morphology |
| stored per-strain values | 1 per record, 3,664 total | 3,664 + 93,570 |
| features of the release with no phenotype class | 25 of 26 | 1 of 26 (`ODmax`) |

93,570 is `19 x 3,664 + 7 x 3,422`: the 19 required features on every kept record, and
the 7 DAPI-derived ones on the 3,422 kept records that determined them (242 of the 3,664
kept strains have no nucleoid channel). The issue's estimate was "roughly 88,000", which
is 24 features at full coverage; the realized figure is higher because the vocabulary
carries the two nucleoid-area symbols the main text's headline 19 leaves out, and lower
per DAPI feature because coverage is not full. Per-feature counts are in
`preprocess/served_features.json`.

Both datasets keep exactly the same 3,664 rows, by construction: the morphology loader
calls the same `read_normalized_table` and `resolve_rows`, so a strain is writable for
both or for neither, and the drop ledger is identical.

```text
BUILT MorphologyCampos2018Dataset: 3664 records at
  $DATA_ROOT/data/torchcell/ecoli_morphology_campos2018 in 9s;
  gene_set size 3664; references 1
```

### L0 to L4, the morphology gate

`torchcell/verification/bacterial_morphology.py`, run as
`python -m torchcell.datasets.ecoli.campos2018 verify-morphology`. The CalMorph verifier
could not be reused: it reads `phenotype["calmorph"]` by literal key and asserts the
literal counts 281 / 220 / 501. The new one reads the vocabulary from the assay each
record names, and refuses a dataset that mixes two assays.

| level | check | result |
| --- | --- | --- |
| L0 | structural | 3,664 records validated |
| L1 | count | observed 3,664, expected 3,664 |
| L1 | pair_uniqueness | 3,664 unique (strain, environment) records, one each |
| L1 | assay_coverage | all 3,664 carry the 19 required features of assay campos2018 and nothing outside its 26 |
| L1 | provenance_gaps | 3,664 documented gaps over 3,664/3,664 records, 0 deferred |
| L1 | canonical_gene_names | 3,664 systematic names, one canonical spelling each, each current |
| L2 | value_fidelity | 93,570 values over 26 features finite and inside the bound of their declared statistic |
| L2 | cv_nonnegative | all 40,062 coefficients of variation non-negative |
| L2 | uncertainty_sanity | 0 labeled uncertainties (the CVs are released features, not a typed dispersion field) |
| L3 | vocabulary_parity | 15 value symbols + 11 CV symbols == 26 features, disjoint, exhaustive, distinct |
| L3 | reference_populated | the reference profile carries the 19 required features in all 3,664 records |
| L3 | media_compound_identity | 18,320 component references carry a structure identifier, 0 unencodable |
| L3 | media_membership | 3,664 records on a medium deriving from a shared library base |
| L4 | gene_containment_sgd | 1.000 of 3,664 measured genes are BW25113 genes |
| L4 | current_genome_genes | every one of the 3,664 names is a gene of the current genome |

L2 `value_fidelity` is PER FEATURE, against the bound each declared statistic implies: a
mean and a CV are non-negative, a Pearson correlation lies in [-1, 1], a fraction of
cells and an inferred relative timing in [0, 1], and a fitted intercept is bounded only
by finiteness. Pooling 26 columns into one list, which is what the CalMorph verifier
does, cannot express those and reports an index no reader can trace to a column. The
bounds hold on the release with room to spare (measured: `rho_CD` in [0.015, 0.980],
`CDN_C0` in [0.145, 0.741], `Rel.timing nuc` up to 1.000, `%2N` in [0.000, 0.560]).

### Schema impact

`scripts/schema_impact_check.py --base origin/main`: **49 impacted datasets, 0 breaking
(additive)**. Nine changed symbols, all additions: the three `BacterialMorphology*`
classes, `SampleUnit` gaining `cell`, and the five unions and maps gaining those classes.
`SampleUnit` is what reaches the other 47 datasets, and only as a widened enum. KG 4.0 is
a full rebuild, so the impacted stores are remade there; the two Campos dev stores were
rebuilt here with `--retire-existing` and read `fresh` under
`python -m torchcell.provenance.build_manifest`.

### No other loader in the repo is extended by this class

Swept `notes/`, all 90 modules under `torchcell/datasets/`, the candidate-table
generators and every open issue. Two findings:

- **Schmidt 2016 Table S28** is recorded as blocked in
  `notes/plan.bacteria-si-phenotype-audit-ecoli.md` with the reason "a bacterial
  cell-size phenotype has no class. `CalMorphPhenotype` is the yeast CalMorph feature set
  and does not apply", which reads like this gap and is NOT one.
  Its `Cell length1` / `Cell width1` columns carry footnote 1, "Cell length, width and
  volume according to Volkmer, B. & Heinemann, M. ... PLoS ONE 6, e23126 (2011)", so they
  are another paper's measurements; the 19 data rows are growth conditions of the wild
  type rather than perturbed strains; and there is no CV and no cell count, so
  `morphology_coefficient_of_variation`, `n_samples` and `sample_unit` would all be
  empty. Writing them under `assay="schmidt2016"` would attribute an assay Schmidt never
  ran. Left unserved. The audit row also states "Table S28, 24 data rows" where the sheet
  has 19 (header on row index 2, data on 3 to 21, then five footnotes), worth correcting
  there.
- **Silvis 2021** is the next real customer and has no loader, no raw mirror and no
  issue. Its recorded `schema_need` is "a shape phenotype whose measured unit is a
  segmented cell, summarized per strain as a location and a dispersion for each
  dimension", which is this class. Serving it needs a `silvis2021` assay plus a `median`
  member of `MorphologyStatistic`: Silvis releases medians and a robust CV, and filing a
  median under `mean` is the misstatement the `statistic` field exists to prevent. Not
  added here, because the enum's own rule is to add members as datasets need them and no
  dataset needs it yet.
