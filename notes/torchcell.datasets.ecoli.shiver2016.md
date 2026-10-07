---
id: qyzc7xg5cugjkoje2ihj7gi
title: Shiver2016
desc: ''
updated: 1791406309864
created: 1791406309864
---

## 2026.10.07 - Loader, sourcing, and the first dev build

Row 27 of the fifty bacterial datasets. Shiver et al. 2016, PLoS Genet 12(6):e1006124,
doi:10.1371/journal.pgen.1006124, PMC4927156, citation key
`shiverChemicalGenomicScreenNeglected2016`. Loader
`torchcell/datasets/ecoli/shiver2016.py`, class `EnvChemgenShiver2016Dataset`, dev store
`$DATA_ROOT/data/torchcell/ecoli_env_chemgen_shiver2016`.

### What the release is

S1 Dataset (`pgen.1006124.s005.txt`) is the integrated fitness-score matrix: 3,975 gene
columns by 292 condition rows. 235 of those rows are Nichols et al. 2011 re-analyzed with
this paper's improved pipeline and carry batch 0; 57 are this study's own screen and carry
batch 1 (30 conditions) or batch 4 (27). 3,975 x 57 = 226,575 released cells, which is the
instance count the candidate table states, and those 57 rows are what the loader serves.
The batch is not decoration: "The batch groups conditions that were measured in the same
experiment and normalized as a group."

### Sourcing table

Every quote below is a verbatim substring of
`$DATA_ROOT/torchcell-library/shiverChemicalGenomicScreenNeglected2016/paper.md`, sha256
`4a1bec97d10f6ef1c02c99329f7adce0d79623819fcf421c098058a7c82ced72`. The OCR renders some
numerals with LaTeX spacing, and the quotes keep that spacing exactly. All 19 module-level
`SourcedValue`s pass `audit_sourced_value` against that hash (run 2026.10.07).

| what | value | section | verbatim quote (abbreviated where long) |
|---|---|---|---|
| background strain | BW25113 | Methods, Media/strains | "The KEIO deletion library is derived from BW25113 (F- λ- Δ(araD-araB)657 ΔlacZ4787(::rrnB-3) rph-1 Δ(rhaD-rhaB)568 hsdR514) [60]" |
| screen size | 3,975 mutants x 57 stresses | Results | "We tested the sensitivities of 3975 mutants of E. coli K-12 to 57 stresses, split between new and previously screened conditions." |
| what the score IS | signed significance of a colony-size change | Results | "These fitness-scores represent the statistical significance of a change in colony size for a particular condition, with negative and positive fitness-scores representing sensitivity and resistance, respectively." |
| scoring pipeline | Collins/Nichols S-score | Methods, Data collection | "The chemical genomics screen was conducted using the same methodology as reported previously [8] with few modifications." |
| readout | Iris colony opacity | Methods, Data collection | "Images were analyzed using Iris to measure the total intensity of pix els within the colony to calculate an opacity metric." |
| default plate | LB Lennox agar, 90 mM NaCl, 2% bacto agar | Methods, Media | "Chemical sensitivity screens used LB Lennox agar plates $1 \%$ (w/v) tryptone, $0 . 5 \%$ (w/v) yeast extract, $9 0 \mathrm { m M }$ sodium chloride, $2 \%$ (w/v) bacto agar) unless otherwise specified." |
| M9 plate | M9 salts + 0.2% (w/v) glucose + 2% bacto agar | Methods, Media | "M9 minimal plates used in the screen contained M9 salts, $0 . 2 \%$ (w/v) glucose, and $2 \%$ ... bacto agar." |
| temperature | 37 C | Methods, Media | "Ordered libraries grown during the chemical-genomics screens were incubated at $3 7 ^ { \circ } \mathrm { C }$ until the majority of colonies reached a defined size ( ${ \sim } 8$ hours)" |
| 4 C exposure | 4 C, 5 weeks = 840 h | Methods, Data collection | "One condition, $4 ^ { \circ } \mathrm { C }$ survival, involved growth of colonies on LB plates at $3 7 ^ { \circ } \mathrm { C }$ for 6 hours, and transfer of the colony array to $4 ^ { \circ } \mathrm { C }$ for 5 weeks." |
| identifier space | gene names, precise deletions unless stated | S1 Dataset legend | "Gene names are used to label the mutation. Unless otherwise specified, the mutations are precise gene deletions." |
| matrix orientation | genes in columns, conditions in rows | S1 Dataset legend | "A table of fitness scores for genes (columns) by conditions (rows) suitable for clustering [62] and other downstream analyses." |
| batch meaning | normalization group; batch 0 = Nichols | S1 Dataset legend | "The batch groups conditions that were measured in the same experiment and normalized as a group. Conditions from Nichols et al. [8] were assigned batch $^ { \alpha } 0 ^ { \gamma }$ ." |
| measurement counts | per condition, not per record | Methods, Clustering | "Unreliable measurements were removed from the dataset at multiple points in the analysis and each condition had a different number of measurements (fitness-scores) that passed analysis." |
| hit threshold band | (-2.0,-1.2) and (+1.2,+2.1) at FDR 5% | Methods, Clustering | "$9 5 \%$ of the cutoff values for negative (sensitization) fitness-scores fell in the range (-2.0,-1.2) while $9 5 \%$ of the cutoff values for positive (resistance) fitness-scores fell in the range $( + 1 . 2 , + 2 . 1 )$" |
| follow-up cassette | kanamycin from pKD4, NOT the array | Methods, Media | "Operon deletions were generated using $\lambda$ -red recombineering to replace the operon with a kanamycin resistance cassette amplified from pKD4." |
| raw-data location | Dryad 10.5061/dryad.f3kc0 | Data Availability | "Raw data associated with colony size and GSEA analysis are available at Dryad" (the DOI follows in the same sentence; the loader pins it as `DRYAD_DOI`) |

### Record type: `EnvironmentResponsePhenotype`, `measurement_type=z_score`

The released value is signed. Measured over the full 57-condition grid of batches 1 and 4
(all 3,975 columns, 218,103 non-blank cells of 226,575): min -30.935, max 10.399, mean
-0.0907. `FitnessPhenotype` clamps
non-positive values to 0 and its verifier wants a 1.0 reference, so it would delete more
than half of this dataset. `EnvironmentResponsePhenotype` with a 0.0 reference is the
honest carrier, and 0 is exactly the paper's neutral point.

`z_score` is the member chosen, and it is a compromise worth stating plainly. The
fitness-score is the Collins/Nichols S-score: the Methods defer the pipeline to Nichols
et al. 2011 (ref [8], Cell 144:143) and the original analysis pipeline to Collins et al.
2006 (ref [18], Genome Biol 7:R63), whose S-score is a modified t-statistic. That is a
standardized colony-size deviation, which is what `MeasurementType.z_score` names
("standardized fitness/growth deviation"). Rejected, with reasons:

- `log2_ratio` -- the number is not a log of a ratio of abundances.
- `differential_fitness` -- defined as a plain subtraction of two normalized fitnesses.
- `colony_size` -- defined as an absolute, unnormalized size.
- `sensitivity_score` -- a one-sided fitness-defect score; this one is two-sided by
  construction, and resistance is half of what the paper reports.

An `s_score` member would be more precise than `z_score`, and that is raised in the PR as
a finding rather than acted on: `schema.py` is untouched here.

`assay_type=colony_size_array` is the typed pinned-array design. The quantified metric is
total pixel intensity (opacity), not area, which `units` states.

### Units: converted inside one dimension, never invented

The release uses seven dose units. Four are typed as released (`mM`, `uM`, `% (w/v)`,
`% (v/v)`). The three mass-per-volume units collapse onto the single typed `ug/mL` by
exact decimal conversion, recorded in `UNIT_CONVERSIONS`:

| released | stored | multiplier | worked example |
|---|---|---|---|
| ug/mL | ug/mL | 1 | pseudomonic acid A 36 -> 36.0 |
| ng/mL | ug/mL | 1e-3 | 5-fluorouridine 250 -> 0.25; ciprofloxacin 1 -> 0.001 |
| mg/mL | ug/mL | 1e3 | azelaic acid 1 -> 1000.0 |

No `ConcentrationUnit` member is added, and the deliberately deferred `mg_per_l` is not
needed at all: 1 ug/mL IS 1 mg/L, so the enum already carries that quantity. Every dose
gets `basis=DoseBasis.fixed` (each is an explicit fixed dose, never a target-inhibition
endpoint).

### The 57 conditions

Each is one row of a committed `ConditionSpec` table keyed on the verbatim released label,
which is also the `screen_id` of every record of that condition. Nine are on the M9
minimal plate (the "M9min" label prefix is exactly what selects it, asserted in the tests);
the other 48 are on LB Lennox agar, the stated default. Temperature lives on
`Environment.temperature`: 37 C except 10 C, 25 C, `UV+10C` (10 C) and `4C survival` (4 C).
UV is `PhysicalFactor.radiation` on three conditions. The M9 carbon source is
`PhysicalFactor.carbon_source` at the recipe's 0.2% (w/v) glucose, or acetate at 0.6% (w/v)
for the one condition whose own label names another carbon source.

**`screen_id` is load-bearing, measured.** Dropping it collapses five groups of otherwise
identical conditions: `EDTA [1 mM]` in batches 1 and 4, `SDS [1% (w/v)]` in 1 and 4,
`ampicillin [4 ug/mL]` in 1 and 4, `gliotoxin-A` with `gliotoxin-B` in batch 1, and the
three `M9min glucose` controls (`-A` and `-B` in batch 1, the plain one in batch 4). Each
of those is an independently normalized screen, so each is its own condition; the verbatim
label is what keeps them L1-distinct. The release does not define the `-A`/`-B` suffix, and
the loader does not invent a meaning for it.

**Two readings are DERIVED and flagged as such.** First, `M9min acetate [0.6% (w/v)]` is
read as acetate at 0.6% (w/v) as the carbon source, by analogy with the release's own
`M9min glucose [0.2% (w/v)]` labels, whose bracketed value matches the stated plate recipe
exactly. Second, the kasugamycin and blasticidin S screen conditions are placed on LB
Lennox agar, not on the buffered M9 the Methods mention in the next sentence ("Media for
kasugamycin and blasticidin S sensitivity was M9 minimal supplemented with metal cations
and buffered at pH 7.5"): their released labels carry no "M9min" prefix, and the Results
attach that medium to the follow-up spot dilutions, "constructing precise deletions of the
ABC-importer operons (Δopp and Δdpp) and analyzing their phenotypes in MG1655, the standard
wild-type background, growing in minimal media at a neutral pH."

### What could not be sourced

- **`n_samples` and `sample_unit`.** The number of colonies behind one fitness-score is in
  neither the paper nor its SI. The only measurement-count statement counts fitness-scores
  per condition, not colonies per fitness-score. The replicate design is deferred to
  Nichols 2011, which is not in the literature mirror, so both fields carry a
  `deferred_pending_source_review` gap whose `resolve_with` names that paper. The deferral
  chain (this Methods -> ref [8] Nichols 2011 -> ref [18] Collins 2006) is the record. A
  comb of the Methods, the three SI figures, the SI reference list, S1 Table and S2 Table
  found nothing else; "replicate" appears in this paper only for the 35S-methionine and
  spot-dilution follow-ups (n = 3 technical, "at least one biological replicate"), never
  for the screen.
- **Per-cell uncertainty.** S1 Dataset releases one number per cell and no dispersion, so
  `environment_response_uncertainty` and `environment_response_se` are
  `not_reported_by_primary` gaps. The score is itself a significance statistic; the
  paper's dispersion statement is the per-condition FDR 5% cutoff band, and the
  per-condition values are not released.
- **The UV dose.** The label reports an exposure TIME ("12 sec") and the paper reports no
  irradiance, so no fluence exists. The radiation perturbation's `magnitude` is `None`
  with a `not_reported_by_primary` gap, and the 12 s exposure survives verbatim in
  `screen_id`. `ConcentrationUnit` carries no time or fluence unit; that is a PR finding,
  not a schema edit.
- **Growth duration, 56 of 57 conditions.** The endpoint is a colony-size rule, the paper
  gives only an approximate 8 hours for the 37 C plates, and the cold conditions
  necessarily ran longer, so `duration_hours` is a `not_reported_by_primary` gap. The 4 C
  survival condition is the exception and stores 840.0 h.
- **The array's cassette.** Shiver names a kanamycin cassette only for the MG1655 operon
  deletions built after the screen. `GenePerturbation` is not a gap carrier, so `cassette`
  is `None` and the absence is recorded here instead of on the record.

### Identifiers

The release labels its columns with gene SYMBOLS, so each resolves straight against the
BW25113 GenBank annotation (GCA_000750555.1) through `reconcile_locus_tags`, and every
perturbation carries `identifier_mapping=DerivedIdentifierMapping(route="gene_symbol")`.
**The ECK crosswalk is not needed and is not used**: no other strain's locus tags appear in
the release.

Measured over the 3,963 distinct labels of the 3,975 columns:

| status | count |
|---|---|
| current | 0 |
| renamed (a symbol resolving to one locus) | 3,649 |
| non_gene_feature (a pseudogene locus) | 130 |
| retired | 182 |
| ambiguous | 2 |

| resolver layer | count |
|---|---|
| locus tag | 0 |
| old locus tag | 0 |
| RefSeq locus tag | 0 |
| gene symbol | 3,635 |
| gene synonym | 146 |
| not found | 182 |

`resolved_fraction` is 0.954, above the loader's stated `MIN_RESOLVED_FRACTION` of 0.95.
3,731 labels are remapped, 48 are kept as given on a merged-locus collision, and 2 are
ambiguous (`rffT` -> BW25113_3793/BW25113_4481, `spr` -> BW25113_2175/BW25113_4043).

### Retention ledger, with the arithmetic

| rule | scope | labels | columns | records |
|---|---|---|---|---|
| `label_is_not_a_gene_deletion` | column | 133 | 134 | 7,638 |
| `label_is_not_in_the_bw25113_annotation` | column | 49 | 49 | 2,793 |
| `label_is_a_fragment_of_a_merged_bw25113_locus` | column | 48 | 48 | 2,736 |
| `label_is_ambiguous_in_bw25113` | column | 2 | 2 | 114 |
| `label_has_two_columns_in_the_release` | column | 11 | 22 | 1,254 |
| `cell_is_blank` | cell | -- | -- | 8,007 |
| **kept** | | | **3,720** | **204,033** |

3,975 - (134 + 49 + 48 + 2 + 22) = 3,720 kept columns, each one distinct locus tag.
3,720 x 57 = 212,040 cells, minus 8,007 blanks = 204,033 records. Against the full grid:
226,575 - 255 x 57 - 8,007 = 204,033. The loader raises if the rule totals and the missing
records disagree.

**The first rule is the interesting one.** The arrayed library is not only Keio: 133 of its
columns are Nichols 2011's essential-gene arrays, and the release marks them with its own
suffixes. Counted over the 134 columns: 114 `-SPA` (sequential peptide affinity tags), 5
`-DAS`/`-DAS+4` (degron alleles), 7 `-kan` (insertions), 6 `{...}` point and indel mutants
(`fabZ{F101Y}`, `lpxC{G210S}`, `bamA{del(64)}`, `bamA{dup(218-219)}`, `ftsA{R286W}`,
`msbA{P18S}`), and 2 columns of the starred `yfiO*`. The S1 Dataset
legend is what makes them the exceptions: "Unless otherwise specified, the mutations are
precise gene deletions." A SPA-tagged hypomorph is not a deletion, and the schema has no
bacterial tagged-allele or degron leaf, so storing one as a
`BacterialDeletionPerturbation` would assert an absence that did not happen. That is a
dataset-level finding, not a loader bug: the essential-gene arm of this screen is
unrepresentable today.

The duplicate-column rule is the other judgment call. Twelve labels head two columns each
(`bamE`, `dppA`, `gnsB`, `hokD`, `mgrB`, `ryfB`, `ydgU`, `yfiO*`, `yhfL`, `ykgO`, `ymjC`,
`ypfM`); `yfiO*` is already gone by rule 1, leaving 11. The two columns of each pair
disagree at every co-present cell (measured: 236 to 256 co-present cells per pair, zero
identical), so they are two independent measurements of two strains, and the release
carries nothing -- no plate, well, or allele id -- that tells them apart. Keeping both puts
two records on one (strain, condition) key; keeping one is an arbitrary choice between two
real measurements. Both go, and the rule says why.

### Compound identity

`resolved_compound` resolves 11 of the 42 distinct agent labels (41 dosed chemicals plus
the two carbon sources, with overlaps) against the pinned `compound_identity_table.json`;
the other 31 come back `UNRESOLVED_PUBLIC` carrying a typed
`deferred_pending_source_review` gap on `inchikey`. No structure is guessed. Issue #726
tracks the curator pass.

**Resolved (11):** DMSO -> dimethyl sulfoxide, 5-fluorouridine, azelaic acid, thiolutin,
urea, acetate -> acetic acid, cinoxacin, ciprofloxacin, sodium fluoride, tetracycline,
glucose -> D-glucose. (The media components agar, tryptone, yeast extract and sodium
chloride also resolve; agar is `RESOLVED_MIXTURE`, the two hydrolysates
`UNDEFINED_MIXTURE`.)

**Unresolved, worth a curator row (31):** 5-methylanthranilic acid, 5-methyltryptophan,
7-azatryptophan, A22, D,L-serine hydroxamate, EDTA, SDS, acriflavine, amoxicillin,
ampicillin, bicyclomycin, bile salts, blasticidin S, casamino acids, chlorhexidine
dihydrochloride, clindamycin, copper(II), deoxycholate, gliotoxin, guanidine hydrochloride,
holomycin, isopentanol, isopropanol, kasugamycin, n-butanol, phenol, pseudomonic acid A,
pyocyanin, rifampicin, silver(II), t-butanol. Three notes for whoever curates them: `SDS` resolves under its full name `sodium dodecyl sulfate` and
`pseudomonic acid A` under `mupirocin`, so those two need only a synonym; `casamino acids`
and `bile salts` are preparations, not single molecules, and belong under
`UNDEFINED_MIXTURE`/`RESOLVED_MIXTURE` rather than an InChIKey; and `silver(II)` and
`copper(II)` are the release's own labels for cation stresses whose counter-ion the paper
never names (Table 1 calls the silver one "silver(lI) nitrate" in OCR, i.e. silver nitrate,
but the condition label is the ion), so a row for them is a curation decision about what
entity the label denotes.

### Media

Two loader-local `Media` objects, both deriving from a `MEDIA_LIBRARY` base, so nothing is
added to `torchcell/datamodels/media.py` and no served store is staled.

- `SHIVER2016_LB_LENNOX_AGAR`, `base_medium="LB_LENNOX"`, solid. Tryptone and yeast extract
  are the shared Lennox components (Shiver's 1% and 0.5% (w/v) are 10 g/L and 5 g/L, the
  Lennox amounts). The salt is NOT reused: Shiver states 90 mM where the library row is
  5 g/L, which is 85.6 mM, so the component is restated at the stated molarity with
  Shiver's own quote.
- `SHIVER2016_M9_MINIMAL_AGAR`, `base_medium="M9"`, solid: the shared salts-only M9 plus
  2% (w/v) bacto agar, with the carbon source held out as the varied factor (the Tong 2020
  convention).

### Raw mirror

`$DATA_ROOT/torchcell-raw/shiverChemicalGenomicScreenNeglected2016/data/pgen.1006124.s005.txt`,
sha256 `9edcf6f6f34b957661b23f2a5e7f6f14e25c387ab825e0b4c79a5f995f8de690`, 20,577,266 bytes.
`RetrievalMethod.pmc_cloud`, `torchcell.literature.retrieve.pmc_cloud_object`, key
`PMC4927156.1/pgen.1006124.s005.txt`, retrieved 2026.10.07. The re-fetch is sha256-identical
to the copy the literature mirror already held under the same key, so the retrieval is
scriptable and reproducible, with no manual recipe anywhere in the chain.

Deliberately NOT mirrored, both named in the manifest's `si_expected`: S1 Table
(`si/si6.docx`, sha256 `6521f4ef...`), which already lives in the literature mirror and is
the parse cross-check; and the Dryad deposit (doi:10.5061/dryad.f3kc0, plate images, Iris
output, GSEA inputs), because S1 Dataset already releases the scored matrix the loader
consumes.

### Parse cross-check against S1 Table

S1 Table prints a "10 C fitness-score" column to one decimal for the cold-sensitive genes.
Against the `10C [-] {4}` row of S1 Dataset, ten of ten agree at the released precision,
which independently fixes the matrix orientation, the condition label and the sign:

| gene | S1 Dataset | S1 Table |
|---|---|---|
| typA | -10.456 | -10.5 |
| ihfB | -8.732 | -8.7 |
| ihfA | -8.679 | -8.7 |
| dinJ | -8.345 | -8.3 |
| rbfA | -6.903 | -6.9 |
| deaD | -6.701 | -6.7 |
| hfq | -5.741 | -5.7 |
| crr | -4.830 | -4.8 |
| ycbK | -4.061 | -4.1 |
| nfuA | -4.041 | -4.0 |

The data-gated test `test_the_ten_degree_row_agrees_with_the_cold_sensitive_table` re-runs
this against both mirrors, verifying S1 Table's sha256 first.

### Build

`PYTHONPATH=$PWD python -m torchcell.database.build_dataset_lmdb --dataset EnvChemgenShiver2016Dataset`

| quantity | value |
|---|---|
| records | 204,033 |
| gene-set size | 3,720 |
| experiment references | 57 |
| wall time | 111 s |
| store | `$DATA_ROOT/data/torchcell/ecoli_env_chemgen_shiver2016` |

`python -m torchcell.provenance.build_manifest` reads the store `fresh` with no drift.

### Verification, L0 to L4

`verify_build` streams the LMDB once through `verify_environment_response_dataset_streaming`
with the BW25113 genome as both the canonical-name resolver and the L4 universe (every
GenBank locus of the assembly), and writes `preprocess/verification_report.json`. All 17
rows PASS.

| level | rule | result |
|---|---|---|
| L0 | structural | 204,033 records validated |
| L1 | count | observed 204,033, expected 204,033 |
| L1 | pair_uniqueness | 204,033 unique (study, strain, condition) records, one each |
| L1 | provenance_gaps | 1,368,604 documented gaps over 204,033/204,033 records; deferred fields `inchikey`, `n_samples`, `sample_unit` |
| L1 | canonical_gene_names | 3,720 systematic names, one spelling each, each current; 89 pseudogene loci the genome resolves to themselves |
| L2 | value_fidelity | 204,033 values checked |
| L2 | se_nonnegative | 0 values checked (no SE is released) |
| L2 | uncertainty_sanity | 0 labeled uncertainties, 0 records with n_samples >= 2 and no uncertainty |
| L3 | measurement_type_consistent | single measurement_type `z_score` |
| L3 | reference_zero | numeric rule: reference response == 0 for all 204,033 records |
| L3 | environment_perturbed | all 204,033 carry an edit; baseline temp 37.0, baseline media the LB Lennox agar |
| L3 | compound_identity | 75,124 references carry a structure identifier, 136,599 declare a typed gap (31 distinct compounds) |
| L3 | media_compound_identity | 526,314 references carry a structure identifier, 0 gaps |
| L3 | media_membership | 204,033 records on a medium deriving from a library base (2 distinct media) |
| L4 | gene_containment_sgd | 1.000 of 3,720 measured genes are in the pinned gene set (>= 0.9) |
| L4 | current_genome_genes | every one of the 3,720 systematic names is a gene of the current genome |

`pair_uniqueness` passing is the interesting row: the environment signature now includes
the environment perturbations and the phenotype's `screen_id`, so a per-compound screen
with repeated compounds is expected to pass, and it does.

Two cosmetic quirks of the shared rules, both pre-existing and not specific to this row.
`gene_containment_sgd`'s message says "S288C reference genes" whatever gene set it is
handed; here it is the BW25113 GenBank loci, which is what `verify_build` passes. And the
`provenance_gaps` row also counts 4,040,985 undeclared `None` values, dominated by the
`Compound` identity fields the shared table leaves empty.

### Tests and checks

`tests/torchcell/datasets/ecoli/test_shiver2016.py`.

- Hermetic: `59 passed, 3 skipped` (the three skips are the `--data` bucket).
- With `--data`: `62 passed`. The data tests audit all 19 `SourcedValue`s against the
  pinned `paper.md`, pin the measured column histograms on the real release, and re-run
  the S1 Table cross-check.
- Hermetic coverage of the loader module: 96% (378 statements, 15 missed: the
  drop-accounting guard, `_bw25113`, and `main`).
- `ruff check`, `ruff format --check` and `python -m mypy` clean on the loader, the test
  and the package `__init__`.
- `pytest tests/torchcell/datasets/test_dataset_registry.py tests/torchcell/database/test_build_dataset_lmdb.py tests/torchcell/datasets/ecoli`: 564 passed, 196 skipped (before the new guard tests were added).
