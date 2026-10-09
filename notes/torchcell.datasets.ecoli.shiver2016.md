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

## 2026.10.07 - BioCypher adapter and its enable-list

`torchcell/adapters/shiver2016_adapter.py`, class `EnvChemgenShiver2016Adapter`, conf
`torchcell/adapters/conf/ecoli_env_chemgen_shiver2016_adapter.yaml`. The loader registered
its dataset class with no adapter, so the two set-level gate tests in
`tests/torchcell/adapters/test_bacterial_adapters.py` failed: every registered E. coli and
P. putida class must be in `dataset_adapter_map` and must be named in `kg_bacteria.yaml`.
The adapter closes both. Wired in `torchcell/adapters/__init__.py`,
`torchcell/knowledge_graphs/dataset_adapter_map.py` (now 72 pairs, the pin in
`tests/torchcell/knowledge_graphs/test_build_time_projection.py` bumped to match) and
`torchcell/knowledge_graphs/conf/kg_bacteria.yaml` (now 21 datasets).

The module follows the landed bacterial pattern: one dataset class per module, and the
conf named after the loader's root slug, which is what `kg_manifest.dataset_adapter_files`
regexes out of the module source (issue #743) and what
`test_each_conf_is_named_after_the_dataset_root_slug` checks.

### Enable-list, one line of why per family

| family | enabled | why |
|---|---|---|
| experiment, genotype, dataset, publication, genome | yes | the record spine of every `ExperimentDataset`: one `BacterialEnvironmentResponseExperiment` per (strain, condition), its `Genotype`, the `AssemblyReferenceGenome` of BW25113 on the reference, and the one `Publication` |
| `bacterial perturbation` | yes | each record's single leaf is a `BacterialDeletionPerturbation`, which is a bacterial leaf, so it is served under `bacterial perturbation` and never under the yeast `perturbation` class |
| `environment`, `media`, `temperature` (and their references) | yes | both loader-local plates appear (46 references on LB Lennox agar, 11 on M9 minimal agar) and every condition carries a `Temperature`: 53 references at 37 C, two at 10 C, one at 25 C and one at 4 C |
| `environment perturbation` (and its reference) | yes | measured over the 57 references: 48 `SmallMoleculePerturbation` plus 14 `EnvironmentPhysicalPerturbation` (11 `carbon_source`, 3 `radiation`). One method pair serves both families, since `_environment_perturbation_node_from` projects compound/concentration and agent/magnitude onto the same columns |
| `environment response phenotype` (and its reference) | yes | the phenotype class of `BacterialEnvironmentResponseExperiment`; `_adapter_init_harness.PHENOTYPE_METHOD` derives the required method from the loader's `experiment_class`, so no other phenotype method is legal here |
| `crispr construct` | no | no record carries one: the arrayed library is deletions, and the adapter's left-off-family check confirms the method emits nothing over the store |
| `phage perturbation` | no | the screen has no phage challenge, and `phage perturbation` is a separate served class, so enabling it would write nothing |

The three temperature-only conditions (`10C [-] {4}`, `25C [-] {4}`,
`4C survival [5 wk] {4}`) carry no environment perturbation at all: temperature is a slot
on `Environment`, not an edit, and the temperature pair already serves it. That is why the
env-perturbation pair is enabled on the strength of the other 54 conditions rather than
all 57.

### Tests and checks

Paired test `tests/torchcell/adapters/test_shiver2016_adapter.py`, five tests over the
shared bacterial harness: the exact conf the constructor loads and the base-adapter
wiring, the refusal naming the exact conf path before `wandb.init`, every conf method
registered on `CellAdapter` with every node class it can emit declared in
`torchcell_schema_config.yaml`, the gate resolving the dataset to its own module and conf,
and a `--data` test running every enabled method over the dev store.

- `pytest tests/torchcell/adapters tests/torchcell/knowledge_graphs -q`:
  `699 passed, 21 skipped, 305 warnings in 58.99s`.
- `DATA_ROOT=/scratch/projects/torchcell-scratch pytest tests/torchcell/adapters/test_shiver2016_adapter.py --data -k dev_store`: 1 passed. The emitted
  graph over the first 200 records plus all 57 references is closed (no dangling edge
  endpoint), every label and property is declared, and the two left-off families emit
  nothing.
- `ruff check`, `ruff format --check` and `python -m mypy` clean on the adapter, its test
  and every wiring file.
- Admission check, no `--neo4j-uri` (the dataset is not served, so the store is never
  read): `EnvChemgenShiver2016Dataset -> BLOCKED`, with `dev LMDB: fresh` and
  `served: no (new dataset)`, and only the four store-wide blockers already recorded for
  the 20 landed bacterial adapters in
  [[plan.bacteria-ontology-genome]]: 11 served datasets whose schema closure changed, the
  served `crispr construct` class changed, plumbing drift on `_crispr_construct_node_from`,
  and the value surface (`compound_identity.py`, `compound_identity_table.json`,
  `media.py`). None of those files is touched here, so this row enters the same full
  rebuild as the other 20.

The loader's three schema findings are issue #749 and are not addressed here.

## 2026.10.09 - The allele columns are stored, the UV dose has a field, and the s_score question is settled (issue #749)

Three findings of this loader, settled now that `BacterialMarkedAllelePerturbation`,
`BacterialDegronPerturbation` and `PhysicalExposurePerturbation` exist in `schema.py`.

### 1. 126 of the 134 non-deletion columns are stored

The rule `label_is_not_a_gene_deletion` is gone, replaced by
`label_is_a_point_or_indel_mutant`, which keeps only the 8 columns that need issue
#731's leaf. The buckets, measured by applying each suffix of `ALLELE_SUFFIXES`
separately to the 3,975 released header fields:

| suffix | columns | labels | leaf | field stored |
|---|---|---|---|---|
| `-SPA` | 114 | 114 | `BacterialMarkedAllelePerturbation` | `tag="SPA"` |
| `-kan` | 7 | 7 | `BacterialMarkedAllelePerturbation` | `cassette="kan"` |
| `-DAS` / `-DAS+4` | 5 | 5 | `BacterialDegronPerturbation` | `degron="DAS"` / `"DAS+4"` |
| `{...}` / `*` | 8 | 7 | none, still dropped | issue #731 |

The 8 that stay dropped are `bamA{del(64)}`, `bamA{dup(218-219)}`, `fabZ{F101Y}`,
`ftsA{R286W}`, `lpxC{G210S}`, `msbA{P18S}` and `yfiO*` (two columns), 439 non-blank
cells. A substituted or indel allele is #731's leaf and inventing a second one here
would file one kind of record under two classes.

### Measured before and after, from the rebuilt dev store

| quantity | before | after |
|---|---|---|
| released cells (3,975 x 57) | 226,575 | 226,575 |
| kept columns | 3,720 | **3,846** (3,720 deletions + 126 alleles) |
| distinct locus tags | 3,720 | 3,838 |
| stored records | 204,033 | **210,998** |
| `label_is_not_a_gene_deletion` | 134 columns, 7,638 records | rule removed |
| `label_is_a_point_or_indel_mutant` | n/a | 8 columns, 456 records |
| `label_is_not_in_the_bw25113_annotation` | 49 columns, 2,793 | 49 columns, 2,793 |
| `label_is_a_fragment_of_a_merged_bw25113_locus` | 48 columns, 2,736 | 48 columns, 2,736 |
| `label_is_ambiguous_in_bw25113` | 2 columns, 114 | 2 columns, 114 |
| `label_has_two_columns_in_the_release` | 22 columns, 1,254 | 22 columns, 1,254 |
| `cell_is_blank` | 8,007 cells | 8,224 cells |

6,965 new records: 126 columns x 57 conditions = 7,182 cells, of which 6,965 are
non-blank. The gene set rises only from 3,720 to 3,838, which is the point: an allele
column is another STRAIN of a gene the deletion columns already name, not another gene.

### Two mechanics the extension needed

**A second resolution pass.** A suffixed label is not a symbol the BW25113 annotation
carries, so the reconciliation leaves all 134 of them in `retired_kept`, which is how
the old rule found them at all. The suffix is now stripped and the BASE symbol resolved
instead. All 121 distinct base symbols resolve to exactly one locus (the loader raises
if one does not); each label's base symbol and the tag it reached is written to
`preprocess/identifier_reconciliation.json` under `allele_base_symbol_resolutions`, so
the second pass is auditable rather than implicit. An allele column whose base symbol is
itself a merged-locus fragment or ambiguous inherits that rule; measured on the release,
none of the 121 is in either set, so that is the invariant rather than a correction.

**The uniqueness key is the strain, not the locus.** The build used to raise if two kept
columns claimed one locus tag. The arrays break that honestly: `lpxC`, `lpxC-SPA` and
`lpxc-kan` are three strains of one gene, and `imp-DAS` and `imp-DAS+4` are two strains
on `BW25113_0054` differing only in their degron. The post-condition now keys on the
`(locus tag, allele kind, allele token)` triple. The same thing had to be added to the
environment-response verifier's strain signature
(`torchcell/verification/environment_response.py`), which previously keyed on
`(systematic_gene_name, perturbation_type, perturbed_gene_name)` plus the CRISPR and
#507 discriminators: the two degrons agree on all three, so `tag`, `terminus`, `degron`,
`cassette` and `insertion_site` join the key. Adding a field can only split a group,
never merge two, so every dataset that passes uniqueness today still passes.

`perturbed_gene_name` is the BASE symbol, not the column label. `thrA-SPA` names a
strain and the gene it perturbs is `thrA`; writing the label there put two common-name
spellings on one locus tag, which the verifier's canonical-name rule flagged (measured:
3 genes, 398 records, on the synthetic build). The released label survives verbatim on
`identifier_mapping.source_identifier`.

### What the allele records deliberately do NOT carry

Every one of these is a mirrored-source question, not a modeling choice:

- `terminus` is `None` on all 119 marked alleles and all 5 degrons. SPA is a C-terminal
  tag in the paper that BUILT that collection (Butland 2008, which IS mirrored), but
  Shiver defers its whole array to Nichols 2011 ("the same methodology as reported
  previously [8]") and Nichols 2011 has no mirror entry (#691), so no mirrored sentence
  places the fusion on THESE strains. This is also why
  `BacterialDegronPerturbation.terminus` is optional rather than required.
- `allele_effect="not_stated"`. The S1 Dataset legend says only "Unless otherwise
  specified, the mutations are precise gene deletions", never that a tagged allele is
  hypomorphic. That is exactly what the leaf's third vocabulary member exists for; a
  bool would have had to be written `False`, which asserts the opposite.
- `collection` is `None`. The deletion columns carry "KEIO deletion library" because the
  Methods name it; the only thing this paper says about the rest of the array is that it
  screened "the previously screened library [8,10]", which names no strain set.
- `protease`, `adaptor` and `inducing_condition` are `None` on the degrons: this paper
  names none of them.

### 2. The UV dose is stored, as a time (#749 item 3)

The three irradiated conditions release their dose in the label's square brackets, read
verbatim from the pinned matrix
(`pgen.1006124.s005.txt`, sha256
`9edcf6f6f34b957661b23f2a5e7f6f14e25c387ab825e0b4c79a5f995f8de690`), rows 272, 275, 276
of column 1:

```
M9min glucose+UV [0.2% (w/v); 12 sec] {4}
UV+10C [12 sec] {4}
UV [12 sec] {4}
```

The governing legend is the S1 Dataset caption, and it is the sentence that states the
mismatch:

> Conditions are labelled with the condition name, concentration in square brackets '[]',
> and "batch" number in curly brackets.

The brackets hold a concentration for 54 conditions and a time for these three, and
`EnvironmentPhysicalPerturbation.magnitude` is a `Concentration`, so the 12 seconds used
to survive only inside the `screen_id` string. It is now a
`PhysicalExposurePerturbation` with `exposure_duration_seconds=12.0`: the dose in its
own field in the unit its name states, the same move `PhagePerturbation` made for a
multiplicity of infection rather than adding a `ConcentrationUnit` member that would
make an exposure time look like a dose.

**No irradiance or fluence is released anywhere in the mirror.** Swept `paper.md`, the
`paper.pdf` text layer, `paper_middle.json`, `paper_content_list.json`, the extracted
text of `si4.docx`, `si6.docx` and `si7.docx`, and the 20.5 MB raw matrix for `irradian`,
`fluence`, `mJ/cm`, `J/m2`, `germicid`, `Stratalinker`, `crosslink`, `lamp`, `254 nm`
and `ultraviolet`: zero hits in all of them. The word "UV" itself appears exactly once in
`paper.md`, and it is the protease name `HslUV`. Both dose fields are therefore typed
gaps whose `resolve_with` names Nichols 2011, not plain `not_reported_by_primary` with no
target: the UV series is a Nichols condition re-run (batch 0 of the same release carries
`UV [6sec]` through `UV [24sec]`), so the irradiance is deferred there rather than
unreported by anyone.

### 3. No `MeasurementType.s_score` member, and the reason is sourcing (#749 item 2)

This is a correction to what the previous section of this note and the loader docstring
asserted. **The claim "the fitness-score IS the Collins/Nichols S-score" was an
unlabeled inference, and it is now labeled as a hypothesis.**

Measured on two independent renderings of the same paper, `paper.md` (sha256
`4a1bec97d10f6ef1c02c99329f7adce0d79623819fcf421c098058a7c82ced72`) and the publisher
PDF's text layer: the strings `S score`, `S-score`, `Z score`, `z-score` and `sscore`
appear **nowhere**, case-insensitively. The paper says "fitness-score", 20 times. S1
Table's own column headers are `10˚C fitness-score` and `16˚C score`. The released
matrix is a condition column followed by 3,975 bare gene symbols, with no statistic
named anywhere in its 20.5 MB. The authors' own machine name for the pipeline is
`fitness_score` (`https://github.com/AnthonyShiverMicrobes/fitness_score.git`).

What IS sourced verbatim, from Results, "The chemical-genomic screen substantially
expands known connections in E. coli":

> These fitness-scores represent the statistical significance of a change in colony size
> for a particular condition, with negative and positive fitness-scores representing
> sensitivity and resistance, respectively.

and, in the same paragraph:

> We assigned fitness-scores to each mutant, using an in-house software package that
> built upon previous analyses [8,18] by implementing additional filtering and
> normalization steps to improve data quality (Methods).

and from Methods, "Data collection and processing":

> The chemical genomics screen was conducted using the same methodology as reported
> previously [8] with few modifications.

and, in the same paragraph:

> Steps added to the original analysis pipeline [18] include ... variance normalization
> of the data (to improve reproducibility of measurements between plates).

[8] is Nichols 2011 and [18] is Collins 2006, and **neither is in the literature
mirror**, so the S-score formula cannot be read from our own documentation. The decision
that follows: no `s_score` member is added, because a `MeasurementType` member named
after a statistic whose definition we cannot read would be a label with no provenance
behind it. `z_score` stays, on the evidence we do hold: the Methods name the
standardizing step outright, and `z_score`'s own definition is a "standardized
fitness/growth deviation". `MeasurementType`'s docstring now records that it covers this
family and why there is no `s_score`.

**Hypothesis (untested, and untestable from our mirror): the released number is the
Collins S-score, a modified t-statistic.** It follows from the deferral chain Shiver ->
[8,18] -> Collins 2006, not from any sentence we hold. Mirroring Nichols 2011 and
Collins 2006 (#691) is what would settle it.

### L0 to L4 on the rebuilt store

`verify_build` over the 210,998-record store: `ecoli_env_chemgen_shiver2016: PASS` on all
17 checks. `python -m torchcell.provenance.build_manifest` reads it `fresh`.

| level | rule | result |
|---|---|---|
| L0 | structural | 210,998 records validated |
| L1 | count | observed 210,998, expected 210,998 |
| L1 | pair_uniqueness | 210,998 unique (study, strain, condition) records, one each |
| L1 | provenance_gaps | 1,426,790 documented gaps over 210,998/210,998 records; deferred fields `inchikey`, `n_samples`, `sample_unit` |
| L1 | canonical_gene_names | 3,838 systematic names, one canonical spelling each, each current; 89 pseudogene loci the genome resolves to themselves |
| L2 | value_fidelity | 210,998 values checked |
| L2 | se_nonnegative | 0 values checked (no SE is released) |
| L2 | uncertainty_sanity | 0 labeled uncertainties; 0 records with n_samples >= 2 and no uncertainty |
| L3 | measurement_type_consistent | single `z_score` |
| L3 | reference_zero | reference response == 0 for all 210,998 |
| L3 | environment_perturbed | all 210,998 carry an edit; baseline temp 37.0, baseline media the LB Lennox agar |
| L3 | compound_identity | 77,736 references carry a structure identifier, 141,328 declare a typed gap (31 distinct compounds) |
| L3 | media_compound_identity | 544,333 references carry a structure identifier, 0 gaps |
| L3 | media_membership | 210,998 records on a medium deriving from a library base (2 distinct media) |
| L4 | gene_containment_sgd | 1.000 of 3,838 measured genes are reference genes |
| L4 | current_genome_genes | all 3,838 names are genes of the current genome |

`pair_uniqueness` passing is again the interesting row, and for a new reason: the
verifier's strain signature now carries `tag`, `terminus`, `degron`, `cassette` and
`insertion_site`, which is what keeps `imp-DAS` and `imp-DAS+4` (two strains on
`BW25113_0054` differing only in their degron) from colliding on one key.
