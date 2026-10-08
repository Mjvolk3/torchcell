---
id: sispfhb4299so445rso0owm
title: Schmidt2016
desc: ''
updated: 1791407471535
created: 1791407471535
---

## 2026.10.07 - Loader, sourcing and build

Row 32 of the fifty bacterial datasets. Schmidt et al. 2016, *The quantitative and
condition-dependent Escherichia coli proteome*, Nat Biotechnol 34:104,
doi:10.1038/nbt.3418, PMID 26641532, citation key
`schmidtQuantitativeConditiondependentEscherichia2016`. The loader is
`torchcell/datasets/ecoli/schmidt2016.py`, the class is `ProteomeSchmidt2016Dataset`,
and the test module is `tests/torchcell/datasets/ecoli/test_schmidt2016.py`.

### What is stored

| field | value |
|---|---|
| experiment class | `BacterialProteinAbundanceExperiment` |
| reference class | `BacterialProteinAbundanceExperimentReference` |
| records | 14, one per loaded BW25113 growth condition |
| protein keys per record | 2,329 (2,034 on the Xylose, Mannose and Fructose records) |
| stored quantity | protein copies per cell |
| `measurement_type` | `absolute_protein_copies_per_cell_label_free_sid_anchored` |
| uncertainty | standard error in copies per cell, derived exactly from the three released replicate columns |
| `n_replicates` | 3 for a dataset-2 protein, 1 for a dataset-1 protein |
| assembly pin | `ecoli_K12_BW25113_ASM75055v1` / `GCA_000750555.1` |
| identifier namespace | `ecoli_k12_bw25113_locus_tag` |
| consumed file | `$DATA_ROOT/torchcell-raw/<key>/data/si2.xlsx` |

### The per-protein quantity stored, with its verbatim definition

Table S6 releases three per-protein quantities per condition and they are not
interchangeable: `Protein copies/cell`, `Protein Mass (fg) / Cell` and
`Coefficient of Variance (%) Between Biological Triplicates (only available for
Dataset 2)`. The loader stores **protein copies per cell**. Its definition, verbatim from
`paper.md` (sha256 `67bedae8f934421086c23b7fa9582b0950a07206bffe7c4ffdd11e4d003a8710`),
Online Methods, *Absolute quantification of selected proteins by targeted LC-MS*:

> Based on the number of cells counted by fluorescenceactivated cell sorting for each
> sample, absolute abundances for the selected proteins (in copies/cell) could be
> calculated across all samples in both data sets (Supplementary Tables 2 and 3).

How the proteome-wide values were extrapolated from those 41 anchors, Online Methods,
*Proteome-wide estimation of protein abundances*:

> The absolute protein concentrations determined for 41 glycolytic proteins were aligned
> with the summed protein intensities as provided by the Progenesis LC-MS software (v4.0,
> Nonlinear Dynamics Limited) divided by the number of expected tryptic peptides as
> recently specified4,25.

The mass block is not stored: it is the same number times the released molecular weight.

### Sourcing table

Every quote below is a verbatim substring of the pinned bytes. `paper.md` carries sha256
`67bedae8f934421086c23b7fa9582b0950a07206bffe7c4ffdd11e4d003a8710`; `si2.xlsx` carries
sha256 `3280a13ff67a73f25440cff6ee73fb99b5ce3ef57854213dbbf6272be241912f`, and its quotes
are row renderings of the `CONTENT_AND_ABBREVIATIONS` sheet (a row's non-empty cells
joined by " | "), the convention `media.py` already uses for binary sources. The data test
`test_every_quote_is_verbatim_in_its_pinned_mirror` re-checks all 21 against the bytes.

| value | source | section | quote (abridged where long) |
|---|---|---|---|
| reference strain BW25113 | paper.md | Online Methods, *Strains and plasmids* | "The Escherichia coli K-12 strain BW25113 (genotype: F-, ∆(araD-araB)567, ∆lacZ4787(:rrnB-3), λ-, rph-1, ∆(rhaD-rhaB)568, hsdR514)19 was used to generate the proteome map for all 22 conditions." |
| the other two strains, and only two conditions each | paper.md | Online Methods, *Strains and plasmids* | "Additionally, the proteome for the glucose and LB condition was also determined for the strains MG1655 ... and NCM3722 ..." |
| 22 conditions, biological triplicates | paper.md | Results | "We grew E. coli BW25113 (ref. 19) under 22 different growth conditions in biological triplicates." |
| n = 3 in dataset 2 | paper.md | Online Methods, *Sample preparation* | "All samples of data set 2 were prepared in biological triplicates." |
| n = 1 in dataset 1 | si2.xlsx | Table S7 title | "... (since this dataset contains no replicate measurements, no statical tests were performed. Therefore, protein quantities in dataset 2 are of higher confidence and should be preferred)" |
| the eleven carbon sources and their g/L | paper.md | Online Methods, *Media* | "The following carbon sources and concentrations were used: acetate (sodium acetate, 3.5 g/L), fumarate (disodium fumarate, 2.8 g/L, galactose (2.3 g/L), glucose (5 g/L), glucosamine (2.1 g/L), glycerol (2.2 g/L), pyruvate (sodium pyruvate, 3.3 g/L), sucnate (disodium succinate hexahydrate, 5.7 g/L, fructose (5 g/L), mannose (5 g/L) and xylose (5 g/L)." (the OCR's LaTeX and the source's "sucnate" are kept) |
| chemostat glucose is 1 g/L, not 5 | paper.md | Online Methods, *Media* | "For chemostat growth only 1 g/L of glucose was used." |
| osmotic stress is 50 mM NaCl | paper.md | Online Methods, *Media* | "Glucose minimal medium for the cells grown with osmotic stress was supplemented with NaCl to a concentration of 50 mM" |
| pH stress is pH 6, set with HCl | paper.md | Online Methods, *Media* | "for the cells grown with pH stress, fuming hydrochloric acid was titrated to the medium until a pH of 6 was reached." |
| 37 C, shaken, aerobic | paper.md | Online Methods, *Cultivation* | "For the batch cultures, ... grown at 37 °C, orbital shaking at 300 r.p.m. and 5-cm shaking diameter (ISF-4-V, Kühner)." |
| temperature stress is 42 C | paper.md | Online Methods, *Cultivation* | "The cells undergoing temperature stress were grown at 42 °C" |
| the glycerol + amino acid recipe | paper.md | Online Methods, *Media* | "The glycerol + amino acid medium was made by supplementing the media with glycerol to a concentration of 2.2 g/L individual amino acids: ... and uracil 20.5 mg/L (0.2 mM)." |
| stationary phase is 1 or 3 days | paper.md | Online Methods, *Cultivation* | "Starved cells were continuously shaken after reaching stationary phase for either 1 or 3 d." |
| the chemostat dilution rates | paper.md | Online Methods, *Cultivation* | "Cells grown in a chemostat were inoculated from a preculture to an OD of 0.1 and allowed to grow in batch mode to an OD of around 0.8 before dilution (rates: 0.12, 0.2, 0.35, 0.5) was started52." |
| the search database carried contaminants | paper.md | Online Methods, *Protein identification and label-free quantification* | "The database consists of 4,431 E. coli proteins as well as known contaminants such as porcine trypsin, human keratins and high abundant bovine serum proteins (Uniprot), resulting in a total of 10,388 protein sequences." |
| dataset 2 is the preferred arm | paper.md | Online Methods, *Proteome-wide estimation of protein abundances* | "Owing to the higher number of quantified membrane proteins, ... protein quantities obtained from data set 2 were employed for all quantitative analysis carry out in this study." |
| Table S6 is the combined table | si2.xlsx | Table S6 title | "Final table with combined global absolute abundance estimations from both datasets including functional annotations using cluster of orthologues groups (COG)" |
| Table S8 carries the replicates | si2.xlsx | Table S8 title | "Global relative quantification of all proteins identifed in dataset 2 ... inlcuding statiscal analysis from biological triplicates ..." |
| Table S25 is the sample map | si2.xlsx | Table S25 title | "Sample names for the individual replicates analyzed for each growth condition and strain included in this study" |

The media objects were already in the library when this loader was written: `LB` (Miller)
and `M9_SCHMIDT2016` (carbon-free), both built from these same quotes, with
`M9_SCHMIDT2016` listed in `CARBON_FREE_MEDIA` and documented as "the carbon source is
the variable across the eleven carbon-source conditions". The loader adds nothing to
`torchcell/datamodels/media.py` and this branch does not touch that file.

PMID 26641532 was resolved from the PMC id converter
(`https://pmc.ncbi.nlm.nih.gov/tools/idconv/api/v1/articles/?ids=PMC4888949&format=json`),
which returned `{"doi":"10.1038/nbt.3418","pmcid":"PMC4888949","pmid":26641532}`; the DOI
agrees with the library manifest, which is why the PMID is recorded rather than guessed.

### The uncertainty: an exact SE, not the released CV

`ProteinAbundancePhenotype.protein_abundance_se` is a standard error and the release
publishes a coefficient of variation. Three facts, each MEASURED on the pinned workbook,
turn one into the other without an assumption, and the first two are re-asserted at build
time (`preprocess/released_statistics_check.json`):

1. `medianNormInt_<group>` of Table S8 equals the median of that group's three `normInt_`
   replicate columns for every released cell. Measured over all 47,334 dataset-2 cells of
   the 23 BW25113 groups: worst relative deviation **0.0** (exact).
2. `cv_<group>` equals `100 * stdev / mean` of those same three columns. Over the same
   47,334 cells the deviation's median is 3.8e-10, its 99th percentile 1.1e-8 and its
   99.9th percentile 5.6e-8, which is float rounding in a ten-significant-digit export.
   The only 23 cells above 1e-6 all belong to `P63284`, the one accession Table S6 files
   on two rows; this loader drops it.
3. Within a condition the stored copies/cell is a per-protein multiple of that protein's
   `medianNormInt`. Measured as a ratio of ratios against the glucose arm,
   `(copies[i,c]/copies[i,glucose]) / (medianNormInt[i,c]/medianNormInt[i,glucose])` is
   one constant per condition for all 2,057 shared dataset-2 proteins and all 21
   condition pairs, with a maximum relative deviation of **9.3e-10**. The constant is a
   per-condition scalar (the volume-corrected total protein mass per cell), and the
   per-protein factor (one over the expected tryptic-peptide count) cancels.

Hence `scale[i,c] = copies[i,c] / medianNormInt[i,c]` and

```
SE[i,c] = scale[i,c] * stdev(three normInt replicates) / sqrt(3)
```

which is exact arithmetic on the released numbers. A dataset-1 protein has no replicate
measurement, so its SE is `nan` and its `n_replicates` is 1; the verifier's supplementary
`se_matches_replicate_count` row asserts a finite SE accompanies `n_replicates == 3` and
`nan` accompanies `n_replicates == 1`, for all 31,721 stored values.

**Why the released CV is not simply divided by sqrt(3) instead.** The CV is
`stdev/mean` while the stored copies/cell is a scalar multiple of the **median** of the
three replicates, so `CV * copies / sqrt(3)` would be off by the per-cell mean-to-median
ratio. The replicate columns are released, so the exact SE is available and the
approximation is not taken.

### A header fault in the released table, found by measurement

Table S6's CV block holds **LB's** coefficient of variation under the header `Glucose`
and **glucose's** under the header `LB`. Its other 20 CV headers are correct, and its
copies/cell and mass headers are all correct. The proof is positional: Table S6's 22 CV
values are Table S8's 23 `cv_` values with `cv_Glucose.2` removed, in order, and
recomputing each CV from Table S8's replicate columns through the Table S25 sample map
reproduces S8's labels exactly. The swap then holds for all 2,058 dataset-2 rows with
zero exceptions (measured; the build asserts it over the 2,056 rows whose cells survive
the accession-duplicate rule).

This loader reads no Table S6 CV column, so nothing stored depends on the finding.
`check_cv_header_swap` asserts it at build time so it stays pinned to the bytes rather
than to this note; a corrected re-export would stop the build and say so.

### Per-condition genotype-versus-environment decisions

**Every loaded condition is wild-type BW25113 and none is a deletion strain.** This
corrects an expectation the task carried. The paper does use three Keio deletion strains,
and they appear ONLY in the N-alpha-acetylation analysis: Supplementary Table 20
(acetylated peptide spectrum matches from WT, ΔrimI, ΔrimJ, ΔrimL in acetate medium),
Supplementary Table 21 (the same in glucose medium) and Supplementary Table 24 (their
growth rates). None of those tables carries a per-protein abundance, so no abundance
record has a genotype perturbation and `Genotype(perturbations=[])` is correct for all 14.
The varying axis is entirely the environment.

| Table S6 condition | strain | genotype | environment | loaded |
|---|---|---|---|---|
| Glucose | BW25113 | wild type | `M9_SCHMIDT2016` + carbon_source glucose 5 g/L, 37 C | reference |
| LB | BW25113 | wild type | `LB`, 37 C, no perturbation | yes |
| Acetate | BW25113 | wild type | + carbon_source sodium acetate 3.5 g/L | yes |
| Fumarate | BW25113 | wild type | + carbon_source disodium fumarate 2.8 g/L | yes |
| Glucosamine | BW25113 | wild type | + carbon_source glucosamine 2.1 g/L | yes |
| Glycerol | BW25113 | wild type | + carbon_source glycerol 2.2 g/L | yes |
| Pyruvate | BW25113 | wild type | + carbon_source sodium pyruvate 3.3 g/L | yes |
| Xylose | BW25113 | wild type | + carbon_source xylose 5 g/L | yes |
| Mannose | BW25113 | wild type | + carbon_source mannose 5 g/L | yes |
| Galactose | BW25113 | wild type | + carbon_source galactose 2.3 g/L | yes |
| Succinate | BW25113 | wild type | + carbon_source disodium succinate hexahydrate 5.7 g/L | yes |
| Fructose | BW25113 | wild type | + carbon_source fructose 5 g/L | yes |
| Osmotic-stress glucose | BW25113 | wild type | glucose + `SmallMoleculePerturbation` NaCl 50 mM | yes |
| 42°C glucose | BW25113 | wild type | glucose, `Temperature` 42 C | yes |
| pH6 glucose | BW25113 | wild type | glucose + physical factor pH 6.0, agent hydrochloric acid | yes |
| Glycerol + AA | BW25113 | wild type | M9 + glycerol + 21 named supplements | no |
| Chemostat µ=0.5 / 0.35 / 0.20 / 0.12 | BW25113 | wild type | M9 + glucose 1 g/L, continuous culture | no |
| Stationary phase 1 day / 3 days | BW25113 | wild type | M9 + glucose 5 g/L, 1 or 3 d into stationary | no |

The stress axes follow the schema's own rule: NaCl is a single named compound, which
`EnvironmentPhysicalPerturbation`'s docstring sends to `SmallMoleculePerturbation`;
temperature lives on `Environment.temperature`; pH is a `PhysicalFactor`. The carbon
source carries the weighed reagent the Methods names (sodium acetate, not acetate) with
the stated g/L, because that is the amount the paper weighed.

Not loaded and not a record for a different reason: the MG1655 and NCM3722 arms of
Table S9 (glucose and LB each). MG1655 is a deposited reference strain of its own with
its own b-number namespace, so those two records belong to a sibling dataset class with
`REFERENCE_STRAIN = "MG1655"`; NCM3722 has no deposited assembly at all. Both are named
in the PR as follow-up and nothing from Table S9 is loaded here. Table S8's `Glucose.2`
group (Table S25 A14-07088/89/90) is the reproducibility arm of Supplementary Figure 7
and is absent from Table S6, so it is not a record either.

### The identifier route, with counts

The released `Bnumber` column is an **MG1655** identifier and the records are BW25113.
Measured against the deposited GenBank annotations: 2,285 of 2,285 released b-numbers are
MG1655 locus tags (1.0000) and **0 of 2,285** are BW25113 locus tags. Keying on them
would write MG1655 identities onto BW25113 records. The hazard is visible directly in the
release: the `ygbT` row carries `b2755` while being an *Erwinia amylovora* protein.

The route taken, recorded as DERIVED, is the released `Gene` column (Table S6 column 3,
the identifier block's own gene name) resolved against the pinned BW25113 GenBank
annotation through `reconcile_locus_tags`. Measured on the 2,340 unique symbols of the
rows that survive the organism and duplicate rules:

| statistic | value |
|---|---|
| unique source names | 2,340 |
| resolved to one locus | 2,329 (0.9953) |
| resolved through the gene-symbol layer | 2,253 |
| resolved through a gene synonym | 76 |
| retired, kept as given | 11 |
| ambiguous | 0 |
| kept on collision | 0 |
| outside the namespace | 11 |
| remapped (stored as a locus other than their own) | 2,329 |

`MIN_RESOLVED_FRACTION` is 0.99, just below the measured 0.9953, so a change in the
release's keying stops the build rather than silently shrinking the abundance map.

The ECK crosswalk (`eck_crosswalk`) is NOT used. It would need the MG1655 genome beside
the BW25113 one, it reaches at most 2,285 of the 2,340 names (69 rows carry no b-number),
and the symbol route already covers more. Cross-checking the two routes against each
other is named in the PR as follow-up.

The whole-table measurements, before the duplicate rule, for the record: column 3 `Gene`
resolves 2,336 of 2,347 unique host symbols (0.9953) with 0 ambiguous and 0 collisions;
the COG block's second `Gene` column (column 74) resolves 2,339 of 2,346 (0.9970) but
carries 1 blank and 2 ambiguous names (`spr`, `ygaD`). Column 3 is the identifier block's
column and is the one consumed.

### Retention ledger, with the arithmetic

`preprocess/dropped_records.json`, whose `check()` refuses a ledger that does not close.

**Conditions: 22 released = 14 records + 1 reference + 7 dropped.**

| rule | n | items | needed addition |
|---|---|---|---|
| `medium_has_no_media_library_entry` | 1 | Glycerol + AA | `M9_GLYCEROL_AA_SCHMIDT2016` |
| `culture_not_batch` | 4 | Chemostat µ=0.5, 0.35, 0.20, 0.12 | a culture-mode / dilution-rate slot on `Environment` |
| `growth_phase_not_representable` | 2 | Stationary phase 1 day, 3 days | a growth-phase slot on `Environment` |

Both structural rules are measured collisions, not preferences, and the test
`test_dropped_chemostat_and_stationary_arms_would_not_be_distinguishable` pins them: the
four chemostat arms build **one** environment between them (they differ only in dilution
rate, which `Environment` cannot state), and the two stationary arms plus the exponential
Glucose arm build **one** environment between the three (same medium, same 5 g/L glucose,
same 37 C, same oxygen regime). `culture_not_batch` is the rule the landed Lamoureux 2023
loader already states under that name and for that reason. That what survives IS
distinguishable is asserted at build time by `check_environments_distinct` over the 15
loaded environments, and again by the verifier's `environment_uniqueness` row over the 14
records.

**Protein rows: 2,359 released - 5 - 14 - 11 = 2,329 keys.**

| rule | n rows | items |
|---|---|---|
| `source_organism_is_not_escherichia_coli` | 5 | `addB` (P23477, *Bacillus subtilis*), `ygbT` (D4HZR9, *Erwinia amylovora*), `cgtA` (Q6E0U3, *Vibrio harveyi*), `cas1` (D8FH86, *Peptoniphilus sp.*), `MettuDRAFT_4149` (E1MTY0, *Methylobacter tundripaludum*) |
| `gene_symbol_filed_on_two_released_rows` | 14 | `nrdA`, `rmlA`, `mrcB`, `rpmE`, `clpB`, `bioD`, `glsA`, two rows each |
| `gene_symbol_resolves_to_no_locus_of_the_pinned_assembly` | 11 | `JW58`, `NA`, `adeP`, `coaB`, `comR`, `ghxP`, `insC`, `insH`, `trmB`, `trpG`, `zapE` |

**Rule order is load-bearing.** The non-host rows go first. Four of those five share an
*E. coli* gene synonym with a host row (`cas1`/`ygbT` both resolve to the *E. coli* `cas1`
locus, `cgtA`/`obgE` both to `obgE`), so running reconciliation first makes those symbols
collide, `reconcile_locus_tags` keeps all four as given by its retain-all policy, and the
genuine *E. coli* `obgE` measurement is lost along with the *Vibrio* contaminant. With the
organism rule first there are **0** collisions. The organism is read from the released
`Description`'s own `OS=` token, and the search database was built to carry contaminants,
which is the quoted justification.

**The dataset-2 preference is sourced and deliberately NOT applied.** The release states
"protein quantities obtained from data set 2 were employed for all quantitative analysis",
which would pick a row for four of the seven twice-filed symbols (`nrdA` and `mrcB` are a
canonical accession against its `-2` isoform, `rpmE` is P0A7M9 against P0A7N1, `glsA` is
P77454 against P0A6W0). It is not applied because `rmlA` (P37744 against P61887) and
`glsA` are two distinct paralogs under one symbol in BOTH rows, so the symbol cannot name
either; only a uniform drop never mis-attributes a measurement. The alternative is
recorded here so a reviewer can ask for it, and every dropped row is listed with its
accession and dataset in the ledger.

**Three records carry 2,034 keys rather than 2,329.** Dataset 1 did not cover the
Glycerol + AA, Xylose, Mannose and Fructose conditions, so its 295 kept proteins are
released as `NA` there. The four `NA` columns are exactly the dataset-1 rows, measured
(295 of 295 in each). Glycerol + AA is dropped for its medium, so the effect is visible on
Xylose, Mannose and Fructose. Each record's reference is restricted to that record's own
keys, which is what the shared verifier's `reference_finite` row requires.

A released copies/cell of 0 is a value and is kept verbatim; 106 of the 50,694 released
cells are 0 and none is negative. Nothing is imputed.

### Data provenance

One file is consumed: `si2.xlsx`, the publisher's Supplementary tables, retrieved from
the PMC OA Cloud bucket. The retrieval is scriptable and was re-run to make the deposit,
so `nature.com`'s auth redirect never came up and no manual recipe was needed.

| field | value |
|---|---|
| mirror path | `$DATA_ROOT/torchcell-raw/schmidtQuantitativeConditiondependentEscherichia2016/data/si2.xlsx` |
| `source_url` | `https://pmc-oa-opendata.s3.amazonaws.com/PMC4888949.1/NIHMS65833-supplement-Supplementary_tables.xlsx` |
| `retrieval_method` | `pmc_cloud` |
| `retrieval_command` | `torchcell.literature.retrieve.pmc_cloud_object(key="PMC4888949.1/NIHMS65833-supplement-Supplementary_tables.xlsx")` |
| `sha256` | `3280a13ff67a73f25440cff6ee73fb99b5ce3ef57854213dbbf6272be241912f` |
| bytes | 17,128,596 |
| `retrieved_at` | 2026-10-07T11:42:57.070583+00:00 (re-run and byte-identical on deposit) |

Recorded in the mirror manifest and deliberately not deposited: ProteomeXchange
**PXD000498**, the raw mass spectra and assigned MS/MS spectra that Supplementary Table 17
names, which no loader reads; and `si/si1.pdf` (Supplementary figures and notes), from
which no value is sourced. PRIDE is where the spectra live, and Mendeley Data is not a
source anywhere in this row.

### Build

```
PYTHONPATH=$PWD python -m torchcell.database.build_dataset_lmdb --dataset ProteomeSchmidt2016Dataset
```

| metric | value |
|---|---|
| records | 14 |
| gene-set size | 2,329 |
| references | 2 (the 2,329-key profile and the 2,034-key one) |
| build time | 5 s in-process, 11.78 s wall |
| store | `$DATA_ROOT/data/torchcell/proteome_schmidt2016` |
| `provenance.build_manifest` | `fresh` |

`compute_gene_set` is overridden: every record is wild type, so the base class's genotype
scan would return the empty set it refuses, and the dataset's genes are the loci it
measures. That is how the landed Caglar 2017 loader states the same situation for its
REL606 panel.

### Verification, L0 to L4

`verify_build(root)` writes `preprocess/verification_report.json`. The shared
`verify_protein_dataset` supplies L0 to L3; three SUPPLEMENTARY rows and a host-aware L4
are added, because the yeast deletion-collection overlap `run_protein` appends would say
nothing about a BW25113 locus tag. Result: **PASS**.

```
proteome_schmidt2016: PASS
  [ok] L0 structural: 14 records validated
  [ok] L1 count: observed 14, expected 14
  [ok] L1 orf_uniqueness: 0 unique knocked-out ORFs, one record each
  [ok] L1 environment_uniqueness: SUPPLEMENTARY: 14 distinct environments over 14 records
  [ok] L2 value_fidelity: 31721 values checked
  [ok] L2 se_nonnegative: 28476 values checked
  [ok] L2 se_matches_replicate_count: SUPPLEMENTARY: 0 of 31721 values disagree with their replicate count
  [ok] L3 reference_finite: reference abundance finite + key-matched for all 31721 values
  [ok] L3 measurement_type_consistent: single measurement_type: 'absolute_protein_copies_per_cell_label_free_sid_anchored'
  [ok] L3 assembly_pin: SUPPLEMENTARY: assembly pins ["('ecoli_K12_BW25113_ASM75055v1', 'GCA_000750555.1')"]
  [ok] L4 gene_containment_bw25113: 2329 measured protein keys; 0 outside the ecoli_K12_BW25113_ASM75055v1 locus universe
```

### What could not be sourced

- The BW25113 background lesions are quoted verbatim from the Methods but have no field
  to live in. `AssemblyReferenceGenome.background` holds a `BacterialStrainBackground`,
  which describes an edit ON TOP OF the pinned assembly, and the deposited BW25113
  GenBank annotation already carries those lesions. So the quote lives in
  `preprocess/sourced_values.json` and here, not on the record. This matches the landed
  Fuhrer 2017 loader, which pins the same strain with no background.
- The thiamine amount of `M9_SCHMIDT2016` is an open gap in the media layer, not in this
  loader: the pinned `paper.md` lost the stock concentration to OCR (it reads ".4.m M").
  That gap is already recorded on the media object.
- The number of expected tryptic peptides per protein, which the abundance model divides
  by, is not a released column. It is not needed: the SE derivation cancels it, measured
  to 9.3e-10.

## 2026.10.07 - BioCypher adapter and its enable-list

`ProteomeSchmidt2016Adapter` (`torchcell/adapters/schmidt2016_adapter.py`) serves this
dataset to BioCypher, with its enable-list in
`torchcell/adapters/conf/proteome_schmidt2016_adapter.yaml`. The adapter is registered
in `dataset_adapter_map`, re-exported from `torchcell/adapters/__init__.py`, and named
in `torchcell/knowledge_graphs/conf/kg_bacteria.yaml`, so the bacteria-only rehearsal
generation covers it. The module names exactly one conf file, because
`kg_manifest._CONF_RE` reads the FIRST quoted `*_adapter.yaml` out of the module source
and would otherwise fingerprint a sibling's conf (issue #743).

### Why no gene-perturbation method is enabled

Every one of the 14 records is wild-type BW25113 with an empty genotype, measured on the
dev store at `$DATA_ROOT/data/torchcell/proteome_schmidt2016`: the perturbation count is
0 in 14 of 14. The paper's three deletion strains carry no abundance data and the loader
does not load them, so the only varying axis is the environment and the glucose arm is
the phenotype reference. Neither `bacterial perturbation (chunked)` nor `perturbation to
genotype (chunked)` is enabled, and `crispr construct (chunked)` is off for the same
reason: there is no perturbation leaf for a construct to hang off. The `genotype
(chunked)` node itself IS served, one empty genotype per record, because the genotype is
what the experiment points at. The landed `ProteomeCaglar2017Adapter` is the precedent;
its REL606 panel is wild type in 105 of 105 records and carries the same shape.

Enabling a perturbation method here would not fail loudly. The method would simply walk
an empty list and write nothing, which is why the admission gate checks the converse:
`assert_dev_store_graph` runs every family the conf leaves OFF over the real records and
requires it to emit nothing, so the enable-list is pinned in both directions.

### The rest of the enable-list

- **Environment, media, temperature, and their references** are enabled: the 14 records
  span LB and 13 M9 variants, with temperature varying (37 C, and 42 C for the heat
  arm), so every one of those sub-objects differs between records.
- **The environment-perturbation pair** is enabled: 13 of the 14 records carry at least
  one `EnvironmentPhysicalPerturbation` or `SmallMoleculePerturbation` (the LB record
  carries none, two records carry two), and the glucose reference environment carries
  one, so the reference method has a node to write.
- **`protein abundance phenotype`** is the phenotype class of
  `BacterialProteinAbundanceExperiment`, which the harness derives from the loader's
  `experiment_class` rather than from the conf.
- **Genome, dataset, publication** are the standard trio; the genome is the pinned
  `ecoli_K12_BW25113_ASM75055v1` assembly reference.

Measured over all 14 records, the adapter emits these node labels: `dataset` 1,
`environment` 15, `environment perturbation` 16, `experiment` 14, `experiment
reference` 2, `genome` 1, `genotype` 14, `interned constant` 14, `media` 15, `protein
abundance phenotype` 16, `publication` 14, `temperature` 15. No `perturbation` and no
`bacterial perturbation` node appears, which
`tests/torchcell/adapters/test_schmidt2016_adapter.py` pins on the conf (hermetic) and
on the emitted graph (data-gated).

### Admission

`kg_manifest admit --dataset ProteomeSchmidt2016Dataset` reads the dev LMDB as `fresh`
and `served: no (new dataset)`, then BLOCKS on four store-wide conditions that predate
this adapter: 11 served datasets whose schema closure changed, the `crispr construct`
graph class, `_crispr_construct_node_from` adapter drift, and the media and
compound-identity value surface. The identical four blocks come back for the already
landed `ProteomeCaglar2017Dataset`, so they belong to the pending full rebuild, not to
this dataset.

## 2026.10.08 - Two sibling loaders for the same paper, and why they are separate modules

This paper now has three dataset families, each with its own loader module, its own
adapter, its own conf and its own dev store:

| family | module | classes | records |
|---|---|---|---|
| Table S6 label-free proteome | `schmidt2016.py` | `ProteomeSchmidt2016Dataset` | 14 |
| Tables S2 and S3 SRM proteome | `schmidt2016_srm.py` | `ProteomeSrmSet1Schmidt2016Dataset`, `ProteomeSrmSet2Schmidt2016Dataset` | 11 + 14 |
| Table S24 rim-deletion growth | `schmidt2016_growth_rate.py` | `GrowthRateSchmidt2016Dataset` | 6 |

Full sourcing, measurements and verification for the two new families:
[[torchcell.datasets.ecoli.schmidt2016_srm]] and
[[torchcell.datasets.ecoli.schmidt2016_growth_rate]].

**This file was not changed except for a pointer in its docstring, and that is the point.**
`build_manifest` decides a built store's staleness from the schema closure of the loader
MODULE's own `torchcell.datamodels` imports (`provenance/schema_deps.py:loader_closure`
parses `from torchcell.datamodels...` out of the module source). Co-locating the new
classes here would have added `FitnessPhenotype`, `BacterialDeletionPerturbation` and
`BacterialFitnessExperiment` to this module's closure, marking the already-served
`proteome_schmidt2016` store stale and forcing a full knowledge-graph rebuild for a change
that touches none of its 14 records. Measured: `python -m torchcell.provenance.build_manifest`
reports `proteome_schmidt2016` as `fresh` both before and after that branch. The siblings
import the pinned artifact, the condition table, `build_environment`,
`check_environments_distinct`, `publication` and the three supplementary verification rules
FROM here, so each is still stated once. A docstring edit changes no import and so changes
no closure.

**The scope correction this makes to the note above.** The section "Per-condition
genotype-versus-environment decisions" says the three Keio deletion strains "appear ONLY in
the N-alpha-acetylation analysis" and carry no per-protein abundance. That remains exactly
right about ABUNDANCE, and it is why no record of this dataset has a genotype perturbation.
It is not a statement about their growth rates, which Table S24 releases and
`GrowthRateSchmidt2016Dataset` now serves as the paper's one gene-perturbation phenotype.

**A third header fault in this release, found while reading Table S23.** The swap this note
already records in Table S6's coefficient-of-variance block is not the only one.
Table S23's `Doubling time (h-1)` column is a doubling time in HOURS, not in h^-1: measured
on all 23 rows with a positive growth rate, the released value equals `ln(2) / rate` to a
worst absolute deviation of 0.0496 h, inside the 0.05 h rounding of a one-decimal column,
while `rate / ln(2)` is off by factors of 5 to 10. And Table S23 spells the strain
`MG1665` on its 2 non-BW25113 rows where Table S25 spells it `MG1655` on 6 samples and the
Methods say "MG1655". Neither is consumed by any loader; both are recorded because the
MG1655 arms of Table S9 are already a named follow-up, and a loader keying on the released
header unit or the released strain string would be wrong.
