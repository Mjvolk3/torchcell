---
id: r681byli0b4wzig48bu0r5x
title: Rachwalski2024
desc: ''
updated: 1791454053140
created: 1791454053140
---

## 2026.10.08 - Rachwalski 2024 mobile CRISPRi loader: sourcing, record type, retention and build

Row 42 of the fifty bacterial datasets, tier 1. Rachwalski K, Tu MM, Madden SJ, French S,
Hansen DM, Brown ED, *A mobile CRISPRi collection enables genetic interaction studies for
the essential genes of Escherichia coli*, Cell Reports Methods 2024;4:100693,
doi:10.1016/j.crmeth.2023.100693, PMID 38215765, PMCID PMC10832289, citation key
`rachwalskiMobileCRISPRiCollection2024`.

Loader: `torchcell/datasets/ecoli/rachwalski2024.py`, class
`CrispriCrossRachwalski2024Dataset`.
Adapter: `torchcell/adapters/rachwalski2024_adapter.py`, conf
`torchcell/adapters/conf/crispri_cross_rachwalski2024_adapter.yaml`.
Tests: `tests/torchcell/datasets/ecoli/test_rachwalski2024.py` and
`tests/torchcell/adapters/test_rachwalski2024_adapter.py`.

### The record, and why it is a fitness record

A record is one (strain, medium, inducer dose) normalized colony growth, stored as a
`BacterialFitnessExperiment`. The genotype carries a
`BacterialCrisprInterferencePerturbation` (dCas9, the 20 nt Table S1 spacer, one guide
per gene), a `BacterialDeletionPerturbation` of the Keio collection, or BOTH. The
environment is solid LB or solid MOPS-glucose minimal at 37 C with the column's
anhydrotetracycline as a `SmallMoleculePerturbation`; a 0 ng/mL column carries NO
perturbation, because no aTc was added to those plates. The phenotype is a
`FitnessPhenotype` with `n_samples = 2` and `screen_id` naming the released table.

`FitnessPhenotype` rather than `EnvironmentResponsePhenotype` because every released
value is a strictly positive relative colony growth whose neutral point is 1, not 0.
Measured on the pinned bytes: minimum 0.00613, maximum 4.752, no blanks, no NaN, and
nothing at or below zero in any of the 63,552 released cells, so the `fitness` clamp
never fires. For Tables S2A and S3 the denominator is a real control strain in the same
condition, and the 17 empty-vector wells mean exactly 1.0000 in each of the 24 columns,
so `fitness == 1` is the release's own normalization rather than a convention imposed
here.

### Why the knockdown is the genotype and the dose is the environment

The strain is the strain at every dose: the guide and the dCas9 cassette are in the cell
on the 0 ng/mL plate too. What the dose changes is how far the designed knockdown is
realized, so aTc rides on the environment, as it does for Rapp 2026's 200 nM.
`ExpressionRangeMultiplier` is left unset, because no record states a realized repression
magnitude. ng/mL is not a schema unit, so a dose is stored as its exact ug/mL thousandth
(500 ng/mL as 0.5 ug/mL).

### The genotype crux: no new leaf was needed

The two-perturbation genotype this row exists for is expressible today.
`BacterialCrisprInterferencePerturbation` and `BacterialDeletionPerturbation` both exist
in `torchcell/datamodels/schema.py` and are both already served as the
`bacterial perturbation` graph node class, so a `Genotype` holding one of each needs no
new leaf, no new graph class and no Biolink parent. 32,499 of the 48,900 records carry
such a genotype. `BacterialGeneInteractionExperiment` and `GeneInteractionPhenotype`
stay unused by this loader, deliberately: see "What is not loaded" below.

### Sourcing

Every statistic below is a verbatim quote against a sha256-pinned mirror artifact. The
loader carries 27 of them (24 in `paper.md`, 2 in the SI figure-legend OCR `si/si1.md`,
1 a Table S2C column header), the whole set is written to
`preprocess/build_accounting.json` at build time, and all 27 audit clean against their
pinned bytes as a `--data` test. The table below is the subset that fixes a stored value.

| value | source | quote |
|---|---|---|
| host strain BW25113 | paper.md, Methods, Plasmids | "All fitness screens with the CRISPRi collection were performed in E. coli BW25113 ... all genes deletions used in the study were obtained from the Keio collection4 (kanamycin resistant single-gene deletions in E. coli BW25113)." |
| 37 C, 16 h | paper.md, Methods, screening | "then were grown for 16 h at 37 C and visualized using transmissive scanners." |
| dCas9 on pFD152 | paper.md, Methods, Plasmids | "sgRNAs targeting different essential genes in E. coli were cloned into the conjugative CRISPRi plasmid pFD152 ... pFD152 was a gift from David Bikard (Addgene plasmid # 125546)." |
| one guide per gene | paper.md, Methods, Plasmids | "an sgRNA with the highest predicted on-target activity score and the least off-target homology was selected for each gene." |
| 15 g/L agar | paper.md, Growth conditions | "For experiments on solid medium, media was prepared with 15 g/L agar." |
| 1.5% w/v agar, MOPS | paper.md, Growth conditions | "MOPS minimal medium was prepared according to the manufacturer's instructions: components were filter sterilized after preparing liquid growth medium or added to sterile water and agar (1.5% w/v) for solid growth medium." |
| the carbon source is glucose | mmc3.xlsx sheet ST2C header | "Normalized Growth MOPS-glucose with 500 ng/ml aTc (This study)" |
| both media | paper.md, Results | "We screened the CRISPRi collection for growth inhibition in both rich (LB) and minimal (MOPS minimal) microbiological media at varying aTc concentrations" |
| Table S3 is minimal | paper.md, Results | "The wildtype (lpp+) and dlpp CRISPRi collections were then screened at varying aTc concentrations on minimal media" |
| SIX doses | paper.md, Figure 3 legend | "the collection was screened at 6 different concentrations of aTc in minimal media." |
| the five-dose Methods list | paper.md, Methods, screening | "Cells were then pinned onto solid media containing five concentrations of aTc (0, 5, 10, 50, 500 ng/mL)" |
| THREE Keio doses | paper.md, Figure 4 legend | "The Keio collection harboring pFD152_lolA was screened at 0, 50, and 500 ng/mL of aTc." |
| n = 2, called technical | paper.md, Methods, normalization | "The average of the two technical replicates was then calculated and is reported in Table S2." |
| n = 2, called technical | paper.md, Methods, screening | "CRISPRi Keio screens were conducted in technical duplicate." |
| n = 2, called biological | paper.md, Methods, normalization | "the 12 Keio collection assay plates screened in biological duplicate" |
| n = 2, called biological | si/si1.md, Figure S1A | "Growth of each CRISPRi mutant was normalized to the mean growth of the 17 empty vector controls on the plate, then two biological replicates (R1 and R2), or the average of the two biological replicates are plotted." |
| a dispersion exists, unreleased | si/si1.md, Figure S1B | "Bars represent the average of biological replicates and the standard error of the data is shown with error bars." |
| the S2/S3 denominator | paper.md, Methods, normalization | "the average raw colony density of the 17 strains harboring the empty vector was calculated, then, this value was used to normalize the growth of strains harboring different CRISPRi plasmids." |
| the S4A normalization | paper.md, Methods, normalization | "First, spatial effects across the 24 assay plates were normalized by dividing each colony with the inter-quartile mean (IQM) ... Then, plate to plate variability was normalized by dividing the growth of each colony, by the IQM of the growth of every other colony harboring the same CRISPRi plasmid within each screening plate." |
| the 272 oracle | paper.md, Results | "With 100 ng/mL aTc added to the growth media, 272 from the 356 strains showed at least a 50% reduction in growth (Figure 1C)." |
| the Keio cassette | paper.md, Methods, Plasmids | "(kanamycin resistant single-gene deletions in E. coli BW25113)." |
| the Keio assay medium | paper.md, Methods, screening | "Then, cultures are spotted onto a 1536-colony density solid agar plates containing different concentrations of aTc" |

### The replicate count is sourced and the replicate type is not

`n_samples = 2` is stated four times. The TYPE is contradicted within the same Methods
section: "technical replicates" and "technical duplicate" against "biological duplicate",
and three figure legends (S1A, S1B, 4C) say biological. `sample_unit` is therefore a typed
`ProvenanceGap`, not a guess. The dispersion is a second typed gap: it exists upstream
(Figure S1B plots a standard error for a handful of selected strains) and no table
releases one per record, so computing a sample SD from two values this loader never sees
would be our statistic, not theirs. Neither absence is ever a zero.

### Findings in the release

1. **The Methods list five inducer doses and the tables carry six.** The prose omits
   100 ng/mL, which Tables S2A and S3 both carry, Figure 3A calls six, and the Results use
   by name ("Looking at 100 ng/mL aTc specifically"). Trusting the prose would drop a sixth
   of the dose response. The build reproduces the paper's own count off that very column:
   exactly 272 of Table S2A's 360 non-empty-vector rows have LB 100 ng/mL growth at or
   below 0.5, and those 360 rows carry 356 distinct target symbols, which is the paper's
   "272 from the 356 strains".
2. **The Results say the Keio screen used two doses; Table S4A and Figure 4B carry three.**
   0, 50 and 500 ng/mL. The release and the figure agree, so the body sentence is the odd
   one out.
3. **`gpsA` is spelled `gspA` in both growth tables.** Table S2A and Table S3 are Table S1's
   plate order position for position, 377 rows for 377 filled wells, with exactly ONE label
   disagreement: well G15 is `gpsA` in Table S1 and `gspA` in both growth tables. Table S1's
   own annotation for that well ("Lipid biosythesis - Glycerol-3-phosphate dehydrogenase",
   operon `yibN,secB,gpsA,grxC`) and the Results ("essential phospholipids (gpsA, psd,
   plsBC, and pssA)") both say gpsA, and `gspA` is a different, non-essential gene. The
   loader stores Table S1's symbol and asserts at build time that this is the only
   disagreement.
4. **The Results cite the LB/MOPS growth table as "Table S1A".** The workbook's own Legend
   sheet calls it Supplemental Table 2A, and Table S1 is the collection layout with no
   growth value in it. The Results citation is off by one.
5. **Table S4A repeats 458 deletion labels with no key.** 4,542 rows over 4,017 distinct
   labels; after resolution, 458 labels head 2 to 5 rows each. No plate, well or clone key
   is released, so two rows naming one deletion cannot be told apart. Baba 2006's plate
   layout would separate them and the mirror does not hold it.
6. **The 360 guide wells carry only 355 distinct spacers.** `rbfA`, `rplK`, `rplX` and
   `ydfO` each head two wells whose 20 nt guide sequences are identical, so these are one
   construct pinned twice and not two guides. The fifth repeated spacer is shared by
   `ispU` and `uppS`, two names for one BW25113 locus, which the merged-locus rule removes
   anyway. 10 wells in all, so 20 rows across Tables S2A and S3.

### Which Table S2A block is which medium, measured rather than read

The group header says only "Average Growth of Replicates" twice. Sheet ST2C names its two
columns ("Normalized Growth LB with 500 ng/ml aTc", "... MOPS-glucose with 500 ng/ml
aTc"), and 50 of its 52 joinable rows match block 1 as LB and block 2 as MOPS to 1e-9. The
2 that do not (`ygeN`, `yqcG`) share one LB value in ST2C while their MOPS values still
match, which is a copy-paste inside the subset. The build runs this check and stops if the
match rate collapses.

### The three screens are three screens

Table S2A and Table S3 both hold the WT (lpp+) collection on MOPS minimal at the same six
doses. Measured over the 377 x 6 overlapping cells, ZERO values agree to 1e-9 and the
per-dose Pearson r runs from 0.514 at 0 ng/mL to 0.945 at 500 ng/mL, so they are two
independent runs. `screen_id` is `Table S2A`, `Table S3` or `Table S4A`, which is what
keeps the two MOPS runs on distinct keys.

That required one change outside this loader: `torchcell/verification/fitness.py` now
folds the phenotype's `screen_id` into `_environment_signature` and the CRISPR construct's
`guide_sequence` and `library_pool` into `_genotype_signature`, which is what the
environment-response verifier has always done and what `FitnessPhenotype.screen_id`'s own
docstring says the field is for. Adding a field to a signature can only SPLIT a group,
never merge two, so no dataset that passed L1 before can start failing.

### Records dropped

| rule | what it removes |
|---|---|
| `row_is_an_empty_vector_control` | the 17 empty-vector wells of Tables S2A and S3; they ARE the denominator and carry no gene perturbation, so they are the `phenotype_reference` |
| `label_is_not_in_the_bw25113_annotation` | `lapA` and `lapB` among the targets; 124 Keio labels (JW strain ids, small RNAs, non-gene labels) |
| `label_is_a_fragment_of_a_merged_bw25113_locus` | `ispU` and `uppS`, two target symbols of one locus; 46 Keio labels |
| `label_is_ambiguous_in_bw25113` | 10 Keio labels matching more than one locus |
| `construct_sits_in_more_than_one_collection_well` | 6 labels over 5 repeated spacers at 10 wells, so 10 rows in each of Tables S2A and S3 |
| `label_heads_more_than_one_row` | 951 rows of Table S4A over 458 labels |

Dropping the whole duplicate group rather than keeping one is the rule Campos 2018 and
Shiver 2016 already apply: mapping both onto one locus would merge strains, and keeping
one would be arbitrary.

Resolution against BW25113 GCA_000750555.1 by gene symbol: 353 of 357 target labels (lpp
included) and 3,837 of 4,017 Keio labels resolve.

### What is not loaded, and why

- **The Zenodo archives** (10.5281/zenodo.10214517, 1.4 GB and 364 MB of plate images plus
  the ImageJ and R analysis code). No record here is built from an image and the released
  tables are the normalized values that code produces, so they are retrieval metadata, not
  a build input, and they are not deposited.
- **Table S2B and Table S2C.** Subsets of Table S2A whose values this build reproduces.
- **Table S3's `Fold Change Growth (dlpp/WT)` block and Table S4A's `Empty Vector
  Normalized Growth` block.** Ratios of columns this dataset already stores, and not the
  ratio of the stored averages: Table S4A's `lolA_0aTC` differs from col5/col1 in the third
  decimal on 498 of the first 500 rows, because the paper normalized per replicate and then
  averaged. Storing them as `GeneInteractionPhenotype` would also mistype them;
  `gene_interaction` is an epsilon or tau, a deviation from an expected product, and a fold
  change is not one. A dedicated interaction arm would need its own statistic and is not
  invented here.
- **Table S4B.** The 68 suppressors and 9 enhancers at a 3-SD cutoff: a hit call over Table
  S4A, not a measurement.
- **mmc6.pdf.** A reprint of the article, redundant with `paper.pdf` in the mirror.
- **pFD152 itself (Addgene 125546).** The effector and spacer are on `CrisprConstruct`; the
  full plasmid is what a future `effector_plasmid_ref` would pin.

### The one environment value the paper does not pin

The base medium of the CRISPRi-Keio assay plates is never named. That paragraph says only
"cultures are spotted onto a 1536-colony density solid agar plates containing different
concentrations of aTc" and opens "the workflow is followed as above with minor
modifications". Every named step of that workflow is on LB agar and the paper names no
other base for it, so LB is what the loader records, as a READING and not a statement. It
is carried in the medium's own provenance note, in `preprocess/build_accounting.json`
under `unpinned_environment_values`, and here. Tables S2A and S3 need no such reading.

### Provenance

Raw mirror: `$DATA_ROOT/torchcell-raw/rachwalskiMobileCRISPRiCollection2024/data/` holds
`mmc2.xlsx` (Table S1), `mmc3.xlsx` (Table S2), `mmc4.xlsx` (Table S3) and `mmc5.xlsx`
(Table S4), each with its `pmc_cloud` `RetrievalRecord`, source URL, sha256 and
`retrieved_at` in `manifest.json`. The retrieval was RE-RUN from the PMC Article Datasets
bucket on 2026-10-08 and returned bit-identical bytes for all four files.

### Build and verification

48,900 records, 32,499 of them two-perturbation, built into
`$DATA_ROOT/data/torchcell/crispri_cross_rachwalski2024`. The build stops on any of: a
header that moved, a row or distinct-label count that moved, a blank growth cell, a second
label disagreement in the plate-order join, a 272-count that moved, an empty-vector mean
off 1, a Table S2C block-order match that collapsed, a resolved fraction below the
threshold, a stored tag that is not a locus, or a record count that does not match the
tables.
