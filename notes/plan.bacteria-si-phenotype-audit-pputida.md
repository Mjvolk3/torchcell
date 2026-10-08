---
id: 56vliesohj17g6z2oqxlb7g
title: Bacteria Si Phenotype Audit Pputida
desc: ''
updated: 1791422527131
created: 1791422527131
---

## 2026.10.07 - SI phenotype audit, P. putida rows plus Caglar 2017

An audit, not a loader task. No loader, schema, adapter, conf or dev store is touched by
this note. The question asked of each paper: does its released supplementary data carry
per-record phenotypes the loader does not store, and does the paper compare itself
against other released data in a way that points at a further loadable dataset or at a
duplication risk.

Scope: the eight `torchcell/datasets/pputida/*.py` rows plus `ecoli/caglar2017.py`, which
is here because it is E. coli B REL606 and sits beside the P. putida group in the same
bacterial expansion.

Method per paper: read the mirrored `paper.md`, list and open every file under the
mirror's `si/`, enumerate the per-record quantities from table captions and data-file
column names, read the loader and its dendron note for what is already recorded as
deliberately not stored, then classify each unstored quantity as loadable now, blocked on
a named schema gap, or not a phenotype.

### Summary of counts

Every released per-record quantity enumerated below is classified exactly once. "Remainder"
is the quantities that are a derived summary, a design parameter, an annotation, or not
released as a table at all. Record counts are in the ranked list at the end.

| paper | SI files opened | quantities enumerated | stored today | loadable now | blocked | remainder |
|---|---|---|---|---|---|---|
| Carruthers 2025 | 9 (31 Source Data sheets) | 24 | 3 | **6** | 5 | 10 |
| de Siqueira 2025 | 3 | 18 | 2 | **4** | 4 | 8 |
| Kang 2026 | 1 (9 SI tables) | 19 | 3 | **2** | 3 | 11 |
| Borchert 2024 | 4 | 13 | 1 partly | 0 | 1 | 11 |
| Lim 2022 | 2 | 17 | 2 partly | 0 | 1 | 14 |
| Lim 2025 | 2 | 21 | 2 | 0 (2 conditional) | 6 | 13 |
| Menasalvas 2025 | 1 on disk, 5 off disk | 20 | 2 | **2** (retrieval-gated) | 2 | 14 |
| Yunus 2026 | 1 (14 SI tables) | 16 | 2 | **1** | 1 | 12 |
| Caglar 2017 | 15 | 16 | 2 | **1** | 3 | 10 |
| **total** | | **164** | **19** | **16** | **26** | **103** |

The answer to the owner's question is yes for six of the nine papers, and the loadable-now
items come to **157 extra records** with no schema change. Three papers (Borchert 2024,
Lim 2022, Lim 2025) release nothing loadable now: what they hold back is either blocked on
#731, excluded by the source's own QC, or released only as a figure.

Three findings sit outside the per-paper tables and are the reason to read the end first:
a **measured duplication risk** between Borchert 2023 and the Borchert 2024 compendium
(identical fitness, max absolute difference 5e-4 over 4,732 loci), a **measured duplication
risk** between Caglar 2017 and Houser 2015 (27 of 152 mRNA and 27 of 105 protein records),
and a **measured NON-duplication** between Carruthers 2025 and Yunus 2026, which was the
isoprenol group's highest-risk pair.

### Citation keys, resolved

The task named six short keys; the mirror and the loaders use the full keys below.

| loader | citation key in the mirror |
|---|---|
| `pputida/carruthers2025.py` | `carruthersAutomationMachineLearning2025` |
| `pputida/borchert2024.py` | `borchertMachineLearningAnalysis2024` |
| `pputida/desiqueira2025.py` | `desiqueiraAlternateRoutesAcetate2025` |
| `pputida/kang2026.py` | `kangMultilayeredMetabolicRemodeling2026` |
| `pputida/lim2022.py` | `limMachinelearningPseudomonasPutida2022` |
| `pputida/lim2025.py` | `limEvolutionguidedToleranceEngineering2025` |
| `pputida/menasalvas2025.py` | `menasalvasBiosensordrivenStrainEngineering2025` |
| `pputida/yunus2026.py` | `yunusPredictiveCRISPRmediatedGene2026` |
| `ecoli/caglar2017.py` | `caglarColiMolecularPhenotype2017` |

### Carruthers 2025, the baseline target dataset

Mirror `$DATA_ROOT/torchcell-library/carruthersAutomationMachineLearning2025/`. Files
under `si/`, by the role its `manifest.json` records and the name
`si/si3.md` ("Description of Additional Supplementary Files") gives each one:

| file | role | what it is |
|---|---|---|
| `si/si1.pdf` + `si/si1.md` | `si_pdf` / `si_ocr` | Supplementary Information: 15 Supplementary Figures, Supplementary Tables 1 to 5 |
| `si/si2.pdf` + `si/si2.md` | `si_pdf` / `si_ocr` | peer review file ("This file contains all reviewer reports in order by version") |
| `si/si3.pdf` + `si/si3.md` | `si_pdf` / `si_ocr` | the Description of Additional Supplementary Files |
| `si/si4.xlsx` | `si_data` | Supplementary Data 1, the 120-row DBTL0 gene-target table |
| `si/si5.xlsx` | `si_data` | Supplementary Data 2, oligonucleotides (104 rows) |
| `si/si6.xlsx` | `si_data` | Supplementary Data 3, plasmids (229 rows) |
| `si/si7.xlsx` | `si_data` | Supplementary Data 4, CRISPRi arrays DBTL1 to DBTL6 (360 rows) |
| `si/si8.pdf` + `si/si8.md` | `si_pdf` / `si_ocr` | Nature Portfolio Reporting Summary |
| `si/si9.xlsx` | `si_data` | the Source Data file, 31 sheets |

`manifest.json` records `si_expected: []` for the literature mirror, so nothing the
paper released is missing from it. The Dryad campaign proteome expectation lives in the
RAW mirror, not here.

Already recorded, not re-reported: the 472-strain campaign proteome behind the Dryad
challenge (issue #739, and `notes/torchcell.datasets.pputida.carruthers2025.md`); the
per-strain `pass filter?` flag kept in `preprocess/pass_filter.csv` and off the record;
the 77 dropped protein keys; and that yield and productivity are four typed
`ProvenanceGap`s with reason `not_reported_by_primary` because the campaign released
neither.

**No growth measurement exists to be missed.** `paper.md` and `si/si1.md` contain zero
occurrences of `OD600`, `optical density`, `biomass`, `growth rate`, `scattered light`,
`backscatter`, `doubling time` or `glucose consum` (measured by grep over both files).
All six `growth` hits in `paper.md` are qualitative prose, for example "although it could
be deleted, PP_4191 strains showed severe growth inhibition and negligible isoprenol
production". Cultures ran in a "48-well BioLector flower plate without optodes", and no
biomass trace is released in any of the 31 Source Data sheets.

The loader reads exactly five of the 31 Source Data sheets (`SHEET_TITER = "Figure 4b"`,
`SHEET_TITER_ALT = "Figure 1D"`, `SHEET_TARGET_MEANS = "Figure 3a"`,
`SHEET_CONTROLS = "Figure 2c"`, `SHEET_PROTEOME = "Supplementary Figure 13abc"`) plus
`si/si6.xlsx` for the pIY670 part composition and `si/si4.xlsx`'s titer column as a
cross-source oracle. The table below covers all 31 sheets and the four Supplementary Data
workbooks.

| released quantity | exact column or verbatim caption quote | SI file / sheet | stored? | classification | records |
|---|---|---|---|---|---|
| per-replicate isoprenol titer, CRISPRi strains | `isoprenoli titer (mg/L)` (and `isoprenol (mg/L)` in `Figure 1D`) | `si/si9.xlsx` `Figure 4b`, `Figure 1D` | yes, 465 records over 1,416 non-control cultures | stored | 1,506 rows |
| per-replicate control titer | `Titer` | `si/si9.xlsx` `Figure 2c` | yes, as the per-cycle `phenotype_reference` | stored | 90 |
| per-target mean DBTL0 titer | `Isoprenol mean`; `Mean isoprenol titer (mg/L)` | `si/si9.xlsx` `Figure 3a`, `Figure 3d`; `si/si4.xlsx` | consumed as an oracle, not stored as its own record | not a phenotype, it is the mean of stored replicates | 121 / 92 / 120 |
| per-protein per-replicate Top3 abundance, PP_0815 off-target panel | `Top_3pep_counts_mean` | `si/si9.xlsx` `Supplementary Figure 13abc` | yes, 19 records plus 1 reference | stored | 90,180 cells |
| **per-replicate isoprenol titer of 12 deletion strains, each carrying the target sgRNA and a non-target sgRNA** | `Strain`, `Type`, `Titer` with `Type` in `{Target, Non-target}`; caption "Comparison of titers between KO strains harboring sgRNAs and their CRISPRi analogues" | `si/si9.xlsx` `Figure 6a` (the sheet `Supplementary Figure 11` is identical to it row for row, 106 rows including the header) | **no** | **loadable now** | 72 cultures, 24 strain-condition groups |
| **per-replicate isoprenol titer of KO-only and CRISPRi-plus-KO array strains** | `Line Name`, `Type`, `Replicate`, `Isoprenol` with `Type` in `{KO Only, CRISPRi and KO}` | `si/si9.xlsx` `Figure 6d` | **no** | **loadable now** | 12 cultures, 4 groups (`PP_0812-15_KO`, `PP_0368_PP_0812-15_KO`, and each with a CRISPRi array) |
| **per-replicate isoprenol titer of the PP_0815 off-target sgRNA panel** | `Sample`, `Replicate`, `titer`; caption "Isoprenol titers of off-target sgRNA strains failed to recapitulate the observed titer" | `si/si9.xlsx` `Supplementary Figure 13d` | **no** | **loadable now** | 58 cultures, 20 groups |
| **per-replicate isoprenol titer of two overexpression strains across seven inducer levels** | `Strain`, `Replicate`, `Inducer concentration`, `Isoprenol`; inducer values `0, 31.25, 62.5, 125, 250, 500, 1000` | `si/si9.xlsx` `Supplementary Figure 12bd` | **no** | **loadable now** | 48 cultures, 15 strain-inducer groups |
| **per-protein per-replicate Top3 abundance of the overexpressed pathway, by inducer level** | `Top_3pep_counts_mean`, `%_of protein_abundance_Top3-method`, `log10_%_abundance`, `Inducer concentration` | `si/si9.xlsx` `Supplementary Figure 12ac` | **no** | **loadable now**, same class the loader already uses | 144 cells, 6 proteins over 16 samples |
| **pathway-protein and dCas9 percent abundance per control culture** | `mvaE`, `mvaS`, `MK`, `PMD*`, `AphA`, `dCas9`; caption "Levels of heterologous pathway proteins in the controls across all DBTL cycles as determined by the Top3 Method" | `si/si9.xlsx` `Supplementary Figure 15` | **no** | **loadable now** | 540 cells, 90 control cultures over 30 line names |
| POI and dCas9 abundance relative to control, per DBTL0 strain | `POI:Control`, `dCas9:Control`; `Target:Control`; `dCas9/control`, `POI/control` | `si/si9.xlsx` `Figure 3c`, `Figure 3a`; `si/si4.xlsx` | no | blocked, gap R below | 92 x 2 plus 120 + 93 |
| normalized POI expression across the DBTL0 library, three example guides | `Normalized Expression` for `Target` in `{PP_3578, PP_1023, PP_0582}`, 120 strains each | `si/si9.xlsx` `Figure 3b` | no | blocked, gap R | 360 |
| per-protein log2 fold change and log10 p-value, 14 single-guide strains | `PP_0368_log2_FC` ... `PP_4192_log2_FC` and the matching `_log10_pval`; caption "Statistically significant Log2(Fold-change) values (paired two-tailed Student's $T-$ test, $p{<}0.05)$" | `si/si9.xlsx` `Figure 5b` | no | blocked, gap R | 5,770 fold changes plus 5,770 p-values over 1,290 protein rows |
| per-protein log2 fold change and log10 p-value, 10 KO and CRISPRi contrasts | same column pattern, plus `Control_log2_FC` | `si/si9.xlsx` `Figure 6b` | no | blocked, gap R | 452 plus 452 over 553 protein rows |
| protein levels significantly changed across the 25 best strains | 26 `POI` rows by 25 array columns; caption "Depicts the protein levels that were significantly changed (paired two-tailed Student's T-test, $p<0.05$) across 25 of the best performing sgRNA combinations" | `si/si9.xlsx` `Supplementary Figure 10` | no | blocked, and the basis is unsourced: the sheet names no unit and the measured range is -7.0576 to 3.9216 with median -1.0118, which is signed and so cannot be the percent-of-total basis the other abundance sheets use | 650 cells |
| ART predicted isoprenol titer per strain | `Predicted Isoprenol (mg/L)` beside `Measured Isoprenol (mg/L)`, with the cell `github source: DBTL ART/dbtl6_art_outputs/art_performance_df_plotting.csv` | `si/si9.xlsx` `Figure 4c` | no | not a phenotype, it is a model output; all 342 of its `Measured Isoprenol (mg/L)` values are already in `Figure 4b` | 342 |
| change in titer when a guide is added to an array | `parent_strain`, `parent_mean`, `child_mean`, `target_diff`, `target`, `N_target` | `si/si9.xlsx` `Figure 4d` | no | not a phenotype, derived from `Figure 4b` | 209 |
| guide occurrence counts and pass/fail/missing counts | `target`, `count`; `n_pass`, `n_fail`, `n_missing`, `n_total` | `si/si9.xlsx` `Figure 4e`, `Figure 4f` | no | not a phenotype, derived summary | 15 / 15 |
| KEGG pathway enrichment of the PP_0815 contrast | `GeneRatio`, `BgRatio`, `RichFactor`, `FoldEnrichment`, `zScore`, `pvalue`, `p.adjust`, `qvalue`, `Count` | `si/si9.xlsx` `Figure 6c` | no | not a phenotype, a derived enrichment | 265 |
| Spearman correlation of each protein with titer | `Spearman_Correlation`, `Average_Abundance_Control` | `si/si9.xlsx` `Supplementary Figure 14` | no | not a phenotype, a derived statistic over DBTL0 | 2,049 |
| ART model R-squared per cycle | `Cycle`, `Dataset`, `R^2` | `si/si9.xlsx` `Supplementary Figure 6c` | no | not a phenotype, a model metric | 12 |
| liquid-handler dispenses and plasmid build failure modes | `Dispenses`; `Failure Mode`, `# Plasmids` | `si/si9.xlsx` `Supplementary Figure 2a`, `Supplementary Figure 2b`, `Figure 2b` | no | not a phenotype, build-stage process QC | 12 / 18 / 18 |
| guide sequences, plasmid parts, array compositions, deletion success rates | `Sequence (5' to 3')`; `Plasmid Description`; `Construct`, `# sgRNAs`, `Vector`; Supplementary Table 5 | `si/si5.xlsx`, `si/si6.xlsx`, `si/si7.xlsx`, `si/si1.md` | design metadata, partly consumed | not a phenotype | 104 / 229 / 360 |
| presence of each target in the RB-TnSeq fitness library | `RbTn-Seq Data?` (Y/N) | `si/si4.xlsx` | no | not a phenotype, an annotation; see the comparison section | 120 |

Three cross-sheet measurements that bear on how a revision would have to be written.

1. **The unstored titer cultures are genuinely new, not a re-export.** Of the 186
   distinct titer values across `Figure 6a` (KO rows), `Figure 6d` (KO rows),
   `Supplementary Figure 13d` and `Supplementary Figure 12bd`, **zero** appear among
   `Figure 4b`'s 1,506 stored values. The sheets' own CRISPRi rows, by contrast, are
   re-exports: all 33 `Figure 6a` `CRISPRi` rows and all 99 `Figure 6d` `CRISPRi` rows
   match a `Figure 4b` value, so a revision must take only the non-CRISPRi rows.
2. **Two of the unstored sheets overlap each other, and disagree.** `Supplementary
   Figure 13d`'s `Non-Target` triplicate (298.0681, 309.4453, 293.9713) is bit-identical
   to `Figure 6a`'s `PP_0815` / `Non-target` triplicate, so that group would be stored
   twice unless deduplicated. Its `Target` triplicate is NOT the same cultures:
   `Figure 6a` gives 422.4448, 495.3173, 464.3335 and `Supplementary Figure 13d` gives
   464.3335, 477.8291, 450.4693, sharing exactly one replicate value. The two sheets
   therefore export two different triplicates for what both captions call the same
   strain.
3. **Supplementary Data 1's two expression headers are swapped relative to the Source
   Data's own naming.** Measured over the 91 targets that join: `si/si4.xlsx`'s
   `dCas9/control` column agrees with `Figure 3c`'s `POI:Control` for 91 of 91 within
   Supplementary Data 1's own two-decimal rounding (tolerance 0.006, zero
   disagreements), and its `POI/control` column agrees with `Figure 3c`'s
   `dCas9:Control` for 91 of 91. Against the like-named columns both disagree for 90 of
   91. A revision that stores the knockdown ratios must take `Figure 3c` as the
   authority on which number is which.

**What the loadable-now Carruthers rows come to, counted under the loader's own
convention** that a non-targeting control culture is a `phenotype_reference` rather than a
record.

| sheet | cultures | groups | records | references |
|---|---|---|---|---|
| `Figure 6a`, `Type` in `{Target, Non-target}` | 72 | 24 | 12 (one per KO background, target sgRNA) | 12 (same backgrounds, non-target sgRNA) |
| `Figure 6d`, `Type` in `{KO Only, CRISPRi and KO}` | 12 | 4 | 4 | 0, no non-target arm |
| `Supplementary Figure 13d` | 58 | 20 | 19 | 1 (`Non-Target`) |
| `Supplementary Figure 12bd` | 48 | 15 | 14 (two plasmids x 7 inducer levels) | 1 (`Control`, 6 cultures not 3) |
| **titer total** | **190** | **63** | **49** | **14** |
| `Supplementary Figure 12ac` (proteome) | 16 samples | 16 | 14 | 2 (`pSTABL1_Control`, `pSTABL2_Control`) |
| `Supplementary Figure 15` (proteome) | 90 | 7 cycle groups | 7 | 0 |

`Supplementary Figure 13d` is the sharpest single item: it is the titer arm of exactly the
20-sample panel whose proteome the loader already stores, so its 19 records pair one to
one with the 19 proteome records that exist today. `Supplementary Figure 15` is the
proteome arm of the control cultures whose titers are already the per-cycle
`phenotype_reference`; its cycle counts (18 in DBTL0, 12 in each of DBTL1 to DBTL6) match
that reference design exactly.

**Gap R, new and not one of the six issues checked.** There is no protein-level
relative-abundance or fold-change phenotype, and no per-protein p-value slot.
`ProteinAbundancePhenotype`'s own docstring rules the ratio out: "``protein_abundance``
maps each measured protein's systematic ORF id to its abundance (absolute per-strain
quantity on a log signal scale, NOT a ratio -- the WT/parent strain supplies the
reference)". The expression family has the ratio slot that the protein family lacks
(`MicroarrayExpressionPhenotype.expression_log2_ratio`, and `PseudobulkExpressionPhenotype`
carries "per-gene pseudobulk log2 fold-change vs the WT profiled in the same"), so the
asymmetry is in the protein family alone. Checked against #731, #739, #749, #753, #756
and #758: none covers it. #753 is the nearest neighbor in spirit, since it also asks for
a dispersion field on a protein phenotype, but its subject is
`ProteinTurnoverPhenotype`'s interval and censoring, not a ratio slot on
`ProteinAbundancePhenotype`.

Gap R partly dissolves if #739 is resolved: for the 92 DBTL0 strains of `Figure 3c` the
absolute POI and dCas9 abundances are in the Dryad campaign proteome, so the ratios
become derivable from stored absolutes rather than needing a ratio field. For
`Figure 5b`, `Figure 6b` and `Supplementary Figure 10` the released value is a fold
change with a p-value and the Source Data carries no absolute companion, so those stay
blocked on gap R itself.

**One factual correction to the loader's own documentation, offered without touching the
code.** `torchcell/datasets/pputida/carruthers2025.py` says the stored sheet is "the only
per-protein, per-replicate abundance matrix in the Source Data file", and
`notes/torchcell.datasets.pputida.carruthers2025.md` repeats it as "the only
per-protein, per-replicate abundance matrix the Source Data carries". `Supplementary
Figure 12ac` is a second one, with the identical column set
(`Protein.Group`, `Protein.Names`, `Protein`, `Protein.Description`, `Sample`, `Strain`,
`Inducer concentration`, `Replicate`, `Top_3pep_counts_mean`,
`%_of protein_abundance_Top3-method`, `log10_%_abundance`) and the same Top3 scale
(`Top_3pep_counts_mean` 0 to 1.0027e8, `%_of protein_abundance_Top3-method` 0.0031 to
13.717). It is narrower, 6 proteins rather than 1,501, which is presumably what the
sentence meant, but as written it is wrong.

**The comparison question.** Two comparisons, and neither is a duplication risk.

- Supplementary Data 1's `RbTn-Seq Data?` column and `paper.md`: "including KOs of 6
  genes that were plausibly essential based on their absence in the $P.$ putida RB-TnSeq
  fitness library (Supplementary Data $1)^{46}$". Reference 46 is "Price, M. N. et al.
  Mutant phenotypes for thousands of bacterial genes of unknown function. Nature 557,
  503-509 (2018)". The KT2440 arm of that compendium is already served: `price2018.py`
  states "The KT2440 arm of the same compendium is the Borchert 2024 loader's". So this
  comparison points at nothing new, and the column is a membership flag rather than a
  measurement, so there is nothing to store twice.
- Reference 47 is "Thompson, M. G. et al. Fatty acid and alcohol metabolism in
  Pseudomonas putida: functional analysis using random barcode transposon sequencing.
  Appl. Environ. Microbiol. 86, e01665-20 (2020)", cited for the isoprenol-degradation
  annotation in Supplementary Table 1. It is mirrored as `thompsonFattyAcidAlcohol2020`
  (doi 10.1128/AEM.01665-20) and has no loader of its own, but it is NOT a missing
  dataset: `borchert2024.py` already carries it as a subsumed source study
  (`THOMPSON2020_KEY = "thompsonFattyAcidAlcohol2020"`, with its own `Provenance` and
  eleven `SourcedValue` quotes), because the Borchert compendium "is the SUPERSET of the
  other KT2440 RB-TnSeq rows, so it is loaded here and the subsumed rows become
  per-record source-study provenance". So this comparison points at nothing new either.

### de Siqueira 2025

Mirror `$DATA_ROOT/torchcell-library/desiqueiraAlternateRoutesAcetate2025/`, `si_expected: []`.
Three SI artifacts, each opened: `si/si1.xlsx` (Data Set S1, one sheet, 34,600 data rows
x 15 columns), `si/si2.xlsx` (Data Set S2, 173 data rows x 27 columns), `si/si3.pdf` with
OCR `si/si3.md` (Tables S1 and S2, Figures S1 to S5). The paper's own list: "Data Set S1
(AEM02123-24-s0001.xlsx). Proteomics data for strains. Data Set S2
(AEM02123-24-s0002.xlsx). SNPs for strains. Supplemental material
(AEM02123-24-s0003.pdf). Table S1 and S2; Figures S1 to S5."

| released quantity | exact column or verbatim caption quote | SI file | stored? | classification | records |
|---|---|---|---|---|---|
| Top3 peptide signal, replicate mean | `Top_3pep_counts_rep_mean` | `si/si1.xlsx` | yes | stored | 5 records, 7,655 values |
| Top3 signal SD | `Top_3pep_counts_rep_std` | `si/si1.xlsx` | yes, as SE = SD / sqrt(3) | stored | same records |
| **percent-of-total relative abundance, replicate mean** | `%_of protein_abundance_Top3_rep_mean` | `si/si1.xlsx` | **no, and nowhere recorded** | **loadable now**: `ProteinAbundancePhenotype` / `BacterialProteinAbundanceExperiment`, whose `measurement_type` is a free `str`, so a second normalization is a second record set | 5 records, 7,655 values now; 20 records and 30,620 values once #731 admits the evolved clones |
| percent-abundance SD | `%_of protein_abundance_Top3-rep_std` | `si/si1.xlsx` | **no, nowhere recorded** | **loadable now** as the SE of the row above | same 5 records |
| **mean of log10 percent abundance** | `log10_%_abundance_rep_mean` | `si/si1.xlsx` | **no, nowhere recorded** | **loadable now**, same class, third `measurement_type` | 5 records, 7,655 values |
| SD of log10 percent abundance | `log10_%_abundance_rep_std` | `si/si1.xlsx` | **no, nowhere recorded** | **loadable now** as its SE | same 5 records |
| CV of percent abundance | `CV%_of_%_protein_abundance` | `si/si1.xlsx` | no | not a phenotype, derived: measured to equal `100 * pct_std / pct_mean` with zero mismatches over 2,000 sampled rows | 0 |
| constant per-protein SEM | `%_of protein_abundance_Top3_rep_mean_sem` | `si/si1.xlsx` | no | not a phenotype, and already recorded in the note ("is NOT used: it is constant across every sample of a protein"); measured, 1,728 of 1,729 proteins carry a single value | 0 |
| 173 variant calls over 5 clones, with coverage and frequency | `Protein Effect`, `Variant Frequency`, `Coverage`, `Amino Acid Change`, `CDS Codon Number` | `si/si2.xlsx` | no | blocked, issue #731, already recorded; the census re-measured and matches: Sigma4 43, PT 33, Sigma5 35, Sigma2 28, Sigma1 34 = 173 | 0 |
| max isoprenol titer, 14 numeric Sigma cells | "Table S2. Maximum isoprenol titers (mM) detected for Sigma-class and PT strains grown in different growth media" | `si/si3.md` | no | blocked, issue #731 (the Sigma strains are unwritable), already recorded | 14 |
| max isoprenol titer, 2 PT acetate cells, both 0 | same caption | `si/si3.md` | no | blocked, the medium is absent from `MEDIA_LIBRARY`, already recorded as the note's open gap 3 | 2 |
| per-time-point titers at 24, 48 and 72 h | "Data points shown represent three independent biological replicates, and the error bars indicate standard deviation from the mean" (Fig. 3 caption) | figure only, Fig. 3B and 3C | no | not released as a table, already recorded verbatim in `si_expected` | 0 |
| OD600 at 24 and 48 h | "Figure S1. Cell densities reached at 24 and 48 hours of cultivation ... The absorbance values for each individual replicate are represented by the black points" | `si/si3.md`, figure only | no | not released as a table, and `MeasurementType` has no optical-density member | 0 |
| residual glucose and acetate, 6 time points over 3 media | "(D) Measurement of residual glucose (left) and acetate (right) in the supernatant of cultures grown in test tubes with model hydrolysate medium" | figure only, Figs. 1B to 1D and 2B to 2D | no | not released as a table | 0 |
| growth curves, 15-minute reads over 48 h | "Readings were taken every 15 min" (Methods, Measurement of growth kinetics) | figure only, Figs. 1A, 2A, 5D, 5E, S2, S5 | no | not released as a table; no growth-rate or lag-time column exists anywhere in the mirror | 0 |
| per-protein log2 fold change with BH-adjusted p | "Comparisons with BH-adjusted P-values (cutoff set to <=0.05) and absolute log2 fold change >2 or <-2" (Fig. 5) | figure only, Figs. 4 and 5B | no | not released as a table, and it would hit gap R | 0 |
| isolation batch and isoprenol selection dose | "Table S1. Summary information about the isolation conditions of the recovered Sigma-class strains" | `si/si3.md` | no | not a phenotype, a design parameter | 0 |
| 192 host protein keys outside the namespace | not applicable | `si/si1.xlsx` | no | blocked on a UniProt-to-locus-tag crosswalk, already recorded as the note's open gap 2 | 0 |

**The headline, verified directly.** Five released per-record proteomics statistics are
neither stored nor mentioned in the loader or the note. The loader's `PROTEOME_HEADER`
asserts only the first nine header cells and the reader consumes `row[7]` and `row[8]`,
so columns 9 through 14 are never touched; the released header is, verbatim from the
bytes, `('Protein.Group', 'Protein.Names', 'Protein', 'Protein.Description', 'Strain',
'Condition', 'Sample', 'Top_3pep_counts_rep_mean', 'Top_3pep_counts_rep_std',
'%_of protein_abundance_Top3_rep_mean', '%_of protein_abundance_Top3-rep_std',
'log10_%_abundance_rep_mean', 'log10_%_abundance_rep_std',
'CV%_of_%_protein_abundance', '%_of protein_abundance_Top3_rep_mean_sem')`. This is more
than bookkeeping, because the paper's own analyses run on the column we do not store:
"Mean protein counts and relative protein abundances for each sample, as well as other
summary statistics for the proteomics data set, are available in File S1." and
"Relative protein abundances of proteins were used for dimensionality reduction."
(Fig. 5A). The percent column is NOT recoverable from what we store: the per-sample ratio
of released percent to percent-of-the-mean-counts runs 0.959 to 0.997, so it is a
replicate-wise mean of per-replicate percentages rather than a rescaling. Likewise
`log10_%_abundance_rep_mean` is the mean of log10s, not the log10 of the mean (1,999 of
2,000 sampled rows disagree; the first row's released -2.23386 against log10 of the mean
-2.19547).

### Kang 2026

Mirror `$DATA_ROOT/torchcell-library/kangMultilayeredMetabolicRemodeling2026/`,
`si_expected: []`. One SI file, `si/si1.docx`, read in full: Tables S1 to S9 and the
captions of Figures S1 to S10, with data-row counts measured from `word/document.xml` as
S1 5, S2 152, S3 173, S4 5, S5 38, S6 40, S7 4, S8 9, S9 7. No spreadsheet and no
proteomics matrix was released; the paper states "Data will be made available on request."

| released quantity | exact column or verbatim caption quote | SI file | stored? | classification | records |
|---|---|---|---|---|---|
| flask max isoprenyl acetate titer, 7 strains | `Titer (mg/L); culture conditions` | `paper.md` main Table 1 | yes | stored | 7 |
| tube isoprenyl acetate titer, 5 AATs | `IPA titer (mg/L)` | `si/si1.docx` Table S4 | yes | stored | 5 |
| fed-batch isoprenyl acetate, 3 phases | `Isoprenyl acetate, aqueous (mg/L)`, `Isoprenyl acetate, organic (mg/L)`, `Isoprenyl acetate, off-gas (mg/L)` | `si/si1.docx` Table S9 | yes, stored as the sum | stored | 7 |
| **fed-batch aqueous isoprenol titer, 7 time points** | `Isoprenol, aqueous (mg/L)` | `si/si1.docx` Table S9 | **no, and nowhere recorded** | **loadable now**: a second `ProductTiterPhenotype` family with `product` = isoprenol, on the same mg/L-as-`ug/mL` convention. `SI_TABLE9_COLUMNS` asserts the column name but `FedBatchRow` declares no field for it, so the value is parsed past and never read | 7 (190.7, 197.0, 175.0, 231.0, 234.7, 307.8, 254.2 mg/L) |
| residual glucose and xylose, 7 time points | `Glucose (g/L)`, `Xylose (g/L)` | `si/si1.docx` Table S9 | parsed into `FedBatchRow`, consumed only by the total-sugar build oracle | **loadable now with a caveat**: `MetabolitePhenotype` / `BacterialMetaboliteExperiment` holds a per-metabolite level, but it has no units field, so `g/L` would sit inside free-text `measurement_type`, and `n_replicates` is a required non-gappable dict, so n = 1 must be asserted from "a single run" rather than quoted | 7 records x 2 analytes |
| total sugar | `Total sugar (g/L)` | `si/si1.docx` Table S9 | no | not a phenotype, derived as glucose + xylose, already recorded as build oracle 2 | 0 |
| off-gas fraction | `Off-gas fraction (%)` | `si/si1.docx` Table S9 | no | not a phenotype, derived, already recorded as build oracle 1 | 0 |
| **per-protein differential expression, 40 proteins** | `Fold Change`, `Log2 (Fold Change)`, `p-Value (Equal Variance)`, `(-Log10 (p-Value))`, `Category`, `Rank`, under "Table S6. List of top 20 accessory genes upregulated and downregulated by sgRNA targeting PP_4854" | `si/si1.docx` Table S6 | **no, and Table S6 is referenced nowhere in the loader** | **blocked, gap R**, found independently here and in Carruthers. Two missing things: no class holds a per-protein log2 fold change, and the only p-value field in all of `schema.py` is `gene_interaction_p_value` on `GeneInteractionPhenotype`. Secondary: the keys are UniProt accessions (`Q88HX1`), so it also needs the crosswalk the de Siqueira note flags | 40, one strain pair (PIPA-D16 against PIPA-C) |
| percent-of-theoretical yield | "corresponding to a yield of $0.067 \mathrm{g}/\mathrm{g}$ and $18.1\%$ of the theoretical maximum (Fig. 8b)" | `paper.md` Results 3.6 | the 0.067 g/g half is stored; 18.1 % is not | blocked: `ProductTiterPhenotype` carries one `product_yield` plus one `product_yield_unit`, so `g_per_g_substrate` and `percent_of_theoretical` cannot both sit on one record | 1 |
| max OD600, fed-batch and flask | "The maximum $\mathrm{OD}_{600}$ was 13.2, lower than the 16.8 observed in flasks (Fig. 7d)" | `paper.md` Results 3.6 | no | blocked: `MeasurementType` has no optical-density member (`log2_ratio`, `z_score`, `sensitivity_score`, `categorical`, `ordinal`, `growth_rate`, `differential_fitness`, `control_regression_residual`, `colony_size`) | 2 |
| ester loss to native hydrolysis | "P. putida cultures showed an additional loss of $34 \mathrm{mg/L}$ in isoprenyl acetate, accompanied by appreciable isoprenol" | `paper.md` Results 3.2 | no | not a phenotype as released, a single prose delta for KT2440 with no table; Fig. S10's per-host values are figure only | 0 |
| compound fuel properties | "Table S1. Comparison of physicochemical and fuel-relevant properties of isoprenol and isoprenyl acetate", including a measured RON of 108.4 | `si/si1.docx` Table S1 | no | not a phenotype, a compound property rather than genotype x environment | 0 |
| putative esterase annotation, 38 rows | "Table S5. Putative esterases in P. putida"; `No.`, `Locus tag`, `Predicted enzyme function`, `Functional class` | `si/si1.docx` Table S5 | partly, the three deleted esterases' functions are sourced | not a phenotype, an annotation | 0 |
| overlay solvent properties, 4 rows | "Table S7. Summary of four overlay solvents evaluated in this study" | `si/si1.docx` Table S7 | no | not a phenotype, a design parameter | 0 |
| acetyl-CoA gene design, 9 rows | "Table S8. Summary of genes engineered to enhance acetyl-CoA biosynthesis in P. putida" | `si/si1.docx` Table S8 | partly, the cassette tokens | not a phenotype, design | 0 |
| strains and plasmids, 152 rows; oligos, 173 rows | "Table S2. Strains and plasmids used in this study.", "Table S3. Oligos used in this study." | `si/si1.docx` | partly | not a phenotype, design | 0 |
| final OD600 in the hydrolysis assay, and titers under 10 other conditions | "(e) Final OD600 after 24 hours under ester hydrolysis assay conditions" (Fig. S1), plus Figs. S2 and S4 to S10 | `si/si1.docx`, figures only | no | not released as a table | 0 |
| isoprenol titers for the 5 AAT strains | "(b) Isoprenyl acetate and isoprenol titers in P. putida denoted by organism of origin and the name of each AAT enzyme" (Fig. 2) | figure only; Table S4 releases only the ester | no | not released as a table | 0 |
| the 5 Table 1 strains with no titer | not applicable | `paper.md` | no | not a phenotype, already recorded in `preprocess/strains_without_a_titer.csv` | 0 |

The Table S9 finding was verified directly against the loader: `SI_TABLE9_COLUMNS`
contains `"Isoprenol, aqueous (mg/L)"` and `class FedBatchRow` declares
`time_hours`, `glucose_g_per_l`, `xylose_g_per_l`, `total_sugar_g_per_l`,
`aqueous_mg_per_l`, `organic_mg_per_l`, `offgas_mg_per_l` and
`released_offgas_fraction_percent`, with no isoprenol field.

### Borchert 2024

Mirror `$DATA_ROOT/torchcell-library/borchertMachineLearningAnalysis2024/`, `si_expected: []`
in the library mirror and 1 entry in the raw mirror. Four released SI files, matching the
paper's own list: "File S1 (mSystems00942-23-S0001.xlsx). Functional annotation clustering
for fModules. / File S2 (mSystems00942-23-S0002.xlsx). Gene names and weights in fModules.
/ File S3 (mSystems00942-23-S0003.txt). Code for generating Sankey diagram in Figure 6. /
Supplemental Materials (mSystems00942-23-S0004.pdf). Figures S1-S10; Tables S1-S4." The
fitness matrix itself is not in the SI: it is `fModule_Metadata.xlsx` in the raw mirror,
from the `beckham-lab/fModule` repository.

| released quantity | exact column or verbatim caption quote | SI file | stored? | classification | records |
|---|---|---|---|---|---|
| gene fitness per (gene, sample) | 332 sample columns such as `set1IT078 Glucose (C) (40mM)`, sheet `fitness_measurements` | raw mirror `data/fModule_Metadata.xlsx` | 1,372,280 of 1,571,024 | already recorded: 42 samples dropped, with the drop table in the note | 198,744 unstored = 4,732 x 42 |
| t-like test statistic per (gene, sample) | sheet `T-like_statistics`, same 332 columns; "(iv) $t$-like test statistic. To estimate the reliability of the fitness measurement for each gene $f$, we use a moderated $t$ statistic" | same file | consumed to derive `environment_response_se` for 867,978 Fitness Browser records, never stored verbatim | blocked: no field for a companion test statistic, and `UncertaintyType` has only `sample_sd`, `standard_error`, `bootstrap_se`, `variance`, `ci95` with "deliberately NO `unknown`". The GAP is already recorded as `_GAP_UNCERTAINTY`; the number travelling nowhere is not. Nearest issue is #749 item 2 (s_score), but that asks for a `MeasurementType` member for the VALUE, not a field for a second statistic | 1,372,280 |
| per-sample replicate count | `total_rep`, `rep`, sheet `metadata` | same file | parsed into `SampleMetadata`, then unused | not a phenotype, a grouping parameter; replicates are already recoverable from identical `Environment` plus distinct `screen_id` | 290 |
| sample pH | `pH` (6.9, 7) | same file | no | not a phenotype: `Environment` has no pH field, and 6.9 against 7.0 is the medium rather than a perturbation | 290 |
| vessel, shaking, time-zero set, operator, date, seq index, inoculum medium | `vessel`, `shaking`, `timeZeroSet`, `person`, `dateStarted`, `seqindex`, `Inoculum media type` | same file | no | not a phenotype; already recorded for vessel and shaking. Worth noting: `CultureEnvironment.culture_format` does carry `vessel` and `shaking_rpm`, and this loader uses the base `Environment` | 290 |
| fModule explained variance | `Explained Variance` | `si/si1.xlsx` | no | not a phenotype, a derived ICA summary; no retention decision exists in loader or note | 84 |
| fModule gene count and annotations | `Number of genes`, `Functional Annotation Clustering (DAVID)`, `Functional Annotations (Manual Curation)`, `Classification`, `Notes on fModule activity` (the released header carries a trailing space), `Genes in fModulon` | `si/si1.xlsx` | no | not a phenotype, annotation or derived | 84 |
| ICA gene weight per (gene, fModule) | `gene_weight` | `si/si2.xlsx`, 465 rows over 40 fModules and 377 loci | no | not a phenotype, a model coefficient | 465 |
| operon, COG, regulator, iModulon membership | `operon`, `COG`, `regulator` (208 non-empty), `iModulon` (176 non-empty) | `si/si2.xlsx` | no | not a phenotype, annotation; `iModulon` is the cross-reference to Lim 2022 | 465 |
| Sankey plotting code | "Supplementary File 3. Python code for generation of Sankey diagram in Figure 6." | `si/si3.txt` | no | not a phenotype | not applicable |
| OD600 growth curves of wild type, transposon mutants and overexpression strains | Figs. 2C to 2F, 4C to 4H, S7 to S10, for example "Error shading indicates the standard deviation from the mean of three biological replicates"; Methods "the optical density at $600\mathsf{nm}$ $(\mathsf{OD}_{600})$ was measured every $15\mathsf{min}$" | `si/si4.pdf` is figure images plus Tables S1 to S4 only | no | not loadable from this SI, figure only. The sibling Borchert 2023 SI DOES release its growth curves (see below) | 0 from this SI |
| replicate-pair Pearson R over the 332 experiments | "(A) Pearson R correlation coefficients between every pair of the 332 experiments" (Fig. S1) | `si/si4.pdf`, figure only | no | not a phenotype, a derived QC summary with no released table | 0 |
| plasmids, primers, strains, synthetic DNA | "Table S1. Plasmids used in this study.", "Table S2. Sequences of DNA oligos used in this study.", "Table S3. Bacteria used in this study, including construction details ... and barcode details for individually arrayed mutants.", "Table S4. Synthetic DNA." | `si/si4.md` | no | not a phenotype, design parameters. The barcode ids (`KT2440 PP_0356::TC1434721(Km)`) would enrich a genotype but carry no measurement | 0 |

Already recorded and therefore not re-reported: the 832 protein-coding genes the compendium
itself eliminated ("In instances where gene fitness data for a particular gene did not
exist across all 332 data sets, the gene was eliminated from analysis"), and the
`n_samples`, `sample_unit` and `duration_hours` gaps. The 22 dropped bioreactor and
adaptation samples need "dissolved-oxygen set point, batch or fed-batch feed rate, sampling
time, solvent overlay and adaptation day", which is the same shape as **#753 item 3**
(`Environment` cannot express a chemostat dilution rate) and belongs there rather than in a
new issue.

**The comparison question, and this one is the sharpest duplication risk in the whole
audit.** The compendium is an explicit aggregation: "collecting 332 publicly available
P. putida RB-TnSeq data sets ... The Fitness Browser ... was used to obtain data for
254/332 of the samples, and the remaining data were generated by the Beckham group ...
PRJNA809672, PRJNA856070, and PRJNA1011287 (22, 38)". The note's superset table already
attributes Thompson 2020 (46 samples), Schmidt 2022 (123) and Borchert 2023 (42). None of
those is separately loaded, and the only mention of `borchertRBTnSeqIdentifiesGenetic2023`
in the tree is `borchert2024.py`'s `BORCHERT2023_KEY`, as provenance.

Borchert 2023 is mirrored, with a 70.9 MB `si/si1.xlsx` holding 13 pairwise comparison
sheets. **Measured, re-run independently for this note:** joining its sheet
`Glu_v_Glu_Van` on `old_locus_tag` against the compendium's `sysName`, all 4,732
overlapping loci agree on every one of six per-replicate columns, with max absolute
difference 0.000500 and 4,732 of 4,732 pairs within 5e-4 on each:

| Borchert 2023 column | compendium column | n | max abs diff |
|---|---|---|---|
| `M9_Glucose_RepA` | `set100IT008 D-Glucose (C)` | 4,732 | 0.000500 |
| `M9_Glucose_RepB` | `set100IT009 D-Glucose (C)` | 4,732 | 0.000500 |
| `M9_Glucose_RepC` | `set100IT010 D-Glucose (C)` | 4,732 | 0.000500 |
| `M9_Glucose_Vanillin_RepA` | `set100IT023 D-Glucose with Vanillin (C)` | 4,732 | 0.000500 |
| `M9_Glucose_Vanillin_RepB` | `set100IT024 D-Glucose with Vanillin (C)` | 4,732 | 0.000500 |
| `M9_Glucose_Vanillin_RepC` | `set100IT025 D-Glucose with Vanillin (C)` | 4,732 | 0.000499 |

5e-4 is exactly the half-unit of the compendium's three-decimal rounding, and Borchert 2023
releases full precision, so this is one measurement exported twice. That is the Rousset
2018 / Cui 2018 pattern of **#760**: landing a Borchert 2023 loader over those comparison
sheets would store the same fitness a second time.

What Borchert 2023 releases that the compendium does NOT, and so is genuinely new if anyone
loads it: `t-statistic`, `p-value`, `q-value` and `adjusted_q-value` per gene per
comparison ("Significant difference in mean fitness between groups was determined with a
two-sample t-test corrected for multiple testing by the pFDR method. q values are
reported."), roughly 65,000 per-gene test results and a DIFFERENT t from Wetmore's
moderated t; barcode-level counts in `Exp1_All_Poolcount` and `Exp2_All_Poolcount`
(186,957 rows, "Contains all counts across all conditions for unqiuely barcoded
transposons in experiments 1 - 12"), which sit below the compendium's gene-level
aggregate; and growth curves in `Figure_2..6_growth_data`, "Growth data measured in
arbitrary units [a.u.] of back-scatted 620 nm light with gain set to 3".

Two further comparisons point at nothing new. Fig. 6 and File S2 cross-reference Lim 2022's
iModulons ("Numerical data for this figure are provided in File S2"; 176 rows carry an
`iModulon` value), and Lim 2022 is already loaded with a different measurement (expression
rather than fitness). The paper names "a companion iModulon data set already exists for
Escherichia coli BW25113 (30)" as a future comparison; that compendium line is already
covered by `ecoli/lamoureux2023.py`.

### Lim 2022

Mirror `$DATA_ROOT/torchcell-library/limMachinelearningPseudomonasPutida2022/`. Two released
SI files, matching "Appendix A. Supplementary data": `si/si1.docx` and `si/si2.xlsx`
(8 sheets), both opened in full.

| released quantity | exact column or verbatim caption quote | SI file | stored? | classification | records |
|---|---|---|---|---|---|
| log2(TPM + 1) expression per (gene, sample) | sheet `5-X`, header `Geneid` plus 503 sample columns, 5,564 genes | `si/si2.xlsx` | 1,001,520 values over 180 samples | split, see the next two rows | 1,797,172 unstored |
| the 141 PRECISE-321 samples the loader cannot write | `putidaPRECISE321` = 1 with an unwritable strain | `si/si2.xlsx` sheet `5-X` | no | already recorded, with the note's drop table: `engineered_strain_code` 71, `non_reference_background` 24, `plasmid_content` 15, `engineered_evolved_sugar_strain` 14, `deleted_gene_unresolved` 11, `evolved_isolate` 4, `sequence_variant_allele` 2. **Partly blocked beyond #731**: the 6 evolved and variant samples fit #731, and the 100 engineered, plasmid and heterologous-cassette samples fit NO open issue, because #731 covers only the sequence-variant half | 784,524 = 5,564 x 141 |
| the 182 samples in `5-X` that are not in PRECISE-321 | 182 columns with `putidaPRECISE321` blank. Measured composition: 94 non-KT2440 (`ScientificName` `Pseudomonas putida` 48, `Pseudomonas putida DOT-T1E` 38, `Pseudomonas putida BIRD-1` 8), 24 KT2440 with `PrimaryQC` = `Fail`, and 64 KT2440 with `PrimaryQC` = `Pass` | `si/si2.xlsx` sheet `5-X` | no | NOT previously recorded as a decision. Not loadable: the paper's own QC excludes them, "non-P. putida KT2440 samples were excluded for consistency of gene expression profiles" and "samples with low correlation within biological replicates $(R^2 < 0.95)$ were discarded". Worth recording explicitly as a deliberate drop | 1,012,648 = 5,564 x 182 |
| raw featureCounts read counts | `counts.csv`, header `Geneid` plus 521 sample columns, 5,564 genes | raw mirror `data/counts.csv`, not the SI | 1,001,520 stored as `expression_count` | already recorded, pairing measured per sample to 5.0e-9 | 1,897,324 unstored |
| iModulon gene weights (M matrix) | sheet `6-M`, header `Geneid`, `0` to `83` | `si/si2.xlsx` | no | not a phenotype, ICA model coefficients; no retention decision in loader or note | 467,376 = 5,564 x 84 |
| iModulon activities (A matrix) | sheet `7-A`, 321 sample columns | `si/si2.xlsx` | no | not a phenotype, a derived latent activity | 26,964 = 84 x 321 |
| per-iModulon enrichment statistics | sheet `4-iM table`: `pvalue`, `qvalue`, `precision`, `recall`, `f1score`, `TP`, `regulon_size`, `imodulon_size`, `n_regs`, `exp_var`, `TU_size`, `reg_type`, `Category`, `Sub_category`, `Function`, `iModulon_type` | `si/si2.xlsx` | no | not a phenotype, derived statistics over the ICA output | 84 |
| TF-gene regulatory interactions | sheet `3-TRN`: `regulator_id`, `gene_id`, `gene_name`, `Pputida`, `evidence`; "We compiled 1993 experimentally validated or predicted interactions ... from multiple databases and literature" | `si/si2.xlsx` | no | not a phenotype, annotation | 1,993 |
| gene coordinates and annotation | sheet `2-Gene table`: `locus_tag`, `gene_name`, `accession`, `old_locus_tag`, `start`, `end`, `strand`, `gene_product`, `COG`, `uniprot`, `operon` | `si/si2.xlsx` | consumed for gene lengths only | not a phenotype, annotation. Its `uniprot` column is the same cross-reference route #753 item 4 asks for | 5,564 |
| per-sample QC verdicts and provenance | `PrimaryQC`, `passed_fastqc`, `passed_reads_mapped_to_CDS`, `passed_global_correlation`, `passed_similar_replicates`, `passed_number_replicates`, `KT2440`, `reference_condition`, `rep_name`, `GEO Series`, `GEO Sample`, `BioProject`, `DOI`, `PMID`, `biosample_strain`, `biosample_genotype`, `Note` | `si/si2.xlsx` sheet `1-Sample_list` | provenance fields go to `preprocess/sample_ledger.json` | not a phenotype, metadata, already recorded | 549 |
| per-iModulon activity change, exponential to stationary | "Supplementary Table 3. iModulons with differential activity changes between the exponential growth phase and the stationary phase", column `Activity change` | `si/si1.docx` | no | not a phenotype, a derived delta of a latent activity | 22 |
| best enriched motif per iModulon | "Supplementary Table 2. The best enriched motif from transcriptional units in iModulons", columns `Name`, `Enriched motif`, `E- value`, `# of TUs`, `# of genes`, `Width` | `si/si1.docx` | no | not a phenotype, sequence-derived | about 30 |
| regulatory-iModulon summary | "Supplementary Table 1. Selected 30 Regulatory iModulons" | `si/si1.docx` | no | not a phenotype, annotation | 30 |
| top-five gene weights per iModulon | "Top five genes with highest weights:" with `Locus tag \| Gene weight \| Gene name \| Gene product`, 84 blocks in Supplementary Note 1 | `si/si1.docx` | no | not a phenotype, a subset of the M matrix | 420 |
| maximum specific growth rate per strain | "(b) Maximum specific rates $(\mathbf{h}^{-1})$ of the six strains in a glucose minimal medium" (Fig. 5b); the text gives four of six, "three evolved P. putida KT2440 strains (A1_F11_I1, A3_F85_I1, and A4_F88_I1 ...) displaying higher growth rates $(0.64 \mathrm{h}^{-1}$, $0.77 \mathrm{h}^{-1}$, and $0.76 \mathrm{h}^{-1}$, respectively) over the wildtype $(0.58 \mathrm{h}^{-1}$" | NOT in `si1.docx` or `si2.xlsx`, figure and prose only | no | not loadable from this paper: four of six values exist only in prose, the measurement is attributed to Lim et al. 2020 (10.1039/D0GC01663B) which is not mirrored, and the genotypes are evolved relA clones, so #731 besides | 6 if ever sourced |
| specific growth rate per carbon source | "we investigated the activity of the 'stationary phase' iModulons ... with specific growth rates $(0.16 \mathrm{h}^{-1}$ to $0.62 \mathrm{h}^{-1}$, Fig. 4c and d)" | not released, figure only, and no column in `1-Sample_list` | no | not loadable, only the range appears in text | 0 |
| differentially expressed genes, exponential against stationary | "A total of 1454 DEGs were obtained" with `padj < 0.05` (Supplementary Figure 7 caption) | not released, volcano plot only | no | not loadable, no per-gene `padj` table released | 0 |

**The comparison question.** The note already states the within-paper conclusion: 225 of
the 321 samples are reprocessed from 21 projects, none of those 21 is its own row among the
fifty, and the one duplicate (PRECISE-321 returned under two keys) was already merged. None
of the other seven `pputida/` loaders carries a transcriptome, so there is no duplication
risk today. The standing forward risk: if Lim 2020 (16 samples), Lim 2021 (14), Bentley
2020 (9) or any of the other 18 sources is ever added as its own row, its RNA-seq is
already in our store at log2 TPM through PRECISE-321.

The paper's external comparisons point at nothing new. Fig. 2l compares gene weights
against "the Cbl+CysB iModulon in E. coli (Lamoureux et al., 2021)" and Sastry 2019, whose
compendium is already `ecoli/lamoureux2023.py` and is a different organism. The regulon
compilation (sheet `3-TRN`) is a network, not measurements. iModulonDB is a view over the
same M and A matrices already in `si2.xlsx`. The one comparison that WOULD be new, Borchert
2024's Fig. 6 join of these iModulons to its fModules gene by gene, is reproducible from
our store already, because both papers are loaded and the join key is the locus tag on
every record.

### Lim 2025

Mirror `$DATA_ROOT/torchcell-library/limEvolutionguidedToleranceEngineering2025/`. Two SI
files, both opened in full: `si/si1.docx` (5,492,159 B) and `si/si2.xlsx` (2,769,528 B, 10
sheets, enumerated as `Fig 2B_Mutation List`, `Table S4_GeneList`,
`ProteomeXchange Sample Key`, `Proteome_A10F63I1vsIPL400_M9G`,
`Proteome_A10F63I1vsIPL400_G+4IP`, `Proteome_A12F53I1vsIPL400_M9G`,
`Proteome_A12F53I1vsIPL400_G+4IP`, `IPL400vsA10F63I1_pIY670_M9G_12h`,
`IPL400vsA10F63I1_pIY670_M9G_24h`, `IPL400vsA10F63I1_pIY670_M9G_48h`).

| released quantity | exact column or verbatim caption quote | SI file | stored? | classification | records |
|---|---|---|---|---|---|
| per-lineage initial growth rate, isoprenol arm | `Initial growth rate (h-1)`; footnote a "Average growth rate, observed in the three first flasks of each experiment." | `si/si1.docx` Table S3 | yes, 2 records | stored | 2 |
| per-lineage final growth rate, 16 values | `Final growth rateb (h-1)`, footnote b naming the three LAST flasks | `si/si1.docx` Table S3 | no | blocked, #731; already recorded as the drop rule `final_growth_rate_is_an_evolved_population` | 16 arms, 4 records if aggregated by lineage group |
| HCHO-arm initial growth rate, 4 values on KT2440 | `Initial growth rate (h-1)`, rows `HCHO TALE KT2440` 13 to 16 (0.284, 0.247, 0.294, 0.288) | `si/si1.docx` Table S3 | no | blocked, no released unstressed denominator; already recorded as `hcho_arm_has_no_strain_other_than_the_reference` | 1 |
| per-lineage passages, generations, cumulative cell divisions | `Passages #`, `Generations`, `CCD (1012)` | `si/si1.docx` Table S3 | kept in `preprocess/table_s3.csv` | not a phenotype, campaign-progress metrics, already recorded as build-time assertions | 0 |
| per-lineage ending dose | `Ending concentration` (8.5, 8, 7.5 g/L; 9 mM) | `si/si1.docx` Table S3 | no | not a phenotype, the dose reached | 0 |
| per-clone variant call frequency, 443 calls on 159 positions | matrix cells under `A1 F0 I1 R1` to `A16 F68 I1 R1`, keyed by `Reference Seq`, `Position`, `Mutation Type`, `Sequence Change`, `Gene`, `starting?`, `Details` | `si/si2.xlsx` `Fig 2B_Mutation List` | no | blocked, #731, already recorded and typed into `preprocess/called_variants.json`. Independently re-counted here: 7 metadata headers in columns 2 to 8 and 49 clone columns in 9 to 57 | 46 evolved clones |
| **`Fit mean`, 49 per-clone values** (56.8, 57, 67.8, 0, 80.5, ..., 36.7) | the literal row label `Fit mean` in column I of row 2, aligned to the 49 clone columns | `si/si2.xlsx` `Fig 2B_Mutation List` | no | **new, and unclassifiable**: a case-insensitive search for `fit mean` finds nothing in `paper.md` or in the extracted `si1.docx` text, so the quantity's definition appears in no released artifact. Not guessed here. The loader's `read_variant_calls` filters on `row[3]` (Position) and so correctly excludes this row | 0 |
| **`Sum mutation`, 49 per-clone values** | the literal row label `Sum mutation` in row 3 | `si/si2.xlsx` `Fig 2B_Mutation List` | no | new, a derived summary: consistent with the sum of each clone's call frequencies (A1 F0 = 0, A5 F0 = 5.8, A9 F0 = 7.9 against the note's 0, 6 and 8 calls with 12 calls at 0.9) | 0 |
| IPL400 proteome under 4 g/L isoprenol, 2,361 keys | `log2_mean_IPL400_M9G+4IP`, `log2_std_IPL400_M9G+4IP` | `si/si2.xlsx` `Proteome_A10F63I1vsIPL400_G+4IP` | yes, 1 record plus a reference | stored | 1 |
| four evolved-isolate proteome arms | `log2_mean_A10_F63_I1_M9G`, `log2_mean_A10F63I1_M9G+4IP`, `log2_mean_A12_F53_I1_M9G`, `log2_mean_A12F53I1_M9G+4IP`, with the matching `log2_std_*` | `si/si2.xlsx`, four comparison sheets (2,367 / 2,374 / 2,367 / 2,338 data rows) | no | blocked, #731, already recorded as the rule `arm_is_an_evolved_isolate` | 4 |
| six pIY670 production-culture proteome arms | `log2_mean_IPL400_pIY670_M9G_12hr/24hr/48hr`, `log2_mean_A10F63I1_pIY670_M9G_12hr/24hr/48hr`, with the matching `log2_std_*` | `si/si2.xlsx` `IPL400vsA10F63I1_pIY670_M9G_12h/24h/48h` (2,350 / 2,350 / 2,378 rows) | no | already recorded as `production_medium_has_no_media_library_entry`. **Refinement the note does not split**: the three A10F63I1 arms are ALSO blocked by #731, so only the three IPL400 arms are reachable with no schema change, and a `MEDIA_LIBRARY` entry alone is not sufficient, because those three sheets pair IPL400 against the evolved strain, so an IPL400-only record needs a new pairing decision (12 h as the reference for 24 h and 48 h, for example) | 3 blocked by #731; 0 to 2 reachable, pending a pairing decision |
| **`A12_F53_I1_pIY670_M9G` at 12, 24 and 48 h** | `ProteomeXchange Sample Key` row 8: `A12_F53_I1_pIY670_M9G` / `HL_0640_` / `R1,R2,R3` / `12hr, 24hr, 48hr` | `si/si2.xlsx` | no | **new**: three processed arms are DECLARED in the sample key and no comparison sheet for them exists in the workbook; the values live only as raw spectra in PRIDE PXD054609. Not loadable from the SI, and #731-blocked besides | 3 |
| per-protein `p-value` and `p_adjusted(BH)` | `p-value`, `p_adjusted(BH)` | `si/si2.xlsx`, all seven proteome sheets | no | new, and mostly derived: `ProteinAbundancePhenotype` has no significance field (gap R again). `p-value` is recomputable from the stored mean and SE at n = 3, and the loader already asserts the Welch identity to 9.2e-13. `p_adjusted(BH)` is NOT exactly recomputable, because BH depends on the full tested set and the loader drops 13 of 2,374 keys | 0 |
| per-protein `t-test_stat` and `log2_Fold_change_A/B` | those two headers | `si/si2.xlsx`, all seven proteome sheets | no | derived, both asserted to reproduce from the stored values, already recorded | 0 |
| **`Proteome change` per mutation per strain** | the column `Proteome change` (Up / Down / NS / ND) under "Details of the mutations in the A10_F63_I1 and A12_F53_I1 strains and their corresponding proteomics shifts on glucose minimal medium with respect to IPL400" | `si/si1.docx` Table S5 | no | new, a derived summary of the proteome sheets; the note cites Table S5 only for the PP_3415 double variant | 18 rows |
| **regulator-enrichment statistics, 46 regulators over 7 columns** | `p value`, `precision`, `recall`, `f1score`, `TP`, `target_set_size`, `gene_set_size`, under "Summary of regulators enriched in the differentially expressed proteins in the evolved production strain A10_F53_I1_pIY670 vs IPL400" | `si/si1.docx` Table S6 | no | new, not a phenotype: keyed to a regulator over a derived differential set. Absent from the note AND the loader | 0 |
| ALE recurrence per commonly mutated region | `# of ALE`, `ALE condition`, under "Commonly Mutated Isoprenol Tolerance Specific Regions" | `si/si1.docx` Table S4 | no | derived summary of the mutation matrix | 11 rows |
| **E. coli ALE recurrence per mutated gene** | `Gene`, `Product`, `# of ALE`, `Mutations in P. putida`, footnote a "Mutated genes in a previous study that tolerized E. coli K-12 MG1655 against isoprenol (Babel and Krömer, 2020)." | `si/si1.docx` Table S7 | no | new, not a phenotype here: another paper's mutation-recurrence count. See the comparison section | 0 |
| the 25- and 17-gene locus annotations | `Locus tag`, `Gene name`, `Product` under "25 Genes between PP_2402 and calA" and "17 Genes between PP_3595 and PP_3610" | `si/si2.xlsx` `Table S4_GeneList` | no | not a phenotype, annotation, already recorded | 0 |
| strain, plasmid and oligo tables | `Name`, `Genotype with notes`, `Reference` (Table S1, 57 rows with JBEI ids); Table S2 oligos | `si/si1.docx` | partly, quoted for genotypes | not a phenotype, design | 0 |
| all remaining phenotyping | Fig. 2A, Fig. 3A, the Fig. 3B / 5A / 5D titers, the Fig. 5B / 5E residual glucose, the Fig. 4D / S3B degradation, Figs. S1C to S1F | `si/si1.docx` and `paper.md` | no | already recorded, 14 items, rule `released_only_as_a_figure`. Two refinements: the note's "Figs. S1C-F (growth curves)" mislabels S1F, whose caption is "(F) Percentage of remaining isoprenol after incubating cells for 48 h in the presence of 150 mg/L isoprenol in M9 minimal medium", a degradation readout; and Fig. S1A plots this paper's own per-gene DESeq2 log2 fold change ("A gene expression fold change was similarly calculated by using DESeq2 ... using the glucose condition as a reference"), whose processed table is not released, only raw reads at GEO GSE281392, already in `si_expected` | 0 |

Two small cross-source disagreements measured, in the same family as the 158-against-159
one the note already reports: Table S4 has 11 rows while the Results name 12 loci (the table
collapses `erdR (PP_1635) or mxtR (PP_1695)` into one row), and Table S5 lists 8 mutations
for A10_F63_I1 and 10 for A12_F53_I1 while the Results say "the total nine mutations" for
each.

**The comparison question.** Four comparisons, one of them a recorded duplication risk.

- **Babel and Krömer 2020**, E. coli MG1655 isoprenol ALE, points at a loadable dataset
  that is not in the mirror. Table S7's footnote is quoted above, and the Results restate
  that paper's numbers: "the evolved E.coli strains exhibited a 47% increase in growth
  rates over the WT in the presence of 4.3 g/L isoprenol and could grow in the presence of
  a maximum of 6.8 g/L isoprenol." Only gene names and ALE counts are re-released here, so
  there is no duplication risk; the comparison is the pointer. `babel*` is absent from the
  mirror.
- **Thompson 2020 RB-TnSeq**: "We identified differential gene expression targets that were
  in good agreement with previous RB-TnSeq fitness profiling data (Thompson et al., 2020)
  (Supplementary Fig. 1A)", with Fig. S1B "Rb-TnSeq fitness profiles (on various alcohols)
  of the genes targeted for deletion". Figure only here, and as the Carruthers section
  records, Thompson 2020 is already subsumed by Borchert 2024. Nothing new.
- **Lim 2022 PRECISE-321, measured, no duplication risk.** The only use is "Its transcript
  levels were also significantly downregulated log2 fold < -2 when P. putida KT2440 was
  grown in the presence of isopentanol (Lim et al., 2022)." Searching the compendium we
  already store (`limMachinelearningPseudomonasPutida2022/si/si2.xlsx`, sheet
  `1-Sample_list`): 0 rows match "isoprenol", 2 match "isopentanol"
  (`SBRG_Stress3__isopentanol__1` / `SRX4083219` and `SBRG_Stress3__isopentanol__2` /
  `SRX4083220`), and 0 match "GSE281392". So Lim 2025's new transcriptome deposit (GEO
  GSE281392) is NOT in the compendium, the statement refers to samples our
  `PutidaPrecise321Lim2022Dataset` already stores, and nothing duplicates in either
  direction.
- **One genuine duplication risk, already recorded:** ALEdb publishes the same mutation
  analysis as `si2.xlsx` ("Mutation analysis results were uploaded to ALEdb v1.0 ...
  accessible at" `http://aledb.org/` "with 'Pputida_isoprenol_TALE' as the project name"). The
  raw mirror's `si_expected` already names it, so a future ALEdb loader must not re-store
  these calls.

### Menasalvas 2025

Mirror `$DATA_ROOT/torchcell-library/menasalvasBiosensordrivenStrainEngineering2025/`. The
only SI artifact on disk is `si/si1.pdf` with its MinerU OCR `si/si1.md` (150,870 B, 468
lines: Notes S1 to S3, Supplementary Methods, Figs. S1 to S21, Tables S1 to S7, and legends
for data S1 to S5), read in full. There is no `.xlsx`, `.csv`, `.txt` or `.docx` in the
mirror.

**Five released supplementary data files are off disk, and that is documented in the RIGHT
place.** `paper.md`: "Data S1, S2, S4, and S5 have been deposited at the Dryad Digital
Repository (doi.org/10.5061/dryad.sbcc2frjq). Data S3 is available at Dryad/Zenodo
(doi.org/10.5281/zenodo.17155686)." The LIBRARY mirror's `manifest.json` carries
`si_expected: []`, while the RAW mirror's carries 7 entries naming the Dryad zip
(`Data_Dryad_Supplementary_Data_Updated_2025-9-18_2.zip`, 78,976,184 B), the Zenodo
AlphaFold output, PRIDE PXD061547 and BioProject PRJNA1226229, each with a reason.
Measured across four keys, the library mirror NEVER carries `si_expected` and the raw mirror
always does (Carruthers 0 against 6, de Siqueira 0 against 6, Lim 2025 0 against 4,
Borchert 0 against 1), so the empty library-mirror value is the convention rather than a
provenance gap. This is the same Anubis proof-of-work situation as #739 but a DIFFERENT
Dryad deposit (`sbcc2frjq` against Carruthers's `gtht76hzh`), and no open issue covers it.

Because those five files are off disk, every data-S column name below is NOT CHECKED and
the verbatim legend is quoted instead.

| released quantity | exact column or verbatim caption quote | SI file | stored? | classification | records |
|---|---|---|---|---|---|
| per-gene enriched-guide call, round 1 | `Gene`, `Function`, `FunctionalCategory`; "Lower pedF-RBS-pyrF Isoprenol Threshold: Enriched gRNAs." | `si/si1.md` Table S1 | yes, 28 of the 58 records | stored | 28 |
| per-gene enriched-guide call, round 2 | the same three columns; "Second Round pedF-RBS-pyrF with Higher Isoprenol Threshold: Enriched gRNAs." | `si/si1.md` Table S2 | yes, 30 records | stored | 30 |
| **relative metabolite concentrations per strain per phase** | "Sheet 1. Metabolite concentrations from selected isoprenol producer strains."; Fig. 7E "the log2 fold abundance change of the metabolites in TEAM-3174 and TEAM-3185 compared to the control strain, TEAM-2595" | data S1-1, Dryad, off disk | no | **loadable now** once retrieved: `MetabolitePhenotype` plus `BacterialMetaboliteExperiment`. Plain product names are permitted keys ("or a plain product name for heterologous products") and the validator imposes no key format, so no schema change. The FILE is already recorded in the note and `si_expected`; naming the class and counting the records is new | about 4 records plus 2 references, estimated as 3 strains x 2 phases from "We analyzed parallel samples from the growth phase [optical density at 600 nm (OD600) ~1] and the 24-hour time point in the production phase", with TEAM-2595 the reference in each phase. Metabolite count per record NOT CHECKED |
| **five separate per-protein abundance datasets** | "Sheet 1. Proteomics Analysis of ∆yiaY ∆yiaZ Complementation by Varied Plasmid Constructs"; "Sheet 2. Analysis of PJ23119-yiaY,yiaZ Constitutive Expression"; "Sheet 3. Proteomics Culture Format Evaluation of Isoprenol Producer Strains"; "Sheet 4. Growth/Production phase Samples of High Isoprenol Producer Strains TEAM-3174 & 3185 Compared to TEAM-2595"; "Sheet 5. Plasmid-born augmentation of Isoprenol pathway overexpression in genomically integrated producer strains" | data S2, Dryad, off disk | no | the note says only "five proteomics sheets"; the enumeration is new. `ProteinAbundancePhenotype` plus `BacterialProteinAbundanceExperiment` can hold them, so **loadable now** for the designed-strain arms (Sheets 1, 2, 3, 5 and the TEAM-2595 arm of Sheet 4). The TEAM-3174 and TEAM-3185 arms are **blocked, #731**, on Table S3's own footnote "# Spontaneous polymorphisms characterized by WGS are described in Supplementary Data 1-5" | Sheet 4 about 4 records plus 2 references at about 2,500 protein keys each ("the ~2500 proteins detected by LC-MS/MS"); Sheets 1, 2, 3, 5 arm counts NOT CHECKED |
| **whole-genome polymorphisms of the producer clones** | "Sheet 5. Illumina Genome Resequencing and Polymorphism Analysis of Selected Isoprenol Clones." | data S1-5, off disk | no | **blocked, #731.** The file is listed in the note but is NOT tied to #731 there, and the consequence is new: the TEAM-3174 and TEAM-3185 genotypes carry uncalled spontaneous variants, so those producer strains are no more representable than Lim's evolved clones | 3 clones, from the three BioSamples in `paper.md`: SAMN46924002 (TEAM-2595), SAMN46924003 (TEAM-3175), SAMN46924004 (TEAM-3185). Variant count NOT CHECKED |
| per-guide library read counts | "Sheet 3. Distribution of gRNA pooled library gRNA sequences."; fig. S12B "Histogram of gRNAs binned by number of reads, ranging from a low of 1 reads up to a maximum of 4,336 reads" | data S1-3, off disk | no | not a phenotype, library-composition QC in E. coli with no strain x condition; already recorded | 0 |
| designed guide sequences and the missing-guide list | "Sheet 2. Pooled gRNA targeting sequences synthesized for the library."; "Sheet 4. Expected gRNA sequences missing from the library." | data S1-2 and S1-4, off disk | no | not a phenotype, design and QC, already recorded | 0 |
| GO-term enrichment of the differential proteome | "Sheet 6. ShinyGO Enrichment Analysis of High Producer Isoprenol Strains." | data S1-6, off disk | no | not a phenotype, a derived enrichment, already listed | 0 |
| AlphaFold3 structures with per-residue confidence | "AlphaFold3 Multimer analysis of YiaZ+YiaY. AlphaFold3 Ligand Analysis of YiaZ+NAD. AlphaFold3 Multimer analysis of YiaZ+PP_2664." | data S3, Zenodo, off disk | no | not a phenotype, a structure prediction, already recorded | 0 |
| per-genome homolog score ratios | "fast.genomics analysis of yiaY and yiaZ homolog co-occurrence in microbial genomes."; fig. S4 "The X and Y axis are the score ratio for homologs of either YiaY and YiaZ respectively, which is a value that compares the bit score of a protein's homolog to the maximum score" | data S4, off disk | no | not a phenotype, comparative-genomics annotation, already listed | 0 |
| raw flow cytometry and ONT amplicon reads | "Flow cytometry raw data for mCherry timecourse analysis. Representative Plasmidsaurus ONT gRNA amplicon sequencing reads." | data S5, off disk | no | not a phenotype, raw instrument data. The derived per-strain quantities are figure only: fig. S7, "The % induced values were calculated by determining how many events (out of 30,000 total) were present above the fluorescence threshold" | 0 |
| genotype edits of the optimization campaign | `Mutation#`, `Encoded Function`, `Reference`, under "Genes Identified in Isoprenol Production Optimization." | `si/si1.md` Table S3 | partly, quoted | new, not a phenotype (design and annotation). Not named in the note; it carries the footnote pointing at data S1-5 | 10 rows |
| strain and plasmid registries | `AccessionNo.`, `TEAM ID#`, `Genotype`, `Reference` (S4); `JBEI ID`, `TEAM #`, `Strain`, `Plasmid #`, `Description`, `Reference` (S5) | `si/si1.md` Tables S4 and S5 | partly, quoted | not a phenotype, design, already recorded | 0 |
| **recombineering guide and oligo sequences** | `Gene target`, `PlasmidNo.`, `gRNA TargetingSequence 5'-3'`, `Recombineering Oligo 5'-3'` | `si/si1.md` Table S6 | no | new, not a phenotype (design). Not named in the note or loader | 0 |
| **techno-economic model inputs** | `Parameter`, `Unit`, `Current state of technology scenario` (3 sub-columns), `Optimal futurecase`, including `Glucose-to-isoprenolyieldδ` `% by mass` 0.040 / 0.048 / 0.056 / 0.31 | `si/si1.md` Table S7 | no | new, not a phenotype (model inputs; the yield row is footnoted "δ Based on the experimental data described in Figure 6 and Supplementary Figure 18 in this study", so it is derived from figure-only titers) | 0 |
| **isoprenol TOLERANCE growth curves of the high producers** | fig. S15 "(B) Isoprenol tolerance growth curves in M9 minimal media supplemented with the indicated concentration of isoprenol. Samples in triplicate were read every 15 minutes with a microtiter dish plate reader at OD600" | `si/si1.md` fig. S15 | no | **new, figure only.** This is an environment-response phenotype in the SAME family as the Lim 2025 tolerance records, and the note's figure-only list is titer-only (it names figs. S13, S14, S19, S20 and no others) | 0 |
| **specific isoprenol titer per OD600** | fig. S16 "Specific isoprenol titer. Refer to Figure 6A, 6C. Strains of the indicated genotypes were normalized to OD600 for samples harvested in growth phase and the 24-hour production phase timepoint" | `si/si1.md` fig. S16 | no | new, figure only, a titer panel the note does not list | 0 |
| **titers with pIY670 against an empty vector at 24 h** | fig. S18 "(A) Isoprenol production was assayed in TEAM-2595, TEAM-3174 and TEAM-3185 harboring an additional copy of the isoprenol pathway with the plasmid-based construct pIY670 compared to an empty vector control" | `si/si1.md` fig. S18 | no | new, figure only, a second unlisted titer panel | 0 |
| per-strain biosensor fold induction, percent induced, basal fluorescence, ligand panel | figs. S1, S2, S3A and S3C, S7, S9, S10, for example fig. S9 "Diol fold cherry activation of the biosensor is shown with a 11.6 mM concentration of each analyte in WT, ∆PP_2664, and ∆yiaY/∆PP_2682 ... and ∆yiaZ/∆PP_2683" | `si/si1.md` | no | new, figure only. fig. S1 is a SECOND analyte entirely (p-coumaric acid, 0 to 1 g/L), a second biosensor dose-response the note never mentions | 0 |
| titer replicate count | "All production runs were performed in quadruplicate, with two biological replicates prepared for each strain when multiple clones were tested." | `paper.md` Methods | no | new sourcing fact: the note's `n_samples = 4` is sourced from the SELECTION sentence, and this is the TITER readout's n, which a future titer loader needs beside the already-recorded `TITER_UNCERTAINTY` of `sample_sd` | not applicable |

**No per-strain titer anywhere, independently reconfirmed.** Every `data S` and `Table S`
reference in `paper.md` and `si1.md` was followed, all seven table captions and all 21
figure legends read. Every titer is a plotted panel, and the five Dryad and Zenodo legends
name no titer table. The note's existing finding holds, with the three additional
figure-only panels above (fig. S15 is tolerance, figs. S16 and S18A are titers).

Three internal cross-reference errors beyond the one the note records ("tables S2 and S3"
against S1 and S2): the Methods name the proteomics strains "TEAM-2595, TEAM-3175, and
TEAM-3184" while the Results and the note name TEAM-3174 and TEAM-3185; the Methods cite
"(data S4)" for "the known gRNA targeting sequences" while the legend assigns data S4 to the
fast.genomics analysis; and the Methods cite "(data S1-4 and S1-5)" for library-validation
sequencing while the legend assigns S1-5 to the WGS polymorphism analysis.

**The comparison question.** One comparison points at loadable data and nothing re-releases
a sibling paper's measurements.

- Fig. 1B is an "RB-TnSeq fitness heatmap of the selected genes (x axis) from the cofitness
  analysis on the indicated carbon sources (y axis)", with the Methods: "This database
  contains quantitative barcoded fitness data from a 200,000-member P. putida barcoded
  transposon mutant library collected across more than 200 experimental conditions.
  Published datasets are available for analysis at the fitness browser, accessible at
  fit.genomics.lbl.gov." Re-displayed, not re-released as numbers. It points at the Fitness
  Browser, which is already served through Borchert 2024.
- **No Carruthers 2025, Yunus 2026, de Siqueira 2025 or Kang 2026 measurement is
  re-reported.** The full reference list was read: the only Carruthers entry is the 2023
  olefinic-ester paper, and Yunus, de Siqueira and Kang 2026 are absent. Lim 2025 and
  Banerjee 2024 are cited with no number of theirs restated.
- **Shared GENOTYPE, not a shared measurement, and worth flagging for cross-dataset key
  uniqueness.** Menasalvas Table S4 lists `TEAM-1510 | P. putida KT2440 ΔPP_ 2675`; Lim
  2025 Table S1 lists `KT2440 ΔPP_2675 Δ14-PP_2676 (JBEI-147164)` sourced to Thompson 2020;
  and the Lim note records that de Siqueira's `PT` strain is the same physical Thompson 2020
  lesion. Three loaders will carry the same genotype with three different phenotypes, so no
  measurement is stored twice, but the genotype key collides across datasets.
- **Shared pathway, no shared titer.** Both Menasalvas and Lim 2025 put the same five-gene
  pIY670 IPP-bypass cassette into KT2440 (Lim episomally, Menasalvas chromosomally
  integrated) and both cite Banerjee 2024 as its source. Since neither releases a per-strain
  titer, nothing is duplicated.

### Yunus 2026

Mirror `$DATA_ROOT/torchcell-library/yunusPredictiveCRISPRmediatedGene2026/`, `si_expected:
[]` in the library mirror. One SI file, `si/si1.docx` (3,517,548 B), read in full: 14 tables
(S1 to S12 plus two protocol tables in Note 2) and 11 figure captions.

| released quantity | exact column or verbatim caption quote | SI file | stored? | classification | records |
|---|---|---|---|---|---|
| relative protein expression per screened strain | `Strain name \| CRISPRi target gene \| Relative expression level` | `si/si1.docx` Table S3, 125 data rows | yes, 102 of 125 | stored | 102 |
| relative expression per array construct per replicate | `\| Replicate \| Relative expression level of PP_4188` (S9 `... of PP_0812`, S10 `... PP_4160`, S11 `... PP_0168`, S12 `... PP_0528`) | `si/si1.docx` S8 (60), S9 (30), S10 (42), S11 (15), S12 (6) = 153 cells over 51 (construct, protein) pairs and 25 distinct constructs | yes | stored | 25 |
| **per-protein fold change against control, PP_4188 strain, downregulated** | `Protein.Group \| Protein.Names \| Protein \| Protein.Description \| Fold Change \| Log2(Fold Change) \| P-Value (Equal Variance) \| (-Log10(P-Value)) \| Rank` | `si/si1.docx` Table S4, 145 rows | no | **split.** The `Fold Change` column is the same quantity type the loader ALREADY stores (a ratio to the control strain), so here it is **loadable now** in `ProteinAbundancePhenotype` with a `measurement_type` naming the basis and `n_replicates = 3` (Fig. S8 caption, "Error bars represent standard deviation from three biological replicates"). `P-Value (Equal Variance)` and `Rank` are **blocked on gap R**, because `ProteinAbundancePhenotype` carries `protein_abundance_se` and no p-value field | 1 record carrying 338 protein keys (145 + 193); PP_4188 itself is absent from both tables, so there is no clash with its Table S3 row |
| same, upregulated | identical header | `si/si1.docx` Table S5, 193 rows | no | same | included above |
| per-strain isoprenol titer | "Supplementary Figure S6. Isoprenol titers obtained from intuition-based target genes."; the text gives only 1469 mg/L (PP_4188) and 958 mg/L (PP_0168) | bar charts Figs. 4B to 4J, S6 and S9; the numeric table sits only behind the Supplementary Note 1 Benchling link | no | already recorded as `NOT_LOADED[0]` and `[1]`; no control titer, so no `ProductTiterExperimentReference` | about 108 if the Benchling table is ever retrieved |
| OD600 at 48 h per strain | "Supplementary Figure S7. OD600 of strains shown in Fig. 4. OD600 was sampled at 48 hr." | figure only | no | already recorded | about 108 |
| PP_4188 relative expression, control against knockdown | "Supplementary Figure S8. Relative expression level of PP_4188 gene in the control and PP_4188 strains." | figure only | no | a duplicate of the Table S3 and S8 values, not a new quantity | 0 |
| **positional relative expression for four more genes** | "Supplementary Figure S5. Relative expression levels of PP_0813, PP_5186, PP_4188, and PP_4189 genes when the sgRNA targeting that specific gene is placed at different position in the array." | figure only, NO companion table | no | **new, recorded nowhere.** Tables S8 to S12 are the tables for Fig. 3J to 3N only; this second positional experiment over four genes has no released table. It would be loadable in the same class if released | 4 genes at an unstated construct count |
| TCA metabolite concentrations at 24, 48 and 72 h | "Concentrations of pyruvate, fumarate, cis-aconitate, and alpha-ketoglutarate in uM from PP_4188 strain extracted at 24 h, 48 h, and 72 h" (Fig. 5A) | figure only | no | already recorded in `NOT_LOADED` | 12 |
| terminal OD600, RFP fluorescence, PP_1607 growth curve | Figs. 3C, 3D, 3F | figures only | no | already recorded | small |
| sgRNA oligo sequences | `Oligo name \| Sequence (5' to 3')` | `si/si1.docx` Table S7, 415 rows = 204 forward/reverse pairs plus 7 primers | partly, the 204 spacers ride on the perturbation | not a phenotype, already recorded | 204 |
| plasmid list | `Plasmid No. \| Plasmid description \| JBEI Accession No.` | `si/si1.docx` Table S6, 475 rows | no | not a phenotype, already recorded | 475 |
| linker oligos and reagent volumes | `Oligo Name \| Sequence (5' to 3') \| Linker Name`; `Volume to add \| stock concentration \| final concentration` | `si/si1.docx` Note 2, two tables | no | new, not itemized in the note's not-loaded list (which names only Table S6 and the three primers); method metadata, not a phenotype | 0 |
| target annotation lists | `No. \| Target Gene \| Enzyme \| Enzyme description \| KEGG Orthology \| Metabolic Pathway \| iPath3` | `si/si1.docx` S1 (52 rows), S2 (57 rows) | kept in `preprocess/target_lists.csv` | not a phenotype, already recorded. Note that S2 carries NO FluxRETAP score column although the paper says it "ranked them based on the inverse value of the overlap area" | 109 |
| Pearson correlation of protein expression against titer | Note 1, "Pearson correlation analysis between gene expression and isoprenol production from all strains can be found here:" then `https://benchling.com/s/etr-mta4DgBAatF0hFYzy275...` | `si/si1.docx` Note 1 | no | a derived summary whose INPUT is the unreleased titers, already recorded | 0 |
| raw DIA spectra | PRIDE PXD062697 | external | no | already recorded | 0 |

**The comparison question, and this is where the isoprenol group was most at risk.**
Carruthers 2025 and Yunus 2026 share an author, a chassis designation (`IY1452b` against
`IY1452`), a product and substantially one CRISPRi target library, and Carruthers releases
as NUMBERS the two quantities Yunus releases only as bar charts. Measured, and then
re-measured here with a column correction:

- **All 120 Carruthers guide-target loci fall inside the Yunus screened set.** 118 of 120
  join Yunus Table S3 on the exact `PP_` tag; the other two (`PP_1607`, `PP_4194`) appear in
  Table S3 only under variant labels (`PP_1607_NT2`, `PP_1607_NT4`, `PP_4194_NT2`,
  `PP_4194_NT3`) and in Table S1. Seven Yunus S3 loci are absent from Carruthers
  (`PP_3365`, `PP_4090`, `PP_4129` plus the four variant labels).
- **The two papers' FluxRETAP-against-heuristic selection labels agree perfectly:**
  Carruthers's `Source (Flux-RETAP or Heuristic) 1` against Yunus table membership gives 54
  `FR` in Table S2 and 50 `H` in Table S1 with zero contradictions, the other 16 being `H`
  in Carruthers and absent from Yunus Table S1. Strong evidence of one shared
  target-selection campaign.
- **The measurements are nonetheless independent, which is the answer.** Carruthers's
  `Mean isoprenol titer (mg/L)` over 120 targets runs min 0.01, median 162.1, max 393.7,
  while Yunus's stated best is 1469 mg/L for `PP_4188`, for which Carruthers gives 209.97.
  On the protein ratio, **using the Carruthers column that measurably holds the POI ratio**
  (the one headed `dCas9/control`, see the Carruthers section's finding 3) against Yunus
  Table S3's `Relative expression level` over the 96 loci where both are numeric: zero
  matches within Supplementary Data 1's own rounding, median absolute difference 0.7384,
  maximum 2.9866. Against the mislabeled column the median difference is 0.1315 over 90
  loci and still zero matches, so the conclusion holds either way and is stronger with the
  correct column.

So this is NOT the Rousset 2018 / Cui 2018 shape of #760 (r = 1.0000 over 54,326 spacers).
Both papers are legitimately loadable and there is **no duplication risk today**, including
none from retrieving the Yunus Benchling titer table later.

Two consequences carry forward. First, **Carruthers's POI and dCas9 ratio columns are the
same quantity Yunus's `CrispriKnockdownYunus2026Dataset` already stores**, over 96 numeric
loci, as an independent measurement, and they are neither stored nor listed as not-stored
(the Carruthers loader consumes Supplementary Data 1 "only as the cross-source oracle" for
the titer). That makes gap R's Carruthers rows look more tractable than they first appear:
Yunus proves a ratio-to-control CAN be stored under `ProteinAbundancePhenotype` with a
`measurement_type` naming the basis, so the Carruthers ratio rows are arguably **loadable
now by precedent** rather than blocked, and the decision is whether
`ProteinAbundancePhenotype`'s docstring prohibition ("NOT a ratio") binds. That contradiction
between the docstring and an existing loader's practice is itself the thing to settle.
Second, **Yunus Table S7 is the only released spacer set for this library**; the Carruthers
loader records verbatim that "the CRISPRi sgRNA SPACER sequences were never released;
Supplementary Data 2's `PP_*_gRNA` entries are the Cpf1 knockout guides." Yunus releases 204
sequence-verified pairs over substantially the same targets. That is a cross-paper genotype
enrichment route, not a phenotype.

### Caglar 2017

Mirror `$DATA_ROOT/torchcell-library/caglarColiMolecularPhenotype2017/`, `si_expected: []`
in the library mirror. `si/si1.md` read in full (223 lines, all 14 table captions and 35
figure captions), and every one of `si2.csv` to `si15.csv` opened. The PMC object naming
maps `-s<N+1>` to Table S<N>, so `si2.csv` is Table S1 and `si15.csv` is Table S14. The two
classes stored today are `BacterialRNASeqExpressionExperiment` (152 records, Table S2) and
`BacterialProteinAbundanceExperiment` (105 records, Table S3).

**The structural finding: 10 of the 14 released tables are addressed nowhere.** The loader
pins `SI_TABLES` for S1 to S4 only, the raw mirror holds exactly `srep45303-s2.csv` to
`-s5.csv`, the loader has no `NOT_LOADED` tuple, and grepping the note and loader for
Tables S5 to S14 returns one incidental hit. So Tables S5 to S14 are not documented
decisions, they are an unexamined gap.

| released quantity | exact column or verbatim caption quote | SI file | stored? | classification | records |
|---|---|---|---|---|---|
| normalized mRNA counts | "Supplementary Table S2: Normalized mRNA counts. Includes data for 4196 distinct proteins each for 152 samples."; columns `MURI_016` onward, row keys `ECB_00001` onward | `si/si3.csv`, 4,196 x 152 | yes | stored | 152 |
| normalized protein counts | "Supplementary Table S3: Normalized protein counts. Includes data for 4196 distinct proteins each for 105 samples."; row keys `YP_003043230.1` onward | `si/si4.csv`, 4,196 x 105 | yes | stored | 105 |
| **doubling time per replicate growth curve** | `name,replicate,doubling.time.minutes,doubling.time.minutes.95m,doubling.time.minutes.95p,r.squared`; "Supplementary Table S5: Doubling time measurements in exponential phase. Includes the mean, ±95% confidence interval, and r2 from the linear fit to OD600 values." | `si/si6.csv`, 55 data rows over 19 conditions | **no** | **loadable now**: `BacterialEnvironmentResponseExperiment` / `EnvironmentResponsePhenotype` with `measurement_type = MeasurementType.growth_rate`, `assay_type = AssayType.liquid_od_growth`, `environment_response_uncertainty_type = UncertaintyType.ci95`, `sample_unit = biological_replicate`. The per-condition replicate counts in the file are 3 everywhere except Gluconate 2 and Lactate 2, matching the loader's own already-sourced `DOUBLING_TIME_REPLICATES` quote | 55 per-replicate records, or 19 condition means |
| doubling time per sample | `doublingTimeMinutes,doublingTimeMinutes.95m,doublingTimeMinutes_95p,rSquared` | `si/si2.csv`, 165 of 171 rows non-NA, **19 distinct values** | no | the same class; the sample-level column is the condition mean repeated, so storing it as well as Table S5 would hold the same 19 measurements twice | 19 |
| **total cell count and cells per tube** | `cellTotal`, `cellsPerTube`, for example 1.50E+10 and 3.00E+09 | `si/si2.csv`, **131 of 171 rows non-NA** | no | **blocked, and described nowhere.** The Table S1 caption lists its columns and does NOT name these two, and a case-insensitive grep over `paper.md` for `cell count`, `cells per`, `CFU`, `optical density` and `OD600` returns nothing. No phenotype class holds an absolute cell count and `MeasurementType` has no member for one. Not covered by #731, #739, #749, #753, #756 or #758 | 131 |
| technical-replicate counts | `RNA_Data_Freq`, `Protein_Data_Freq`; "number of RNA samples (technical replicates), number of protein samples (technical replicates)" | `si/si2.csv` | a sourced value only | not a phenotype, already recorded | 0 |
| mean flux ratio with dispersion | `"Phase","Salt","Conc","Branch","MeanFluxRatio","SDEFluxRatio","Phase2"`; "Supplementary Table S4: Mean flux ratios for 13 branches each" | `si/si5.csv`, 260 rows = 13 branches x 20 (phase, salt, concentration), 130 EXP and 130 STA, 71 rows with `SDEFluxRatio` = 0 | no | **already recorded** in the loader docstring, "it holds flux RATIOS, which neither `MetabolitePhenotype` (pool sizes) nor `FluxPhenotype` (signed net flux) can store", and as the note's gap 2. No open issue carries it | 260 |
| cophenetic clustering z-scores, mRNA | `"Variable","Overall_Z_score","Condition","Z_score","num_elements"` | `si/si7.csv` (Table S6), 33 rows | no | not a phenotype, a statistic over the whole sample set rather than a strain or condition record | 33 |
| the same, protein | identical header | `si/si8.csv` (Table S7), 29 rows | no | not a phenotype | 29 |
| **DESeq2 per-gene differential expression** | `"","X","id","baseMean","log2FoldChange","lfcSE","stat","pvalue","padj","gene_name","signChange","pick_data","growthPhase.x","test_for","contrast","base","fullFileName","dataType","carbonSource","Mg","Na","growthPhase.y","investigatedEffect","testVSbase"`, with `testVSbase` in {gluconateVSglucose, glycerolVSglucose, lactateVSglucose, lowMgVSbaseMg, highMgVSbaseMg, highNaVSbaseNa} | `si/si9.csv` (Table S8), **201,408 rows** = 4,196 x 2 data types x 6 contrasts x 2 phases x 2 control models; 8,392 distinct ids, 48 contrasts | no | **blocked, gap R again**: no phenotype class holds a per-gene `log2FoldChange` with `lfcSE`, `pvalue` and `padj` against a named contrast. The largest single unstored per-record quantity in the whole audit. Attributable to no existing issue | 201,408, or 100,704 if only the batch-only control model is kept |
| filtered DE gene list | `"","genes","fullFileName","thresholds_for_input_genes","dataType","carbonSource","Mg","Na","growthPhase","investigatedEffect","testVSbase"` | `si/si10.csv` (Table S9), 19,197 rows | no | not a phenotype, a thresholded subset of Table S8 (`P0.05Fold2`) carrying no value of its own | 19,197 |
| DAVID enrichment with joined per-gene statistics | `"","X","Category","KEGG_Path","Count","PValue","List.Total","Pop.Hits","Pop.Total","Fold.Enrichment","Bonferroni","Benjamini","FDR_KEGG_Path",...,"gene_number","gene_name_ez","gene_name","padj_gene","log2","signChange","score_gene","rank",...` | `si/si11.csv` (Table S10), 26,454 rows over 42 columns | no | not a phenotype at the pathway level, and its `log2` and `padj_gene` columns re-release Table S8's per-gene values restricted to enriched pathways, so storing it beside S8 would duplicate | 26,454 |
| extra DE proteins when controlling for doubling time | `"","genes","dataType","growthPhase","test_for","exist_in"`, values `"controlled for Growth"` | `si/si12.csv` (Table S11), 4,084 rows | no | not a phenotype, a membership list with no value | 4,084 |
| enrichment of those | `,Category,Term,Count,X.,PValue,Genes,List.Total,Pop.Hits,Pop.Total,Fold.Enrichment,Bonferroni,Benjamini,FDR,phase,name,test` | `si/si13.csv` (Table S12), 82 rows | no | not a phenotype | 82 |
| flux ratio against ion concentration regression | `,Salt,Phase,Branch,term,estimate,std.error,statistic,p.value,p.adjusted` (the file uses CR-only line terminators) | `si/si14.csv` (Table S13), 26 rows = 13 branches x 2 salts | no | a derived summary over Table S4, not a measurement | 26 |
| flux ratio against doubling time regression | `"","Branch","term","estimate","std.error","statistic","p.value","p.adjusted"` with `term` = `meanDT` | `si/si15.csv` (Table S14), 13 rows | no | a derived summary | 13 |

**The comparison question: 27 of 152 mRNA and 27 of 105 protein records are a re-release of
Houser 2015.** Caglar says so. Results, verbatim: "We also measured central metabolic fluxes
for a subset of conditions using glucose as carbon source. **Results from one of these
conditions, long-term glucose starvation, have been presented previously10.**" Reference 10
is "Houser, J. R. et al. Controlled Measurement and Comparative Analysis of Cellular
Components in E. coli Reveals Broad Regulatory Changes in Response to Glucose Starvation.
PLOS Comput Biol 11, e1004400 (2015)". The data-availability section splits the deposits to
match, verbatim: "accession GSE67402 for the glucose time-course previously published10,
accession GSE94117 for all other experiments ... accession PXD002140 for the glucose
time-course previously published10, accession PXD005721 for all other experiments".

Measured and re-verified for this note, by joining `si2.csv`'s `sampleNum` to the
`MURI_NNN` column names of `si3.csv` and `si4.csv`: `experiment == 'glucose_time_course'`
covers 27 samples (MURI 16 to 33 and 97 to 105), and **all 27 appear as columns in BOTH the
152-sample mRNA table and the 105-sample protein table**, i.e. 17.8 % of the stored RNA-seq
records and 25.7 % of the stored protein records. A separate 9 samples labeled
`glucose_time_course (repeated between MURI 97- 105)` (MURI 7 to 15) appear in neither omics
table.

Houser 2015 is not in the mirror and has no loader, so the duplication is latent rather
than realized. But a Houser 2015 loader would re-store those 27 plus 27 samples, and GEO
GSE67402 with PRIDE PXD002140 are the same material under the earlier accessions. This is
the exact shape of #760 and nothing in the Caglar loader or note records it; the note
mentions Houser 2015 only as the unmirrored source of a deferred log-base method detail.

Three further comparison points. The log-base deferral chains to the same paper
("Measurements of RNA and protein abundances were carried out as previously described10" and
"Our experimental approach was identical to the one used in our prior work on glucose
starvation10"), and the loader back-solved the base rather than following the deferral,
which is documented. Caglar benchmarks against four external datasets in prose only, naming
no table of theirs: Schmidt (22 conditions, more than 2,300 proteins), Soufi (10
conditions), and two Lewis sets; `ecoli/schmidt2016.py` already exists and Caglar
re-releases none of its values, so there is no overlap to resolve. And Caglar's `cellTotal`
column plays the same role Carruthers's `RbTn-Seq Data?` does: a column flagging a
measurement that lives in another release.

### Loadable now, ranked largest first

Record counts are what each item would ADD, counted under each loader's own reference
convention. "Values" is given where one record is a profile rather than a scalar.

| rank | item | where | records added | notes |
|---|---|---|---|---|
| 1 | **doubling time per replicate growth curve** | Caglar 2017 `si/si6.csv` (Table S5) | **55** (or 19 condition means) | `BacterialEnvironmentResponseExperiment` with `measurement_type = growth_rate`, `assay_type = liquid_od_growth`, `uncertainty_type = ci95`. The file's replicate counts match the loader's already-sourced `DOUBLING_TIME_REPLICATES` quote. Nothing in loader or note addresses Tables S5 to S14 at all |
| 2 | **four sheets of unstored isoprenol titer cultures** | Carruthers 2025 `si/si9.xlsx` `Figure 6a`, `Figure 6d`, `Supplementary Figure 13d`, `Supplementary Figure 12bd` | **49** records over 190 cultures, plus 14 references | Zero of the 186 distinct values appear in the stored `Figure 4b`. `Supplementary Figure 13d` alone is 19 records pairing one to one with the 19 proteome records already stored. Needs `BacterialDeletionPerturbation` (already used by this loader for PP_0815) and `GeneAdditionPerturbation` (whose validator is relaxed to accept a `PP_` tag) |
| 3 | **two unstored per-protein abundance sheets** | Carruthers 2025 `si/si9.xlsx` `Supplementary Figure 12ac` and `Supplementary Figure 15` | **21** records, about 630 values | Same class and same Top3 percent-of-total basis the loader already uses. `Supplementary Figure 15` is the proteome arm of the control cultures already serving as the per-cycle `phenotype_reference` |
| 4 | **fed-batch isoprenol titer and residual sugars** | Kang 2026 `si/si1.docx` Table S9 | **21** (7 titer + 14 metabolite) | The 7 isoprenol titers are unconditional: `SI_TABLE9_COLUMNS` already asserts `Isoprenol, aqueous (mg/L)` and `FedBatchRow` declares no field for it. The 14 metabolite records carry a caveat: `MetabolitePhenotype` has no units field, so `g/L` would sit in free-text `measurement_type`, and `n_replicates` is required so n = 1 must be asserted from "a single run" |
| 5 | **two further released proteomics normalizations** | de Siqueira 2025 `si/si1.xlsx` columns 9 to 12 | **10** (2 normalizations x 5 samples), 15,310 values plus SEs | `measurement_type` is a free `str`, so each normalization is its own record set. Neither is recoverable from what we store: the percent column is a replicate-wise mean of per-replicate percentages (ratio 0.959 to 0.997 against percent-of-the-mean), and the log10 column is the mean of log10s (1,999 of 2,000 sampled rows disagree with log10 of the mean). The paper's own dimensionality reduction runs on the percent column |
| 6 | **metabolite concentrations and five proteomics sheets**, RETRIEVAL-GATED | Menasalvas 2025 data S1-1 and data S2, Dryad `10.5061/dryad.sbcc2frjq` | about 4 metabolite records plus the designed-strain proteome arms at about 2,500 keys each | `MetabolitePhenotype` and `ProteinAbundancePhenotype` both fit with no schema change, but the files are not on disk and `datadryad.org` serves the same Anubis proof-of-work challenge as #739 on a DIFFERENT deposit. The TEAM-3174 and TEAM-3185 arms are #731-blocked regardless. Arm counts for Sheets 1, 2, 3 and 5 are NOT CHECKED |
| 7 | **per-protein fold change against control, PP_4188 strain** | Yunus 2026 `si/si1.docx` Tables S4 and S5 | **1** record carrying 338 protein keys | The `Fold Change` column is the same quantity this loader already stores. Its `P-Value (Equal Variance)` and `Rank` columns are blocked on gap R |
| 8 | **knockdown ratios, CONDITIONAL on the ratio question** | Carruthers 2025 `si/si9.xlsx` `Figure 3c` (authoritative) and `si/si4.xlsx` | 92 records of a 2-protein profile | Conditional, because `ProteinAbundancePhenotype`'s docstring forbids a ratio while the Yunus loader already stores one. Settling that contradiction either unblocks this row or confirms gap R. `Figure 3c` must be the authority: Supplementary Data 1's two headers are measurably swapped |
| 9 | **three IPL400 production-proteome arms, CONDITIONAL** | Lim 2025 `si/si2.xlsx` `IPL400vsA10F63I1_pIY670_M9G_12h/24h/48h` | 0 to 2 | Needs a `MEDIA_LIBRARY` entry for the production medium AND a pairing decision, because each sheet pairs IPL400 against an evolved strain that is #731-blocked |

**Firm total: 157 records across items 1 to 5 and 7** (55 + 49 + 21 + 21 + 10 + 1), plus 14
new phenotype references, plus the retrieval-gated Menasalvas arms and the two conditional
items. Items 1 to 5 and 7 need no schema change and no new class.

### Duplication risks, ranked by how much would be stored twice

| rank | risk | evidence | status |
|---|---|---|---|
| 1 | **Borchert 2023's comparison sheets ARE the Borchert 2024 compendium's fitness** | Measured here: joining `borchertRBTnSeqIdentifiesGenetic2023/si/si1.xlsx` sheet `Glu_v_Glu_Van` on `old_locus_tag` against `fModule_Metadata.xlsx` `sysName`, all 4,732 overlapping loci agree on six per-replicate columns with max absolute difference 0.000500, which is the half-unit of the compendium's three-decimal rounding. 12 further comparison sheets were header-inspected and NOT numerically compared | Latent: Borchert 2023 has no loader. A loader over those sheets would re-store the fitness. Its t, p, q and adjusted-q columns, its 186,957 barcode-level rows and its growth curves ARE new |
| 2 | **Caglar 2017's glucose time course is Houser 2015 re-released** | The paper says so: "Results from one of these conditions, long-term glucose starvation, have been presented previously10", and splits the deposits, "accession GSE67402 for the glucose time-course previously published10 ... accession PXD002140 for the glucose time-course previously published10". Measured here: `experiment == 'glucose_time_course'` covers 27 samples (MURI 16 to 33 and 97 to 105) and all 27 appear as columns in BOTH the 152-sample mRNA table and the 105-sample protein table, i.e. 17.8 % and 25.7 % of the stored records | Latent: Houser 2015 is not mirrored and has no loader. Nothing in the Caglar loader or note records this |
| 3 | **Carruthers 2025, internal: two Source Data sheets export the same control triplicate** | `Supplementary Figure 13d`'s `Non-Target` rows (298.0681, 309.4453, 293.9713) are bit-identical to `Figure 6a`'s `PP_0815` / `Non-target` rows. Their `Target` triplicates are NOT the same cultures: 422.4448, 495.3173, 464.3335 against 464.3335, 477.8291, 450.4693, sharing exactly one value | Live for any revision that loads both sheets. Must deduplicate the reference group and decide what the disagreeing Target triplicates are |
| 4 | **Carruthers 2025, internal: the KO-comparison sheets re-export the stored titers** | All 33 `Figure 6a` `CRISPRi` rows and all 99 `Figure 6d` `CRISPRi` rows match a `Figure 4b` value; the KO rows match none | Live. A revision must take only the non-CRISPRi rows |
| 5 | **Lim 2022, forward: 225 of 321 samples are reprocessed from 21 projects** | The note's by-source table carries each source's verbatim `DOI` or `PMID` and BioProject | Already recorded. If Lim 2020 (16 samples), Lim 2021 (14), Bentley 2020 (9) or any of the other 18 is ever added as its own row, its RNA-seq is already in our store at log2 TPM |
| 6 | **Lim 2025 against ALEdb** | "Mutation analysis results were uploaded to ALEdb v1.0 ... accessible at" `http://aledb.org/` "with 'Pputida_isoprenol_TALE' as the project name" | Already recorded in the raw mirror's `si_expected` |
| 7 | **Caglar 2017, internal: Tables S9, S10 and S1 re-release Table S8's and Table S5's values** | Table S9 is a thresholded subset of S8 (`P0.05Fold2`) with no value of its own; Table S10's `log2` and `padj_gene` columns are S8's per-gene values restricted to enriched pathways; `si2.csv`'s `doublingTimeMinutes` holds 19 distinct values over 165 non-NA rows, i.e. the Table S5 condition mean repeated per sample | Live for any revision; each pair must be loaded once |

**The isoprenol group's duplication risk is measurably LOW, which is the headline negative
result.** The highest-risk pair was Carruthers 2025 against Yunus 2026, which share an
author, a chassis designation, a product and substantially one CRISPRi target library, and
where Carruthers releases as numbers exactly the two quantities Yunus releases only as bar
charts. Measured: all 120 Carruthers targets fall inside the Yunus screened set, 118 of them
on the exact tag, and the two papers' FluxRETAP-against-heuristic labels agree with zero
contradictions over 104 shared loci, so they clearly share one target-selection campaign.
But the numbers are independent. Carruthers's per-target titer runs min 0.01, median 162.1,
max 393.7 mg/L while Yunus's stated best is 1469 mg/L for `PP_4188`, for which Carruthers
gives 209.97. On the protein ratio, taking the Carruthers column that measurably holds the
POI ratio against Yunus Table S3's `Relative expression level` over the 96 loci where both
are numeric: zero matches within rounding, median absolute difference 0.7384, maximum
2.9866. This is nothing like #760's r = 1.0000 over 54,326 spacers. Both are legitimately
loadable.

The other four isoprenol papers do not overlap either. Kang measures isoprenyl acetate
rather than isoprenol, and its only released isoprenol numbers are Table S9's 7 fed-batch
values for `PIPAxyl-E3-K3-O15`, a strain no other paper builds. de Siqueira's PT, Lim 2025's
IPL300 and IPL400, Menasalvas's TEAM strains and Carruthers's IY1449b are four distinct
chassis. Menasalvas's reference list contains no Yunus, de Siqueira or Kang entry and
restates no Lim 2025 or Banerjee number. The shared elements across the group are the pIY670
cassette and the `PP_2675` lesion, which are parts and genotype provenance rather than
measurements. One consequence worth carrying: the same `PP_2675` genotype will appear under
three different loaders with three different phenotypes, so the GENOTYPE key collides across
datasets even though no measurement duplicates.

### Missing-dataset leads the comparisons surfaced

| paper | why it matters | mirror status |
|---|---|---|
| **Banerjee et al. 2024**, *Metab. Eng.* 82:157-170, doi 10.1016/j.ymben.2024.02.004, "Genome-scale and pathway engineering for the sustainable aviation fuel precursor isoprenol production in Pseudomonas putida" | the common ancestor of the whole isoprenol group: the source of the IY1449b and PIPA chassis AND of pIY670, cited by Carruthers (ref 31 and Supplementary Reference 2), de Siqueira (ref 20), Kang (Table 1 and Table S2), Lim 2025 and Menasalvas. Kang quotes its numbers second-hand: "this strain supported isoprenol titers of up to 762 mg/L in shake flasks and 3.5 g/L in fed-batch cultures (Banerjee et al., 2024)" | **NOT mirrored.** The only `banerjee*` key is `banerjeeAddressingGenomeScale2025` (doi 10.1038/s41540-024-00480-z), a different aromatic-conversion paper. Hypothesis (untested): it releases per-strain isoprenol titers; its SI has not been read |
| **Houser et al. 2015**, PLOS Comput Biol 11:e1004400 | the source of 27 of Caglar's 152 mRNA and 27 of its 105 protein samples; the duplication in risk 2 above | **NOT mirrored**, no loader |
| **Babel and Krömer 2020**, E. coli K-12 MG1655 isoprenol ALE | Lim 2025's Table S7 and Results restate its gene list and its numbers ("a 47% increase in growth rates over the WT in the presence of 4.3 g/L isoprenol and could grow in the presence of a maximum of 6.8 g/L isoprenol") | **NOT mirrored**, no loader. Only gene names and ALE counts are re-released, so there is no duplication risk, and the comparison is the pointer |
| **Borchert, Bleem, Beckham 2023** (`borchertRBTnSeqIdentifiesGenetic2023`, doi 10.1016/j.ymben.2023.04.007) | its t, p, q and adjusted-q test results, its 186,957 barcode-level rows and its released growth curves are all absent from the compendium | **mirrored**, no loader, and risk 1 above applies to its fitness columns |
| **Thompson et al. 2020** (`thompsonFattyAcidAlcohol2020`, doi 10.1128/AEM.01665-20) | cited as a comparison by Carruthers, de Siqueira, Lim 2025 and Menasalvas | **mirrored**, no loader of its own, and NOT missing: `borchert2024.py` already carries it as a subsumed source study |

Adding any reference to the library is a curation decision, so these are flagged, not acted
on.

### Gap R, stated once for the record

Three papers hit the same missing thing from three directions, which is why it is named
rather than filed under one of them: **there is no protein-level relative-abundance or
fold-change phenotype, and no per-protein p-value slot.**
`ProteinAbundancePhenotype`'s docstring forbids the ratio ("absolute per-strain quantity on
a log signal scale, NOT a ratio -- the WT/parent strain supplies the reference"), the
expression family has the slot the protein family lacks
(`MicroarrayExpressionPhenotype.expression_log2_ratio`, and
`PseudobulkExpressionPhenotype`'s "per-gene pseudobulk log2 fold-change vs the WT profiled
in the same"), and the only p-value field anywhere in `schema.py` is
`gene_interaction_p_value` on `GeneInteractionPhenotype`.

Where it bites, in descending size: Caglar Table S8 (201,408 rows of `log2FoldChange`,
`lfcSE`, `pvalue`, `padj` against a named contrast), Carruthers `Figure 5b` and `Figure 6b`
(6,222 fold changes with matching p-values), Yunus Tables S4 and S5 (338 keys), Kang Table
S6 (40 proteins), Lim 2025's seven proteome sheets (`p-value`, `p_adjusted(BH)`), de
Siqueira's figure-only differential, and Carruthers's ratio columns. Checked against #731,
#739, #749, #753, #756 and #758: none covers it. #753 is the nearest in spirit, since it
also asks for a dispersion field on a protein phenotype, but its subject is
`ProteinTurnoverPhenotype`'s interval and censoring.

**The complication that must be settled first.** The Yunus 2026 loader ALREADY stores a
ratio to control under `ProteinAbundancePhenotype`, over 102 strains. So either the
docstring's prohibition is narrower than it reads, in which case the Carruthers ratio
columns and the Yunus fold-change tables are loadable now, or the Yunus loader is already
over the line. That contradiction is the decision, and it gates five of the rows above.

### Two other gaps with no issue

- **No absolute cell count.** Caglar `si2.csv` carries `cellTotal` and `cellsPerTube`
  (131 of 171 rows non-NA), which the Table S1 caption does not list and which `paper.md`
  never mentions (measured: zero hits for `cell count`, `cells per`, `CFU`,
  `optical density` and `OD600`). No phenotype class holds an absolute cell count and
  `MeasurementType` has no member for one.
- **No optical-density member on `MeasurementType`.** The enum is `log2_ratio`, `z_score`,
  `sensitivity_score`, `categorical`, `ordinal`, `growth_rate`, `differential_fitness`,
  `control_regression_residual`, `colony_size`. Kang's max OD600 (13.2 fed-batch and 16.8
  flask), de Siqueira's Figure S1 cell densities, Yunus's Supplementary Figure S7, Borchert's
  growth curves and Menasalvas's Figure S15 tolerance curves all land on it. Most of those
  are figure-only anyway, so the enum member is the second blocker, not the first.
- **A Dryad deposit behind the same Anubis challenge as #739, on a different DOI.**
  Menasalvas's five supplementary data files sit at `10.5061/dryad.sbcc2frjq` and
  `10.5281/zenodo.17155686`. The raw mirror records the by-hand recipe; no open issue names
  this deposit. It gates loadable-now item 6.

### What this audit did NOT check

- **Menasalvas data S1 to S5 column names, sheet dimensions and row counts.** The files are
  not in the mirror. Every data-S claim rests on the verbatim legend in `si/si1.md`, never on
  the bytes.
- **Banerjee 2024, Houser 2015 and Babel and Krömer 2020.** Not mirrored, not read. The
  Caglar duplication claim rests on Caglar's own two verbatim statements plus the 27-sample
  join measured in Caglar's own Table S1, not on reading Houser.
- **Twelve of Borchert 2023's 13 comparison sheets.** Header-inspected; only `Glu_v_Glu_Van`
  was numerically compared. The six-column, 4,732-gene match there is strong evidence the
  others follow, and it is not a measurement of them.
- **The meaning of Lim 2025's `Fit mean` row.** Measured absent from `paper.md` and from the
  extracted `si1.docx` text. Not guessed.
- **Raw deposits.** PRIDE PXD063733 / 063737 / 063738 / 063740 / 063743 / 063744 / 063746
  (Carruthers), PXD055153 (de Siqueira), PXD067010 (Kang), PXD054609 (Lim 2025), PXD061547
  (Menasalvas), PXD062697 (Yunus), PXD005721 and PXD002140 (Caglar); SRA PRJNA1153078,
  PRJNA1187681, PRJNA1226229, PRJNA809672, PRJNA856070, PRJNA1011287; GEO GSE281392,
  GSE94117, GSE67402. No loader consumes raw reads or spectra and none was fetched.
- **Repositories named in Data Availability sections.** `github.com/fModules/putida-code`
  (which the Borchert note says may hold the figure growth curves, a hypothesis, untested),
  `github.com/beckham-lab/RB-TnSeq`, `github.com/JBEI/Isoprenol_CRISPRi`,
  `github.com/umutcaglar/ecoli_multiple_growth_conditions` (named in Caglar as holding "All
  processed data and analysis scripts", so it may carry per-record data beyond the 14 SI
  tables), the Texas Data Repository `10.18738/T8/UG3TUR`, `mipreadr v0.1`, and the two Yunus
  Benchling pages. None fetched.
- **Figure images.** All figure CAPTIONS were read for every paper; no numeric value was
  digitized off a plot, so every "figure only" verdict rests on the absence of a released
  table rather than on reading bars.
- **The rendered PDFs.** The MinerU OCR markdown was read for each mirrored paper and SI, and
  the raw WordprocessingML for each `.docx`. Visibly garbled OCR spans are quoted as the OCR
  has them.
- **No build, loader, slurm job or dataset process was run**, and nothing outside this note
  was modified. The only code executed was read-only `openpyxl`, `csv` and `zipfile`
  inspection of the two mirrors.

## 2026.10.07 - Correction: Caglar 2017 Table S5 is NOT loadable now, and si2 is not its mean

This section corrects the "Caglar 2017" table row above and rank 1 of "Loadable now". Both
said the doubling times are **loadable now** as `BacterialEnvironmentResponseExperiment` /
`EnvironmentResponsePhenotype` with `measurement_type = growth_rate`, `assay_type =
liquid_od_growth`, `environment_response_uncertainty_type = UncertaintyType.ci95`,
`sample_unit = biological_replicate`, for 55 records.
`[[plan.bacteria-si-phenotype-audit-ecoli]]` rank 15 said the same quantity is blocked by
its gap 1 and gap 9, for 19 records. The E. coli note has the verdict right. The original
text above is left in place.

Everything below is measured by
`experiments/036-dataset-fixes-before-kg-build/scripts/caglar2017_doubling_time_loadability.py`,
which sha256-verifies both SI files against the library mirror's `manifest.json`, then
builds each candidate record form as real pydantic records with the Caglar loader's own
environment helpers and scores it with the environment-response family's own rules.
Results: `experiments/036-dataset-fixes-before-kg-build/results/caglar2017_doubling_time_loadability.json`
and `..._conditions.csv`.

- `si/si6.csv`, sha256 `76411accacbdc28310622cc15289b65ad44937bdc915051bbcfc8c4da1b04c60`
- `si/si2.csv`, sha256 `1486290bf6a340ae64ee20c915435c0a00ff5eede489de1f56b62733b66f8940`

### The 55-versus-19 half is not a disagreement

Both counts are right at their own grain, measured: `si6.csv` holds **55 data rows over 19
distinct `name` values**, and the per-condition replicate counts are 3 for 17 conditions
and 2 for exactly `Gluconate.tab` and `Lactate.tab`. That matches the loader's already
sourced `DOUBLING_TIME_REPLICATES` quote verbatim, Methods, Cell Growth: "Means and
confidence intervals were calculated from three replicate growth curves for all conditions
except for gluconate and lactate, which had measurements for only two replicates." So 55 is
the per-replicate row count and 19 is the condition count. Neither note miscounted.

### What the file actually is, and the reading error above

The row above describes `si6.csv` as holding "the mean, +/-95% confidence interval". It does
not. Its columns are `name,replicate,doubling.time.minutes,doubling.time.minutes.95m,
doubling.time.minutes.95p,r.squared` and **every row is one replicate's own linear fit**,
carrying that fit's own `r.squared`. Methods, Cell Growth, verbatim: "Doubling times were
calculated as $\log_{\mathrm{e}} 2$ divided by the fit slope for each biological replicate
separately." The file holds no condition mean and no confidence interval of a mean; the
mean and the interval of the mean are what Fig. 2 draws ("error bars represents $95\%$
confidence intervals of the mean"), and they are not released as numbers.

### Blocker 1: the interval is asymmetric in 55 of 55 rows, so `ci95` would falsify it

`UncertaintyType.ci95` is defined in `torchcell/datamodels/schema.py` as "95% CI
half-width -> SE = hw / 1.96", a single symmetric number. Measured over the 55 rows:

| statistic | value |
|---|---|
| rows whose interval is symmetric about the value | **0 of 55** |
| upper half-width wider / lower wider | 54 / 1 |
| ratio of upper to lower half-width, min / median / max | -26.3920 / **1.3187** / 10.1066 |
| absolute asymmetry in minutes, median / max | 2.5783 / **1150.7309** |
| rows whose upper bound is not above the value | **1** |

The one row is `Glycerol.tab` replicate 1: value 80.95212424, `95m` 38.94241445, `95p`
**-1027.769034**. A negative doubling-time upper bound is what a slope confidence interval
that straddles zero becomes under `DT = log_e 2 / slope`, so that row has no half-width at
all. Passing either side as `ci95` is accepted silently: the lower half-width derives
`environment_response_se` = 21.4339, the upper derives **-565.6845**, a negative standard
error. So the field as the row above proposes it does not merely round the interval, it can
record a number that is not a standard error.

The repo already has the lossless pattern and says why it is required.
`FluxPhenotype` carries `net_flux_lower` / `net_flux_upper` / `confidence_level` with
`label_statistic_name = None`, docstring verbatim: "a two-sided confidence bound is not a
single number, and naming one of the two bounds as "the" statistic would misreport it."
`EnvironmentResponsePhenotype` has no such pair. That is exactly the E. coli note's gap 9.

### Blocker 2: three verifier rules reject the absolute form, not one

Each form below was built as real records and scored by
`torchcell.verification.environment_response`'s own `_l1_pair_uniqueness`,
`_l3_environment_perturbed` and `_l3_reference_zero`.

| form | records | `pair_uniqueness` | `environment_perturbed` | `reference_zero` |
|---|---|---|---|---|
| A absolute, per replicate (this note's proposal) | 55 | FAIL, 39 duplicates, 16 unique triples | FAIL, 9 records | FAIL, max&#124;v&#124; = 53.3 |
| B absolute, per condition (the E. coli note's grain) | 19 | FAIL, 3 duplicates, 16 unique | FAIL, 3 records | FAIL, max&#124;v&#124; = 53.3 |
| C log2 ratio, all 19 conditions, `screen_id` set | 19 | PASS | FAIL, 3 records | PASS |
| D log2 ratio, in-experiment base only, `screen_id` set | **11** | PASS | PASS | PASS |

- `reference_zero` is the E. coli note's gap 1, confirmed: the reference record's
  `environment_response` must be 0 for a numeric readout, and the reference's absolute
  doubling time is 53.25 min. An absolute rate cannot satisfy it, and the phenotype
  validator forbids a `None` response for a non-categorical `measurement_type`, so the
  reference cannot decline to carry a number.
- `environment_perturbed` and `pair_uniqueness` are blockers neither note named. Three of
  the 19 conditions ARE the base condition measured in three separate experiments
  (`Glucose.tab`, `MgSO4_000.800_mM.tab`, `NaCl_005_mM.tab`): glucose, 0.8 mM Mg2+, 5 mM
  Na+, so they carry no environmental edit and collide on the condition signature. A fourth
  collision is `MgSO4_000.080_mM` against `MgSO4-2_000.080_mM`, the same 0.08 mM Mg2+ run in
  two experiments. `screen_id` is the honest discriminator for the latter (Table S1's own
  `experiment` column names the run), and it is what makes forms C and D unique.

### Blocker 3: the ratio form is available for 11 of 19 conditions, and the base is a choice

Form D passes every rule, but it stores 11 records: the three base rows become references,
and **the whole `MgSO4_stress_low` series has no base-condition row of its own** --
`si6.csv` releases no 0.8 mM Mg2+ curve under the `MgSO4-2` prefix, so five conditions
(15 of the 55 rows) have no in-experiment reference. Borrowing another experiment's base is
not harmless: the three released base measurements are 53.2538, 61.9140 and 58.3515 min, a
**0.2174 log2** spread, which is larger than several of the effects being described. And a
ratio of doubling times is a quantity the paper never releases, so storing it would record
11 derived numbers, drop 8 of 19 conditions, and gap every uncertainty.

### Verdict

**Nothing from Table S5 is loadable today.** The released measurement is an absolute
doubling time with an asymmetric per-replicate interval; `EnvironmentResponsePhenotype` can
hold neither. The two capabilities needed are small and already exist in spirit elsewhere
in the schema, but both would add fields to `EnvironmentResponsePhenotype`, which moves the
schema closure of every served environment-response dataset and so forces a FULL KG
rebuild. That is the driver's call, not a loader's. Filed as #776; rank 1 of "Loadable
now" above should be read as blocked, and the firm total of 157 records drops to 102.

### Duplication risk 7 was right on the grain and wrong on the value

The risk-7 row above says `si2.csv`'s `doublingTimeMinutes` holds "19 distinct values over
165 non-NA rows, i.e. the Table S5 condition mean repeated per sample". The grain is
confirmed and the join is exact; the characterization of the value is wrong.

Measured. `si2.csv` has 171 rows, 165 with a doubling time (the 6 blanks are the
`pilot_24_hour` and `pilot_mid-log` samples), and those 165 rows hold **19 distinct
`(doublingTimeMinutes, .95m, _95p, rSquared)` tuples**, not merely 19 distinct values. Each
tuple's condition joins **1:1** onto one of Table S5's 19 `name` values on
`(experiment, carbonSource, Mg_mM, Na_mM)`, covering all 165 rows; the join raises if any
condition selects more or fewer than one value, so the correspondence is measured. So yes:
Table S1 repeats one condition-level fit across that condition's samples, and the two
sources must be loaded once, not twice.

But Table S1's value is **not** an aggregate of Table S5's replicate values:

| aggregate of Table S5's replicates | exact matches to Table S1 (< 1e-6) | median &#124;difference&#124; | max &#124;difference&#124; |
|---|---|---|---|
| arithmetic mean | **0 of 19** | 0.5571 min | 6.5763 min |
| geometric mean | 0 of 19 | 0.5902 min | 6.9147 min |
| harmonic mean | 5 of 19 | 0.7582 min | 7.2351 min |

Table S1 also carries one `rSquared` per condition where Table S5 carries one per
replicate, and its interval is only mildly asymmetric (half-width ratio 1.0981 to 1.4632
over 165 rows) against Table S5's -26.3920 to 10.1066. So Table S1 holds a **separate
condition-level fit of the same OD600 curves**, not a summary of Table S5. Hypothesis
(untested): it is one linear fit to the pooled exponential-phase points of all replicate
curves, which equals the harmonic mean of the per-replicate doubling times exactly when the
replicates share a time grid, and 5 of 19 conditions match that exactly. Consequence for a
future revision: the two tables are two fits of one experiment, so the duplication is real
at the level of the EXPERIMENT even though 14 of 19 numbers differ. Pick the per-replicate
table, which is the finer grain.

### One further finding neither note recorded

Table S5 contains an internal duplicate: `MgSO4-2_000.020_mM.tab` replicate 3 and
`MgSO4-2_000.040_mM.tab` replicate 3 are identical in all four numeric columns
(64.6719153, 58.70481327, 71.98933404, 0.994667398). One fit appears under two different
Mg2+ concentrations, so the 55 rows hold 54 distinct fits. A per-replicate load would store
one growth curve twice under two different environments.

## 2026.10.08 - Corrections from building ranks 4, 5 and 7

Three "loadable now" rows were implemented. Two of the three needed a correction, both
measured on the same sha256-pinned bytes the audit read. The original text above is left in
place. Loaders, dev stores and verification:
[[torchcell.datasets.pputida.kang2026]], [[torchcell.datasets.pputida.desiqueira2025]],
[[torchcell.datasets.pputida.yunus2026]].

### Rank 4 is NOT 21 records, it is 0, and the blocker is a missing denominator

The audit called Kang's 7 fed-batch isoprenol titers "unconditional". They are not
storable at all today. `ProductTiterExperimentReference` requires a `phenotype_reference`
and `ProductTiterPhenotype.titer` is a required float, so an isoprenol record needs an
isoprenol REFERENCE titer, and Table S9 is the only place this paper releases any
isoprenol number. Measured over every SI table and the whole `paper.md`: Table S1 is a
physicochemical property comparison, Table S4's only titer column is `IPA titer (mg/L)`
(the ester), Tables S2, S3, S5 to S8 release no titer, and Fig. 2b's isoprenol bars have
no companion table.

The one isoprenol baseline the paper states is second-hand on three counts, which is why
it cannot be the reference: "In the previously reported engineered P. putida KT2440
background, this strain supported isoprenol titers of up to 762 mg/L in shake flasks and
$3 . 5 ~ \mathrm { g } / \mathrm { L }$ in fed-batch cultures (Banerjee et al., 2024)."
The measurement is Banerjee's; the strain is the background PIPA was ADAPTED FROM rather
than PIPA; and no medium, sugar load or replicate count travels with either number.
Banerjee 2024 is not mirrored, which this audit's own "did NOT check" list already says.
The sibling Yunus 2026 loader refuses its entire titer family for the same shape of gap,
and that precedent was followed.

What the revision did instead: read the column (it was asserted by `SI_TABLE9_COLUMNS` and
declared nowhere on `FedBatchRow`, so it was parsed past), carry it per record in
`preprocess/titer_rows.csv`, oracle-check it, and re-join it to the deposited bytes with a
new verification level. The decline is `ISOPRENOL_NOT_A_RECORD` in the loader and is
written into the build accounting.

**The audit's proposed source for the 14 metabolite records does not exist.** The caveat
reads "n = 1 must be asserted from 'a single run'". That phrase is in no mirrored byte.
Measured: `replicate` occurs ONCE in `paper.md` and the hit is "To replicate this
composition, glucose and xylose are commonly added in a 2:1 ratio"; every figure caption
except Fig. 8's says "Error bars indicate the standard deviation of biological
triplicates"; Fig. 8's says only "Cultivations were performed with 1 L medium and 200 mL
overlay with sampling approximately every 12 h"; and the Table S9 caption states no
replicate count. So `MetabolitePhenotype.n_replicates`, which is required with a `>= 1`
validator and therefore non-gappable, has nothing to be filled from, and that is a second
blocker independent of the missing units field.

### Rank 5 is right, and the two statistics are re-measured over all 34,600 cells

The audit measured the de Siqueira non-recoverability on a 2,000-row sample. Re-measured
over every released cell, both findings hold and the percent one is stronger than reported.

| claim | audit | re-measured over 34,600 cells |
|---|---|---|
| percent is not percent-of-the-mean | per-sample ratio 0.959 to 0.997 | per-CELL ratio 0.914506 to 1.083554, median 1.000007, and ZERO cells agree exactly |
| log10 is the mean of logs | 1,999 of 2,000 sampled rows disagree | 33,914 strictly below, 686 within 5e-5, **0 above**, largest gap 1.378950 |

The one-sided log10 result is Jensen's inequality for a mean of logarithms, which is a
stronger statement than a disagreement count: a log10-of-the-mean column would agree
everywhere. The audit's first-row numbers reproduce exactly (`Csda`, -2.23385590492606
against -2.19547). Two additions the audit did not make: the log10 SD is not the
delta-method transform of the percent SD either (34,496 of 34,600 cells disagree), and the
two columns the audit classified as derived are now build oracles rather than prose (the
CV is exactly `100 * pct_sd / pct_mean` on every cell; the SEM is single-valued for 1,728
of 1,729 proteins).

**One structural correction.** The audit says `measurement_type` being a free `str` makes
each normalization "its own record set" inside one dataset. It cannot be: the shared
`verify_protein_dataset` asserts a single `measurement_type` per DATASET ("no silent
cross-assay mixing"), so each normalization is its own dataset CLASS, with the full adapter
gate. Rank 3 (Carruthers's two unstored abundance sheets) is on the same Top3
percent-of-total basis the Carruthers loader already stores, so it is unaffected; any
future row that proposes a second scale inside an existing protein dataset is.

### Rank 7 is right, and the no-clash claim is right for the wrong reason

The audit says "PP_4188 itself is absent from both tables so there is no clash with its
Table S3 row". No row's `Protein` column is `PP_4188`, and the build asserts it. But the
protein is in the tables: Table S4 row 40 is `Kgdb` (`Q88FB0`, "Dihydrolipoyllysine-residue
succinyltransferase component of 2-oxoglutarate dehydrogenase complex") at a fold change of
0.238537433, which is the enzyme this paper's own Table S2 names for `PP_4188`
("2-oxoglutarate dehydrogenase dihydrolipoyltranssuccinylase subunit"), and the pinned
annotation resolves `sucB` to `PP_4188`. Row 39 is `Kgda` (`Q88FA9`, the E1 component) =
`PP_4189` by the same route. `kgdB` resolves to no locus of this assembly, so both land in
the 33 released keys that are dropped for want of a gene node.

So the record carries **305 of the audit's 338 keys**, not 338, and the reason nothing
clashes is the missing UniProt-to-locus-tag crosswalk rather than the protein's absence. A
crosswalk that recovered `Kgdb` would put 0.238537433 beside Table S3's 0.2213 for the same
strain, on a different `measurement_type` -- the two independent runs the Yunus note already
documents, not a contradiction.

### Firm total, revised

| rank | audit | built |
|---|---|---|
| 4 | 21 | **0** |
| 5 | 10 | **10** |
| 7 | 1 record, 338 keys | **1 record, 305 keys** |

**11 records, not 32.** The audit's "firm total: 157 records across items 1 to 5 and 7"
becomes 136 once rank 1 (55, corrected to 0 by the Caglar section above) and rank 4 (21,
corrected to 0 here) come out: 49 + 21 + 10 + 1 for ranks 2, 3, 5 and 7, of which ranks 5
and 7 are now built and ranks 2 and 3 (Carruthers, 70 records) are still open.
