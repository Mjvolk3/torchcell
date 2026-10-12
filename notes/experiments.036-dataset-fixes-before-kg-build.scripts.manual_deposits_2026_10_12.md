---
id: nhnsj7yfynrboe7p92f53nu
title: Manual_deposits_2026_10_12
desc: ''
updated: 1791803314904
created: 1791803314904
---

## 2026.10.12 - Six bacterial manual deposits recorded and measured

The owner downloaded, in a browser, the supplementary files scripts could not fetch for
six bacterial papers, and they were copied into `$DATA_ROOT/torchcell-raw/<key>/`
(arrival times are the downloads' mtimes, 2026-10-12 03:40 to 03:41 CDT). Every sha256
measured at copy time was re-measured here against both the mirror copy and the owner's
original; all 28 files match.

`--record KEY ...` writes `DEPOSIT.md` and `SHA256SUMS.txt` beside each deposit (one pair
per directory) and extends the key's `manifest.json` through
[[torchcell.literature.manual_deposit]]; `--measure` writes
`experiments/036-dataset-fixes-before-kg-build/results/manual_deposits_2026_10_12.json`.

```bash
python experiments/036-dataset-fixes-before-kg-build/scripts/manual_deposits_2026_10_12.py --record thompson nichols teteneva ishii
python experiments/036-dataset-fixes-before-kg-build/scripts/manual_deposits_2026_10_12.py --measure
```

### What was recorded

| key | files | recorder | manifest | scripted equivalent measured |
|---|---|---|---|---|
| `tongGeneDispensabilityEscherichia2020` | `data/carpe/CarPE.pdf` (+ layout text, parsed CSV) | `tong2020.record_carpe_deposit` | extended | none (a Shiny app) |
| `balakrishnanPrinciplesGeneRegulation2022` | SM PDF, MDAR checklist, Tables S3-S7 zip and its 5 members (+ SM layout text) | `balakrishnan2022.record_si_deposit` | extended | none (PMC proof-of-work, #853) |
| `ishiiMultipleHighThroughputAnalyses2007` | `si/ishii_som.pdf` (+ MinerU OCR, 103 figures) | `ishii2007.record_som_deposit` | extended | none (Cloudflare, no PMCID) |
| `thompsonFattyAcidAlcohol2020` | SI figure PDF, 2021 correction (+ layout text) | this script | extended | none (ASM answers 403) |
| `nicholsPhenotypicLandscapeBacterial2011` | `si/mmc1.xls` to `mmc8.pdf` | this script | CREATED | all 8 byte-identical to `ars.els-cdn.com` (`elsevier_mmc`) |
| `tetenevaGenomewideScreenGenetic2024` | 5 supplementary files | this script, `teteneva2024.SI_DEPOSIT` | CREATED | all 5 byte-identical to `pmc-oa-opendata/PMC11188689.1/` (`pmc_cloud_object`) |

Every file is `RetrievalMethod.manual_browser` with the recipe in its retrieval
`params`; where a scripted route was fetched on 2026-10-12 and returned the same bytes,
`params.scripted_equivalent` names it, so a rebuild of those two keys needs no person.
Elsevier's CDN answers 404 for `mmc9` in pdf, xls, xlsx, docx, doc and zip, so the eight
Nichols files are the complete supplement.

### Balakrishnan 2022: Table S3 against GEO, and Tables S4 to S7

The Science Table S3 (`data/science.abk2066_table_s3.xlsx`, 1,752,886 B) is not
byte-identical to the GEO copy the loader reads (2,270,095 B), and it is cell-identical
in everything the loader reads:

| sheet | rows (GEO, Science) | cells differing | of which float | max relative float difference | other differences |
|---|---|---|---|---|---|
| `1 - RNAseq-ss (description)` | 30, 30 | 0 | 0 | 0 | none |
| `2 - RNAseq-ss (fractions)` | 4,344, 4,344 | 33 | 33 | 2.2e-16 | none |
| `3 - RNAseq-deg (description)` | 17, 17 | 2 | 0 | 0 | rows 2 and 10, column 7 |
| `4 - RNAseq-deg (fractions)` | 4,344, 4,344 | 14 | 14 | 1.9e-16 | none |

The 47 float differences are the last binary digit of a 7-significant-digit value (GEO
stores `6.132878000000001e-05`, Science `6.132878e-05`). The two text differences are
the nitrogen source of the M9 rifampicin samples `RIF_A0` and `RIF_B0`: GEO reads
`10 mM NH4Cl`, the Science copy `11.34 mM (NH4)2SO4`, which is the salt every M9 sample
of sheet 1 carries. Sheet 3 is not consumed by any loader.

| table | sheet | non-empty rows | columns | what it holds |
|---|---|---|---|---|
| S4 | `1 - Proteomics (description)` | 30 (29 samples) | 10 | sample, group, growth rate, strain, medium, carbon, nitrogen, supplement, description, SWATH file |
| S4 | `2 - Proteomics (fractions)` | 4,343 (4,342 genes) | 32 | `gene`, `locus`, `gene length (nt)` and 29 sample columns of protein number fractions |
| S5 | `list of fold changes` | 4,344 | 11 | per gene and per limitation (C, A, R): fit, log10 fold change of relative translation efficiency, fit error |
| S5 | `top outliers` | 39 | 6 | genes whose translation efficiency changes more than 3-fold |
| S6 | `reference condition` | 4,343 (4,342 genes) | 10 | eight per-gene parameters (below) |
| S6 | `C-limited condition` | 4,343 (4,342 genes) | 10 | the same eight at slow C-limited growth |
| S7 | `1 - description of samples` | 20 (19 samples) | 5 | sample, description, carbon source, growth medium, proteins detected |
| S7 | `2 - protein variability data` | 1,416 (1,415 proteins) | 23 | 19 sample columns of protein mass fractions, the median and the maximum fold change |

Table S4's 29 samples (8 NCM3722, 8 NQ1243, 6 NQ1390, 7 NQ393) are all sample names of
Mori 2021 Dataset EV9 as `mori2021.SAMPLES` declares them, each with the same strain.
That supports the #853 reading that Table S4 re-serves Mori 2021 EV9, at the level of
sample names and strains; the values were not compared.

**Table S6, the new dataset candidate (#853).** Reported (non-NaN) cells per parameter,
of 4,342 gene rows (4,325 distinct loci; the 17 repeats are the `0` locus of the
insertion-element rows):

| parameter | reference condition | C-limited condition |
|---|---|---|
| protein concentration [Pi] (1/µm3) | 1,922 | 1,972 |
| mRNA concentration [mRi] (1/µm3) | 4,239 | 4,254 |
| gene concentration [Gi] (1/µm3) | 4,330 | 4,330 |
| translation inititation rate αpi (1/min) | 1,921 | 1,970 |
| mRNA degradation rate, δi (1/min) | 2,696 | 2,883 |
| translational burstiness, bi | 1,846 | 1,897 |
| transcriptional initiation rate, αmi (1/min) | 2,694 | 2,880 |
| promoter on-rate, ki (µm3/min) | 2,696 | 2,883 |

The parameter names are the released column headers, verbatim. No loader is written for
Table S6; the inventory is for the candidate gate.

### Nichols 2011: the Cell supplemental information

| file | format | title (cell A1) | content, measured |
|---|---|---|---|
| `mmc1.xls` | BIFF | Table S1 - Screen List | 115 condition rows: condition, supplier, item no., target, concentrations screened, screens; the `Screens` column sums to 648 |
| `mmc2.xls` | OOXML under an `.xls` name | Table S2 - Complete Data Set | THE chemical-genomic fitness-score matrix: 3,979 strain rows x 324 condition columns |
| `mmc3.xls` | BIFF | Table S3 - Multi-Stress Responsive Genes | 94 strain labels |
| `mmc4.xls` | BIFF | Table S4 - Conditionally-Essential Genes | two lists: 116 rich-media conditionally essential, 102 auxotrophs |
| `mmc5.xls` | BIFF | Table S5. Rich-media CE Lethal Events | 831 (strain, lethal condition) pairs |
| `mmc6.xls` | BIFF | Table S6 - Highly Correlated Orphan-Annotated Gene Pairs (r ≥ 0.5) | 832 orphan to annotated gene pairs with r and function |
| `mmc7.xls` | BIFF | Table S7 - Drug-GO Interactions at p<= 10-3 | 719 (GO process, drug, score) rows; the sheet itself is named `TableS5-Drug-GO interactions.tx` |
| `mmc8.pdf` | PDF, 23 pages | Resource | the article (pages 1 to 14) followed by the Supplemental Information (Extended Experimental Procedures and supplementary figures, from page 15) |

**Table S2, the matrix a loader would read.** 3,979 strain rows (3,967 distinct labels;
12 labels sit on two rows each), 324 distinct condition columns, 1,286,962 numeric cells
and 2,234 empty ones (3,979 x 324 = 1,289,196), values from -44.33 to 9.30. The Extended
Experimental Procedures call the scores "analogous to ‘‘unaveraged’’ S-scores (Collins
et al., 2006)". Condition labels are `<condition>-<concentration> - <suffix>`, the suffix
`UNSPECIFIED` on 309 columns and empty on 15. Row labels, by shape:

| label shape | rows |
|---|---|
| ECK number and symbol (`ECK1963-HCHA`) | 3,678 |
| SPA-tagged essential gene (`... - SPA`) | 117 |
| ECK number and symbol, primed (`ECK2350-YFDM'`) | 112 |
| ECK number only (`ECK2331`) | 41 |
| other: alleles, truncations, insertions (`ECK0095-B-FTSA(R286W)-KAN`) | 27 |
| several ECK numbers, a multi-gene deletion (`ECK1205/1207/1209-RDLABC`) | 4 |

What the prime marks is not stated in the tables; hypothesis (untested): the second of
two independently derived clones the Extended Experimental Procedures mention. Shiver
2016 (`torchcell/datasets/ecoli/shiver2016.py`) re-analyzes 235 of these conditions as
its batch 0, so a loader must take the G4 overlap against `shiver2016` first.

**G3.** `python -m torchcell.candidates inventory --citation-key
nicholsPhenotypicLandscapeBacterial2011` now runs: "8 file(s) in raw, 0 worksheet row(s)
counted; all intact". G3 counts worksheet rows only in files named `.xlsx`, so neither
the BIFF tables nor the OOXML `mmc2.xls` are counted; the counts above are this
script's.

### Teteneva 2024: Tables S2 to S4

- **Table S4** (`supplementary_table_s4_wrae096.xlsx`, sheet `DATA`): three header rows,
  then 11,027 gene-by-water-sample rows (water sample 1: 3,675, 2: 3,679, 3: 3,673) with
  six fitness columns (filtered and non-filtered water at days 2, 4 and 8), 66,162
  non-empty fitness values (the schedule's count, reproduced), PC1, PC2, a Euclidean
  distance and a protein description. Genes are named ONLY by gene symbol (column
  `Gene name`), 3,691 distinct. The `LEGEND` sheet does not name the statistic.
- **Symbol routes** over those 3,691 symbols, by `/gene` alone: the GenBank member
  `GCA_000010245.1` lands 3,036 uniquely and 655 on no gene feature; the RefSeq member
  `GCF_000010245.2` lands 3,607 uniquely, 44 on two or more features and 40 on none.
- **Table S3**: 63 sequencing samples (per water sample 3 controls at day 0, 9 filtered,
  9 non-filtered: 3 days x 3 technical replicates); the legend says "One sample that
  failed the QC is omitted from the further analysis and is highlighted in red", and the
  red-filled row is `sample_10`.
- **Table S2** (PDF): five lake water samples, #1 to #5, with date, pH, air and water
  temperature; Table S4 carries only #1 to #3.
- **G3**: "5 file(s) in raw, 11100 worksheet row(s) counted; all intact".

### Thompson 2020: what the 2021 correction says, and what it touches

The correction (`si/thompson2021_correction_PMC8091123.pdf`, sha256 `2c7e387f...`, one
page, ASM's PDF of DOI 10.1128/AEM.00177-21) has one paragraph after the citation line.
Verbatim from its layout text (line breaks folded to spaces; the text layer prints the
"fi" of "first" as the ligature "ﬁ"):

> Readers should note that when the authors created the deletion mutant of PP_2675,
> they deleted the entire region between the gene’s start and stop codons. However, this
> also deleted the DNA that codes for the first 14 amino acids of the overlapping ORF
> PP_2676.

It then says "Furthermore, the originally published URL for the JBEI public strain
repository was incor-" (the word breaks across the line) and gives the correct link as
"The correct link to all strains and plasmids in this study is
https://public-registry.jbei" followed on the next line by ".org/folders/641.". So it
rescopes two things: the targeted `ΔPP_2675` strain (the paper's Table 1; the growth
curves of Figs. 7 and 9A and the alcohol consumption of Fig. 9C) also lacks the first 14
codons of `PP_2676`, and the strain registry folder is 641, not the printed 456.

Mapped onto the Borchert 2024 compendium (`borchert2024.attribute_sample` over the
release's metadata): Thompson 2020 is the source of **46 samples** (set15: 24, set16: 12,
set12: 10), **217,672 records** at 4,732 loci, every one of them mutant library
`Putida_ML5_JBEI`, experiment group `carbon source`. None of the 46 names PP_2675,
PP_2676, pedF, a deletion or a knockout in its description, conditions or medium. They
are pooled RB-TnSeq fitness assays of the transposon library in wild-type KT2440, and
the correction concerns a separately built deletion strain and a URL. So it rescopes
**0 of the 46 samples and 0 stored values**. (The Thompson subsumption record of
2026-10-08 counts 47 samples and 222,404 records for this paper; the 46 here are the
loader's own attribution, which #986 also uses.)

One stored dataset IS touched by the correction's content: de Siqueira 2025's PT strain
is "ΔPP_2675 Δ14-PP_2676" and defers its construction to Thompson 2020, and
`desiqueira2025.py` carries a `ProvenanceGap` on `deleted_span` because the source does
not say whether 14 is base pairs or codons. The correction answers the unit: 14 amino
acids, so 14 codons (42 bp) at the start of PP_2676. No coordinates are given. Not
changed here; it is a candidate fix for the de Siqueira loader.
