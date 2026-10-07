---
id: cynwfom5hghtqlf0z6jhu6b
title: Choe2025
desc: ''
updated: 1791407627991
created: 1791407627991
---

## 2026.10.07 - Row 22 of the fifty bacterial datasets

Choe D, Lee E, Kim K, Hwang S, Jeong KJ, Palsson BO, Cho BK, Cho S. "Rapid
identification of key antibiotic resistance genes in E. coli using high-resolution
genome-scale CRISPRi screening." iScience 2025 28:112435,
[doi:10.1016/j.isci.2025.112435](https://doi.org/10.1016/j.isci.2025.112435). Citation
key `choeRapidIdentificationKey2025`, PMC12063145.

Loader: `torchcell/datasets/ecoli/choe2025.py`, class
`CrispriChemgenChoe2025Dataset`, dev store
`$DATA_ROOT/data/torchcell/crispri_chemgen_choe2025`. Tests:
`tests/torchcell/datasets/ecoli/test_choe2025.py`.

### What the screen is

A 39,591-member sgRNA library, one guide per 100 nt of all 4,198 MG1655 coding
sequences, transformed into *E. coli* K-12 MG1655 carrying pdCas9, grown in LB and in LB
plus a sub-inhibitory dose of each of twelve antibiotics, with each guide's abundance read
by amplicon sequencing of the spacer. The three released tables the loader consumes:

| table | file | what it holds |
|---|---|---|
| S1 | `mmc2.xlsx` | per-guide normalized abundance (RPM), 39,580 guides x (2 library samples + 2 replicates x 13 conditions) |
| S2 | `mmc3.xlsx` | per-gene enrichment ratio (ER) in 13 conditions, 4,198 genes, with the symbol-to-b-number map and `gRNA #` |
| S6 | `mmc5.xlsx` | the 39,591 synthesized oligos; the 20 nt spacer is sliced from the stated flanks |

### Record type, and why

RECORD = one (guide x antibiotic) `BacterialEnvironmentResponseExperiment`, 466,569 of
them. `EnvironmentResponsePhenotype` with `measurement_type=log2_ratio` and
`assay_type=pooled_competitive_growth_barcode`; the value is

```
environment_response = log2( a_abx / a_LB )
```

where `a_c` is the guide's mean abundance over condition `c`'s two replicate samples,
which is the paper's own definition of an sgRNA's abundance under a condition. The
reference is the SAME clone in untreated LB, so its value is log2(1) = 0.

Three decisions sit behind that line.

1. **The denominator is the untreated LB control, not the initial library.** The paper's
   gene-level statistic ER divides by the pool WITHOUT dCas9, so ER in plain LB measures
   the knockdown's own fitness cost and ER under a drug conflates that cost with the drug
   effect. The paper's own antibiotic-response statistic is the ratio between the two,
   released in Table S4 as "Log2 ER ratio fold-change (Abx/LB)", and the library cancels
   out of it: `log2(ER_abx / ER_LB) = log2(a_abx / a_LB)`. That is a treatment-over-control
   barcode abundance ratio, exactly what `MeasurementType.log2_ratio` documents, and it
   makes the reference a real control environment whose response is 0. Storing log2(ER)
   instead would make the thirteenth, untreated condition a record with no environmental
   edit, which the environment-response verifier's L3 `environment_perturbed` rule
   correctly refuses.
2. **Not `FitnessPhenotype`.** The stored number is signed (234,744 of 466,569 values are
   negative, 13,176 of them below -1; min -7.2275, max +5.3219, median -0.0031) and its
   baseline is 0, not 1. `FitnessPhenotype` is a strictly positive growth ratio that clamps
   non-positive values and whose verifier requires a 1.0 reference.
3. **Per guide, not per gene.** Table S2's ER is a median the authors took over a gene's
   guides; Table S1 is the per-guide release it is computed from, and this row's sequence
   basis is `K-12+guide`. The genotype is one
   `BacterialCrisprInterferencePerturbation` whose `crispr` construct carries the effector
   `dCas9`, the 20 nt Table S6 spacer and the gene's Table S1 guide count, and the spacer is
   what keeps sibling guides of one gene distinct strains
   (`verification.environment_response._genotype_signature`). The two replicate columns
   are averaged because that average IS the paper's definition of an sgRNA's abundance;
   pairing replicate 1 of an antibiotic with replicate 1 of LB would assert a pairing the
   release never states, so the replicate count rides on `n_samples = 2` instead.

### Sourcing table

Every value is a module-level `SourcedValue` in `SOURCED_VALUES` (or a per-condition
`dose_sourced`), anchored to `paper.md`
sha256 `d29923ce2276b17882e89407b683a1edce74d4da74535f7398f7e68c86abcbf7` or to
`si/si1.md` sha256 `bc26a5be98957b07ed65addd027fe7240a6ec52ad2710dcd9b7a3a245e1e8895`
in the literature mirror. The data-gated test
`test_every_sourced_value_is_backed_by_its_quote` audits all 20 against those pins.

| field | value | verbatim quote (source, section) |
|---|---|---|
| host strain | MG1655 | "Purified sgRNA library plasmids (100 ng) were transformed into E. coli K-12 MG1655 competent cells containing the pdCas9 plasmid through electroporation." (paper.md, METHOD DETAILS, 'Generation of antibiotics treated populations') |
| `CrisprConstruct.effector` | dCas9 | "dCas9 and sgRNA were expressed with separate plasmids, pdCas9 plasmid73 (cat. #46569; Addgene, Cambridge, MA, USA) and pgRNA-bacteria plasmid4 (cat. #44251; Addgene) targeting mrfp (Figure S1A)" (paper.md, 'Bacterial strains') |
| library size | 4,198 CDSs, 39,591 oligos | "we designed and synthesized a library of evenly spaced sgRNAs targeting every 100 bp of the 4,198 coding sequences (CDSs) in E. coli (targeting 39,591 positions, Figure 1A)" (paper.md, Results) |
| library coverage | 39,574 | "Of the sgRNAs, 39,574 sgRNA sequences were present in the initial library (lib), representing 99.96% coverage (Table S1)." (paper.md, Results) |
| guide spacing | 100 nt | "we designed sgRNAs for the CRISPRi system with a ratio of one within 100 nt of all CDSs in E. coli K-12 MG1655." (paper.md, 'sgRNA library construction') |
| oligo layout | 26 nt prefix + 20 nt spacer + 33 nt suffix | "A pool of 39,591 oligos consisting of a 20 nt random spacer ... -GCTCAGTCC TAGGTATAATACTAGTA-20 nt spacer-GTTTTAGAGCTAGAAATAGCAAGTTAAAATAAG- ... by CustomArray" (paper.md, 'sgRNA library construction') |
| window length | 23 nt (20 + PAM) | "candidate sgRNAs were acquired by dissecting all genic regions into 23 mers that began with the protospacer adjacent motif (PAM) sequence" (paper.md, 'Design of sgRNA library') |
| design uniqueness | one locus | "Sequences that aligned to more than one locus in the genome were excluded." (paper.md, 'Design of sgRNA library') |
| CDS source | NC_000913.3 RefSeq | "gRNAs positioned in the CDS were extracted from the CDS list (CDSs extracted from the NC_000913.3 RefSeq annotation)" (paper.md, 'Design of sgRNA library') |
| `n_samples` | 2 | "The abundance change of each sgRNA was calculated by dividing its mean abundance (reads per million mapped reads) from two replicate measurements under a given condition by its mean abundance in the initial library." (paper.md, 'Quantification of gene fitness') |
| ER definition | median over a gene's guides | "The enrichment ratio (ER) of a gene is the median of abundance changes of all sgRNAs targeting that gene." (paper.md, 'Quantification of gene fitness') |
| medium + selection | LB, 35 ug/mL chloramphenicol, 100 ug/mL ampicillin | "E. coli cells for sgRNA library construction and for CRISPRi screening were each cultivated in LB medium supplemented with 100 ug/ml ampicillin and 35 ug/ml chloramphenicol plus 100 ug/ml ampicillin, respectively." (paper.md, 'Bacterial strains') |
| formulation | LB Miller broth | the KEY RESOURCES TABLE row "LB Miller broth / BD Difco / 244620" (paper.md) |
| temperature, aerobicity, harvest | 37 C, shaken 250 rpm, harvested at a per-condition OD600 | "Then, pre-cultured cells were re-inoculated to an OD600 0.05 in 50 ml LB medium or LB media with antibiotics containing a sublethal dose of antibiotics (Table S3), and cultured shaking with 250 rpm at 37 C until the required cell density was achieved." (paper.md, 'Generation of antibiotics treated populations') |
| 12 antibiotics | 12 | "Next, we exposed the culture to sub-inhibitory concentrations of 12 different antibiotics to identify genes affecting cellular responses to antibiotics (Figure S3; Table S3)" (paper.md, Results) |
| `assay_type` | pooled_competitive_growth_barcode | "The number of reads fall onto each gRNA were counted (Table S1) and ratio of gRNA abundance compared to initial library was calculated. The enrichment ratio (ER) was determined as the median of all gRNA abundance ratios in a gene (Table S2)." (paper.md, 'sgRNA library sequencing') |
| raw reads | ENA PRJEB33267 | "Original data of CRISPRi amplicon sequencing have been deposited at European Nucleotide Archive as ENA: PRJEB33267 ... and are publicly available as of the date of publication." (paper.md, 'Data and code availability') |

Doses, solvents and sampling densities come from Table S3 in `si/si1.md`; each
`ScreenCondition.table_s3_row` holds its own HTML row verbatim.

| condition | solvent / media | lethal dose | treatment | stored dose | sampling OD600 |
|---|---|---|---|---|---|
| LB (control) | LB | -- | -- | -- | 1.80 |
| CCCP | 0.4% DMSO/LB | 16 | 5.3 ug/ml | 5.3 `ug/mL` | 1.22 |
| Polymyxin B | LB/LB | 2.7 | 0.5 ug/ml | 0.5 `ug/mL` | 1.53 |
| Pyocyanin | 0.4% DMSO/LB | 26 | 5 ug/ml | 5.0 `ug/mL` | 1.50 |
| Rifampicin | 0.4% DMSO/LB | 9.5 | 3.2 ug/ml | 3.2 `ug/mL` | 1.32 |
| Sulfamethizole | 0.4% DMSO/LB | 0.6 | 0.2 mg/ml | 200.0 `ug/mL` | 1.36 |
| Verapamil | LB/LB | 10.8 | 3.6 mM | 3.6 `mM` | 1.50 |
| Erythromycin | 0.4% DMSO/LB | 40.3 | 13.4 ug/ml | 13.4 `ug/mL` | 1.12 |
| Puromycin | LB/LB | 77.8 | 26 ug/ml | 26.0 `ug/mL` | 2.00 |
| Phleomycin | LB/LB | 121.2 | 5 ug/ml | 5.0 `ug/mL` | 1.12 |
| Mitomycin C | 0.4% DMSO/LB | 79.6 | 26.5 ug/ml | 26.5 `ug/mL` | 1.10 |
| MMS | LB/LB | 0.11 | 0.04 % (w/v) | 0.04 `percent_w/v` | 1.30 |
| Novobiocin | LB/LB | 104.1 | 35 ug/ml | 35.0 `ug/mL` | 1.28 |

Sulfamethizole is the only unit conversion: `ConcentrationUnit` has no mg/mL member and
0.2 mg/ml is 200 ug/ml, a conversion inside the paper's own mass-per-volume dimension.
The two deliberately deferred enum members (`ConcentrationUnit.mg_per_l`,
`MeasurementType.normalized_colony_size`) are not needed and were not added;
`torchcell/datamodels/schema.py` is untouched by this branch.

### What is NOT asserted

- **No inducer.** dCas9 sits under a tetracycline-inducible promoter (Figure S1A) and the
  paper names no inducer and no dose for the screen, so the environment carries none.
- **No duration.** Each condition was harvested at its own OD600 (1.10 to 2.00) with no
  wall time and no doubling count, so `duration_hours` and `duration_generations` are
  typed `not_reported_by_primary` gaps whose note carries that condition's sampling OD600.
- **No replicate TYPE.** The Methods say only "two replicate measurements", never whether a
  replicate is a separate culture or a re-sequencing of one, so `n_samples = 2` is stored
  and `sample_unit` is a typed gap.
- **No uncertainty.** Table S1 releases the two replicate abundances and the paper reports
  no dispersion of the ratio they define; a sample SD computed from two values by the
  loader would be our statistic, not a released one, so
  `environment_response_uncertainty` is a typed gap.
- **No dose rule.** Every treatment dose is 0.04 to 0.36 of its Table S3 lethal dose, a
  regularity the paper never states as a rule, so `DoseBasis.fixed` records the doses as
  the explicit numbers Table S3 prints. (Hypothesis, untested: most arms were set at about
  one third of the lethal dose; nine of the twelve ratios fall between 0.33 and 0.36.)
- **Six name-only compounds.** CCCP, polymyxin B, pyocyanin, rifampicin, puromycin and
  phleomycin have no row in the shared compound-identity table, so `resolved_compound`
  returns them with an `inchikey` gap. That is the honest shipped state and the Wang 2015
  precedent; adding identity rows is a separate, sourced act.

### Fidelity checks run at build time

All three stop the build on disagreement, and their results are written to
`preprocess/extraction.json`.

1. **`spacer_matches_the_released_window`.** For each of the 39,577 Table S1 guides Table
   S6 names, the Table S6 spacer is the first 20 nt of the 23 nt MG1655 window the row's
   Start and End delimit, in the orientation whose remaining 3 nt are an NGG PAM.
   **Measured: 19,183 on the reverse read, 20,394 on the forward read, 0 both ways, 0
   neither.** Spacer, coordinates and pinned assembly are therefore mutually consistent.
2. **`enrichment_ratio_reproduced`.** The median over a gene's Table S1 guides of
   (condition mean abundance / library mean abundance) against Table S2's released ER.
   **Measured: 4,192 genes x 13 conditions = 54,496 cells, max absolute difference
   5.0e-5**, which is Table S2's 4-decimal rounding. The six genes carrying a guide absent
   from the initial library (`sbp`, `nmpC`, `ybgK`, `yrhA`, `ftsX`, `yiiR`) are excluded
   from the CHECK, not from the dataset: that guide's ratio is infinite, which moves an
   even-length median, and the paper does not state how it handled it. Before the
   exclusion, `sbp` was the only gene over tolerance, in all 13 conditions (up to 0.293
   under phleomycin). The library columns are used for nothing else; the stored statistic
   does not divide by them.
3. **`library_coverage_reproduced`.** 39,580 Table S1 rows minus the 6 whose two library
   samples are both zero = **39,574**, the paper's own count.

### Identifier reconciliation

Table S1 names a guide's gene by symbol only; Table S2 is the released symbol-to-b-number
table and its 4,198 symbols are exactly Table S1's (asserted). Each released b-number goes
through `reconcile_locus_tags` against MG1655 `GCA_000005845.2`. No ECK crosswalk is
involved: the screen host IS MG1655, so the released b-numbers are already in the stored
`ecoli_k12_mg1655_bnumber` namespace.

| quantity | count |
|---|---|
| released b-numbers | 4,198 |
| resolved to one locus | 4,194 (0.999, above `MIN_RESOLVED_FRACTION = 0.99`) |
| resolver layer: locus tag | 4,191 |
| resolver layer: gene synonym (ECK) | 4 |
| resolver layer: not found | 3 |
| status `current` | 4,070 |
| status `non_gene_feature` (pseudogene loci resolving to themselves) | 124 |
| status `retired` | 3 (`b0017` insL1, `b4590` ybfK, `b4694` yagP) |
| status `ambiguous` | 1 (`b4659` yabP, matching both b0056 and b0057) |
| loci in the built gene set | 4,156 |
| released symbol is an older spelling of the annotation's | 213 (e.g. `yadD` -> `rpnC`, `ligT` -> `thpR`, `tsaA` -> `trmO`) |

`perturbed_gene_name` is the pinned annotation's own symbol for the stored locus when it
resolves back to it (`canonical_symbol`), so one locus carries one spelling and the
verifier's `canonical_gene_names` rule holds.

**Table S2's `gRNA #` disagrees with Table S1's row count for 10 genes**, in both
directions, written to `preprocess/identifier_reconciliation.json`:
`insH1` 38 vs 9, `ybiO` 22 vs 21, `ypdA` 16 vs 1, `insF1` 15 vs 9, `ydfJ` 12 vs 2,
`insD1` 10 vs 2, `insE1` 7 vs 3, `insB1` 3 vs 12, `insN` 3 vs 9, `insA` 2 vs 5. The
records store the Table S1 count, which is the one the ER reproduction confirms the
authors grouped on.

### Retention ledger

Rules apply in order; a guide appears under the first rule that removes it. Full items in
`preprocess/dropped_records.json`.

| # | rule | scope | guides | records |
|---|---|---|---|---|
| 1 | `b_number_is_not_in_the_mg1655_annotation` | guide | 17 (4 genes) | 204 |
| 2 | `guide_window_does_not_overlap_the_resolved_locus` | guide | 301 (66 genes) | 3,612 |
| 3 | `guide_strand_disagrees_with_the_resolved_locus` | guide | 12 (`lomR` 9, `insO` 3) | 144 |
| 4 | `guide_has_no_spacer_in_table_s6` | guide | 1 (`ypjA` 2778354-2778376) | 12 |
| 5 | `abundance_is_zero_in_the_untreated_lb_control` | guide | 283 | 3,396 |
| 6 | `abundance_is_zero_in_the_antibiotic` | cell | -- | 1,023 |

Arithmetic: `39,580 - 17 - 301 - 12 - 1 - 283 = 38,966` kept guides, and
`38,966 x 12 - 1,023 = 466,569` records against `39,580 x 12 = 474,960` source cells, so
8,391 records are dropped and the six rules account for `204 + 3,612 + 144 + 12 + 3,396 +
1,023 = 8,391`. `process` refuses to finish if that identity fails.

**Rule 2 is the interesting one.** The row's released Start-End window lies outside the
span of the locus its released b-number resolves to, so the stored gene identity would not
be supported by the stored coordinates. This is not an annotation-version difference: the
RefSeq `NC_000913.3` gene spans the library was designed against were parsed and are
identical to the GenBank ones for every one of the 318 flagged rows (0 rescued). Two
groups dominate.

- **The IS families**, whose released symbol does not name one locus: 58 guides under
  `insC1` (a 366 bp locus) and 38 under `insH1` (981 bp), which at one guide per 100 nt is
  arithmetically impossible. RefSeq calls `b1370` `insH5`, `b1578` `insD7`, `b0542` `renD`
  and `b0372` `insF2`, so even the released symbol and its own b-number disagree there.
- **The Qin prophage cluster b1548-b1578** (`ydfK` 13, `ycjV` 10, `rem` 10, `yehH` 9,
  `ypjC` 8, `ydfU` 7, `nohQ` 7, `ydfR` 6, plus `dicA`, `dicB`, `dicC`, `cspB`, `cspF`,
  `cspI`, `flxA`, `hokD`, `gnsB`, `essQ`, `intK` ... at 1 to 4 guides each), where the
  released symbol-to-b-number column and the released coordinate columns disagree by a
  shift.

Rule 6 is the honest handling of the strongest signal a pooled screen can show: both
replicates of one antibiotic read 0 for a guide whose LB control did not, so the log2
ratio is negative infinity. A pseudocount would fabricate the value, so that one record is
dropped and the guide's other eleven are kept.

### Raw mirror

`$DATA_ROOT/torchcell-raw/choeRapidIdentificationKey2025/data/` holds exactly the three
files the loader consumes, each with a `pmc_cloud` `RetrievalRecord` in `manifest.json`.
The route is scriptable (`torchcell.literature.retrieve.pmc_cloud_object`) and was
re-retrieved bit-identically on 2026-10-07; nothing here needed a manual recipe.

| file | bytes | sha256 |
|---|---|---|
| `data/mmc2.xlsx` (Table S1) | 9,020,670 | `7d1224ccf399fb5f89fb33915ee01680cfc13a98ef8df1aa94d87e646339a8cc` |
| `data/mmc3.xlsx` (Table S2) | 703,998 | `23eba0baea3e2d4e4041af6db61eaee6b425667d01597824c7aee564d416c681` |
| `data/mmc5.xlsx` (Table S6) | 1,999,427 | `4f8642f007e403fb2676942d07308482ba14c87836f4626dcf5effe9fabe59f9` |

Bucket prefix `https://pmc-oa-opendata.s3.amazonaws.com/PMC12063145.1/`. `si_expected`
records what was deliberately not mirrored: ENA PRJEB33267 (the raw reads), `mmc1.pdf`
(quoted through the literature mirror's OCR), `mmc4.xlsx` (Table S4, which is log2 of
Table S2 and would store the same measurement twice), `mmc6-8.zip` (the design scripts),
and the BW25113 KEIO knockout growth curves, which are a separate assay released only as
figures.

### Build numbers

`PYTHONPATH=$PWD python -m torchcell.datasets.ecoli.choe2025 build` on GilaHyper,
2026-10-07:

| quantity | value |
|---|---|
| records | 466,569 |
| guides | 38,966 |
| gene set (MG1655 loci) | 4,156 |
| references | 1 (the same clone in untreated LB, response 0) |
| distinct environments | 12 |
| wall time | 4 min 20 s (260 s) |
| `processed/lmdb` | 1.9 GB |
| `processed/interned` | 124 KB |
| build manifest | `fresh` |

The single reference and the twelve environments are interned, so a stored record is the
genotype plus the phenotype.

### Follow-ups, not done here

- The six name-only compounds (CCCP, polymyxin B, pyocyanin, rifampicin, puromycin,
  phleomycin) want sourced rows in the compound-identity table.
- The LB-over-library ratio, which is the knockdown's own fitness cost in LB and is a
  positive ratio with baseline 1, is a second dataset of a different record type
  (`BacterialFitnessExperiment`) over the same release. It is not served here because a
  loader emits one experiment class, and it is not the antibiotic-response readout this row
  is listed for.
- The BW25113 KEIO single-knockout area-under-curve arm (Figures S5 to S10, three
  biological replicates, mean +/- SD) is a separate assay on a different strain released
  only as figures.
- Rule 2's 318 flagged rows are a defect in the release's own symbol-to-coordinate
  bookkeeping, not in our mapping. They are recoverable if the guide's gene is reassigned
  from its coordinates rather than from its released symbol, which is a change of identity
  policy and wants an explicit decision.
