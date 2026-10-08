---
id: gatc6vn78pwcb9t2om8y7i5
title: Mori2021
desc: ''
updated: 1791424568696
created: 1791424568696
---

## 2026.10.07 - Loader, sourcing, Schmidt overlap and the build

Row 34 of the fifty bacterial datasets. Loader
`torchcell/datasets/ecoli/mori2021.py`, adapter
`torchcell/adapters/mori2021_adapter.py`, conf
`torchcell/adapters/conf/proteome_mori2021_adapter.yaml`, tests
`tests/torchcell/datasets/ecoli/test_mori2021.py` and
`tests/torchcell/adapters/test_mori2021_adapter.py`.

Mori et al. 2021, *Mol Syst Biol* 17:e9536, doi `10.15252/msb.20209536`, PMID 34032011,
PMC8144880, citation key `moriCoarseFineAbsolute2021`.

### Which per-protein quantity is stored

The release carries several per-protein quantities and they are not interchangeable:
ribosome-profiling synthesis mass fractions from Li 2014, xTop / TopPep1 / TopPep3 /
iBAQ mass fractions for the seven calibration samples (Dataset EV6), the
ribosome-profiling-corrected absolute mass fractions (Datasets EV8 and EV9), the
per-limitation response slopes (Dataset EV11) and the peptide-precursor intensities
(Datasets EV4 and EV5).

The loader stores the **absolute protein mass fraction** of Dataset EV9
(`si10.xlsx`, sheet `EV9-AbsoluteMassFractions-2`). `measurement_type` is
`absolute_protein_mass_fraction_xtop_dia_swath_riboprofiling_scaled`.

Verbatim definition, from the pinned `paper.md`:

> The resulting absolute protein abundances are expressed in "protein mass fractions",
> i.e., mass of a given protein over the total mass of all detected proteins, which can
> readily be converted to cellular protein concentration (Appendix Note S1).

And the unit the paper refuses, in the same paragraph:

> Note that the frequently used absolute unit "protein copies/cell" is avoided here, as
> cell size is highly variable across growth conditions

So the stored number is dimensionless and normalized per sample. That is asserted against
the bytes rather than read off the description: `check_normalization` sums every one of
the 30 EV8 columns and the 36 EV9 columns and refuses a deviation from 1 above
`NORMALIZATION_ATOL` = 1e-7. Measured on the pinned workbooks, the worst absolute
deviation over all 66 columns is below 3e-8.

Dataset EV9's own description, pinned as a workbook quote:

> Absolute protein mass fractions computed from xTop protein intensities and corrected
> with ribosome profiling synthesis rates for the samples described in Dataset EV3
> (Samples-2).

### Sourcing table

Every quote below is a substring of the pinned bytes. `paper.md` quotes are checked by
the test module against the literature mirror; Appendix and workbook quotes are checked
inside `process()` by `check_quotes`, so the build stops if a re-export moves them.

| value | source | sha256 (first 16) | verbatim quote |
|---|---|---|---|
| `REFERENCE_STRAIN` = MG1655 | `si1.docx` (Appendix, Extended Experimental Methods) | `3c9490d042c152bd` | "For the calibration samples A1, C1 and F1, we used strain EQ353, which is the specific MG1655 strain used in Li et al. (Li et al, 2014) 2014." |
| strain label `MG1655 (EQ353)` | `si2.xlsx` (Dataset EV1, `EV1-Strains`) | `78b8629a8e13229d` | "MG1655 (EQ353) \| Wild type E. coli strain - same strain used in Li et al. (2014) \| Originarily obtained from Carol Gross Lab" |
| stored quantity + unit | `paper.md` | `c86b36d90657aceb` | the mass-fraction definition above |
| abundance model | `si10.xlsx` (Dataset EV9) | `50efbf4a3c7f9c1b` | the EV9 description above |
| quantification workflow | `paper.md` | `c86b36d90657aceb` | "we developed a versatile mass spectrometric workflow based on data-independent acquisition proteomics (DIA/SWATH) together with a novel protein inference algorithm (xTop)" |
| 66 released samples | `paper.md` | `c86b36d90657aceb` | "The workflow outlined above allowed us to analyze E. coli proteomes over 66 different samples representing an array of different treatments, strains, and growth conditions." |
| 2,335 detected proteins | `paper.md` | `c86b36d90657aceb` | "A total of 2,335 proteins were detected from 66 samples across these conditions." |
| base media (the drop rules) | `si1.docx` | `3c9490d042c152bd` | "Unless otherwise indicated, growth media used are based on one of the following base media: modified Record's MOPS medium (Chen et al., 1990), phosphate-buffered "N-C-" medium (Csonka et al, 1994; Gutnick et al, 1969), M9 medium (Kochanowski et al, 2013), and Luria-Bertani (LB) medium." |
| temperature 37 C, aerobic | `si1.docx` | `3c9490d042c152bd` | "Batch cultures were grown in a 37°C water bath shaker shaking at 250 rpm for aeration." |
| Neidhardt MOPS NaCl 50 mM | `si1.docx` | `3c9490d042c152bd` | "This medium is Neidhardt's MOPS minimal medium (Neidhardt et al, 1974), but with 1/4 of the stated MOPS buffer (10 mM final concentration); further, NaCl concentration was adjusted to a final concentration of 10 mM (instead of 50 mM)." |
| the calibration sample is the reference | `si1.docx` | `3c9490d042c152bd` | "Exclusively to the three biological replicates of the calibration sample (E. coli strain K-12 MG1655 grown in glucose minimal media at exponential growth phase) a set of 29 stable isotope labeled peptides (AQUA peptides (Gerber et al, 2003)) was spiked after digestion and before C18 purification." |
| the calibration samples match Li 2014 | `si1.docx` | `3c9490d042c152bd` | "For the A1, C1 and F1 samples, both strain, growth media and experimental procedures are identical to those used in a ribosome profiling study (Li et al., 2014) to measure protein synthesis rates for E. coli in glucose-limited media." |
| Schmidt comparison | `paper.md` | `c86b36d90657aceb` | "We compared the latter to our results for MG1655 (EQ353) in glucose match culture. While the number of detected proteins were comparable (1,901 vs 1,843), we observed a systematic underestimation of low-abundant proteins" |

The NaCl line is the one worth pausing on. The library's `MOPS_MINIMAL` states 50 mM NaCl
and 9.5 mM NH4Cl for Neidhardt 1974's MOPS minimal without a carbon source. Mori states
its biofilm medium as a *departure* from that formulation, and in doing so prints 50 mM as
the unmodified amount. Dataset EV3 then releases "9.5 mM NH4Cl" as the calibration samples'
nitrogen source, which is the object's own ammonium component. Both halves of the library
object are therefore corroborated by the paper, not assumed.

### Raw mirror

Six files, all from the PMC OA Cloud bucket (`PMC8144880.1`), all scriptable through
`torchcell.literature.retrieve.pmc_cloud_object`, deposited under
`$DATA_ROOT/torchcell-raw/moriCoarseFineAbsolute2021/data/` with a per-file
`RetrievalRecord` (method `pmc_cloud`, the exact key, sha256, `retrieved_at`). Nothing was
retrieved by hand and no source needed a manual recipe.

| file | dataset | sha256 (first 16) | bytes | why it is consumed |
|---|---|---|---|---|
| `si1.docx` | Appendix | `3c9490d042c152bd` | 18,407,364 | the only statement of the base media, the culture temperature, EQ353 and the calibration sample |
| `si2.xlsx` | EV1 strains | `78b8629a8e13229d` | 13,639 | the EQ353 stock row |
| `si3.xlsx` | EV2 Samples-1 | `2bb6c8c3b647740c` | 19,326 | the strain and medium of the 30 EV8 columns |
| `si4.xlsx` | EV3 Samples-2 | `3c13d83bc11996c0` | 17,652 | the strain, medium, carbon, nitrogen and growth rate of the 36 EV9 columns |
| `si9.xlsx` | EV8 mass fractions 1 | `15795fc3405735fd` | 1,094,736 | the 30 column headers that are the Samples-1 half of the 66, and the identity block that pins the EV9 fault |
| `si10.xlsx` | EV9 mass fractions 2 | `50efbf4a3c7f9c1b` | 1,284,400 | the stored values |

Not mirrored, with why, in `NOT_MIRRORED`: SWATHAtlas PASS01421 (spectral library),
Panorama Public (AQUA Skyline documents), Datasets EV4/EV5 (peptide intensities),
EV6/EV7/EV10/EV11/EV12 (the other per-protein quantities) and the review process file.
Mendeley Data is not involved anywhere in this release.

### Identifier route, with counts

The records are MG1655, so the release's `Gene locus` column is the records' **own**
namespace rather than a foreign identifier. That is the opposite of the Schmidt 2016
situation, where the released b-number was an MG1655 identifier on BW25113 records and had
to be abandoned.

Measured with `reconcile_locus_tags` against the pinned MG1655 GenBank annotation:

| route | unique names | resolved | fraction | layers | left outside |
|---|---|---|---|---|---|
| released `Gene locus` b-number (**stored**) | 2,073 | 2,073 | 1.0000 | locus tag 2,070, gene synonym 3 | none |
| released `Gene name` symbol | 2,077 | 2,072 | 0.9976 | gene symbol 2,011, gene synonym 62, not found 4 | `gapC_1`, `ilvG_1`, `rdoA`, `yedS_1` (retired), `rffT` (ambiguous: `b3793` / `b4481`) |

The b-number route is stored: it resolves completely, has no ambiguity and no collision,
and 2,070 of 2,073 need no remapping at all. Statuses: 2,065 current, 8 non-gene feature.
The `Gene name` symbol is kept in `preprocess/protein_identifiers.csv` as the source's
cross-reference.

### The Schmidt 2016 overlap, measured

Both papers are absolute *E. coli* proteomes from Heinemann-adjacent groups and Mori reused
Schmidt's sample-preparation protocol ("The proteomic sample preparation was performed
using an optimized E. coli protocol described previously by Schmidt et al. (2016)"), so the
two releases were checked against each other before loading both.

**There is no overlap to resolve.** Mori's loaded records are MG1655 (EQ353) in Neidhardt
MOPS glucose; the landed Schmidt 2016 records are BW25113 in M9. No loaded
(protein, strain, condition) triple is shared, and the two store different quantities
(mass fraction against copies per cell).

The nearest pair was compared anyway, on UniProt accession, with Schmidt's copies/cell
converted to a mass fraction through its own released molecular weights
(`test_mori_does_not_republish_the_schmidt_2016_proteome`):

| measurement | value |
|---|---|
| Mori `A1-1` proteins quantified with a UniProt accession | 1,901 |
| Schmidt `Glucose` proteins with copies/cell > 0 | 2,355 |
| shared UniProt accessions | 1,812 |
| Pearson r on log10 mass fraction | 0.796 |
| Spearman rho | 0.832 |
| pairs agreeing to 1e-6 relative | **0 of 1,812** |
| Mori / Schmidt ratio, median | 1.61 |
| Mori / Schmidt ratio, 5th and 95th percentile | 0.44 and 29.9 |

Two things are worth naming. The 1,901 is the paper's own figure for this comparison
("the number of detected proteins were comparable (1,901 vs 1,843)"), which anchors
`A1-1` as the sample the paper compared. And the ratio's long upper tail is the direction
the paper reports: "we observed a systematic underestimation of low-abundant proteins".
Zero agreeing pairs is the duplication test, and it is negative: these are two
independent measurements. Contrast Rousset 2018, whose growth screen correlated with
Cui 2018's at r = 1.0000 and was therefore not stored twice.

### Retention ledger, with arithmetic

`preprocess/dropped_records.json` carries both ledgers and `DropLog.check()` refuses one
whose rules do not account for every drop.

**Samples: 66 - 52 - 2 - 5 = 7.** The 66 released mass-fraction columns are 30 of Dataset
EV8 plus 36 of Dataset EV9, which is the paper's own "66 different samples". Every drop is
a medium, and `media.py` is a value-surface file this branch does not edit.

| rule | n | items | needed addition |
|---|---|---|---|
| `medium_has_no_media_library_entry` | 52 | modified Record's MOPS (19), M9 as Kochanowski 2013 states it (23), phosphate-buffered N-C- (7), the anaerobic phosphate-buffered medium (1), the low-osmolarity MOPS of the biofilm and maltose-pyruvate samples (2) | `MOPS_RECORDS_MORI2021`, `M9_MORI2021`, `NC_MINUS_MORI2021`, `NC_MINUS_ANAEROBIC_MORI2021`, `MOPS_LOW_OSMOLARITY_MORI2021` |
| `medium_differs_from_the_library_entry_it_names` | 2 | `Lib-28`, `Lib-30`: they name "MOPS (Neidhardt's)" but release 20 mM NH4Cl where `MOPS_MINIMAL` states 9.5 mM | `MOPS_MINIMAL_20MM_NH4CL_MORI2021` |
| `medium_formulation_not_stated_by_the_source` | 5 | `Lib-12`, `Lib-19`, `Lib-20`, `Lib-21`, `Lib-22`: the Appendix names "Luria-Bertani (LB) medium" with no amounts, and the library holds both `LB` (Miller, 10 g/L NaCl) and `LB_LENNOX` (5 g/L NaCl) | a sourced decision on which LB Mori weighed |
| **loaded** | **7** | `A1-1`, `A1-2`, `A1-3`, `C1`, `F1-1`, `F1-2`, `F1-3` | |

The nitrogen check is not decorative: `check_sample_metadata` refuses a loaded sample
whose released nitrogen source is anything other than `9.5 mM NH4Cl`, which is exactly the
difference that put `Lib-28` and `Lib-30` in their own rule.

**Protein rows: 4,342 - 2,265 - 4 - 0 = 2,073.**

| rule | n | note |
|---|---|---|
| `not_quantified_in_any_loaded_sample` | 2,265 | the release writes 0 in all seven loaded columns. A released 0 is "not quantified in this sample", not a measured zero: each column sums to 1 over its non-zero rows |
| `release_files_no_gene_locus_for_this_row` | 4 | `ygaU`, `yghZ`, `yifE`, `ymfP`: Dataset EV9 leaves `Gene locus` blank |
| `b_number_resolves_to_no_locus_of_the_pinned_assembly` | 0 | every released b-number resolves |

Per-record keys are that column's own quantified subset: `A1-1` 1,897, `A1-2` 1,879,
`A1-3` 1,911, `C1` 1,930, `F1-1` 1,896, `F1-2` 1,931, `F1-3` 1,914. The reference
quantifies all 2,073 and is restricted to each record's keys.

### An identity fault in the released tables, found and not consumed

Datasets EV8 and EV9 hold the same 4,342 rows in the same `Gene name` order, but their
identity blocks disagree. EV9 leaves `Gene locus` blank for 30 rows; for 12 of them EV8
supplies a b-number (`ybaO` b0447, `ybbB` b0503, `ydfP` b1553, `yfaW` b2247, `yfhG` b2555,
`ygaU` b2665, `yghZ` b3001, `yhfG` b3362, `yifE` b3764, `ykgN` b4505, `ymfO` b1151,
`ymfP` b1152). The other 18 are the IS-element rows (`insAB-*`, `insCD-*`, `insEF-*`,
`insJK`), which neither sheet identifies.

The loader keys on EV9's own identity block and deliberately does **not** read EV8's as a
substitute, so the four dropped rows stay dropped. `check_identity_fault` asserts the
exact supplied set and the 18 at build time, so the finding stays pinned to the bytes
rather than to this note.

### Record granularity and the reference

The seven loaded samples are three biological replicates of one culture condition, two of
which were injected three times: `A1-1` .. `A1-3` are injections of culture `A1`,
`F1-1` .. `F1-3` of culture `F1`, and `C1` is a single injection ("Biological replicate of
A1"). One record per released column, values verbatim, nothing averaged. `n_replicates` is
1 for every protein of every record and `protein_abundance_se` is `None`: one LC-MS
injection has no replicate, and the spread of a culture's injections is a property of the
culture, not of one injection.

`phenotype_reference` is the calibration sample's own profile, which is what the release
designates as the absolute-scale anchor (the AQUA quote above). Per protein it is the mean
of the three culture means, a culture's mean being over its own quantified injections;
`n_replicates` is how many of the three cultures quantified the protein and the SE is
`stdev(culture means) / sqrt(n)`, exact arithmetic on the released columns rather than a
released summary statistic. A protein only one culture quantified stores `nan`.

Measured on the pinned workbook, over the 1,719 proteins all seven columns quantify, the
reference relative SE is 0.031 at the median, 0.122 at the 95th percentile and 0.918 at
the maximum.

All seven records carry one environment (`MOPS_MINIMAL` + a 0.2% w/v glucose
`EnvironmentPhysicalPerturbation`, 37 C, aerobic) and one empty wild-type genotype. The
Schmidt-style environment-uniqueness rule is therefore NOT applied: these are seven
measurements of one condition, not seven conditions. The released growth rate (0.69 to
0.73 per hour) has no `Environment` slot and lives in `preprocess/samples.csv`.

`EQ353` is recorded as a `BacterialStrainBackground` on the MG1655 assembly rather than
flattened into bare MG1655, because this paper's own Figure 9 result is that the stock
matters: "different laboratory strains of E. coli, even those labeled as common MG1655
strain, exhibit important biological differences". `alleles` is empty, which is what the
release's "Wild type E. coli strain" states.

### Build numbers

```
BUILT ProteomeMori2021Dataset: 7 records at
  /scratch/projects/torchcell-scratch/data/torchcell/proteome_mori2021 in 12s;
  gene_set size 2073; references 7
```

- records 7, gene-set size 2,073, references 7, wall time 12 s
- stored values 13,358 across the seven records
- `python -m torchcell.provenance.build_manifest` reports
  `dataset_name='proteome_mori2021' status='fresh' drift=[]`
- the previous store was retired with `scripts/deprecate.sh` for both `processed/` and
  `preprocess/` before the rebuild, into
  `/scratch/projects/torchcell-deprecated/2026-10-07_205458__{processed,preprocess}/`

### L0 to L4 verification

`verify_build` adds four SUPPLEMENTARY rows and the host-aware L4 containment on top of
the shared `verify_protein_dataset` gate. `proteome_mori2021: PASS`, all twelve rows ok:

| level | rule | result |
|---|---|---|
| L0 | structural | 7 records validated |
| L1 | count | observed 7, expected 7 |
| L1 | orf_uniqueness | 0 knocked-out ORFs (every record is wild type) |
| L1 | strain_label | `['MG1655 (EQ353)']` |
| L2 | value_fidelity | 13,358 values checked |
| L2 | se_nonnegative | 0 values checked (records carry no SE) |
| L2 | one_injection_per_record | 0 of 13,358 disagree |
| L3 | reference_finite | finite and key-matched for all 13,358 |
| L3 | measurement_type_consistent | one type |
| L3 | assembly_pin | `('ecoli_K12_MG1655_ASM584v2', 'GCA_000005845.2')` |
| L3 | reference_se_matches_culture_count | 0 of 13,358 disagree |
| L4 | gene_containment_mg1655 | 2,073 keys, 0 outside the MG1655 locus universe |

### Could not be sourced

Nothing the loader stores is unsourced. Two observations about the release are recorded
rather than resolved:

1. The paper's headline "A total of 2,335 proteins were detected from 66 samples" does not
   reproduce as a plain union of the non-zero rows: the union over all 66 released columns
   is 2,428 `Gene name` rows (2,372 in EV8, 2,235 in EV9, 2,179 in both). The paper states
   no filter for the 2,335, so the number is carried as a sourced value and is NOT used as
   a build assertion. Hypothesis (untested): 2,335 counts distinct protein groups after
   whatever quantitation filter Dataset EV6/EV11 apply, rather than distinct released rows.
2. The four blank-locus rows the loader drops are identified in Dataset EV8 but not in
   Dataset EV9 (above). Using EV8's column would be a cross-sheet fallback, so it is not
   done; a future revision could key both sheets off one identity block if a sourced
   reason to prefer EV8's is found.
