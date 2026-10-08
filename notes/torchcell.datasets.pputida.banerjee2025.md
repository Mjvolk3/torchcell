---
id: xlq6pv8yuvtmw2atp8em3kk
title: Banerjee2025
desc: ''
updated: 1791452189926
created: 1791452189926
---

## 2026.10.08 - Banerjee 2025 p-coumarate growth-coupling proteome loader

`torchcell/datasets/pputida/banerjee2025.py`, one dataset class
`ProteomeBanerjee2025Dataset` (`BacterialProteinAbundanceExperiment` /
`BacterialProteinAbundanceExperimentReference`), row 39 of the bacterial expansion
list. Template: [[torchcell.datasets.pputida.desiqueira2025]]. Skeleton:
[[torchcell.datasets.bacteria_common]].

Banerjee D, Menasalvas J, Chen Y, Gin JW, Baidoo EEK, Petzold CJ, Eng T, Mukhopadhyay
A. "Addressing genome scale design tradeoffs in Pseudomonas putida for bioconversion of
an aromatic carbon source", npj Syst. Biol. Appl. 2025, doi:10.1038/s41540-024-00480-z,
citation key `banerjeeAddressingGenomeScale2025`.

NOT to be confused with two siblings in the candidate table: "Banerjee 2024"
(doi:10.1016/j.ymben.2024.02.004, isoprenol, not mirrored) and "Banerjee 2020
indigoidine" (doi:10.1038/s41467-020-19171-4, not mirrored). Neither is in the fifty,
and a value needed from either is a deferral to an unmirrored paper.

### Pins

| artifact | where | sha256 |
|---|---|---|
| `paper.md` (MinerU 2.7.6 OCR) | `$DATA_ROOT/torchcell-library/banerjeeAddressingGenomeScale2025/` | `74d7040a0a7ec721f18e5fc49d855a5c9c6eaeee9ff7bf166ec675824e27c489` |
| `si/si1.md` (OCR of the SI PDF) | same | `25cc7bbffa0e0d2ad89fe0c2748cd1966940f16e8dd8a1662a101ec1b55362b4` |
| `si/si1.pdf` (publisher MOESM1) | same | `e59a55234b50d5f64f5e941baa4d58c0e1f8fe8bb9ffe248a7594b55becb8ebf` |
| `si/si3.zip` = Supplementary Data S2 (MOESM3) | same | `09648ccbac2f12bc89bd5ea1582837a7b6e1fa771c3eede4929dc666120f6d0f` |
| `si/si2.zip` = Supplementary Data S1 (MOESM2), the COBRA archive, NOT consumed | same | `cf9b5d1b7a33c6558d7cc54989f5e592fee51c0b624596a1b1c91be763d86b14` |

Raw mirror created at `$DATA_ROOT/torchcell-raw/banerjeeAddressingGenomeScale2025/`
with five members extracted from `si3.zip`, each with a `zip_member`-style retrieval
record carrying the archive's own sha256 as `container_sha256`:

| member | sha256 |
|---|---|
| `si/t-test_pJ_PP_0897_vs_Control_20230307-000644.xlsx` | `02bb64f4f6cfe49507430478c41b83e28d52374cc728d052f581821ee575d80e` |
| `si/t-test_0415pPP_0897_vs_Control_20230425-000301.xlsx` | `53e36bec86b8c97cea094501a0ed6bfb9eef21da9d5eff519ccb8d71b42d21ed` |
| `si/t-test_TEAM-2487_M9_pCA_alanine_malate_vs_TEAM-2370_M9_pCA_alanine_malate_20230626-224656.xlsx` | `3817e0501952f93f2459bfcf9cfca8bb4a4a1cfb52dd3674c97ab61c11f71ea4` |
| `si/t-test_TEAM-2487_M9_pCA_alanine_malate_vs_TEAM-2487_M9_alanine_malate_20230626-224656.xlsx` | `cc4a05f1e19074a99e25e87ba2d5ef6b5f95d3d2bce8177e7f24e28277006104` |
| `si/DataAnalysis.xlsx` (curated, read only as a build oracle) | `a888c57f06e47f3cf35b232607f2c36b4a2ecd7807b0109c84089038dcf405e1` |

The retrieval re-runs as-is: the archive came from the PMC Article Datasets bucket at
`https://pmc-oa-opendata.s3.amazonaws.com/PMC11732973.1/41540_2024_480_MOESM3_ESM.zip`.
PRIDE `PXD050285` holds the raw DIA files and is retrieval metadata only; no loader
consumes raw spectra.

### Measured record counts, and the correction to the row's estimate

The candidate row (`experiments/database/scripts/build_bacteria_candidate_datasets_table.py`,
`name="Banerjee 2025"`) carries `genotypes_n=12`, `env_n=3`, `instances_n=36` on an
`estimate` basis and `dim=2000` on a `reported` basis. Measured on the released files:

| quantity | row | measured | note |
|---|---|---|---|
| instances | 36 (estimate) | **6** | one record per released (strain, medium) proteome sample |
| genotypes | 12 promoter variants | **4 strains** | D1b_gf, two promoter variants, one deletion |
| environments | 3 | **3** | the row is right |
| dim | 2000 (reported) | **2,494 / 2,470** | the paper's "about 2,000 proteins quantified" is approximate; the workbooks hold more |

The 36 came from 12 x 3. There are not 12 promoter variants: the paper makes TWO
(`PJ23109-PP_0897` and `Ppp_0415-PP_0897`), and the released proteomes cover four
strains over three media, not a full cross. So **instances 36 -> 6** and
**genotypes 12 -> 4**, and `dim` is better recorded as the released protein count per
arm than as the text's round number. The row's prose about the PRIDE file layout
(26 raw runs = an 8-run promoter arm at n=4 plus an 18-run cross-feeding arm at n=3) is
confirmed by the released data: the promoter arm's t statistics back-solve to n=4 and
the cross-feeding arm's to n=3, and the cross-feeding arm's 6 (strain x medium) groups
times 3 replicates is the 18.

The released analysis covers 3 of the cross-feeding arm's 6 groups; the other 3 (the
`M9 alanine` plate and the 2370 arms of the two alanine media) were run but never
released as a table, so they are not records.

### The split: one dataset, two refusals, one blocked arm

The row's phenotype field bundles three quantities. They are different phenotype
classes, and `verify_protein_dataset`'s L3 `measurement_type_consistent` allows one
`measurement_type` per dataset, so the split was decided by what is RELEASED:

- **Proteome: one dataset class.** Six records, one `measurement_type`
  (`diann_lfq_log2_intensity_replicate_mean`). Both arms store the same kind of number
  (a log2 DIA-NN LFQ quantity, replicate mean), so one class is correct and a second
  would be a split with no schema reason.
- **Indigoidine titer: refused whole.** Figs. 2C-2E, Supplementary Figs. 2, 3A and 8
  plot it in mg/L, in g/g and as mg/CFU, and no table releases a number. Supplementary
  Data S1 is the COBRA model and flux distributions, Supplementary Data S2 is the
  proteomics, and Supplementary Tables 1-4 are strains, plasmids, oligos and the BIOLOG
  grid. `ProductTiterExperimentReference.phenotype_reference` is required and
  `ProductTiterPhenotype.titer` is a required float, so a titer family needs a released
  reference titer and there is none. Same precedent as Yunus 2026 and Kang 2026.
  (The mg/L -> `ug/mL` unit question never arises, because no titer is stored.)
- **Growth rate: refused whole.** Figs. 2B, 2F, 2G and Supplementary Figs. 1, 6, 7, 8
  are plate images and growth curves with no companion table. The only per-hour numbers
  in the paper (1.14/h, 0.43/h, 0.45/h) are flux-balance PREDICTIONS of the
  context-specific models. Had a signed growth readout been released it would have gone
  to `EnvironmentResponsePhenotype`, since `FitnessPhenotype` clamps non-positive
  values; nothing was released, so nothing is written.
- **BIOLOG grid: released, measured, blocked on one enum member.** Supplementary
  Table 4 releases a measured `OD595` for 95 PM1-plate carbon sources on
  `D1b_gf dPP_0897` beside the matched in-silico value, all 95 parsing as floats. The
  readout is a tetrazolium redox dye, not biomass, and the values are signed, so
  `EnvironmentResponsePhenotype` is the right shape -- but `MeasurementType` has no
  member for a respiration endpoint, and adding one changes the serialized closure of
  every served `EnvironmentResponsePhenotype` dataset, which forces a full
  knowledge-graph rebuild. Recorded as `BIOLOG_NOT_A_DATASET` and raised on issue #749.
  The PM2A plate's OD values were never released, only the narrative that dextrin and
  D,L-carnitine were the two extra positives.

### The promoter-variant modality HAS a perturbation leaf

`torchcell/datamodels/schema.py` already carries
`PromoterReplacementPerturbation(ExpressionModulationPerturbation)`: the gene stays
`state="present"`, its coding sequence is unedited, the mechanism is SO:1000032
`delins`, `expression_direction` is required with no default, and `crispr` is re-declared
optional so a promoter swap carries no guide. It is already in
`cell_adapter.BACTERIAL_PERTURBATION_LEAVES`, so it serves as a `bacterial perturbation`
node. **No issue was needed and none was opened**: the leaf exists, is served, and fits
this paper without bending. Both variants are written with
`expression_direction="decreased"`, which is sourced AND measured (below).

### Sourcing: every quote verbatim against the pinned bytes

Every statistic carries a `SourcedValue` whose quote is asserted to be a verbatim
substring of the sha256-pinned mirror by
`tests/torchcell/datasets/pputida/test_banerjee2025.py::test_every_module_quote_is_verbatim_in_the_pinned_mirrored_bytes`
(29 sourced values over `paper.md` and `si/si1.md`).

| fact | value | quote anchor |
|---|---|---|
| medium | modified M9, salts deferred to ref 21 | Methods, Cultivation |
| proteomics media | M9 60 mM p-CA; M9 50 mM p-CA + 70 mM D-alanine + 70 mM L-malate | Methods, Shotgun proteomics |
| supplemented doses, restated | 50 mM p-CA, 70 mM L-malate, 70 mM D-alanine | Results, Metabolite supplementation |
| harvest | mid-log, OD600 0.8-1 | Methods, Shotgun proteomics |
| instrument | Orbitrap Exploris 480 | Methods, Shotgun proteomics |
| search | DIA-NN library-free, global FDR 0.01, LFQ | Methods, Shotgun proteomics |
| replicates, cross-feeding arm | 3 ("grown in triplicates") | Methods, Shotgun proteomics |
| replicates, promoter arm | 4 ("four biological replicates", "(n=4) in B") | Supplementary Figure 4 caption |
| promoter parts | pJ23109 (Anderson collection), PP_0415 promoter | Results |
| direction | decreased ("strongly reduced ~8 fold") | Results |
| cutset | PP_1378 + PP_0897 + fumC1/PP_0944 + fumC2/PP_1755 | Results |
| chassis | D1b_gf, with ΔfleQ named as PP_4373 | Results + Supplementary Table 1 |
| aerobicity | aerobic | Methods, Constraint-based modeling |

### Three things the released numbers settled that no quote states

**1. The replicate count differs by arm, and the t statistic decides.** The Methods say
"grown in triplicates"; Supplementary Figure 4 says "four biological replicates". Both
are true of different arms. Back-solving
`t = (m1 - m2) / sqrt((s1^2 + s2^2) / n)` from the released means, SDs and t:

| workbook | rows back-solved | n |
|---|---|---|
| `t-test_pJ_PP_0897_vs_Control` | 2,494 | **4** exactly |
| `t-test_0415pPP_0897_vs_Control` | 2,494 | **4** exactly |
| `t-test_TEAM-2487..._vs_TEAM-2370...` | 2,470 | **3** exactly |
| `t-test_TEAM-2487...PAM_vs...AM` | 2,470 | **3** exactly |

Machine precision over the first 400 rows (mean absolute deviation 2e-14 and 4e-14);
every other n in 2..6 is off by 0.16 to 2.9. The ONE exception is asserted rather than
excused: the two merged-accession labels `Pyrc` and `Ubid`, each filed twice under one
name with two protein groups, back-solve to exactly **2n** (8 and 6), because the
analysis pooled both groups' replicates. Those are exactly the two labels the merge rule
drops, so no stored value comes from a row whose replicate count does not hold.
`assert_replicates_back_solve` re-runs this over every released row at build time.

**2. Which group is which strain.** Neither the paper nor the SI maps `2370` or `2487`
onto Supplementary Table 1. PP_0897's own released fold changes settle it:

| workbook | PP_0897 log2 FC | p | reading |
|---|---|---|---|
| pJ vs control | **-4.042** | 7.8e-5 | titration |
| 0415 vs control | **-3.283** | 1.1e-5 | titration |
| 2487 vs 2370 in PAM | **-7.928** | 5.9e-4 | deletion (effectively absent) |
| 2487 PAM vs 2487 AM | **+0.048** | 0.83 | flat, which only a deleted gene can be |

So `2370` = D1b_gf and `2487` = D1b_gf ΔPP_0897, and the two titrations are the promoter
variants. Independently, the significance-sheet row counts reproduce the Results text
EXACTLY, which is what assigns each workbook its promoter: the paper says "only 28
metabolic and 63 non-metabolic proteins ... in the PJ23109 strain and ... only 19
metabolic and 37 non-metabolic ... in the PPP_0415 strain", and the sheets hold
20 + 71 = 91 = 28 + 63 and 24 + 32 = 56 = 19 + 37. The curated
`Promoter STrainsAnalysis` sheet holds 134 = 91 + 56 - 13 rows, with exactly 13 carrying
both arms' fold changes at matching sign, reproducing "Only 13 proteins showed matching
amplitudes of varying abundance (i.e., both up or down)".

**3. The record count is six, not eight.** `log2_mean_2370` / `log2_std_2370` appear in
BOTH promoter workbooks and are equal cell for cell (0 of 2,494 differ), and
`log2_mean_2487_M9_pCA_alanine_malate` likewise across both cross-feeding workbooks. So
each is ONE sample, asserted by `assert_shared_groups_agree`.

### The one derived step in the key

The released `Protein` column is a UniProt gene label: a bare locus tag for 1,376-1,381
proteins and a title-cased symbol otherwise. **The four workbooks disagree on its CASE**:
one cross-feeding workbook writes `PP_0002` where the other writes `Pp_0002`, for 1,376
labels, with no other difference. `canonical_label` upper-cases a locus-tag-shaped label
and leaves everything else verbatim, which is what lets the two workbooks be joined and
also keeps six collision-held tags inside the namespace.

Reconciliation against `pputida_KT2440_ASM756v2` over the union of all four workbooks:
2,763 distinct labels, 1,586 locus tags, 916 gene symbols, 261 resolving through no
layer; 270 end up outside the namespace and 2 more drop as merged accessions, leaving
**2,491** candidates. Resolved fraction 0.905 (promoter) and 0.904 (cross-feeding);
`MIN_RESOLVED_FRACTION = 0.90`. The drop list goes to
`preprocess/dropped_protein_labels.csv` every build; the UniProt crosswalk that would
recover the host proteins is raised on issue #753.

### What is NOT typed, and why

The chassis also carries a chromosomally integrated indigoidine cassette, written
`PP_5402-intergenic::PBAD-Sc.bpsA,Bc.sfp,Ps.glnA`. `GeneAdditionPerturbation` requires
`source_organism` with no default, and this article and its SI never expand `Sc.`, `Bc.`
or `Ps.` ("Bacillus subtilis" occurs only in an unrelated reference title). The organisms
are stated in Banerjee 2020 and in reference 21, neither of which is mirrored, so
expanding them is a deferral to an unmirrored source. The cassette is kept verbatim in
`BacterialStrainBackground.genotype_statement` and the three genes are NOT written as
`HeterologousPathwayPerturbation`.

Temperature is a typed `ProvenanceGap` on every record: the shotgun-proteomics Methods
state the medium, the vessel, the inoculum and the harvest state but no incubation
temperature. Every temperature the paper does state elsewhere is 30 C, which is why the
absence is recorded rather than borrowed. The `M9 alanine malate` medium is named only by
a released column header and never described, so both its substrates carry a gap on
`magnitude` rather than borrowing the 70 mM doses stated for the p-CA-containing medium.

### Media library

`M9_DEFERRED_BANERJEE2025` in `torchcell/datamodels/media.py` is the one bacterial medium
in the library whose SALTS are a deferred composition rather than a stated recipe:
"Engineered strains were grown in a modified M9 minimal medium as previously
described21", and reference 21 (Eng et al. 2023, Cell Rep. 42, 113087, which the SI cites
as "Eng andBanerjee,2023") is not mirrored. It is its OWN base rather than a derivative
of the stated M9 salts, because a derivative must restate or drop every base component
and this medium can do neither without asserting amounts the source withheld. It is
carbon-free; the loader carries each condition's p-CA, D-alanine and L-malate as
`EnvironmentPhysicalPerturbation(factor=carbon_source)`, the de Siqueira convention.
The production runs add 30 mM MOPS at pH 7 and 1.5% w/v L-arabinose on top of the same
base; no medium object carries them, because this paper releases no titer and no growth
rate, so nothing is served from a production run.

### Build and verification

Dev store: `$DATA_ROOT/data/torchcell/proteome_banerjee2025`, 6 records over 13,413
stored abundances. `python torchcell/datasets/pputida/banerjee2025.py` builds and
verifies. All twelve rows PASS:

| level | rule | result |
|---|---|---|
| L0 | structural | 6 records validated |
| L1 | count | observed 6, expected 6 |
| L1 | orf_uniqueness | 1 ORF, 1 with multiple strains (expected) |
| L1 | group_uniqueness | 6 distinct (strain edit, carbon regime) samples |
| L2 | value_fidelity | 13,413 values checked |
| L2 | se_nonnegative | 13,413 values checked |
| L2 | replicate_split | every stored n is 3 or 4 |
| L3 | reference_finite | finite + key-matched for all 13,413 |
| L3 | measurement_type_consistent | single `diann_lfq_log2_intensity_replicate_mean` |
| L3 | assembly_pin | all 6 pin `pputida_KT2440_ASM756v2` / `D1b_gf` |
| L4 | gene_containment_kt2440 | all 2,493 stored identifiers are loci |
| L4 | stored_scale_is_the_released_log2_mean | all 13,413 re-join a released cell |

Adapter: [[torchcell.adapters.banerjee2025_proteome_adapter]]. No new Biolink graph
class was declared: `protein abundance phenotype` and `bacterial perturbation` already
exist in `biocypher/config/torchcell_schema_config.yaml`, and
`PromoterReplacementPerturbation` is already in `BACTERIAL_PERTURBATION_LEAVES`.

### Follow-ups filed

- #749 -- `MeasurementType` has no respiration-endpoint member, which blocks the 95-row
  BIOLOG grid.
- #726 -- `D-alanine` and `L-malate` have no compound-identity row (and D-alanine is NOT
  mapped onto the L-alanine row; the enantiomer is load-bearing here).
- #753 -- the UniProt-to-locus-tag crosswalk, measured at 270 dropped labels.
