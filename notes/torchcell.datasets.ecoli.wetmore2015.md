---
id: ydmkv84j6lujkpj9fxfxa51
title: Wetmore2015
desc: ''
updated: 1791382414929
created: 1791382414929
---

## 2026.10.07 - Subsumed by the Price 2018 compendium: a provenance record, not a loader

Module: `torchcell/datasets/ecoli/wetmore2015.py`. Tests:
`tests/torchcell/datasets/ecoli/test_wetmore2015.py`. Raw mirror:
`$DATA_ROOT/torchcell-raw/wetmoreRapidQuantificationMutant2015/`. Plan:
[[plan.bacteria-ontology-genome]] section 4, checklist item 7. Shared API:
[[torchcell.datasets.bacteria_common]].

Wetmore et al. 2015 (mBio, doi:10.1128/mBio.00306-15) is the RB-TnSeq method paper,
rank 2 of the fifty bacterial rows. Its E. coli arm is the BW25113 Tn5 library KEIO_ML9,
grown in 101 condition samples, 92 of them successful by the paper's quality rules.

### Decision

**This row is SUBSUMED by the Price 2018 E. coli compendium (rank 21,
`priceMutantPhenotypesThousands2018`, Fitness Browser `orgId` `Keio`) and is NOT loaded.**
The module registers no dataset (a test pins that), so there is no LMDB, no
`len(dataset)` and no verifier run for this row. It holds three things instead: the
provenance record naming which compendium samples this paper first reported
(`subsumption_record()`), the RB-TnSeq statistics every transposon row defers to, as
`SourcedValue` constants, and the identifier finding the compendium loader needs.

Rows that de-duplicate against this decision (Mutalik 2020, rank 3, and every other
`Keio_ML9` row): the compendium's E. coli experiment set is exactly the 162 `Keio` rows
of Price 2018 Supplementary Table 5 (`si/si3.xlsx`, sheet `TableS5_Experiments`), which
equal the 162 columns of the compendium release's `fit_logratios_good.tab` (statistics
version 1.0.3). 92 of those 162 are this paper's samples; the other 70 are not.

### Evidence

Every number below is printed by
`python -m torchcell.datasets.ecoli.wetmore2015 report` from sha256-pinned files (Data
Set S1 and Table S5 in the literature mirror, the release tables in the raw mirror), and
each is asserted by a `--data` test.

| measured | value |
|---|---|
| condition samples in Data Set S1 `Expts_Keio` (time-zero rows excluded) | 101 |
| successful by this paper (`Expt_Quality_Keio` `u`) | 92 |
| of this paper's samples, carried by the compendium (Table S5 `Keio`) | 92 |
| successful AND carried | 91 |
| successful, NOT carried | 1 (`set1IT029`, fumarate) |
| failed here, carried there | 1 (`set2IT045`, LB) |
| compendium `Keio` samples that are not this paper's | 70 of 162 |
| carried samples whose condition the authors withdrew in 2021 | 4 (`set1IT007`, `set1IT008`, `set1IT043`, `set1IT044`) |
| shared samples compared value by value | 91 |
| this paper's genes / compendium genes / this paper's genes in the compendium | 3,646 / 3,789 / 3,646 |
| per-sample Pearson r between the releases, min / median | 0.9890 / 0.9971 |
| largest single-gene difference between the releases | 2.57 (log2 units) |

For all 92 carried samples, `short`, `Media`, `Condition_1` and `Concentration_1` agree
between Data Set S1 and Table S5 (`SHARED_METADATA`); `build_subsumption` refuses a
shared name whose metadata differ, so a matching name is checked to be the same sample
rather than assumed.

**The two verdict flips**, each traced to the rule it trips (`QualityMetrics.failed_rules`
against `SUCCESS_RULES`):

- `set1IT029` (fumarate): gMed 50.0 under 1.0.0, exactly at the gMed >= 50 rule, so it
  passed; 48 under the compendium's 1.0.3, so it fails. Its replicate `set1IT030` fails
  gMed in both releases, so **the compendium carries no E. coli fumarate sample**. Not
  loading this release therefore loses one condition (3,646 gene values at 1.0.0). That
  loss is accepted on purpose: the same authors' later analysis rejects the sample, and
  holding one 1.0.0 sample beside 162 at 1.0.3 would mix processing versions inside one
  library.
- `set2IT045` (LB): cor12 0.0976 under 1.0.0 fails the cor12 >= 0.1 rule; 0.109 under
  1.0.3 passes, and the compendium carries it. It is still this paper's sample, so the
  provenance record names it.

**The 2021 withdrawal.** The compendium page (`data/bigfit/index.html`) carries "Please
disregard any of the data from this publication regarding sucrose or D-mannitol." Two
facts measured on THIS paper's own release show its samples are the ones meant, rather
than assuming it: in `set1IT043` and `set1IT044` (D-mannitol) mtlA and mtlD fitness is
within 0.2 of zero while manX, manY and manZ are -2.1 to -4.8, the exact symptom the note
describes; and `set1IT007`/`set1IT008` (sucrose) grew to an end OD600 of 1.26 and 1.24
(Data Set S1 `EndOD`) although the note states BW25113 should not grow on sucrose. The
compendium release still contains all four, so its loader must drop or flag them;
`disregarded_by_authors` names them.

### What the compendium loader takes from this record

- **Provenance.** Each of the 92 carried samples gets "first reported in Wetmore 2015"
  from `subsumption_record().carried`; the 70 `superset_only` samples do not.
- **Withdrawn samples.** `subsumption_record().disregarded` (four names). Other
  compendium sucrose or mannitol samples, if any, are the compendium loader's to check.
- **Identifiers.** The release's gene table names MG1655 b-numbers on MicrobesOnline
  scaffold 7023, i.e. MG1655 coordinates, for a BW25113 library. Measured on the 3,789
  compendium genes with `reconcile_locus_tags` and `eck_crosswalk`:

  | route | resolved | histogram |
  |---|---|---|
  | `sysName` on MG1655 | 99.89% | 3,660 current, 1 renamed, 124 non-gene feature, 3 retired, 1 ambiguous; 5 decided at the ECK synonym layer |
  | `sysName` on BW25113 | 0% | all 3,789 not found (no b-number is a BW25113 tag) |
  | symbol on BW25113 | 96.91% | 3,514 renamed, 158 non-gene feature, 111 retired, 6 ambiguous; 3,191 at the symbol layer, 487 at the synonym layer |
  | b-number through a one-to-one ECK pair to a BW25113 tag | 3,768 of 3,789 (99.45%) | 1 pair whose numerics disagree |

  The ECK crosswalk is therefore the stronger route to `ecoli_k12_bw25113_locus_tag`, and
  per [[torchcell.datasets.bacteria_common]] its use must be recorded on each record as
  a derived mapping.
- **Phenotype class.** The stored number is a signed log2 ratio centered on zero
  (`GENE_FITNESS_IS_LOG2`). `FitnessPhenotype.validate_fitness` clamps a non-positive
  fitness to 0, so storing it there would erase every defect. The signed home is
  `EnvironmentResponsePhenotype(measurement_type=log2_ratio,
  assay_type=pooled_competitive_growth_barcode)` in the assembly-pinned
  `BacterialEnvironmentResponseExperiment` family, as the Hillenmeyer HIP/HOP loader
  does for its log ratio. The plan's section 3c table maps RB-TnSeq to
  `FitnessPhenotype`; that row needs revisiting before the compendium loader is written.
- **Uncertainty.** `UncertaintyType.standard_error` from `fit_standard_error_obs.tab`; it
  is already an SE, so `derive_se` needs no `n_samples`. The moderated t is a
  significance statistic, not an uncertainty, and has no `UncertaintyType`.
- **Genotype.** A gene fitness value averages a gene's usable insertion strains, so a
  gene-level record is a gene-level genotype: `TransposonInsertionPerturbation` with
  `barcode`, `insertion_position` and `insertion_strand` left `None`. Only the
  per-strain tables (`strain_fit.tab`, `all.poolcount`, the `pool` file) carry barcodes
  and mapped positions, and `pool` positions are on scaffold 7023 (MG1655), so a
  barcode-level record would also need a coordinate liftover to the BW25113 assembly.

### Sourcing decisions

Wetmore `paper.md` sha256 `ca3e7ef27a22a2a28e52ccbe5fdbb60890fea93d03b18b46e1d2b77b033b3cdb`;
Table S1 `si/si7.md` `f10134f0d2ecde8f5c7043c514cbd043240fb33e6866f1c4efad0fff9a18a279`;
Data Set S1 `si/si1.xlsx` `428a06cae37867c8d21541f64d80da082e67743e1b1def9238a14e7726bc7150`;
Price `paper.md` `f3443cdcb2f722b5e6aa6d999f67d68a6f845eb9eb45ad0bac1cc4c24ea53e2d`;
Price `si/si3.xlsx` `e5dbf3d5c97cfc12f49d7fd83f84bc95c16cbe963309a561fff20f442788b879`.
All 34 `SourcedValue` constants pass `audit_sourced_value` (hash and verbatim quote)
under `--data`; release-page values audit against `torchcell-raw`, the rest against
`torchcell-library`.

| constant | value | verbatim quote | source |
|---|---|---|---|
| `STRAIN` | BW25113 | `the model bacterium Escherichia coli BW25113 (a K-12 strain; parent strain of the Keio deletion collection [20])` | paper.md, Results |
| `STRAIN_TABLE_S1` | BW25113 | `<td rowspan=1 colspan=1>Escherichia coli strain BW25113</td><td rowspan=1 colspan=1>wild-type strain</td>` | Table S1 |
| `MUTANT_LIBRARY` | KEIO_ML9 | `<td rowspan=1 colspan=1>KEIO_ML9</td><td rowspan=1 colspan=1>Escherichia coli strainBW25113 transposon mutantlibrary</td>` | Table S1 |
| `TRANSPOSON` | Tn5 transpososome | `<td>Transposon</td><td>Tn5 transpososome</td>` | Table 1 |
| `N_BARCODED_STRAINS` | 152,018 | `<td>No. of strains with unique bar codesa</td><td>152,018</td>` | Table 1 |
| `GENES_WITH_FITNESS` | 3,471 protein-coding | `<td>No. with fitness estimates (% of total)</td><td>3,471 (84)</td>` | Table 1 |
| `STRAIN_FITNESS` | log2 treatment / time zero | `Roughly, strain fitness is the normalized $\log _ { 2 }$ ratio of counts between the treatment sample ...` | Methods |
| `GENE_FITNESS` | weighted mean of strains | `Gene fitness is the weighted average of the strain fitness, and a t score is computed based on the consistency of the strain fitness values for each gene.` | Methods |
| `STRAIN_INCLUSION` | 3 reads/strain, 30/gene, central 10-90% | `(3 reads per strain and 30 reads per gene, considering only the adequate strains)` | Methods (i) |
| `GENE_FITNESS_IS_LOG2` | `log2_ratio` | `<P>Gene fitness is a log<SUB>2</SUB> ratio.` | release page |
| `T_STATISTIC`, `T_STATISTIC_SIGMA` | moderated t, sigma 0.1 | `where $\sigma$ is a small constant (we use 0.1) ...` | Methods (iv) |
| `T_STATISTIC_N` | n = strains of the gene | `where $n$ is the number of strains and $V _ { g }$ is a prior estimate of the variance in gene fitness.` | Methods (iv) |
| `SIGNIFICANT_ABS_T` | 4 | quoted below the table (it contains a pipe) | Results |
| `UNCERTAINTY_TYPE` | `standard_error` | `<LI><B>se</B> -- an estimate of how noisy this gene's measurement is (se is short for standard error)` | release page |
| `STANDARD_ERROR_TABLE` | `fit_standard_error_obs.tab` | `estimated standard error</A> (based on variation across strains)` | release page |
| `N_STRAINS_PER_GENE_FIELD` | `n` (R image only) | `<LI><B>n</B> -- number of usable strains for each gene` | release page |
| `MEDIAN_STRAINS_PER_GENE` | 16 | `<td>Median no. of strains per genec</td><td>16</td>` | Table 1 |
| `REPLICATE_DEFINITION` | same `short` = replicates | `<LI><B>short</B> -- shortened description. Samples with the same value are replicates.` | release page |
| `REPLICATE_DESIGN` | at least 2 | `... all but 5 with at least two biological replicates.` | Results |
| `TIME_ZERO_POOLING` | summed | `We sum the per-strain counts across replicate time-zero samples.` | Methods |
| `TIME_ZERO_REPLICATES` | independent gDNA and PCR | `Also, we usually have multiple replicates of any given time zero, with independent extraction of genomic DNA and independent PCR with a different index.` | Methods |
| `GENERATIONS` | typically 4 to 6 | `(typically 4 to 6 generations)` | Results |
| `SUCCESS_RULES` | gMed >= 50, mad12 <= 0.5, cor12 >= 0.1, abs gccor <= 0.2, abs adjcor <= 0.25 | quoted below the table (it contains pipes) | release page |
| `RELEASE_VERSION` | 101 / 92 / 1.0.0 | `<P><small>101 condition samples (92 successful), Sat Sep 27 07:53:29 2014, statistics version 1.0.0</small>` | release page |
| `SUPERSET_RELEASE` | 207 / 162 / 1.0.3 | `<P><small>207 condition samples (162 successful), Fri Feb 19 10:53:17 2016, statistics version 1.0.3</small>` | compendium release page |
| `SUPERSET_INCLUDES_THIS_PAPER` | 385 | `Our analysis includes 385 successful experiments from Wetmore et al.9 and 36 successful experiments from Melnyk et al.12.` | Price paper.md |
| `STOCK_SOLUTION_CORRECTION` | sucrose, D-mannitol | `Please disregard any of the data from this publication regarding sucrose or D-mannitol.` | compendium page |

The two quotes that contain a pipe, verbatim:

- `SIGNIFICANT_ABS_T`: `Genes with $\left| t \right|$ of ${ > } 4$ have highly significant phenotypes that are largely reproducible in biological replicate experiments (Fig. 2A).`
- `SUCCESS_RULES` (four consecutive lines of the release page):
  `<LI>gMed &ge;50`, `<LI>mad12 &le; 0.5`, `<LI>cor12 &ge; 0.1`,
  `<LI> |gccor| &le; 0.2 and |adjcor| &le; 0.25`

`n_samples`: a gene fitness value averages its usable strains inside ONE sample, and the
count is per gene and per sample. It is released only as field `n` of the R image
(`fit.image`), never in a tab-delimited table; Table 1 gives only its median, 16. Neither
back-solve nor a conservative range applies, since it is not one number for the dataset.
`N_SAMPLES_GAP` records it as `deferred_pending_source_review` naming `fit.image`, and it
does not block an SE. Biological replication is across records (two sample names per
condition), never inside `n_samples`.

### Raw mirror

Retrieved 2026-10-07 with `torchcell.literature.retrieve.direct_url` (`python -m
torchcell.datasets.ecoli.wetmore2015 retrieve --dest DIR`, then `deposit --source-dir
DIR`); `deposit_raw_mirror` checks every source and every existing mirror file before
writing anything, leaves matching files alone, refuses a differing one, and refuses an
existing manifest that records other retrievals. A second deposit was run and changed no
file. Only files the record reads are kept; the per-strain tables and the R images are
listed in `si_expected` as not mirrored.

| path under `data/` | sha256 (first 16) | read for |
|---|---|---|
| `rbarseq/html/Keio/index.html` | `9085883893edd0d5` | release-page quotes |
| `rbarseq/html/Keio/fit_logratios_good.tab` | `f65a093b7fba660f` | value comparison, mannitol check |
| `rbarseq/html/Keio/fit_genes.tab` | `3eae36f697a32008` | identifier finding |
| `bigfit/index.html` | `c4819e452ab54dfd` | the 2021 withdrawal |
| `bigfit/html/Keio/index.html` | `f05323ffce7701df` | compendium version quote |
| `bigfit/html/Keio/fit_quality.tab` | `7d4b15733d8e2586` | compendium verdicts |
| `bigfit/html/Keio/fit_logratios_good.tab` | `b37f038702ef0b79` | value comparison, gene set |

### Checklist answers (plan section 4)

1. Paper pinned: `paper.md` `ca3e7ef2...`; every value above is a verbatim substring.
2. Exact tables: Data Set S1 sheets `Expts_Keio` and `Expt_Quality_Keio`; Table S5 sheet
   `TableS5_Experiments`; release `fit_logratios_good.tab` (normalized gene fitness of
   successful samples) and `fit_quality.tab`. No value column is stored.
3. Background strain: BW25113, assembly set `ecoli_K12_BW25113_ASM75055v1` for the
   compendium loader (`STRAIN`, `STRAIN_TABLE_S1`).
4. Identifiers: the histograms above; no records, so no threshold applies here. The
   compendium loader should state its threshold against the ECK route (99.45%).
5. `n_samples` and uncertainty type: SE (sourced); `n_samples` is a typed gap.
6. Media: not applied, nothing is loaded. The compendium loader takes `LB_LENNOX`,
   `M9_NOCARBON_PRICE2018` and `M9_NONITROGEN_PRICE2018` (its own Table S18), not the
   Wetmore entries, because its values are its release's.
7. Superset: this row is the subsumed one; the record names 92 carried samples.
8. This note.
9. No LMDB: nothing is loaded.

### Gaps and open questions

- **Fumarate.** The one condition lost by not loading this release (`set1IT029`). If the
  owner wants it, a one-sample loader from the 1.0.0 release is possible, but it would be
  a sample the authors' re-analysis rejects, at a different processing version.
- **`FitnessPhenotype` and a signed log ratio** (above): the plan's section 3c mapping
  of RB-TnSeq rows to `FitnessPhenotype` does not survive its clamp.
- **`MOPS Rich Defined media_noCarbon`** (four samples here, `set1IT067` to `set1IT070`)
  has no `MEDIA_LIBRARY` entry; [[torchcell.datamodels.media]] records its
  micronutrient units as corrupt in Data Set S1.
- **The live Fitness Browser** (`fit.genomics.lbl.gov`) answers a script with HTTP 403,
  so its post-2021 replacement sucrose and mannitol samples were not inspected.
