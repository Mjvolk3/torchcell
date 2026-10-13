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

## 2026.10.10 - Loaded from its own release: RbTnseqWetmore2015Dataset (owner policy)

Module: `torchcell/datasets/ecoli/wetmore2015.py` (`RbTnseqWetmore2015Dataset`, root `data/torchcell/rbtnseq_wetmore2015_ecoli`). Adapter: `torchcell/adapters/wetmore2015_adapter.py`, conf `rbtnseq_wetmore2015_ecoli_adapter.yaml`. Tests: `tests/torchcell/datasets/ecoli/test_wetmore2015.py`, `tests/torchcell/adapters/test_wetmore2015_adapter.py`. Release inventory: [[experiments.036-dataset-fixes-before-kg-build.scripts.wetmore2015_release_inventory]]. Schema: [[torchcell.datamodels.schema]] (2026.10.10, the released t statistic; 2026.10.11, why no schema type carries a withdrawal). Compendium: [[torchcell.datasets.ecoli.price2018]].

### Decision: the 2026.10.07 section above is superseded

Owner policy of 2026-10-10: a served compendium is a processing reference and a sanity check, not the source of record. An original study with its own per-record release is loaded from that release, attributed to itself, and cross-checked against the compendium's copy. Wetmore 2015 released its own values (statistics version 1.0.0), so the row is now LOADED, and the Price 2018 store is its L4 sanity check rather than its replacement.

The 2026.10.07 measurements still hold and are still computed by the same functions (`subsumption_record`, `release_comparison`): Price 2018 carries 92 of this paper's 101 condition samples, re-analyzed at 1.0.3. What changed is the policy reading them. The duplication is now explicit and joinable: both datasets key a sample as `Keio:<name>`. The Price 2018 loader keeps its own records, its attribution of 92 samples to this paper, and its drop of the withdrawn samples; nothing in it changed.

### What was built (measured 2026.10.10)

| quantity | value |
|---|---|
| Data Set S1 condition samples / successful (`u`) / failed | 101 / 92 / 9 |
| the paper's five rules reproduce `u` | yes, all 101 (the build refuses a disagreement) |
| stored samples | 84 (carbon source 57, nitrogen source 26, LB 1) |
| withdrawn samples, REFUSED (2026.10.11 decision) | 4: `set1IT007`, `set1IT008` (sucrose), `set1IT043`, `set1IT044` (D-mannitol) |
| release genes / mapped to BW25113 | 3,646 / 3,628 |
| records (`len(dataset)`) | **304,752** = 84 x 3,628 |
| references | 84 (one per sample) |

Every released value is accounted for (`preprocess/dropped_records.json`): 101 x 3,646 = 368,246 released values (the all-sample table) = 304,752 stored + 14,584 (the 4 withdrawn samples) + 32,814 (the 9 failed samples) + 14,584 (the 4 MOPS Rich Defined samples) + 1,512 (18 unmapped genes x 84 stored samples).

| rule | scope | items | released values refused |
|---|---|---|---|
| `authors_withdrew_sucrose_and_mannitol_2021` | sample | the four sucrose and D-mannitol samples | 14,584 |
| `failed_paper_quality_rules` | sample | `set1IT030`, `set1IT033`, `set1IT034` (gMed); `set2IT024`, `set2IT034`, `set2IT045`, `set2IT055`, `set2IT079`, `set2IT090` (cor12) | 32,814 |
| `medium_not_in_media_library` | sample | `set1IT067` to `set1IT070` (`MOPS Rich Defined media_noCarbon`) | 14,584 |
| `no_one_to_one_eck_pair` | gene | 18 b-numbers: 7 not in the MG1655 annotation, 6 whose ECK id BW25113 lacks, 5 not one-to-one | 1,512 |

The first two rules state the source's own words with their sha256 in the ledger (`DropRuleCount.source`): the withdrawal note for the first, the release page's `u` definition for the second. The last two rest on this loader's own typing (the media library, the ECK crosswalk), so they carry no quote.

### The records

- `EnvironmentResponsePhenotype(measurement_type=log2_ratio, assay_type=pooled_competitive_growth_barcode)` in `BacterialEnvironmentResponseExperiment`; value = the gene-fitness table's normalized log2 ratio ("Gene fitness is a log<SUB>2</SUB> ratio.").
- **Standard error as the paper defines it.** The moderated t of Methods (iv) is `t = f / sqrt(sigma^2 + max(V_e, V_n))`, sigma 0.1 ("where $\sigma$ is a small constant (we use 0.1) ..."), with V_e the strain-consistency variance (the release's `se`, `fit_standard_error_obs.tab`) and V_n the naive Poisson one (`sdNaive`, `fit_standard_error_naive.tab`). The per-gene SE the paper's own statistic uses is therefore `sqrt(max(V_e, V_n)) = max(se, sdNaive)` (`PER_GENE_STANDARD_ERROR`, quoting the formula line). Measured on all 106 x 3,646 released values: the released t equals `fit / sqrt(0.1^2 + max(se, sdNaive)^2)` to 1.99e-13, while `se` alone misses by up to 30.1. `sdNaive` is the larger for 24.35% of the stored values. This refines the 2026.10.07 note's "`UncertaintyType.standard_error` from `fit_standard_error_obs.tab`", exactly as the Price 2018 loader refined it for 1.0.3.
- **The released t** is stored on every record in `environment_response_t_statistic` (new field). The verifier's supplementary L2 row re-checks the identity on the stored values.
- **The withdrawal is a refusal, not a field.** No record of the four sucrose and D-mannitol samples is stored; see the 2026.10.11 section. The 2026.10.07 section's evidence that this paper's samples are the ones the note means (the mtlA / mtlD versus manX / manY symptom in its own release, and growth on sucrose to EndOD 1.24 to 1.26) stands and is what makes the refusal this paper's.
- `n_samples` and `sample_unit`: typed gaps (`N_SAMPLES_GAP`, `SAMPLE_UNIT_GAP`); the usable strain count is released only in the R image.
- `screen_id = Keio:<sample>`; reference = the typical gene of the same sample, response 0.
- Genotype: `TransposonInsertionPerturbation(transposon="Tn5 transpososome", library_pool="KEIO_ML9")`, Table 1 and Table S1 verbatim; the strain is BW25113 as Table S1 states it ("wild-type strain"), so the reference is `assembly_reference("BW25113")` with no background beyond the assembly. The release's MG1655 b-number is typed on each perturbation: `identifier_mapping = DerivedIdentifierMapping(source_identifier=<b-number>, route="eck_crosswalk")`. Route histogram over the 3,628 placed ECK ids: renamed 3,493, non_gene_feature 135; perturbed_gene_name is the BW25113 symbol for 3,493 and the tag for the 135 pseudogene loci.
- Environment, from Data Set S1's own columns: `Media` through this paper's own media-library entries (`M9_NOCARBON_WETMORE2015`, `M9_NONITROGEN_WETMORE2015`, `LB_LENNOX`, each quoting the Media tab); `Condition_1` / `Concentration_1` / `Units_1` as `EnvironmentPhysicalPerturbation(factor=carbon_source | nitrogen_source)` (`mg/mL` stored as g/L); `Temperature (°C)` (37 for 87 samples, 28 for the microplate LB sample); `Total Generations` as `duration_generations` ("Total number of population doublings of experiment (if determined)."), a typed gap for `set2IT069` where the column is blank; `duration_hours` a typed gap on every environment (the assays ran to saturation). 29 of the 39 stored `Condition_1` labels are not in the pinned compound-identity table and carry the resolver's typed `inchikey` gap (issue #726); the 30th was D-mannitol, refused with its samples.

### Duplication by content against Price 2018, and why it is admitted anyway

Measured with the release tables (`release_comparison`, `all_sample_comparison`, the inventory script) and with the two dev stores (the L4 row below). 92 of this paper's 101 samples are Price 2018 Table S5 samples under the same name, condition, concentration and medium. On the 3,646 shared genes, per-sample Pearson r of Wetmore 1.0.0 against Price 1.0.3 is min 0.98895, median 0.99713 over all 92 (largest single-gene difference 2.5725 log2, `set1IT028`, `b2500`). These are two analyses of the same reads: Price 2018 "Fitness data was analysed as previously described9 ." (ref 9 is this paper) at statistics version 1.0.3. The candidate verdict records the store-level match (G4-value, below). By the copy test (a value equal to a served value within 1e-9) none is subsumed; by sample identity all are duplicated. The row is admitted anyway because the 2026-10-10 policy makes the original release the source of record and the compendium its check.

**What this release carries that the compendium store does not:**

- `set1IT029` (fumarate): successful here (gMed 50.0, exactly at the rule), failed at 1.0.3 (gMed 48), so the compendium carries no E. coli fumarate sample at all. 3,628 records.
- The released moderated t on every value (the compendium's records store none).
- `duration_generations` per sample from Data Set S1 (the compendium's environments carry none).
- The 1.0.0 statistics themselves (values, SE, t) for the 83 samples both stores hold.

Not carried by either: the four MOPS Rich Defined samples (no library medium on both sides).

### Verification (L0 to L4)

Superseded: the store this section first reported (slurm job 3871, 88 samples, 319,264 records) was rebuilt after the 2026.10.11 refusal decision. The L0-L4 table and its verbatim output are in the 2026.10.11 section below.

### Candidate gate

`python -m torchcell.candidates gate --row "Wetmore 2015" --phenotype-class EnvironmentResponsePhenotype --issue 726 --compound <each stored Condition_1 label> --json`, then the G4-value record added and the composed verdict written with `validate --write`: `database/candidates/wetmoreRapidQuantificationMutant2015.json`, outcome **admissible_with_gaps**, `source_kind` **primary**, `torchcell_commit` `9f708b4e1` (re-gated after the rebase onto main `d88f3865c`).

| gate | outcome | reason, short |
|---|---|---|
| G1 | pass | primary (the candidate row now records the fetched release: `accession_confirmed=True`) |
| G2 | pass | MG1655 and BW25113 resolve and read (the row names the K-12 lineage; the records pin BW25113) |
| G3 | pass | 68 files in library + raw, all intact |
| G4 | pass | no key held elsewhere; values compared against the Price store (below); admitted by policy |
| G5 | gap #726 | `EnvironmentResponsePhenotype` exists; 30 of 41 `Condition_1` labels unresolved |

**G4, the duplication measurement the verdict carries** (`g4_value`, computed by reading both dev stores through `stream_records`, because `candidates overlap` refuses the Price store as built from a dirty tree): 83 released columns against the 147 Price samples on 3,628 genes. The nearest served sample by max abs difference is the column's own name for 81 of 83; `set1IT083` and `set1IT094` sit nearer their exact replicate by that statistic (margins 0.13 and 0.09), which says max abs difference over 3,628 genes is driven by single outlier genes, not that the samples are misidentified (their own-name Pearson r is above 0.98). No released column is within the copy-test tolerance (1e-9) of any served sample; the smallest best max abs difference is 0.52. So: duplicated by content (two analyses of the same reads), not a copy, and admitted under the 2026-10-10 policy, which the G4 reason states.

**The Price 2018 decomposition now records this dataset.** Price 2018's verdict carries a per-source `AggregationRecord.sources` (plan decision 16, [[torchcell.candidates.aggregation-decomposition]]). Its Wetmore entry is built by `wetmore_source` in `aggregation_decomposition_audit.py`, which reads `loaded_as_own_dataset` off the registry: now yes, `RbTnseqWetmore2015Dataset`. Its cross-check moved from a pin to a test: the all-sample release, 92 samples, min r 0.98895 against the same 0.87 bound as the L4 row. Price was re-gated with `--decomposition` (admissible; loadable from own release 0, loaded 1).

### Schema impact

`PYTHONPATH=<worktree> python scripts/schema_impact_check.py --base origin/main`: one changed symbol, `EnvironmentResponsePhenotype` modified (one added optional field, `environment_response_t_statistic`, plus the `_check` validator); **37 impacted dataset groups (this dataset's own included), 0 breaking** (additive), measured against main `d88f3865c`. Every dev store whose closure reaches `EnvironmentResponsePhenotype` reads stale until rebuilt, including the stores array 3733 just rebuilt; this lands with the KG 4.0 full rebuild.

### Raw mirror

`fit_t.tab` (`fcebd1de3403898b`), `fit_standard_error_obs.tab` (`06cc6122f7a3551d`), `fit_standard_error_naive.tab` (`6b2c70e805c337f1`) and `fit_logratios.tab` (`adf240fcb677193c`) were retrieved on 2026-10-10 by the recorded route (`python -m torchcell.datasets.ecoli.wetmore2015 retrieve --dest DIR`, `torchcell.literature.retrieve.direct_url`, every file sha256-checked on write; the seven 2026-10-07 files re-fetched in the same run are byte-identical upstream) and deposited with `deposit --source-dir DIR`. `deposit_raw_mirror` now EXTENDS a manifest whose every record it repeats verbatim (the first deposit's `created_at` kept, each file with its own `retrieved_at`) and still refuses one that records anything else; a second deposit changed no file. Not mirrored: the per-strain tables and the R image.

### Gaps and owner items

- **MOPS Rich Defined medium.** 4 successful samples (14,584 released values) are not stored on either side. Data Set S1's Media tab prints six micronutrients in mM at 40,000 to 3,000,000 ("Ammonium Molybdate | 3000000 | mM"), so the recipe cannot be taken as printed; a `MEDIA_LIBRARY` entry would come from Price 2018 Table S18 or a curated adjudication.
- **The failed samples are not stored.** The paper's analyses exclude them and the release's gene-fitness table omits them; their values stay deposited (`fit_logratios.tab`).
- **The candidate gate's G4 still reads "subsumed = refused".** The verdict records the copy-test tolerance and states the policy in its G4 reason; a gate rule for the policy (a primary release cross-checked against a compendium) is the owner's call.
- **The first `--verify` refused at the L4 row** (a shell run, 2026-10-10): it required identical gene sets per sample, while the compendium holds 3,768 genes against this release's 3,628; the row now scores the shared genes, counts each side's extras, and fails only on a gene held here and not there. Its build manifest was retired (`build_manifest.json.unverified.20261011-004629`). A second shell run was stopped mid-verify on the driver's instruction that dev-store builds go through slurm on a memory-tight node; its unverified manifest was retired too (`...unverified.20261011-023512`), and the store was rebuilt under slurm (job 3871, 48G, 8 CPUs).
- **Every array-built dev store reads `torchcell_dirty: true`** (`git status --porcelain` counts untracked files, and the primary checkout has untracked notes), so `candidates overlap` (`read_store_values`) refuses the Price store; this verdict's G4-value was computed by reading both stores through `stream_records`.
- **Compound identities** (#726): 29 `Condition_1` labels of stored samples (30 before the withdrawn samples were refused; D-mannitol was the 30th).
- **(#986) Thompson 2020's published correction is unread**, and 46 of its samples are stored in `RbTnseqBorchert2024Dataset`. Found by the 2026.10.11 sweep; blocked on a manual retrieval, so it is a different paper's issue, not this one's.

## 2026.10.11 - Author-withdrawn data is refused, not stored and flagged (owner decision)

Owner decision, 2026-10-11, reversing the 2026.10.10 build's treatment of the four sucrose and D-mannitol samples: **author-withdrawn data is bad data and must not be stored.** The owner's words: "if it is author withdrawn is it with good reason... maybe we shouldn't add bad data."

### The refusal reason, verbatim

The reason is the authors' own, read from `data/bigfit/index.html` in this key's raw mirror (sha256 `c4819e452ab54dfd3748242a3d4cf0f04021071aefadb2f07f2806afa32e3555`), section "Note added September 10, 2021":

> Please disregard any of the data from this publication regarding sucrose or D-mannitol.

and its cause, from the same page and section:

> After publishing this data, Morgan Price and Adam Deutschbauer discovered that our stock solutions for sucrose and D-mannitol were problematic.

The cause is what makes the flag insufficient. The stock solution was wrong, so what was measured is not the response to sucrose or to D-mannitol: the value is wrong, not merely caveated. A record kept beside a typed flag still reaches every consumer that does not filter on the flag, and the authors asked readers to disregard the data, not to qualify it.

### The carrier: `AuthorWithdrawal`, one object both loaders refuse on

`SourceWithdrawal`, `SourceWithdrawalReason` and `EnvironmentResponsePhenotype.source_withdrawal` are removed from the schema, with the graph property, the adapter projection and the ontology lane root ([[torchcell.datamodels.schema]], 2026.10.11). A refused record is absent, so no record type needs to carry a withdrawal; the withdrawal is a property of the BUILD and belongs with the other refusals.

In their place, `wetmore2015.AuthorWithdrawal` (frozen, `extra="forbid"`) and the module constant `AUTHOR_WITHDRAWAL`:

| field | value |
|---|---|
| `rule` | `authors_withdrew_sucrose_and_mannitol_2021` (the drop-rule name in both ledgers) |
| `conditions` | `{"Sucrose", "D-Mannitol"}`, the `Condition_1` labels the note covers |
| `instruction` | the "Please disregard..." `SourcedValue`, pinned file + sha256 + section |
| `cause` | the "...stock solutions ... were problematic." `SourcedValue`, same pin |

It replaces `DISREGARDED_CONDITIONS`, `WITHDRAWAL_REASON`, `WITHDRAWAL`, and the Price 2018 loader's own copy of the rule-name string literal, so the two loaders now agree by construction rather than by coincidence. `DropRuleCount` in both loaders gains `source: SourcedValue | None`, so the quote and its sha256 ride into `preprocess/dropped_records.json` beside the count; the medium and ECK rules rest on this loader's own typing and state `None` there.

`classify_samples` checks the withdrawal FIRST, in the order the Price 2018 loader already used, so a withdrawn sample is refused for the withdrawal whatever else it would have tripped.

### Counts, before and after (measured)

`loader_inventory()` off the deposited bytes, `PYTHONPATH=<worktree>`:

| quantity | 2026.10.10 build (job 3871) | after the refusal (job 3913) |
|---|---|---|
| stored samples | 88 | **84** (carbon source 57, nitrogen source 26, LB 1) |
| stored records | 319,264 | **304,752** = 84 x 3,628 |
| records carrying a withdrawal flag | 14,512 | 0 (the field does not exist) |
| released values refused by the withdrawal rule | 0 (they were stored) | **14,584** = 4 x 3,646 |
| references | 88 | 84 |
| `Condition_1` labels with the typed `inchikey` gap (#726) | 30 of 41 | 29 of 39 (D-mannitol went with its samples, and sucrose, which resolves) |

The released-value accounting still balances over the all-sample table: 101 x 3,646 = **368,246** = 304,752 stored + 14,584 withdrawn + 32,814 failed + 14,584 MOPS Rich Defined + 1,512 (18 unmapped genes x 84 stored samples).

Price 2018 (`rbtnseq_price2018_ecoli`): **record delta 0.** Measured from the dev store built on main, `preprocess/samples.json` already gives all four the drop rule `authors_withdrew_sucrose_and_mannitol_2021`, and `dropped_records.json` already counts 15,156 refused released values (4 x 3,789); 147 samples kept, 553,896 records. That loader needed no behavior change, only the shared typed reason ([[torchcell.datasets.ecoli.price2018]], 2026.10.11).

### Verification of the rebuilt store (L0 to L4)

`python -m torchcell.database.build_dataset_lmdb --dataset RbTnseqWetmore2015Dataset --io-workers 8 --retire-existing --verify` under slurm (**job 3913**, 48G, 8 CPUs) from `392bb4132`: 304,752 records written, then `verify_build`; 50 min 22 s end to end (the L0 structural pass over every record is the cost). **Every row PASS.** Report: `data/torchcell/rbtnseq_wetmore2015_ecoli/preprocess/verification_report.json`; the cross-check in full: `preprocess/compendium_cross_check.json`. The store reads fresh (`build_dataset_lmdb --list-stale --include-private` does not name it). The 2026.10.10 build's `processed/` and `preprocess/` were retired as `*.superseded.20261011-044320`.

| level | rows | result |
|---|---|---|
| L0 | structural (304,752 records validated against the `ExperimentType` union) | pass |
| L1 | count (304,752 = 84 x 3,628), pair uniqueness, provenance gaps (3 deferred fields: inchikey, n_samples, sample_unit), canonical gene names (135 pseudogene loci resolve to themselves), stored tags are loci (supplementary), **withdrawn samples refused not stored** (supplementary: 0 records of the 4, 84 samples stored) | pass |
| L2 | value fidelity, SE non-negative, interval orientation, uncertainty sanity, the t identity (supplementary: max deviation 1.99e-13 over every record) | pass |
| L3 | single measurement type, reference = 0, every record carries an environmental edit, compound identity (29 labels with the typed gap), media compound identity, media membership (3 library media) | pass |
| L4 | gene containment and current-genome genes on BW25113 (3,628 of 3,628), the Price 2018 cross-check (supplementary: 83 samples, min r 0.9889, median 0.9971, against 0.87) | pass |

Verbatim:

```
RbTnseqWetmore2015Dataset: PASS
  [ok] L0 structural: 304752 records validated
  [ok] L1 count: observed 304752, expected 304752
  [ok] L1 pair_uniqueness: 304752 unique (study, strain, condition) records, one each
  [ok] L2 value_fidelity: 304752 values checked
  [ok] L2 se_nonnegative: 304752 values checked
  [ok] L2 interval_orientation: 0 of 0 stored intervals do not bracket their value, as declared
  [ok] L3 measurement_type_consistent: single measurement_type: 'log2_ratio'
  [ok] L3 reference_zero: numeric rule: reference response == 0 for all 304752 records
  [ok] L3 environment_perturbed: all 304752 experiments carry an environmental edit (perturbation, non-baseline temperature, or non-baseline media; baseline temp=37.0, media="M9 minimal medium, no carbon source (Wetmore 2015 'M9 minimal media_noCarbon')")
  [ok] L1 provenance_gaps: 1135564 documented provenance gaps over 304752/304752 records; 3 deferred field(s): ['inchikey', 'n_samples', 'sample_unit']; 6708172 undeclared None values over 15 carrier fields (top: Compound.inchi x2514204, Compound.chebi_id x689320, Environment.dilution_rate_per_hour x304752, EnvironmentResponsePhenotype.category x304752, EnvironmentResponsePhenotype.category_label x304752)
  [ok] L1 canonical_gene_names: 3628 systematic names, one canonical spelling each, each current in the genome; 135 common names the resolver cannot place; 135 are pseudogene loci the genome resolves to themselves
  [ok] L2 uncertainty_sanity: 304752 labeled uncertainties, none a zero dispersion; 0 records report n_samples >= 2 with no uncertainty
  [ok] L3 compound_identity: environment edits: 83444 compound references carry a structure identifier; 217680 declare a typed gap (29 distinct compounds, unencodable)
  [ok] L3 media_compound_identity: medium components: 1810372 compound references carry a structure identifier; 0 declare a typed gap (0 distinct compounds, unencodable)
  [ok] L3 media_membership: 304752 records on a shared MEDIA_LIBRARY medium, 0 on a medium deriving from one (3 distinct media)
  [ok] L4 gene_containment_sgd: 1.000 of 3628 measured genes are reference genes (>= 1.0)
  [ok] L4 current_genome_genes: every one of the 3628 measured systematic names is a gene of the current genome
  [ok] L1 stored_tags_are_loci_of_the_pinned_assembly: SUPPLEMENTARY: 3628 stored tags, statuses {'current': 3493, 'non_gene_feature': 135}; 0 do not resolve to themselves
  [ok] L1 withdrawn_samples_refused_not_stored: SUPPLEMENTARY: 0 records of the 4 samples the authors withdrew (['set1IT007', 'set1IT008', 'set1IT043', 'set1IT044']); 0 of them are in the store; 84 samples stored, expected 84
  [ok] L2 t_statistic_matches_the_stored_se: SUPPLEMENTARY: max |t - fit / sqrt(0.1^2 + SE^2)| = 1.99e-13 over 304752 records (tolerance 1e-09); 0 records lack a t or an SE
  [ok] L4 compendium_cross_check_price2018: SUPPLEMENTARY: 83 of 92 shared samples compared against data/torchcell/rbtnseq_price2018_ecoli (301124 values); per-sample Pearson r min 0.9889 / median 0.9971 against the bar 0.87 (the compendium's median exact-replicate r); largest single-gene difference -2.573 log2 (set1IT028 BW25113_2500), a finding, not an error; absent here {'set1IT007': 'authors_withdrew_sucrose_and_mannitol_2021', 'set1IT008': 'authors_withdrew_sucrose_and_mannitol_2021', 'set1IT043': 'authors_withdrew_sucrose_and_mannitol_2021', 'set1IT044': 'authors_withdrew_sucrose_and_mannitol_2021', 'set1IT067': 'medium_not_in_media_library', 'set1IT068': 'medium_not_in_media_library', 'set1IT069': 'medium_not_in_media_library', 'set1IT070': 'medium_not_in_media_library', 'set2IT045': 'failed_paper_quality_rules'}, absent there {'set1IT007': 'authors_withdrew_sucrose_and_mannitol_2021', 'set1IT008': 'authors_withdrew_sucrose_and_mannitol_2021', 'set1IT043': 'authors_withdrew_sucrose_and_mannitol_2021', 'set1IT044': 'authors_withdrew_sucrose_and_mannitol_2021', 'set1IT067': 'medium_not_in_media_library', 'set1IT068': 'medium_not_in_media_library', 'set1IT069': 'medium_not_in_media_library', 'set1IT070': 'medium_not_in_media_library'}
```

### The cross-check still compares 83 samples, not 79

The refusal was expected to drop the comparison to 79. It does not, and the reason is measurable: the four withdrawn samples were **never in the comparison**, because the Price store already refused them. Before the refusal they were "absent there" with Price's own rule; now they are absent on both sides, which is what the L4 row prints. Measured directly from the two stores' `preprocess/samples.json` against `subsumption_record().carried`:

| set | count |
|---|---|
| samples of this paper the compendium's Table S5 carries | 92 |
| of those, the Wetmore store holds | 83 (was 87) |
| of those, the Price store holds | 84 |
| compared (in both) | **83** (was 83) |
| absent here | `set1IT007`, `set1IT008`, `set1IT043`, `set1IT044` (withdrawn), `set1IT067` to `set1IT070` (MOPS Rich Defined), `set2IT045` (failed cor12) |
| absent there | the same four withdrawn, and the four MOPS Rich Defined |

The candidate verdict's `g4_value.matches` is the independent check: 83 entries, none of them a withdrawn sample, before and after.

### Candidate verdict re-gated

`database/candidates/wetmoreRapidQuantificationMutant2015.json`, `torchcell_commit` `83388017a`: still **admissible_with_gaps**, `source_kind` **primary**, G1 to G4 pass and G5 a #726 gap. What moved: G5 reads **29** unresolved compounds instead of 30 (D-mannitol left with its samples; the gate was re-run on the 39 `Condition_1` labels the refusal leaves stored), and `g4_value.store_commit` is now `cb310e33`, the commit the Price dev store was last rebuilt from (2026-10-11T07:35Z, by the #726 compound-curator branch; dev stores are shared between worktrees). The G4 numbers are unchanged: 83 columns compared, best-is-own-name for 81 of 83, smallest best max abs difference 0.521 against the 1e-9 copy tolerance, so still duplicated by content and not a copy.

### Other loaders swept

`disregard | withdraw | retracted | corrigendum | erratum` (case-insensitive) over `torchcell/datasets/`, `torchcell/datamodels/` and `notes/`, plus `retraction | expression of concern | published correction | advisory | note added`. No loader other than this one stored data its own authors withdrew. The hits and their verdicts: `kemmeren2014.py`'s withdrawn SGD gene NAME for YHR127W (not data); Niu 2019's 2020 erratum, already read, which corrects a competing-interest statement and no data; Borchert 2024's advisory check, already measured, where the KT2440 release has 0 sucrose or mannitol samples; `shiver2016.py`'s "Unreliable measurements were removed", the authors' own pre-publication filtering; and our own analysis retractions in the `experiments.018` / `experiments.019` notes.

One unresolved case, filed rather than guessed at: **Thompson 2020's published correction** (`PMC8091123`) is unread and 46 of its samples are stored in `RbTnseqBorchert2024Dataset` (217,672 records at its 4,732 mapped genes). The notice is not scriptable and not mirrored, so its scope is UNKNOWN; **issue #986** carries the measurement and the manual-retrieval steps that close it.

## 2026.10.12 - #1057 on main: released rows stored, rebuilt and verified

Owner rule of 2026-10-12: a released row is stored under the identifier its release gives, and the genome never gates storage. Rebuilt from main `db632f2b0` with `--retire-existing --verify` (slurm array 4506; the stores with no per-dataset verifier entry point verified by job 4582).

### `rbtnseq_wetmore2015_ecoli`: 304,752 -> 320,848 records

Row 16 MOPS rich medium (+4 samples) and the ECK-unplaced genes stored as released.

| level | rule | result | message |
|---|---|---|---|
| L0 | `structural` | pass | 320848 records validated |
| L1 | `count` | pass | observed 320848, expected 320848 |
| L1 | `pair_uniqueness` | pass | 320848 unique (study, strain, condition) records, one each |
| L2 | `value_fidelity` | pass | 320848 values checked |
| L2 | `se_nonnegative` | pass | 320848 values checked |
| L2 | `interval_orientation` | pass | 0 of 0 stored intervals do not bracket their value, as declared |
| L3 | `measurement_type_consistent` | pass | single measurement_type: 'log2_ratio' |
| L3 | `reference_zero` | pass | numeric rule: reference response == 0 for all 320848 records |
| L3 | `environment_perturbed` | pass | all 320848 experiments carry an environmental edit (perturbation, non-baseline temperature, or non-baseline media; baseline temp=37.0, media="M9 minimal medium, no carbon source... |
| L1 | `provenance_gaps` | pass | 984420 documented provenance gaps over 320848/320848 records; 3 deferred field(s): ['inchikey', 'n_samples', 'sample_unit']; 7448778 undeclared None values over 15 carrier field... |
| L1 | `canonical_gene_names` | finding (genome reference; reports, does not gate) | 0 genes carry conflicting common-name spellings (0 records; 0 case-only); 18 systematic names are not the genome's current name; 1 common names resolve to another gene |
| L2 | `uncertainty_sanity` | pass | 320848 labeled uncertainties, none a zero dispersion; 0 records report n_samples >= 2 with no uncertainty |
| L3 | `compound_identity` | pass | environment edits: 298972 compound references carry a structure identifier; 18230 declare a typed gap (2 distinct compounds, unencodable) |
| L3 | `media_compound_identity` | pass | medium components: 2475634 compound references carry a structure identifier; 0 declare a typed gap (0 distinct compounds, unencodable) |
| L3 | `media_membership` | pass | 320848 records on a shared MEDIA_LIBRARY medium, 0 on a medium deriving from one (4 distinct media) |
| L4 | `gene_containment_sgd` | finding (genome reference; reports, does not gate) | 0.995 of 3646 measured genes are reference genes (>= 1.0) |
| L4 | `current_genome_genes` | finding (genome reference; reports, does not gate) | 18 systematic names are absent from the current genome over 1584 records: b0500 x88, b1172 x88, b2650 x88, b2863 x88, b3683 x88, b4104 x88, b4294 x88, b4339 x88, b4486 x88, b449... |
| L1 | `stored_tags_are_loci_of_the_pinned_assembly` | pass | SUPPLEMENTARY: 3628 stored tags, statuses {'current': 3493, 'non_gene_feature': 135}; 0 do not resolve to themselves |
| L1 | `withdrawn_samples_refused_not_stored` | pass | SUPPLEMENTARY: 0 records of the 4 samples the authors withdrew (['set1IT007', 'set1IT008', 'set1IT043', 'set1IT044']); 0 of them are in the store; 88 samples stored, expected 88 |
| L2 | `t_statistic_matches_the_stored_se` | pass | SUPPLEMENTARY: max /t - fit / sqrt(0.1^2 + SE^2)/ = 1.99e-13 over 320848 records (tolerance 1e-09); 0 records lack a t or an SE |
| L4 | `compendium_cross_check_price2018` | pass | SUPPLEMENTARY: 87 of 92 shared samples compared against data/torchcell/rbtnseq_price2018_ecoli (317202 values); per-sample Pearson r min 0.9890 / median 0.9971 against the bar 0... |

## 2026.10.12 - #1045 part 2: the rest of the released rows, under their released identifiers

Every row #1057 still dropped that is not an author withdrawal is stored, and no identifier is translated: the stored key is the one the release prints, and a crosswalk to an annotation is a modeling-time join outside the record. Rebuilt from the part-2 branch with `--retire-existing --verify` (slurm array 4583; the stores with no per-dataset verifier entry point verified by jobs 4645 and 4644).

### `rbtnseq_wetmore2015_ecoli`: 320,848 -> 353,662 records

The 9 quality-flagged samples stored, the authors' flag kept in samples.json (88 -> 97 samples), and every gene under its released b-number.

| level | rule | result | message |
|---|---|---|---|
| L0 | `structural` | pass | 353662 records validated |
| L1 | `count` | pass | observed 353662, expected 353662 |
| L1 | `pair_uniqueness` | pass | 353662 unique (study, strain, condition) records, one each |
| L2 | `value_fidelity` | pass | 353662 values checked |
| L2 | `se_nonnegative` | pass | 353662 values checked |
| L2 | `interval_orientation` | pass | 0 of 0 stored intervals do not bracket their value, as declared |
| L3 | `measurement_type_consistent` | pass | single measurement_type: 'log2_ratio' |
| L3 | `reference_zero` | pass | numeric rule: reference response == 0 for all 353662 records |
| L3 | `environment_perturbed` | pass | all 353662 experiments carry an environmental edit (perturbation, non-baseline temperature, or non-baseline media; baseline temp=37.0, media="M9 minimal medium, no carbon source... |
| L1 | `provenance_gaps` | pass | 1115676 documented provenance gaps over 353662/353662 records; 3 deferred field(s): ['inchikey', 'n_samples', 'sample_unit']; 8148810 undeclared None values over 15 carrier fiel... |
| L1 | `canonical_gene_names` | finding (genome reference; reports, does not gate) | 0 genes carry conflicting common-name spellings (0 records; 0 case-only); 7 systematic names are not the genome's current name; 6 common names resolve to another gene |
| L2 | `uncertainty_sanity` | pass | 353662 labeled uncertainties, none a zero dispersion; 0 records report n_samples >= 2 with no uncertainty |
| L3 | `compound_identity` | pass | environment edits: 298972 compound references carry a structure identifier; 29168 declare a typed gap (3 distinct compounds, unencodable) |
| L3 | `media_compound_identity` | pass | medium components: 2563138 compound references carry a structure identifier; 0 declare a typed gap (0 distinct compounds, unencodable) |
| L3 | `media_membership` | pass | 353662 records on a shared MEDIA_LIBRARY medium, 0 on a medium deriving from one (4 distinct media) |
| L4 | `gene_containment_sgd` | finding (genome reference; reports, does not gate) | 0.998 of 3646 measured genes are reference genes (>= 1.0) |
| L4 | `current_genome_genes` | finding (genome reference; reports, does not gate) | 7 systematic names are absent from the current genome over 679 records: b0500 x97, b4104 x97, b4339 x97, b4590 x97, b4600 x97, b4659 x97, b4694 x97 |
| L1 | `withdrawn_samples_refused_not_stored` | pass | SUPPLEMENTARY: 0 records of the 4 samples the authors withdrew (['set1IT007', 'set1IT008', 'set1IT043', 'set1IT044']); 0 of them are in the store; 97 samples stored, expected 97 |
| L2 | `t_statistic_matches_the_stored_se` | pass | SUPPLEMENTARY: max /t - fit / sqrt(0.1^2 + SE^2)/ = 1.99e-13 over 353662 records (tolerance 1e-09); 0 records lack a t or an SE |
| L4 | `compendium_cross_check_price2018` | pass | SUPPLEMENTARY: 88 of 92 shared samples compared against data/torchcell/rbtnseq_price2018_ecoli (320848 values); per-sample Pearson r min 0.9890 / median 0.9971 against the bar 0... |
