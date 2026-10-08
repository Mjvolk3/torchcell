---
id: 9lz0k9qcxqdvu824ytjmd5e
title: Bacteria Si Phenotype Audit Ecoli
desc: ''
updated: 1791422872525
created: 1791422872525
---

## 2026.10.07 - E. coli SI phenotype audit, all 14 loaded papers

The question: for each bacterial paper already loaded, does the paper **also release
additional phenotypes in its supplementary data that our loader does not store**, and does
it **compare its own results against other released data** in a way that points at further
loadable measurements? This section answers it for the 14 *E. coli* papers that have a
loader on `main`. It changes no loader, no schema, no adapter, no conf and no store.

Each paper's SI was read from `$DATA_ROOT/torchcell-library/<citation_key>/si/`, every data
file was opened and its real sheet names and column headers printed, and the loader plus its
dendron note were read to separate a genuine omission from a decision the project already
recorded. Per-record quantities are enumerated from the column headers and the SI captions,
never from the abstract. Every row cites its SI file and either a verbatim quote or a real
column name, and anything not read is labeled under each paper's **Not checked**.

Two conventions, so the counts read correctly. A row of a paper's table can carry two
classifications (consumed as a normalizer and not stored, or blocked and already recorded),
so a paper's bucket counts sometimes exceed its row count; the per-section totals line is
the authority for that paper. And **"already recorded" retires an item**: where the loader
docstring, its retention ledger or its note already states the decision, this audit cites
that line and moves on rather than re-reporting it.

**On the repo's strict artifact rule, honestly.** `CLAUDE.md` requires any number in
`notes/` to come from a committed script. The counts here were produced by one-off reads of
the mirrored SI (openpyxl, pandas, `wc -l`, streaming passes) rather than by a committed
script, so every one of them instead states **the file, the sheet or column, and the counting
rule**, which is what makes it recomputable. Where a count took more than a header read, the
rule is written out beside it. No number in this note is reused by a figure, a table or the
manuscript; if one ever is, it needs a committed generator first.

The citation keys differ from the module names in five cases, which is worth stating once:
`caglar2017` is `caglarColiMolecularPhenotype2017`, `fuhrer2017` is
`fuhrerGenomewideLandscapeGene2017`, `lamoureux2023` is
`lamoureuxMultiscaleExpressionRegulation2023`, `tong2020` is
`tongGeneDispensabilityEscherichia2020`, and `wang2015` is
`wangDynamicInterplayMultidrug2015`.

### Summary

Stored record counts are measured from each dev LMDB at
`$DATA_ROOT/data/torchcell/<dataset>/processed/lmdb` by `txn.stat()['entries']`.

| paper | loader | stored records (measured) | released quantities enumerated | loadable-now items | records they would add | blocked items | comparison flag |
|---|---|---|---|---|---|---|---|
| Caglar 2017 | `ecoli/caglar2017.py` | 152 + 105 | 25 | 1 | 19 | 1 | Houser 2015 re-release, latent |
| Fuhrer 2017 | `ecoli/fuhrer2017.py` | 3,735 | 22 | 0 | 0 | 2 | Sevin 2014 reused, Ishii 2007 unloaded |
| Goodall 2018 | `ecoli/goodall2018.py` | 8,203 | 14 | 0 | 0 | 3 | points at Keio (Baba 2006) |
| Lamoureux 2023 | `ecoli/lamoureux2023.py` | 241 | 24 | 5 | 1,778 (+ <=20) | 4 | Public K-12, the #760 pattern waiting |
| Price 2018 | `ecoli/price2018.py` | 553,896 | 38 | 4 | 516 + <=24,626,916 | 4 | Wetmore resolved; Fitness Browser hazard |
| Tong 2020 | `ecoli/tong2020.py` | 111,420 | 22 | 0 | 0 | 3 | **S1B is Baba 2006's data** |
| Wang 2015 | `ecoli/wang2015.py` | 46 | 20 | 1 | 46 | 2 | Rutherford 2010 re-plotted |
| Cui 2018 | `ecoli/cui2018.py` | 141,542 | 36 | 0 | 0 | 0 | #760 resolved, note stale |
| Gupta 2024 | `ecoli/gupta2024.py` | 13 | 38 | 1 | 1 | 9 | internal, two ceiling conventions |
| Rapp 2026 | `ecoli/rapp2026.py` | 1,496 | 27 | 5 | 3,001 | 3 | **Table S5 vs S7 disagree** |
| Schmidt 2016 | `ecoli/schmidt2016.py` | 14 | 33 | 3 | 54 | 7 | points at Ishii 2007, Li 2014 |
| Shiver 2016 | `ecoli/shiver2016.py` | 204,033 | 14 | 1 | 835,337 | 2 | **Nichols plates re-scored** |
| Wang 2018 | `ecoli/wang2018.py` | 240,481 | 31 | 0 | 0 | 3 | Cui leakage, not duplication |
| Rousset 2018 | `ecoli/rousset2018.py` | 68,436 | 13 | 0 | 0 | 1 | #760 resolved, note stale |
| **total** | | **1,333,813** | **357** | **21** | **840,752 + <=24,626,916** | **44** | |

**Headline answer: yes, in 8 of the 14 papers.** Twenty-one loadable-now opportunities
exist, together worth **840,752 records with a measured or exactly bounded count**, plus one
item whose exact count is unmeasured (Price's per-strain fitness, upper bound 24,626,916
records), for an upper bound of **25,467,668**, plus at most 20 more from rank 21. Six papers have nothing loadable left:
Fuhrer 2017, Goodall 2018, Tong 2020, Cui 2018, Wang 2018 and Rousset 2018. Three of those
six are complete because the loader already took everything the schema can hold, and three
are complete because what remains is blocked.

**The single biggest measured opportunity is Shiver 2016**, and it is the one that carries
the worst duplication risk. The 235 Nichols batch-0 condition rows of its S1 Dataset are
**835,337 records** (measured: 235 rows x 3,720 kept gene columns = 874,200 cells minus
38,863 blanks), in the same file, with the same strain columns, the same score semantics and
the same experiment and phenotype classes as the 204,033 records we already serve from that
key. It needs no schema change and 4.1x's that dataset. But those rows are Nichols's
physical plates re-scored by Shiver's pipeline, verbatim: "After reanalyzing the original
images from the Nichols et al. [8] screen with our improved workflow, we integrated both
datasets." Nichols 2011 is rank 10 of the bacterial expansion plan and is not in the mirror
(#691), so if it later lands from its own release the store would hold one set of plates
scored twice. Unlike #760, the two copies would **not** be numerically identical, so a
correlation check cannot catch it: Shiver's own re-screens of 15 Nichols conditions at
identical dose correlate at Pearson r = 0.5402 over 55,368 co-present cells. Only the
provenance says it is one experiment.

**Three new duplication risks, none of them the #760 shape.**

1. **Tong 2020's sheet `S1B - Comparison to Keio Data` carries Baba 2006's own growth
   numbers**, in three columns the loader never reads: `LB_22hr (Baba06)`,
   `MOPS_24hr (Baba06)`, `MOPS_48hr (Baba06)`, 10,931 cells over the 3,644 Keio strains the
   build keeps. Baba 2006 is mirrored but holds no data file (`si_expected: []`, verified),
   its own growth table is its Supplementary Table 3 which we do not have, and there is no
   Baba loader. So Tong's S1B is currently our only copy of Baba's per-strain growth data,
   and loading it under Tong's key would attribute Baba's measurements to Tong and collide
   with any later Baba loader. This points the opposite way from #760: the risk is
   mis-attribution, not double storage.
2. **Shiver's Nichols batch-0 block**, above.
3. **Lamoureux 2023's Public K-12 arm** folds 1,675 other labs' SRA samples in beside
   PRECISE-1K, carrying 1,675 BioProject and 783 PMID values. Measured and bounded: the
   public `counts.csv` has 3,125 sample columns and **zero** `p1k_` columns, so nothing
   stored is duplicated today, but any future *E. coli* K-12 RNA-seq loader built from
   GEO/SRA will overlap it. That is the #760 pattern waiting to happen, and the dedup key
   has to be the accession.

**Two internal release defects that would corrupt a load.** Gupta 2024's `si13.zip` Source
Data tables republish Supplementary Data 1's half-lives with the ceiling set to the
**doubling time** rather than 8 h: 8 of 6,458 compared rows differ by exactly 1.000 h,
always on a protein with one ceiling replicate. The loader consumed Supplementary Data 1 and
is internally consistent; the Source Data tables must never be loaded. And Rapp 2026's
Table S7 `Mean_Int` column equals `Mean_FC` on all 9,462 rows and the mean of
`R1_Int`/`R2_Int` on none, while Table S5's `Mean_Int` is correct on all 1,385 rows.
Separately, Table S7 reports a **third** set of fold changes for pairs we already store: 689
joined pairs, 0 identical, Pearson r 0.955, so it is not a superset.

**Convergent pointer: four of the fourteen papers point at the Keio collection's own
per-strain growth data as a loadable dataset we lack.** Tong's S1B republishes it, Goodall's
Table S2 benchmarks against its essential-gene list, Price's Supplementary Note 1 uses a
PEC-plus-Keio list of 330 genes as its essentiality benchmark, and Wang 2018's Fig. 5a scales
its scatter by "the $1 / \mathsf { O D } _ { 6 0 0 }$ value of the relevant gene knockout
reported with the Keio collection" with Supplementary Note 1 naming the cutoff OD600 < 0.09
at 48 h in MOPS. Baba 2006 is in the literature mirror with no data file, no raw mirror, no
loader and no row in the twenty-dataset plan. **Naming it is allowed; adding it is a curation
decision that needs its own go-ahead.**

**Twelve schema gaps surfaced that none of #749, #753, #756, #758 or #760 names.** Ranked by
how many of the fourteen papers they block:

| # | missing capability | papers blocked |
|---|---|---|
| 1 | a member of `MeasurementType` for an **absolute growth readout** (optical density, absolute growth rate in h^-1), plus relief from the environment-response verifier's L3 `reference_zero` rule, which an absolute rate cannot satisfy | Tong (Baba OD600), Rapp (sampling OD), Gupta (OD600 curves), Lamoureux (`approx_OD`), Fuhrer (absolute rate), Caglar and Schmidt (absolute rates) |
| 2 | a **significance** field (p value, q value, FDR) on `EnvironmentResponsePhenotype` | Rousset (`padj`), Schmidt (`qvalue_*`), Wang 2018 (`FDRvalue`, `FPRvalue`), Price (the moderated t statistic) |
| 3 | a **time-series / growth-curve** phenotype | Rapp (Table S2, 1,515 x 3 x 181 points), Gupta (two OD600 series) |
| 4 | a **post-translational-modification** phenotype | Schmidt (Tables S17 to S22, including quantification in three deletion strains) |
| 5 | a **between-environment per-protein abundance ratio** carrier (`ProteinAbundancePhenotype`'s docstring forbids a ratio) | Gupta (si8's two log2 columns), Schmidt (Tables S7, S8 `medianRatio_*`) |
| 6 | a tri-state call and a continuous field on `GeneEssentialityPhenotype` | Goodall (317 `Unclear` rows, plus the insertion index and the log likelihood ratio on 8,520 records) |
| 7 | an **MIC / IC50** member whose value is a concentration | Shiver (Table 2, 14 records), Price (Table S4, 48 numeric and 3 censored E. coli values) |
| 8 | a typed **censoring** flag outside #753's scope | Price (Table S4's "Above maximum" / "Below minimum") |
| 9 | an **asymmetric interval** field (#753 asks for an interval; this needs an asymmetric one) | Caglar (the doubling time's `.95m` / `.95p` pair) |
| 10 | a **bacterial cell-size** phenotype | Schmidt (Tables S28, S29) |
| 11 | a **targeted RT-qPCR panel** phenotype | Wang 2015 (Table S4, 33 cells over 4 strains) |
| 12 | a key for an **unidentified m/z feature** | Rapp (Table S7's 6,615 unannotated features) |

Gap 1 is the load-bearing one: it blocks measurements in seven of the fourteen papers, and
`torchcell/datasets/pputida/lim2025.py:113-118` already records the `reference_zero` half of
it verbatim.

**Five loader or note claims that the SI contradicts**, each a factual statement a reader
would rely on:

1. `price2018.py:638` records the solvent gap as "no vehicle is stated per stress compound in
   the paper, Table S4 or Table S5". `TableS4_Stress` has a column named `Solvent` (water 47,
   DMSO 7, Ethanol 1), and all 35 distinct kept stress conditions match a Table S4 compound
   by case-insensitive exact name. The gap may still be right, because that is the prescreen
   stock solvent, but the recorded reason is false.
2. The Price note's "derived mapping has no typed slot" gap is stale:
   `TransposonInsertionPerturbation.identifier_mapping: DerivedIdentifierMapping | None`
   exists with an `eck_crosswalk` route, and the loader does not set it.
3. `cui2018.py:62-64` says `pos`, `ori`, `coding` and `essential` are "kept verbatim in
   `preprocess/guide_retention.csv`". The built file's header is
   `guide,n_targets,source_genes,positions,stored_tags,fit18,fit75,drop_reason`: only `pos`
   survives, and `coding` is the covariate the paper's whole analysis turns on.
4. The Gupta raw manifest's `si_expected` dismisses Supplementary Data 2 to 6 as "an analysis
   derived from Supplementary Data 1 rather than an independent measurement". That is wrong
   for Data 5 and 6: si1.md:616 reads "Absolute protein abundances ... are calculated using
   label free mass spectrometry (Supplementary Data 6)", which is a separate acquisition.
5. The Rousset note is stale by one landed change (its ledger still reads 91,609 records and
   its Cui overlap is still filed as unresolved), and the Cui note's Rousset section still
   asserts "Four of Rousset's conditions are environments Cui never ran" and files the
   shared-spacer question as an untested hypothesis. #760 settled both.

**Two measured corrections to numbers already on record.** #760 priced option 1 as losing
"the 4,920 spacers' LC-E75 measurements"; at the grain the Rousset loader stores the loss is
**1,687 records**, because the other 3,233 Rousset-only spacers are template-strand or
intergenic and rule 1 or 2 drops them regardless (`rousset2018.py:123`, and
`5,063 + 30,811 + 21,685 + 1,687 = 59,246`). And the Wang 2018 note says the gene level is
"genuinely additional information and not an aggregate"; measured, **20,059 of 20,158
released gene rows reproduce exactly** as the median of that gene's N most start-proximal
`Quality == "Good"` guide fitness values, and the gene-level `Z score` divides out to the
same per-screen sigma the loader already back-solved. The genuinely unstored information is
exactly `FDRvalue` and `FPRvalue`, which strengthens the note's case for a significance
field rather than weakening it.

### Ranked loadable-now opportunities, largest first

Record counts are measured unless marked. "Basis" states how.

| rank | opportunity | records | basis | schema change needed |
|---|---|---|---|---|
| 1 | **Price 2018 per-strain fitness** (`strain_fit.tab`, plus `strain_se`) as strain-level `EnvironmentResponsePhenotype` with a barcoded `TransposonInsertionPerturbation` | **<= 24,626,916** | **upper bound, not a count**: 152,018 unique barcodes (Table S20, measured) x 162 samples. Only `used = TRUE` strains carry a value and the file is not mirrored, so the real count is unmeasured | none; `barcode`, `insertion_position` and `insertion_strand` all exist on the leaf and are currently `None` |
| 2 | **Shiver 2016 Nichols batch-0 block** of S1 Dataset, 235 further conditions | **835,337** | measured: 235 x 3,720 kept columns = 874,200 cells minus 38,863 blanks, counted with the loader's own 243 dropped labels; the same script reproduces 204,033 on the served rows | none. Read the duplication warning first |
| 3 | **Lamoureux 2023 Public K-12**, 1,675 public RNA-seq samples as a separate dataset | **1,675** | measured: 1,675 non-`p1k_` rows of `k12_modulome/metadata_qc.csv`, matching the paper's "the final set of 1675 high-quality publicly-available samples" | none; needs a dedup policy on the accession |
| 4 | **Rapp 2026 growth**, the paper's own AUC over Table S2's curves, as `FitnessPhenotype` against the 16 control strains | **1,514** | measured: 4,593 Table S2 rows = 1,515 genes x 3 replicates plus 48 control rows, minus `phnE` by an existing drop rule. 18 of those genes have no metabolome sample, so this extends strain coverage | none for the AUC scalar; the 181-point curve itself is blocked by gap 3 |
| 5 | **Rapp 2026 targeted LC-MS/MS fold change** (Table S6), a second platform | **411** records, 1,256 values | measured with pandas: `len(df) = 1256`, `Gene.nunique() = 411`, `Abbreviation.nunique() = 289`, all 1,256 non-null | none; a distinct `measurement_type`. Not a duplicate of the stored FI-MS: r = 0.6722, median absolute log2 difference 1.1649 |
| 6 | **Rapp 2026 FI-MS absolute intensities** (Table S5 `Mean_Int` / `R1_Int` / `R2_Int`) | **411** records, 1,385 values | measured: `len(df) = 1385`, 411 genes; `Mean_Int` equals `mean(R1_Int, R2_Int)` on 1,385 of 1,385 rows | none; a distinct `measurement_type`, and it carries a real per-replicate SE |
| 7 | **Rapp 2026 Table S6 `Intensity PrecMz`** | **411** records | same sheet as rank 5 | none, but low value: an instrument-scale intensity with no normalization |
| 8 | **Rapp 2026 Table S7's 2,847 annotated features** | **254** records, 2,847 values | measured: `Table_S7` filtered to `Metabolite != "empty"`, 1,946 distinct (gene, polarity, mass) keys | none, but gated on the Table S5-versus-S7 disagreement above |
| 9 | **Price 2018 Table S1 likely-essential E. coli genes** as `GeneEssentialityPhenotype` | **324** | measured: `orgId == "Keio"` rows of `TableS1_LikelyEssentialGenes`, and measured **disjoint** from the 3,789 fitness genes (intersection 0) | none. Caveat: the paper's own label is "essential or important for growth" in LB at 37 C, and Note 1 puts the FDR at 6% to 16% |
| 10 | **Price 2018 Tables S2 and S3 wild-type carbon and nitrogen growth calls** | **192** | measured: 96 + 96 data rows, E. coli True 38 / False 58 in each | none, **but I would not load these as they stand**: the sheet's own legend makes TRUE a disjunction of a growth observation and a data-availability fact |
| 11 | **Lamoureux 2023 per-sample growth rate** (`metadata_qc.csv` column `Growth Rate (1/hr)`) | **103** of the 241 built records, ceiling 354 | measured intersection of the 241 `p1k_` ids in `preprocess/record_samples.json` with the 354 non-blank cells | blocked in its absolute form by gap 1; storable now as a log2 ratio against a matched wild-type rate, which exists for some conditions only |
| 12 | **Wang 2015 no-isoprenol OD600** as `FitnessPhenotype` | **46** plus 1 reference | measured: 47 Table S3 rows minus the BW25113 reference, the same partition the stored dataset uses | none. **Already a recorded decision** (`notes/...wang2015.md:246`): it would be "a second phenotype family in one dataset", so the open question is whether it becomes a separate dataset |
| 13 | **Schmidt 2016 Table S23 per-condition growth rate + Stdev** | **26** | measured: 26 strain-condition rows (31 non-empty rows minus 2 header and 3 footnote rows) | blocked in its absolute form by gap 1. Caveat: the `Stdev`'s replicate design is not stated, so `n_samples` would be a typed gap |
| 14 | **Schmidt 2016 Tables S2 and S3 SRM absolute abundances**, a different assay of the 41 anchor proteins | up to **22** condition records, 1,461 cell values | measured: 779 + 682 non-empty data rows, each one (protein, peptide, condition) | none; a distinct `measurement_type`. It re-measures proteins Table S6 covers, so it must not be mixed with the stored block |
| 15 | **Caglar 2017 doubling time** as `EnvironmentResponsePhenotype`, `measurement_type=growth_rate` | **19** | measured two ways that agree: 19 distinct non-NA `doublingTimeMinutes` values in si2.csv, and 19 distinct `name` values in si6.csv | blocked in its absolute form by gap 1, and its asymmetric CI by gap 9. The values are already in the raw mirror |
| 16 | **Schmidt 2016 Table S24 deletion-strain growth rates** (WT, dRimI, dRimJ, dRimL in glucose and acetate) | **6** plus 2 references | measured: the sheet read in full, 4 strains x 2 media, 3 replicates except two cells with 2 | none if filed as `FitnessPhenotype` (a ratio to the WT row of the same medium). **The only genuinely new gene-perturbation phenotype in that paper** |
| 17 | **Gupta 2024 absolute protein concentration** (Supplementary Data 6) as `BacterialProteinAbundanceExperiment` | **1** record, 2,994 values | measured: every one of the 2,994 `TableS6` rows has a non-empty `Concentration (M)` | none, but two things must be resolved first and neither is a schema change: `n_replicates` is unsourced (conservative lower end, 1, with a gap), and the release never says which of the 13 conditions the label-free run was |
| 18 | **Price 2018 Table S4 `Solvent`** into the existing stress records | **0 new**, enriches 207,240 | measured: 55 of 147 kept samples are stress, 55 x 3,768 genes | none; fills `SmallMoleculePerturbation.solvent` in place of a typed gap |
| 19 | **Lamoureux 2023 Public K-12 `aerobicity` + `time`** | 0 standalone (1,425 and 391 inside rank 3) | measured fill counts | none; `time` has no unit in its header, so the unit would itself be a gap |
| 20 | **Lamoureux 2023 the 98 short / low-FPKM genes** | **0 new**, 23,618 values on built records | measured: 4,355 minus 4,257 rows | none. Already recorded, and the source's own QC removed them, so this is a note rather than a recommendation |
| 21 | **Lamoureux 2023 the 20 QC-failed libraries** | **<= 20**, uncounted | 1,055 minus 1,035 samples; how many survive the loader's drop rules is not measurable without running the loader | none |

Ranks 1, 2, 3 and 4 are the four worth a decision. Ranks 1 and 2 are each larger than the
entire current *E. coli* store (1,333,813 records measured), rank 2 needs no retrieval
because the file is already mirrored, and rank 2 is the one that must not be landed before
the Nichols 2011 provenance question is settled.

### What could not be checked, across all fourteen

- **Nichols 2011 is not in the literature mirror** (#691). It is the one paper that would
  settle whether Shiver's batch-0 block duplicates a future Nichols loader, and the question
  cannot be answered until it is mirrored.
- **Baba 2006's Supplementary Table 3** is not mirrored (its key holds `paper.pdf`,
  `paper.md` and one `si1.pdf`, with `si_expected: []`), so the Keio growth data four papers
  point at was never read at its source.
- **No figure value was digitized anywhere.** Figure-only quantities were enumerated from
  captions and axis labels, never from a panel. This affects Wang 2015 most (six of its seven
  solvents exist only as Figure 5 bar heights), and also Fuhrer, Goodall, Tong, Rousset,
  Rapp, Wang 2018 and Gupta.
- **Binary images with no SI OCR**: Tong's four TIFs, Rousset's ten PNGs. Their content is
  described only from the main text that cites them.
- **Every PDF was read through its MinerU OCR**, not the PDF bytes. Gupta's si12 Reporting
  Summary OCR is badly mangled; Rapp's Figure 7 caption is absent from the OCR.
- **Unmirrored bulk releases, none fetched** (an audit does no retrieval): Price's `bigfit`
  per-organism files and the 84 GB tarball plus three figshare archives; the PRECISE-1K Zenodo
  archive's member sha256s (its unmirrored members were read from the GitHub tree at the commit
  the archive root names, which is the same content by git identity but not the pinned read
  path); Gupta's PRIDE PXD042444 and Zenodo code, which is the only remaining place the
  molarity conversion and the abundance run's condition could be sourced; Rapp's three MassIVE
  deposits and the Zenodo MD trajectories; Fuhrer's MassIVE MSV000078963 and the two 200 MB
  raw matrices; Cui's GitLab `badSeed_public` notebook, the only place a per-guide quantity
  beyond `fit18`/`fit75` could live; Wang 2018's bioRxiv preprint, the possible home of the
  tiling screen's unreleased fitness table; Caglar's GEO, PRIDE and Texas Data Repository
  deposits and the authors' analysis repository; Goodall's ENA PRJEB24436 and the C code its
  Text S1 says is deposited; Shiver's Dryad per-colony deposit; Tong's CarPE app, which holds
  113,880 kinetic curves that no SI file releases.
- **Three papers cite comparison datasets absent from the mirror**: Sevin and Sauer 2014
  (Fuhrer's osmotic-stress metabolome, reused rather than measured), Donati 2021 (Rapp's parent
  pooled CRISPRi library, with the proteome and metabolome Rapp's 5-fold knockdown claim rests
  on), Rutherford 2010 / GEO GSE16973 (re-plotted as Wang 2015's Figure S1), Joyce 2006
  (Tong's glycerol Keio comparison), Zheng 2013 (the only *E. coli* isoprenol titer anywhere in
  Wang 2015, and the project has no *E. coli* isoprenol production record at all), Li 2014,
  Pedersen 1978, Lu 2006, Ishihama 2005/2008 and Masuda 2009 (Schmidt's six abundance
  benchmarks). Ishii 2007 and Taniguchi 2010 **are** mirrored, with `paper.pdf` and `paper.md`
  only, no `si/` directory, no raw mirror and no loader.

### caglarColiMolecularPhenotype2017 -- `ecoli/caglar2017.py`

**SI inventory.** Mirror `$DATA_ROOT/torchcell-library/caglarColiMolecularPhenotype2017/si/`.
The SI PDF names the file behind every table (`File name: tableSN_...csv`), which confirms
the loader's own mapping comment: PMC object `si<N+1>` is Table S`<N>`.

| file | format | what it is | bytes |
|---|---|---|---|
| `si1.pdf` / `si1.md` | PDF + OCR | the Supplementary Information: Figures S1 to S35 and the "List of Supplementary Tables" captions for S1 to S14 | 1,605,834 / 31,661 |
| `si2.csv` | CSV | **Table S1** `tableS1_meta_data.csv`, 170 data rows x 22 columns, one row per sample | 33,095 |
| `si3.csv` | CSV | **Table S2** `tableS2_mRNA_normalized_raw_data.csv`, 4,196 `ECB_` genes x 152 samples | 11,074,929 |
| `si4.csv` | CSV | **Table S3** `tableS3_protein_normalized_raw_data.csv`, 4,196 `YP_` proteins x 105 samples | 7,732,680 |
| `si5.csv` | CSV | **Table S4** `tableS4_fluxData.csv`, 260 rows, 13 branches x salt x concentration x phase | 20,656 |
| `si6.csv` | CSV | **Table S5** `tableS5_doubling_times.csv`, 55 rows, per-replicate doubling-time fits for 19 conditions | 3,915 |
| `si7.csv` | CSV | **Table S6** `tableS6_clustering_mrna_cophenetic.csv`, 33 rows of mRNA clustering z-scores | 1,369 |
| `si8.csv` | CSV | **Table S7** `tableS7_clustering_protein_cophenetic.csv`, 29 rows of protein clustering z-scores | 1,240 |
| `si9.csv` | CSV | **Table S8** `tableS8_combinedOutputDF_DeSeq.csv`, 201,408 rows of DESeq2 output (48 contrasts x 4,196 genes, measured) | 91,055,296 |
| `si10.csv` | CSV | **Table S9** `tableS9_combinedDifferentiallyExpressedGenes_DeSeq.csv`, 19,197 filtered gene rows | 5,951,696 |
| `si11.csv` | CSV | **Table S10** `tableS10_combinedResultList_DAVID.csv`, 26,454 DAVID KEGG/GO enrichment rows | 17,968,819 |
| `si12.csv` | CSV | **Table S11** `tableS11_changed_protein_carbonSource_ExpSta.csv`, 4,084 gene rows | 243,682 |
| `si13.csv` | CSV | **Table S12** `tableS12_changed_DAVID_P05.csv`, 81 enrichment rows | 170,257 |
| `si14.csv` | CSV | **Table S13** `tableS13_flux_vs_conc_Pvalues.csv`, 26 flux-vs-concentration regression rows | 2,495 |
| `si15.csv` | CSV | **Table S14** `tableS14_flux_vs_doublingTime_Pvalues_tog.csv`, 13 flux-vs-doubling-time regression rows | 1,631 |

Also present and skipped for the reader: `si1_middle.json`, `si1_content_list.json`,
`si1_ocr_provenance.json`, and `si/images/si1/`.

**File hazard, measured.** `si14.csv` has CR-only line terminators (`file`: "ASCII text,
with CR line terminators"; `wc -l` returns 0; 26 lines after `tr '\r' '\n'`). A naive
`pandas.read_csv` sees one row. Every other SI CSV is LF.

**What the loader stores.** Two registered classes, both wild-type REL606, both already
built and L0-L4 verified (`notes/torchcell.datasets.ecoli.caglar2017.md`, heading
"2026.10.07 - The RNA-seq and proteome loaders"):

- `RnaseqCaglar2017Dataset`, `RNASeqExpressionPhenotype`, Tables S1 + S2, **152 records**,
  `expression_count` (reconstructed HTSeq count) and `expression_tpm` (derived here).
- `ProteomeCaglar2017Dataset`, `ProteinAbundancePhenotype`, Tables S1 + S3 + 42 NCBI
  GenPept batches, **105 records**, `protein_abundance`
  (`lcmsms_spectral_count_deseq2_size_factor_normalized`).

Table S1 columns the loader consumes are listed at `caglar2017.py:1509-1521` (`COL_SAMPLE`
to `COL_PROTEIN`): `dataSet`, `experiment`, `growthTime_hr`, `batchNumber`,
`carbonSource`, `Mg_mM`, `Mg_mM_Levels`, `Na_mM`, `Na_mM_Levels`, `growthPhase`,
`uniqueCondition`, `RNA_Data_Freq`, `Protein_Data_Freq`. The raw mirror holds Tables S1 to
S4 only (`SI_TABLES`, `caglar2017.py:823-849`); Tables S5 to S14 are not mirrored.

**Already recorded as not stored.**

- **Flux ratios, Table S4.** Loader module docstring, `caglar2017.py:71-72`: "The flux arm
  (Table S4) is not loaded: it holds flux RATIOS, which neither ``MetabolitePhenotype``
  (pool sizes) nor ``FluxPhenotype`` (signed net flux) can store." Also note heading
  "Per-family design for the loader" bullet **Flux**, and "Gaps carried forward" item 2 of
  the 2026.10.07 loader section. Retired, no new finding.
- **`SDEFluxRatio` is SD or SE, undefined in the mirror**, and which flux condition had two
  replicates rather than three is not named. Note "Gaps carried forward" item 2 (both
  sections). Retired.
- **Table S4 carries stationary-phase rows** although the Results analyze exponential only.
  Note "Per-family design", **Flux** bullet. Retired. (Measured here: 130 EXP and 130 STA
  rows of `si5.csv`.)
- **Growth phase has no typed `Environment` slot.** Note heading "Environment", "**Growth
  phase has no slot**", and "Gaps carried forward" item 1. Retired, and it is an
  environment axis, not a phenotype.
- **`n_mapped_reads`** is a typed gap resolving to GEO GSE94117. Note "What is stored",
  RNA-seq bullet, and "Gaps carried forward" item 3. Retired.
- **Technical-replicate counts** `RNA_Data_Freq` / `Protein_Data_Freq` travel in
  `record_samples.json`; whether the 12 two-run protein samples were summed or averaged is
  not stated. Note "What is stored", Replicates bullet, and item 5. Retired.
- **The two QC-flagged RNA samples** (`MURI_091`, `MURI_130`) are kept, as the authors kept
  them. Note "What is stored", QC bullet. Retired.
- **Raw deposits not mirrored**: GEO GSE94117, PRIDE PXD005721, Texas Data Repository
  doi:10.18738/T8/UG3TUR. Loader `si_expected`, `caglar2017.py:1058-1065`. Retired.

**Enumerated released quantities.** Column names are verbatim from the files' real headers.

| released quantity | source file + column | stored | classification | evidence |
|---|---|---|---|---|
| normalized, log-transformed mRNA count | `si3.csv` (Table S2), one column per `MURI_*` sample | yes | stored | caption "Supplementary Table S2: Normalized mRNA counts. Includes data for 4196 distinct proteins each for 152 samples." |
| normalized, log-transformed protein count | `si4.csv` (Table S3), one column per `MURI_*` sample | yes | stored | caption "Supplementary Table S3: Normalized protein counts. Includes data for 4196 distinct proteins each for 105 samples." |
| sampling time | `si2.csv` `growthTime_hr` | yes | stored as `Environment.duration_hours` | `SAMPLE_TIME_COLUMN`, `caglar2017.py:445-450` |
| technical-replicate counts | `si2.csv` `RNA_Data_Freq`, `Protein_Data_Freq` | ledger | already recorded | caption "number of RNA samples (technical replicates), number of protein samples (technical replicates)" |
| batch | `si2.csv` `batchNumber` | ledger | not a phenotype (design/annotation) | `replicate_groups.json` |
| carbon source, Mg2+, Na+ levels | `si2.csv` `carbonSource`, `Mg_mM`, `Na_mM`, `Mg_mM_Levels`, `Na_mM_Levels` | yes | not a phenotype (environment design) | note heading "Environment" |
| growth phase | `si2.csv` `growthPhase` | no | already recorded (no typed slot) | note "Gaps carried forward" item 1 |
| **doubling time, per condition** | `si2.csv` `doublingTimeMinutes`; `si6.csv` (Table S5) `doubling.time.minutes` | **no** | **loadable now: `BacterialEnvironmentResponseExperiment` + `EnvironmentResponsePhenotype.environment_response`, `measurement_type=growth_rate`, `assay_type=liquid_od_growth`, `sample_unit=biological_replicate`** | abstract "Additionally, we provide measurements of doubling times and in-vivo metabolic fluxes through the central carbon metabolism."; caption "Supplementary Table S5: Doubling time measurements in exponential phase." |
| **95% confidence interval of the doubling time** | `si2.csv` `doublingTimeMinutes.95m` + `doublingTimeMinutes_95p`; `si6.csv` `doubling.time.minutes.95m` + `.95p` | **no** | **blocked**: the released interval is ASYMMETRIC and `EnvironmentResponsePhenotype` carries one scalar `environment_response_uncertainty` with `UncertaintyType.ci95`, which `derive_se` reads as a half-width (`schema.py:3526-3527`). Same missing-interval-field shape as **#753** item 1 (`ProteinTurnoverPhenotype` has no interval field), one class over | measured on `si2.csv`: max `\|(95p - dt) - (dt - 95m)\|` = 5.9504 min; on `si6.csv`: 1150.7309 min, and 1 row has an upper bound BELOW the estimate (`Glycerol.tab` replicate 1, `.95p` = `-1027.769034`) |
| r-squared of the doubling-time fit | `si2.csv` `rSquared`; `si6.csv` `r.squared` | no | not a phenotype (goodness-of-fit diagnostic of the stored estimate; no `Phenotype` field holds it) | caption "r2 from the linear fit to OD600 values" |
| per-replicate doubling-time fit | `si6.csv` `replicate` + `doubling.time.minutes` | no | loadable now, as the `n_samples` basis of the record above (not as 55 separate records: the env-response verifier's L1 `pair_uniqueness` is one record per strain x condition) | 55 data rows over 19 `name` values, measured by `pandas` groupby on `si6.csv` |
| harvested cell number | `si2.csv` `cellTotal`, `cellsPerTube` | no | not a phenotype (harvest bookkeeping, and UNDOCUMENTED: neither column appears in the Table S1 caption) | Methods "aliquots of these cultures were removed as necessary to harvest a constant number of cells given the changes in cell density over the growth curve."; measured: 131 of 171 rows non-NA, `cellTotal` 1.13e8 to 4.6e10, `cellTotal/cellsPerTube` is 5 or 10 |
| harvest date | `si2.csv` `harvestDate` | no | not a phenotype (metadata; 62 distinct dates) | caption "harvest date" |
| condition grouping keys | `si2.csv` `uniqueCondition`, `uniqueCondition02` | partly | not a phenotype (annotation; 35 and 17 distinct) | loader uses `uniqueCondition` for replicate groups only |
| mean flux ratio + its dispersion | `si5.csv` `MeanFluxRatio`, `SDEFluxRatio` | no | already recorded | loader docstring `caglar2017.py:71-72` |
| mRNA / protein clustering z-scores | `si7.csv`, `si8.csv` `Overall_Z_score`, `Z_score`, `num_elements` | no | not a phenotype (derived summary of the stored matrices: cophenetic distances of a clustering) | Table 1 legend "The z scores represent mean cophenetic distances between all pairs of conditions with the same label, normalized by the distribution of mean distances obtained after randomly reshuffling condition labels." |
| DESeq2 differential expression per gene x contrast | `si9.csv` `baseMean`, `log2FoldChange`, `lfcSE`, `stat`, `pvalue`, `padj`, `signChange` | no | not a phenotype (derived from the stored counts by DESeq2) | caption "Supplementary Table S8: Combined results from tests for differential expression for all genes and all distinct tests considered."; measured: 201,408 rows = 48 (dataType, effect, contrast, phase) groups x 4,196 genes |
| filtered differentially expressed gene list | `si10.csv` `genes`, `thresholds_for_input_genes` | no | not a phenotype (a filter over `si9.csv`) | caption "Supplementary Table S9: Filtered version of Supplementary Table S8, retaining only genes with P < 0.05 and log2FoldChange > 2." |
| DAVID KEGG/GO enrichment statistics | `si11.csv` `Count`, `PValue`, `Fold.Enrichment`, `Bonferroni`, `Benjamini`, `FDR_KEGG_Path`, `padj_gene`, `log2`, `score_gene`, `rank` | no | not a phenotype (pathway enrichment / annotation) | caption "Supplementary Table S10: Complete results from DAVID enrichment analysis for KEGG pathways and molecular functions." |
| doubling-time-controlled extra protein list | `si12.csv` `genes`, `exist_in` | no | not a phenotype (derived gene list) | caption "Supplementary Table S11: List of the additional proteins identified as differentially expressed when controlling for doubling time" |
| enrichment of that list | `si13.csv` `Count`, `PValue`, `Fold.Enrichment`, `Bonferroni`, `Benjamini`, `FDR` | no | not a phenotype (enrichment) | caption "Supplementary Table S12: Enriched KEGG pathways and molecular functions based on the genes listed in Supplementary Table S11." |
| flux-vs-ion-concentration regression | `si14.csv` `estimate`, `std.error`, `statistic`, `p.value`, `p.adjusted` | no | not a phenotype (regression over Table S4) | caption "Supplementary Table S13: Results from linear regressions of flux ratios against ion concentrations (Mg2+ and Na+)." |
| flux-vs-doubling-time regression | `si15.csv` `estimate`, `std.error`, `statistic`, `p.value`, `p.adjusted` | no | not a phenotype (regression over Table S4) | caption "Supplementary Table S14: Results from linear regressions of flux ratios against doubling times" |

**The doubling time is the finding, and it is new.** The word "doubling" appears in the
loader only at `caglar2017.py:317` (`DOUBLING_TIME_REPLICATES`, a `SourcedValue` that is
defined and never used in a record) and at `caglar2017.py:829` (the `SI_TABLES` description
of Table S1). It appears in the note only in the Table S1 content row and in the
"Replicate structure" bullet list. **Nothing in the loader or the note records a decision
not to store it**, so unlike the flux arm this is not retired. It is one of the paper's
three headline measurements: "Additionally, we provide measurements of doubling times and
in-vivo metabolic fluxes through the central carbon metabolism." (abstract) and "Finally,
we measured doubling times in exponential phase for all experimental conditions
(Supplementary Table S5)." (Results).

Everything the schema needs already exists: `MeasurementType.growth_rate`,
`AssayType.liquid_od_growth`, `SampleUnit.biological_replicate`, `UncertaintyType.ci95`,
and the wrapper pair `BacterialEnvironmentResponseExperiment` /
`BacterialEnvironmentResponseExperimentReference` (`schema.py:5495-5508`). The method is
quotable: "Doubling times were calculated as log_e 2 divided by the fit slope for each
biological replicate separately. Means and confidence intervals were calculated from three
replicate growth curves for all conditions except for gluconate and lactate, which had
measurements for only two replicates." (Methods, Cell Growth), on OD600 curves grown
"separately from the main batches used for harvesting cells but under identical
conditions".

Two obstacles that are verifier rules, not schema gaps, and both are already named
elsewhere in the repo:

1. **An absolute rate cannot pass L3 `reference_zero`.** The environment-response verifier
   requires the reference response to be numerically 0
   (`torchcell/verification/environment_response.py:18-21`, `387`). `lim2025.py:113-118`
   already records this exactly: "``MeasurementType.growth_rate`` exists, but
   ``torchcell.verification.environment_response``'s L3 ``reference_zero`` requires a
   numeric reference of exactly 0, which an absolute rate in h^-1 cannot be." The
   no-change encoding is therefore `log2(doubling_time / base_condition_doubling_time)`
   with `measurement_type=log2_ratio`, the Lim 2025 and Wang 2015 pattern, and the paper
   defines that baseline: "The red points and dashed orange lines represent the doubling
   time at the base condition (glucose, 5 mM Na+, 0.8 mM Mg2+)" (Fig. 2 legend). Storing
   the minutes verbatim needs the verifier to accept a declared non-zero baseline, which
   `lim2025.py` says is in its PR body.
2. **The base condition itself carries no environmental edit**, so an L3
   `environment_perturbed` record for it would fail; it is the reference, which is what the
   RNA and protein loaders already do with the same condition
   (`REFERENCE_CONDITIONS`, `caglar2017.py:451-459`).

**The two released doubling-time tables disagree, and a loader has to pick one.** This is
measured, not inferred. Pairing all 19 Table S5 `name` values onto Table S1 by
(`experiment`, `carbonSource`, `Mg_mM`, `Na_mM`) is one-to-one and covers all 165 Table S1
rows that carry a doubling time:

- 5 of 19 conditions match exactly (`< 1e-6`) the HARMONIC mean of Table S5's per-replicate
  fits, that is the mean growth rate converted back to a time: `Glycerol.tab` and the four
  `MgSO4-2_000.0{10,20,40,80}_mM.tab` low-Mg conditions.
- The other 14 do not match any Table S1 value, arithmetic, harmonic or geometric. The
  largest gap is `NaCl_100_mM.tab`: Table S5 harmonic mean 58.800521 min against Table S1's
  66.035662 min, a difference of 7.235141 min (11.0% of the Table S1 value). Next are
  `NaCl_200_mM` 5.376736, `NaCl_300_mM` 5.189551, `NaCl_005_mM` 4.411182 min.
- The fit qualities also differ: Table S1 `rSquared` runs 0.859135 to 0.993397, Table S5
  `r.squared` runs 0.957308 to 0.999800.

Counting rule, so this is reproducible without the script: pair every Table S5 `name`
onto Table S1 by (`experiment`, `carbonSource`, `Mg_mM`, `Na_mM`), then compare Table S1's
`doublingTimeMinutes` against the arithmetic, harmonic and geometric mean of that
condition's Table S5 `doubling.time.minutes` values.
Hypothesis (untested): Table S1 reports a fit pooling a condition's replicate OD600
curves (its caption says "r2 from the linear fit to OD600 values", one r-squared per
condition) while Table S5 reports the per-replicate fits the Methods describe, and the
NaCl and high-Mg series were refit between the two tables. The mirror does not say.

**Comparisons against other data.**

- **Duplication risk, already recorded.** The glucose time-course columns of Tables S2 and
  S3 are the authors' own earlier release: "Results from one of these conditions, long-term
  glucose starvation, have been presented previously10." (ref 10 = Houser et al. 2015, PLOS
  Comput Biol 11, e1004400; GEO GSE67402, PRIDE PXD002140). The note already records this
  (`GLUCOSE_TIME_COURSE_PRIOR`, `caglar2017.py:461-469`, and checklist item 7: "ref 10 has
  no loader"). So the risk is one-directional and latent: loading Houser 2015 later would
  re-serve samples that are already columns of these two LMDBs. Nothing to change now; it
  belongs in a Houser 2015 admission check, the same way **#760** handled Rousset against
  Cui.
- **No numeric benchmark against another lab's data.** The Discussion compares SCOPE only:
  "Our data provides a comprehensive picture of E. coli in terms of number, range, and
  depth of different stresses, comparable and complementary to other recently published
  datasets. For example, Schmidt et al.12 considered 22 unique conditions and measured
  abundances of >2300 proteins. mRNA abundances were not measured. Soufi et al.11
  considered 10 unique conditions and also measured abundances of >2300 proteins. ...
  Lewis et al.13 considered only 3 different carbon sources but measured mRNA and protein
  abundances in different strains adapted to these growth conditions. Finally, Lewis et
  al.14 compiled a database of 213 mRNA expression profiles covering 70 unique conditions
  ... In comparison, we considered 34 unique conditions, measured 152 mRNA expression
  profiles, 105 protein expression profiles, and 59 flux profiles, and used the exact same
  E. coli genotype throughout." Those four are loadable E. coli condition-series datasets
  we do not have: ref 12 Schmidt et al. 2016 Nat. Biotechnol. 34, 104-110 (22 conditions,
  >2,300 absolute protein abundances, the natural `ProteinAbundancePhenotype` companion);
  ref 11 Soufi et al. 2015 Front. Microbiol. 6, 103; ref 13 Lewis et al. 2010 Mol. Syst.
  Biol. 6, 390; ref 14 Lewis et al. 2009 J. Bacteriol. 191, 3437-3444. They are candidates
  to NAME, not to add: adding a reference is a curation decision that needs an explicit
  go-ahead.
- **No K-12 overlap.** "used the exact same E. coli genotype throughout", and the strain is
  REL606 (E. coli B), so none of the K-12 rows of the fifty can hold these samples. The
  note already measured the PRECISE-1K check as null (checklist item 7).
- **Qualitative cross-reference only** to Soufi: "Such genes were similarly down-regulated
  in our study during stress induced by high Na+ concentrations." No shared values.

**A paper cross-reference that does not resolve in the release.** "105 protein samples, and
65 flux samples (Supplementary Table S1). 59 of the flux samples are associated with high
Mg2+ and high Na+ experiments." (Results). Measured: the released Table S1 (`si2.csv`) has
22 columns and none of them counts flux samples (there is `RNA_Data_Freq` and
`Protein_Data_Freq`, no flux analogue). So the per-sample flux design the text points at is
not in the mirror, and the flux arm's `n_samples` would still rest on the prose
(`FLUX_REPLICATES`, `caglar2017.py:300-308`) even if the ratio phenotype existed. This
strengthens, rather than changes, the already-recorded flux gap.

**Loadable-now estimate.**

- **Doubling time: 19 records.** Basis, measured two ways that agree: `si2.csv` has 19
  distinct non-NA `doublingTimeMinutes` values, and grouping its 165 doubling-time rows by
  (`experiment`, `carbonSource`, `Mg_mM`, `Na_mM`) gives the same 19 cells; `si6.csv` has 19
  distinct `name` values. Each record aggregates 3 biological replicates, except
  `Gluconate.tab` and `Lactate.tab` with 2 (Methods quote above; 55 replicate rows in
  `si6.csv` over 19 conditions). Of the 19, 3 are the glucose / 0.8 mM Mg2+ / 5 mM Na+ base
  condition measured in three separate experiment series (`glucose_time_course`,
  `MgSO4_stress_high`, `NaCl_stress`), with three different numbers (53.679955, 63.608097,
  62.596307 min), so they are the three references, one per series, not one.
- Not 165. Attaching the doubling time to each omics sample would write 19 numbers into 165
  records and collide with L1 `pair_uniqueness`.
- Not 55 either, for the same rule; the 55 per-replicate fits are the `n_samples` basis and
  belong in `preprocess/`.
- **Retrieval cost.** `doublingTimeMinutes` is already in the raw mirror the loader consumed
  (`data/srep45303-s2.csv`, sha256 `1486290b...`), so the 19 values need no new fetch.
  Table S5's per-replicate fits are NOT mirrored and would need one recorded retrieval of
  PMC object `PMC5394689.1/srep45303-s6.csv`, which the provenance rule for a later
  revision allows because the SI lists the file by name.
- No other loadable-now item. Every remaining unstored quantity is either already recorded
  (flux) or classified not a phenotype.

**Not checked.**

- `si9.csv` (91 MB), `si11.csv` (18 MB), `si10.csv` (6.0 MB): header plus a streaming pass
  for the contrast structure of `si9.csv` only. I did not read their value distributions,
  per the briefing's instruction for huge files. Nothing in the classification depends on
  those values: all three are DESeq2 or DAVID output over the two matrices the loader
  already stores.
- `si3.csv` and `si4.csv` value contents: not re-measured here. The note already reports the
  full back-solve on both, and the loader's L0-L4 verifiers pass.
- `si1.pdf` figures S1 to S35: read only through `si1.md`'s captions. Figure S35 plots the
  Table S4 flux ratios and Figure 2 the Table S5 doubling times, so both figures are views
  of tables I did read; no figure-only quantity was identified.
- GEO GSE94117, PRIDE PXD005721, the Texas Data Repository GC-MS deposit
  (doi:10.18738/T8/UG3TUR) and the authors' analysis repository
  `https://github.com/umutcaglar/ecoli_multiple_growth_conditions` ("All processed data and
  analysis scripts are available on github") are outside the mirror and were not fetched.
  The first three are recorded in the loader's `si_expected`; the GitHub repository is not,
  and it is the stated home of the processing scripts.
- Whether any of the 19 doubling times is already served by another bacterial dataset: not
  measured. Hypothesis (untested): none, since no other row is REL606 and the note's
  PRECISE-1K check found no E. coli B strain.

### fuhrerGenomewideLandscapeGene2017 -- `ecoli/fuhrer2017.py`

Audit only. No loader, schema, conf, note or store was changed; no build, slurm job or
`git` command was run.

**SI inventory** (`$DATA_ROOT/torchcell-library/fuhrerGenomewideLandscapeGene2017/si/`).
Sidecars `si1_middle.json`, `si1_content_list.json`, `si1_ocr_provenance.json` and the
same three for `si6` also exist, plus `si/images/si1/` (13 jpgs) and `si/images/si6/`
(7 jpgs); they are OCR layout artifacts, not released data.

| file | format | bytes | what it is |
|---|---|---|---|
| `si1.pdf` + `si1.md` | PDF + MinerU OCR | 1,424,240 + 7,799 | "Expanded View Figures": captions of Figures EV1 to EV10. No data table. |
| `si2.xlsx` | xlsx, 4 sheets | 1,854,096 | Table EV1. `Legend EV1A`, `Legend EV1B`, `Table EV1A` (4,320 Keio strains, 8 columns, header on row 4), `Table EV1B` (7,534 ion rows, up to 54 columns). |
| `si3.xlsx` | xlsx, 2 sheets | 11,130 | Table EV2. `Legend Table EV2` + `Table EV2`: a 12-row annotation-coverage summary of the KEGG eco and Orth 2011 databases. No per-strain row. |
| `si4.zip` | zip, 1,276 entries | 2,952,907 | Table EV3. `index.html` (one row per y-gene, 1,273 rows x 17 data columns), `Table_EV3_Legend.docx`, and `details/data_<gene>.html` x 1,273. |
| `si5.xlsx` | xlsx, 4 sheets | 513,698 | The ion-annotation reliability table. Its own legend sheet says `Legend Table EV5`; the paper's Methods cite the same content as `Table EV4` (see "Comparisons"). `negative mode` 2,191 data rows, `positive mode` 1,828 data rows, 13 columns each, `Sheet3` empty. |
| `si6.pdf` + `si6.md` | PDF + MinerU OCR | 290,754 + 42,480 | MSB review process file: referee reports, author responses, and the EMBO author checklist. |

The per-strain matrices are NOT in the publisher SI. The paper deposits them in
BioStudies S-BSST5, and the raw mirror
(`$DATA_ROOT/torchcell-raw/fuhrerGenomewideLandscapeGene2017/`) holds
`zscore_neg.tsv`, `zscore_pos.tsv`, `sample_id_zscore.xls`, `sample_id_all.xls`,
`neg_ionMz.xls`, `pos_ionMz.xls`, `S-BSST5.json` and a copy of `si2.xlsx`.

**What the loader stores.** `MetabolomeFuhrer2017Dataset`
(`torchcell/datasets/ecoli/fuhrer2017.py:1216`), `BacterialMetaboliteExperiment` /
`...Reference` with `MetabolitePhenotype`, `measurement_type =
"fia_tof_ms_ion_modified_z_score"` (`fuhrer2017.py:511`). The value is the deposit's
summarized modified z-score, keyed `neg_0001`..`neg_3169` / `pos_0001`..`pos_4365` (the
1-based ion row = Table EV1B's `Ion Index`). Record count **3,735**, stated in
`notes/torchcell.datasets.ecoli.fuhrer2017.md` under "Records and drops" and again under
"Build and verification" ("3,735 records, 1 reference group, 83 s"). `n_replicates` = 2
clones per strain; `metabolite_level_se` is a typed `ProvenanceGap`
(`fuhrer2017.py:522`). Table EV1A is read only for the strain join:
`read_table_ev1a` (`fuhrer2017.py:734`) builds `KeioEntry` from four columns --
`jw_id`, `blattner_id`, `gene_name`, `delivery_status` (`fuhrer2017.py:691-698`).

**Already recorded as not stored.** Retired, not re-reported below.

- Per-replicate raw ion intensities (`rawdata_neg_all.tsv`, `rawdata_pos_all.tsv`, 219 MB
  - 300 MB): `fuhrer2017.py:294` constant `NOT_MIRRORED`, and the note's "Where the data
  is" ("Not mirrored, because nothing reads them").
- The deposit's putative KEGG annotation files (`neg_kegg_all_3mD.xls`,
  `pos_kegg_all_3mD.xls`): same `NOT_MIRRORED` constant, `fuhrer2017.py:298`.
- Per-strain uncertainty on the metabolite level: typed gap, `fuhrer2017.py:522`
  (`METABOLITE_LEVEL_SE_GAP`) and the note's "Uncertainty type: none released, a typed
  gap".
- The dataset-level replicate-noise statistic (z = 2.765):
  `SOURCED_VALUES["replicate_noise"]`, `fuhrer2017.py:479` -- "a dataset-level noise
  statistic over all strains and ions, not a per-record uncertainty; it is not stored on
  any record".
- Table EV1B's putative compound annotations: the note's "Open, and what is a gap" --
  "`target_metabolite_ids` is None: ions carry ambiguous putative annotations (EV1B), and
  linkage to iML1515 is not decided."
- The 25 pooled names and the 46 unresolved JW ids: the note's "Records and drops" table
  and `dropped_records.json`.
- Culture format (1 ml, 96-deep-well, 300 rpm): the note's "Open, and what is a gap".
- Median-vs-mean replicate summary: recorded conflict,
  `SOURCED_VALUES["replicate_summary"]`, `fuhrer2017.py:470`.

**The growth rate is NOT in that ledger.** The note mentions it once, descriptively, at
line 28 ("Table EV1A (`si/si2.xlsx`) is the strain list with growth rates") and never
again; `grep -n -i growth` over the note returns only that line and an unrelated
"mid-exponential growth phase" quote. No retention rule, `si_expected` entry, provenance
gap or open item covers it. It is the one genuinely unrecorded released phenotype.

| released quantity | source file + column | stored | classification | evidence |
|---|---|---|---|---|
| Per-strain growth rate, mean over 2 clones | `si2.xlsx` sheet `Table EV1A`, merged header `Growth-ratesc` (row 2) -> pandas column `Mean` with `header=3` | no | **blocked**: no schema class can hold an UNPERTURBED ABSOLUTE rate. `FitnessPhenotype.fitness` is typed "ko_growth_rate/wt_growth_rate" with `reference_one` = 1.0 and non-negative values; Fuhrer releases no WT rate (measured: no wild-type row in EV1A) and 3 kept values are negative. `EnvironmentResponsePhenotype` + `MeasurementType.growth_rate` is the only typed home for an absolute rate, but the environment-response verifier's L3 `reference_zero` requires a numeric reference of exactly 0 and L3 `environment_perturbed` requires a genuine environmental edit on every record. NOT one of the five open issues; the `reference_zero` half is already stated in `torchcell/datasets/pputida/lim2025.py:113-119` | Footnote c of the same sheet (row 4330): "cMean and standard deviations are calculated from duplicate cultivations(each clone per knock-out strain from the strain collection)". Methods: "Growth rates were calculated as the slopes of linear fits to log-transformed experimentally determined OD values." Measured: 4,319 of 4,320 rows numeric, min -0.41, median 0.81, max 1.88 |
| Per-strain growth-rate sample SD | same sheet, pandas column `Standard Deviation` | no | **blocked** with the item above: it is the uncertainty of a value that has no home. Once a home exists it is `..._uncertainty` + `UncertaintyType.sample_sd`, `n_samples = 2`, `SampleUnit.biological_replicate` | Same footnote c. Measured: 3,995 of 4,320 numeric, 324 are the literal string `n.d.`; min 0.01, max 1.17 |
| Keio delivery status | `si2.xlsx` `Table EV1A` column `Keio Delivery Statusb` | read, not stored | not a phenotype: collection metadata, and the loader uses it as a build gate (`READY_STATUS`, `fuhrer2017.py:513`, `UndistributedKeioEntryError`) | Legend EV1A: "3870 strains are currently classified as 'ready to deliver', 450 strains as 'not current JW ORF, eliminated wrong primers or eliminated weak growth." Measured counts match exactly: 3,870 / 418 / 20 / 11 / 1 |
| Gene functional annotation | `si2.xlsx` `Table EV1A` column `Annotationa` | no | not a phenotype: an annotation | Footnote a (row 4328): "aAnnotation is based on \"Global functional atlas of Escherichia coli encompassing previously uncharacterized proteins., Hu et al., PLoS Biology, 2009\"" |
| `Sample Number`, `JW ID`, `Blattner ID`, `Gene Name` | `si2.xlsx` `Table EV1A` | JW id stored as `construction.strain_accession` | not a phenotype: identifiers | `fuhrer2017.py:691-698` |
| Ion m/z and ion index | `si2.xlsx` `Table EV1B` columns `Ionization Mode`, `Ion Index`, `m/z` | yes | stored: the record key plus `preprocess/ions.csv` | note, "Phenotype, n and the uncertainty type"; measured 3,169 neg + 4,365 pos = 7,534 rows |
| Putative ion annotations | `si2.xlsx` `Table EV1B` column `Annotations (Compound Name, Modification, Sum-Formula, KEGG Code, HMDB Code)` | no | not a phenotype: an annotation. Already recorded (note, "Open, and what is a gap", `target_metabolite_ids`) | Legend EV1B: "Putative annotations are indicated based on accurate mass with 3mD tolerance." |
| Annotation-coverage counts (compounds in database, theoretical ions, unknown / annotated ions, matched formulas) | `si3.xlsx` sheet `Table EV2`, 12 rows x 4 numeric columns | no | not a phenotype: database-level summary, no strain or sample attached | sheet rows 5 to 12, e.g. "Ions (centroids) detected \| 4365 \| 3169 \| 4365 \| 3169" |
| CLR profile-similarity: `hits`, `top hit`, `top score`, `Prediction-score` | `si4.zip` `index.html` header group `CLR`, 1,273 gene rows | no | not a phenotype: a derived summary of the stored z-score matrix (a gene-gene profile-similarity index and a composite ranking score), plus annotation | `Table_EV3_Legend.docx`: "Orange columns report: number of gene deletions with significantly similar metabolome profiles; gene deletion with the highest similarity and corresponding similarity score; prediction score calculated as explained in the material and method section." |
| Differential-ion summary: `hits`, `top hit`, `top score` | `si4.zip` `index.html` header group `DIFF IONS` | no | not a phenotype: a count and a max over values we already store (the AUC-weighted z-score of the strain's largest-changing annotated ion) | Legend docx: "Grey columns report: number of individuated differential ions; annotated metabolite with the largest change and corresponding Z-score." Methods: "The changes in ion abundance (i.e., $z$ -score) were then weighted by corresponding AUC indices (Table EV3, DIFF IONS hits, top hit, and top score)." |
| KEGG-pathway and COG enrichment: `hits`, `top hit`, `top score` (a p-value) | `si4.zip` `index.html` header groups `KEGG PATHWAYS by CLR` and `COG` | no | not a phenotype: enrichment statistics over annotation sets | Legend docx: "Light green columns report: Number of significantly enriched KEGG metabolic pathways among most similar gene deletions; most significant enriched pathway and corresponding significance by means of p-value." |
| Metabolite-prediction summary: `hits`, `top hit`, `top score`, `overlap` | `si4.zip` `index.html` header group `MET Prediction` | no | not a phenotype: a prediction and an overlap indicator (0 or 1) | Legend docx: "Dark green columns report: Number of significantly enriched metabolites among those directly linked (e.g. substrate or products) to most similar gene deletions; most significant overrepresented metabolite and whether the metabolite is also detected as significantly changed in our metabolome screen (i.e. 0 or 1)." |
| Per-partner `CLR_index`, per-pathway `pvalue_MS` / `qvalue_MS`, differential-ion lists | `si4.zip` `details/data_<gene>.html` x 1,273 | no | not a phenotype: the same derived quantities, expanded per gene | read from `details/data_yaaA.html`: "CLR Gene_matching CLR_index yadC 5.8 yfjU 5.5 ... Pathway_MS pvalue_MS qvalue_MS Valine, leucine and isoleucine biosynthesis 0.002 0.1416" |
| Per-ion-annotation `score`, `rank`, `mzDelta`, `TIC correl`, `average int`, `AUC`, `Z-cutoff` | `si5.xlsx` sheets `negative mode` (2,191 rows) and `positive mode` (1,828 rows); the `ion` column is the Ion Index, so it joins our `neg_NNNN` / `pos_NNNN` keys | no | not a phenotype: annotation-reliability metrics per (compound, ion, adduct), not per strain. `average int` is a per-ion mean over all samples of the raw matrices, a scale the stored z-score standardization already removes | `Legend Table EV5`: "For each annotated ion, we ranked enzyme deletions according to their respective absolute Z-score. Then we performed an ROC analysis ... The AUC score reflect whether annoated enzymes directly linked to the metabolite are eliciting the strongest changes, and it is used as a measure of annotation of reliability." |
| Modified z-score per (strain, ion) | `zscore_neg.tsv`, `zscore_pos.tsv` | yes | stored: `MetabolitePhenotype.metabolite_level` | note, "Phenotype, n and the uncertainty type" |
| Biological-replicate rank correlation (mean rho 0.1059) | `si6.md` author response | no | not a phenotype: a dataset-level reproducibility statistic, and the authors say so | si6.md line 181: "The mean rho is 0.1059. This information wasn't included because rho is not a good benchmark for reproducibility given that we measured thousands of features but only a little fraction is significantly associated to a given genotype." |
| Four OD600 readings per culture, and the approximate harvest OD | NOT RELEASED in any file | no | measured but unreleased: no file in the SI or in S-BSST5 carries an OD value | Methods: "Growth was followed via absorbance at $6 0 0 ~ \mathrm { { n m } }$ measured at four time-points"; "Approximate harvest OD values were estimated by the calculated growth rates and harvest time." Verified by listing all 10 S-BSST5 files in `S-BSST5.json`: none is an OD file |
| Deoxycholate / iron / enterobactin dose-response maximum growth rates, triplicate, for WT and named mutants | FIGURE ONLY (Fig 6B, Fig 6C, Figs EV10A, EV10B) | no | not released as a table: no SI file carries these numbers | Methods: "Maximum growth rates during exponential growth phase were calculated from triplicate cultivations." Figure 6 caption: "Error bars represent standard deviations from three biological replicates." |
| Succinate-dehydrogenase activity (3 replicates) and succinate / galactose / gluconate growth defects (>= 2 replicates) for `ygfY`, `yidK`, `yidR`, `sdhC` | FIGURE ONLY (Figs EV8B, EV8C, EV8D) | no | not released as a table | si1.md, Figure EV8: "Data are shown as mean and standard deviation of three replicates"; "Data are shown as mean and standard deviation of two replicates" |
| WT metabolome under nutrient limitation and stress (phosphate, sulfur, anaerobic, iron, osmotic, deoxycholate), triplicate | NOT RELEASED: not in the SI and not among the 10 S-BSST5 files | no | measured but unreleased (figure-only, Fig 6A and Fig EV9) | Methods: "Perturbation experiments were performed in $5 ~ \mathrm { m l }$ cultures ... $1 \ \mathrm { m l }$ of cells at $\mathrm { O D } 0 . 5$ was harvested in triplicates by fast filtration". File list verified in `S-BSST5.json` |
| Raw MS profile spectra, > 34,000 analyses | MassIVE `MSV000078963`, not mirrored | no | not a phenotype: instrument spectra, not a per-record table | Data availability: "Profile data for $> 3 4 , 0 0 0$ mass spectrometric analysis can be downloaded from <http://massive.ucsd.edu/>, accession code: MSV000078963." |

**Comparisons against other data.**

- **A reused dataset from another paper, which we do not have.** Osmotic-stress
  metabolome data in this paper is not this paper's measurement: "Metabolomics
  measurements were performed by flow-injection TOF-MS as described previously (Fuhrer
  et al, 2011), and for osmotic stress $5 0 0 ~ \mathrm { m M }$ NaCl), previously
  published data were used (Sevin & Sauer, 2014)." Sevin and Sauer 2014 (Nat Chem Biol
  10:266) is a candidate E. coli metabolome source; it is NOT in the literature mirror
  (measured: `ls $DATA_ROOT/torchcell-library/ | grep -i sevin` returns nothing). No
  duplication risk with our Fuhrer build, which stores only the deletion matrix.
- **M3D expression compendium.** Used as an external corroboration of the y-gene calls:
  "or M3D showing good expression level correlation with genes involved in carbohydrate
  metabolism such as fucR, fucI, fucK, xylAB, and rfaB (Fig EV8E)", and si1.md's EV8E
  caption "Correlation of expression levels over all 907 experiments in the M3D
  database." A separate E. coli expression compendium from the PRECISE releases this
  repo already loads (`ecoli/lamoureux2023.py`); a possible dataset, not a duplicate.
- **Earlier metabolome observations, qualitative only.** "consistent with earlier
  observations (Ishii et al, 2007; Fendt et al, 2010)" -- no numeric comparison and no
  shared table, so no duplication risk; both are candidate E. coli metabolome sources.
  Ishii 2007 is ALREADY in the literature mirror
  (`$DATA_ROOT/torchcell-library/ishiiMultipleHighThroughputAnalyses2007/`, measured by
  `ls | grep -i ishii`) with no loader in `torchcell/datasets/ecoli/`; Fendt 2010 is not
  mirrored.
- **Framing only, no numeric comparison.** "Large-scale phenotypic screening of single-
  and double-gene deletion mutants has proven powerful ... (Costanzo et al, 2010;
  Nichols et al, 2011). However, screening of large mutant libraries typically comes at
  the cost of measuring only one or few phenotypic traits (e.g., growth rate or
  viability)."
- **Duplication risk for the growth rate: LOW, and it is worth stating why.** The repo's
  `ecoli/tong2020.py:48` records that Tong's Table S1B compares its glucose values
  "with the Keio collection's own MOPS measurements (Baba 2006)". Fuhrer's EV1A rates are
  a different measurement: footnote c attributes them to "duplicate cultivations(each
  clone per knock-out strain from the strain collection)" in Fuhrer's own glucose-M9
  plus casein hydrolysate screening medium, and the sheet carries 4,320 rows (Fuhrer's
  screened set), not Baba's 3,985. So a Fuhrer growth-rate dataset would not re-store
  Baba's or Tong's numbers. If a Baba 2006 loader is ever added, re-check: Baba's
  growth tables are NOT in the mirror (only `si1.pdf` is there).
- **A publisher labeling inconsistency worth pinning.** `si5.xlsx`'s own legend sheet is
  titled `Legend Table EV5`, but the Methods cite that exact content as Table EV4: "we
  derived for each ion-annotation an estimate of the AUC index representing annotation
  confidence (Table EV4)". Measured over `paper.md`: `Table EV1` x2, `Table EV2` x1,
  `Table EV3` x13, `Table EV4` x1, `Table EV5` x0. So no SI file is missing; the file
  and the paper disagree on the number. The dendron note (line 28) follows the file's
  own label ("EV5 the annotation AUCs"), which is defensible, but the paper's citation
  is `Table EV4` and a reader chasing it will not find the string.

**Loadable-now estimate.** None. The single unstored phenotype is blocked, so the
estimate below is the size it WOULD have, for prioritizing the schema question.

- Fuhrer growth rate, if a home existed: **3,735 records** matching the metabolome build
  one-for-one, roughly 3,470 of them with an SD. Basis, measured: `sample_id_zscore.xls`
  has 3,807 columns, one of them `wt`; of the 3,806 strain names, 3,781 match exactly one
  Table EV1A row and all 3,781 carry a numeric growth-rate `Mean` (3,516 also carry a
  numeric `Standard Deviation`, 265 read `n.d.`); the loader then drops 46 of those 3,781
  for a JW id not on a BW25113 locus, leaving 3,735 (the note's kept count). The
  3,470 SD figure is an estimate, not a measurement: it scales 3,516/3,781 onto 3,735
  and I did not re-run the JW reconciliation to get the exact overlap.
- A SUPERSET build keyed on Table EV1A rather than on the deposit's columns would reach
  **3,869 records** (measured: 3,870 rows with `Keio Delivery Statusb` = "ready to
  distribute", of which 3,869 carry a numeric `Mean` and 3,590 a numeric `Standard
  Deviation`), i.e. about 88 strains that have a growth rate but no metabolome column,
  before any JW reconciliation drop. The ceiling over all delivery statuses is 4,319
  numeric means of 4,320 rows.

**Not checked.**

- `si1.pdf` and `si6.pdf` were read only through their MinerU OCR (`si1.md`, `si6.md`)
  and the OCR's figure jpgs were not opened. Numbers that live only inside a plotted
  figure (Fig 6B/6C, EV8B-D, EV10A/B growth rates) cannot be recovered from either, so
  "figure only" above rests on the captions plus the absence of any matching data file.
- `S-BSST5`'s `rawdata_neg_all.tsv` / `rawdata_pos_all.tsv` and
  `neg_kegg_all_3mD.xls` / `pos_kegg_all_3mD.xls` were not downloaded or opened: they
  are deliberately not mirrored (`fuhrer2017.py:294`), and their contents are described
  by `S-BSST5.json`, which I did read.
- The MassIVE deposit `MSV000078963` was not opened (no network retrieval was attempted
  and nothing from it is in either mirror).
- Whether Sevin and Sauer 2014 and Fendt 2010 are in the literature mirror was checked by
  directory name only (`ls $DATA_ROOT/torchcell-library/ | grep -i "sevin\|fendt"`), not
  by reading any manifest, so a mirror entry under a differently spelled citation key
  would have been missed.
- The exact number of the loader's 3,735 kept records that carry a numeric EV1A
  `Standard Deviation` was not measured: that needs the BW25113 JW reconciliation, which
  needs the genome, and running it is a build step this audit does not take.
- The unit and log base of the EV1A growth rate are not stated anywhere I read. Methods
  give only "the slopes of linear fits to log-transformed experimentally determined OD
  values". Hypothesis (untested): a specific growth rate in h^-1 on a natural-log scale,
  because the measured median of 0.81 corresponds to a 51-minute doubling time, which is
  the right order for glucose plus casein hydrolysate at 37 C. A loader must not assume
  this; it is exactly the kind of value the dataset-sourcing rule says to resolve from
  the SI or record as a gap.

### goodallEssentialGenomeEscherichia2018 -- `ecoli/goodall2018.py`

Goodall et al. 2018, mBio 9:e02096-17, doi 10.1128/mBio.02096-17, PMC5821084. A TraDIS
mini-Tn5 transposon screen of E. coli K-12 BW25113 in two conditions.

**SI inventory.** All under
`/scratch/projects/torchcell-scratch/torchcell-library/goodallEssentialGenomeEscherichia2018/si/`.
The paper's own listing (`paper.md` lines 145-151) is `TEXT S1, DOCX file, 0.1 MB. / FIG
S1, PDF file, 0.02 MB. / FIG S2, TIF file, 0.1 MB. / TABLE S1, XLSX file, 0.2 MB. / TABLE
S2, PDF file, 0.04 MB. / TABLE S3, PDF file, 0.03 MB. / TABLE S4, XLSX file, 0.3 MB.`, so
the `siN` numbering is by byte size of the publisher deposit, not by the listing order.
Each `siN.md` has `siN_middle.json`, `siN_content_list.json` and `siN_ocr_provenance.json`
sidecars; those are not listed below. Extracted images live under `si/images/si1/` (1 jpg)
and `si/images/si6/` (2 jpg).

| file | bytes | what it is |
|---|---|---|
| `si1.pdf` | 20,103 | **Fig. S1**: two R histograms of the insertion-index distribution (panel A = TL, panel B = LB) with the fitted exponential (red) and gamma (blue) modes. Axes `Insertion Index` x `Probability mass function` / `Frequency`. Read as the extracted image `si/images/si1/c7b93b39....jpg`. |
| `si1.md` | 84 | the OCR of `si1.pdf`. **It is 84 bytes and contains nothing but one image reference** (`![](images/si1/c7b93b39....jpg)`), because the page is a single bitmap with no text layer. No caption, no numbers. |
| `si2.xlsx` | 257,700 | **Table S1**, the TL (input transposon library) per-gene call table. One sheet. Consumed by the loader. |
| `si3.pdf` | 40,795 | **Table S2. Comparison of essential genes identified by Keio, PEC and TraDIS** -- a 7-column Venn-partition gene list over 6 pages. Not consumed. |
| `si3.md` | 1,860 | the OCR of `si3.pdf`. The 7-column layout collapsed: the header row and the column boundaries are lost, leaving a run-on gene list. I read the real table from `si3.pdf` directly (rendered pages + `pdftotext -layout`), not from this OCR. |
| `si4.docx` | 116,918 | **Text S1**, "Supplementary Methods / Statistical Analysis": the geometric-model and simulation argument for insertion-free-region (IFR) p-values. No OCR sibling in the mirror. Read by unzipping `word/document.xml` and stripping tags. Prose + 3 references, no data table. |
| `si5.tif` | 128,210 | **Fig. S2**. Despite the extension `file` reports `PNG image data, 800 x 600, 16-bit/color RGBA`. Four panels (A-D) of simulated IFR statistics (expected IFR count and P(>=1 IFR) vs length `l`, per genome and per gene) for Langridge, Barquist and "This study". Read as an image after copying to `.png`. |
| `si6.pdf` | 28,484 | **Table S3. Causes of discrepancies between datasets** -- 88 gene rows, 3 columns. Not consumed. |
| `si6.md` | 6,359 | the OCR of `si6.pdf`, as an HTML table. Mostly faithful; it mis-merges the Venn-group cell for rows whose gene carries a parenthetical synonym (`mazE (chpR) K` lands in the Gene cell with an empty group cell). I cross-read `si6.pdf` with `pdftotext -layout`. |
| `si7.xlsx` | 270,726 | **Table S4**, the LB-outgrowth per-gene call table. One sheet. Consumed by the loader. |

**Real sheet and column headers (measured).** `openpyxl`, read-only, over both workbooks:

- `si2.xlsx`, one sheet `Table_S1_Essential genes identi`, `max_col=6`, 4,315 rows.
  Row 0 is the title `Table S1. Essentiality classification for genes of the TL data`;
  row 1 is the header `('Gene', 'Insertion Index Score', 'Log Likelihood Ratio',
  'Essential', 'Non-essential', 'Unclear')`; **4,313 data rows, 4,269 distinct gene
  names, 0 rows with an empty `Gene`.** Calls: Essential 358, Non-essential 3,793,
  Unclear 162 (sum 4,313). `Insertion Index Score` range 0 to 0.701149;
  `Log Likelihood Ratio` range -172.162 to 53.4526.
- `si7.xlsx`, one sheet `Table_S4_Essential genes identi`, `max_col=6`, 4,315 rows.
  Title `Table S4. Essentiality classification for genes following outgrowth in LB`;
  the **same six headers**; 4,313 data rows, 4,269 distinct gene names. Calls:
  Essential 356, Non-essential 3,802, Unclear 155. `Insertion Index Score` range 0 to
  0.820896; `Log Likelihood Ratio` range -173.552 to 152.418.

There is no seventh column in either workbook: no read count, no per-replicate value, no
p-value, no q-value, no log2 fold change, no insertion position, no gene length, no
condition contrast. The TraDIS-screen quantities the brief asked me to look for that are
NOT in the release: per-gene **read counts**, **log2 fold change** (TL vs LB), and any
**p/q value**. The only per-gene statistic besides the index is the two-mode mixture log
likelihood ratio, which is the evidence for the call, not a test statistic against a null.

`si3.pdf` (Table S2) real header, read from the rendered page 1:
`TraDIS only | Keio only | PEC only (W3110) | TraDIS-Keio | TraDIS-PEC | Keio-PEC | All 3`,
with footnote a `BW25113 (CP009273.1) gene names have been used unless otherwise
specified. Alternative gene names used by Keio or PEC are shown in brackets` and footnote
b `The Keio naming convention is used for all genes in bold`.

`si6.pdf` (Table S3) real header, from `pdftotext -layout`: `Gene | Venn group | Cause of
discrepancy`. **88 gene rows** (measured by tallying the third field). Venn groups: K 25,
P 18, PT 18, KP 16, KT 11. Causes: `Errors in library construction` 34, `Genes containing
a transposon free region` 16, `RNA genes not considered in our analysis` 14, `Unclear` 9,
`Conditionally essential` 7, `Polar insertions` 5, `Anti-Toxin` 3.

Data-availability statement (`paper.md` line 139):
> Accession number(s). TraDIS sequencing data are available from the European Nucleotide Archive under accession no. PRJEB24436.

**What the loader stores.** `GeneEssentialityPhenotype.is_essential` on
`BacterialGeneEssentialityExperiment` / `...Reference`, from the `Essential` /
`Non-essential` one-hot columns of both workbooks, one record per (condition, locus).
**8,203 records** over two conditions (TL 4,098 = 358 essential + 3,740 non-essential;
LB 4,105 = 356 + 3,749), 4,172 distinct loci, 2 references
(`notes/torchcell.datasets.ecoli.goodall2018.md`, "### Records and drops" table and
"### Build and verification"). Genotype is one gene-level
`TransposonInsertionPerturbation` per record. Both released conditions are loaded; nothing
in the two consumed tables is left on the floor at the condition level.

**Already recorded as not stored.** Every unstored per-record quantity of the two
consumed workbooks is already a written decision. I am retiring these, not re-reporting
them:

- **`Insertion Index Score`** -- already recorded in
  `torchcell/datasets/ecoli/goodall2018.py:348`, `SOURCED_VALUES["insertion_index"]`
  note: `released in the 'Insertion Index Score' column; not stored on records (no
  field), kept in preprocess/calls.csv`; and in the note's `### Open, and what is a gap`
  bullet "Schema, the main gap".
- **`Log Likelihood Ratio`** -- same two places; note `### Phenotype, n and the
  uncertainty type`: "the released values per gene are the insertion index ..., the
  mixture log likelihood ratio (the evidence for the call, not a spread) and the call".
- **The `Unclear` call** -- already recorded: loader drop rule
  `unclear_call_not_representable` (`DROP_RULE_DESCRIPTIONS`, around line 802), and the
  note's `### Records and drops` table, 162 TL + 155 LB = 317 rows.
- **`n_samples` = 2 per condition and `sample_unit`** -- already recorded, note
  `### Phenotype, n and the uncertainty type` ("n_samples, sourced but not storable"),
  including the unresolved technical-vs-independent-culture conflict.
- **No per-gene uncertainty exists** -- already recorded in the same section, with the
  synonym comb listed ("Combed for `replicate`, `independent`, `triplicate` ...,
  `standard deviation`, `n =`, `bootstrap`, `error`, `confidence` over `paper.md` and
  every SI OCR file"). I re-ran that comb and agree: the only standard deviation in the
  paper is the beta-galactosidase assay's.
- **Tables S2 and S3, and the ENA reads** -- already recorded as deliberately not
  mirrored: note "Not mirrored, because nothing reads them: the TraDIS reads (ENA
  PRJEB24436) and Tables S2/S3", and the raw manifest's `si_expected` list.
- **Per-insertion positions and strands** -- already recorded, note `### Genotype` table
  and `### Open, and what is a gap` ("Per-insertion records (positions, strands) would
  need the ENA reads; not in scope").
- **The 106 multi-copy insertion-sequence rows** -- already recorded, drop rule
  `symbol_not_on_one_bw25113_locus` and the note's `### Strain and identifiers`.

One precision correction to the note, not a new finding. `### Records and drops` ends:
"The text's manual reassessments (for example `ftsK`, essential by its N-terminal
insertion-free region but non-essential by index) are not in the tables and are not
applied." They are not in the two CONSUMED tables, but they are partly in a released
table: `si6.pdf` (Table S3) carries `ftsK | KP | Genes containing a transposon free
region`, which is exactly that reassessment's reason, for 88 genes. The decision not to
consume Table S3 is unchanged and correct (see the classification below); only the phrase
"are not in the tables" is loose.

#### Released quantities

| released quantity | source file + column | stored | classification | evidence |
|---|---|---|---|---|
| essentiality call, TL | `si2.xlsx` sheet `Table_S1_Essential genes identi`, cols `Essential` / `Non-essential` | **yes** | stored as `GeneEssentialityPhenotype.is_essential` | 358 / 3,793 measured by openpyxl; paper: `sufficient insertions were found in 3,793 genes for them to be classed as nonessential, 162 genes were situated between the two modes and classed as unclear, and 358 genes in the mutant library were identified as essential (Table S1).` |
| essentiality call, LB | `si7.xlsx` sheet `Table_S4_Essential genes identi`, cols `Essential` / `Non-essential` | **yes** | stored, second condition | 356 / 3,802 measured; paper: `Insertion index scores were calculated as before (Table S4).` |
| `Unclear` call, TL + LB | `si2.xlsx` / `si7.xlsx` col `Unclear` | no | **blocked**: `GeneEssentialityPhenotype` has only `is_essential: bool` (`torchcell/datamodels/schema.py:3649-3656`), so a third state has nowhere to go. None of #749, #753, #756, #758, #760 names this field. **Already recorded** (loader drop rule `unclear_call_not_representable`). | 162 + 155 = 317 rows measured |
| `Insertion Index Score`, TL + LB | `si2.xlsx` / `si7.xlsx` col `Insertion Index Score` | no | **blocked**: no continuous field on `GeneEssentialityPhenotype`. Not loadable as `FitnessPhenotype.fitness`, whose contract is `ko_growth_rate/wt_growth_rate` (`schema.py:3552`); this is unique insertion points per base. Not in any of the five issues. **Already recorded.** | definition: `To normalize for gene length, the number of unique insertion points within the CDS was divided by the CDS length in bases. This value was termed the insertion index score`; ranges 0-0.701149 (TL), 0-0.820896 (LB) |
| `Log Likelihood Ratio`, TL + LB | `si2.xlsx` / `si7.xlsx` col `Log Likelihood Ratio` | no | **blocked**: same missing field. It is the two-mode mixture evidence, not a p-value and not an uncertainty, so `UncertaintyType` cannot type it. **Already recorded.** | `the ratio of these values was used to calculate a log likelihood score`; ranges -172.162 to 53.4526 (TL), -173.552 to 152.418 (LB) |
| Venn-group membership over TraDIS / Keio / PEC, 417 genes | `si3.pdf` (Table S2), the 7 column headings `TraDIS only`, `Keio only`, `PEC only (W3110)`, `TraDIS-Keio`, `TraDIS-PEC`, `Keio-PEC`, `All 3` | no | **not a phenotype** for Goodall: the TraDIS half is a re-encoding of the `Essential` column we already store, and the Keio / PEC halves are two OTHER datasets' binary calls reproduced here. They are also only the positive side: the essential members are listed, the non-essential genes of Keio and PEC are not, so the table is not a loadable essentiality table for either. See the comparison section. | caption `Table S2. Comparison of essential genes identified by Keio, PEC and TraDIS`; footnote a `BW25113 (CP009273.1) gene names have been used unless otherwise specified.` |
| `Cause of discrepancy`, 88 genes | `si6.pdf` (Table S3), col `Cause of discrepancy` | no | **not a phenotype**: a curation label (7 distinct strings) explaining why three datasets disagree, assigned by manual inspection. Turning `Genes containing a transposon free region` or `Anti-Toxin` into an essentiality call is our interpretation, not a released value. | caption `Table S3. Causes of discrepancies between datasets`; tally measured: 34 / 16 / 14 / 9 / 7 / 5 / 3 |
| `Venn group`, 88 genes | `si6.pdf` (Table S3), col `Venn group` | no | **not a phenotype**: the same Venn membership as Table S2, abbreviated (K / P / KP / KT / PT) | tally measured: K 25, P 18, PT 18, KP 16, KT 11 |
| reads with a matching inline barcode, per sample | `paper.md` TABLE 1, col `No. of sequence reads with matching inline barcode` | no | **not a phenotype**: a sequencing QC metric for 6 sample rows (TL1, TL2, TL combined, LB1, LB2, LB combined), not attached to a strain or gene | `TABLE 1 Parameters for TraDIS data set derived from the E. coli K-12 strain BW25113 transposon library`; TL1 4,818,864 ... LB combined 12,311,487 |
| mapped reads and % of raw, per sample | `paper.md` TABLE 1, col `No. of mapped sequence reads (% of the raw data)` | no | **not a phenotype**: same | TL1 `3,891,339 (80.75)`, LB2 `5,382,477 (84.06)` |
| genome-wide insertion sites, per sample | `paper.md` TABLE 1, col `No. of genome-wide insertion sites` | no | **not a phenotype**: a per-sample library-density metric | TL combined 901,383; LB combined 595,233 |
| gene function + `Comment`, 81 genes | `paper.md` TABLE 2 `Genes with low insertion index scores identified by TraDIS only` | no | **not a phenotype**: an annotation table; `Functions are taken from the Ecocyc website.` The `Comment` column is `Duplicated in Keio collection` provenance, citing Yamamoto et al. | footnote a `ACP, acyl carrier protein; PTS, phosphotransferase system; DMK, demethylmenaquinone. Functions are taken from the Ecocyc website.` |
| beta-galactosidase activity (Miller units), transposon::lacZ fusions | Fig. 4 only; **no released table or data file** | no | **not a phenotype we can load**: (a) no numbers are released, only bars in a bitmap; (b) the construct is a plasmid-borne reporter fusion in pRW224/pRW225, not a genome perturbation of a gene whose phenotype is being measured. It is the paper's control that the transposon permits read-through. | `$\beta$ -Galactosidase activity was measured in triplicate for three technical replicates. Values are mean values plus standard deviations between replicates (error bars).`; Methods: `$\beta$ -galactosidase activity was calculated in Miller units.` |
| insertion-index correlation between replicates | Fig. 1B and 1C only | no | **not a phenotype**: a per-condition summary of two sequenced samples (0.96 for TL, 0.97 for LB), figure-only, already sourced in the loader as `SOURCED_VALUES["replicate_kind"]` / `["lb_pooled"]` | `There was a high correlation coefficient of 0.96 between the samples (Fig. 1B).`; `there was a high correlation coefficient of 0.97 between the gene insertion index scores of each technical replicate (Fig. 1C)` |
| simulated IFR statistics (expected counts, P(>=1 IFR), corrected lengths) | `si4.docx` Text S1 + `si5.tif` Fig. S2 | no | **not a phenotype**: simulation output about the method's resolution, per study, not per record. `Our study (901,383 inserts, 4,631,469 base genome) gives a corrected p_genome = 0.05 of ~75 (and p_gene = 0.05 of ~36).` | `we computationally investigate these statistics, simulating many instances of N random non-coincident insertions in a genome of length G and reporting the statistics of resultant IFRs (C code available as SI)` |
| insertion-index distribution + fitted modes | `si1.pdf` Fig. S1 | no | **not a phenotype**: a histogram of the `Insertion Index Score` column we already have, with the exponential/gamma fits | axes `Insertion Index` / `Probability mass function` / `Frequency`; `an exponential distribution model was fitted to the mode that includes essential genes, and a gamma distribution model was fitted to the nonessential mode` |

**Totals.** 14 released per-record quantities enumerated. 2 stored (the essentiality call
in each of the two conditions). 0 loadable now. 3 blocked (the `Unclear` call, the
insertion index, the log likelihood ratio -- each released twice, once per condition, so
6 released columns). 9 not a phenotype.

#### Comparisons against other data

This paper is built as a benchmark against other datasets, so the comparison question has
real content here.

- **The whole paper benchmarks TraDIS against the Keio collection and the PEC database.**
  > The 358 putative essential genes identified in the TL data were compared to the essential genes as defined by the Keio collection and the Profiling of the E. coli Chromosome (PEC) database (1, 2). This comparison revealed 248 genes $( 5 9 . 5 \% )$ that were common to all three data sets (Fig. 2A and Table S2). ... These genes comprise 16 genes in the Keio and PEC lists that were not identified by our analysis, 25 exclusive to Keio, 18 exclusive to PEC, and 11 and 18 that overlapped between our method and Keio or PEC, respectively (Fig. 2A). However, the largest subcategory of 81 genes is unique to our data set.

  Those seven numbers sum to 417 and are the seven columns of Table S2 (`si3.pdf`). They
  imply a Keio essential set of 25 + 11 + 16 + 248 = 300 and a PEC essential set of
  18 + 18 + 16 + 248 = 300. My own column tally off `pdftotext -bbox` reproduced the two
  largest columns exactly (`All 3` = 248, `Keio only` = 25) but over-counted the others
  by 1 to 3 because the column x-positions shift between pages; the paper's stated counts
  are the ones I rely on, and my tally is an unreliable cross-check, not a measurement.

  **This does not create a duplication risk, and it does not create a loadable dataset
  either.** Table S2 lists only the genes each source calls ESSENTIAL; it never lists the
  non-essential side, so it cannot be ingested as a Keio or PEC essentiality table. It is
  also a third party reproducing two other releases, which is precisely the weaker
  provenance the project's rules reject.

- **It does point at a dataset we do not have: Keio (Baba 2006).** Baba 2006 is already in
  the literature mirror as `babaConstructionEscherichiaColi2006` (paper.md, paper.pdf,
  one `si/si1.pdf`), and three loaders already cite it for strain provenance
  (`torchcell/datasets/ecoli/fuhrer2017.py:155`, `wang2015.py:197`, `tong2020.py:48`),
  but there is **no `torchcell-raw/baba*` mirror and no Baba 2006 loader**, and it is not
  one of the twenty ranked rows in `notes/plan.bacteria-ontology-genome.md`. Its own
  release, not Goodall's reproduction of it, is where the Keio essential-gene list and
  the Keio growth measurements would come from. Flagging it for the Baba audit; I did not
  open Baba's SI, so I make no claim about what it contains.

- **PEC is a database, not a mirrored paper.** `grep -rn -i 'PEC database'` over
  `torchcell/` and `notes/` returns nothing outside the Goodall files. Fetching it would
  mean ingesting a live web resource, which the provenance rules do not allow without a
  pinned artifact. Not a candidate.

- **Goodall is the only essentiality loader, so nothing is stored twice.**
  `grep -l GeneEssentialityPhenotype torchcell/datasets/ecoli/*.py` returns
  `goodall2018.py` alone, and `notes/plan.bacteria-ontology-genome.md:512` reads
  `| GeneEssentialityPhenotype | reuse | Goodall 2018 TraDIS essentiality calls |`.
  The already-recorded deduplication finding (note `### Deduplication (checklist item 7)`)
  stands: "No superset in the fifty covers this TraDIS library. Keio-derived and RB-TnSeq
  BW25113 rows are different libraries and assays."

- **The paper calls its own LB table a fitness readout, but releases no fitness column.**
  > Scrutiny of our data revealed substantially fewer degS and folK mutants after growth in LB, supporting our hypothesis that they are conditionally essential (Fig. 3G and H). Other genes showing similar fitness costs can be identified in the LB outgrowth data set (Table S4).

  and
  > The fitness effect was confirmed by growing the library in LB, as such mutants are outcompeted (Fig. 5F)

  A TL-to-LB depletion would be a competitive-fitness measurement, but **no log fold
  change, ratio or p-value column is released** -- only the two independent index columns.
  Computing it would be our derivation off two values we cannot currently store, so it is
  downstream of the same schema gap, not a separate loadable item.

- **A warning against loading the insertion index as fitness.** The paper itself lists the
  non-fitness causes of a low index:
  > A low number of transposon insertion events within a gene, which fall below the statistical cutoff threshold, can be due to inaccessibility of the gene to transposition because of extreme DNA structure, exclusion by DNA-binding proteins, polarity effects due to insertion in a gene upstream of a cotranscribed essential gene, and location of the gene close to the replication terminus (17).

  So the index is not a growth-rate ratio even in spirit. If a continuous field is added,
  it should be a named insertion-index field, not `FitnessPhenotype.fitness`.

#### Loadable-now estimate

**Zero loadable-now items.** Everything released and unstored needs a schema field that
does not exist. For sizing the blocked work, measured from the workbooks and from the
note's drop table:

- `Unclear` call: **317 new records** (162 TL + 155 LB), on top of the 8,203 already
  stored. The drop rules apply `symbol_not_on_one_bw25113_locus` first, so these 317 are
  disjoint from the 106 dropped insertion-sequence rows. Basis: openpyxl count of the
  `Unclear` column in each workbook, which equals the note's `### Records and drops` row.
- `Insertion Index Score` + `Log Likelihood Ratio`: **0 new records, 2 values on each of
  8,203 existing records**, plus 2 values on each of the 317 records the tri-state call
  would add. Basis: 4,313 rows x 2 columns x 2 workbooks = 17,252 released values, of
  which 17,040 sit on rows that survive or would survive the drop rules
  (8,203 + 317 = 8,520 records x 2).

#### Not checked

- **The ENA reads, PRJEB24436.** Not in either mirror and not downloaded. Per-insertion
  positions, orientations and read counts would come only from there; already recorded as
  out of scope.
- **The C code Text S1 says is deposited** (`C code available as SI`). No `.c` file is in
  the mirror and no separate code item appears in the paper's supplemental listing. Not
  retrieved.
- **Exact per-column gene counts of Table S2.** My `pdftotext -bbox` tally is unreliable
  because the column x-positions shift between pages (it reproduced `All 3` = 248 and
  `Keio only` = 25 exactly, but over-counted the five narrow columns). I used the paper's
  stated counts instead. The 88 rows of Table S3 ARE a measurement: tallied from
  `pdftotext -layout`, and they equal 169 discordant genes minus the 81 TraDIS-only ones.
- **Baba 2006's own SI** (`si/si1.pdf` in `babaConstructionEscherichiaColi2006`). Out of
  this paper's scope; flagged above for whoever audits Baba.
- **Fig. 3 and Fig. 5 insertion profiles.** They are per-gene browser tracks in bitmaps
  with no released underlying values, so there is nothing per-record to enumerate from
  them beyond what the tables already carry.

### lamoureuxMultiscaleExpressionRegulation2023 -- `ecoli/lamoureux2023.py`

PRECISE-1K. The SI inventory is genuinely short, and I say so plainly: there is exactly
ONE supplementary file and it contains ZERO supplementary tables. The audit therefore
turns on the raw-data mirror and on the release archive the paper's Data Availability
points at.

#### SI inventory

| file | format | what it is | bytes |
|---|---|---|---|
| `si/si1.pdf` | PDF | the journal's single supplemental file (`gkad750_supplemental_file.pdf`, retrieved from the PMC OA bucket per `manifest.json` `si_data_sources`) | 2,819,174 |
| `si/si1.md` | markdown | MinerU OCR of si1.pdf | 8,307 |

Sidecars exist and are skipped here for the reader: `si1_middle.json`,
`si1_content_list.json`, `si1_ocr_provenance.json`, plus `si/images/si1/*.jpg` (14 figure
images).

**si1.pdf holds 14 Supplemental Figures and no tables.** Measured: the OCR layout
`si1_content_list.json` is `Counter({'image': 14, 'text': 2})`, zero blocks of type
`table`; and `grep -i -E "Supplemental Table|Supplementary Table|Table S" paper.md`
returns nothing. So the paper releases no tabular SI at all. Every per-record quantity
beyond the expression matrix lives in the Zenodo/GitHub release, not in the SI PDF.

#### Raw-data mirror: every file and its real column headers

`$DATA_ROOT/torchcell-raw/lamoureuxMultiscaleExpressionRegulation2023/` holds `manifest.json`
plus four files, exactly the four members the loader consumes.

| file | shape (measured) | what it is |
|---|---|---|
| `data/precise1k/log_tpm_qc.csv` | 4,257 rows x 1,035 sample columns (`wc -l` 4,258; header `,p1k_00001,...,p1k_01055`) | PRECISE-1K itself, log2(TPM+1), row key = b-number |
| `data/precise1k/counts.csv` | 4,355 rows x 1,055 sample columns (`wc -l` 4,356; header `Geneid,p1k_00860,...`) | raw featureCounts counts, PRE-QC sample set |
| `data/precise1k/log_tpm_qc_w_short_low_fpkm.csv` | 4,355 rows x 1,035 sample columns | the same log2(TPM+1) values before the short / low-FPKM genes were removed |
| `data/precise1k/metadata_qc.csv` | 1,035 rows x 43 columns | post-QC per-sample metadata |

`metadata_qc.csv` column headers, verbatim and in file order, with the non-blank count I
measured (`pandas`, `dtype=str`, `keep_default_na=False`, 1,035 rows):

`Unnamed: 0` (1035, the `p1k_#####` id), `sample_id` (1035), `study` (1035, 45 unique),
`project` (1035, 45 unique), `condition` (1035, 518 unique), `rep_id` (1035),
`Strain Description` (1035, 278 unique), `Strain` (1035, 5 unique), `Culture Type` (1035),
`Evolved Sample` (1035), `Base Media` (1035, 9 unique), `Temperature (C)` (1035, 4 unique),
`pH` (1035, 5 unique), `Carbon Source (g/L)` (795), `Nitrogen Source (g/L)` (795),
`Electron Acceptor` (857), `Trace Element Mixture` (469), `Supplement` (319),
`Antibiotic for selection` (188), **`Growth Rate (1/hr)` (354)**, `Isolate Type` (449),
`Additional Details` (145), `project_reference` (1035), `Sequencing Machine` (643),
`LibraryLayout` (1035), `Platform` (1035), `Biological Replicates` (859), `DOI` (565),
`GEO` (591), `SRX` (396), `Run` (396), `R1` (1035), `R2` (964), `contact` (1035),
`creator` (1035), `passed_fastqc` (1035), `passed_pct_reads_mapped` (1035),
`passed_reads_mapped_to_CDS` (1035), `passed_global_correlation` (1035), `full_name`
(1035), `passed_similar_replicates` (1035), `passed_number_replicates` (1035), `run_date`
(1035).

**There is a per-sample growth rate and the loader does not read it.** 354 of 1,035 cells
are non-blank, all numeric, 338 strictly positive (min 0.07, median 0.62, max 1.42 h^-1)
and 16 exactly 0, spread over 17 of the 45 projects. The loader's `COL_*` constants
(`lamoureux2023.py:435-455`) do not include it, and `grep "Growth Rate"` over the loader,
its dendron note and its test file returns nothing.

**There is NO per-sample OD, strain-level quality metric, or iModulon activity in the
mirror.** The six `passed_*` booleans are the only per-sample quality cells, and they are
near-constant by construction (this is the QC-passing subset): `passed_fastqc`,
`passed_reads_mapped_to_CDS`, `passed_global_correlation` and `passed_similar_replicates`
are TRUE for all 1,035; `passed_pct_reads_mapped` is FALSE for 5; `passed_number_replicates`
is FALSE for 55. The only OD and dilution-rate information in the mirror is free text in
`Additional Details`: `"Cells grown till OD 0.3, rhamnose added to 0.1g/L, and harvested
after 2 hours"` (94 rows, the `pcoli` project), `"chemostat w/ dilution rate 0.31 h^-1"`
(3 rows) and `"chemostat w/ dilution rate 0.44 h^-1"` (2 rows). iModulon activities are
NOT mirrored (next section).

#### The release archive, and what is in it that is not mirrored

The mirror's `manifest.json` already says so: `si_expected` = `"the four consumed members
of data/precise1k/ are mirrored; the rest of the archive (iModulon matrices, Public K-12
data, notebooks) is not consumed and not mirrored"`. I enumerated the rest from the GitHub
API tree of `SBRG/precise1k` at tag `v1.0`, which resolves to commit
`71e1157f83d8362bac9039113cccfcbdb8873119`, i.e. the same commit the archive root
`SBRG-precise1k-71e1157/` names, so the tree IS the archive's membership (364 blobs). I
then fetched the real header rows from `raw.githubusercontent.com` at that commit (small
range requests; files written to an isolated scratch directory, never executed). The
archive itself is not on disk, so its member sha256s are not pinned by me.

| release file (not mirrored) | bytes | measured shape | real column headers |
|---|---|---|---|
| `data/precise1k/A.csv` | 3,971,613 | 201 iModulons x 1,035 samples | row key = iModulon name (`Sugar Diacid`, `Translation`, `OxyR`, ...), columns = `p1k_00001 ... p1k_01055`; 1,035 of 1,035 columns are `p1k_` |
| `data/precise1k/M.csv` | 16,673,231 | 4,257 genes x 201 iModulons | row key = b-number, columns = iModulon names |
| `data/precise1k/imodulon_table.csv` | 58,241 | 201 rows x 31 cols | `exp_var, imodulon_size, enrichment_category, system_category, functional_category, function, regulator, n_regs, pvalue, qvalue, f1score, precision, recall, TP, regulon_size, confidence, note, trn_enrich_max_regs, trn_enrich_evidence, trn_enrich_method, compute_regulon_evidence, single_gene_dominant_technical, tcs, regulon_discovery, ko, PRECISE 2.0, PRECISE 2.0_pearson, PRECISE 2.0_spearman, PRECISE, PRECISE_pearson, PRECISE_spearman` |
| `data/precise1k/component_stats.csv` | 13,362 | 201 rows x 4 cols | `S_mean_std, A_mean_std, count, L2_norm` |
| `data/precise1k/deg_dima_result.csv` | 513,577 | 6,103 condition PAIRS x 6 cols | `cond1, cond2, n_dima, n_deg, exp_var_dimas, exp_var_all` |
| `data/precise1k/multiqc_stats.tsv` | 447,779 | 1,055 rows x 56 cols | `Sample, Total, Assigned, Unassigned_rRNA, ..., percent_assigned, reads_processed, reads_aligned, reads_aligned_percentage, not_aligned, ..., Total Sequences, Sequence length, %GC, total_deduplicated_percentage, avg_sequence_length, basic_statistics, per_base_sequence_quality, ... adapter_content, cutadapt_version, r_processed, r_with_adapters, ... pe_sense, pe_antisense, failed, se_sense, se_antisense` |
| `data/precise1k/metadata.csv` | 567,820 | 1,055 rows x 35 cols | the PRE-QC metadata: same columns as `metadata_qc.csv` minus the six `passed_*` flags, `full_name` and `run_date`, plus a leading `Experiment`; it carries the SAME `Growth Rate (1/hr)` column |
| `data/precise1k/log_tpm.csv` | 63,419,050 | pre-QC log2[TPM] (1,055 samples) | not opened |
| `data/precise1k/log_tpm_norm_qc.csv` | 83,938,684 | the control-centered matrix | not opened |
| `data/precise1k/crp_binding.csv` | 7,380 | 188 rows, HEADERLESS, two float columns | no column names in the file |
| `data/precise1k/multiqc_data/*` (21 files), `subsamples/*` (20 files), `precise1k.json.gz` | -- | -- | not opened |
| `data/k12_modulome/metadata_qc.csv` | 1,188,124 | **2,710 rows x 55 cols** | the 43 PRECISE-1K columns PLUS `BioProject, BioSample, ScientificName, GEO Sample, PMID, aerobicity, approx_OD, dilution_rate, growth_phase, reference_condition, time, passed_replicate_correlations` |
| `data/k12_modulome/counts.csv` | 46,664,485 | 3,125 sample columns, ALL SRA accessions (`Geneid, DRX085453, DRX021107, SRX10647059, ...`), **zero `p1k_` columns** | header read |
| `data/k12_modulome/A.csv` | 9,987,692 | 194 iModulons x **2,710 samples (1,035 `p1k_` + 1,675 SRX/DRX)** | header read |
| `data/k12_modulome/M.csv`, `imodulon_table.csv` (194 rows), `component_stats.csv`, `multiqc_stats.tsv`, `k12_modulome.json.gz`, `k12_only_p1k_ctrl.json.gz`, `k12_only_proj_ref.json.gz`, `bioproject_list{,_curated}.csv`, `metadata_qc_part*`, `metadata_qc_with_p1k.csv`, `K12_metadata.tsv`, `Escherichia_coli_20220127.tsv` | -- | -- | `k12_modulome/imodulon_table.csv` header read (194 rows; cross-dataset columns `PRECISE 2.0`, `PRECISE-1K`, `PRECISE` with `_pearson`/`_spearman`); the rest not opened |
| `data/precise/log_tpm.csv` (14,417,636), `log_tpm_norm.csv`, `A.csv`, `M.csv`, `M_thresholds.csv`, `iM_table.csv`, `imodulon_table.csv`, `metadata_qc.csv`, `sample_table.csv`, `precise.json.gz` | -- | the ORIGINAL 278-sample PRECISE | `precise/sample_table.csv` header read: `project, study, condition, rep_id, Strain Description, Strain, Base Media, Carbon Source (g/L), Nitrogen Source (g/L), Electron Acceptor, Trace Element Mixture, Supplement, Temperature (C), pH, Antibiotic, Culture Type, Growth Rate (1/hr), Evolved Sample, Isolate Type, Sequencing Machine, Additional Details, doi, GEO, n_replicates, run_date` |
| `data/annotation/TRN.csv`, `gene_info.csv` | 358,044 / 1,063,339 | RegulonDB TRN + gene annotation | not opened |

**The Public K-12 extra columns carry nothing for a PRECISE-1K sample.** This is the one
hypothesis I had that the data refuted, so it is stated as a measurement:
`k12_modulome/metadata_qc.csv` is a union of two annotation schemas, and every one of the
12 extra columns is non-blank ONLY on the 1,675 public rows: `aerobicity` 1,425 (p1k 0),
`approx_OD` 788 (p1k 0), `growth_phase` 732 (p1k 0), `time` 391 (p1k 0), `dilution_rate`
92 (p1k 0), `BioProject`/`BioSample`/`ScientificName`/`reference_condition`/
`passed_replicate_correlations` 1,675 (p1k 0), `GEO Sample` 758 (p1k 0), `PMID` 783
(p1k 0). Conversely `Growth Rate (1/hr)` is non-blank on 354 rows, all of them `p1k_`,
never on a public row. So the Public K-12 table does NOT resolve the PRECISE-1K loader's
`duration_hours` gap, its `oxygen_regime_not_stated` drop or its `culture_not_batch` drop.
It would supply those fields for the public arm only.

#### What the loader stores

`RnaseqLamoureux2023Dataset`, root `data/torchcell/rnaseq_lamoureux2023`.
`BacterialRNASeqExpressionExperiment` + `RNASeqExpressionPhenotype`,
`measurement_type="rnaseq_tpm"`, one record per RNA-seq library.

- Value: `expression_tpm = 2**x - 1` over `log_tpm_qc.csv`, with the pseudocount 1
  back-solved (`lamoureux2023.py:28-37`); `expression_count` from `counts.csv`.
- **241 records** of 1,035 (794 dropped), 1 reference, gene set 42, verified at L0-L4
  (`notes/torchcell.datasets.ecoli.lamoureux2023.md`, "Build and verification (2026.10.07)":
  "**241 records** (of 1,035; 794 dropped by the rules above), 1 reference, gene set 42").
- Per-record value count measured by the verifier: 1,025,937 TPM and 1,025,937 count
  values (same note, L2 row).

#### Already recorded as not stored

Retire these; the project has already decided them.

- **`n_mapped_reads`**, the per-library read depth. `lamoureux2023.py:408-425`
  (`MAPPED_READS_GAP`), a typed `deferred_pending_source_review` gap whose `resolve_with`
  names `data/precise1k/multiqc_stats.tsv` verbatim: "the per-sample figure is in the
  release's MultiQC table, not in a consumed file". Note: "Phenotype and unit" section.
- **Replicate count / `n_samples`.** `lamoureux2023.py:40-45`: "The phenotype class has no
  `n_samples` slot, so no record claims more than its own library". Note: "Replicate
  structure", which also measured the disagreements between the `Biological Replicates`
  cell and the kept group size.
- **Chemostat dilution rate.** Note, "Environment settling and encoding", rule
  `culture_not_batch` (3 records): "chemostat (dilution rates 0.31, 0.44); `Environment`
  has no culture-type slot". Issue **#753** point 3 already names "`Environment` cannot
  express a chemostat dilution rate".
- **Harvest time / `duration_hours`.** `lamoureux2023.py:394-402` (`DURATION_GAP`): "cells
  were harvested at an optical density, not after a stated time".
- **The 98 short / low-FPKM genes.** Note, "Phenotype and unit": "The 98 genes the paper
  removed ... keep their TPM share, so a stored sample sums to less than 1e6"; the loader
  reads `log_tpm_qc_w_short_low_fpkm.csv` "only to back-solve the pseudocount"
  (`lamoureux2023.py:26-27`).
- **PRECISE (278 samples) is subsumed.** Note, "Superset and deduplication (checklist item
  7)": 266 of PRECISE's 278 sample ids are PRECISE-1K ids, so "PRECISE-1K is therefore the
  loaded superset and PRECISE gets no loader of its own".
- **The centered matrix is derived.** Note, "Reference": `log_tpm_qc - log_tpm_norm_qc`
  equals the mean of the two `control:wt_glc` samples for all 1,035 samples, max absolute
  difference 3.6e-15.

#### Enumeration

| released quantity | source file + column | stored | classification | evidence |
|---|---|---|---|---|
| log2(TPM+1), 4,257 genes x 1,035 samples | mirror `log_tpm_qc.csv`, row = b-number, col = `p1k_#####` | yes | stored: `RNASeqExpressionPhenotype.expression_tpm` | paper: "the final expression dataset was reported in units of $\log _ { 2 }$ -transformed Transcripts Per Million" |
| featureCounts read count, 4,355 x 1,055 | mirror `counts.csv`, col `Geneid` + sample cols | yes, for stored cells | stored: `RNASeqExpressionPhenotype.expression_count` | measured `wc -l` 4,356, header 1,056 fields |
| log2(TPM+1) for the 98 removed genes, 98 x 1,035 | mirror `log_tpm_qc_w_short_low_fpkm.csv` | no | loadable now: `RNASeqExpressionPhenotype.expression_tpm` -- ALREADY RECORDED, and the source's own QC removed them | paper: "extremely low-expression transcripts $\mathrm { ( F P K M } < 1 0 )$ were also removed to reduce noise." |
| pre-QC log2[TPM] + metadata for the 20 libraries QC removed | release `precise1k/log_tpm.csv` + `metadata.csv` (1,055 rows) | no | loadable now in principle, but the source's own QC excluded them | measured: `counts.csv` and `metadata.csv` carry 1,055 samples vs 1,035 in `log_tpm_qc.csv` / `metadata_qc.csv` |
| **`Growth Rate (1/hr)` per sample, 354 non-blank values** | mirror `metadata_qc.csv` column 20 `Growth Rate (1/hr)` | **no** | **loadable now**: `EnvironmentResponsePhenotype.environment_response` with `MeasurementType.growth_rate` + `AssayType.liquid_od_growth`, wrapped in `BacterialEnvironmentResponseExperiment`, which exists (`schema.py:5503`). Verifier blocker named below. | measured: 354 non-blank, all numeric, 338 > 0 (min 0.07, median 0.62, max 1.42), 16 exactly 0; 17 of 45 projects. Not read by the loader (`COL_*` block `lamoureux2023.py:435-455`), not in the note. |
| `Biological Replicates` per sample | mirror `metadata_qc.csv` column 27 | no | not a phenotype (replicate count) -- ALREADY RECORDED | `lamoureux2023.py:43`, note "Replicate structure" |
| six `passed_*` QC booleans per sample | mirror `metadata_qc.csv` columns 36-39, 41-42 | no | not a phenotype (QC filter indicator, not a measurement of the strain) | measured: 4 constant TRUE over 1,035; `passed_pct_reads_mapped` 5 FALSE; `passed_number_replicates` 55 FALSE |
| chemostat dilution rate (2 distinct rates, 5 samples) | mirror `metadata_qc.csv` `Additional Details` free text | no | blocked: **#753** point 3 -- ALREADY RECORDED as the `culture_not_batch` drop | verbatim cells `"chemostat w/ dilution rate 0.31 h^-1"` (3 rows), `"chemostat w/ dilution rate 0.44 h^-1"` (2 rows) |
| harvest OD + induction time for 94 samples | mirror `metadata_qc.csv` `Additional Details` free text | no | blocked: NEW gap, no optical-density slot anywhere (`CultureFormat` has `vessel/working_volume_ul/shaking_rpm/inoculum_cells/endpoint`; `EndpointRule` is only `fixed_duration \| fixed_generations \| until_control_saturation`) | verbatim cell `"Cells grown till OD 0.3, rhamnose added to 0.1g/L, and harvested after 2 hours"`, 94 rows; all 94 are the `pcoli` project the loader drops as `heterologous_expression_construct` |
| per-sample MultiQC metrics, 1,055 x 56 | release `precise1k/multiqc_stats.tsv` | no | `reads_aligned` / `Assigned` ALREADY RECORDED as `MAPPED_READS_GAP`; the other 54 are not a phenotype (sequencing QC) | header read in full, listed above; measured 1,055 data rows |
| iModulon activities, 201 x 1,035 = 207,435 values | release `precise1k/A.csv` | no | not a phenotype: a DERIVED linear projection of the matrix we already store. The paper defines it as the solution of `X = MA`, so A is recoverable from the stored expression plus the released `M.csv`; storing it would be storing a transform of our own values. | paper: "FastICA numerically solves the matrix decomposition equation $\mathbf { X } = \mathbf { M } \mathbf { A }$ ; X is the input matrix; M is the 'iModulon' matrix, and A is the 'Activity' matrix" and "the final M matrix has dimensions of 4257 genes by 201 independent components, and the final activity matrix A has dimensions of 201 iModulons by 1035 samples" |
| iModulon gene weights, 4,257 x 201 | release `precise1k/M.csv` | no | not a phenotype (model parameter / gene-level loading) | paper: "the M matrix contains weightings that specify how much each gene (row) belongs to each independent component" |
| per-iModulon enrichment statistics, 201 x 31 | release `precise1k/imodulon_table.csv` cols `pvalue, qvalue, f1score, precision, recall, TP, regulon_size, exp_var, ...` | no | not a phenotype (annotation of a derived module, not of a strain) | header read; 201 data rows; paper: "All parameters and statistics related to calculation of TRN enrichments for regulatory iModulons are recorded in the iModulon metadata table, available in the GitHub repository." |
| per-iModulon component statistics, 201 x 4 | release `precise1k/component_stats.csv` | no | not a phenotype | header `S_mean_std, A_mean_std, count, L2_norm`; 201 data rows |
| per-condition-pair DEG and DiMA counts, 6,103 x 4 | release `precise1k/deg_dima_result.csv` cols `n_deg, n_dima, exp_var_dimas, exp_var_all` | no | not a phenotype (derived COUNTS; the per-gene DESeq2 log2 fold changes and FDRs behind them are NOT released) | measured 6,103 data rows, 549 distinct conditions, median `n_deg` 471, 4,483 rows with `n_dima > 0`, median `n_dima` over those rows 27. These reproduce the paper exactly: "A median of 471 differentially expressed genes (DEGs) were found across all pairwise within-project comparisons", "a median of just 28 iModulons", SI Fig 11 "$\scriptstyle ( \mathsf { n } = 4 4 8 3 )$", and "a total of 6104 pairwise computations" |
| control-centered log2[TPM], 4,257 x 1,035 | release `precise1k/log_tpm_norm_qc.csv` | no | not a phenotype (derived) -- ALREADY RECORDED, identity measured to 3.6e-15 | note, "Reference" |
| Public K-12 expression + counts for 1,675 public samples | release `k12_modulome/counts.csv` (3,125 SRA columns) and `k12_modulome.json.gz` | no | loadable now as a SEPARATE dataset: `RNASeqExpressionPhenotype` again; duplication risk, see below | paper: "The $\log _ { 2 } [ \mathrm { T P M } ]$ , raw read count, QC data files and sample metadata for the high-quality public samples may be found in the data directory of this project's GitHub repository." |
| Public K-12 per-sample `aerobicity` (1,425) and `time` (391) | release `k12_modulome/metadata_qc.csv` | no | loadable now: `Environment.aerobicity` and `Environment.duration_hours` (`schema.py:3349-3354`) -- public arm only, 0 fill on every `p1k_` row | measured fill counts above; `time` values are bare numbers with no unit in the header, so the unit would itself be a gap |
| Public K-12 per-sample `approx_OD` (788) and `growth_phase` (732) | release `k12_modulome/metadata_qc.csv` | no | blocked: NEW gap, same as the OD gap above. No field holds a harvest optical density or a growth phase; `EndpointRule` has no at-target-OD member. Not named by #749, #753, #756, #758 or #760. | measured fill counts above; example `approx_OD` values `0.5`, example `growth_phase` values `exponential` |
| Public K-12 `dilution_rate` (92 samples) | release `k12_modulome/metadata_qc.csv` | no | blocked: **#753** point 3 | measured 92 non-blank, example value `0.2/h` |
| Public K-12 iModulon decomposition, 194 iModulons x 2,710 samples | release `k12_modulome/A.csv`, `M.csv`, `imodulon_table.csv` | no | not a phenotype (derived) | measured: `A.csv` header has 2,710 sample fields, 1,035 of them `p1k_`; `imodulon_table.csv` has 194 data rows; paper: "ICA decomposition of the K-12 Dataset yields 194 iModulons." |
| PRECISE (278-sample) expression, activities and sample table | release `data/precise/*` | no | not a phenotype here -- ALREADY RECORDED as subsumed | note, "Superset and deduplication" |
| RegulonDB TRN + gene annotation | release `data/annotation/TRN.csv`, `gene_info.csv` | no | not a phenotype (annotation) | paper: "the gold-standard TRN reference annotation downloaded from RegulonDB v10.5" |
| `crp_binding.csv`, 188 rows, 2 unlabeled float columns | release `precise1k/crp_binding.csv` | no | not checked (headerless, no caption in paper or SI naming it) | measured: 189 lines, first line is already data (`0.9912374732449338,0.999646525493167`) |

Totals: **24 released quantities enumerated; 2 stored; 5 loadable now; 4 blocked; 12 not a
phenotype; 1 not checked.** Seven of the 24 are already recorded as decisions and are
retired above.

#### Comparisons against other data

PRECISE-1K is explicitly a compendium that subsumes earlier releases, and the paper says
so four times.

1. **Its own earlier release, PRECISE (Sastry 2019, ref 1).** Paper, Results: "PRECISE-1K
   constitutes a nearly 4-fold increase in size from the original 278-sample PRECISE(1)
   (Figure 1B)." **Already recorded and already measured**: 266 of PRECISE's 278 sample ids
   are PRECISE-1K sample ids, so PRECISE gets no loader and the 12 non-overlapping libraries
   are the only ones left out (note, "Superset and deduplication"). No action.

2. **Public SRA data, folded in as the 'Public K-12' / 'K-12 Dataset'.** Methods: "Data was
   compiled from NCBI SRA as described previously (22). ... After initial curation, 3125
   samples remained. ... Finally, the 0.95 minimum replicate correlation threshold was
   applied, yielding the final set of 1675 high-quality publicly-available samples $54 \%$
   of the original set). Next, these 1675 samples were combined with the 1035 samples of
   PRECISE-1K to yield the 'Public K-12' dataset, comprising 2710 curated, high quality
   expression profiles". Results: "We combined these samples with PRECISE-1K to yield the
   'K-12 Dataset', a high-quality transcriptomics dataset consisting of 2710 expression
   profiles (Figure 5A)."

   **Duplication risk, measured and bounded.**
   - `k12_modulome/counts.csv` has 3,125 sample columns and **zero** `p1k_` columns (all are
     SRX/DRX accessions), so the public COUNTS file does not duplicate the PRECISE-1K counts
     we store. 3,125 is exactly the Methods' post-curation number. (Results line 213 says
     "From 3230 K-12 samples", which disagrees with Methods' 3125; the released file agrees
     with Methods.)
   - `k12_modulome/metadata_qc.csv` (2,710 rows) and `k12_modulome/A.csv` (2,710 sample
     columns) DO contain all 1,035 `p1k_` samples. The metadata overlap is annotation, and
     the A overlap is a derived activity, so neither is a second copy of a stored
     measurement. But a future loader built naively off `k12_modulome/*` would re-ingest the
     1,035 PRECISE-1K samples.
   - The real risk is forward-looking: those 1,675 public samples are OTHER labs' RNA-seq,
     re-processed by this lab. The table carries 1,675 `BioProject`, 1,675 `BioSample`, 758
     `GEO Sample` and 783 `PMID` values, so any future E. coli K-12 RNA-seq loader built from
     GEO/SRA directly will overlap it. **This is the #760 pattern** (Rousset's growth screen
     IS Cui's `fit75` screen): if Public K-12 is ever loaded, it must be loaded with the
     accession as the dedup key, and whichever release lands first becomes primary.
   - Our existing E. coli RNA-seq loader is NOT at risk. `RnaseqCaglar2017Dataset` is REL606,
     an E. coli **B** strain, and the public curation discarded non-K-12 data: "RNA-seq
     samples were discarded if the strain was not from a K-12 strain".
   - The paper itself flags one public project as outside PRECISE-1K: "the example new data
     comes from the public K-12 Dataset AAT (67) (anaerobic-aerobic transition) (not in
     PRECISE-1K, but in public K-12 metadata)", ref 67 = Bui and Selvarajoo 2020.

3. **An unnamed intermediate, 'PRECISE 2.0'.** `precise1k/imodulon_table.csv` carries
   `PRECISE 2.0`, `PRECISE 2.0_pearson`, `PRECISE 2.0_spearman` beside the `PRECISE`
   columns, and 173 of the 201 PRECISE-1K iModulons have a non-blank `PRECISE 2.0` match
   (103 have a `PRECISE` match); `k12_modulome/imodulon_table.csv` matches 150 to PRECISE 2.0,
   159 to PRECISE-1K and 91 to PRECISE. **`grep -i "PRECISE 2" paper.md` returns nothing**, so
   the mirrored paper and SI never name this dataset. What PRECISE 2.0 is, and whether it is
   a separate release with its own samples, is not answerable from the mirror.

4. **Proteomics, as a gene-level availability split only.** Paper, Results: "Genes for which
   proteomics data is available in two large datasets (46,47) have significantly higher
   expression $\cdot P = 1 . 2 \mathrm { E } { - } 1 5 0$ , Mann-Whitney $U _ { ☉ }$ , $m = 2 0 3 1$ ,
   $n = 2 2 2 6 )$". Ref 46 = Schmidt 2016, which we already load
   (`ProteomeSchmidt2016Dataset`). Ref 47 = Heckmann et al. 2020 PNAS, "Kinetic profiling of
   metabolic specialists demonstrates stability and consistency of in vivo enzyme turnover
   numbers", which has no loader in `torchcell/datasets/ecoli/`. The comparison releases no
   per-record number (it is a binary has-proteomics split used to color a 2-D histogram), so
   it is a dataset POINTER, not a loadable quantity here.

5. **RegulonDB, as the TRN gold standard.** `data/annotation/TRN.csv`; annotation, not
   phenotype.

6. **iModulonDB** is a portal over the same A and M matrices, not a separate release: "The
   PRECISE-1K and Public K-12 iModulons, along with those for the other organisms mentioned
   above, can also be explored at iModulonDB.org (20)."

#### Loadable-now estimate

1. **Per-sample growth rate.** **103 of the 241 records the loader already builds** carry a
   numeric `Growth Rate (1/hr)`; 89 are strictly positive, 14 are exactly 0. Basis: measured
   intersection of the 241 `p1k_` ids in
   `$DATA_ROOT/data/torchcell/rnaseq_lamoureux2023/preprocess/record_samples.json` with the
   non-blank cells of the mirrored `metadata_qc.csv` (all 241 matched the metadata by id).
   Those 103 span 45 distinct `full_name` conditions and 9 projects (`ytf` 28, `ica` 28,
   `misc2` 10, `cra_crp` 10, `fur` 4, `ssw` 4, `rpoB` 2, `nquinone` 2, `42c` 1), all strain
   MG1655. The ceiling over the whole compendium is 354 records (338 positive), reachable
   only if a growth-rate dataset relaxes the expression loader's genotype drops, which a
   growth rate does not need the MG1655-keyed expression matrix for.
   Three caveats that belong in the PR, not in the number:
   - **The 16 zeros cannot be read as measured zeros.** A 0 h^-1 rate is incompatible with a
     library harvested at OD600 ~0.5, and the release states nothing about the convention, so
     the honest encoding is to drop them or carry them as a typed gap, never as a measurement.
     `p1k_00003` (`fur__wt_dpd__1`, 0.2 mM DPD) is one of them.
   - **A verifier rule blocks the obvious encoding.** `torchcell/verification/environment_response.py`
     L3 `reference_zero` requires "the reference (parent-strain) response is 0 for a numeric
     readout", which an absolute rate in h^-1 cannot be. `torchcell/datasets/pputida/lim2025.py:113-118`
     already documents exactly this and chose a log2 ratio instead: "``MeasurementType.growth_rate``
     exists, but ``torchcell.verification.environment_response``'s L3 ``reference_zero``
     requires a numeric reference of exactly 0, which an absolute rate in h^-1 cannot be."
     So either the verifier gains a declared non-zero baseline (which lim2025's PR body already
     proposes) or the rate is stored as a ratio against a matched wild-type rate in the same
     condition. There are wild-type MG1655 rows with rates, so a ratio is constructible for
     some conditions, but not all; this is a design decision, not a mechanical port.
   - **No uncertainty and no replicate count are released** for the rate. The replicates of a
     condition each carry the same cell (a per-condition value repeated per library), so a
     sample SD over replicates would be a fabrication.
2. **The 98 short / low-FPKM genes.** 98 x 1,035 = 101,430 released values; adding them to the
   built records adds 98 genes x 241 records = 23,618 values and **zero new records**. Basis:
   measured row counts (4,355 minus 4,257). Already recorded, and the source's own QC removed
   them, so this is a note, not a recommendation.
3. **Public K-12, as a separate dataset.** **1,675 records** (one per public sample), 1,675 x
   4,257 = 7,131,675 expression values. Basis: measured 1,675 non-`p1k_` rows in
   `k12_modulome/metadata_qc.csv`, matching the paper's "the final set of 1675 high-quality
   publicly-available samples". The 3,125-column `counts.csv` is the pre-QC superset, so the
   loader would subset it by the 1,675 QC-passing accessions. Read the duplication section
   first; this is the item that needs a policy decision before code.
4. **Public K-12 `aerobicity` + `time`.** 1,425 and 391 records respectively, inside item 3;
   no standalone records.
5. **The 20 QC-failed libraries.** At most 20 new records (1,055 minus 1,035). I cannot count
   how many would survive the loader's genotype and environment drop rules without running the
   loader, which this audit does not do, so this is uncounted.

#### Not checked

- **The Zenodo archive's member sha256s.** The 278 MB archive is not on disk and I did not
  re-download it. Everything I report about unmirrored release files comes from the GitHub
  tree and raw blobs at commit `71e1157f83d8362bac9039113cccfcbdb8873119`, which is the commit
  the mirrored archive root `SBRG-precise1k-71e1157/` names. That is the same content by git
  identity, but it is NOT the pinned-archive read path, so treat these as unpinned reads.
- **`k12_modulome.json.gz`, `k12_only_p1k_ctrl.json.gz`, `k12_only_proj_ref.json.gz`,
  `precise1k.json.gz`, `precise.json.gz`.** Packaged `IcaData` objects; not opened (gzipped
  JSON, and the Public K-12 log2[TPM] matrix is presumably inside `k12_modulome.json.gz`,
  since the tree shows no `k12_modulome/log_tpm*.csv`).
- **The 4,257 x 1,035 cells of `precise1k/log_tpm.csv` and `log_tpm_norm_qc.csv`** (63 MB and
  84 MB); headers not even fetched. Their relationship to the stored matrix is already
  measured in the note for `log_tpm_norm_qc.csv`.
- **`data/precise1k/crp_binding.csv`** is headerless, so I cannot say what its two float
  columns are. No caption in `paper.md` or `si1.md` names it.
- **`k12_modulome/Escherichia_coli_20220127.tsv`, `K12_metadata.tsv`, `bioproject_list*.csv`,
  `metadata_qc_part*`, `metadata_qc_with_p1k.csv`** -- the SRA curation intermediates; not
  opened. `metadata_qc_with_p1k.csv` may differ from `metadata_qc.csv` in ways I did not check.
- **`data/precise1k/multiqc_data/*` (21 files) and `subsamples/*` (20 files).** Not opened;
  `multiqc_stats.tsv` is the summary of the former and I read it in full.
- **Whether the 16 zero growth rates are measured zeros or placeholders.** The release states
  no convention; neither `paper.md` nor `si1.md` mentions the growth-rate column at all.
- **What 'PRECISE 2.0' is.** Named only in two released `imodulon_table.csv` files, never in
  the mirrored paper or SI.
- **Whether the 1,675 public accessions overlap any dataset we might add.** I read the
  BioProject / PMID fill counts but did not cross-match the accession list against our
  bacterial loader inventory beyond ruling out Caglar 2017 on strain grounds.

### priceMutantPhenotypesThousands2018 -- `ecoli/price2018.py`

**SI inventory.** `$DATA_ROOT/torchcell-library/priceMutantPhenotypesThousands2018/si/`

- `si1.pdf` (PDF, 1,418,164 bytes, role `si_pdf`) -- the Supplementary Information document:
  Supplementary Figures 1 to 5 and Supplementary Notes 1 to 6, plus its own reference list.
  The 22 Supplementary Tables are NOT in it (its own contents page says "In separate excel
  file").
- `si1.md` (Markdown, 69,403 bytes, role `si_ocr`) -- the OCR of si1.pdf. Read.
- `si2.pdf` (PDF, 79,139 bytes, role `si_pdf`) -- the Nature Research Reporting Summary.
- `si2.md` (Markdown, 10,224 bytes, role `si_ocr`) -- the OCR of si2.pdf. Read in full. It
  carries the authoritative Data Availability statement, which is the most complete
  enumeration of what the release contains.
- `si3.xlsx` (XLSX, 5,631,030 bytes, role `si_data`) -- the Supplementary Tables workbook,
  22 sheets, Tables S1 to S22, one sheet each. Opened; every sheet and column listed below.
- `si/images/si1/` (5 jpg) and `si/images/si2/` (1 jpg) -- OCR-extracted figure images.

The OCR sidecars `si1_content_list.json`, `si1_middle.json`, `si1_ocr_provenance.json` and
their `si2_` counterparts also exist and are not described further. The paper OCR is
`paper.md` (117,812 bytes, sha256 `f3443cdcb2f722b5...`), read.

The release's bulk data is NOT in the SI: it lives at `https://genomics.lbl.gov/supplemental/bigfit/`
(archived at figshare 10.6084/m9.figshare.5134837) and in the Fitness Browser
(figshare 10.6084/m9.figshare.5134840). Four of its pages/files are already mirrored under
`torchcell-raw/wetmoreRapidQuantificationMutant2015/` and three under
`torchcell-raw/priceMutantPhenotypesThousands2018/`; the per-organism file list below is
read from the mirrored `data/bigfit/index.html` and `data/bigfit/html/Keio/index.html`.

#### si3.xlsx: every sheet, every column, row counts

Counted with openpyxl, where "data rows" means non-empty rows below the header row. Each sheet carries a free-text preamble above its header row.

| sheet | total rows | header row | data rows | columns |
|---|---|---|---|---|
| `TableS1_LikelyEssentialGenes` | 13,883 | 14 | 13,869 | organism, orgId, locusId, sysName, locus_tag, protein_id, uniprotId, scaffoldId, begin, end, strand, name, desc, GC, nReads, normreads, nPosCentral, dens, geneClass |
| `TableS2_Carbon` | 158 | 41 | 96 | Compound, CAS number, concentration, units, PlatePosition, + one column per each of 28 bacteria (`Escherichia coli BW25113` among them) |
| `TableS3_Nitrogen` | 182 | 67 | 96 | Compound, CAS number, concentration, units, PlatePosition, + one column per each of 27 bacteria |
| `TableS4_Stress` | 220 | 40 | 55 | Compound, CAS, CoreSet_forMutantFitnessAssays, Stock solution, Stock solution units, Solvent, Maximum concentration tested, Minimum concentration tested, + one column per each of 30 bacteria |
| `TableS5_Experiments` | 4,889 | 19 | 4,870 | orgId, organism, name, Group, short, Media, Condition_1, Concentration_1, Units_1, Temperature, pH, Aerobic_v_Anaerobic, Shaking, Growth.Method, has_exact_replicate, has_similar_replicate |
| `TableS6_4G11carbon` | 62 | 3 | 59 | locusId, group, scaffoldId, begin, end, desc, comment, fitness_fructose, fitness_4hydroxybenzoate |
| `TableS7_Ecoli_SpecificPheno` | 255 | 11 | 244 | name, sysName, Condition, conserved, orgs, desc, code, comment |
| `TableS8_ConservedLinks` | 13,202 | 10 | 13,192 | organism, orgId, locusId, sysName, locus_tag, protein_id, uniprotId, geneClass, desc, specific, cofit |
| `TableS9_CisplatinGenes` | 76 | 7 | 69 | Section, Gene name or annotation, Comment, + one locus-tag column per each of 28 bacteria |
| `TableS10_XyloseGenes` | 32 | 12 | 20 | Section, Gene Name, Comment, + one locus-tag column per each of 12 bacteria |
| `TableS11_ABCtransporter` | 119 | 18 | 101 | organism, orgId, locusId, sysName, locus_tag, protein_id, uniprotId, desc, Condition_1, class, comment |
| `TableS12_GeneAnnotations` | 475 | 19 | 456 | Category, organism, orgId, locusId, sysName, locus_tag, protein_id, uniprotId, new_annotation, comment, original_description, SEED_description, KEGG_description, protein_sequence |
| `TableS13_UncharProteins` | 345 | 10 | 335 | organism, domainName, domainId, orgId, locusId, sysName, locus_tag, protein_id, uniprotId, desc, specific, cofit |
| `TableS14_RB_TnSeq_Bacteria` | 33 | 1 | 32 | Bacteria, Source, Isolation, Public availability of wild-type strain, Reference |
| `TableS15_Oligos` | 95 | 1 | 94 | Name, Sequence, Notes |
| `TableS16_Plasmids` | 23 | 1 | 22 | Plasmid, Description, Construction note |
| `TableS17_SingleStrains` | 31 | 1 | 30 | Strain, Description, Source |
| `TableS18_Medias` | 651 | 9 | 597 | Controlled vocabulary, Concentration, Units (stacked per-medium blocks, each headed `Media` / `Description` / `Minimal`) |
| `TableS19_GenomeSequencing` | 7 | 1 | 6 | Organism, Nickname, Accession, Assembly Size (nt), Closed, #Contigs, #Proteins, GC content, Origin, Technologies, Assembly Strategy, PacBio average coverage |
| `TableS20_Mutagenesis` | 42 | 1 | 40 (32 strains + 8 footnote rows) | Strain, Mutant library name, Transposon1, Number of unique barcodes2, Mutants used in fitness calculation per gene3, Method of delivery, Conjugation D:A ratio4, Conjugation Time (hrs), Conjugation Temperature (Celsius), Media5, Temperature for selecting mutants, Antibiotic; concentration (in ug/mL), Reference |
| `TableS21_subsystem_CarbonLinks` | 37 | 5 | 32 | Group, Carbon source, subsystem |
| `TableS22_GenomeAnnotations` | 38 | 6 | 32 | Source, Organism, IDs, Comment |

#### The per-organism release files (one directory per bacterium)

From the mirrored `data/bigfit/html/Keio/index.html` (sha256 `f05323ffce7701df...`), whose
header reads "207 condition samples (162 successful), Fri Feb 19 10:53:17 2016, statistics
version 1.0.3". Each of the 32 organism directories carries the same file set:
`fit_quality.tab`, `expsUsed`, `fit_genes.tab`, `fit_logratios_good.tab`,
`fit_logratios.tab`, `fit_logratios_unnormalized.tab`,
`fit_logratios_unnormalized_naive.tab`, `fit_t.tab`, `fit_standard_error_obs.tab`,
`fit_standard_error_naive.tab`, `cofit`, `specific_phenotypes`, `gene_counts.tab`,
`all.poolcount`, `strain_fit.tab`, `fit.image`, `log`, plus plots. The compendium-level
page adds `orginfo.tab`, `essential_proteins.tab`, `AllConsLinks.tab`,
`FEBA_anno_withrefseq.tab`, `high_fitness_feba.tab`, `genomes.tar.gz`,
`refseq_mapping.tsv`, `uniprot.map`, `strainusage.tar.gz`, `mapping/mapping_<orgId>.tar.gz`
and three dated R images.

**Multi-organism scope, measured.** `TableS5_Experiments` holds 4,870 rows over exactly 32
distinct `orgId` values; exactly one is E. coli (`Keio` = `Escherichia coli BW25113`, 162
rows). There is no second E. coli organism in the release. The other 31 organisms carry
4,708 experiments. Per-organism experiment counts (measured): psRCH2 303, Phaeo 262,
Marino 249, pseudo3_N2E3 205, Caulo 196, Cola 196, SB2B 194, Dino 184, pseudo6_N2E2 180,
pseudo5_N2C3_1 176, MR1 176, Koxy 173, Miya 170, Pedo557 166, **Keio 162**, PV4 160, Korea
148, WCS417 148, acidovorax_3H11 145, pseudo1_N1B4 140, SynE 129, pseudo13_GW456_L13 110,
Kang 108, Cup4G11 104, Ponti 104, ANA3 103, azobra 93, BFirm 89, Smeli 86, PS 79, Dyella79
69, HerbieS 63.

**What the loader keeps, exactly.** Of the 162 `Keio` rows: Group = carbon source 66,
stress 55, nitrogen source 32, motility 7, lb 2 (measured). The loader keeps 147 and the 15
drops are already documented. Set prefixes: set1 88, set2 64, set6 10. Every `Keio` row is
`Aerobic`; the `pH` column is empty for all 162 (measured), so no pH value is lost for a
stored sample.

**What the loader stores.** `EnvironmentResponsePhenotype(measurement_type=log2_ratio,
assay_type=pooled_competitive_growth_barcode)` inside
`BacterialEnvironmentResponseExperiment`, one record per (gene, sample), from
`data/bigfit/html/Keio/fit_logratios_good.tab` (the 162 successful Keio columns; the
loader keeps 147). Uncertainty is `UncertaintyType.standard_error` holding
`max(fit_standard_error_obs, fit_standard_error_naive)`. **553,896 records = 147 samples x
3,768 mapped genes** (`notes/torchcell.datasets.ecoli.price2018.md`, "What was built
(measured 2026-10-07)" table, row "records (`len(dataset)`)"). The file's 4th column `comb`
is a concatenated label string (`"b0001 thr operon leader peptide (NCBI)"`), not a value
(measured), so nothing is lost by ignoring it.

**Already recorded as not stored.**

- The released moderated t statistic has no slot on the phenotype --
  `price2018.py:40` ("The t statistic itself has no slot on the phenotype and is not
  stored") and the note's "Gaps and open questions" bullet "**The t statistic has no slot**".
- `n_samples` / `sample_unit` (the per-gene usable strain count, released only in the R
  image field `n`; Table S20 gives only its median 12) -- `price2018.py:42-44` and
  `SOLVENT_GAP`'s neighbors `N_SAMPLES_GAP` / `SAMPLE_UNIT_GAP` at `price2018.py:601-627`.
- The 15 dropped Keio samples (4 sucrose/D-mannitol withdrawn 2021, 4
  `MOPS Rich Defined media_noCarbon`, 7 soft-agar motility) and the 21 genes with no
  one-to-one ECK pair -- `price2018.py:959-977` (`DROP_RULES`) and the note's
  "Sample inventory by source and the drops" table.
- Table S5 `Shaking` and `Growth.Method` -- note, "Environment" section: "Shaking and
  vessel are not stored: the bacterial experiment class takes a plain `Environment`, not
  `CultureEnvironment`."
- The 53 unresolved stress-compound identities (typed `inchikey` gap) -- note,
  "Environment" section, "**53 Condition_1 labels of kept samples are not in the pinned
  compound-identity table**".
- The per-strain tables, `expsUsed` and the R image are deliberately not mirrored --
  `price2018.py:755-761` (`si_expected`) and the raw manifest's `si_expected` field.
- Duplication with Wetmore 2015 -- `wetmore2015.py:11-32`, "DECISION: SUBSUMED, NOT
  LOADED".

#### Released quantities and their classification

| released quantity | source file + column | stored | classification | evidence |
|---|---|---|---|---|
| gene fitness, normalized log2 ratio, 162 successful Keio samples | `bigfit/html/Keio/fit_logratios_good.tab`, columns 5..166 (`set1IT003 D-Glucose (C)` ...) | yes (147 samples) | stored | 3,790 lines x 166 columns measured; page: "Gene fitness (normalized log2 ratios for successful experiments )" |
| estimated standard error (strain consistency) | `fit_standard_error_obs.tab`, 220 experiment columns | partly | stored as the max of the two | measured 3,790 lines x 223 columns |
| naive standard error (total counts) | `fit_standard_error_naive.tab`, 220 experiment columns | partly | stored as the max of the two | same shape |
| moderated t statistic, the 147 stored samples | `fit_t.tab`, the 162 good columns | no | blocked: `EnvironmentResponsePhenotype` has no t-statistic / test-statistic field. **Already recorded** (note, Gaps) | page: "t-like test statistic (based on consistency of measurements for the gene)" |
| gene fitness, t and both SEs for the 45 FAILED Keio condition samples and the 13 Time0 controls | `fit_t.tab` / `fit_standard_error_*.tab` columns absent from `fit_logratios_good.tab`; the fitness is in the unmirrored `fit_logratios.tab` | no | blocked: nothing on `Phenotype` records that a value failed the release's own quality rule (`u = FALSE`) or is a Time0-vs-Time0 negative control, so these would read as equal to passing measurements. Not named by #749, #753, #756, #758 or #760 | measured: `fit_quality.tab` has 220 rows, `u` True 162 / False 58, `short == "Time0"` 13, so 45 failed condition samples. Their columns include a complete **LB pH5/pH6/pH7/pH8 series (12 samples)** and Vancomycin, Cephalothin, Polymyxin B, Gentamicin, Neomycin, Tetracycline, Carbenicillin, Lomefloxacin, Sisomicin, Phosphomycin, Benzethonium chloride, 5-Hydroxymethylfurfural, nitrate, nitrite, perchlorate, 1-ethyl-3-methylimidazolium lysine, NAG (N), Fumarate (C), L-Lactate (C) |
| unnormalized gene fitness; naive unnormalized log ratios | `fit_logratios_unnormalized.tab`, `fit_logratios_unnormalized_naive.tab` | no | not a phenotype: the pre-normalization form of a value we store | Keio page links "Unnormalized logratios" and "Naive unnormalized logratios" |
| reads per gene per sample | `gene_counts.tab` | no | not a phenotype: a sequencing read count, the assay's raw intensity | page: "Counts per gene per sample (#reads, summed over strains)" |
| **per-strain fitness (log2 ratio, normalized) and its rough SE** | `strain_fit.tab`; `strain_lrn` / `strain_se` in `fit.image` | no | **loadable now**: `EnvironmentResponsePhenotype.environment_response` with a strain-level `TransposonInsertionPerturbation(barcode=..., insertion_position=..., insertion_strand=...)` -- all three fields exist on the leaf and the loader currently sets them `None` | page: "Strain fitness (log2 ratios, normalized, large)"; "strain_se -- a rough estimate of how noisy the per-strain fitness values are" |
| per-strain BarSeq counts per sample | `all.poolcount` | no | not a phenotype: read counts | page: "BarSeq count for each strain by each sample (large)" |
| per-gene `nEff`, `tot`, `tot0`, `tot1`, `tot2`, `tot1_0`, `tot2_0`, `lrn1`, `lrn2`, `lr1`, `lr2` | `fit.image` | no | not a phenotype: count totals and half-gene re-computations of a value we store | Keio page, "The R image" section |
| per-sample quality metrics (nMapped, nPastEnd, nGenic, nUsed, gMed, gMedt0, gMean, cor12, mad12, mad12c, mad12c_t0, opcor, adjcor, gccor, maxFit, u) | `bigfit/html/Keio/fit_quality.tab`, 20 columns x 220 rows (measured) | no | not a phenotype: per-sample assay QC, not a strain x condition measurement | header read directly |
| top cofitness hits per gene (Pearson r) | `cofit`; `cofit` in `fit.image` | no | not a phenotype: a derived correlation between two genes' stored fitness vectors | paper: "Cofitness is the Pearson (linear) correlation of all of the fitness values for a pair of genes in the same bacterium" |
| specific-phenotype calls | `specific_phenotypes`; `specphe` in `fit.image` | no | not a phenotype: a derived call thresholded on fitness and t ("abs(fit) > 1 and abs(fit) > percentileFit + 0.5 and abs(t) > 5", Keio page) | Keio page, R image section |
| strong positive fitness effects across all organisms, with `fit`, `t`, `se`, `gMean`, `maxFit`, `used` | `bigfit/high_fitness_feba.tab` | no | blocked: same `u = FALSE` gap as above. It is the authors' own sanctioned route into the failed experiments ("It include experiments that did not meet our quality thresholds due to strong positive selection") | compendium page, "Strong positive fitness effects (tab-delimited)" |
| **likely-essential protein-coding genes, E. coli** | `si3.xlsx` `TableS1_LikelyEssentialGenes`, membership + `geneClass` | no | **loadable now**: `GeneEssentialityPhenotype.is_essential` in `BacterialGeneEssentialityExperiment` | 324 `orgId == "Keio"` rows measured; sheet preamble: "This table lists all of the likely-essential protein-coding genes in the 32 bacteria"; Methods: "Protein-coding genes were considered essential or important for growth (nearly essential) if we did not estimate fitness values for the gene and both the normalized insertion density and the normalized read density were under 0.2" |
| per-gene TnSeq coverage: `GC`, `nReads`, `normreads`, `nPosCentral`, `dens` | `TableS1`, columns 14 to 18 | no | not a phenotype: the sequencing-coverage evidence the essentiality call is computed from, plus a sequence property (GC) | sheet preamble: "nReads -- thetotal number of TnSeq reads that lie within the gene"; "dens -- the density of central insertions, normalized so that the median across all genes is 1" |
| `locus_tag`, `protein_id`, `uniprotId` for the essential genes | `TableS1` | no | not a phenotype: identifier annotation (320 of 324 Keio rows carry a `BW25113_RS*` RefSeq tag, measured) | header read |
| **wild-type growth call on 94 carbon substrates, E. coli column** | `TableS2_Carbon`, column `Escherichia coli BW25113` | no | **loadable now with a serious caveat**: `EnvironmentResponsePhenotype(measurement_type=categorical, category=...)` with `Genotype(perturbations=[])` (precedent: `caglar2017.py:2226`, `gupta2024.py:85`, `schmidt2016.py:1680`). The caveat is that the TRUE label is a disjunction, not a growth observation | 96 data rows, E. coli True 38 / False 58 (measured). Sheet legend, verbatim: "Postitive for growth on the indicated carbon substrate with the wild-type bacterium or a successful genome-wide mutant fitness assay was done." and "A call of FALSE does not necessary mean that the bacterium does not grow on a given substrate." |
| **wild-type growth call on 45 nitrogen substrates, E. coli column** | `TableS3_Nitrogen`, column `Escherichia coli BW25113` | no | **loadable now, same caveat** | 96 data rows, E. coli True 38 / False 58 (measured); same disjunctive legend |
| substrate dose for the growth calls (`concentration`, `units`, `CAS number`) | `TableS2` / `TableS3` | no | not a phenotype: the condition's design parameter and a compound identifier | `units` measured: mM 91, mg/mL 2, % 1 (S2) |
| **wild-type IC50 for 55 stress compounds, E. coli column** | `TableS4_Stress`, column `Escherichia coli BW25113` | no | **blocked**: `MeasurementType` has no `ic50` member and the number is a concentration (the stock's own unit), not a response score; and 3 of the E. coli values are censored text, for which nothing typed exists (the same shape as #753's "no per-protein censoring flag") | 55 data rows; E. coli: 48 numeric, 2 "Above maximum" (Sodium Chlorate, L-Lysine), 1 "Below minimum" (Polymyxin B sulfate), 4 empty (measured). Sheet preamble: "The values reported are the half-maximum inhibitory concentrations (IC50)"; "Above maximum -- the maximum concentration of the compound tested did not inhibit the growth of the bacterium by 50%" |
| **per-compound stock `Solvent`** | `TableS4_Stress`, column `Solvent` | no | **loadable now** into `SmallMoleculePerturbation.solvent`, which the loader currently sets `None` under `SOLVENT_GAP`. See the correction section below | measured: all 35 distinct kept Keio stress `Condition_1` labels match a `TableS4` `Compound` by case-insensitive exact name, covering 55 of 55 kept stress samples; solvents are water 29, Dimethyl Sulfoxide 5, Ethanol 1 |
| `Stock solution`, `Stock solution units`, `Maximum / Minimum concentration tested`, `CoreSet_forMutantFitnessAssays` | `TableS4_Stress` | no | not a phenotype: design parameters of the dose-response prescreen | measured; stock units mg/ml 26, mM 25, vol% 4; CoreSet yes 32 / no 23 |
| `has_exact_replicate`, `has_similar_replicate` | `TableS5_Experiments` | no | not a phenotype: a derived flag over the sample list (Keio: exact True 107 / False 55, similar True 144 / False 18, measured) | sheet preamble: "has_exact_replicate -- TRUE if this experiment has a replicate with the same concentration of the compound" |
| E. coli specific-important phenotypes with `conserved`, `orgs`, `code` | `TableS7_Ecoli_SpecificPheno` | no | not a phenotype: a derived call over stored fitness plus the unstored t, with a hand-assigned DIRECT/INDIRECT/UNKNOWN/UNCHAR code | 244 rows, 173 distinct genes x 40 distinct conditions; conserved True 86 / False 158; code ? 95, DIRECT 82, UNKNOWN 41, INDIRECT 15, UNCHAR 11 (measured) |
| conserved specific phenotypes / conserved cofitness partners | `TableS8_ConservedLinks` columns `specific`, `cofit` | no | not a phenotype: derived, and released as semicolon-joined TEXT with no numeric cofitness value (measured) | 13,192 rows, 515 with `orgId == "Keio"` |
| DUF/UPF members with conserved links | `TableS13_UncharProteins` | no | not a phenotype: derived, same two text columns | 335 rows, 7 Keio (measured) |
| ABC transporter reannotation class | `TableS11_ABCtransporter` columns `class`, `comment` | no | not a phenotype: a hand-assigned annotation | 101 rows, 2 Keio (measured) |
| revised gene annotations + `protein_sequence` | `TableS12_GeneAnnotations` | no | not a phenotype: annotation and sequence | 456 rows, 3 Keio (measured) |
| cisplatin / xylose ortholog-group membership | `TableS9_CisplatinGenes`, `TableS10_XyloseGenes` | no | not a phenotype: ortholog-group annotation (locus tags, not values) | S9 69 rows with 39 non-empty E. coli cells; S10 20 rows with 8 (measured) |
| library metadata: 152,018 unique barcodes, median 12 strains per gene | `TableS20_Mutagenesis` | no | not a phenotype: library metadata. The median-12 half is **already recorded** as the `n_samples` gap | measured row: `Escherichia coli BW25113 \| KEIO_ML9 \| Tn5 \| 152018.0 \| 12.0 \| electroporation \| ... \| LB \| 37.0 \| Kanamycin; 50 \| PMID 25968644` |
| media formulations, oligos, plasmids, single strains, genome assemblies, SEED subsystems, annotation sources | `TableS18`, `TableS15`, `TableS16`, `TableS17`, `TableS19`, `TableS21`, `TableS22` | partly (`TableS18` through the media library) | not a phenotype: design parameters and reference metadata | headers read |
| single-strain OD600 growth curves (Keio-collection deletions and constructed mutants) | Extended Data Figs 3 to 8, Supplementary Figure 5; no released data table | no | not a phenotype we can load: figure-only, no table. The curves are plotted, the numbers are not released | Methods: "These growth assays were performed in a Tecan microplate reader (either Sunrise or Infinite F200) with absorbance readings $\mathrm{(OD_{600}})$ ) every $15\mathrm{{min}}$"; si2.md: "All of the effects of mutations in individual strains on growth (shown in Extended Data Figures 3-8 and Supplementary Figure 5) were observed consistently across multiple wells" |
| **per-gene-per-experiment fitness, t and both SEs for the other 31 bacteria** | one directory per `orgId` under `bigfit/html/` | no | **out of scope for an E. coli loader, not "loadable now"**: these are released per-gene-per-experiment quantities, but they need a host genome tier, an assembly pin and a loader per organism. One of the 31 is already landed from a superset release elsewhere in this repo (`torchcell/datasets/pputida/borchert2024.py`, which loads the KT2440 `Putida_ML5` matrix and whose docstring names this compendium's statistics identity); the other 30 have no host directory | 4,708 non-Keio experiments measured from `TableS5` |
| likely-essential genes for the other 31 bacteria | `TableS1`, non-`Keio` rows | no | out of scope for an E. coli loader (same reason) | 13,869 - 324 = 13,545 rows (measured); per-organism range SynE 614 to PV4 289 |
| wild-type carbon / nitrogen growth calls and IC50 for the other 29 / 26 / 29 bacteria | `TableS2`, `TableS3`, `TableS4` non-E. coli columns | no | out of scope for an E. coli loader | 28, 27 and 30 organism columns respectively (measured) |
| `fitness_fructose`, `fitness_4hydroxybenzoate` for 59 Cupriavidus basilensis 4G11 genes | `TableS6_4G11carbon` | no | out of scope (another organism), and a figure-only subset of that organism's own release | 59 rows; preamble: "This table lists the genes that are color coded in Figure 1c" |

Counts: **38 released quantities enumerated**; 3 stored (gene fitness plus the two SE
estimates it is the max of); **4 loadable now** (per-strain fitness + its SE, Table S1
E. coli essentiality, Tables S2/S3 wild-type growth calls, Table S4's `Solvent`);
**4 blocked** (the t statistic, the failed/Time0 experiments, `high_fitness_feba.tab`, the
IC50 column); **23 not a phenotype**; **4 out of scope for an E. coli loader** (the other
31 organisms' fitness, their essentiality, their growth/IC50 columns, Table S6).

#### Comparisons against other data

- **Its own earlier release, Wetmore 2015.** Paper Methods, verbatim: "Our analysis
  includes 385 successful experiments from Wetmore et al.9 and 36 successful experiments
  from Melnyk et al.12. The other 4,449 successful fitness assays are described here for
  the first time." 385 + 36 + 4,449 = 4,870, which is exactly the `TableS5` row count
  (measured). **The duplication risk is real and is already resolved in this repo:**
  `torchcell/datasets/ecoli/wetmore2015.py` registers no dataset
  (`wetmore2015.py:11`, "DECISION: SUBSUMED, NOT LOADED") because "The compendium's values
  are a re-analysis of the same samples (statistics version 1.0.3 against this paper's
  1.0.0) over a gene set containing every one of this paper's genes. Loading both releases
  would store each shared sample twice" (`wetmore2015.py:30-32`). 92 of the 162 Keio samples
  are Wetmore's and their records cite Wetmore 2015 as publication. The Price raw mirror
  consumes four release files through the Wetmore mirror by sha256 rather than copying
  them, and its manifest's `si_expected` says so. **No further action: landing both is not
  on the table, because only one of them registers a dataset.** The one loss is recorded:
  the E. coli fumarate condition (`set1IT029`), which Wetmore passed and the compendium
  re-analysis rejects at gMed 48.
- **Melnyk 2015** (36 successful experiments) is the Dechlorosoma suillum PS arm, not
  E. coli, so it carries no E. coli duplication risk.
- **The Keio single-gene deletion collection (Baba 2006 / Yamamoto 2009) and PEC**, as an
  essentiality benchmark. si1.md, Supplementary Note 1, verbatim: "By combining the
  profiling of E. coli chromosome database (PEC) and the results of systematically
  attempting to delete every gene in E. coli 1,6, we obtained a list of $330\ E.$ coli genes
  that were previously reported to be essential. 257 of these 330 genes $(78\%)$ were in our
  list of likely-essential genes from TnSeq analysis." And: "we expect that the true rate of
  false positives in our list of $E.$ coli proteins that are essential, or nearly so, for
  growth in rich media is somewhere between $6\%$ and $16\%$ ." This **points at a loadable
  dataset we do not have**: the Keio-collection / PEC essentiality list for E. coli K-12 is
  an external release (Baba 2006, Yamamoto 2009, PEC), not part of this paper. It is also
  the benchmark that makes the Table S1 essentiality call interpretable, and it bears on
  #749's note that Nichols 2011 is not in the mirror (#691).
- **Nichols 2011** (ref 5) is cited ONCE, in the introduction, as prior art for the general
  approach ("One approach for investigating the function of an unknown protein is to assess
  the consequences of a loss-of-function mutation of the corresponding gene under multiple
  conditions3-6"). There is **no quantitative comparison against Nichols 2011 anywhere in
  `paper.md` or `si1.md`** (grep: `Nichols` appears only in the reference list), so this
  paper gives no evidence about Nichols overlap. The Shiver 2016 loader (#749) remains the
  only route to that question.
- **A TnSeq study in another organism.** si1.md, Supplementary Note 1, on Shewanella
  oneidensis MR-1: "We compared this to a list of likely-essential genes that we had
  previously generated using a different transposon and Sanger sequencing 3. ... Of the 397
  genes in our new list, 298 were previously classified as essential and 83 were classified
  as unknown, just 16 were previously identified as dispensable". Reference 3 is
  Deutschbauer 2011 (PLoS Genet 7:e1002385, "Evidence-based annotation of gene function in
  Shewanella oneidensis MR-1 using genome-wide fitness profiling across 121 conditions"), a
  separate releasable fitness compendium. Out of scope for an E. coli loader.
- **Cofitness against protein-protein data: NOT benchmarked.** The cofitness accuracy test
  is against TIGRFAM subroles, not against any interaction dataset: "we determined for each
  query gene whether its functional role (TIGR subrole16) could be accurately predicted by
  the roles of its cofit genes ... the top 2,000 predictions from cofitness ( $r>0.86$ for
  gene pairs from one bacterium) had $63\%$ agreement in TIGR subroles, while the top 2,000
  predictions from conserved cofitness ... had $74\%$ agreement". STRING (Szklarczyk 2017,
  si1.md ref 37) is cited exactly once and qualitatively, for a single gene triple: "These
  proteins are predicted to be functionally related because of conserved proximity,
  cooccurrence, and coexpression 37" (si1.md, Supplementary Note 5, DUF444/SpoVR). So the
  release contains no cofitness-versus-interaction benchmark and points at no interaction
  dataset to load.
- **The authors' own 2021 withdrawal** (sucrose and D-mannitol) is a self-correction, not a
  cross-dataset comparison, and is already handled by `DROP_DISREGARDED`.
- **The Fitness Browser diverges from this release.** Compendium page, verbatim: "Also note
  that for some organisms, the Fitness Browser contains additional experiments beyond those
  described here. For these organisms, the cofitness values will not match." A future pull
  from `fit.genomics.lbl.gov` would therefore not be the same dataset as this one, which is
  a duplication hazard for any later Fitness Browser ingest.

#### Loadable-now estimate

1. **Per-strain fitness, E. coli** -- upper bound **24,626,916 records** = 152,018 unique
   barcodes (`TableS20` "Number of unique barcodes2", measured) x 162 samples. This is an
   upper bound, not a count: only strains inside the central 10-90% of a gene with enough
   Time0 reads carry a value ("Only strains containing insertions within the central
   $10{-}90\%$ of the gene and with sufficient abundance, on average, in the start samples
   were included in these calculations"), and `strain_fit.tab` is not mirrored, so the
   `used = TRUE` subset cannot be counted here. It would be the largest single E. coli
   dataset in the repo by an order of magnitude and would need an insertion-position-level
   record design, not a gene-level one.
2. **Table S1 likely-essential genes, E. coli** -- **324 records**, one per gene, measured
   as the `orgId == "Keio"` row count of `TableS1_LikelyEssentialGenes`. Measured and
   relevant: the 324 essential sysNames and the 3,789 genes in `fit_logratios_good.tab` are
   **disjoint (intersection 0)**, which the Methods predict by construction ("if we did not
   estimate fitness values for the gene"), so these records extend the gene universe rather
   than contradicting a stored fitness. Two honest caveats: the paper's own label is
   "essential or important for growth (nearly essential)" in the library-selection condition
   (LB plates at 37 C), not unconditional essentiality, and Note 1 puts the E. coli FDR at
   6% to 16%.
3. **Tables S2 and S3 wild-type growth calls, E. coli** -- **192 records** (96 carbon + 96
   nitrogen compounds, measured row counts), each a `Genotype(perturbations=[])` record. I
   would not load these as they stand: the TRUE label is a disjunction of a growth
   observation and a data-availability fact, so a TRUE record is not a measurement of
   growth, and the FALSE label is hedged by the sheet's own note.
4. **Table S4 `Solvent` into the existing 553,896 records** -- no new records; it fills
   `SmallMoleculePerturbation.solvent` on the stress records. 55 of 147 kept samples are
   stress (measured), so 55 x 3,768 = **207,240 records** would gain a vehicle in place of
   a typed gap.

Not estimable here: the other 31 organisms' per-gene-per-experiment value count. Estimate
(basis stated): the paper restricted analysis to "the 123,255 different non-essential
protein-coding genes for which we collected gene fitness data" over 32 bacteria, so the mean
is 3,852 genes per organism; 3,852 x 4,708 non-Keio experiments gives roughly **18.1 million
per-gene fitness values** for the other 31 organisms, with the same count again for t and
for each SE estimate. This is an estimate from a mean, not a measurement, because the
per-organism gene counts are not in the SI and the per-organism tables are not mirrored.

The failed-experiment block is measurable exactly: 3,789 genes x 58 non-good Keio columns
= **219,762 values** of t and of each SE estimate already sitting in our raw mirror,
whose matching fitness values are in the unmirrored `fit_logratios.tab`. Of those 58
columns, 45 are failed condition samples and 13 are Time0 controls (measured).

#### Loader-note accuracy: two stale claims found while auditing

Both are factual claims in the current loader/note that the SI contradicts. Neither is a
phenotype question, but both change what the records say.

1. **`SOLVENT_GAP`'s note is wrong about Table S4.** `price2018.py:638` states the gap as
   "no vehicle is stated per stress compound in the paper, Table S4 or Table S5", and the
   note repeats it ("`solvent=None` with a typed gap (no vehicle stated in the paper, Table
   S4 or Table S5)"). `TableS4_Stress` has a column literally named `Solvent`, with values
   water (47 rows), Dimethyl Sulfoxide (7) and Ethanol (1) (measured). All 35 distinct kept
   Keio stress `Condition_1` labels match a `TableS4` `Compound` by case-insensitive exact
   name, covering 55 of 55 kept stress samples. Example rows, verbatim: `Kanamycin sulfate |
   25839-94-0 | no | 100.0 | mg/ml | water`, `Chloramphenicol | 56-75-7 | no | 34.0 | mg/ml
   | Ethanol`. The one real caveat, which the gap note does not make, is that this is the
   solvent of the stock used for the **wild-type IC50 prescreen**, and whether the mutant
   fitness assays drew on the same stocks is not stated: `paper.md` never contains the
   strings "stock solution", "dissolved", "solvent", "DMSO" or "dimethyl" (grep, zero hits),
   and Table S4's own caveat is about concentration only ("The concentrations reported here
   are not necessarily the concentrations used for the mutant fitness assays"). So the gap
   may still be the right call, but the reason recorded for it is false and should be
   restated as "Table S4 gives the prescreen stock solvent; the paper does not say the
   fitness assays used those stocks".
2. **The "derived mapping has no typed slot" gap is stale.** The note's Gaps bullet says
   "The typed form would be a field on `TransposonInsertionPerturbation`, e.g.
   `source_identifier: str | None` plus `identifier_route: Literal["direct",
   "eck_crosswalk"] | None` (a schema change; the Mutalik 2020 loader names the same need)".
   That field exists today: `TransposonInsertionPerturbation.identifier_mapping:
   DerivedIdentifierMapping | None`, where `DerivedIdentifierMapping` carries
   `source_identifier` verbatim plus `route: DerivedIdentifierRoute`, and its validator
   explicitly handles the `eck_crosswalk` route ("an eck_crosswalk route starts from another
   strain's locus tag"). The Price loader does not set it (grep: no `identifier_mapping=` in
   `price2018.py`); it carries the derivation in the leaf's free-text `description` and in
   `preprocess/identifier_mapping.json`. All 553,896 records could carry the typed mapping
   with no schema change.

Also worth the owner's eye, not previously recorded: the verifier's L1 row reports
`Environment.duration_generations x553896` as an undeclared None. That absence is genuine
and the paper says why -- the generation count was measured only for six bacteria, and
E. coli is not one of them: "We tested the impact of rescaling the fitness values before
computing cofitness for six bacteria (C. crescentus N1000, E. vietnamensis, D. japonica
UNC79MFTsu3.2, K. michiganensis M5a1, P. actiniarum, and Pedobacter sp.
GW460-11-11-14-LB5). For these bacteria, we have accurate measurements of the total number
of generations of growth for every successful fitness assay." The paper gives only the range
for everything else ("typically 4-8 population doublings"), so the honest form is a declared
gap rather than an undeclared None.

#### Not checked

- **si1.pdf and si2.pdf as PDFs**: read only through their mirrored OCR (`si1.md`,
  `si2.md`). si2.md was read end to end; in si1.md, the contents page, Supplementary Figure
  captions 1 to 5 and Supplementary Note 1 were read end to end, Note 6's opening read, and
  Notes 2 to 5 searched by keyword rather than read end to end. Those notes are per-gene
  annotation rationales; the per-record numbers they discuss are the Table S7 to S13
  columns already enumerated above. No value in them was taken on trust.
- **Every unmirrored release file**: `fit_logratios.tab`, `fit_logratios_unnormalized.tab`,
  `fit_logratios_unnormalized_naive.tab`, `gene_counts.tab`, `cofit`,
  `specific_phenotypes`, `all.poolcount`, `strain_fit.tab`, `fit.image`, `expsUsed`, `log`,
  `high_fitness_feba.tab`, `orginfo.tab`, `essential_proteins.tab`, `AllConsLinks.tab`,
  `FEBA_anno_withrefseq.tab`, `strainusage.tar.gz`, `mapping/mapping_Keio.tar.gz` and the
  31 non-Keio organism directories. Reason: an audit does no network retrieval, and the
  compendium tarball is 84 GB ("or as a tarball for all genomes here (large! 84 GB)"). Their
  existence, file names and column definitions are read from the two mirrored release
  index.html pages, which is why every claim about them cites a page quote rather than a
  measured row count.
- **The three figshare archives** (10.6084/m9.figshare.5134837, .5134840, .5146309) and the
  Fitness Browser SQLite database: not fetched, so I cannot say whether the Browser's
  additional E. coli experiments beyond these 162 exist.
- **The OCR figure images** under `si/images/` and `images/`: not opened.
- **`TableS18_Medias` beyond its header shape**: 597 data rows of stacked per-medium blocks
  were not parsed per medium; the media that matter here are already in
  `torchcell/datamodels/media.py`.
- **Whether `fit_logratios.tab`'s failed columns are numerically complete**: the t and SE
  columns for the 58 non-good samples are populated in our mirror, but I did not verify that
  the fitness file carries the same 58 columns, because that file is not mirrored.

### tongGeneDispensabilityEscherichia2020 -- `ecoli/tong2020.py`

**SI inventory.** Ten files under
`$DATA_ROOT/torchcell-library/tongGeneDispensabilityEscherichia2020/si/`. There is NO
`siN.md` for any of them (no OCR of the SI), so the xlsx sheet names plus column headers
and `paper.md` are the whole evidence base. `paper_middle.json`,
`paper_content_list.json` and `paper_ocr_provenance.json` sidecars exist beside
`paper.md`. The publisher's own `SUPPLEMENTAL MATERIAL` block in `paper.md` lines 169-178
gives only file type and size, no captions:

> FIG S1, TIF file, 0.4 MB.
> FIG S2, TIF file, 0.2 MB.
> FIG S3, TIF file, 0.1 MB.
> FIG S4, TIF file, 0.1 MB.
> TABLE S1, XLSX file, 1.7 MB.
> TABLE S2, XLSX file, 0.02 MB.
> TABLE S3, XLSX file, 0.01 MB.
> TABLE S4, XLSX file, 0.1 MB.
> TABLE S5, XLSX file, 0.03 MB.
> TABLE S6, XLSX file, 0.01 MB.

The mirror's `manifest.json` carries each file's `original_filename`, which fixes the
mapping from `siN` to the paper's own numbering:

| file | bytes | publisher name | what it is |
|---|---|---|---|
| `si1.tif` | 443,662 | `mBio.02259-20-sf001.tif` | Fig. S1, the Biolog phenotype-microarray survey (190 carbon sources, 74 supporting growth of BW25113) |
| `si2.tif` | 207,284 | `mBio.02259-20-sf002.tif` | Fig. S2, the screening workflow ("The workflow is summarized in Fig. S2.") |
| `si3.tif` | 133,708 | `mBio.02259-20-sf003.tif` | Fig. S3, replicate correlation, the 3 SD cutoff, and the WT-vs-interquartile-mean check |
| `si4.xlsx` | 2,091,607 | `mBio.02259-20-st001.xlsx` | **Table S1**, the loaded workbook. 2 sheets |
| `si5.xlsx` | 18,994 | `mBio.02259-20-st002.xlsx` | Table S2, 3 sheets of gene lists |
| `si6.xlsx` | 13,965 | `mBio.02259-20-st003.xlsx` | Table S3, the Venn-category gene lists |
| `si7.tif` | 84,136 | `mBio.02259-20-sf004.tif` | Fig. S4, the 33 model-paradox genes and the liquid re-screen |
| `si8.xlsx` | 156,348 | `mBio.02259-20-st004.xlsx` | Table S4, 1,402 genes x 30 conditions of TP/TN/FP/FN |
| `si9.xlsx` | 29,656 | `mBio.02259-20-st005.xlsx` | Table S5, the same for the 198 genes with >= 1 false prediction |
| `si10.xlsx` | 10,229 | `mBio.02259-20-st006.xlsx` | Table S6, per-condition confusion counts and accuracy |

`si4.xlsx` sha256 `d291160d41f65a1a3bd4fc8fe7d58e42bbdd92b4be2ef8c74bb2f284db8e8c84` is
byte-identical to the raw mirror's `data/mBio.02259-20-st001.xlsx` (measured with
`sha256sum`). Reading it reproduces the loader's finding: the full bytes are refused
(`BadZipFile: Bad magic number for file header`) and the leading archive is 1,803,783 of
2,091,607 bytes with sha256 `1c3f7b8aea226f90e5bcb237ecf483f62dcac84c63569a57b13307fba54aa166`,
exactly `LEADING_ARCHIVE_SHA256` in `tong2020.py:220`. Every sheet below was read from
that leading archive.

Real sheet names and headers, with row counts measured by `pandas.read_excel`
(`header=None` shapes, so the header row is included):

- `si4.xlsx` sheets `['S1A - Endpoint Biomass', 'S1B - Comparison to Keio Data']`
  - `S1A - Endpoint Biomass`, raw shape (3797, 32) = 3,796 strains. Headers: `B-numbers`,
    `Gene`, then 30 carbon sources in sheet order `Galactose, L-alanine, D-alanine,
    Mannose, Glucosamine, Thymidine, Adenosine, Saccharate, Acetate, α-ketoglutarate,
    Malate, Succinate, Fumarate, Ribose, Fucose, Glycerol, Lactate, Oxaloacetate,
    Pyruvate, Galacturonate, Maltose, Fructose, Trehalose, Mannitol, Sorbitol,
    Glucuronate, Gluconate, Xylose, Glucose, N-acetyl Glucosamine`. Measured: 113,880 of
    113,880 value cells non-null.
  - `S1B - Comparison to Keio Data`, raw shape (3728, 6) = 3,727 rows. Headers verbatim,
    including the trailing spaces on the first two: `B-number`, `Gene names`,
    `MOPS glucose solid minimal media (This study)`, `LB_22hr (Baba06)`,
    `MOPS_24hr (Baba06)`, `MOPS_48hr (Baba06)`.
- `si5.xlsx` sheets `['S2A - Genes with phenotype', 'S2B - Y-genes with phenotypes',
  'S2C - Essential 27 conditions']`. Every sheet's headers are `B-numbers`, `Gene` and
  nothing else. Data rows measured: 342, 51, 74.
- `si6.xlsx` sheet `['Sheet1']`, raw shape (343, 3) = 342 data rows. Headers `Names`,
  `total`, `elements`. 25 rows carry a `Names` + `total`; the rest carry only `elements`.
- `si8.xlsx` sheet `['Sheet1']`, raw shape (1403, 32) = 1,402 genes. Headers `B-numbers`,
  `Gene`, then the 30 conditions under the sheet's own spellings (`alphaKG`, `D.alanine`,
  `L.alanine`, `NacetylGlucosamine`; the rest as in S1A). Measured: 42,060 of 42,060
  cells non-null, distinct values exactly `{TP, FP, TN, FN}`.
- `si9.xlsx` sheet `['Sheet1']`, raw shape (199, 32) = 198 genes. Same 30 condition
  headers; the first header is `Accession Number` here, not `B-numbers`. Distinct values
  `{TP, FP, TN, FN}`. Measured: every S5 b-number is also an S4 b-number.
- `si10.xlsx` sheet `['Sheet1']`, raw shape (13, 31). No header row of its own: column 0
  holds the row labels `TP, TN, FP, FN, Total:, Accuracy` and row 0 holds the 30 carbon
  sources. Rows 8-10 hold the decoding legend as a 2x2 block: `All conditions` /
  `Model Growth` / `Model No Growth` against `Experiment Growth` / `Experiment No Growth`.

**What the loader stores.** `FitnessPhenotype.fitness` on `BacterialFitnessExperiment`,
one record per (deletion strain x carbon source), from `si4.xlsx` sheet
`S1A - Endpoint Biomass` (`tong2020.py:232`, `ENDPOINT_SHEET`) across the 30 columns of
`CARBON_SOURCE_COLUMNS` (`tong2020.py:241-272`). 111,420 records built from 113,880
source cells; 3,644 Keio strains (109,320 records, BW25113) and 70 library strains (2,100
records, MG1655), per `EXPECTED_RECORDS` at `tong2020.py:1240` and the note's
`### Records dropped` table. From sheet `S1B - Comparison to Keio Data` the loader reads
ONLY the b-number column (`KEIO_ID_COL`, `tong2020.py:238`), as the Keio-vs-library
background discriminator; it reads no value column of S1B.

**Already recorded as not stored.** Retire these, do not re-report them:

- Carbon-source concentrations are a typed `ProvenanceGap` resolving to CarPE
  (`tong2020.py:884-899` `carbon_source_gap`; note `### Gaps (typed, never guessed)`,
  first row).
- Per-cell dispersion: Table S1A releases one value and no uncertainty
  (`tong2020.py:934-942` `_uncertainty_gap`; note `### Gaps`, second row).
- Fig. S3a's whole-data-set replicate correlation R = 0.911 is already named as the
  paper's only replicate statistic, and as an image (note `### Gaps`, second row;
  `tong2020.py:97-99`).
- The library strains' `n_samples` / `sample_unit` (`tong2020.py:919-931`; note `### Gaps`,
  third row).
- S1B's duplicated b-numbers (b0621, b2218) and its 3,725 distinct b-numbers (note
  `### Background strain (checklist item 3)`).
- That S1B's own glucose column agrees with S1A: "every one of them is in Table S1A and
  its glucose value is identical" (`tong2020.py:49-50`). Confirmed here by measurement
  (max absolute difference 0.0 over 3,727 rows), so storing S1B's `MOPS glucose solid
  minimal media (This study)` column would duplicate S1A `Glucose`.
- The 30 carbon-source compound `inchikey` gaps, the agar amount, and the two verifier
  rules that fail on the shared yeast wording (note `### Gaps`, `### Build, manifest and
  verification`).

NOT recorded anywhere in the loader or the note: the three `(Baba06)` value columns of
sheet S1B, Tables S2 through S6 (none is mentioned in either file; `grep -n -i
"Table S2\|Table S3\|Table S4\|Table S5\|Table S6\|EcoCyc\|Biolog"` over both returns
nothing), and the kinetic growth curves as a release.

| released quantity | source file + column | stored | classification | evidence |
|---|---|---|---|---|
| normalized 24 h end-point growth per (strain, carbon source), 113,880 values | `si4.xlsx` sheet `S1A - Endpoint Biomass`, 30 carbon-source columns | yes | stored as `FitnessPhenotype.fitness` | caption-equivalent in text: "These endpoint biomass values formed a highly replicating final data set of growth amplitudes for 3,796 genes tested across 30 different carbon sources (see Fig. S3 and Table S1A in the supplemental material)." |
| our own glucose value per Keio strain, 3,727 values | `si4.xlsx` sheet `S1B`, column `MOPS glucose solid minimal media (This study)` | yes (via S1A) | not a phenotype: a re-release of the stored S1A `Glucose` cell | measured max absolute difference 0.0 over the 3,727 rows against S1A `Glucose` |
| Baba 2006 LB rich-medium end-point OD600 at 22 h per Keio strain, 3,727 values | `si4.xlsx` sheet `S1B`, column `LB_22hr (Baba06)` | no | **blocked**: no carrier for an absolute end-point optical density | measured: min 0.1270, max 1.1300, median 0.7630, 513 distinct values, all at <= 3 decimals, none equal to 1.0. Baba's own Methods (mirrored `babaConstructionEscherichiaColi2006/si/si1.md:172`): "Mutants were tested for growth in $2 0 0 \mathrm { - } \mu \mathrm { l }$ LB medium with kanamycin as a rich medium in 96-well microplates ... incubated for $2 2 \mathrm { h }$ at $3 7 ^ { \circ } \mathrm { C }$ without shaking. Absorbance at $6 0 0 \mathrm { n m }$ was measured" |
| Baba 2006 MOPS minimal end-point OD600 at 24 h per Keio strain, 3,727 values | `si4.xlsx` sheet `S1B`, column `MOPS_24hr (Baba06)` | no | **blocked**, same gap | measured: min 0.0000, max 0.8170, median 0.3080, 417 distinct. Baba Methods: "Mutants were transferred with 96-inoculation pins ... from LB into $2 0 0 { \cdot } \mu 1 0 . 4 \%$ glucose MOPS medium with 2 mM Pi (Wanner, 1994) and kanamycin as minimal medium and incubated for 24 and $4 8 \mathrm { h }$ at $3 7 ^ { \circ } \mathrm { C }$ without shaking." |
| Baba 2006 MOPS minimal end-point OD600 at 48 h per Keio strain, 3,726 values | `si4.xlsx` sheet `S1B`, column `MOPS_48hr (Baba06)` | no | **blocked**, same gap | measured: min 0.0000, max 1.0570, median 0.3430, 637 distinct, 1 blank cell. Same Baba Methods quote |
| strain gene symbol | `si4.xlsx` S1A `Gene`, S1B `Gene names` | no (the pinned annotation's symbol is used instead) | not a phenotype: an annotation | `canonical_symbol`, `tong2020.py:697-709`, takes the symbol from the genome, not the release |
| the 342 genes with a low-growth phenotype in >= 1 carbon source | `si5.xlsx` sheet `S2A - Genes with phenotype`, columns `B-numbers`, `Gene` | no | not a phenotype: a derived gene list with no number | "In total, there were 342 genes $( 9 . 0 1 \% )$ showing a low growth phenotype on at least one carbon source (see Table S2A in the supplemental material)." Measured 342 data rows, 2 columns, no value column |
| the 51 poorly annotated genes with a phenotype | `si5.xlsx` sheet `S2B - Y-genes with phenotypes` | no | not a phenotype: a derived list plus an annotation class | "We also identified 51 genes with poorly annotated functions that showed a low growth phenotype in at least 1 carbon source (Table S2B)." Measured 51 data rows, 2 columns |
| the 74 genes with low growth in >= 90% of conditions | `si5.xlsx` sheet `S2C - Essential 27 conditions` | no | not a phenotype: a derived list, a count over the stored values | "Finally, we identified 74 genes that showed low growth in $9 0 \%$ of conditions, defining a set of genes which are essential in all minimal media, regardless of the carbon source (Table S2C)." Measured 74 data rows, 2 columns |
| Venn-category membership of the 342 genes, 25 categories with a `total` | `si6.xlsx` sheet `Sheet1`, columns `Names`, `total`, `elements` | no | not a phenotype: a derived set-intersection annotation | "Overall, 96 genes showed a low growth phenotype in at least 1 carbon source in each category (Table S3)." Measured: the `Names` = `Entner-Doudoroff Lower Glycolysis Pentose Phosphate Pathway TCA cycle Upper Glycolysis` row carries `total` 96 |
| experimental growth / no-growth call per (gene, carbon source), 42,060 cells | `si8.xlsx` sheet `Sheet1`, decoded from `TP\|FN` = growth, `TN\|FP` = no growth | no | not a phenotype: a threshold on the value we already store | The legend is in the release itself (`si10.xlsx` rows 8-10: `Model Growth`/`Model No Growth` against `Experiment Growth`/`Experiment No Growth`). MEASURED: a single global threshold on the stored S1A value reproduces all 42,060 calls, 42,060/42,060 agree at 0.413769; growth values min 0.4138, no-growth values max 0.4129, a clean gap. The paper's stated rule lands inside that gap: interquartile mean of all 113,880 S1A cells minus 3 x SD of all cells = 0.413126, against "we used a 3 standard deviation cutoff below the interquartile mean of our entire data set (40) (Fig. S3)" |
| EcoCyc flux-balance predicted growth / no-growth per (gene, carbon source), 42,060 cells | `si8.xlsx` sheet `Sheet1`, decoded from `TP\|FP` = model growth, `TN\|FN` = model no growth | no | not a phenotype: an in-silico prediction, never measured | "Genes that had a predicted growth rate above our cutoff of $1 \times 1 0 ^ { - 3 } / \mathrm { h }$ were predicted to be dispensable in the model." Same `si10.xlsx` legend block |
| the same two calls for the 198 genes with >= 1 false prediction, 5,940 cells | `si9.xlsx` sheet `Sheet1` | no | not a phenotype: a measured subset of `si8.xlsx` | "We also identified 198 genes that have a false prediction in at least one carbon source (see Table S5 in the supplemental material)." Measured: every S5 b-number is an S4 b-number |
| per-carbon-source TP/TN/FP/FN counts, total 1,402, accuracy percent | `si10.xlsx` sheet `Sheet1`, rows `TP`, `TN`, `FP`, `FN`, `Total:`, `Accuracy` | no | not a phenotype: an aggregate over 1,402 genes | "Also, breakdown of the accuracy in each carbon source can be found in Table S6." Measured: Acetate column reads TP 1275, TN 61, FP 24, FN 42, Total 1402, Accuracy 95.292439 |
| data-set summary statistics (3,796 genes, 102 avg hits, 342, 74, 51) | main `TABLE 1 Summary statistics of data set` | no | not a phenotype: aggregates of the stored values | "aHits are defined as gene deletion mutants that had a final growth 3 SDs below the average growth." |
| global 2x2 confusion matrix (38,362 / 1,167 / 645 / 1,886) | main `TABLE 2` | no | not a phenotype: an aggregate | "The model predicted the same phenotype as experiments with $9 5 . 7 \%$ accuracy." |
| maximum growth rate of E. coli on each carbon source, 30 values | Fig. 3b only; no SI data file | no | not a phenotype here: a figure-only aggregate over strains, not a per-strain readout | "Using this growth curve, we calculated the maximum growth rate of E. coli grown on each carbon source (Fig. 3b)." The curve is "the interquartile mean ... for every mutant at every time point $( n = 1 , 8 9 8 )$" |
| WT (interquartile-mean) growth curve per carbon source, 72 time points x 30 | Fig. 3a only; no SI data file | no | not a phenotype here: figure-only aggregate | "The growth curves of E. coli for each carbon source were calculated by taking the interquartile mean (40) of integrated density of every colony at each time point." |
| 113,880 individual kinetic growth curves, 72 time points each | NOT in any SI file; CarPE web app only | no | not released in this SI: a per-record quantity with no mirrored source | "In our experiment, kinetic measurements were obtained at 20-minute intervals, adding up to over 8 million data points collected which generated 113,880 individual growth curves for each mutant under every condition." and "Users can select which carbon source conditions to show and download the data used to make such plots." |
| qRT-PCR fold change (dd-CT) for ydhC, nupG, nupC in glycerol and adenosine vs glucose | Fig. 5d only; no SI data file | no | not a phenotype here: figure-only, no released numbers | "(d) Reverse transcription-quantitative PCR (qRT-PCR) data showing the fold change in response (threshold cycle [ΔΔCT] values) for each gene (nupC, nupG and ydhC), relative to that in glucose" |
| intracellular adenosine accumulation by LC-MS | Fig. 5e only; no SI data file | no | not a phenotype here: figure-only, no released numbers | "(e) Accumulation assays measuring the amount of adenosine in cells that were treated with adenosine for 10 minutes." |
| ydhC complementation growth curves and the nupC/nupG/ydhC final amplitudes | Fig. 5b, 5c only | no | not a phenotype here: figure-only; 5c is a re-plot of stored S1A adenosine cells | "Shown is a representative example of many replicates." |

**Comparisons against other data.** Four, and the first is the material one.

1. **Table S1B is a secondhand release of Baba 2006's own growth data, and Baba's primary
   release is not in our mirror.** The comparison sentence:

   > We note here that, while our experiments were conducted in solid media, this glucose
   > data set had a $9 8 \%$ agreement with the results of a previous study of the Keio
   > collection grown in liquid cultures of MOPS glucose (Table S1B) (11).

   Reference 11 is `11. Baba T, Ara T, ... 2006. Construction of Escherichia coli K-12
   in-frame, single-gene knockout mutants: the Keio collection.` Baba 2006 IS mirrored
   (`babaConstructionEscherichiaColi2006`, the sha256 two other loaders already pin at
   `wang2015.py:199` and `fuhrer2017.py:157`), but its mirror holds only `paper.pdf`,
   `paper.md`, `si/si1.pdf` and `si/si1.md`: no data file, and `si_expected` and
   `si_data_sources` are both `[]` in its `manifest.json`. Baba's growth numbers live in
   its own Supplementary Table 3, which we do not have:

   > All mutants were profiled for growth yield in both rich (LB) and minimal glucose
   > MOPS media (Figure 6). ... Complete information is in Supplementary Table 3.

   There is no `baba*.py` in `torchcell/datasets/ecoli/`. So Tong's S1B is currently the
   ONLY copy of Baba 2006's per-strain growth data in our mirrors, for 3,727 strains and
   three conditions.
   - **Duplication risk, in the opposite direction from #760.** Loading the three
     `(Baba06)` columns under this citation key would attribute Baba 2006's measurements
     to Tong 2020, and would then collide with any later Baba 2006 loader built from
     Supplementary Table 3. The clean route is a Baba 2006 loader from Baba's own
     release; the fallback, if Supplementary Table 3 cannot be retrieved, is to load the
     columns with Baba 2006 as the `Publication` and Tong's workbook as the artifact, and
     to say so.
   - **The "98% agreement" is a call agreement, not a value agreement.** MEASURED on the
     3,727 paired rows of S1B: Pearson r between
     `MOPS glucose solid minimal media (This study)` and `MOPS_24hr (Baba06)` is 0.3829,
     against `LB_22hr (Baba06)` 0.1286 and `MOPS_48hr (Baba06)` 0.2843 (n = 3,727, 3,727,
     3,726). Hypothesis (untested): the 98% is the fraction of strains whose binary
     growth/no-growth call matches; the paper does not define it and no agreement column
     is released.

2. **Joyce 2006 (reference 14), a glycerol minimal-media Keio screen, is a loadable
   dataset we do not have.** Cited twice as a comparison:

   > On average, we identified 102 genes per carbon source that showed a growth defect, a
   > number similar in scale to the 119 genes that were found to have an indispensable
   > phenotype for growth in minimal media containing glucose or glycerol (Table 1) (11,
   > 14).

   and

   > Of the 17 remaining paradoxical genes, we compared our findings for these to 2 other
   > data sets, in which 1 uses glucose (11) as the sole carbon source and another uses
   > glycerol (14).

   Reference 14 is `Joyce AR, Reed JL, ... 2006. Experimental and computational
   assessment of conditionally essential genes in Escherichia coli. J Bacteriol
   188:8259-8271.` Not in the mirror (checked by listing
   `$DATA_ROOT/torchcell-library/`). The comparison itself is figure-only (Fig. S4), so
   nothing per-record comes from Tong here.

3. **Goodall 2018 (reference 12) is already loaded.** "Of these genes, seven were
   determined to be essential for growth in previous studies (12)", where 12 is
   `Goodall ECA, ... 2018. The essential genome of Escherichia coli K-12`. We have
   `torchcell/datasets/ecoli/goodall2018.py` and the key
   `goodallEssentialGenomeEscherichia2018` in the mirror, so this is a cross-check we
   already hold, not a new dataset.

4. **Yamamoto 2009 (reference 43) names 7 Keio strains whose deletion is not the
   genotype we store.** This bears on the designed-versus-realized perturbation question,
   not on a new phenotype:

   > An update to the Keio collection showed that these mutants contained a gene
   > duplication which explains why these mutants grew in our experiment despite being
   > essential for growth in rich media (43).

   Reference 43 is `Yamamoto N, ... 2009. Update on the Keio collection of Escherichia
   coli single-gene deletion mutants.` Not in the mirror. The paper does not name the
   seven genes in any released file; they are in Fig. S4 (a tif). Our 111,420 Tong
   records store these strains as clean single deletions, which is what Tong screened
   and what Tong's table says, so nothing is wrong today. Not a phenotype.

**Loadable-now estimate.** Zero items. Every per-record NUMBER Tong 2020 measures itself
and releases in its SI is either the stored S1A cell, a re-release of it (S1B's own
glucose column, measured identical), or a threshold call on it (Tables S2, S4, S5, S6,
with the threshold measured as exactly reproducible). The three unstored per-record
numbers in the SI are Baba 2006's, and they are blocked.

Blocked item, with its record count:

- **Baba 2006 end-point OD600, 10,931 records.** Basis: MEASURED, not estimated. The
  3,644 kept Keio strains of the current build (`preprocess/identifier_reconciliation.json`,
  3,644 crosswalk entries) all appear in S1B; among those 3,644 deduplicated rows,
  `LB_22hr (Baba06)` has 3,644 non-null, `MOPS_24hr (Baba06)` 3,644 and
  `MOPS_48hr (Baba06)` 3,643, so 10,931 cells. Over all 3,727 S1B rows the three columns
  hold 11,180 non-null cells. Two new conditions come with them: a RICH medium (LB), which
  no Tong record has, and a second MOPS time point (48 h beside 24 h).
  - **The exact missing thing.** The value is an absolute absorbance at 600 nm of a
    static 200 microliter 96-well culture, with no wild-type normalization: measured
    median 0.7630 (LB), 0.3080 (MOPS 24 h), 0.3430 (MOPS 48 h), and no cell equal to
    1.0 in any of the three columns. `FitnessPhenotype.fitness` is declared
    `Field(description="ko_growth_rate/wt_growth_rate")` (`schema.py:3552`), so an
    absolute OD is not what that field means. `EnvironmentResponsePhenotype` cannot take
    it either: `MeasurementType` has nine members
    (`log2_ratio, z_score, sensitivity_score, categorical, ordinal, growth_rate,
    differential_fitness, control_regression_residual, colony_size`) and none is an
    optical density, while `colony_size` is defined as "ABSOLUTE end-point colony size
    (mean radius in image pixels for the Bloom 2019 control plates)"
    (`schema.py:4653-4654`), and the verifier's L3 `reference_zero` rule requires
    "numeric rule: reference response == 0 for all ... records"
    (`torchcell/verification/environment_response.py:375-397`), which an absolute OD
    cannot satisfy. So the gap is one of: a `MeasurementType.optical_density` member
    plus an exemption from `reference_zero`, or an absolute-growth-readout phenotype.
  - **Not one of the five open issues.** #749 is the bacterial tagged-allele/degron leaf,
    `s_score`, and non-concentration doses; #753 is the turnover interval, censoring,
    chemostat dilution rate and the UniProt `db_xref`; #756 is the adapter's
    environment-versus-phage conf rule; #758 is quote verbatimness; #760 is the
    Rousset/Cui growth-screen duplication. None names an optical-density readout.

Not blocked by the schema but not retrievable from our mirrors:

- **The 113,880 kinetic growth curves, 72 time points each.** No SI file holds them; the
  paper puts them on CarPE, which the loader already records as a non-scriptable Shiny
  app (`tong2020.py:227-230`). As records they are 113,880 x 72 = 8,199,360 per-(strain,
  carbon source, time) values, which agrees with the paper's "over 8 million data
  points". Whether the downloadable series is the raw integrated density or the
  CarPE-plotted relative growth is NOT determinable from the mirror, so whether they
  would hit the same absolute-readout gap is unmeasured. Worth a line in the loader's or
  note's retention ledger, which does not currently mention the curves as a release at
  all.

**Not checked.**

- `si1.tif`, `si2.tif`, `si3.tif`, `si7.tif`: binary TIFF images, and there is no `siN.md`
  OCR for this key's SI, so nothing in them was read. Their content is described above
  only from the main text that cites them. In particular, whether Fig. S1 (the Biolog
  survey of 190 carbon sources) or Fig. S4 (the liquid re-screen of 26 genes in MOPS
  glucose and glycerol, 9 of which did not grow) carries numbers rather than marks is
  unverified; neither has a released data file, so any numbers in them would be
  figure-only.
- The CarPE "carbon conditions" tab and its per-mutant curve downloads: not fetched. The
  loader records a scripted GET returning only the app's loader page, measured 2026-10-07.
- Baba 2006's Supplementary Table 3, Joyce 2006 and Yamamoto 2009: absent from
  `$DATA_ROOT/torchcell-library/` (checked by listing it), so their own releases were not
  read. Baba's Methods quote above comes from the mirrored `si/si1.md`, which IS present.
- The 98% figure was not reproduced; no agreement column is released and the paper does
  not define the statistic.

### wangDynamicInterplayMultidrug2015 -- `ecoli/wang2015.py`

Wang, Yang, Shah, Choi and Kim 2015, Sci Rep 5:16505, doi 10.1038/srep16505,
"Dynamic interplay of multidrug transporters with TolC for isoprenol tolerance in
Escherichia coli".

**SI inventory.** The mirrored SI is one PDF and its OCR, nothing else. There is no
spreadsheet, no csv, no data deposit, and no GEO/PRIDE accession of the authors' own.

- `si/si1.pdf` -- 1,098,793 bytes, PDF, the single supplementary file
  (`srep16505-s1.pdf`), holding Tables S1-S8 and Figures S1-S6.
- `si/si1.md` -- 20,017 bytes, MinerU OCR of that PDF. All eight tables transcribe into
  it as HTML `<table>` blocks; the six supplementary figures transcribe as captions plus
  image references.
- `si/images/si1/*.jpg` -- 15 OCR-extracted figure images (the six supplementary figures
  plus panel crops).
- The `si1_middle.json`, `si1_content_list.json` and `si1_ocr_provenance.json` sidecars
  also exist and are skipped here.

**Raw-data mirror** `$DATA_ROOT/torchcell-raw/wangDynamicInterplayMultidrug2015/`, every
file:

| file | bytes | sha256 (head) | what |
|---|---|---|---|
| `data/si1.pdf` | 1,098,793 | `ed2a029a...` | the same SI PDF, from the PMC Open Access bucket (`pmc_cloud`, key `PMC4643228.1/srep16505-s1.pdf`) |
| `manifest.json` | 2,163 | n/a | provenance record for the one raw file |

**There are no tabular raw files, so there are no column headers to print.** The loader
does not read a released table; it runs `pdftotext -layout -enc UTF-8` (poppler 21.01.0,
recorded in `manifest.json` under `processing`) over the SI PDF's born-digital text layer
and parses Tables S2 and S3 out of it. The nearest thing to a column header is Table S3's
own two-level header, verbatim from `si1.md`:

> `<tr><td rowspan="2">Strains</td><td colspan="2">Cell growth (OD600)</td></tr><tr><td>No isoprenol</td><td>0.5% (v/v) isoprenol</td></tr>`

Measured by me from `si1.md` with
a one-off scratch script over `si1.md`'s HTML tables: Table S3 has **47 strain rows**
(first `BW25113` `8.30 ± 0.16` `4.12 ± 0.01`, last `BWΔtolC` `8.16 ± 0.04` `5.39 ± 0.08`)
and **94 cells of the form mean ± SD**, so 188 numbers. Table S4 has **9 gene rows** and
**33 numeric cells** (9 x 4 minus the 3 blanked self-deletion cells). Table S2 lists
**48 strain names** (46 Keio entries plus the two strains built in this study).

**What the loader stores.** `EnvChemgenWang2015Dataset`, a
`BacterialEnvironmentResponseExperiment` per strain with an
`EnvironmentResponsePhenotype` (`measurement_type = log2_ratio`,
`assay_type = liquid_od_growth`, `n_samples = 2`, `sample_unit = biological_replicate`).
The value is the log2 of the paper's own relative tolerance capacity to BW25113,
`log2[(OD_iso / OD_0)_strain / (OD_iso / OD_0)_BW25113]`, computed from BOTH Table S3
columns (`wang2015.py:1099` `log2_relative_tolerance`, `:1104` `response_phenotype`).
**46 records plus 1 reference** (`torchcell.datasets.ecoli.wang2015.md`, "Records and
drops", kept = 46; "Build and verification", "46 records, 1 reference"). Range -0.453
(emrA) to +0.633 (acrA). Uncertainty is a typed `ProvenanceGap` on every record
(`wang2015.py:548` `UNCERTAINTY_GAPS`).

**Already recorded as not stored.** The loader's module docstring
(`wang2015.py:81-85`, "NOT LOADED, because the paper releases them only as figures or as
another readout") and the note's "Not loaded, and why" heading
(`notes/torchcell.datasets.ecoli.wang2015.md:235-251`) already retire almost everything
below. Repeating them here only as pointers:

- The Fig. 2 time courses; the pTrc99A overexpression strains (Fig. 3C with IPTG, Fig. S3
  without); BWΔacrAB, BWΔABC and the 0.75% isoprenol arm (Fig. 4); **the other alcohols
  (Fig. 5)**; pT-tolC in BWΔABC (Fig. S5). Already recorded at note:237-239 and
  `wang2015.py:82-84`, and in the raw manifest's `si_expected`.
- Table S4, the 9-transcript RT-qPCR panel: "no phenotype class models a targeted qPCR
  panel" (note:244-245, `wang2015.py:84-85`). Used only as identity evidence for the
  swapped acrA/acrB strains (`SOURCED_VALUES["identity_evidence_acrA"]`, `..._acrB`).
- **The no-isoprenol OD600 column**: "consumed only as the normalizer; a fitness record
  of each strain in plain 2YT would be a second phenotype family in one dataset"
  (note:246-247).
- The Table S3 column SDs: they belong to the four OD600 means, not to the stored ratio,
  so none is stored and the ratio's SD is a typed gap (note:164-172, `wang2015.py:70-73`).
  The columns themselves are kept in `preprocess/table_s3.csv`.
- Kanamycin in the screen cultures, and the culture format (4 mL in a 55 mL tube), both
  quoted and deliberately unstored (note:248-249, `SOURCED_VALUES["culture_format"]`).

**Not in either ledger** (new to this audit, all of them figure-only or not a phenotype):
Figure S1 (the third-party butanol microarray), Figure S2 (the no-isoprenol time
courses), Figure S6 (transcript abundance relative to acrB), Table S1 (compound physical
properties) and Table S6 (amplicon sizes). None of them is loadable; the ledger is
complete in substance.

| released quantity | source file + column | stored | classification | evidence |
|---|---|---|---|---|
| OD600 mean, no isoprenol, 47 strains | `si1.pdf`/`si1.md` Table S3 column `No isoprenol` | consumed as the normalizer, not stored | **loadable now**: `FitnessPhenotype.fitness` in a `BacterialFitnessExperiment`; **already recorded as a decision**, note:246-247 | `<tr><td>BW25113</td><td>8.30 ± 0.16</td><td>4.12 ± 0.01</td></tr>`; caption "Table S3. Cell growth of the MDT null mutants in the absence and presence of isoprenol." |
| sample SD, no isoprenol, 47 strains | Table S3, same column | no | not a phenotype: the uncertainty of the row above; already recorded, note:164-172 | `Note: The results are presented as means $\pm$ standard divisions.` |
| OD600 mean, 0.5% (v/v) isoprenol, 47 strains | Table S3 column `0.5% (v/v) isoprenol` | consumed into the stored ratio | not a phenotype on its own: the stored `log2_ratio` is this value over the row above, over the parent's | same table row |
| sample SD, 0.5% isoprenol, 47 strains | Table S3, same column | no | not a phenotype: uncertainty of the row above; already recorded | same note line |
| transcript change to isoprenol, fold ± SD, 9 genes, n = 3, wild type | Table S4 column `Transcript changes to isoprenol (fold)a` | no | **blocked**: no phenotype class for a targeted RT-qPCR panel. `MicroarrayExpressionPhenotype` requires `expression` (absolute linear-scale per gene) and `n_replicates`, which this table does not release, so it cannot hold a fold-change-only panel. **None of #749, #753, #756, #758, #760 names this gap.** Already recorded as a decision, note:244-245 | `<tr><td>macB</td><td>2.36 ± 0.41</td>...`; note a) `E . coli BW25113 was grown in 2YT medium with $0 . 5 \%$ (v/v) of isoprenol at $3 0 \mathrm { { ^ \circ C } }$ for $^ { 6 \mathrm { h } }$ . Changes (folds) were normalized to those of the cultures without isoprenol` |
| transcript fold ± SD in BWΔacrA, BWΔacrB, BWΔtolC, 24 cells | Table S4 columns `BWΔacrA`, `BWΔacrB`, `BWΔtolC` | no (used as identity evidence only) | **blocked**, same gap as the row above | `<tr><td>acrD</td><td>1.65 ± 0.16</td><td>1.44 ± 0.15</td><td>1.91 ± 0.17</td><td>2.40 ± 0.16</td></tr>` |
| molar mass, API gravity, heat of combustion, logPo/w, fuel in water, 9 compounds | Table S1 columns `Molar mass (g/mol)`, `API gravities a`, `Heat combustion (MJ/L)`, `logPo/wc`, `Fuel in water (%)` | no | not a phenotype: physical properties of a compound, and three of the five are calculated or predicted, not measured | note: `API gravities are calculated according to chemical density ... logPO/W is predicted at website www.molinspiration.com` |
| Keio number and MDT family, 48 strains | Table S2 columns `Keio No.`, `MDT families` | Keio number stored as `construction.strain_accession` when it names the same locus; family in the ledger only | not a phenotype: strain annotation | `<tr><td>BWΔmacA</td><td>JW0862</td><td>ABC</td>...` |
| amplicon size, 11 primer pairs | Table S6 column `Amplicon sizes` | no | not a phenotype: assay design parameter. Tables S5, S7, S8 are primers and plasmids, also design | `<tr><td>QacrA-F QacrA-R</td><td>...</td><td>189 bp</td><td>2</td></tr>` |
| growth inhibition %, 46 strains, 5-bin color | Figure 1C (main text, figure only) | no | not a phenotype: a derived summary of Table S3, `1 - OD_iso / OD_0` | `By comparison of growth inhibition $( \% )$ of each mutant relative to wild type strain, the mutants were categorized into 5 groups` |
| OD600 time course, 10 strains, 0-24 h every 6 h, in 0.5% isoprenol | Figure 2A, figure only | no | not a phenotype: figure-only value; already recorded, note:237 | `Cell growth was measured every 6 h.` |
| growth inhibition time course, same 10 strains | Figure 2B, figure only | no | not a phenotype: figure-only, and derived from 2A and Fig. S2; already recorded | `Initial growth inhibition was assumed as $5 0 \%$ in all strains.` |
| growth inhibition %, 7 overexpression constructs x 2 hosts + 2 empty-vector controls, 0.1 mM IPTG | Figure 3C, figure only | no | not a phenotype: figure-only; already recorded, note:237-239 (with the modeling route if ever digitized) | `All strains were grown in 2YT medium with $0 . 5 \%$ (v/v) of isoprenol and $0 . 1 \mathrm { m M }$ of IPTG at $3 0 ^ { \circ } \mathrm { C }$ for $1 2 \mathrm { h }$ .` Constructs read off the panel: AcrD, EmrAB, MacAB, MdtBC, MdtJI, YdiM, TolC |
| growth inhibition %, 5 strains x 2 isoprenol doses (0.5%, 0.75%) | Figure 4, figure only | no | not a phenotype: figure-only; already recorded, note:239. BWΔacrAB and BWΔABC appear in no table, and 0.75% is the only second dose in the paper | `All mutants were grown in 2YT medium with $0 . 5 \%$ (v/v) (dark cyan bars) or $0 . 7 5 \%$ (v/v) (red bars) of isoprenol` |
| **growth inhibition %, 6 further alcohols x 2 strains (BW25113, BWΔacrAB)** | **Figure 5, figure only, no table anywhere** | no | not a phenotype: figure-only value; already recorded, note:239 ("the other alcohols (Fig. 5)") | caption `E. coli BW25113 (dark cyan bars) and BWΔacrAB (red bars) were grown in 2YT medium with various $\mathrm { C } _ { \mathrm { m } } \mathrm { O H s }$ at a given concentration $( \% , \bf { v } / v )$ at $3 0 ^ { \circ } \mathrm { C }$ for $1 2 \mathrm { h }$ .` Axis labels read off the figure image: `C4OH (0.6%,v/v)`, `C5OH (0.3%,v/v)`, `3M-C4OH (0.3%,v/v)`, `C6OH (0.1%,v/v)`, `2M-3=C4OH (0.5%,v/v)`, `3M-2=C4OH (0.5%,v/v)` |
| OD600 time course without isoprenol, 10 strains, every 6 h | Figure S2, figure only | no | not a phenotype: figure-only. NOT in either ledger (the ledger covers the Table S3 no-isoprenol column, not this figure) | `Cell growth was measured every $^ \textrm { \scriptsize 6 b }$ . Results are the means of two biological replicates.` |
| growth inhibition %, 6 constructs x 2 hosts, no IPTG | Figure S3, figure only | no | not a phenotype: figure-only; already recorded, note:237-238 | `Growth inhibition by isoprenol on the strains expressing the identified transporters in the absence of IPTG.` |
| growth inhibition %, BWΔABC + pT-tolC, with and without 0.2 mM IPTG | Figure S5, figure only | no | not a phenotype: figure-only; already recorded, note:239 | `Growth inhibition was determined from BWΔABC expressing TolC in absence (dark cyan bar) and presence (red bar) of $0 . 2 ~ \mathrm { m M }$ of IPTG.` |
| transcript abundance relative to acrB, with and without isoprenol, n = 3 | Figure S6, figure only | no | not a phenotype: figure-only, and the same qPCR gap as Table S4. NOT in either ledger | `Transcript changes (folds) were normalized to acrB gene. Results are the means of three biological replicates.` |
| log2 expression of 46 transporter genes x 4 time points under 0.8% butanol | Figure S1, figure only, and **not this lab's data** | no | not a phenotype of this paper: a re-plot of GEO GSE16973. NOT in either ledger | `The data are adopted from Gene expression Omnibus (Accession No. GSE16973)9. Butanol was added at a concentration of $0 . 8 \%$ for a given time.` Time points read off the figure image: 0, 30, 80, 195 min |

Counts: **20 released per-record quantities enumerated; 0 stored verbatim** (2 of them,
the two Table S3 OD600 mean columns, are consumed into the one stored derived value);
**1 loadable now** (already a recorded decision); **2 blocked** (both halves of the same
Table S4 qPCR gap); **17 not a phenotype**, 11 of those figure-only.

**The multi-solvent question, answered exactly.** The paper measures tolerance to seven
alcohols, and we store one. But the six others are released **only as the bar heights of
Figure 5**. There is no Table, no column, and no number for them anywhere in the SI text:
`si1.md` contains Figure 5's caption only through the main paper, and the SI's own tables
run S1 (compound properties), S2 (strains), S3 (isoprenol OD600), S4 (qPCR), S5-S7
(primers), S8 (plasmids). The six, with the dose each was tested at, read off the Figure 5
axis image (`images/07e4ea4ed91a37f63877891e0bf82cc5ef0ad019f4a8c1cb77152e6334bd8266.jpg`):
`C4OH` 0.6% (n-butanol), `C5OH` 0.3% (n-pentanol), `3M-C4OH` 0.3% (i-pentanol),
`C6OH` 0.1% (n-hexanol), `2M-3=C4OH` 0.5% (isoprenol isomer 2) and `3M-2=C4OH` 0.5%
(isoprenol isomer 1, prenol), each for BW25113 and BWΔacrAB, n = 2 with SD error bars:
12 values. Table S1 names the same compounds and gives their properties, which is how the
isomer shorthands resolve. Isoprenol itself (`3M-3=C4OH`) is NOT in Figure 5; its
two-strain comparison is Figure 4. Geraniol and farnesol are not measured in this paper
at all; geraniol appears only in the title of SI reference 7 (Shah 2013, MarA
overexpression). No MIC, no survival fraction, no lag time, no titer and no growth rate
is released for any strain: every growth readout in the paper is an end-point OD600 at
12 h, or the growth inhibition derived from it, or the Figure 2 / S2 time courses.

**Isoprenol production: this paper measures none.** The one titer in the text is a
literature value attributed to another paper, not a measurement of Wang's:

> Production of isoprenol has been achieved in the tractable host $E .$ . coli at a titer of $1 . 3 \mathrm { g } / \mathrm { L }$ $( 0 . 1 5 \%$ , v/v) to date18.

Reference 18 is `Zheng, Y. et al. Metabolic engineering of Escherichia coli for
high-specificity production of isoprenol and prenol as next generation of biofuels.
Biotechnol. Biofuels 6, 57, doi: 10.1186/1754-6834-6-57 (2013)`, which is **not in the
673-key literature mirror** (checked by `ls /scratch/projects/torchcell-scratch/torchcell-library | grep -i zheng`;
the only hit is `zhengPyGSSLGraphSelfSupervised2024`). The `pMBO` plasmid in Figure 1A
is a schematic of endogenous isoprenol synthesis, not an experiment: `pMBO indicates the
plasmid for endogenous isoprenol synthesis` is the whole of its appearance, and no
isoprenol was quantified in any culture. Every isoprenol in the experiments is exogenous,
added to the medium. So for the bacterial priority product, **Wang 2015 contributes
tolerance only, and Zheng 2013 is the E. coli titer source we do not have.** The isoprenol
titers we do store are P. putida (`torchcell/datasets/pputida/desiqueira2025.py`,
`carruthers2025.py`), so there is no E. coli isoprenol production record in the project at
all.

**Comparisons against other data.**

- **Rutherford 2010 / GEO GSE16973, a loadable dataset we do not have.** The paper
  re-plots another lab's butanol microarray as its own Figure S1 and compares it with its
  qPCR panel:
  > A microarray analysis was used to study the response of transporters to butanol15. It reported up-regulation of several transporters upon exposure to butanol (Supplementary Fig. S1), but the up-regulated transporters are not identical to the transporters observed under isoprenol exposure (Fig. 3A).

  and the figure caption:
  > The data are adopted from Gene expression Omnibus (Accession No. GSE16973)9.

  Reference 15/9 is `Rutherford, B. J. et al. Functional genomic study of exogenous n-butanol stress in Escherichia coli. Appl. Environ. Microbiol. 76, 1935-1945, doi:10.1128/AEM.02323-09 (2010)`.
  Neither `rutherford` nor `GSE16973` appears in the literature mirror (673 keys) or
  anywhere under `torchcell/` or `notes/` (grepped). This is an E. coli expression
  dataset with a public accession, directly on the solvent-tolerance axis.
  **No duplication risk with Wang:** the Figure S1 numbers are Rutherford's, and Wang
  stores nothing from them.
- **Atsumi 2010 and Minty 2011, isobutanol-tolerance evolution.** The paper checks its
  acrAB result against them:
  > In accordance with our observation, mutations at acrAB locus conferring an improved tolerance have been also identified from isobutanol-adaptive E. coli strains20,21.

  `Atsumi, S. et al. Evolution, genomic analysis, and reconstruction of isobutanol tolerance in Escherichia coli. Mol. Syst. Biol. 6, 449 (2010)` and
  `Minty, J. J. et al. Evolution combined with genomic study elucidates genetic bases of isobutanol tolerance in Escherichia coli. Microb. Cell Fact. 10, 18 (2011)`.
  Neither is in the mirror. The comparison is qualitative (a locus agreeing), not a
  shared measurement, so it is a candidate dataset rather than a duplication risk.
  Hypothesis (untested): both releases carry per-strain isobutanol tolerance for
  reconstructed mutants, which would be loadable as `EnvironmentResponsePhenotype`.
- **Baba 2006** is consumed, not compared: the Keio construction paper supplies the
  background strain and cassette strings (`SOURCED_VALUES["keio_background"]`,
  `["cassette"]`), and it is in the mirror.
- **No duplication risk inside the project.** Wang's 46 strains are Keio deletions of
  multidrug transporters scored by end-point OD600 in 2YT with isoprenol. No other loader
  stores an E. coli isoprenol tolerance; `isoprenol` elsewhere in `torchcell/datasets/`
  appears only under `pputida/` (grepped). The strain set does overlap the Keio-based
  E. coli loaders in gene space, but the condition (0.5% v/v isoprenol) is unique to this
  dataset.
- **One internal cross-check the loader already runs** is worth repeating because it is a
  comparison of the paper against itself: Table S4's transcript columns prove that the
  strains the paper calls BWΔacrA and BWΔacrB are swapped relative to their Table S2 Keio
  numbers (`wang2015.py:464-478`). The paper's own text and Table S3 are internally
  inconsistent on strain count (44 vs 45 vs 46) and on two derived percentages (133.5% vs
  133.07%; 88.3% vs 86.68%), all recorded at note:203-213.

**Loadable-now estimate.**

- No-isoprenol fitness from Table S3's `No isoprenol` column: **46 records** (one per
  deletion strain) plus 1 BW25113 reference. Basis: measured, 47 Table S3 strain rows
  counted from `si1.md`, minus the BW25113 row which becomes the reference, which is the
  same partition the existing dataset uses (note:129, "47 Table S3 rows = BW25113 (the
  reference) + 46 deletion strains"). It would be a `BacterialFitnessExperiment` with
  `FitnessPhenotype.fitness = OD600(strain, no isoprenol) / OD600(BW25113, no isoprenol)`,
  all values positive (range 7.28 to 8.55 over 8.30, measured from the Table S3 cells in
  `si1.md`), `n_samples = 2`, `sample_unit = biological_replicate`. The per-strain SD is
  released for both numerator and denominator, so the same non-derivable-ratio-SD argument
  as the stored phenotype applies and the uncertainty would again be a typed gap. **This
  is already a recorded decision** (note:246-247: it would be "a second phenotype family
  in one dataset"), so the open question is only whether it should be a separate dataset,
  not whether it is loadable. It is the single largest unstored per-record quantity in the
  paper.
- Nothing else is loadable now. Every other unstored quantity is figure-only, a design
  parameter, a compound property, or the Table S4 qPCR panel, which is blocked.

**Not checked.**

- `si1.pdf` itself was not re-parsed by me; I read its OCR `si1.md` and trusted the
  loader's own extraction checks (`TABLE_S3_SHA256`, and
  `test_real_text_layer_agrees_with_the_independent_ocr`, which the note records as
  agreeing on all 47 rows and 188 numbers). My counts come from `si1.md`.
- Figure values were read as labels and bar positions from the OCR-extracted JPEGs, not
  digitized. Every per-strain number I quote comes from a table, never from a bar height.
  The exact Figure 5, Figure 4, Figure 3C, Figure 2 and Figure S1 values are therefore
  "not measured" here, only their structure (which strains, which compounds, which doses,
  which time points).
- Figures S2, S3, S4, S5 and S6 were read from their captions in `si1.md` only; I opened
  the image for Figure S1 but not for those five, so their series counts come from the
  caption text.
- `paper.pdf` was not opened; `paper.md` (sha256 `9cd90f00...`) was read in full.
- I did not query GEO for GSE16973, and did not try to retrieve Rutherford 2010,
  Atsumi 2010, Minty 2011 or Zheng 2013. Whether each has per-record released data is
  unverified.
- I ran no build, no verifier, and no `git` command, per the briefing.

### cuiCRISPRiScreenColi2018 -- `ecoli/cui2018.py`

**SI inventory.** The mirror's `siN` numbering maps onto the publisher's Supplementary
Data numbering with an offset of three, established from `si3.md`'s own "Description of
Additional Supplementary Files".

| file | format | what it is | bytes |
|---|---|---|---|
| `si1.pdf` + `si1.md` | PDF + OCR | Supplementary Information: Supplementary Figures 1-16 and Supplementary Tables 1-9 (Table 7 = strain constructions, Table 1 = the guide-repartition counts) | 1,285,975 / 33,206 |
| `si2.pdf` + `si2.md` | PDF + OCR | Peer-review file (three reviewers' comments plus the authors' rebuttal). Carries the Keio/essentiality exchange and the only statement of where the column descriptions live | 1,483,549 / 69,077 |
| `si3.pdf` + `si3.md` | PDF + OCR | Description of Additional Supplementary Files: the captions of Supplementary Data 1-6 | 54,934 / 662 |
| `si4.csv` | TSV (tab-delimited despite the `.csv` name) | **Supplementary Data 1**: the 64 essential-gene promoter operons. 64 rows, 4 columns `pos`, `genes`, `end`, `ori` | 1,466 |
| `si5.csv` | TSV | **Supplementary Data 2**: the reverse-polar-effect gene list. 106 rows, 11 columns `name`, `left`, `right`, `ori`, `gene_len`, `essential`, `fit75_coding_median`, `nguides_coding`, `fit75_template_median`, `nguides_template`, `rank` | 9,021 |
| `si6.xlsx` | XLSX, one sheet `Sheet1` | **Supplementary Data 3**: the per-seed statistics. 1,022 data rows, 12 columns (an unnamed index plus `seeds`, `N_targets_E18`, `mean_E18`, `pval_E18`, `pval.adj_E18`, `std_E18`, `N_targets_E75`, `mean_E75`, `pval_E75`, `pval.adj_E75`, `std_E75`) | 153,620 |
| `si7.txt` | FASTA | **Supplementary Data 4**: plasmid sequences. 2 records, `>psgRNA` and `>psgRNAc` | 4,898 |
| `si8.csv` | CSV | **Supplementary Data 5**: the screen results. Header read + `wc -l` only, per instruction: 85,382 lines = 85,381 data rows, 10 columns `guide,gene,essential,pos,ori,coding,fit18,fit75,ntargets,seq`. `sha256 95ebaa5a0c92c63849617f48889e2d28b7805fdffdd960527f8a501381143c1e`, byte-identical to the loader's pinned `SCREEN_SHA256` and to the raw mirror's `data/41467_2018_4209_MOESM8_ESM.csv` | 12,080,844 |
| `si9.txt` | text | **Supplementary Data 6**: the modeling split. 6 lines, three comma-separated index lists: 58,935 train, 7,367 validation, 7,367 test (73,669 total) | 430,971 |

Also present and not listed above: `images/` and the `_middle.json` / `_content_list.json`
/ `_ocr_provenance.json` OCR sidecars for `paper`, `si1`, `si2` and `si3`.

**What the loader stores.** `EnvironmentResponsePhenotype` with
`measurement_type=log2_ratio`, wrapped in `BacterialEnvironmentResponseExperiment`, one
record per (retained guide, screened strain) from Supplementary Data 5
(`cui2018.py:1064-1090`, `cui2018.py:1246-1272`). 141,542 records = 70,771 retained guides
x 2 screens (`EXPECTED_RECORDS`, `cui2018.py:234`; ledger in the note at
`### Retention ledger, with the arithmetic`).

**Assignment item 1 resolves to "no gap": BOTH dose regimes are stored.**
`SCREENS: tuple[tuple[str, str], ...] = (("LC-E18", "fit18"), ("LC-E75", "fit75"))`
(`cui2018.py:229`) and `process()` loops `for screen_id, _ in SCREENS` writing a record
per screen (`cui2018.py:1246`). The two are kept apart both by
`EnvironmentResponsePhenotype.screen_id` and by distinct `BacterialStrainBackground`
objects on the reference (`cui2018.py:591-603`), which is the right treatment, because the
paper itself calls them different experiments rather than replicates:

> Strain LC-E75 carrying this fine-tuned Ptet-dCas9 cassette was then used to perform a
> genome-wide dCas9 knockdown screen following the same protocol as the screen previously
> performed with strain LC-E18.

and

> The expression cassette selected in this manner displayed an expression level 2.6-time
> lower than the original strain LC-E18 and was integrated in strain LC-E75.

**There are no column descriptions to quote.** Neither `paper.md`, `si1.md` nor `si3.md`
defines any column of any released data file; `si3.md`'s captions are one sentence per
file ("Description: Screen results."). Reviewer 1 asked for exactly this and the authors
declined to put it in the SI (`si2.md:142-146`):

> 5- To aid reproducibility and interpretation, the large supplementary data tables could
> be more adequately annotated. For example, it is not clear what the column labels in the
> files mean. It would be relevant to deposit raw files and any relevant computer
> processing code.

and the authors' reply:

> We have now made all code and corresponding data tables available as jupyter notebooks
> at the following address, where the content of the tables is also extensively described:
> <https://gitlab.pasteur.fr/dbikard/badSeed_public>

So every column meaning in this release is established from the Methods plus the released
values, which is what the loader's `si_expected` already states.

**Already recorded as not stored.**

- Supplementary Data 3's per-seed `mean` / `SD` / `n` / Bonferroni-corrected p-value:
  `cui2018.py:39-44` ("THE PER-SEED STATISTICS ARE NOT RECORDS, AND ARE NOT LOADED"),
  `si_expected` entry 2 (`cui2018.py:696-700`), and the note's
  `### What the release reports, and the one thing it does not separate` ("Supplementary
  Data 3 is the bad-seed quantification, and it is deliberately NOT loaded").
- Supplementary Data 1 and 2 as analysis gene lists: `si_expected` entry 3
  (`cui2018.py:701-704`).
- Supplementary Data 4 plasmid sequences: `si_expected` entry 4 (`cui2018.py:705-707`).
- Supplementary Data 6 train/validation/test indices: `si_expected` entry 5
  (`cui2018.py:708-710`).
- The `pos` column and the NC_000913.2 vs ASM584v2 coordinate gap: `cui2018.py:57-64` and
  the note's `### Positions are not stored, and the reason is a genome version`.
- DESeq2's per-guide standard error, absent from the release: `UNCERTAINTY_UNRELEASED`
  (`cui2018.py:481-489`) and the three `ProvenanceGap` entries (`cui2018.py:1050-1054`).
- No sequencing-read accession, so the fold changes cannot be recomputed from reads:
  `si_expected` entry 6 (`cui2018.py:711-717`).
- The model's per-guide predictions are not a released column: `cui2018.py:15-28` and
  `BAD_SEED_CONFOUND` (`cui2018.py:526-533`).
- The medium's unstated formulation: `MEDIUM` (`cui2018.py:490-498`) and the note's
  `### Medium: the one value the source does not pin`.

**Released quantities, column by column.** 36 columns across Supplementary Data 1, 2, 3
and 5; 5 stored, 0 loadable now, 0 blocked, 31 not a phenotype.

| released quantity | source file + column | stored | classification | evidence |
|---|---|---|---|---|
| 20-nt guide spacer | si8.csv `guide` | yes | stored on `CrisprConstruct.guide_sequence` | `cui2018.py:1104-1106` |
| target gene symbol | si8.csv `gene` | yes | stored as a resolved b-number plus `DerivedIdentifierMapping(route="gene_symbol")` | `cui2018.py:1112-1120`; measured: 77,953 of 85,381 rows name a gene, over 4,360 distinct symbols |
| log2FC in LC-E18 | si8.csv `fit18` | yes | `EnvironmentResponsePhenotype.environment_response`, `screen_id="LC-E18"` | `cui2018.py:229`, `cui2018.py:1074-1081` |
| log2FC in LC-E75 | si8.csv `fit75` | yes | `EnvironmentResponsePhenotype.environment_response`, `screen_id="LC-E75"` | same |
| perfect-match multiplicity | si8.csv `ntargets` | yes (as structure) | consumed: one knockdown perturbation per distinct targeted gene, and asserted equal to the guide's row count | `cui2018.py:808-814`, `cui2018.py:1093-1122` |
| target-gene essentiality flag | si8.csv `essential` | no | not a phenotype: an imported annotation, not Cui's measurement. Measured: constant per gene (0 of 4,360 genes carry two values), 297 TRUE and 4,063 FALSE genes. The paper names no source for this column anywhere in `paper.md`, `si1.md` or `si3.md`; `si2.md:313` names Ecogene only for a separate comparison | counted in si8.csv by `drop_duplicates("gene")` |
| chromosomal target position | si8.csv `pos` | no | not a phenotype: a design parameter in a stale coordinate system | already recorded, `cui2018.py:57-64` |
| guide target strand | si8.csv `ori` | no | not a phenotype: a design parameter. Values measured: only `+` and `-` | see the retention-claim defect below |
| coding vs template strand of the target gene | si8.csv `coding` | no | not a phenotype: a derived design annotation (`ori` relative to the gene's strand). Measured: 34,634 TRUE, 43,319 FALSE, 7,428 NA | same |
| 60-nt genomic window around the target | si8.csv `seq` | no | not a phenotype: sequence context, a model feature. Measured: empty on 9,056 rows, all multi-target | note, `### The released table's structure is checked, not assumed` |
| promoter-operon coordinates and orientation | si4.csv `pos`, `genes`, `end`, `ori` (4 columns, 64 rows) | no | not a phenotype: an operon annotation used to select a figure subset. No measurement column at all | caption: "List of 64 operons containing essential genes used to study the effect of dCas9 binding in promoter regions." (`si3.md`) |
| gene coordinates, length, orientation, essentiality flag, rank | si5.csv `name`, `left`, `right`, `ori`, `gene_len`, `essential`, `rank` (7 columns, 106 rows) | no | not a phenotype: annotation plus a rank, which the briefing names explicitly as a derived summary | caption: "List of genes used in the reverse polar effect analysis." (`si3.md`) |
| gene-level median log2FC per strand | si5.csv `fit75_coding_median`, `fit75_template_median` | no | not a phenotype: a derived summary of what we already store, and MISNAMED. Measured: both reproduce as the median of the **`fit18`** column over that gene's rows, exactly for 105/106 and 103/106 genes; against `fit75` they reproduce for 0/106 | measured over si8.csv + si5.csv: group si8 by `gene` and `coding`, take the median of each group's `fit18` and of its `fit75`, and compare against si5's two median columns |
| guides per gene per strand | si5.csv `nguides_coding`, `nguides_template` | no | not a phenotype: a count over the rows we already store. Measured: both equal the si8.csv row count for that gene and strand, exactly for 103/106 genes | same script |
| per-seed mean, SD, n, p-value, adjusted p-value, in each strain | si6.xlsx `Sheet1` columns `N_targets_E18`, `mean_E18`, `pval_E18`, `pval.adj_E18`, `std_E18` and the five `_E75` counterparts, plus `seeds` (11 columns, 1,022 rows) | no | not a phenotype, and already recorded: the unit of observation is a five-nucleotide SEQUENCE, so there is no genotype for the record | `cui2018.py:39-44`; sheet read with openpyxl, `max_row=1023` |
| plasmid sequences | si7.txt, 2 FASTA records | no | not a phenotype: a design parameter. Already recorded as the artifact a future `CrisprConstruct.effector_plasmid_ref` would pin | `cui2018.py:705-707` |
| train / validation / test row indices | si9.txt, 3 lists (58,935 / 7,367 / 7,367) | no | not a phenotype: a modeling split over Data 5. Already recorded | `cui2018.py:708-710` |

**Quantities the text names that the release does not carry at all** (so they are not
loadable from anything in the mirror, and the first three are already recorded):

- Per-guide read counts per sample. The Methods give only a per-condition aggregate, "we
  obtained on average 7.5 million and 17 million reads per experimental condition for
  LC-E18 and LC-E75, respectively", and there is no SRA, ENA or GEO accession
  (`si_expected` entry 6). No per-guide count column exists in any released file.
- DESeq2's per-guide `lfcSE`. Already recorded (`cui2018.py:481-489`).
- The neural network's per-guide predicted log2FC, and any bad-seed-corrected per-guide
  value. Already recorded (`cui2018.py:15-28`). The notebook is the only route.
- Guide efficiency or activity predictions as such: none exist. The model predicts log2FC,
  not repression efficiency.
- Position relative to the start codon (Figure 1f, "the effect of the relative distance
  along the gene"). NOT a released column; it is computable from `pos` plus gene
  coordinates. Figure-only.
- Off-target counts per guide (Supplementary Figures 13, 15; "a median of 4 off-targets
  that carry a perfect match of 9 nt or more"). NOT a released column. Figure-only, and
  the median is already carried as `OFF_TARGET_BURDEN` (`cui2018.py:541-548`).
- A gene-level essentiality call from this screen. Explicitly deferred:
  "A detailed analysis of what our screen can teach us about gene essentiality in E.coli
  will be part of a separate study." (`si2.md:313`)
- Per-strain low-throughput measurements with replicate counts, all figure-only with no
  numeric release: RT-qPCR relative target expression (Fig 2c, Supplementary Fig 2),
  flow-cytometry mCherry fluorescence at n = 3 (Supplementary Figs 9, 12), Western-blot
  dCas9 quantification over 3 blots (Supplementary Fig 8d), spot-dilution plating (Fig 2b,
  Fig 4c, Supplementary Fig 7), and the six resequenced bad-seed suppressor genomes
  (Supplementary Table 2, which releases mutation calls, not a phenotype).

**Two defects found in this audit, neither already recorded.**

1. **A retention claim that the built tree contradicts.** `cui2018.py:62-64` says "The
   ``pos``, ``ori``, ``coding`` and ``essential`` cells are kept verbatim in
   ``preprocess/guide_retention.csv``", and the note repeats it at
   `### Positions are not stored, and the reason is a genome version`. MEASURED on the
   built tree: `head -1 /scratch/projects/torchcell-scratch/data/torchcell/crispri_knockdown_cui2018/preprocess/guide_retention.csv`
   prints `guide,n_targets,source_genes,positions,stored_tags,fit18,fit75,drop_reason`.
   Only `pos` survives, as `positions`. `ori`, `coding` and `essential` are written
   nowhere, so three released columns leave no trace in the build at all. The writer is
   `cui2018.py:1274-1290`. Either the sentence or the writer is wrong; the honest fix is
   to add the three columns, because `coding` is the single covariate the paper's own
   analysis turns on.
2. **Supplementary Data 2's `fit75_*` columns hold `fit18` values.** Measured above: both
   `fit75_coding_median` and `fit75_template_median` reproduce as medians of the `fit18`
   (LC-E18) column and not at all as medians of `fit75`. This is a third defect in the
   paper's own bookkeeping, alongside the two the note already records at
   `### Two defects in the paper's own bookkeeping, recorded`. It changes nothing we
   store, and it is worth recording so that nobody later reads Data 2 as an LC-E75
   summary.

**Comparisons against other data.**

- **The Keio collection: asked for, and answered against Ecogene instead.** Reviewer 3
  asked, verbatim: "D. Essential genes. Did some genes considered essential by other
  methods (e.g., Keio collection) not show a fitness defect when targeted by CRISPR?"
  (`si2.md:311`). The rebuttal answers against a different source and releases no table:
  "A detailed analysis of what our screen can teach us about gene essentiality in E.coli
  will be part of a separate study. We indeed find 65 genes in our analysis that do not
  show a fitness defect but are annotated as essential in the Ecogene database
  (<http://www.ecogene.org>)." (`si2.md:313`). The 65 genes are named nowhere, and the
  breakdown (4 toxin-antitoxin, 14 contested, 45 incomplete repression) appears only in
  the rebuttal prose. Nothing loadable, and no duplication risk.
- **Peters 2016 is cited for a different organism.** Reference 8 is "A comprehensive,
  CRISPR-based functional analysis of essential genes in bacteria", and the comparison is
  qualitative and about B. subtilis: "A reverse polar effect was reported in B. subtilis
  where targeting downstream of a gene was seen to block the expression of the upstream
  gene likely through destabilization of the interrupted transcript8." No E. coli
  measurement is compared, so Peters 2016 is not a dataset this paper points us at.
- **The only quantitative benchmark in the paper is internal.** "Supplementary Figure 10.
  Comparision of fitness measurments in strain LC-E18 and LC-E75. Scatter plot of the
  log2FC value (LC-E18 vs LC-E75) for all the guides in the library." (`si1.md:32`) and
  Supplementary Figure 11 on the narrower template-strand distribution in LC-E75. Both
  axes are already stored, so there is no new data behind either figure.
- **Baba 2006, Goodall 2018 and Rousset 2018 are cited nowhere in this paper.** Grep over
  `paper.md`, `si1.md` and `si2.md` finds no `Keio` outside the reviewer exchange, no
  `Baba`, no `Goodall`, and `Rousset` only as the third author's name. The paper's
  30-reference list contains no E. coli screen to deduplicate against.
- **Duplication risk, and the note is now stale on it.** Issue #760 established that
  Rousset 2018's growth screen IS this paper's `fit75` screen at Pearson r = 1.0000 over
  54,326 spacers. That is invisible from this paper's own text, which is presumably why
  the note's `### Relationship to Rousset 2018, the other Bikard-lab E. coli CRISPRi row`
  still asserts the opposite: "Four of Rousset's conditions are environments Cui never
  ran", and files the shared-spacer question as "**Hypothesis (untested):** the two
  libraries share a large fraction of their spacers ... The two guide sets have NOT been
  compared here". #760 compared them. The note section needs a dated correction saying the
  essentiality/growth arm is the same measurement and that Rousset now serves only its
  phage screens. No new double-storage risk is introduced by anything in Cui's own SI.

**Loadable-now estimate.** Zero records. Every unstored released column is a design
parameter (`pos`, `ori`, `seq`, the plasmid sequences), an annotation (`essential`,
`coding`, the Data 1 operon coordinates, Data 2's gene coordinates), a derived summary of
what we already store (Data 2's medians, guide counts and rank; Data 6's split indices),
or a statistic whose unit of observation is a five-nucleotide sequence rather than a
strain (all of Data 3). No new phenotype class or field is needed, so nothing is blocked
either.

**Not checked.**

- `paper.pdf`, `si1.pdf`, `si2.pdf`, `si3.pdf` were not opened as PDFs; the MinerU OCR
  markdown was read instead, which is the project's own convention.
- The figure images under `images/` and `si/images/` were not read; figure-only values were
  established from their legends, not from the panels.
- <https://gitlab.pasteur.fr/dbikard/badSeed_public> was not fetched, so the authors' own
  column descriptions and their per-guide model predictions are unexamined. This is the
  one place a per-guide quantity beyond `fit18` / `fit75` could still exist, and it is a
  code repository rather than a released data file.
- si8.csv was read in full for the counts in the table above (gene, essential, coding and
  seq tallies, and the Data 2 reproduction test), which goes beyond the header-only
  instruction; the header and `wc -l` are reported separately as instructed.
- The LMDB store itself was not opened; stored-field claims come from the loader source
  and from `preprocess/guide_retention.csv`.

### guptaGlobalProteinTurnover2024 -- `ecoli/gupta2024.py`

Nat Commun 15:5890, doi:10.1038/s41467-024-49920-8. Mirror
`$DATA_ROOT/torchcell-library/guptaGlobalProteinTurnover2024/`. Every count below was
computed with `openpyxl` over the mirrored files.

**SI inventory.** The `siN` numbering maps onto the publisher's MOESM numbering, which
the raw manifest's `si_expected` states: MOESM1 = Supplementary Information, MOESM3 = the
Supplementary Data descriptions, MOESM4 = Supplementary Data 1, ..., MOESM11 =
Supplementary Data 8.

- `si1.pdf` (2,980,518 B) + `si1.md` (63,952 B): Supplementary Information. Supplementary
  Figures 1-10 legends, the full monoisotopic-decay derivation, the protease-assignment
  criteria, and the "Percentage of active proteome turnover per unit hour" section. 71
  extracted images under `si/images/si1/`.
- `si2.pdf` (2,918,795 B) + `si2.md` (172,473 B): the **Peer Review File**, not a
  Supplementary Note. Five reviewers' comments plus the authors' replies, which is where
  Supplementary Figures 8, 9 and 10 and the protease volcano plot were introduced. 84
  images under `si/images/si2/`.
- `si3.pdf` (103,265 B) + `si3.md` (755 B): the Description of Additional Supplementary
  Files, i.e. the one-line caption of each of Supplementary Data 1-8.
- `si4.xlsx` (1,446,285 B): **Supplementary Data 1**, "Half-lives for 3262 proteins across
  13 growth conditions". Sheets `TableS1` (3,262 data rows, 72 columns of which 41 are
  used) and a vestigial `TableS3_Old`. This is one of the two files the loader consumes.
- `si5.xlsx` (69,204 B): **Supplementary Data 2**, "Assignment of protein substrates to
  proteases". Sheet `TableS2`, 308 data rows. Plus `TableS3_Old`.
- `si6.xlsx` (69,149 B): **Supplementary Data 3**, "Rapidly degrading proteins". Sheet
  `TableS3`, 286 data rows. Plus `TableS3_Old`.
- `si7.xlsx` (61,817 B): **Supplementary Data 4**, "N terminus residue of proteins". Sheet
  `TableS4`, 809 data rows. Plus `TableS3_Old`.
- `si8.xlsx` (194,537 B): **Supplementary Data 5**, "Relative protein levels". Sheet
  `TableS5`, 2,994 data rows. Plus `TableS3_Old`.
- `si9.xlsx` (149,472 B): **Supplementary Data 6**, "Absolute protein levels". Sheet
  `TableS6`, 2,994 data rows. Plus `TableS3_Old`.
- `si10.xlsx` (810,962 B): **Supplementary Data 7**, "95% Confidence intervals on
  half-lives for 3262 proteins across 13 growth conditions". Sheet `TableS7`, 3,262 data
  rows, 26 interval columns. The loader's second consumed file.
- `si11.xlsx` (16,860 B): **Supplementary Data 8**, "Glossary of all the variables used in
  the derivation of monoisotopic peak decay". Sheet `Sheet1`, 27 rows.
- `si12.pdf` (1,672,595 B) + `si12.md` (2,571 B): the Nature Portfolio Reporting Summary.
  Checkbox form; the OCR is largely unusable, but the Life-sciences design block is
  legible and carries "Replication A in duplicates. These are biological replicates
  measures months apart." and "No data was excluded from analyses."
- `si13.zip` (23,457,000 B): the **Source Data** archive, 13 entries under `Source_Data/`.
  **NOT raw mass spectrometry.** It is per-panel plotting tables:
  `Figure{1..6}_source_data.xlsx`, `FigureS{1,2,6,10}_source(_)data.xlsx`,
  `FigureS{8,9}_sourcedata.csv`, and `Reviewer1_Comment4_sourcedata.xlsx`. The raw spectra
  are at PRIDE PXD042444, which is NOT in the mirror. Extracted to a new empty directory
  (one-off, outside the project tree); nothing was run from inside it.

The `_middle.json`, `_content_list.json` and `_ocr_provenance.json` sidecars exist for
`si1`, `si2`, `si3`, `si12` and are omitted above.

**What the loader stores.** `ProteinTurnoverPhenotype` inside
`ProteinTurnoverExperiment`, 13 records, one per released condition, keyed on MG1655
b-numbers. The consumed column is si4 sheet `TableS1`'s per-condition "Average of
half-lives (hrs)" cell, stored verbatim in `half_life`, with `degradation_rate =
ln(2)/half_life` and a `degradation_rate_se` computed on the rate scale from the two
replicate columns (`torchcell/datasets/ecoli/gupta2024.py:18-46`). si10's intervals are
consumed as the censoring oracle only. Record count and label count are stated in the
note's retention ledger: 13 records, 3,260 protein keys, **33,187 stored per-protein label
values** (`notes/torchcell.datasets.ecoli.gupta2024.md`, "### Retention ledger, with
arithmetic").

**Already recorded as not stored.** Retired here, not re-reported:

- Per-replicate 95% confidence intervals (si10, 26 columns) have no interval field.
  Already recorded in `notes/torchcell.datasets.ecoli.gupta2024.md` "### Did
  ProteinTurnoverPhenotype suffice" item 1 and `gupta2024.py:103-109`. **Issue #753 (1).**
- Per-protein right-censoring flag for the 2,082 stored ceiling cells. Already recorded in
  the note's "### Ceiling cells are kept, and the class cannot say they are censored" and
  `gupta2024.py:48-56`. **Issue #753 (2).**
- `Environment` has no chemostat dilution rate / doubling time. Already recorded in the
  note's "### Genotype and environment". **Issue #753 (3).**
- `DerivedIdentifierRoute` has no UniProt `db_xref` member, so the per-key route lives in
  `preprocess/identifier_route.csv`. Already recorded in the note's "### Identifier route:
  two layers, both measured" and `gupta2024.py:58-76`. **Issue #753 (4).**
- The 26 per-replicate half-life columns are consumed for the SE and not stored as labels;
  the two alternative-isoform rows `sp|P07363-2|CHEA_ECOLI` and `sp|P63284-2|CLPB_ECOLI`
  are dropped. Already recorded in the note's "### Retention ledger, with arithmetic" and
  `gupta2024.py:78-82`.
- The active degradation rate `k_D` is deliberately not stored because `k_total - D` goes
  negative for stable proteins. Already recorded in `gupta2024.py:30-33`.

**What is NOT recorded anywhere.** Supplementary Data 5 and Supplementary Data 6 are
absent from the loader docstring and absent from the dendron note. The only place they
appear is the raw manifest's `si_expected`, which lumps them into a single blanket
dismissal:

> "Supplementary Data 2-6 and 8 (MOESM5-9, MOESM11) -- protease-substrate assignments,
> rapidly degrading proteins, N-terminal residues, relative and absolute protein levels,
> and the model-variable glossary. NOT deposited: no loader reads them, and each is an
> analysis derived from Supplementary Data 1 rather than an independent measurement"

The second half of that sentence is **factually wrong for Supplementary Data 5 and 6**.
si1.md line 616 states their origin:

> "Absolute protein abundances $( \mathsf { P } _ { \mathsf { a } } )$ are calculated
> using label free mass spectrometry (Supplementary Data 6)."

A label-free MS quantification is a separate acquisition from the TMTproC turnover fit, so
Supplementary Data 6 is an independent measurement, not a derivation of Supplementary Data

1. The same holds for Supplementary Data 5, whose values are log2 ratios of protein
levels between chemostats, a quantity no half-life can produce.

**Released quantities.**

| released quantity | source file + column | stored | classification | evidence |
|---|---|---|---|---|
| per-condition mean total half-life, 13 conditions | si4 `TableS1` cols `MinimalMedia_mean` ... `smpBN-lim6_mean`; 33,201 released cells | **yes** | stored | caption "Average of half-lives (hrs) in ..."; per-column non-empty counts 2555/2665/2651/2697/2469/2619/2467/2460/2491/2390/2393/2809/2535, matching Table 1 |
| per-replicate total half-life, 26 columns | si4 `TableS1` cols `*_1` / `*_2`; 61,811 cells | no (consumed for the SE) | not a phenotype: the stored mean plus its SE is the same information at the stored granularity | already recorded, note "### Retention ledger" |
| ceiling flag (trailing `*`) | si4 `TableS1`, 2,084 cells | no | blocked, **#753 (2)** | si4 row 3 "* Protein total half-life was set to ceiling for this dilution rate" |
| per-replicate 95% CI, 26 columns | si10 `TableS7` cols `*_confidence_1` / `*_confidence_2` | no | blocked, **#753 (1)** | si10 row 4 "Total half-life 95% confidence interval in ..." |
| `Undetermined` censoring marker | si10 `TableS7`, same columns | no (consumed as oracle) | blocked, **#753 (2)** | si10 row 3 "*Undetermined = Model fitting resulted in half-lives much greater than dilution rate; total half-life was set to ceiling for that condition" |
| **absolute protein concentration** | **si9 `TableS6` col `Concentration (M)`; 2,994 proteins** | **no** | **loadable now: `ProteinAbundancePhenotype.protein_abundance` in `BacterialProteinAbundanceExperiment`** | si9 row 4 "Molar concentration per unit cell"; si1.md:616 "Absolute protein abundances ... are calculated using label free mass spectrometry (Supplementary Data 6)" |
| **relative protein level, C-lim12 vs P-lim12** | **si8 `TableS5` col `log2 (C-lim12/P-lim12)`; 2,585 values** | **no** | **blocked: new gap, see below** | si8 row 4 "Log 2 fold change of relative protein levels (C-lim / P-lim). Both in chemostat at 12 hr doubling time." |
| **relative protein level, N-lim12 vs P-lim12** | **si8 `TableS5` col `log2 (N-lim12/P-lim12)`; 2,585 values** | **no** | **blocked: new gap, see below** | si8 row 4 "Log 2 fold change of relative protein levels (N-lim / P-lim). Both in chemostat at 12 hr doubling time." |
| protease assignment, 6-category call | si5 `TableS2` col `Protease assignment`; 308 rows | no | not a phenotype: a derived classification of stored half-lives | si5 row 4 "Classification into 6 different categories: ClpP only, Lon only, HslV only, Additive, Redundant, Unexplained" |
| fractional protease contributions, 5 columns | si5 `TableS2` cols `Redundant`, `ClpP`, `Lon`, `HslV`, `Unexplained`; 5 x 308 = 1,540 values | no | not a phenotype: derived fractions computed from stored half-lives by the si1 criteria | si5 row 4 "Fractional contribution of the protease ClpP towards stabilizing a substrate"; criteria at si1.md:363 |
| rapidly-degrading half-lives, 3 conditions | si6 `TableS3` cols `P-lim6_mean` (253), `C-lim6_mean` (267), `N-lim6_mean` (251) | no | not a phenotype: a filtered republication of si4's same three columns | si6 row 4 "Average half-life (in hrs) across replicates for P-lim bacteria doubling at 6 hrs" |
| `Degrading_*` yes/blank call, 3 conditions | si6 `TableS3` cols `Degrading_P6` (184), `Degrading_C6` (153), `Degrading_N6` (235) | no | not a phenotype: a t-test call on the stored half-life; the underlying p-value is not released | si6 row 4 "Is the protein actively degrading in P-lim 6-hr doubling time"; test at `gupta2024.py:36-40` |
| mean half-life across the three limitations | si6 `TableS3` col `PCN6_mean`; 286 values | no | not a phenotype: a mean over values we already store | si6 row 4 "Average half-life (in hrs) of all nutrient limitations at 6-hr doubling time" |
| observed N-terminal residue | si7 `TableS4` col `This study`; 639 values | no | not a phenotype: a per-protein sequence observation (one amino-acid letter), not a quantity attached to a condition | si7 row 4 "Label-free proteomics." |
| literature N-terminal residue | si7 `TableS4` col `Literature`; 231 values | no | not a phenotype: third-party annotation republished | si7 row 4 cites "Link, A.J., Robison, K., and Church, G.M. (1997) ... Electrophoresis 18, 1259-1313" |
| model-variable glossary | si11 `Sheet1` cols `Variable`, `Unit`, `Description`; 27 rows | no | not a phenotype: documentation | si11 row 3 "D \| hr¹ \| Constant \| Dilution rate out of the chemostat" |
| peptide-level M0 signal over time, 2 proteins | si13 `Figure1_source_data.xlsx` sheet `Figure1D`; 8 time points each for OmpF and RpoS | no | not a phenotype: the raw fit input for two example proteins | sheet row 2 "Fgure 1D: Data for ompF" |
| replicate half-lives, C-lim 6 h | si13 `Figure1_source_data.xlsx` sheet `Figure1E`; 2,651 rows, cols `Replicate1`/`Replicate2` | no | not a phenotype: republication of si4 `C-lim6_1`/`C-lim6_2` | first row `yciZ 3.93264216330523 3.60114375725656` equals si4's `C-lim6_1`/`C-lim6_2` for `sp\|A5A614\|YCIZ_ECOLI` |
| `Confidently degrading` flag, 3 conditions | si13 `Figure2_source_data.xlsx` sheets `Figure2A,B,C-{C,P,N}-LIM`; 374 / 316 / 892 `yes` of 2,302 / 2,097 / 2,059 rows | no | not a phenotype: the same derived call as si6's `Degrading_*`, extended to all detected proteins | sheet col header `Confidently degrading` |
| `Subcellular location` | si13 `Figure2`/`Figure4` sheets | no | not a phenotype: annotation | sheet col header `Subcellular location` |
| per-protein active degradation rate by location | si13 `Figure2_source_data.xlsx` sheet `Figure 2E` (2,073 rows), `Figure3_source_data.xlsx` sheet `Figure 3B` (1,332 rows), cols `Degradation Rate` | no | not a phenotype: `k_D` derived from the stored rate minus the known dilution, figure-only | col headers `Subcellular Location`, `Degradation Rate`, `Experiment`/`Strain` |
| protease substrate counts | si13 `Figure3` sheet `Figure 3E`; 6 rows | no | not a phenotype: counts over si5's assignment | col header "Y axis for the bar plot " |
| active degradation % of proteome per hour, 5 strains | si13 `Figure3` sheet `Figure 3G`; 5 rows | no | not a phenotype: a bulk summary computed from stored half-lives weighted by si9 abundances and molecular weight | col header "Active degradation %of proteome per hour"; formula at si1.md:616-618 |
| module CV and randomized CV | si13 `Figure5_source_data.xlsx` sheet `Figure5A`; 657,000 rows across complexes / operons / pathways | no | not a phenotype: a dispersion statistic over stored half-lives plus a permutation null | col headers `Protein-Complexes`, `CV`, `Gene groups`, `Randomized CVs` |
| `Gene groups` membership | same sheet | no | not a phenotype: complex / operon / pathway annotation | same |
| **OD600 growth curve on thymidine as sole N source** | **si13 `Figure5_source_data.xlsx` sheet `Figure 5G`; 48 rows, cols `Strain`, `Culture.ID`, `Time_hr`, `OD_600`** | **no** | **blocked: new gap, see below** | paper Fig. 5 caption panel D "The wild-type and TKO strain were grown on glucose minimal media with thymidine as the sole nitrogen source. The mutant strain grows to a higher maximum OD600 than the wild-type strain." |
| **OD600 growth curve, starvation to LB upshift** | **si13 `FigureS9_sourcedata.csv`; 162 rows, cols `Culture.ID`, `Strain`, `Limitation`, `Time_hr`, `OD600`** | **no** | **blocked: new gap, see below** | si1.md:41 "We measured growth curves following nutrient upshift from starvation in minimal media to LB media for both the wild-type strain and the mutant lacking lon, clpP, and hslV. We find that the triple knock-out (TKO, brown) is less able to adapt to the new conditions and has a significant growth delay $( \sim 8 0$ minutes) compared to the wild-type (blue) in nitrogen limitation." |
| `kd_active`, 38 genes x 4 conditions | si13 `FigureS8_sourcedata.csv`; 152 rows, cols `Gene`, `name`, `kd_active` | no | not a phenotype: the `k_D` the loader declines to store, released only for an FDR-0.01 selected subset and capped at 1 | si1.md:38 Fig S8 legend "Text values on each tile are the active degradation rate in units of hr-1 and are capped at 1. Proteins were excluded if they did not have a total half-life significantly less than the dilution-only half-life in at least one condition" |
| chloramphenicol-assay normalized abundance profiles | si13 `FigureS1_source_data.xlsx` sheets `Fig S1B` (10 time points), `Fig S1C` (101 fitted points) | no | not a phenotype: fitted curves from a control assay the paper runs to argue against translation inhibition | si1.md:2 "Using translation inhibition to measure protein turnover might lead to unintended side effects" |
| M0 relative-level clusters over time | si13 `FigureS6_sourcedata.xlsx`; 17 rows x 8 time points | no | not a phenotype: k-means cluster centroids | si1.md:31 "Clusters of $\mathbf { M } _ { 0 }$ relative levels plotted with time. Clustering is done using k means algorithm." |
| half-life by physicochemical category, 3 conditions | si13 `FigureS2_source_data.xlsx` sheets `FigureS2A/B/C`; 2,651 rows each | no | not a phenotype: stored half-lives tagged with molecular weight / isoelectric point / charge bins | col headers `C-lim`, `Category` |
| smpB vs WT half-life pairs | si13 `FigureS10_sourcedata.xlsx`; 1,983 rows | no | not a phenotype: republication of si4's `smpBN-lim6_mean` and `N-lim6_mean` | col headers `Half life hours smpB mutant`, `Half life hours N-lim Wild type` |
| absolute concentration, superset | si13 `Reviewer1_Comment4_sourcedata.xlsx` sheet `All proteins in grey` col `Concentration (M)`; 3,048 proteins | no | same quantity as si9, 54 proteins more (see duplication below) | measured: 2,994 shared ids, 0 value mismatches, 0 ids only in si9 |
| `Log N-lim/P-lim`, superset | same sheet; 2,639 values | no | same quantity as si8's `log2 (N-lim12/P-lim12)`, 54 values more | measured: 2,585 shared, 0 mismatches |
| `Confidence (Y-axis)` differential p-value | same sheet; 2,639 values | no | blocked, and floored: range -7.0 to -0.301272460255 with 836 values exactly -7.0 | the statistic is never named in the release; si2.md:204 only says "We compared the expression of proteases in nitrogen and phosphorus limited conditions using a volcano plot" |
| 74-protease subset table | same workbook, sheet `Reviewer1_Comment5_sourcedata`; 73 rows | no | not a phenotype: a named subset of the volcano, with a protease list taken from a third-party web page | si2.md:204 "We compiled the list of 74 proteases in E.coli obtained from the website" |
| steady-state chemostat OD600, 3 conditions | paper.md:163 Methods, prose only | no | not a phenotype: a culture process parameter, given to two significant figures with a tilde and not attached to any strain-gene record | "The $\mathrm { \mathbf { O D _ { 6 0 0 } } }$ for the chemostats are as follows: P-lim $\mathrm { O D } \sim 0 . 7 1 $ , N-lim OD \~0.51, C-lim OD $\mathord { \sim } 0 . 6 8$ ." |
| doubling time per condition | paper.md:44 Table 1 col `Doubling time` | no (named on `measurement_type`) | blocked, **#753 (3)** | Table 1 "Doubling time: 42 min / 3h / 6h / 12 h" |
| replicate count per condition | paper.md:44 Table 1 col `Replicates`; 2 for all 13 | yes (as n = 2 in the SE) | stored | Table 1 col `Replicates` = 2 on every row |
| protein count per condition | paper.md:44 Table 1 col `# of proteins` | yes (as the L1 oracle) | not a phenotype: a coverage count | note "### Retention ledger", Table 1 matches 13 of 13 |

Counts: **38 released quantities enumerated**; 3 stored (the mean half-life, the replicate
count as n = 2, the per-condition protein count as an oracle); **1 loadable now**; 9
blocked (5 of them already carried by #753, 4 new); 25 not a phenotype.

#### The new gaps

**1. A between-environment per-protein abundance RATIO has no home.**
`ProteinAbundancePhenotype`'s own docstring forbids it:

> "``protein_abundance`` maps each measured protein's systematic ORF id to its abundance
> (absolute per-strain quantity on a log signal scale, NOT a ratio -- the WT/parent strain
> supplies the reference)."

(`torchcell/datamodels/schema.py:4536-4538`.) si8's two columns are ratios between two
environments of the SAME unperturbed strain, so the reference is not a parent strain and
the number is not an absolute level. `EnvironmentResponsePhenotype` does not substitute:
it is `graph_level = "global"` with a single `environment_response: float | None`
(`schema.py:4745,4759`), i.e. one growth-response scalar per record, not a per-protein
dict. Exact missing capability: either a `ProteinAbundancePhenotype` variant that admits a
log2 between-condition ratio with a declared reference environment, or an
`environment_reference` on the abundance phenotype. Not named by #749, #753, #756, #758 or
#760.

**2. No class holds an OD600 readout or a growth curve.** `AssayType.liquid_od_growth`
exists (`schema.py:4605`, "``liquid_od_growth``: liquid-culture optical-density growth
curve / MIC"), so the HOW axis anticipates this assay, but `MeasurementType`
(`schema.py:4619-4665`) has members `log2_ratio, z_score, sensitivity_score, categorical,
ordinal, growth_rate, differential_fitness, control_regression_residual, colony_size` and
no member for an optical density, a maximum OD, or a lag / growth delay. The only `od600`
fields in the schema are design inputs (`CultureFormat.inoculum_od600:3248`,
`PreCulture.od600_at_transfer:3304`), not readouts. Both released growth tables are
therefore blocked. Also not named by any of the five issues.

**3. Provenance, not schema: si9's condition and replicate count are unsourced.** Combed
thoroughly per the SI-sourcing rule: grepped `paper.md`, `si1.md`, `si2.md`, `si3.md`,
`si12.md` for `absolute`, `abundance`, `molar`, `concentration`, `copies`, `iBAQ`, `per
cell`, `cell volume`, `label free`, `triplicate`, `duplicate`. The release says exactly
two things about Supplementary Data 6: the caption "Molar concentration per unit cell" and
si1.md:616. It never states which of the 13 conditions the label-free run was, which
strain, how many replicates, or the recipe that turns an MS signal into molarity. The
back-solve route of the sourcing rule is precluded: no companion statistic for the
abundance is released. The deferral route is also unavailable, because the released method
sentence cites no earlier paper. The usable evidence is indirect and I label it as such.
Hypothesis (untested): the condition is N-lim at a 12 h doubling, because
`Reviewer1_Comment4_sourcedata.xlsx` pairs the same `Concentration (M)` column with
`Log N-lim/P-lim` on one axis set, and because si1.md:616-618 applies the single `Pa`
vector to all five N-lim strains of Fig 3G. That is a plausible reading, not a sourced
fact, and it must NOT be written into a record as if sourced.

#### Comparisons against other data

- Protease-trap pull-downs, three of them. Verbatim (paper.md:70):

  > "To validate our classifications, we compared our proteasesubstrate relationships with
  > previous proteome-wide measurements. We see a significant overlap $\scriptstyle ( p
  > \cdot \lor \mathsf { a l u e } = 6 \mathsf { E } - 9 )$ of our identified ClpP
  > substrates with substrates identified via a trap mutant (Fig. 3F)20. However, we do
  > not observe an overlap of our putative Lon substrates with a previous Lon-trap
  > experiment $( p \mathrm { - v a l u e } = 0 . 2 2 ) ^ { 2 1 }$ ."

  Refs 21 and 63 are Arends 2018 (Lon) and Arends 2016 (FtsH, "In vivo trapping of FtsH
  substrates by label-free quantitative proteomics"). A trap pull-down releases a SUBSTRATE
  LIST, not a `genotype x environment -> phenotype` record, so none of the three is a
  dataset this schema could load. No duplication risk.
- N-terminal residues against Edman degradation. Verbatim (paper.md:93-94):

  > "Encouragingly, when we compared a small subset of the identified N-termini with
  > previous data in literature obtained using Edman Degradation73we found nearly perfect
  > agreement (55/61 proteins)"

  si7's `Literature` column (231 values) is that third-party data republished under this
  paper's key. Do not load it; it belongs to Link, Robison and Church 1997 if it is ever
  wanted.
- **No comparison against any prior E. coli turnover or abundance compendium.** Grepped
  `paper.md` for `Schmidt`, `compendium`, `previously published`, `published data`: no hit
  of that kind. So the stored turnover labels carry no external duplication risk, unlike
  Rousset / Cui (#760).

**Internal duplication risk, measured.** Three of the mirrored files republish numbers
that are already stored or that would be stored, and two of them disagree with the
deposited version:

1. si6 (`TableS3`) and si13's `Figure2`, `Figure1E`, `FigureS10`, `Figure 5F`, `Figure6C`
   sheets all republish si4 columns. Measured on the Figure 2 sheets against si4's
   `{C,P,N}-lim6_mean`: 2,302 / 2,097 / 2,059 rows compared, **6 / 1 / 1 disagreements**,
   and every one of the 8 is **exactly 1.000 h**, always on a protein whose condition has
   one `8.000*` ceiling replicate (e.g. `sp|Q46893|ISPD_ECOLI` P-lim6 = `('8.000*',
   6.299952397)`, si4 mean 7.1499761985, Source Data 6.14997619845974). So the Source Data
   figure tables were computed with the ceiling set to the DOUBLING TIME, which is the
   convention si1.md:617 states ("The proteins with half-life greater than the doubling
   time are called stable and assigned a half-life equal to the doubling time $( T _ { 1 /
   2 , \mathsf { c a p } } )$"), while Supplementary Data 1 caps at 8 h for a 6 h doubling.
   **Two ceiling conventions coexist in one release.** The loader consumed si4, so it holds
   the 8 h convention and is internally consistent; the Source Data tables are a stale
   version and must never be loaded.
2. `Reviewer1_Comment4_sourcedata.xlsx` sheet `All proteins in grey` is a strict SUPERSET
   of si9 and of si8's N-lim column: 3,048 proteins with `Concentration (M)` containing
   all 2,994 of si9 with **0 value mismatches and 0 ids only in si9**, and 2,639 values of
   `Log N-lim/P-lim` containing all 2,585 of si8 with **0 mismatches**. If abundance is
   loaded, this is a deliberate choice between the numbered Supplementary Data file (2,994
   rows, cited by the SI text) and a larger table that exists only inside `si13.zip` under
   a reviewer-comment filename. The numbered file is the citable one; the extra 54 rows are
   not worth sourcing from a filename that names no Supplementary Data number.
3. `TableS3_Old` appears as a second sheet in si4, si5, si6, si7, si8 and si9, always 305
   rows, always holding the same stale column-description list for `TableS1`
   (`Protein ID`, `Gene names`, `Average_hl_P6`, ...). It is a vestigial sheet, not data.

#### Loadable-now estimate

**si9 absolute protein concentration, as `BacterialProteinAbundanceExperiment` holding
`ProteinAbundancePhenotype`.** No schema change: `protein_abundance: dict[str, float]`
takes the molar concentration keyed on the b-number the existing two-layer identifier
route already resolves, `measurement_type` says what the number is (the class's own
docstring: "``measurement_type`` records WHAT the number is so heterogeneous proteomics
assays are never silently mixed"), and `BacterialProteinAbundanceExperimentReference`
carries the same MG1655 `AssemblyReferenceGenome` pin the turnover records use
(`schema.py:5543-5556`).

- **Records added: 1.** One experiment record, because the release gives one concentration
  column, not one per condition. Basis: si9 `TableS6` has exactly 4 populated columns and
  one of them is a value column.
- **Per-protein label values added: 2,994.** Measured: every one of the 2,994 rows with a
  `Protein ID` has a non-empty `Concentration (M)`. Of those, 2,994 would survive the
  identifier route only up to the same two isoform drops the turnover records take;
  I did NOT re-run the reconciliation against si9's id list, so treat 2,994 as the ceiling
  and expect 2,992 to 2,994 keys.
- **Two things must be resolved before a load, and neither is a schema change.**
  `ProteinAbundancePhenotype.n_replicates` is a required `dict[str, int]` whose keys must
  match `protein_abundance` and whose values must be `>= 1` (`schema.py:4552,4564-4568`),
  and the release states no replicate count for the label-free run. Under the sourcing
  rule's resolution order the back-solve is precluded, so the conservative lower end
  applies: `n_replicates = 1` for every key, with a `ProvenanceGap` on `n_samples`
  recording that the release is silent. The `Environment` the record needs is the harder
  one: see new gap 3 above. A load that names a condition without sourcing it would be a
  guess, so the honest form is an `Environment` carrying a `ProvenanceGap` on the medium,
  or the loader waits until the Zenodo analysis code
  (<https://doi.org/10.5281/zenodo.10895828>, not in the mirror) is retrieved and mirrored.

No other item classifies as loadable now.

#### Not checked

- `si1.pdf`, `si2.pdf`, `si3.pdf`, `si12.pdf` were read only through their OCR markdown,
  not the PDFs. `si12.md` in particular is badly mangled (checkbox form), so anything the
  Reporting Summary states beyond the two sentences quoted above is not checked.
- `si4.xlsx`-`si11.xlsx` cell FORMATTING was not inspected, only values (`data_only=True`).
  A censoring convention encoded as a fill color rather than a trailing `*` would have been
  missed. There is no evidence of one; si4 and si10 both use text markers.
- PRIDE PXD042444, the raw mass spectrometry, is not in the mirror and was not fetched.
  The release's own `si13.zip` is NOT that data; it is per-figure plotting tables.
- The Zenodo analysis code (10.5281/zenodo.10895828) is not in the mirror and was not
  fetched. It is the only place the molarity conversion and the label-free run's condition
  could still be sourced from.
- `Figure5_source_data.xlsx` sheet `Figure5A` has 657,002 rows; I read its headers and
  first rows and counted nothing within it, because it is a permutation null.
- I did not run the loader, any build, any slurm job, or any `git` command.

### rappMetabolomeColiCRISPRi2026 -- `ecoli/rapp2026.py`

Rapp et al. 2026, *Cell Systems*, doi:10.1016/j.cels.2025.101518. Mirror:
`/scratch/projects/torchcell-scratch/torchcell-library/rappMetabolomeColiCRISPRi2026/`.

**SI inventory.** 17 publisher files (`mmc1`-`mmc17`); every `siN` maps to `mmc N`, and
every workbook holds exactly one data sheet named `Table S(N-1)`.

| file | format | bytes | what it is |
|---|---|---|---|
| `si1.pdf` + `si1.md` | PDF + OCR | 1,246,521 / 25,084 | Supplemental information: Figure captions S1-S12 plus the full text + results table of **Data S3** (MurQ molecular docking / MD). No supplementary-table captions are in it; the Tables are captioned only by their own column legends. |
| `si2.xlsx` | xlsx, sheet `Table_S1` | 107,114 | 1,515 rows. Cols: `Gene`, `sgRNA Nr.`, `b-Nr.`, `base pairing region`, `Oligo sequence`. The library's guide design. **Consumed.** |
| `si3.xlsx` | xlsx, sheet `Table_S2` | 7,390,438 | 4,593 rows x 187 cols (header on row 2). Cols: `Replicate Nr.`, `RXN Nr.`, `Gene`, `Guide Nr.`, `b-Nr.`, `Plate ID`, then **181 OD600 time columns `0` .. `30`** (10 min spacing, 0-30 h). The growth curves. |
| `si4.xlsx` | xlsx, sheet `Table_S3` | 199,690 | 3,026 rows. Cols: `Target gene`, `b number`, `OD`, `Plate ID`, `Well`, `Replicate`, `Sample ID`. **Consumed** (OD not stored on records). |
| `si5.xlsx` | xlsx, sheet `Table_S4` | 43,827,170 | 1,880 feature rows x 3,030 cols. Cols: `Abbr`, `Metabolite`, `Mass`, `Kegg`, then **3,026 sample columns** (`aaeA_R1_msAV932_B1` ...). The FI-MS fold-change matrix. **Consumed: this is what we store.** |
| `si6.xlsx` | xlsx, sheets `TableS5` + `Legend` | 318,926 | 1,385 rows x 29 cols. `Gene`, `Metabolite`, `Metabolite Abbreviation`, `Kegg ID`, `ECMDB`, `CAS`, `Polarity`, `Mode`, `Mass`, `MonoMass`, `Mean_FC`, `R1_FC`, `R2_FC`, `Mean_Int`, `R1_Int`, `R2_Int`, `LC-MS/MS`, `Reactant`, `Reactant Abbreviation`, `Subsystem`, `FBA`, `Substrate Abb`, `Product Abb`, `Operon effect`, `Pathways`, `PathwaysAbb`, `Position`, `Upstream Accumulation`, `Downstream Accumulation`. **Consumed for the `Mean_FC` cross-check only.** |
| `si7.xlsx` | xlsx, sheets `Table_S6` + `Legend` | 590,835 | 1,256 rows x 54 cols. `QC passed`, `Gene`, `Abbreviation`, `metName`, `Kegg`, `IDsmiles`, `CAS`, `SMILES`, `ExpSpectrum`, `ExpSpecMerged`, `Reactant`, `ReactantAbb`, `SubstrateAbb`, `ProductAbb`, `Operon effect`, `Polarity`, `Mode`, `PrecMz`, `Monoisotopic Mass`, `Sum formula`, `retention time (sec)`, `fold-change`, `Intensity PrecMz`, `Deviation (Da)`, `Scan # MS1`, `Scan # CE10/20/40`, and per-collision-energy MS2 fragment m/z, intensity, normalized intensity, matching predicted fragment, carbon counts, SMILES, `# predicted fragments`, `total # of fragments`, `At least one predicted fragment`, `DataFile`. The targeted LC-MS/MS screen. |
| `si8.xlsx` | xlsx, sheet `Table_S7` | 948,938 | 9,462 rows x 10 cols. `Gene`, `Metabolite`, `Mass`, `Mode`, `Mean_FC`, `R1_FC`, `R2_FC`, `Mean_Int`, `R1_Int`, `R2_Int`. Every accumulating m/z feature, annotated and not. |
| `si9.xlsx` | xlsx, sheet `Table_S8` | 26,200 | 51 data rows x 27 cols (header on row 3, title "Non-annotated iML1515 m/z-features"). SIRIUS/CSI:FingerID structure predictions: `structurePerIdRank`, `formulaRank`, `ConfidenceScoreExact`, `ConfidenceScoreApproximate`, `CSI:FingerIDScore`, `ZodiacScore`, `SiriusScore`, `molecularFormula`, `adduct`, `precursorFormula`, `InChIkey2D`, `InChI`, `name`, `smiles`, `xlogp`, `pubchemids`, `links`, `dbflags`, `ionMass`, `retentionTimeInSeconds`, `retentionTimeInMinutes`, `formulaId`, `alignedFeatureId`, `mappingFeatureId`, `overallFeatureQuality`, `Selected`. |
| `si10.xlsx` | xlsx, sheet `TableS9` | 70,584 | 802 rows. `Abbreviation`, `BIGG`, `Metabolite`, `KEGG`, `Monoisotopic mass`, `Neutral Formula`. **Consumed: the metabolite-identity layer.** |
| `si11.xlsx` | xlsx, sheet `TableS10` | 464,867 | 7,683 rows. `gene`, `Gene Number`, `Abbreviation`, `Metabolite`, `Subsystem`, `KEGG`, `BIGG`, `Monoisotopic Mass`, `Neutral Formula`. iML1515 gene-reactant map. |
| `si12.xlsx` | xlsx, sheet `Table_S11` | 228,386 | 3,340 rows. `Position`, `Abb`, `Pathway`, `GeneID`, `GeneAccession`, `GeneName`, `ReactionId`, `ReactionEC`, `EnzymaticActivity`, `Evidence`, `Sub- or superpathway`. EcoCyc pathway membership. |
| `si13.xlsx` | xlsx, sheet `Table_S12` | 1,081,379 | 11,851 rows. The `Table_S11` columns plus `metAbb`, `metAbNames`, `subSys`, `KEGGID`, `BIGG`, `mass`, `mass_13C`, `NeutralFormula`. EcoCyc pathways with their reactants. |
| `si14.pdf` + `si14.md` | PDF + OCR | 18,676,134 / 5,815 | **Data S1**: replicate parity plots, one panel per strain. Caption: "Log2 fold-changes of all annotated m/z-features in the Fl-MS screen (positive and negative ionization mode). X-axes show log2 foldchange in replicate 1. Y-axes show log2 fold-change of replicate 2. Shown are parity plots of 1498 CRISPRi strains and 15 control strains." Each panel prints that strain's MRE (e.g. "yqhD ... MRE = 0.14"). Figures only, no tabular data. |
| `si15.zip` | zip | 62,188 | **Data S2**: one file, `DataS2.mgf` (733,915 bytes uncompressed), the putative MS2 reference spectra. Verified with `unzip -l`; extracted to a separate empty scratch dir only for the listing. |
| `si16.pdf` + `si16.md` | PDF + OCR | 1,701,137 / 68,099 | **The transparent peer review record**, not data: "Initial Submission: Received Aug 12, 2024", the two editorial decision letters, three reviewers' comments and the authors' point-by-point response. It contains no data table and no released quantity. Grep for `proteom`, `transcriptom`, `growth rate`, `growth defect`, `Fuhrer`, `Donati`, `Keio`, `doubling`, `correlat`, `Table S2`, `Table S3` returns 0 hits each. |
| `si17.pdf` + `si17.md` | PDF + OCR | 14,925,355 / 120,325 | **A duplicate of the article**, the corrected online version with `si1`'s supplemental information appended. `diff paper.md si/si17.md` differs only in image hashes, three whitespace-level table/method lines, and the appended `# Supplemental information` block that is verbatim `si1.md`. Not a supplementary methods document and not data. |

Sidecars exist beside each OCR'd PDF (`siN_middle.json`, `siN_content_list.json`,
`siN_ocr_provenance.json`) and extracted images in `si/images/si1` (14), `si/images/si14`
(63), `si/images/si17` (24).

**What the loader stores.** `BacterialMetaboliteExperiment` /
`BacterialMetaboliteExperimentReference` carrying a `MetabolitePhenotype`
(`torchcell/datasets/ecoli/rapp2026.py:50-66`). `metabolite_level` is Table S4's
**linear fold change relative to the per-batch median**, keyed by Table S4's `Abbr`
verbatim plus the adduct (`frdp[M-H]-`), as the arithmetic mean of the strain's two
plates; `metabolite_level_se` is `|r1 - r2| / 2`; `n_replicates` is 2 for every key.
Measured from the built dev store: **1,496 records**
(`preprocess/strains.csv`, 1,496 data rows) x **1,321 feature keys**
(`preprocess/metabolites.csv`, 1,321 data rows). The stored mean reproduces the paper's
own `Mean_FC` exactly: `preprocess/mean_fc_crosscheck.json` reports
`"n_pairs": 1385, "n_checked": 1385, "max_abs_difference": 0.0`.

**Already recorded as not stored.**

- The eight unread SI workbooks and the four SI PDFs are already listed as deliberately
  not mirrored, naming Table S2 explicitly: `rapp2026.py:275-287` (`NOT_MIRRORED`),
  `"mmc3.xlsx (Table S2 growth curves), mmc7.xlsx (Table S6 LC-MS/MS spectra), mmc8.xlsx
  (Table S7 non-annotated features), mmc9.xlsx (Table S8 SIRIUS predictions), mmc11.xlsx
  (Table S10 iML1515 reactants), mmc12.xlsx (Table S11 EcoCyc pathways), mmc13.xlsx
  (Table S12 pathway reactants), mmc15.zip (Data S2 reference spectra): not consumed"`,
  repeated in the note under `### Raw mirror`. That records non-consumption of the FILES;
  it does not classify the phenotypes inside them, which is what this audit adds.
- The three MassIVE deposits (MSV000098712 / MSV000098755 / MSV000098714) are recorded as
  raw spectra rather than released matrices: `rapp2026.py:276-278` and the note's
  `data_availability` row.
- The dataset-level replicate-noise statistic (1.7 log2 units) is recorded as not stored
  on any record: `rapp2026.py:556-559`, `"a dataset-level reproducibility statistic over
  all strains and features, not a per-record uncertainty; it is not stored on any
  record"`.
- The 559 all-empty Table S4 feature rows are recorded: `rapp2026.py:548-553`, `"of 1,880
  rows, 1,321 are finite in all 3,026 columns and 559 are empty in all of them, with no
  partial row"`.
- The two dropped records (`argR` with no sgRNA, `phnE` `b4104`) and the missing
  `DerivedIdentifierRoute` member are recorded: note `### Retention ledger` and
  `### Finding for the owner: a missing DerivedIdentifierRoute member`.
- Keys whose abbreviation is a merged isobaric set carry no `target_metabolite_ids`
  entry, recorded in `preprocess/metabolite_identity.json`: `rapp2026.py:56-63`.

**Released per-record quantities.**

| released quantity | source file + column | stored | classification | evidence |
|---|---|---|---|---|
| sgRNA number, b-number, 20-nt spacer, full oligo | `si2.xlsx` `Table_S1` `sgRNA Nr.` / `b-Nr.` / `base pairing region` / `Oligo sequence` | spacer + b-number on the perturbation | not a phenotype: design parameter | 1,515 rows read with openpyxl |
| OD600 growth curve, 181 points 0-30 h, per strain per replicate (n=3) | `si3.xlsx` `Table_S2` columns `0` .. `30` | no | **blocked**: no time-series or growth-curve phenotype. `grep "time_series\|timepoint\|growth_curve\|curve\|auc\|area_under" schema.py` hits only the `liquid_od_growth` docstring; no `Phenotype` subclass has a time axis. Not named by #749, #753, #756, #758 or #760. | Figure S1 caption (`si1.md`): "Growth curves of all strains in the library were determined in plate readers. dCas9 expression was induced at $\mathfrak{t}=0$ h by addition of aTc. Growth curves show means from $\mathsf{n}{=}3$ cultures cultivated in minimal glucose medium in a plate reader." Methods: "${\tt OD}_{600}$ was measured every 10 min for $^{24\mathrm{~h~}}$ at ${}^{37}\^{\circ}\mathrm{C}$ and 800 rpm. Data was analyzed with custom MATLAB scripts to obtain the area under the curve using the trapz.m function." |
| AUC (trapz) per strain, and the growth-defect call at AUC < 18 | NOT a released column; defined in the Figure S1 caption and Methods, and asserted in Results | no | **loadable now, but only as a derivation we compute**: `FitnessPhenotype.fitness` (an AUC or mu_max ratio against the 16 control strains, which Table S2 carries as `Guide Nr.` `ctrl1`..`ctrl16`), or `EnvironmentResponsePhenotype` with `MeasurementType.growth_rate` + `AssayType.liquid_od_growth`. The latter class is documented for a response to an ENVIRONMENTAL perturbation, which this is not, so `FitnessPhenotype` + `BacterialFitnessExperiment` is the honest home. | Figure S1: "The area under the curve (AUC) was used to classify CRISPRi strains with and without growth defect (AUC cutoff of 18)." Results: "Of the 1,515 CRISPRi strains, 489 CRISPRi strains showed growth defects in minimal glucose medium, whereas 1,026 CRISPRi strains showed no tangible growth defect (Figure S1; Table S2)." MEASURED by me from `si3.xlsx`: trapezoid AUC over the 181 columns, averaged over the 3 replicates, gives **490 strains below 18 and 1,025 at or above**, and taking the AUC of the mean curve gives the same 490/1,025 (AUC range 0.33 to 26.26). So the paper's statistic reproduces from the released curves to within one strain. |
| sampling OD600 at 6.5 h, per sample | `si4.xlsx` `Table_S3` column `OD`, 3,026 rows (range 0.2052 to 1.1893, mean 0.7135) | no, on the record; it travels only as `preprocess/strains.csv` column `optical_density` (`rapp2026.py:2042-2044`, semicolon-joined per replicate) | **blocked**: nothing holds a single-timepoint absolute OD. `MeasurementType` has no `optical_density` member (members are `log2_ratio`, `z_score`, `sensitivity_score`, `categorical`, `ordinal`, `growth_rate`, `differential_fitness`, `control_regression_residual`, `colony_size`), and `FitnessPhenotype.fitness` wants a ko/wt ratio. Same shape as #753's censoring flag (released number reachable only through a preprocess sidecar) but #753 does not name an OD, so this is a separate gap. Caveat: this OD is also a QC/normalization covariate rather than the paper's reported phenotype. | Results: "The remaining 1,498 strains had sampling ODs between 0.2 and 1.2, and sampling ODs were consistent across the two replicates (Figure S2; Table S3)." Figure S2: "89 strains have more than $20\%$ error between replicates and are shown in orange." |
| Plate ID, Well, Replicate, Sample ID, `RXN Nr.`, `Guide Nr.` | `si4.xlsx` `Table_S3`; `si3.xlsx` `Table_S2` | plate / well / batch in `preprocess/strains.csv` | not a phenotype: design and annotation | column headers |
| FI-MS linear fold change vs the per-batch median, per feature per sample | `si5.xlsx` `Table_S4`, 1,880 feature rows x 3,026 sample columns | **yes** | stored: `MetabolitePhenotype.metabolite_level` / `metabolite_level_se` | `rapp2026.py:50-66`; Methods: "The intensity of the annotated $m/z$ features was used to calculate fold changes relative to the median on a per batch basis (Table S4)." |
| `Mean_FC` / `R1_FC` / `R2_FC` for the 1,385 accumulating pairs | `si6.xlsx` `TableS5` | **yes**, the same numbers | already stored: do not store twice. `max_abs_difference 0.0` over all 1,385 pairs. | `preprocess/mean_fc_crosscheck.json` |
| `Mean_Int` / `R1_Int` / `R2_Int`: absolute FI-MS intensity of the annotated feature | `si6.xlsx` `TableS5` (1,385 rows, 411 genes, 319 metabolite abbreviations) | no | **loadable now**: a second `MetabolitePhenotype` record per strain with `measurement_type='fi_ms_absolute_intensity'`, which is what that field exists for. Sparse, because intensities are released only for accumulating pairs. | Legend sheet: "Mean_Int \| Mean intensity of annotated m/z-feature of both replicates." / "R1_Int+R2_Int \| Intensity of annotated m/z-feature in replicate 1 (R1) and replicate 2 (R2)." MEASURED: `Mean_Int` equals `mean(R1_Int, R2_Int)` on 1,385 of 1,385 rows. |
| `LC-MS/MS`, `Reactant`, `Reactant Abbreviation`, `Subsystem`, `Substrate Abb`, `Product Abb`, `Operon effect`, `Pathways`, `PathwaysAbb`, `Position`, `Upstream Accumulation`, `Downstream Accumulation` | `si6.xlsx` `TableS5` | no | not a phenotype: iML1515 / EcoCyc annotations and derived pathway-topology flags | Legend sheet, e.g. "Reactant \| 1 - accumulating metabolite (or at least one isobaric metabolite) is a reactant of the CRISPRi targeted gene (based on iML1515 model)." MEASURED: `Reactant==1` on 132 rows, `LC-MS/MS==1` on 1,256, `Operon effect==1` on 53. |
| `FBA` activity call | `si6.xlsx` `TableS5` column `FBA`: `Active` 860, `Not Active` 471, `Inactive in Active Operon` 44, `control` 10 | no | not a phenotype: an iML1515 FBA **prediction**, not a measurement. **This is the only flux content in the whole SI** -- no flux value is released anywhere, so `FluxPhenotype` has nothing to hold from this paper. | Legend sheet: "FBA \| Active - Gene is active under growth on glucose according FBA (flux balance analysis). 'Not Active' - Gene is not active. 'Inactive in active operon' - Gene is not active, but the gene is located in an operon with active genes." |
| targeted LC-MS/MS EIC peak-height `fold-change` per strain-metabolite pair | `si7.xlsx` `Table_S6` column `fold-change`, 1,256 rows, 411 unique `Gene`, 289 unique `Abbreviation`, all 1,256 non-null | no | **loadable now**: `MetabolitePhenotype` on the existing `BacterialMetaboliteExperiment` pair, `measurement_type='lc_msms_eic_peak_height_fold_change'`, `n_replicates` 1 per key, no released uncertainty. This is a SECOND PLATFORM, not a copy of what we store (see the comparison section). | Legend sheet: "fold-change \| Intensity of highest peak in EIC of the strain compared to median peak intensity of all other strains." Figure 2C caption: "${\mathsf{Log}}_{2}$-fold changes for 1,256 strain-metabolite pairs measured by targeted LC-MS/MS." |
| `Intensity PrecMz` (precursor intensity at EIC peak maximum) | `si7.xlsx` `Table_S6` | no | **loadable now** alongside the row above, as a separate `measurement_type` record; low value because it is an instrument-scale intensity with no normalization | Legend: "Intensity PrecMz \| Intensity of precursor ion at peak maxiumum within the EIC." |
| `retention time (sec)`, `Deviation (Da)`, `Scan # MS1/CE10/CE20/CE40`, MS2 fragment m/z + intensity + normalized intensity at 3 collision energies, matching predicted fragment m/z + carbon count + SMILES, `# predicted fragments`, `total # of fragments`, `At least one predicted fragment`, `DataFile`, `ExpSpectrum`, `ExpSpecMerged` | `si7.xlsx` `Table_S6` | no | not a phenotype: spectral and instrument metadata | Legend sheet, 26 entries |
| `QC passed` flag | `si7.xlsx` `Table_S6`; MEASURED 569 of 1,256 rows | no | not a phenotype: a quality flag, but it IS the gate on any loaded LC-MS/MS value | Legend: "QC passed \| Measurement met the following criteria: Precursor intensity >5000, log2-fold change >1, high precursor purity in MS1 (mass deviation <0.003, no side peaks within isolation width of quadrupole) and at least one MS2 fragment." |
| `Mean_FC` / `R1_FC` / `R2_FC` of the 2,847 **annotated** accumulating m/z features (adducts, neutral losses, 13C isotopes beyond Table S4's six adducts) | `si8.xlsx` `Table_S7`; MEASURED 2,847 annotated rows over 254 genes, 1,946 distinct (gene, polarity, mass) keys | no | **loadable now**, as a `MetabolitePhenotype` record with its own `measurement_type`, BUT see the duplication warning below: these are a THIRD set of numbers for strain-feature pairs we already store | Figure 6B: "2,847 accumulating $m/z$ features (across all CRISPRi strains) could be annotated to an iML1515 metabolite either (de-)protonated or as an adduct or isotope shown in (A). 6,615 accumulating $m/z$ features were not annotated." |
| `Mean_FC` / `R1_FC` / `R2_FC` of the 6,615 **unannotated** accumulating m/z features | `si8.xlsx` `Table_S7`, rows whose `Metabolite` is the literal string `empty`; MEASURED exactly 6,615 such rows over 262 genes and 2,588 distinct masses | no | **blocked**: `MetabolitePhenotype.metabolite_level` keys are documented as "metabolite_id -> measured level (Yeast9 s_NNNN id, or product name)" and an m/z feature with no structure is neither, so `target_metabolite_ids` has nothing to map and the key would be a mass masquerading as an identity. Needs a typed unidentified-feature key (or an explicit decision that a `m/z + adduct + polarity` string is a legitimate key). Not named by any of the five open issues. | Results: "This analysis identified 9,462 cases in which m/z features accumulated in a CRISPRi strain (Table S7). ... This filtering step resulted in 6,615 cases in which unannotated $m/z$ features accumulate in a CRISPRi strain" |
| `Mean_Int` / `R1_Int` / `R2_Int` in Table S7 | `si8.xlsx` `Table_S7` | no | **`Mean_Int` is unusable as released.** MEASURED: `Mean_Int` equals `Mean_FC` on **9,462 of 9,462 rows** and equals `mean(R1_Int, R2_Int)` on **0** rows, so the column holds the fold change, not an intensity. `R1_Int` / `R2_Int` are real intensities, so the mean can be recomputed; Table S5's `Mean_Int` is correct on all 1,385 rows. A release error, not a schema gap. | computed from the workbook with pandas |
| SIRIUS / CSI:FingerID structure predictions for the MetE features: `ConfidenceScoreExact`, `ConfidenceScoreApproximate`, `CSI:FingerIDScore`, `ZodiacScore`, `SiriusScore`, `molecularFormula`, `adduct`, `InChI`, `smiles`, `xlogp`, `ionMass`, `retentionTimeInSeconds`, `overallFeatureQuality` | `si9.xlsx` `Table_S8`, 51 data rows | no | not a phenotype: in-silico structure annotation of features, not a measurement of a strain | Methods: "Structure prediction was performed with SIRIUS $6.0.7^{45}$ including CSIFingerID ... The top 3 SIRIUS results per $m/z$ feature are shown in Table S8." |
| 802 isobaric metabolite identities (BiGG, KEGG, monoisotopic mass, neutral formula) | `si10.xlsx` `TableS9`, 802 rows | **consumed** as the identity layer | not a phenotype | `rapp2026.py:22-26` |
| 7,683 iML1515 gene-reactant pairs | `si11.xlsx` `TableS10` | no | not a phenotype: GPR annotation | Methods: "Reactants (substrate or product) of all genes were extracted from the gene-protein-reaction associations from the iML1515 model9 (Table S10)." |
| 3,340 EcoCyc pathway-gene rows with `Evidence` codes | `si12.xlsx` `Table_S11` | no | not a phenotype: pathway annotation | Methods: "Pathways of E. coli K-12 substr. MG1655 were extracted from the EcoCyc database ... with a customized Matlab script (Table S11)." |
| 11,851 pathway-reactant rows with `mass` and `mass_13C` | `si13.xlsx` `Table_S12` | no | not a phenotype: pathway annotation | Methods: "Reactants of iML1515 model were added to the genes of the pathway list from EcoCyc (Table S12)." |
| per-strain replicate mean relative error (MRE), printed in each Data S1 panel | `si14.pdf` / `si14.md` | no | not a phenotype: figure-only, and derivable from the two replicate values we already store | Results: "with only 37 strains showing a mean relative error greater than $40\%$ between replicates"; `si14.md` panel text "yqhD ... MRE = 0.14" |
| MS2 reference spectra (QC-passed, three collision energies, with predicted-fragment SMILES and retention times) | `si15.zip` -> `DataS2.mgf` | no | not a phenotype: a spectral library | Results: "MS2 spectra at three collision energies, SMILES codes of predicted fragments, and retention times are integrated into an MGF file, which can be used as putative reference spectra (e.g., for spectral library search in mzMine; Data S2)." |
| MurQ docking / MD: `Stable reps`, per-residue interaction frequency %, `dG`, `Cou`, `Hbond`, `Lipo`, `LN` for 5 ligands | `si1.md` Data S3 table | no | not a phenotype: in-silico, and not keyed to a strain | "Data S3 Protein ligand interaction frequency $(\%)$ and predicted binding energy" |
| carotenoid absorbance at 470 nm normalized to sampling OD, IspB vs control | Figure 7C only; Methods "Quantification of carotenoid levels" | no | not a phenotype here: **figure-only**, no released table, 2 strains. If it were released it would want `ProductTiterPhenotype`, which has no bacterial experiment wrapper. | Results: "The IspB strain produced almost 4-fold more carotenoids than the control"; Methods: "Extracts were diluted 5-fold in ethanol and absorbance was measured at $470\mathsf{nm}$ in a plate reader (Spark Tecan). Resulting data was normalized to the OD at the time point of sampling." |
| chorismate-intermediate time course, AroC vs control, n=3, 8 h, MRM | figure only; Methods "Quantification of intermediates in chorismate biosynthesis" | no | not a phenotype here: figure-only, no released table | caption: "normalized to the mean of the control strain at t $=1$ h. Big dots are the mean of $n=3$ replicates that are shown as smaller dots." |
| MetE triplicate LC-MS/MS fold changes for L-homocysteine 5.15, O-acetyl-L-serine 26.68, O-succinyl-L-homoserine 8.6 | Figure S11 caption in `si1.md` | no | not a phenotype here: three numbers in a caption, no table | "Accumulation of L-homocysteine (fold change: 5.15, compared to control), O-acetyl-L-serine (fold change: 26.68) and O-succinyl-L-homoserine (fold change: 8.6) was verified via targeted LC-MS/MS and SIRIUS predictions." |
| pairwise Jaccard indices of differential-metabolite sets across strains | Figure S3 | no | not a phenotype: a derived summary of what we already store | "Pairwise Jaccard indices were calculated based on the sets of differential metabolites in each CRISPRi strain" |

Counts: **27 released quantities enumerated**; 2 stored (the Table S4 matrix and its Table
S5 restatement); **5 loadable now** (Table S5 intensities, Table S6 `fold-change`, Table S6
`Intensity PrecMz`, Table S7's 2,847 annotated features, and the growth AUC as a
derivation); **3 blocked** (the Table S2 growth curves, the Table S3 sampling OD, Table
S7's 6,615 unannotated features); **17 not a phenotype**.

**No second condition, no second dose, no other omics.** One medium (M9 + 5 g/L glucose),
one inducer dose (200 nM aTc), one temperature (37 C), one sampling time (6.5 h). The
paper releases **no proteomics and no transcriptomics**: case-insensitive grep of
`paper.md` for `proteom` returns exactly one hit, reference 51 (Proteowizard), and
`transcriptom`, `ribosome`, `doubling time` and `growth rate` return zero. The one protein
statement, "In our conditions, protein abundance decreased on average 5-fold after $4.5~\mathsf{h}$,
but the extent of decrease varies with the target gene and the efficiency of the sgRNA",
cites reference 18 (Donati 2021), not data released here.

**Comparisons against other data.**

- **Fuhrer 2017, which we already load.** Introduction: "A study with the E. coli KEIO
  knockout collection measured metabolome changes across 3,807 gene deletion strains and
  showed that single-gene deletions often result in broad and sometimes unanticipated
  metabolite changes.17 However, such studies with gene deletions are restricted to
  non-essential genes and exclude many enzymes in major de novo biosynthesis pathways that
  produce essential amino acids, nucleotides, and other biomass building blocks." This is a
  **complementarity** claim, not a duplication: Fuhrer is Keio deletions scored as modified
  z-scores over 3,169 negative + 4,365 positive ions
  (`torchcell/datasets/ecoli/fuhrer2017.py:5-14`), Rapp is CRISPRi knockdowns scored as
  linear per-batch-median fold changes over 1,880 annotated features. Different strains,
  different platform, different statistic. **No duplication risk against Fuhrer.**
- **Internal cross-platform comparison, the one real duplication hazard.** Figure 2C
  compares FI-MS against targeted LC-MS/MS on the same 1,256 strain-metabolite pairs, and
  Results says the targeted data "confirmed metabolite increase in $85\%$ of the tested
  pairs". MEASURED by me: the 1,256 `Table_S6` rows join 1:1 onto `TableS5` on
  (`Gene`, `Abbreviation`, `Mode`), and `Mean_FC` vs `fold-change` gives **Pearson r =
  0.6722** on the linear fold change (0.6701 on log2) with a **median absolute log2
  difference of 1.1649**. These are NOT the same number, so loading the LC-MS/MS values
  adds a measurement rather than duplicating one, provided `measurement_type` separates
  them.
- **Table S5 and Table S7 disagree on the same strain-feature pairs.** MEASURED: joining
  `TableS5` to `Table_S7`'s annotated rows on (`Gene`, polarity, `Mass` to 3 dp) gives 689
  pairs, **0 of which have equal `Mean_FC`**, at Pearson r = 0.955 and median absolute log2
  ratio 0.209 (example: `aas` / 450.2626 / neg is 5.3040 in Table S5 and 5.0167 in Table
  S7, with different `R1_FC` / `R2_FC` too). Since Table S4 and Table S5 agree exactly
  (`max_abs_difference 0.0`), Table S7 is a THIRD set of numbers for pairs we already
  store. Hypothesis (untested): Table S7 comes from a separate per-m/z-feature peak-picking
  pass rather than the merged-isobaric annotation pipeline that produced Tables S4 and S5.
  **Any admission of Table S7 must give it its own `measurement_type` and must not be
  described as a superset of the stored matrix.**
- **Three CRISPRi screens cited for agreement, none of them loaded here.** Results: "The
  other 46 essential genes in our library did not show a growth defect, which is consistent
  with other CRISPRi screens in E $.coI i^{21-23}$ and probably due to overcapacities of
  enzymes and other compensatory mechanisms.18" Those are Rachwalski 2023 (already row-listed
  in `notes/plan.bacteria-ontology-genome.md:511`), Anglada-Girotto 2022 (already surveyed in
  `notes/experiments.database.expansion-100.md:188`) and Hawkins 2020 (already surveyed in
  `notes/experiments.database.expansion-bacteria.md:311`). The comparison is qualitative, so
  it points at datasets we have already triaged, not at a new one.
- **Donati 2021 is the one cited dataset that is nowhere in the repo or the mirror.**
  Reference 18, Donati et al. 2021, "Multi-omics Analysis of CRISPRi-Knockdowns Identifies
  Mechanisms that Buffer Decreases of Enzymes in E. coli Metabolism", Cell Syst. 12, 56-67.e6,
  doi:10.1016/j.cels.2020.10.011. It is the **parent pooled CRISPRi library** this arrayed
  library was sorted from ("The 1,515 CRISPRi strains were derived from a pooled CRISPRi
  library,18 by sorting single strains into 96-well plates") and it is the source of the
  5-fold knockdown figure Rapp relies on, i.e. it releases the proteome and metabolome
  measurements that this paper does not. `grep -rli donati` over `notes/` and `torchcell/`
  returns only an unrelated word match in `notes/database.docker.build.overview.md`, and
  there is no `torchcell-library/*onati*` directory. Flagging it as the loadable dataset this
  paper's own lineage points at.
- **Model-versus-data, not dataset-versus-dataset.** Figure S10 and the Discussion contrast
  the screen with iML1515 FBA predictions: "flux balance analysis with iML1515 predicts that
  peptidoglycan recycling is inactive during exponential growth on glucose in minimal medium.
  Our data, however, indicate tha[t]". The `FBA` column of Table S5 is that prediction. Not a
  measurement and not a second dataset.
- **ECMDB spectral comparison.** "we obtained ${\mathsf{M}}{\mathsf{S}}^{2}$ spectra of 174
  unique metabolites, for 102 of which no experimental spectrum is available in the ECMDB (E.
  coli metabolome database) database." A reference-spectrum gap, not a phenotype.

**Loadable-now estimate.**

- **Targeted LC-MS/MS `fold-change` (the biggest clean item).** **411 records**, one per
  CRISPRi strain, each a `MetabolitePhenotype` whose dict holds that strain's 1 to 10
  measured metabolite keys; **1,256 strain-metabolite measurements** in total, 289 distinct
  metabolite abbreviations. Basis: measured from `si7.xlsx` sheet `Table_S6` with pandas,
  `len(df) = 1256`, `df["Gene"].nunique() = 411`, `df["Abbreviation"].nunique() = 289`,
  `df["fold-change"].notna().sum() = 1256`. Only 569 rows carry `QC passed == 1`, so a
  QC-gated variant is **569 measurements** over an unmeasured (smaller) strain count.
- **Table S5 FI-MS absolute intensities.** **411 records** (the same 411 strains),
  **1,385 strain-metabolite intensity values** with per-replicate `R1_Int` / `R2_Int`, so
  `n_replicates` 2 and a real `metabolite_level_se`. Basis: `si6.xlsx` `TableS5`,
  `len(df) = 1385`, `df["Gene"].nunique() = 411`.
- **Table S7's 2,847 annotated features.** **254 records**, **2,847 fold-change values**
  over 1,946 distinct (gene, polarity, mass) keys. Basis: `si8.xlsx` `Table_S7` filtered to
  `Metabolite != "empty"`. Gated on the duplication warning above.
- **Growth, as the paper's AUC.** **1,515 records** if a per-gene scalar is adopted
  (**1,514** after the loader's existing `b_number_remapped_by_the_annotation` rule removes
  `phnE`; `argR` is already absent from Table S2), against 16 control strains. Basis:
  measured from `si3.xlsx` `Table_S2`, 4,593 rows = 1,515 genes x 3 replicates (4,545) plus
  48 control rows (`Guide Nr.` `ctrl1`..`ctrl16` x 3). **18 of those genes have no metabolome
  sample at all** (`btuD`, `cyoC`, `gltA`, `ilvD`, `ispH`, `lysP`, `mgtA`, `mngB`, `napH`,
  `pabB`, `patA`, `pgi`, `ppsA`, `serC`, `thyA`, `torA`, `yejA`, `ynfF` -- the paper's 17 OD
  failures plus one), so growth records would extend the dataset's strain coverage, not just
  its phenotype coverage. This item is listed as loadable only under the derivation caveat in
  the table; the released form (the 181-point curve) is blocked.
- **Sampling OD.** **1,513 records** (1,498 CRISPRi + 15 controls), **3,026 measurements**.
  Basis: `si4.xlsx` `Table_S3`, 3,026 rows, 1,513 unique `Target gene`, `Replicate` R1 1,514
  / R2 1,512. Blocked today for want of a field.
- **Table S7's 6,615 unannotated features.** 262 records, 6,615 values, 2,588 distinct
  masses, if the unidentified-feature key question is settled. Blocked today.

**Not checked.**

- `si1.pdf`, `si14.pdf`, `si16.pdf`, `si17.pdf` were read through their MinerU `siN.md` OCR,
  not the PDF bytes. The figure PANELS in `si1.pdf` and `si14.pdf` are images; any number
  that exists only inside a plotted panel and not in its caption was not read.
- `si15.zip`'s `DataS2.mgf` content was listed but not parsed. It is a spectral library, not
  a per-strain phenotype, so parsing it would not change the enumeration.
- The three MassIVE deposits (MSV000098712, MSV000098755, MSV000098714) were not retrieved.
  They hold raw spectra, already recorded as not mirrored at `rapp2026.py:276-278`.
- The Zenodo deposit for the MurQ MD trajectories (10.5281/zenodo.16100343, named in Data
  S3) was not retrieved; trajectories are not a phenotype.
- `si5.xlsx` was read with openpyxl `read_only=True` for its sheet name, header row and
  dimensions (`A1:DLN1881`, 1,880 feature rows x 3,030 columns) plus three sample data rows.
  Its 5.7 million cells were not loaded; the stored values were taken instead from the built
  dev store's `preprocess/` sidecars.
- Figure 7's caption is absent from the `paper.md` OCR (the OCR carries captions for Figures
  1 to 6 only, while the body cites Figure 7A/7B/7C), so the carotenoid panel's replicate
  count and units come from the Methods paragraph rather than from the caption.

### schmidtQuantitativeConditiondependentEscherichia2016 -- `ecoli/schmidt2016.py`

**SI inventory.** Only two files, and one of them carries everything:

| file | format | what it is | bytes |
|---|---|---|---|
| si1.pdf (+ si1.md, 45 KB OCR) | pdf | Supplementary Figures 1 to 18 and Supplementary Notes | 6,659,254 |
| si2.xlsx | xlsx | **Supplementary Tables S1 to S29, 30 sheets** (29 tables plus `CONTENT_AND_ABBREVIATIONS`) | 17,128,596 |

The OCR sidecars (`si1_middle.json`, `si1_content_list.json`, `si1_ocr_provenance.json`)
exist. Every sheet name and column header below was read with
`openpyxl.load_workbook(read_only=True, data_only=True)`; the non-empty row count per
sheet was counted the same way.

**What the loader stores.** `ProteomeSchmidt2016Dataset`, `BacterialProteinAbundanceExperiment`
/ `ProteinAbundancePhenotype`, **one quantity: protein copies per cell, from Table S6**,
`measurement_type="absolute_protein_copies_per_cell_label_free_sid_anchored"`. **Measured:
14 entries** in `$DATA_ROOT/data/torchcell/proteome_schmidt2016/processed/lmdb`, which
matches the note's `### What is stored` table ("records | 14, one per loaded BW25113 growth
condition", `notes/torchcell.datasets.ecoli.schmidt2016.md:24`), each carrying 2,329
protein keys (2,034 on three records).

**Already recorded as not stored.** Retire these; the project has decided them.

- **Table S6's `Protein Mass (fg) / Cell` block**: "The mass block is not stored: it is the
  same number times the released molecular weight" (`notes/...schmidt2016.md:56`).
- **Table S6's `Coefficient of Variance (%)` block**: the loader derives an exact SE from
  Table S8's three replicate columns instead (`notes/...schmidt2016.md:99`, section "The
  uncertainty: an exact SE, not the released CV").
- **Seven of the 22 released conditions**: `Glycerol + AA` (no `MEDIA_LIBRARY` entry), the
  four chemostat arms (`culture_not_batch`, no dilution-rate slot on `Environment`, the
  same gap #753 item 3 names) and the two stationary-phase arms
  (`growth_phase_not_representable`). `notes/...schmidt2016.md:239` with the closing
  arithmetic, and `schmidt2016.py:68`.
- **Table S9's MG1655 and NCM3722 arms**: "MG1655 is a deposited reference strain of its own
  with its own b-number namespace, so those two records belong to a sibling dataset class
  with `REFERENCE_STRAIN = "MG1655"`; NCM3722 has no deposited assembly at all. Both are
  named in the PR as follow-up" (`notes/...schmidt2016.md:192`).
- **Table S8's `Glucose.2` group**: the Supplementary Figure 7 reproducibility arm, absent
  from Table S6 (`notes/...schmidt2016.md:198`).
- **That the three Keio deletion strains carry no per-protein abundance**: "The three Keio
  deletion strains in the paper (``rimI``, ``rimJ``, ``rimL``) appear only in the
  N-alpha-acetylation tables (Supplementary Tables 20, 21 and 24) and carry no per-protein
  abundance at all" (`schmidt2016.py:55`). Note the scope: the loader rules out an
  ABUNDANCE record for those strains. It does not address their growth rates, which is the
  gap below.
- **ProteomeXchange PXD000498 and si1.pdf**: deliberately not mirrored,
  `schmidt2016.py:245` `NOT_MIRRORED`.

**Which tables the loader has ever named.** Measured by grepping both the loader and the
note: **Table S6, S7, S8, S9, S25, and Supplementary Tables 2, 3, 17, 20, 21, 24**. Tables
**S1, S4, S5, S10, S11, S12, S13, S14, S15, S16, S18, S19, S22, S23, S26, S27, S28 and
S29 appear nowhere in either file.** Seventeen of the 29 released tables have never been
classified, which is what this audit closes.

**Released quantities and their classification.**

| released quantity | source sheet + column | stored | classification | evidence |
|---|---|---|---|---|
| protein copies per cell, 22 conditions, combined | Table S6, group header `Protein copies/cell` at col 8 | **yes** (15 of 22 conditions: 14 records + 1 reference) | stored | S6 title: "Final table with combined global absolute abundance estimations from both datasets including functional annotations using cluster of orthologues groups (COG)" |
| protein mass fg/cell, 22 conditions | Table S6 col 30 group `Protein Mass (fg) / Cell` | no | not a phenotype: derived, copies x released MW | already recorded, `notes/...schmidt2016.md:56` |
| CV % between biological triplicates, 22 conditions | Table S6 col 52 group | no | not a phenotype: superseded by the exact SE the loader derives | already recorded, `notes/...schmidt2016.md:99` |
| **per-condition growth rate (h-1) + Stdev** | **Table S23 cols `Growth rate (h-1)`, `Stdev`**, 26 strain-condition rows | **no** | **loadable now**: `EnvironmentResponsePhenotype` with `measurement_type=MeasurementType.growth_rate` ("absolute or normalized growth rate / doubling time", `schema.py:4642`) + `BacterialEnvironmentResponseExperiment` | S23 title: "Experimental details for all E. coli samples analzed in this study including growth rate, harvesting conditions, OD-values and number of identified proteins". Measured rows include `LB \| BW25113 \| 1.9 \| 0.03` and `Acetate \| BW25113 \| 0.3 \| 0.04` |
| **deletion-strain growth rate (h-1), 3 replicates + Average + Stdev** | **Table S24**, `WT`, `ΔrimI`, `ΔrimJ`, `ΔrimL` in `Glucose:` and `Acetate:` | **no** | **loadable now**: `FitnessPhenotype` (ratio to the WT row of the same medium) or `EnvironmentResponsePhenotype` `growth_rate`, with `BacterialDeletionPerturbation`; `fitness_uncertainty_type=sample_sd`, `n_samples=3` (2 for WT glucose and ΔrimJ acetate, which release two replicates), `sample_unit=biological_replicate` | S24 title: "Growth rates determined for WT, ΔrimI, ΔrimJ and ΔrimL E. coli strains grown in glucose and acetate medium". Measured: `ΔrimJ \| 0.3502 \| 0.3709 \| 0.365 \| 0.3620333333333334 \| 0.010664114277957324` in glucose |
| per-condition single-cell volume (fl) | Table S23 col `Single cell volume [fl]1` | no | not a phenotype: a derived conversion factor, computed not measured (the column carries footnote 1, "Volume calculated for a...") | read from Table S23 |
| per-condition doubling time | Table S23 col `Doubling time (h-1)` | no | not a phenotype: reciprocal of the growth rate above | read from Table S23; `LB \| 1.9 \| ... \| 0.4` |
| per-condition OD600 at harvesting, 3 replicates | Table S23, the three unlabeled columns under `OD @ harvesting. replicates` | no | **loadable now in principle, but not worth a record**: an OD at harvest is a sampling checkpoint, not a growth phenotype, and the Methods say every sample was taken at the same physiological point ("Samples for proteome analyses were taken from cells that were grown until they reached ten divisions in exponential state") | read from Table S23, e.g. `LB \| ... \| 1.8 \| 1.37 \| 1.34` |
| per-condition number of proteins identified | Table S23 col `Number of Proteins Identified` | no | not a phenotype: an assay-coverage statistic | read from Table S23, `1752` for LB BW25113 |
| absolute copies/cell per (protein, peptide, condition) by targeted SRM, dataset 1, + SD from technical replicates | Table S2, cols `Protein Abundance (copies/cell)`, `Standard Deviation (copies/cell) from Te...`, 779 data rows | no | **loadable now, with a caveat**: a genuinely different assay (targeted SRM against synthetic heavy peptides) of the 41 anchor proteins, so a distinct `measurement_type` on `ProteinAbundancePhenotype`. The caveat is that it re-measures proteins Table S6 already covers, so it must not be mixed with the stored block | S2 title: "Absolute quantification of selected proteins for dataset 1 (see supplemental Figure 4 for dataset details)" |
| the same by SRM, dataset 2, + SD from biological replicates | Table S3, 682 data rows | no | loadable now, same caveat | S3 title: "Absolute quantification of selected proteins for dataset 2" |
| copies/cell and mass/cell per dataset, before combination | Table S4 (19 conditions), Table S5 (22 conditions) | no | not a phenotype: the per-dataset precursors of the stored Table S6, and Table S6 names which dataset each row came from in its `Dataset` column | S6 title "Final table with combined ..." against S4/S5 "Global absolute abundance estimations for all proteins identified in dataset 1 / 2" |
| `medianRatio_<condition>_vs_glucose`, dataset 1, 18 conditions | Table S7, 2,039 data rows x 18 | no | **blocked**: `ProteinAbundancePhenotype.protein_abundance` is specified as "absolute per-strain quantity on a log signal scale, NOT a ratio" (`schema.py:4538`), so a released fold change has nowhere to go. **Not named in #749, #753, #756, #758 or #760** | S7 title, and the class docstring |
| `medianRatio_<condition>_vs_Glucose` + `qvalue_<condition>` + `cv_<condition>`, dataset 2, 23 conditions | Table S8, 2,058 data rows; the replicate `normInt_` columns ARE consumed for the SE | partly (replicates only) | **blocked**, two ways: the ratio as above, and the q-value because no phenotype class carries a p-value or FDR slot. The q-value gap is the SAME one the Rousset loader records for `padj` (`notes/torchcell.datasets.ecoli.rousset2018.md:154`) and it is in no issue | S8 title: "... inlcuding statiscal analysis from biological triplicates using label-free quantification and SafeQuant data analysis and the coefficient of variation for each protein across the growth conditons" |
| copies/cell, fold change, q-value, CV and raw and normalized replicate intensities for BW25113, MG1655 and NCM3722 in LB and glucose | Table S9, 2,038 data rows, 7 column groups | no | already recorded as PR follow-up for the two non-BW25113 strains; the two BW25113 arms duplicate Table S6 | `notes/...schmidt2016.md:192`; paper.md Online Methods: "Additionally, the proteome for the glucose and LB condition was also determined for the strains MG1655 ... and NCM3722 ..." |
| per-protein COG letter, description and class | Table S10 (2,057 rows), and the same four columns at the tail of Table S6 | no | not a phenotype: annotation | S10 title: "List of proteins that could be functionally annotated using cluster of orthologues groups (COG)" |
| protein mass per COG group per sample | Table S11 (47 rows x 69 sample columns), Table S12 (6 rows x 69) | no | not a phenotype: derived summary, a per-sample sum over the stored copies x MW | S12 title: "Total protein mass (in fg/cell) that could be functionally assigned to the different four COG classes for the individual growth condtions and biological replicates analyzed" |
| per-protein subcellular location, and protein mass per sample | Table S13, 1,174 data rows x 69 sample columns | no | not a phenotype: the location is annotation ("Cellular protein location (accordi[ng to]...)"), the mass is derived | S13 title: "Distribution of proteins and their cellular mass assigned to different subcellular locations" |
| protein mass by subcellular location per condition | Table S14, 21 rows x 22 conditions | no | not a phenotype: derived summary | S14 title |
| periplasmic binding protein mass fraction per condition | Table S15, 121 data rows x 22 | no | not a phenotype: derived summary | S15 title: "Distribution of annotated periplasmic binding protein mass to total periplamsic protein mass (in %) across the individual growth conditions" |
| periplasmic binding protein : ABC transporter stoichiometry, per condition | Table S16, 122 data rows x 22 | no | not a phenotype: a derived ratio of stored values | S16 title |
| identified post-translational modifications: `Variable Modification`, `Modification Position`, `Mascot ion score (best hit)`, `Mass error (ppm, best hit)` | Table S17, 358 data rows | no | **blocked**: there is no PTM phenotype class, and the per-site modification is a measured property of a protein in a condition, not an abundance. **Not named in any of the five issues** | S17 title: "Post-translational modifications identified across all growth conditions (all raw data and assigend MS/MS spectra can be obtained via PRIDE ...)" |
| lysine sites carrying more than one PTM type | Table S18, 20 data rows | no | not a phenotype: a site annotation with no measured value | S18 title: "Single protein sites (lysine residues) for which different types of post-translational modifications were identified" |
| peptide-spectrum-match count per modification per condition | Table S19, 17 data rows x 22 conditions | no | not a phenotype: a spectral count, an assay-coverage statistic | S19 group header: "Number of peptide spectrum matches" |
| Nα-acetylated PSMs in WT, ΔrimI, ΔrimJ, ΔrimL in acetate, with `Replicate`, `Strain`, spectrum counts and sequence coverage | Table S20, 138 data rows | no | **blocked**, same PTM gap. The strain axis is real (three deletions), so this is a PTM measurement on perturbed strains | S20 title: "List of Nα-acetylated peptide spectrum matches identified from WT, ΔrimI, ΔrimJ and ΔrimL strains grown in acetate medium" |
| Nα-acetylated peptide precursor intensities, 4 strains x 3 replicates, in glucose | Table S21, 22 data rows x 24 intensity columns | no | **blocked**, same PTM gap. This one carries QUANTITIES per strain, measured: the header row is `WT, WT, WT, ΔrimJ, ΔrimJ, ΔrimJ, ΔrimI, ΔrimI, ΔrimI, ΔrimL, ΔrimL, ΔrimL` twice over, as raw then normalized intensities | S21 title: "List of Nα-acetylated peptides relatively quantified from WT, ΔrimI, ΔrimJ and ΔrimL strains grown in glucose medium using label-free quantification from biological triplicates" |
| modified-peptide `medianRatio_<condition>_vs_Glucose`, 22 conditions, plus confidence score and mass error | Table S22, 96 data rows | no | **blocked**, PTM gap plus the ratio gap | S22 title: "List of modified peptides relatively quantified using label-free quantification from biological triplicates inlcuding statisitcal analysis" |
| file name to condition and strain map | Table S25, 80 data rows | consumed | not a phenotype: sample metadata, read by the loader | `schmidt2016.py:485` reads `Table S25` |
| SRM transition list (precursor, product, RT, charge, peptide, light/heavy) | Table S26, 423 rows | no | not a phenotype: a design parameter | S26 title: "Transition list employed for abolute quantification of 41 reference proteins using selected reation monitoring and stable isotope dilution" |
| expected and measured subunit ratios of protein complexes, per condition | Table S27, 54 data rows x 22 conditions | no | not a phenotype: a derived ratio of stored copies, benchmarked against an external expectation | S27 title (quoted under comparisons below) |
| per-condition cell length, cell width, cytosolic and periplasmic mass fractions + SD, computed volumes, geometric correction factor | Table S28, 24 data rows | no | **blocked**: a bacterial cell-size phenotype has no class. `CalMorphPhenotype` is the yeast CalMorph feature set and does not apply. **Not named in any of the five issues** | S28 title: "Periplasmic protein mass distribution geometrically corrected for increase in cell size at higher growth rates"; columns `Cell length1`, `Cell width1` |
| cryo-EM periplasm and cytoplasm width measurements, per cell, two conditions | Table S29, 35 data rows x 2 condition blocks | no | not a phenotype: per-image measurements with no strain or condition-level value released as such | S29 title: "Cryo-electron microscopy analysis of E. coli cells"; row labels `Measurement`, `PP1`, `CP`, `PP2`, `PPb` |
| selected proteins and their proteotypic peptides, with heavy reference peptide concentration | Table S1, 41 data rows | no | not a phenotype: a design parameter | S1 title: "Proteins selected for absolute quantification and their selected proteotypic peptides for which heavy reference peptides were synthesized and employed for quantification by stable isotope dilution" |

**Comparisons against other data. This paper is the richest comparison source in the set.**

1. **Supplementary Figure 8 benchmarks the proteome against six published absolute-abundance
   datasets, named with full citations in si1.md's own reference list:**

   > Supplementary Figure 8: Correlation of our absolute protein abundance estimates with
   > various published small datasets including only a few growth conditions. Specifically,
   > the quantitative data of this study was compared to Li et al.6 (A), Pedersen et al.7,9
   > (B), Taniguchi et al.10 (C), Lu et al.11 (D), Ishii et al.12 (E) and Ishihama et al.13
   > (F).

   (`si1.md:113`.) The supplemental reference list resolves them (`si1.md:236-249`):
   Pedersen 1978 Cell ("a catalog of the amount of 140 individual proteins at different
   growth rates"), **Taniguchi 2010 Science** ("Quantifying E. coli proteome and
   transcriptome with single-molecule sensitivity in single cells"), Lu 2006 Nat
   Biotechnol, **Ishii 2007 Science** ("Multiple High-Throughput Analyses Monitor the
   Response of E. coli to Perturbations"), Ishihama 2005 and 2008, Masuda 2009.

   **Two of these are already in the literature mirror and have no loader and no raw
   mirror**: `ishiiMultipleHighThroughputAnalyses2007` and
   `taniguchiQuantifyingColiProteome2010`, each holding `paper.pdf`, `paper.md` and
   `manifest.json` with **no `si/` directory** (measured by `ls`). Ishii 2007 is the
   stronger candidate of the two: it is a multi-omic survey of *E. coli* single-gene
   deletions, which is the exact record shape this program already serves for yeast. The
   remaining four are in neither mirror. **This is a pointer, not a request**: adding a
   reference or a dataset is a curation decision that needs its own go-ahead.

2. **Table S27 benchmarks measured complex stoichiometries against Li 2014**, verbatim from
   the workbook's own contents sheet:

   > Table S27 | Stoichiometries determined for quantified components of protein complexes
   > with known subunit composition as indicated in <www.uniprot.org> and Li, G.-W.,
   > Burkhardt, D., Gross, C. & Weissman, J. S. Quantifying absolute protein synthesis
   > rates reveals principles underlying allocation of cellular resources. Cell 157,
   > 624-635 (2014).

   Li 2014 releases per-gene absolute synthesis rates from ribosome profiling. Not in the
   mirror (checked by `ls`); another pointer, not a request.

3. **The three-strain comparison is internal, and is already a recorded follow-up**: Table
   S9's MG1655 and NCM3722 arms. No duplication risk, because nothing in the repo stores
   either strain's proteome.

4. **RegulonDB** supplies the regulon membership used for the co-regulation analysis
   ("proteins sharing at least one transcriptional repressor or activator; as reported in
   RegulonDB40", `paper.md:63`). A live annotation database, not a pinned measurement
   release, so not a candidate.

**No duplication risk from this paper.** Nothing it releases is another paper's data
re-released, and no other loader in the repo stores an *E. coli* proteome. The one overlap
inside the release is Table S9's two BW25113 arms against Table S6's LB and Glucose
columns, which matters only if Table S9 is ever loaded for its non-BW25113 strains; the
loader already states that nothing from Table S9 is read.

**Loadable-now estimate, largest first.**

1. **Table S2 + Table S3 SRM absolute abundances: 1,461 cell-level values**, or **41
   protein keys across up to 22 condition records** if filed the way the stored block is.
   Basis: 779 + 682 non-empty data rows counted with openpyxl, each row one (protein,
   peptide, condition) cell. Caveat stated above: it re-measures proteins Table S6 covers,
   so it is an assay-distinct addition, not new biology.
2. **Table S23 per-condition growth rate: 26 records.** Basis: 26 strain-condition rows
   counted in the sheet (31 non-empty rows minus 2 header rows and 3 footnote rows). Of
   those, 22 are BW25113 conditions that align 1:1 with Table S6's conditions, and 4 are
   the MG1655 and NCM3722 LB and glucose arms. **Caveat:** the `Stdev` column's replicate
   design is not stated. The Methods say only "The growth rate was calculated from at least
   four consecutive measurements", which describes the fit, not the replicate count, so
   `n_samples` would be a typed gap and the uncertainty type would have to be sourced or
   gapped rather than guessed.
3. **Table S24 deletion-strain growth rates: 6 records** (3 deletions x 2 media), plus 2
   wild-type references. Basis: the sheet read in full, 4 strains x 2 media, replicate
   counts 3 except WT-glucose and ΔrimJ-acetate which release 2. **This is the only
   genuinely new gene-perturbation phenotype in the paper**, and it is small but clean: a
   named Keio deletion, a stated medium, three biological replicates and a released SD.

**Not checked.**

- si1.pdf's figure images. The si1.md OCR was read in full and every figure caption was
  enumerated from it, which is how the Supplementary Figure 8 comparison was found. No
  numeric value was digitized from a figure panel.
- ProteomeXchange PXD000498 (raw spectra), deliberately not mirrored.
- Table S29's 16,384-column sheet was read only to row 6; its stated extent is 35 data
  rows across two condition blocks, and the trailing columns are empty.
- The six comparison papers' own SI. Ishii 2007 and Taniguchi 2010 are in the mirror as
  `paper.pdf` + `paper.md` only, with no `si/` directory, so their released tables were
  not examined and no record estimate for them is offered.

### shiverChemicalGenomicScreenNeglected2016 -- `ecoli/shiver2016.py`

Audit only. No loader, schema, adapter, conf, dev store, build or `git` command was touched.

**SI inventory.** All under
`$DATA_ROOT/torchcell-library/shiverChemicalGenomicScreenNeglected2016/si/`. Each TIF was
opened and read (PIL render at reduced width), so the S-number mapping below is checked,
not inferred.

- `si1.tif` -- 1,590,186 bytes, 4285 x 4880 RGB TIFF. **S1 Fig**, two panels: (A) significance
  cutoffs vs number of conditions sampled from Nichols; (B) gene-pair correlation scatter,
  current vs integrated, with the R = 0.78 and R = 0.47 FDR lines drawn. No tabular data.
- `si2.tif` -- 5,675,993 bytes, 2570 x 5250 RGB TIFF. **S2 Fig**, spot-dilution photographs:
  6 strains (MG1655 and Δopp Δdpp, each +/- ΔgcvA/ΔgcvB) x 3 plates (LB, LB blasticidin S
  [130 uM], LB kasugamycin [240 uM]), dilutions 10^0 to 10^-6. Images only, no scored values.
- `si3.tif` -- 458,450 bytes, 1427 x 2332 RGB TIFF. **S3 Fig**, two 35S-Met incorporation
  decay curves (1 mM kanamycin, 2 mM spectinomycin) for MG1655 vs Δopp Δdpp. Plot only.
- `si4.docx` -- 124,127 bytes. **S1 Text**, "Supporting Information References", refs 69 to
  82. 15 text lines, no table, no data.
- `si5.txt` -- 20,577,266 bytes. **S1 Dataset**, the integrated fitness-score matrix. Measured:
  header has 3,976 tab fields (`Condition` + 3,975 gene columns), 292 data rows.
- `si6.docx` -- 42,120 bytes. **S1 Table**, "Cold-sensitive genes from the screen". One table,
  145 data rows x 7 columns.
- `si7.docx` -- 100,109 bytes. **S2 Table**, "Strains and plasmids used in this study". One
  table, 21 rows (14 MG1655 strains, 6 plasmids, 1 header). No measurements.

Sidecars `paper_middle.json`, `paper_content_list.json`, `paper_ocr_provenance.json` and
`manifest.json` also exist at the key root. There is no `siN.md` for any SI file; the docx
text above was extracted by unzipping `word/document.xml` into fresh empty directories and
walking the XML with a script kept outside those directories.

**What the loader stores.** `BacterialEnvironmentResponseExperiment` carrying
`EnvironmentResponsePhenotype`, `measurement_type=z_score`, `assay_type=colony_size_array`,
one record per (BW25113 deletion strain x screen condition), from the 57 batch-1/batch-4
rows of `si5.txt`. 204,033 records. Stated in the module docstring
`torchcell/datasets/ecoli/shiver2016.py:1-15` and in the retention-ledger arithmetic at
lines 158-163; the dendron note repeats it at
`notes/torchcell.datasets.ecoli.shiver2016.md:17-25` and `:190-205`.

I re-derived the record count independently from the released file plus the build's own
dropped-column lists: with the 243 dropped labels (255 columns) removed, the batch-1/4
block has 212,040 cells of which 204,033 are non-blank. Exact agreement.
(Counting rule: stream `si5.txt`, drop the 243 labels listed in the read-only
`$DATA_ROOT/data/torchcell/ecoli_env_chemgen_shiver2016/preprocess/dropped_records.json`,
then count non-blank cells in the batch-1 and batch-4 rows.)

**Already recorded as not stored.** Retire all of these; they are decisions, not gaps.

- The 235 Nichols batch-0 condition rows are out of scope: "235 of those rows are Nichols
  et al. 2011 re-analyzed with this paper's improved pipeline and carry batch 0 ... those 57
  rows are what the loader serves" (`notes/torchcell.datasets.ecoli.shiver2016.md:17-25`,
  and the same sentence in the loader docstring lines 10-15). The scoping decision is
  recorded; what is NOT recorded is what that block is worth and what it collides with, which
  is the "Comparisons" section below.
- Every one of the 255 unloaded columns is named, with a rule and a count, in the retention
  ledger at `notes/torchcell.datasets.ecoli.shiver2016.md:190-229` and loader docstring lines
  120-163, and the per-label lists are materialized in `preprocess/dropped_records.json`.
  Measured against `si5.txt`, the five column rules partition exactly:
  134 tagged alleles (#749), 49 not in the BW25113 annotation, 48 merged-locus fragments,
  2 ambiguous, 22 duplicate-label columns; 8,007 blank cells in kept columns.
  3,975 - 255 = 3,720 kept; 3,720 x 57 - 8,007 = 204,033. **There is no third, unexplained
  category: the arithmetic closes to the record.**
- `n_samples` / `sample_unit` absent, deferring to Nichols 2011 which is not mirrored (#691);
  per-cell uncertainty absent; UV dose is an exposure time with no irradiance; growth duration
  absent for 56 of 57 conditions; `cassette` is None on the array
  (`notes/torchcell.datasets.ecoli.shiver2016.md:129-158`). All three gaps in #749 are these.
- The Dryad deposit (doi:10.5061/dryad.f3kc0) holding per-colony size, opacity, circularity
  and the GSEA inputs is deliberately not mirrored
  (`notes/torchcell.datasets.ecoli.shiver2016.md:280-284`, loader docstring lines 178-182).

#### Released per-record quantities

| released quantity | source file + column | stored | classification | evidence |
|---|---|---|---|---|
| fitness-score, Shiver batches 1 and 4 | `si5.txt`, 57 rows `{1}`/`{4}` x 3,975 gene columns | yes | stored | 226,575 released cells, 218,103 non-blank (measured: blanks 3,717 in batch 1 + 4,755 in batch 4) |
| fitness-score, Nichols batch 0 | `si5.txt`, 235 rows `{0}` x 3,975 gene columns | **no** | **loadable now**: `EnvironmentResponsePhenotype.environment_response`, `measurement_type=z_score`, `assay_type=colony_size_array`, in `BacterialEnvironmentResponseExperiment`, exactly as the 57 served rows are | 934,125 released cells, 888,431 non-blank (measured). Caption: "Conditions from Nichols et al. [8] were assigned batch `0`." Same strain columns, same scoring pipeline, same release file as what is already stored |
| `10˚C fitness-score` | `si6.docx` S1 Table col 5, 145 rows | yes (full precision) | stored, lower-precision copy | measured: 138 of 140 non-NaN rows agree within 0.05 of the `10C [-] {4}` row of `si5.txt`; the 2 misses are `yfiO*` and `hokD`, both duplicate-label columns, which independently corroborates loader rule 5 |
| `16˚C score` | `si6.docx` S1 Table col 7, 145 rows | **no** | subsumed by the batch-0 block above | measured: 143 of 145 rows agree within 0.05 of the `16C [-] {0}` row of `si5.txt`, same 2 duplicate-label misses. The paper's own results table is drawn from the block we do not load |
| `10˚C sensitive`, `16˚C sensitive` (TRUE/FALSE) | `si6.docx` S1 Table cols 4 and 6 | no | **not a phenotype**: derived FDR 5% hit call on the stored score | measured: 107 TRUE / 38 FALSE at 10 C, 59 TRUE / 86 FALSE at 16 C. Caveat, not a reason to load: the per-condition cutoff is not released, only the band "95% of the cutoff values for negative (sensitization) fitness-scores fell in the range (-2.0,-1.2) ... positive ... (+1.2,+2.1)", so the boolean is not exactly recomputable |
| `COG category` | `si6.docx` S1 Table col 2 (123 of 145 populated) | no | **not a phenotype**: annotation | column header read verbatim from the docx XML |
| `Cold shock or translation reference` | `si6.docx` S1 Table col 3 (12 of 145 populated) | no | **not a phenotype**: a citation | as above |
| `MIC Kasugamycin (μg/mL) (Minimal Media` | `paper.md` Table 2, 7 strains | **no** | **blocked**: `MeasurementType` has no member for an absolute inhibitory concentration. Not the #749 gap (that asks for `s_score`); this asks for an `mic` member whose value is a concentration in ug/mL, not a signed score | MG1655 50, Δopp 100, ΔoppA 100, ΔoppB 100, Δopp Δdpp 200, ΔoppA ΔdppA 200, ΔoppB ΔdppB 200 |
| `MIC Blasticidin S (μg/mL) (Minimal Media` | `paper.md` Table 2, 7 strains | **no** | **blocked**, same member | MG1655 "3-4", all six deletion strains 60 |
| 35S-Met translation-inhibition time courses and fitted decay rates | Fig 4A-C, Fig 5A, `si3.tif` S3 Fig | no | **not a phenotype** (figure only, no released numbers): the prose gives fold-changes, not values -- "the ABC-importer deletion strains showing a 7 to 10-fold slower rate of decrease in translation than MG1655"; "Error bars represent standard deviation of technical replicates $( \mathsf { n } = 3 )$" | rendered `si3.tif`: a plot, no table |
| OppA apparent K_D for Pro-Phe-Lys, +/- 1 mM kasugamycin | Fig 5B | no | **not a phenotype**: an in vitro protein-ligand constant, and no number is released | "Addition of 1 mM kasugamycin increased the effective concentration of PFK required to reach half-maximal fluorescence shift" -- no value given anywhere in `paper.md` |
| spot-dilution growth, 6 strains x 3 plates | Fig 3C, Fig 3D, `si2.tif` S2 Fig | no | **not a phenotype** (figure only): `AssayType.spot_dilution` exists, but the release scores nothing; the photographs are the datum | rendered `si2.tif`: plate images with 10^0 to 10^-6 labels, no scores |
| stress / drug family / biological target for the 26 new stresses | `paper.md` Table 1 | no | **not a phenotype**: condition annotation | table read verbatim from `paper.md:57` |
| per-colony size, opacity, circularity; GSEA input files | Dryad doi:10.5061/dryad.f3kc0 (not in the SI) | no | **not a phenotype** for our purposes, and already a recorded decision | "The raw data and metadata files for this analysis are available online (<http://dx.doi.org/10.5061/dryad.f3kc0>)" |

Totals: 14 released quantities enumerated, 2 stored (one of them a lower-precision copy),
1 loadable now, 2 blocked, 8 not a phenotype, 1 subsumed by the loadable-now block.

#### Comparisons against other data

This paper compares against exactly one external dataset, Nichols et al. 2011, and it does so
three ways. All three matter.

1. **The paper re-scored Nichols's raw images and shipped the result inside its own SI.**
   > "After reanalyzing the original images from the Nichols et al. [8] screen with our
   > improved workflow, we integrated both datasets."

   That is what the 235 batch-0 rows of `si5.txt` are. They are not a reprint of Nichols's
   published numbers; they are Shiver's pipeline applied to Nichols's plates. **Duplication
   risk, and it is a different shape from #760.** Nichols 2011 is rank 10 of the bacterial
   expansion plan (`notes/plan.bacteria-ontology-genome.md:926`, candidate,
   `ecoli/nichols2011.py`) and is not in the literature mirror (#691, confirmed: no
   `*ichols*` key under `$DATA_ROOT/torchcell-library/`). If Nichols 2011 later lands from
   its own release and the batch-0 block lands from Shiver, the store holds the **same
   physical plates scored twice by two pipelines**. Unlike Rousset/Cui (#760), the two would
   not be numerically identical, so a correlation check would not catch it; only the
   provenance would say it is one experiment. Hypothesis (untested): the right resolution is
   that the batch-0 block belongs to Nichols 2011's `publication` with Shiver 2016 as the
   processing provenance, not to a second Dataset node.

2. **Shiver re-screened 15 of Nichols's exact conditions himself, and the two disagree.**
   > "A scatter plot of individual fitness-scores for conditions present in both screens
   > $( \mathsf { n } = 1 7 ^ { \cdot }$ ). Measurements between screens are reproducible, with
   > a Pearson's correlation of 0.61."

   Measured over `si5.txt` by matching condition labels exactly including dose: 15 labels
   appear in both batch 0 and a Shiver batch at identical dose. Pooled over all 55,368
   co-present cells, Pearson r = 0.5402. Per pair it ranges from 0.182
   (`EDTA [1 mM] {0}` vs `{1}`) to 0.735 (`M9min glucose [0.2% (w/v)] {0}` vs `{4}`). My 15
   is not the paper's 17; exact-label-plus-dose matching finds 15, and compound-level matching
   ignoring dose finds 16 distinct compounds, so the paper's n = 17 counts pairs on some
   looser rule it does not state. I report my measurement and the discrepancy; I did not
   resolve which 17 the figure used.

   Consequence: a Shiver record and a (future) Nichols record for the same compound at the
   same dose are **not** interchangeable measurements. r ~ 0.54 is reproducibility, not
   identity.

3. **The paper's own results table reads out of the unloaded block.** S1 Table's
   `16˚C score` column is the `16C [-] {0}` row, a batch-0 condition (measured, 143 of 145
   rows agree). The 16 C comparison that motivates the cold-sensitivity result
   > "Growth at $1 0 ^ { \circ } \mathrm { C }$ resulted in multiple sensitivities ... with
   > almost double the number of cold-sensitive mutations as the next lowest temperature
   > $( 1 6 ^ { \circ } \mathrm { C } )$"

   cannot be reproduced from what we store, because 16 C is only in batch 0.

Other named prior work is cited as background, not benchmarked, so it implies no loadable
measurement from this paper: the Keio collection ("The KEIO deletion library is derived from
BW25113 ... [60]", Baba 2006 via Grenier 2014 for the genome), the Iris / Paradis-Bleau 2014
screen [10] from which the library also came, Collins 2006 [18] for the S-score, and the
bundle [3-10] (Joyce 2006, Hu 2009, Tamae 2008, Liu 2010, Tran 2011, Nichols 2011,
Nakayashiki 2013, Paradis-Bleau 2014) cited once as "Chemical-genomic screens in the model
bacterium Escherichia coli K-12 [3-10] have already provided a critical resource". No number
from any of them is compared against a number of Shiver's.

#### Loadable-now estimate

**Nichols batch-0 block of S1 Dataset: 835,337 records.** Basis, measured, not inferred: the
235 batch-0 rows of `si5.txt` restricted to the 3,720 columns the loader already keeps
(the 243 dropped labels taken verbatim from the build's `dropped_records.json`) give
235 x 3,720 = 874,200 cells, of which 38,863 are blank, leaving 835,337. The same rule applied
to the batch-1/4 rows reproduces 204,033 exactly, which is the control that the counting rule
is the loader's rule. This is **4.1x the dataset we currently serve from this key**, and it
needs no schema change: identical strain columns, identical score semantics, identical
experiment and phenotype classes, only 235 more `screen_id`s and their condition parsing.

The two blocked MIC columns would be 14 records (7 MG1655 strains x 2 drugs) if
`MeasurementType` gained an `mic` member. One cell is a range ("3-4"), which would need a
documented resolution rule.

#### Not checked

- **Nichols et al. 2011 itself** (Cell 144:143). Not in the literature mirror (#691), so I
  could not confirm whether its own release covers the same 235 conditions, nor compare its
  published scores against Shiver's re-scoring. That comparison is what would settle the
  duplication question in point 1, and it cannot be done until the paper is mirrored.
- **The Dryad deposit** (doi:10.5061/dryad.f3kc0). Not mirrored by design, so I did not
  enumerate its per-colony columns; the Methods name "colony size, opacity, and circularity
  from Iris (read_data.m)" and the GSEA input files, and nothing more.
- **Which 17 conditions Fig 1A used.** The figure's point coordinates are not released and the
  caption does not list the conditions, so I could only measure the 15 exact-label matches.
- **`paper.pdf` itself.** I read the OCR `paper.md` (sha256
  `4a1bec97d10f6ef1c02c99329f7adce0d79623819fcf421c098058a7c82ced72`), not the PDF. Table 1
  and Table 2 are OCR-reconstructed HTML tables in that markdown; their numbers are as the OCR
  rendered them. The S1 Table and S2 Table numbers above come from the docx XML, not OCR.
- **si5.txt was never loaded whole.** Every count above came from a single streaming pass
  (header split, then line-by-line), per the briefing.

### wangPooledCRISPRInterference2018 -- `ecoli/wang2018.py`

Wang et al. 2018, Nat Commun 9, 2475, doi:10.1038/s41467-018-04899-x. Pooled CRISPRi in
E. coli K-12 MG1655. All counts below are measured on the mirrored bytes at
`$DATA_ROOT/torchcell-library/wangPooledCRISPRInterference2018/si/` unless marked an
estimate. The measurement scripts were one-off and live outside the project tree.

#### SI inventory

`si<N>` is `41467_2018_4899_MOESM<N>_ESM.<ext>`, confirmed from the library
`manifest.json` `original_filename` field, so `si<N>` = MOESM N and Supplementary Data
`k` = `si<k+3>`. Sidecars `si1_middle.json`, `si1_content_list.json`,
`si1_ocr_provenance.json` and the same three for si2 and si3 exist and are skipped here.

| file | bytes | what it is |
|---|---|---|
| `si1.pdf` + `si1.md` | 16,778,920 / 62,441 | Supplementary Information: Supplementary Figs. 1-30, Supplementary Tables 1-8, Supplementary Note 1 |
| `si2.pdf` + `si2.md` | 2,820,645 / 166,390 | Peer Review File: three reviewers' reports and the authors' point-by-point responses. No released data table |
| `si3.pdf` + `si3.md` | 177,576 / 1,383 | "Description of Additional Supplementary Files": the 14 data-file captions, quoted below |
| `si4.xlsx` | 116,696 | Supplementary Data 1, in silico TILING sgRNA library. 1 sheet `Sheet1`, 2,281 data rows, columns `sgRNA ID`, `nucleotide sequence` |
| `si5.xlsx` | 123,127 | Supplementary Data 2, gene clusters. Sheets `protein-coding genes` (4,090 rows) and `ncRNA-coding genes` (115), columns `Cluster`, `Genes in cluster`. CONSUMED by the loader |
| `si6.xlsx` | 1,723,119 | Supplementary Data 3, genome-wide sgRNA library. 1 sheet `sheet1`, 56,071 rows, columns `sgRNAID`, `nucleotide sequence`. CONSUMED by the loader |
| `si7.xlsx` | 127,948 | Supplementary Data 4, sgRNAs per gene. 1 sheet, 4,205 rows, columns `Gene`, `sgRNA number` |
| `si8.xlsx` | 1,597,937 | Supplementary Data 5, GENE-level metrics. 5 sheets (`essential genes` 4,142 rows; `auxotrophy in MOPS media`, `amino acid addition`, `furfural tolerance`, `isobutanol tolerance` 4,004 each; 20,158 total), columns `gene`, `sgRNA number used to calculate the metrics`, `gene fitness`, `Z score`, `FDRvalue`, `FPRvalue` |
| `si9.xlsx` | 3,498,341 | Supplementary Data 6. 1 sheet `essential genes`, 54,116 rows, columns `sgRNA`, `gene`, `sgRNA fitness`, `Z score`, `Quality`. STORED |
| `si10.xlsx` | 3,082,270 | Supplementary Data 7. 1 sheet `auxotrophy in MOPS media`, 48,308 rows, same five columns. STORED |
| `si11.xlsx` | 3,111,675 | Supplementary Data 8. 1 sheet `amino acid addition`, 48,308 rows, same five columns. STORED |
| `si12.xlsx` | 3,110,921 | Supplementary Data 9. 1 sheet `furfural tolerance`, 48,308 rows, same five columns. STORED |
| `si13.xlsx` | 3,127,601 | Supplementary Data 10. 1 sheet `isobutanol tolerance`, 48,308 rows, same five columns. STORED |
| `si14.xlsx` | 34,163 | Supplementary Data 11, Keio essential gene set. 1 sheet, 313 rows, columns `Essential gene suggested by Keio collection`, `Overlap with CRISPRi data in this work`, and a third column whose header is the sentence `Note that genes with fitness score < -6 in pooled screening are regarded as essential genes` over 313 EMPTY cells |
| `si15.xlsx` | 4,522,735 | Supplementary Data 12, sgRNA libraries for six other microbes. 13 sheets: `README` (6 rows, `strain name`, `genome accession`) plus `<Genus>_protein` / `<Genus>_ncRNA` for Bacteroides, Bacillus, Corynebacterium, Mycobacterium, Listeria, Pseudomonas, each with columns `sgRNAID`, `nucleotide sequence`; 164,426 design rows in total |
| `si16.xlsx` | 475,889 | Supplementary Data 13, "Gene level metrics" for those six microbes. Same 13-sheet layout, but the real columns are only `Gene`, `sgRNA number`; 24,695 rows. The caption says "metrics"; the file holds a per-gene DESIGN COUNT and no measurement |
| `si17.xlsx` | 129,896 | Supplementary Data 14, sublibrary composition. 1 sheet, 4,205 rows, columns `Cluster`, `Genes in cluster`, `Sub library` |

The briefing's guess that si10 to si13 are per-screen RAW COUNT tables is wrong. Measured:
all four carry the five columns `sgRNA`, `gene`, `sgRNA fitness`, `Z score`, `Quality`,
identical in shape to si9, one sheet each named for its screen. They are the four
non-essentiality screens' guide fitness tables, which the loader already reads. **No
released file anywhere in this SI carries a per-guide read count.**

#### What the loader stores

`CrispriGuideFitnessWang2018Dataset`: one `BacterialEnvironmentResponseExperiment` per
(guide, screen) row of Supplementary Data 6-10 (si9 to si13), phenotype
`EnvironmentResponsePhenotype` with `measurement_type=log2_ratio`,
`assay_type=pooled_competitive_growth_barcode`, value = the `sgRNA fitness` column.
**240,481 records**, 242,294 knockdown perturbations, 4,218 distinct b-numbers, five
`screen_id` values (`essentiality`, `auxotrophy`, `trp_biosynthesis`,
`furfural_tolerance`, `isobutanol_tolerance`), from `SCREENS` at
`torchcell/datasets/ecoli/wang2018.py:307-385` and the build numbers in
`notes/torchcell.datasets.ecoli.wang2018.md` under "### Build numbers".
`n_samples=2`, `sample_unit=biological_replicate`.

Table 2 of the paper names exactly five phenotypes and the loader serves all five:

> Table 2 The phenotypes studied in this work ... Essentiality | dCas9, LB | Empty plasmid, LB ... Auxotrophy | MOPS | LB ... L-Trp biosynthesis | 0.5 g/L casamino acid, MOPS | LB ... Furfural tolerance | 0.4 g/L furfural, MOPS | initial, see Supplementary Fig. 7 ... Isobutanol tolerance | 4 g/L isobutanol, MOPS | initial, see Supplementary Fig. 7

#### Already recorded as not stored, retire these

- **The `Z score` column of si9 to si13** is a per-screen constant rescaling of the
  fitness and is deliberately not stored twice. Loader docstring,
  `wang2018.py:25-33`; note heading "### The sigma back-solve, which is the one value
  the paper does not print".
- **1,942 non-targeting rows, 4,880 `Quality == "Bad"` rows, 55 retired-b-number rows.**
  Note heading "### Retention ledger", with the arithmetic
  247,348 - 1,942 - 4,880 + 10 - 55 = 240,481.
- **Supplementary Data 5 (si8), the gene level, is not loaded** because
  `EnvironmentResponsePhenotype` has no significance field. Note heading
  "### Finding: the gene level needs a significance field the schema does not have";
  loader docstring `wang2018.py:120-131`. I sharpen it below rather than restate it.
- **Supplementary Data 4 (si7) is the per-cluster count of Supplementary Data 3**,
  verified identical for all 4,205 clusters. Raw manifest `si_expected` entry 8.
- **Supplementary Data 1 and 11-14 (si4, si14-si17) feed no record**; the SI PDFs live in
  the literature mirror. Raw manifest `si_expected` entries 10 and 11.
- **BioProject PRJNA450392 holds raw reads no loader consumes.** Raw manifest
  `si_expected` entry 12.
- **No per-guide uncertainty exists**, since the replicates are combined before the
  ratio. Note heading "### What could not be sourced".
- **Quote verbatimness is issue #758** and is not re-audited here.

#### Released quantities

| released quantity | source file + column | stored | classification | evidence |
|---|---|---|---|---|
| per-guide `sgRNA fitness`, five screens | si9-si13, column `sgRNA fitness`, 247,348 rows | **yes** (240,481) | stored | the loader's whole payload |
| per-guide `Z score`, five screens | si9-si13, column `Z score` | no | not a phenotype: `fitness / Z` is one constant per screen | already recorded, `wang2018.py:25-33` |
| per-guide `Quality` flag | si9-si13, column `Quality`, values `Good` / `Bad` only | no | blocked, but already recorded as a drop rule, not a new gap | note "### Retention ledger" |
| gene-level `gene fitness`, 5 screens | si8, column `gene fitness`, 20,158 rows | no | **not a phenotype: a derived summary of rows we already store.** MEASURED | see "The gene level is 99.0% derivable" below |
| gene-level `Z score` | si8, column `Z score` | no | not a phenotype: `gene fitness / Z` is the SAME per-screen sigma the loader back-solved from the guide level | measured, all five sheets, see below |
| gene-level `FDRvalue` | si8, column `FDRvalue`, 20,158 rows | no | **blocked**: `EnvironmentResponsePhenotype` has no significance field anywhere (`schema.py:4722-4810` lists `environment_response`, `category`, `environment_response_se`, `environment_response_uncertainty`, `n_samples`, `sample_unit`, `units`, `screen_id`, and no p/q/FDR). Already the note's open finding | note "### Finding: the gene level needs a significance field..." |
| gene-level `FPRvalue` | si8, column `FPRvalue`, 20,158 rows | no | **blocked**, same field. The quasi-gene FPR interpolation is the one quantity not recomputable from the guide rows | Supplementary Fig. 25 describes the interpolation |
| gene-level `sgRNA number used to calculate the metrics` | si8, column 2 | no | **blocked**, secondary: it would be the gene record's `n_samples`, but `SampleUnit` (`schema.py:3490-3500`) has members `colony, screen, biological_replicate, technical_replicate, pooled` and no guide-level member | measured from the schema source |
| per-guide read count per NGS library or replicate | **not released anywhere** | no | n/a: it is the input to the ratio, not a phenotype | see "Per-guide read counts" below |
| per-NGS-library QC: `Pair-end read number`, `After pretreatment`, `Mapping ratio`, `Overall remaining ratio`, `Zero-count sgRNA`, `Gini Index`, 16 libraries | si1.md Supplementary Table 2 | no | not a phenotype: sequencing QC per library, not per strain | the loader already quotes this table |
| per-replicate-pair Pearson r, 7 pairs | si1.md Supplementary Fig. 8 caption: "Pearson correlation coefficient: 0.934 (NC-R1/2), 0.921 (dCas9-R1/2), 0.994 (LB-R1/2), 0.995 (MOPS-R1/2), 0.993 (MOPSC-R1/2), 0.990 (MOPSF-R1/2) and 0.974 (MOPSI-R1/2)." | no | not a phenotype: a screen-level QC statistic on normalized read counts | verbatim from si1.md line 35 |
| Keio essential gene list, 313 genes | si14, column 1 | no | not a phenotype: an external annotation (Baba 2006 / Yamamoto 2009 via EcoCyc) | caption "Supplementary Data 11 Description: Essential gene set reported by Keio collection" |
| `Overlap with CRISPRi data in this work`, Yes 194 / No 119 | si14, column 2 | no | not a phenotype: a threshold call on si8 `gene fitness < -6`, over an external gene list | 194/313 = 61.98%, matching the paper's "62.0% have a fitness value below -6" |
| tiling-library screen, per-guide fitness and Z, 2,281 guides over 44 genes, MOPS vs LB, 10 doublings, 2 replicates | **not released**; only si4's library design is | no | not loadable: the numbers exist only inside Fig. 1b, 1c and Supplementary Figs. 1-4, 9 | see "The sixth screen" below |
| tiling-library gene fitness and MWU P value | **not released** | no | not loadable, same reason | Supplementary Fig. 9 plots it against the genome-scale auxotrophy gene fitness |
| sfGFP repression ratio per dCas9 promoter and per sgRNA, 12 h and 26 h, plus or minus IPTG | **not released**, Supplementary Fig. 22b only | no | figure-only; no released number | see "The validation experiments" below |
| lycopene titer normalized to OD600, crtE1 / crtE2 vs control | **not released**, Supplementary Fig. 22c only | no | figure-only; no released number, and there is no bacterial `ProductTiterExperiment` wrapper | Methods: "The titer was normalized to the culture $\mathrm{OD}_{600}$ value." |
| sucrose-plate growth of E. coli Msac, sacB1 / sacB2 vs control | **not released**, Supplementary Fig. 22d only | no | figure-only; would be a `VisualScorePhenotype` if a score existed | Methods: "the growth phenotype was measured (Supplementary Fig. 22d)" |
| single-cell sfGFP fluorescence distribution and average repression fold, 6 sgRNAs, 10,000 events each | **not released**, Supplementary Fig. 23 only | no | figure-only; no released number | si1.md line 102: "Lower right presents the average repression fold for each sgRNA" |
| Sanger-confirmed 2 bp indel mutants chpS, gpsA, yhhQ and their viability in liquid LB | **not released**, Supplementary Fig. 15 only | no | not a phenotype: a qualitative viability confirmation, no number | Methods: "the obtained mutants were cultivated in liquid LB culture to confirm their growth" |
| tiling-library gene annotations: `Gene`, `Operon (RegulonDB)`, `Library`, `Description`, `Function` | si1.md Supplementary Table 1 | no | not a phenotype: annotation | 44 genes, Library I / Library II |
| Supplementary Table 4: `Gene`, `Fitness`, `FDR`, `Annotation`, `CRISPR/Cas9 knockout`, `number of sgRNAs`, `Gene length (bp)`, `Essentiality by Kato et al.` | si1.md | no | not a phenotype: a 22-row subset of si8 plus literature annotation. `Essentiality by Kato et al.` is external (Kato and Hashimoto 2007, ref 34) | si1.md line 152 |
| Supplementary Table 5: 7 ncRNAs, `function`, `knockout growth phenotype`, `reference` | si1.md | no | not a phenotype: literature annotation | si1.md line 156 |
| Supplementary Table 6: `gene`, `fitness in CRISPRi`, `FPR`, `strain`, `chemical`, `mutation in paper`, `reference` | si1.md | no | not a phenotype: a subset of si8 plus literature annotation | si1.md line 160 |
| Supplementary Tables 7, 8: strains, plasmids, primers | si1.md | no | not a phenotype: design parameters | si1.md lines 180, 186 |
| guide spacers and ids, E. coli tiling and genome-scale | si4 (2,281), si6 (56,071) | si6 consumed | not a phenotype: design parameters | si6 is the loader's spacer source |
| guide spacers and ids, 6 other microbes | si15, 164,426 rows across 12 sheets | no | not a phenotype: design parameters for organisms with NO screen in this paper | caption: "Genome-wide sgRNA library for model prokaryotic microorganisms" |
| per-gene sgRNA counts, 6 other microbes | si16, 24,695 rows | no | not a phenotype: a design count despite the caption word "metrics" | columns are `Gene`, `sgRNA number` only |
| sublibrary assignment per cluster | si17, 4,205 rows, column `Sub library` | no | not a phenotype: library-design partition into ten sublibraries | caption: "Sublibrary composition of E. coli genome-scale sgRNA library" |
| cluster membership | si5, 4,205 clusters over 4,317 genes | consumed | not a phenotype: annotation. CONSUMED by the loader for multi-gene perturbations | note "### Library shape, asserted at build time" |

Totals: **31 released quantities enumerated; 1 stored (the per-guide fitness, 5 screens);
0 loadable now; 3 blocked (si8 `FDRvalue`, si8 `FPRvalue`, si8 `sgRNA number ...`); 27
not a phenotype or not released.**

#### The gene level is 99.0% derivable, which narrows the open gap to two columns

The note says of Supplementary Data 5 that "the quasi-gene FPR interpolation behind them
cannot be recomputed from the stored guide rows, so they are genuinely additional
information and not an aggregate." That is right for `FDRvalue` and `FPRvalue` and WRONG
for the other two columns, and the released bytes prove it.

**Measured** over si8 against si9 to si13: for every
gene row whose `sgRNA number used to calculate the metrics` is at least 1, the released
`gene fitness` equals, to within 1e-9, the median of that gene's N most start-proximal
`Quality == "Good"` guide fitness values, where position is the `_<position>` suffix of
the guide id. Per sheet, matched / rows:

| sheet | matched | rows | rows with N = 0 |
|---|---|---|---|
| `essential genes` | 4,141 | 4,142 | 1 |
| `auxotrophy in MOPS media` | 3,955 | 4,004 | 49 |
| `amino acid addition` | 3,955 | 4,004 | 49 |
| `furfural tolerance` | 4,004 | 4,004 | 0 |
| `isobutanol tolerance` | 4,004 | 4,004 | 0 |
| **total** | **20,059** | **20,158** | **99** |

Every one of the 99 unmatched rows is exactly a row with N = 0, which carries the
sentinel triple `gene fitness = 0.0`, `FDRvalue = 1.0`, `FPRvalue = 1.0` (example
`rdlA` in `essential genes`, `tolA` in `auxotrophy in MOPS media`). No row with N >= 1
fails. The two sheets with no N = 0 rows are the two that also report no `Bad` row,
which matches the retention ledger.

**Measured** for the gene-level `Z score`: `gene fitness / Z score` is flat within each
sheet and equals the sigma the loader already back-solved at the guide level, to twelve
decimal places:

| sheet | ratio min .. max | `SCREENS[*].nc_sigma` |
|---|---|---|
| `essential genes` | 0.849067168855 .. 0.849067168869 | 0.8490671688617482 |
| `auxotrophy in MOPS media` | 1.076137733048 .. 1.076137733066 | 1.0761377330569217 |
| `amino acid addition` | 0.730557195701 .. 0.730557195711 | 0.7305571957056847 |
| `furfural tolerance` | 1.847242807220 .. 1.847242807244 | 1.8472428072313638 |
| `isobutanol tolerance` | 1.602397640139 .. 1.602397640162 | 1.6023976401505866 |

Consequence for the open finding: **of Supplementary Data 5's six columns, four are
already in the store or derivable from it, and the unstored information is exactly
`FDRvalue` and `FPRvalue`.** That strengthens the note's case rather than weakening it:
if a significance field is added, the gene level is not a new ingest of a parallel
measurement, it is two numbers attached to a median this project can already recompute.
It also means a gene-level record is NOT blocked on provenance, only on the field.

#### Per-guide read counts are not released

The Methods state the combination step verbatim:

> Subsequently, the read counts for each sgRNA in two biological replicates were averaged as the geometric mean.

and the data availability statement defers the reads to an archive:

> Data availability. NGS raw data of CRISPR screening results for the tiling library and genome-scale library can be accessed from the NCBI Short Read Archive with BioProject ID PRJNA450392.

Supplementary Table 2 enumerates the 16 NGS libraries by name (`initial`, `dCas9-R1`,
`dCas9-R2`, `NC-R1`, `NC-R2`, `loginitial`, `LB-R1`, `LB-R2`, `MOPS-R1`, `MOPS-R2`,
`MOPSC-R1`, `MOPSC-R2`, `MOPSF-R1`, `MOPSF-R2`, `MOPSI-R1`, `MOPSI-R2`) but gives only
six aggregate QC numbers per library. The 16 libraries map onto the five screens of
Table 2 with no library left over, so there is **no sixth genome-scale condition hiding
in the sequencing set**. Classification: a per-guide read count is the numerator and
denominator of the stored ratio, not a phenotype, so even if it were released it would
not be a record.

#### The sixth screen exists and its numbers were never released

The paper's first half is a separate pooled screen that the five stored `screen_id`
values do not cover:

> We performed screening with MOPS medium as the selective condition and LB broth as the control condition throughout a period of ten cell doublings.

against a tiling library of 2,281 guides over 44 genes in two sub-libraries plus 400
controls, with two biological replicates (Supplementary Fig. 1). Its per-guide fitness
and Z score are computed (Fig. 1b uses "all 468 sgRNAs" of the true-positive set) and its
gene fitness is plotted against the genome-scale auxotrophy gene fitness in Supplementary
Fig. 9. **Only si4, the in silico library design, is released.** No table of tiling
fitness exists in any mirrored artifact, so this is a data-absence block, not a schema
block. Hypothesis (untested): the tiling fitness table may exist in the authors' earlier
bioRxiv post, doi 10.1101/129668, which both the paper and the rebuttal cite, but that
preprint is not in the mirror and I did not fetch it.

#### The validation experiments release no numbers

There is a real second experiment set (Supplementary Figs. 22 and 23) with four distinct
readouts, all figure-only: bulk sfGFP fluorescence normalized to OD600 across five
Anderson-promoter dCas9 variants, plus or minus IPTG, at 12 h and 26 h; lycopene titer
normalized to OD600; sucrose-plate growth of the sacB strain; and flow-cytometry sfGFP
distributions for six sgRNAs with an average repression fold per sgRNA. The one
quantitative statement that reaches the text is a ratio, not a table:

> It is also worthy noting that the diversity of sgRNA repression efficiency is observed here (as much as 10-fold, sgRNA_166 vs. sgRNA_414), which is consistent with our conclusions at the functional level

No qPCR or knockdown-efficiency-per-guide measurement exists in this paper. One reviewer
asked for exactly that and the authors declined:

> - Line 97: the authors do not demonstrate here that their positioning actually matters for expression level (would need qPCR, WB etc.)
> Response: we thank the reviewer for these useful suggestions. The manuscript is revised accordingly (Line 90-92).

#### Comparisons against other data

**Rousset 2018, the preprint that became our `rousset2018.py` (ref 35 here). No
duplication, and the paper says why.**

> Among seven potentially false negative essential genes they mentioned in the main text (alsK, bcsB, chpS, entD, mazE, yafN and yefM; alsK, bcsB, and entD were confirmed by knockout; no supplementary data was provided, disabling systematic comparison), six of them are also identified by our work (Supplementary Table 4) except for yafN, while this gene is not annotated as essential by Keio collection data.

The comparison is a seven-gene-name overlap because no data was available to the authors
at the time. It implies no duplication of our Rousset records.

**Cui 2018 is NOT cited by this paper**, so the duplication question has to be answered
empirically. **Measured** against
`$DATA_ROOT/torchcell-raw/cuiCRISPRiScreenColi2018/data/41467_2018_4209_MOESM8_ESM.csv`):

- Wang's si6 holds 55,646 distinct gene-targeting spacers. **8,278 of them appear
  verbatim in Cui's `guide` column** (78,137 distinct guides). A further 523 match
  Cui's reverse complement, which is the opposite-strand guide at the same locus and is
  therefore a different guide, not a duplicate.
- Restricting to Wang's `essentiality` screen, `Quality == "Good"`, with numeric Cui
  values on both strains: **7,836 shared spacers**. Pearson r of Wang `sgRNA fitness`
  against Cui `fit75` = **0.8911** (median absolute difference 0.4929); against Cui
  `fit18` = 0.7927 (median absolute difference 0.8636).

**This is NOT the #760 situation.** #760 found r = 1.0000 and a median absolute
difference of 0.0000 over 54,326 spacers, which is one screen released twice. Here two
labs, two dCas9 cassettes, two induction regimes and two generation counts (Wang 15, Cui
17) give r = 0.89 on a shared seventh of the guides. These are independent replications
and both belong in the store. The real consequence is a **leakage hazard, not a
duplication**: 7,836 guide spacers will carry a Wang `essentiality` record and a Cui
`LC-E75` record of nearly the same biology, so any split that is random over records
puts near-copies of the same measurement on both sides. A split keyed on the 20-mer
spacer, not on the record, is the fix. That is a modeling note, not a loader change.

**Rousset's phage screens are complementary, not duplicated.** **Measured**: 4,727 of
Wang's 55,646 distinct targeting spacers appear in the `target` column of
`pgen.1007749.s014.csv` and `pgen.1007749.s016.csv` (17,220 targets each), the four phage
screens that Rousset now serves after #760. Those carry `log2FC_lambda`, `log2FC_T4`,
`log2FC_186`, a different phenotype from Wang's growth fitness, so the shared guides are
a feature, not a collision.

**The Keio collection appears as a QUANTITATIVE covariate we do not have.** Fig. 5a uses
a per-gene Keio growth number, not just the essential-gene list:

> The size of the scatter is proportional to the $1 / \mathsf { O D } _ { 6 0 0 }$ value of the relevant gene knockout reported with the Keio collection1 .

and Supplementary Fig. 28 thresholds it:

> Supplementary Fig. 28 Selection of $\mathrm { O D } _ { 6 0 0 }$ (from Keio dataset, $4 8 \mathrm { ~ h ~ }$ cultivation in MOPS minimal medium) threshold to identify auxotrophic genes.

with Supplementary Note 1 naming the cutoff: "we hence checked the OD600 distribution of
Keio collection in MOPS medium (Supplementary Fig. 28) and chose $\mathrm { O D } _ { 6 0
0 } { < } 0 . 0 9$ as threshold to define auxotroph." That per-strain OD600 at 48 h in MOPS
is Baba 2006 (ref 1) with the Yamamoto 2009 update (ref 27), and there is **no
`ecoli/baba2006.py` or Keio loader in the repo** (`torchcell/datasets/ecoli/` holds
caglar2017, cui2018, fuhrer2017, goodall2018, gupta2024, lamoureux2023, mutalik2020,
price2018, rapp2026, rousset2018, schmidt2016, shiver2016, tong2020, wang2015, wang2018,
wetmore2015). It is a per-strain growth measurement over the whole single-gene deletion
collection and would be a `BacterialFitnessExperiment` or
`BacterialGeneEssentialityExperiment`. **This is the loadable dataset we do not have that
this paper points at.** Adding it is a curation decision, flagged here, not taken.

**Tn-seq benchmarks we already have.** The ROC comparison is against four classifiers:

> The results indicated that the CRISPRi screening achieved performance generally comparable to that of Tn-seq method with a 16-fold larger library size (area under the curve (AUC)-ROC value: CRISPRi, 0.952; Goodall et al., 0.950), despite moderately but significantly poorer performance in the low false positive rate range. The performance of Tn-seq decreased significantly as the decrease of library size (Wetmore et al. AUC-ROC, 0.878), followed by genetic footprinting strategy (AUC-ROC, 0.821).

Goodall 2018 (ref 29) and Wetmore 2015 (ref 25) both have loaders. Gerdes 2003 (ref 28)
is the genetic-footprinting set and has none; it is a binary gene list rather than a
per-strain measurement, so it is a weak candidate. Both Tn-seq sets measure transposon
insertion abundance, not CRISPRi guide depletion, so none of this duplicates anything.

**Nichols 2011 (ref 39) is the auxotrophy gold standard** in Supplementary Fig. 17:
"True positive rates and false positive rates are calculated using a gold standard set of
essential and nonessential genes by suggested by Nichols et al (Cell 2011)." Nichols 2011
not being in the mirror is already #691 and is already named in #749; nothing new here.

**Peters 2016 (ref 17) is cited for a mechanism, not a data comparison**: "the more
moderate reverse polarity of CRISPRi using our data, which was first described by Peters
et al.17". No measurement is compared.

#### Loadable-now estimate

**Zero.** Nothing released by this paper and not already stored fits an existing
phenotype class and field. Everything unstored is one of: not released (the tiling
screen, the validation readouts, the per-guide read counts), a measured derivative of
what is stored (gene fitness, gene Z, the si14 overlap flag, si7), a design parameter or
annotation (si4, si5, si6, si15, si16, si17, Supplementary Tables 1, 3, 5, 7, 8), or
blocked on the already-recorded significance field (si8 `FDRvalue`, `FPRvalue`).

For sizing the blocked item if the field is ever added: a gene-level dataset would write
**20,049 records**. Basis: 20,158 released rows (4,142 + 4 x 4,004, counted from si8),
minus the 99 rows with `sgRNA number used to calculate the metrics` = 0, which carry the
sentinel triple rather than a measurement, minus 10 rows (`ybfK` and `sokE`, present in
all five sheets, measured) that the loader's existing retired-b-number rule would drop.
The two sets do not intersect, measured.

#### Not checked

- **Supplementary Figs. 1-30 as images.** Only the OCR'd captions are text; the plotted
  values (tiling fitness, Supplementary Fig. 22's repression ratios and lycopene titer,
  Supplementary Fig. 23's distributions, Supplementary Fig. 7's screen workflow) are not
  recoverable from `si1.md`. Images exist under `si/images/si1/`.
- **The bioRxiv preprint doi 10.1101/129668**, cited by the paper and the rebuttal as the
  home of the tiling-screen detail and a cross-feeding growth experiment. Not in the
  mirror, not fetched.
- **BioProject PRJNA450392.** Raw reads, not fetched, and no loader would consume them.
- **The dev LMDB.** The 240,481 record count is quoted from the note's build section, not
  re-measured; nothing was built or run.
- **si2.pdf beyond the OCR text.** The rebuttal embeds figures (gel images, a plasmid
  maintenance check) that OCR did not render; no data table was found in the text.

### roussetGenomewideCRISPRdCas9Screens2018 -- `ecoli/rousset2018.py`

**SI inventory** (`$DATA_ROOT/torchcell-library/roussetGenomewideCRISPRdCas9Screens2018/si/`,
20 files, no OCR markdown for any of them because every PDF-free supplement is a png, csv,
xlsx or docx):

| file | format | what it is | bytes |
|---|---|---|---|
| si1 to si10 | png | S1 to S10 Figs | 77 KB to 2.1 MB |
| si11.csv | csv | S1 Table, per-sgRNA growth screen, 59,246 rows | 7,092,023 |
| si12.csv | csv | S2 Table, gene-level medians of the growth screen, 4,213 rows | 470,001 |
| si13.xlsx | xlsx | S3 Table, 56 rows, one sheet `Feuil1` | 10,676 |
| si14.csv | csv | S4 Table, per-sgRNA log2FC for the three phages, 17,220 rows | 1,989,407 |
| si15.csv | csv | S5 Table, gene-level phage resistance scores, 3,708 rows | 464,151 |
| si16.csv | csv | S6 Table, per-sgRNA log2FC after transduction, 17,220 rows | 1,353,365 |
| si17.csv | csv | S7 Table, gene-level transduction estimates, 3,708 rows | 289,586 |
| si18.docx | docx | S8 Table, cloning and sequencing primers | 14,540 |
| si19.docx | docx | S9 Table, individual sgRNAs used in the paper | 12,831 |
| si20.docx | docx | S10 Table, qPCR primers | 12,881 |

Measured column headers, read directly from the csv first lines:

- si11.csv: `target,position,ori,coding,gene,essential,gene_left,gene_right,gene_ori,log2FC,padj,gamma`
- si12.csv: `gene,essential,gene_ori,gene_left,gene_right,median_coding,median_template,mad_coding,mad_template,coding,template,operon`
- si14.csv: `target,position,ori,gene,essential,gene_left,gene_right,gene_ori,log2FC_lambda,log2FC_T4,log2FC_186`
- si15.csv: `gene,essential,Resistance_Score_lambda,mad_lambda,Resistance_Score_T4,mad_T4,Resistance_Score_186,mad_186,sgRNAs,operon`
- si16.csv: `target,pos,ori,gene,essential,gene_right,gene_left,gene_ori,log2FC`
- si17.csv: `gene,essential,operon,sgRNAs,estimate,pvalue,FDR`

**What the loader stores.** `CrispriScreenRousset2018Dataset`, `EnvironmentResponsePhenotype`
with `measurement_type=MeasurementType.log2_ratio`, one record per (sgRNA, screen) over the
four phage-derived screens from S4 and S6 Tables. **Measured: 68,436 entries** in the dev
LMDB at `$DATA_ROOT/data/torchcell/ecoli_crispri_rousset2018/processed/lmdb`, which matches
the arithmetic the loader docstring states at `torchcell/datasets/ecoli/rousset2018.py:136`:
"rules 5 to 7 remove 444 of the 68,880 phage-derived cells, leaving 68,436 records over
3,671 genes".

**Already recorded as not stored.**

- **S2, S5 and S7 Tables as derived gene-level summaries**, and deliberately not mirrored:
  `notes/torchcell.datasets.ecoli.rousset2018.md:21` ("S2, S5 and S7 Tables are gene-level
  medians and model estimates derived from them") and `:236` ("`si_expected` records what
  was deliberately NOT mirrored: S2, S5 and S7 Tables (derived gene-level summaries) and
  ENA BioProject PRJEB28256 (the raw reads)").
- **`padj` and `gamma` on S1 Table**: `rousset2018.py:913` in `_uncertainty_gap`, and
  `notes/...rousset2018.md:147` and `:154`. `gamma` is undefined in both mirrored papers
  and was measured, not sourced; `padj` has no schema slot.
- **DESeq2 `lfcSE`**: not a released column, typed as a gap on
  `environment_response_uncertainty` (`rousset2018.py:905`).
- **The template-strand guides (30,811 S1 rows)**: dropped with the argument in
  `rousset2018.py:109` and `notes/...rousset2018.md:206`.
- **The growth screen as a whole**: issue #760, and `torchcell/datasets/ecoli/__init__.py`
  ("Its fifth released screen, growth over 17 generations, is Cui 2018's screen released
  again (r 1.0000 over the 54,326 spacers they share), so it is accounted for in the
  retention ledger rather than stored twice").
- **The aTc-into-media move**: issue #756.

**Released quantities and their classification.**

| released quantity | source file + column | stored | classification | evidence |
|---|---|---|---|---|
| per-sgRNA phage log2FC, 3 phages | si14.csv `log2FC_lambda`, `log2FC_T4`, `log2FC_186` | yes | stored | S4 Table caption: "computed fold change for each phage (log2F-C_lambda, log2FC_T4 and log2FC_186)" |
| per-sgRNA transduction log2FC | si16.csv `log2FC` | yes | stored | S6 Table caption: "computed fold change after transduction (log2FC)" |
| per-sgRNA growth log2FC | si11.csv `log2FC` | no | not a phenotype here: the same measurement Cui 2018 serves | #760; `rousset2018.py:120` rule `growth_screen_measurement_is_served_by_cui2018` |
| per-sgRNA adjusted p-value | si11.csv `padj` | no | blocked: no `p_value`/`fdr` field on `EnvironmentResponsePhenotype`. **This gap is NOT in #749, #753, #756, #758 or #760** | already recorded at `notes/...rousset2018.md:154`: "a `p_value`/`fdr` field on `EnvironmentResponsePhenotype` would let three of the fifty bacterial rows keep a released statistic they currently drop" |
| per-sgRNA `gamma` | si11.csv `gamma` | no | not a phenotype: meaning undefined in both mirrored papers | already recorded, `notes/...rousset2018.md:147` |
| gene-level median growth log2FC per strand + MAD + guide count | si12.csv `median_coding`, `median_template`, `mad_coding`, `mad_template`, `coding`, `template` | no | not a phenotype: derived summary of a screen we do not store | already recorded, `notes/...rousset2018.md:21` |
| gene-level phage resistance score + MAD + guide count | si15.csv `Resistance_Score_lambda`, `mad_lambda`, ... , `sgRNAs` | no | not a phenotype: derived summary of the stored guide-level values | S5 Table caption: "median log2FC (Resistance_Score_lambda, ...) and median absolute deviation of guides targeting the coding stand for each phage" |
| gene-level transduction estimate, p-value, FDR | si17.csv `estimate`, `pvalue`, `FDR` | no | not a phenotype: derived summary. The caption makes the derivation explicit, so it is not an independent measurement | S7 Table caption: "For each gene, a linear model was built to take the effect on cell fitness into account (see Methods). The regression coefficient of the Boolean parameter termed as ``estimate'' is used for gene ranking." |
| essential genes not essential in the screen, with an explanation and a reference | si13.xlsx sheet `Feuil1`, columns `Gene`, `Explanation`, `Reference` | no | not a phenotype: annotation and literature commentary, carries no measured value | read from si13.xlsx; 56 rows, first data row `('chpS', 'TA system', 'Zhang et al. (2005)')` |
| cloning, sequencing and qPCR primer sequences; individual sgRNA sequences | si18.docx, si19.docx, si20.docx | no | not a phenotype: design parameters | S8 Table "List of primers used for cloning and sequencing", S9 "List of individual sgRNAs used in this study", S10 "List of primers used for qPCR" |
| RT-qPCR relative `lexA` and `rho` expression, 3 biological x 2 technical replicates | S6 Fig (si6.png) only | no | not a phenotype: figure-only, no released table | S6 Fig caption: "RT-qPCR results are shown for 3 biological replicates and 2 technical replicates." |
| normalized sfGFP fluorescence per OD600 after 12 h for the lexA and rho promoter fusions, n=3 | S6 Fig (si6.png) only | no | not a phenotype: figure-only | S6 Fig caption: "Raw GFP fluorescence after $1 2 \mathrm { h }$ was normalized by OD600 ... Bar plot shows mean $\pm$ standard deviation $\left( \mathbf { n } = 3 \right)$" |
| phage and cosmid titer by plaque assay and transduction, n=9 | S9 Fig (si9.png) only | no | not a phenotype: figure-only | S9 Fig caption: "the relative concentrations of phage and cosmid were measured by plaque assay and by transduction into strain MG1655::λ respectively ... mean $\pm$ standard deviation $\mathbf { \rho } ( \mathbf { n } = 9 \mathbf { \rho } _ { , }$" |

The whole phage panel is stored: the paper challenged three phages (lambda, T4, 186cIts)
plus the lambda transduction assay, which is exactly the four screens the loader serves.
There is no fifth phage.

**Comparisons against other data.**

- The duplication against Cui 2018 is settled and acted on (#760). Worth one correction to
  the issue's arithmetic, measured from the loader's own ledger: #760 priced option 1 as
  losing "the 4,920 spacers' LC-E75 measurements", but at the grain the loader stores
  (coding-strand, in-gene gene perturbations) the loss is **1,687 records**, not 4,920. The
  other 3,233 Rousset-only spacers are template-strand or intergenic and would have been
  dropped by rules 1 and 2 regardless. Evidence: `rousset2018.py:123` rule 4
  `growth_screen_guide_is_below_cui2018_read_floor` (1,687 guides), and the identity
  `5,063 + 30,811 + 21,685 + 1,687 = 59,246`.
- The paper benchmarks its gene ranking against the **EcoGene** essentiality annotation
  ("genes with the most depleted sgRNAs included a large majority of genes annotated as
  essential in the EcoGene database [34]") and uses the **STRING** database for S2 Fig.
  Neither is a dataset of measurements we would load; both are annotation sources.
- S4 Fig uses "a recent transcription start site dataset" for predicted internal promoters.
  That is a third-party annotation, not a phenotype release, and the paper names it only by
  reference.

**Loadable-now estimate.** **None.** Every released per-record quantity Rousset carries is
either stored, served by Cui 2018, a derived gene-level summary of a stored or subsumed
screen, a design parameter, or blocked on the missing `p_value`/`fdr` field. This is the
one paper in the set where the audit's answer is that nothing was left on the table.

**Documentation finding (not a phenotype finding).** The dendron note is stale by one
landed change. Its retention ledger at `notes/torchcell.datasets.ecoli.rousset2018.md:213`
still reads "128,126 - 36,517 = **91,609 records**" and its open item at `:268` still says
the Cui overlap is "Not resolved here; flagged so it is not discovered after both" land.
Both were superseded by #760 and by commit `c80b0088d`, after which the loader stores
68,436. The note has no dated section recording the change.

**Not checked.**

- si1.png through si10.png: binary figure images, not read. Their captions in `paper.md`
  were read in full, which is how the three figure-only quantities above were enumerated.
- ENA BioProject PRJEB28256 (the raw sequencing reads), deliberately not mirrored.
- The analysis code at `gitlab.pasteur.fr/dbikard/dCas9_genome_wide_screen`, which the note
  names as the thing that would define `gamma`. Not fetched; this audit reads only the
  mirror.

## 2026.10.07 - Caglar 2017 doubling time: rank 15's verdict holds, gap 1's first half does not

Rank 15 above and the "Caglar 2017" section of
`[[plan.bacteria-si-phenotype-audit-pputida]]` disagreed about whether Caglar's Table S5
doubling times are loadable. **Rank 15's verdict is correct and the P. putida note is
wrong**; that note now carries the full measurement and a dated correction. Two details
here need fixing, and three blockers were found that rank 15 does not list. The original
text above is left in place.

Everything below is measured by
`experiments/036-dataset-fixes-before-kg-build/scripts/caglar2017_doubling_time_loadability.py`
(results under the same experiment's `results/`), which sha256-verifies `si/si6.csv`
(`76411acc...`) and `si/si2.csv` (`1486290b...`) against the library mirror manifest.

**The 19-versus-55 count was never a disagreement.** `si6.csv` holds 55 data rows over 19
conditions; rank 15 counted conditions, the P. putida note counted per-replicate rows. Both
are right. Per-condition replicate counts are 3 for 17 conditions and 2 for exactly
`Gluconate.tab` and `Lactate.tab`, verifying the loader's sourced
`DOUBLING_TIME_REPLICATES` quote.

**Gap 9 is confirmed, and is stronger than stated.** 0 of 55 rows have a symmetric
interval; the upper-to-lower half-width ratio runs -26.3920 to 10.1066 with median 1.3187,
and `Glycerol.tab` replicate 1 has `95p` = **-1027.769034**, a negative doubling-time upper
bound, so that row has no half-width to give. `UncertaintyType.ci95` accepts either side
silently: the upper half-width derives a standard error of **-565.6845**. The repo already
holds the lossless pattern gap 9 asks for, on a different phenotype --
`FluxPhenotype.net_flux_lower` / `net_flux_upper` / `confidence_level` with
`label_statistic_name = None`, whose docstring states the principle verbatim: "a two-sided
confidence bound is not a single number, and naming one of the two bounds as "the"
statistic would misreport it."

**Gap 1's first half is wrong: the enum member exists.** `MeasurementType.growth_rate` is
documented in `torchcell/datamodels/schema.py` as "``growth_rate``: absolute or normalized
growth rate / doubling time", so `MeasurementType` is not missing a member for an absolute
growth RATE. What gap 1 gets right is its second half, and that is the whole of the
blocker: the environment-response verifier's L3 `reference_zero` requires the reference
record's `environment_response` to be 0 for a numeric readout, and
`EnvironmentResponsePhenotype`'s own validator forbids a `None` response for a
non-categorical `measurement_type`, so the reference must carry a number and that number
must be 0. Caglar's base-condition doubling time is 53.25 min. Measured: the absolute form
fails `reference_zero` with max|v| = 53.3 at both the 55-row and the 19-condition grain.
Gap 1 should be restated as **relief from `reference_zero` for an absolute readout** plus,
separately, a `MeasurementType` member for optical density, which genuinely has none. That
re-split does not change which papers gap 1 blocks.

**Three blockers rank 15 does not list**, all measured on real records:

| form | records | `pair_uniqueness` | `environment_perturbed` | `reference_zero` |
|---|---|---|---|---|
| absolute, per replicate | 55 | FAIL, 39 duplicates | FAIL, 9 | FAIL |
| absolute, per condition | 19 | FAIL, 3 duplicates | FAIL, 3 | FAIL |
| log2 ratio, all 19, `screen_id` set | 19 | PASS | FAIL, 3 | PASS |
| log2 ratio, in-experiment base only | **11** | PASS | PASS | PASS |

1. Three of the 19 conditions ARE the base condition (glucose, 0.8 mM Mg2+, 5 mM Na+) run
   in three separate experiments, so they carry no environmental edit and collide on the
   condition signature.
2. `MgSO4_000.080_mM` and `MgSO4-2_000.080_mM` are the same 0.08 mM Mg2+ condition in two
   experiments. `screen_id` resolves this honestly, because Table S1's own `experiment`
   column names the run.
3. The `MgSO4_stress_low` series has NO base-condition row in `si6.csv` (no 0.8 mM Mg2+
   curve under the `MgSO4-2` prefix), so 5 conditions and 15 of the 55 rows have no
   in-experiment reference. The three released base measurements are 53.2538, 61.9140 and
   58.3515 min, a 0.2174 log2 spread, so borrowing a base from another experiment is not a
   neutral choice.

**Verdict: nothing is loadable, and the fix forces a full rebuild.** The only verifiable
form stores 11 derived log2 ratios the paper never released, drops 8 of 19 conditions and
gaps every uncertainty, so it is worse than storing nothing. Both missing capabilities
would add fields to `EnvironmentResponsePhenotype`, which moves the schema closure of every
served environment-response dataset and so requires a full KG rebuild rather than an
incremental admission. Filed as #776. Rank 15's record count should read **0 loadable,
19 conditions / 55 rows blocked**.

**One further finding about rank 11 and rank 13's shape.** Rank 11 (Lamoureux per-sample
growth rate) and rank 13 (Schmidt Table S23) are blocked by the same `reference_zero` rule
and would hit the same "is a matched reference released for every condition?" question that
sinks the Caglar ratio form. Worth measuring per paper before either is called storable as
a ratio.

## 2026.10.08 - Rank 3 landed, and two corrections to this audit

Rank 3 (Lamoureux 2023 Public K-12) is loaded as its own dataset. Full record in
[[torchcell.datasets.ecoli.lamoureux2023_public_k12]]; the two corrections belong here,
because both revise statements above.

**Correction 1: the Public K-12 log2[TPM] matrix is not released.** The "Not checked"
section above says it is "presumably inside `k12_modulome.json.gz`, since the tree shows no
`k12_modulome/log_tpm*.csv`". Checked: the archive has no such member, the live
`SBRG/precise1k` tree at HEAD has none either, and all three packaged `IcaData` objects
(`k12_modulome.json.gz`, `k12_only_p1k_ctrl.json.gz`, `k12_only_proj_ref.json.gz`) carry
`X: null` and `log_tpm: null`, as do `precise1k.json.gz` and `precise.json.gz`. So the
paper's Data Availability sentence overstates what was deposited: the counts, the MultiQC
table and the metadata are there and the expression matrix is not. The loader therefore
computes TPM from the released counts and the release's own gene spans (the Caglar 2017
convention) under a distinct `measurement_type`, `rnaseq_tpm_from_released_counts`, because
the derivation does NOT reproduce the paper's own released TPM where both are published:
on PRECISE-1K the maximum absolute difference is 9.48 log2 units and no sample agrees to
within 0.01 across all genes.

**Correction 2: rank 3's record count is 1,675 ROWS, not 1,675 records.** The table above
reads "**1,675** | measured: 1,675 non-`p1k_` rows", and that row count is right. The
dataset built from them holds **240 records**, because the same genotype and environment
discipline the PRECISE-1K arm applies drops 1,435: 669 for a strain other than MG1655 (the
curation deliberately kept 15 K-12 substrains, and the expression is keyed by MG1655
b-numbers), 175 plasmid-borne constructs, 250 for a base medium with no `MEDIA_LIBRARY`
key, 164 non-batch cultures, and nine smaller rules. The audit's number was an upper bound
on the arm, not a projection through the loader, and it did not say so.

**Rank 19 split in two.** `aerobicity` is loaded: the `Electron Acceptor` column PRECISE-1K
reads is blank on every public row, and the `aerobicity` column replaces it (`aerobic`
1,135, blank 250, `O2` 206, `aerobic(30% DO)` 52, `anaerobic` 17, `transition` 15; the last
two are named drop rules, since `Environment.aerobicity` holds neither a dissolved-oxygen
setpoint nor a regime that changes). `time` is NOT loaded, exactly as this audit warned: it
has no unit in its header, neither `paper.md` nor `si/si1.md` mentions the column, and its
13 distinct values on the kept rows mix bare numbers (`12`, `0.25`, `0.08`) with an H:MM:SS
clock (`12:00:00`, `0:00:30`). `duration_hours` is a typed `not_reported_by_primary` gap on
every record and the note records the measurement.

**The dedup policy rank 3 asked for, measured and asserted.** The record identity is the SRA
experiment accession: the 1,675 rows carry 1,675 distinct experiment and 1,675 distinct run
accessions. `BioSample` is NOT the key, and the release proves it: the rows map to 1,568
distinct BioSamples, 38 of which carry 2 to 12 rows apiece (145 rows), and 7 of those span
several conditions or strains under one BioSample. The value-level rule is profile identity,
and both measurements are zero: no public count column repeats another, and none is
identical to any of the 1,055 `data/precise1k/counts.csv` columns the PRECISE-1K dataset
serves. Against our own store, `PRJNA645443` is the one overlapping identifier (12
`phage_resist` BW25113 RNA-seq samples here; `PhageRbTnseqMutalik2020Dataset`'s manifest
records it as holding that paper's raw reads, and that dataset serves RB-TnSeq fitness), and
none of the 38 PMIDs or 30 GEO series matches any other *E. coli* or *P. putida* loader.
That is the measured negative the #760 pattern asked for.
