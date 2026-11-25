---
id: oxqeikyz9zitxmnuerglb4y
title: Hale2024
desc: ''
updated: 1791716076096
created: 1791716076096
---

## 2026.10.11 - Scoping (work in progress)

Hale 2024 (Nat Commun 15:4234, doi:10.1038/s41467-024-48626-1, citation key
`haleGenomescaleAnalysisInteractions2024`): CRISPRi guides integrated into barcoded
BYx3S segregants, relative fitness per double-barcode lineage under ATC induction (two
replicate assays) and uninduced (one control assay). The cross is BY4716 x 322134S
(3S), not BY x RM. The results are in the next section.

## 2026.10.11 - Loader, deposits and verification

`torchcell/datasets/scerevisiae/hale2024.py`, class `CrispriSegregantHale2024Dataset`,
dev store `$DATA_ROOT/data/torchcell/crispri_segregant_hale2024`. First dataset whose
genotype is a segregant background plus an engineered perturbation, held by the new
`PerturbedSegregantGenotype` ([[torchcell.datamodels.eqtl-data-model]] design kept: the
background is the inferred mosaic, the guide the designed edit).

### What one record is

One double-barcode lineage (`<geno_bc>_<guide_bc>`, the paper's own lineage key) in one
fitness assay. A lineage is one independently integrated transformant of a (segregant,
guide) design, so lineages of one design are biological replicates; nothing is averaged
that the source did not average.

| field | value | source |
|---|---|---|
| genotype.background | HMM hard-call mosaic of Supplemental Data 12, run-length encoded per chromosome over called markers, `chrmt` = code 17 | SD12; Matsui 2022 Methods |
| genotype.background_role | `segregant`, or `parent_1` for `x_BY_BYx` | "into each segregant and the BY parent" |
| genotype.perturbations | one `CrisprInterferencePerturbation`: released `gene_orf` through the shared resolver, the genome's standard name, `dCas9-Mxi1`, the 20-nt guide | SD6-8; Results |
| environment | `SC_URA`, 21 C, shaken, 10 generations; ATC1 and ATC2 add anhydrotetracycline 0.25 ug/mL | Methods 'Fitness assay' |
| phenotype | released `fitness`, `MeasurementType.relative_fitness`, pooled barcode assay; uncertainty = released `error` as `standard_error`; n = 1 lineage, `biological_replicate`; `screen_id` = ATC1/ATC2/CON | SD6-8; `fitseq.py` |
| reference | 0.0 = the mean of all CON lineages, one reference per assay | Methods 'Fitness estimation' |

### Counts (measured on the built store and the release)

- Records: 8,692,635 = ATC1 2,923,436 + ATC2 2,857,376 + CON 2,911,823.
- Released rows: 2,959,500 + 2,892,633 + 2,947,298 = 8,799,431; the difference is the
  106,796 control-gRNA rows (36,064 + 35,257 + 35,475), ledgered in
  `preprocess/dropped_records.json` (rule `control_guide_targets_no_gene`, 83 guides).
  0 rows dropped for an unresolved ORF (all 1,721 resolve to current R64-4-1 genes).
- Lineages: 2,722,120 in all three assays, 3,054,459 in any.
- Designs (background, guide): 895,494 across the three assays; 7,963 gene-targeting
  guides; 1,721 target genes (the store's gene set); 170 backgrounds (169 segregants and
  the BY parent) over 224 segregant barcodes. The schedule's 1.4e6 counted a different
  unit; the release is per lineage.
- Mosaics: 182 Supplemental Data 1 columns encoded, median 69 blocks (min 17 for
  `x_BY_BYx`, one per chromosome; max 94); 12 of the 182 never carry a lineage and are
  not served.

### Sourcing (verbatim, sha256-pinned; audited at L3)

- Temperature and aeration, paper.md: "assays were grown for $2 4 \mathrm { h }$ at
  $2 1 ^ { \circ } \mathrm { C }$ with shaking." (21 C confirmed in the PDF text layer.)
- Induction: "ATC was added to experimental flasks (assays) at a final concentration of
  $2 5 0 \mathrm { n g / m L }$ ." and "ATC was not added to the control assay."
- Medium: "All assays were done in synthetic complete medium containing glucose as the
  fermentable carbon source and lacking uracil, ..."
- Duration: "We performed the assays for 10 generations in serial batch culture, ..."
- Normalization: "This normalization was performed using 83 control gRNAs ..." and "Six
  of the 83 control gRNAs were removed at this step, ..." and "... such that the mean
  fitness of all lineages in CON was exactly zero."
- The statistic: "PyFitSeq estimates the fitnesses of lineages relative to the mean
  fitness of the whole population." The error's kind is settled by `fitseq.py` (darachm
  commit 340194a, the repository the released README names, last commit 2021-12-28):
  `estimation_error = 1/np.sqrt(second_derivative)` of `calculate_likelihood_of_fitness_
  vector`, which returns the negative log-likelihood, so the error is the
  observed-information standard error. Time axis: `EIGHT_fitseq.job` "-t 0 2 4 6 10".
- Segregant calls (deferred to Matsui 2022, ref. 35, mirrored): the HMM correction, the
  30 kb chromosome-end omission and the collapse of adjacent non-recombining SNPs. Parent
  genotypes verbatim from Matsui: "BY fcy1Δ flo8Δ flo11Δ ura3Δ ho YBR209W::pGal1-Cre -
  Lox71 - Lox2272/71" and the same for 3S.
- Allele coding, si3.md: "with 0 indicating a BY allele and 1 indicating a 3S allele at
  that locus." BY as S288C: "The S288C reference differs from the BY strain by only
  roughly 125 SNPs."
- Assay labels: the released `analysis_code.R` says "Note that the names of ATC1 and ATC2
  have been switched in the main text for clarity. Supplemental Data 6 is labelled ATC1,
  but this same file is referred to as ATC2 in this script." The store follows the SI
  description and the main text: SD6 = ATC1 (primary), SD7 = ATC2 (replicate), SD8 = CON.

### Measured properties of the release

- The released `fitness` is `raw_fitness` minus one constant per assay; the constant
  recomputes from the 77 retained control gRNAs and the CON mean to within 5.1e-15
  (ATC1 0.0717783, ATC2 0.0582968, CON 0.0489100). L2 re-derives it on every build.
- Supplemental Data 2 (5,717 markers, the paper's analysis matrix) is a strict subset of
  Supplemental Data 12 (14,395 markers, the whole panel) and agrees at all 1,039,423
  called cells of the 182 shared columns, so the mosaics use SD12's finer boundaries.
- Supplemental Data 13 (the 3S variant table): 43,558 SNVs, 270 single-base deletions,
  771 single-base insertions; every SNV and deletion carries the SGD R64-4-1 base, which
  settles the coordinate system (Hale mapped to R64-2-1; the sequence is the same).
- Released tables already satisfy two of the paper's filters (T0 >= 5 normalized reads,
  minimum 5.86; >= 100 guides per segregant, minimum 132); 1 ATC1 and 2 CON guides sit in
  a single segregant. The log-likelihood rule cannot be re-checked on post-filter data.
- 14,374 of the 14,395 Supplemental Data 12 marker positions are rows of the 3S variant
  table; the other 21 are not (issue #1013 records it beside the missing assembly).
- Supplemental Data 4 barcode counts do not equal the released rows per segregant (3 of
  170 equal), so they count barcodes before the fitness filters and are not consumed.
- Supplemental Data 1's `genotype_data` column is undocumented and unused by the
  released code; it is not consumed.

### Decisions taken by recommendation (owner to confirm)

1. Per lineage, per assay, not per (segregant, guide): the release is per lineage, the
   per-design means exist only inside the paper's mixed models.
2. The 3S parent has no public assembly (NCBI assembly search for 322134S: 0 hits; not in
   Peter 2018 Table S1 nor SGRP2), so `SegregantParent(3S).assembly_ref` points at the
   genomes-tier set `hale2024_322134S_variants` (Supplemental Data 13). It is a variant
   table, not an assembly: larger indels, structural variants and 3S-specific sequence
   are not represented. Alternatives: assemble 3S from SRA reads, or a variant-panel
   parent class.
3. The BY parent is a one-parent mosaic with `background_role="parent_1"`, validated to
   carry parent 1 at every block, rather than a plain `Genotype` on S288C, so all 170
   backgrounds share one representation.
4. Guides near a BY/3S polymorphism (74 overlap one, paper) are served: the paper drops
   them from its analyses, but the lineage was measured and the mosaic says which allele
   the binding site carries.
5. Not served: Supplemental Data 9-11 (gRNA mixed-model effects, deviation values, QTL
   intervals) are model outputs, not measurements; the per-time-point normalized counts
   and PyFitSeq log-likelihoods have no phenotype field and stay in the raw mirror.
6. Perturbations and the publication are always interned: inline they kept every record
   at 2,299 bytes, over LMDB's half-page node limit, which put all 8,692,635 records on
   overflow pages (34 GB). At 1,824 bytes there are 0 overflow pages, but the store is
   still 31 GB with 7,910,297 leaf pages for 8,692,635 entries: `str(idx)` keys arrive
   out of lexicographic order, so page splits leave about one record per 4 KB page. A
   shared-writer property, not changed here.
7. The candidate row named neither parent and gave Kruglyak L as last author (it is
   Ehrenreich IM); both corrected in `build_candidate_datasets_table.py`. The rendered
   notes-tex tables are not regenerated in this branch. A `322134S` host token resolves
   to the variant set, which `READABLE_ASSEMBLY_SETS` now lists because this loader and
   its verifier read it.

### Deposits

- Raw mirror `$DATA_ROOT/torchcell-raw/haleGenomescaleAnalysisInteractions2024/`:
  `data/` = Supplemental Data 1, 2, 3, 6, 7, 8, 12 (copied from the literature mirror
  `si/`, PMC-cloud retrieval recorded; byte-identical to Figshare
  10.6084/m9.figshare.24578995.v1, md5 per file matched), `code/` = five members of the
  Figshare `code.zip` (`zip_member`, container sha256 dc8dac03...) and `fitseq.py` at
  340194a. Retrievals re-run 2026-10-11: PMC object, Figshare zip member and GitHub file
  all reproduce their sha256.
- Genomes tier `$DATA_ROOT/torchcell-genomes/hale2024_322134S_variants/`
  (`supplemental_data13.txt`, sha256 5a698776...; registry constant
  `HALE2024_3S_VARIANTS`, role `variants`). tc-data serves it with no change: the
  `/genomes` route lists every manifested set; Radiant needs the rsync the seed hint
  prints.
- No RM11-1a set was deposited: this cross has no RM parent.

### Schema impact

`scripts/schema_impact_check.py --base origin/main`: 8 changed symbols (new
`PerturbedSegregantGenotype`, `SegregantPerturbationExperiment`,
`SegregantPerturbationExperimentReference`; `MeasurementType` gained
`relative_fitness`; the four unions and maps gained members); 44 datasets stale via
`MeasurementType` / `ExperimentType`, 0 breaking. KG 4.0 is a full rebuild, so no
incremental admission is attempted.

### Build and verification

Built and verified in one slurm job (3918, 4 CPUs, 40 G) at `9ac9f4754` with
`python -m torchcell.database.build_dataset_lmdb --dataset CrispriSegregantHale2024Dataset
--retire-existing --verify`: build 364 s, verification about 1 h 33 min, peak RSS 36.4 GB,
wall 1:39:47. Report: `preprocess/verification_report.json`; all 25 levels pass.

| level | check | result |
|---|---|---|
| L0 | structural | 8,692,635 / 8,692,635 validate as `SegregantPerturbationExperiment` |
| L1 | count | 8,692,635 (expected 8,692,635) |
| L1 | lineage_uniqueness | 0 duplicate, 0 unreleased lineages, 0 released gene rows unserved |
| L1 | screen_partition | ATC1 2,923,436 / ATC2 2,857,376 / CON 2,911,823 = the release |
| L1 | ledger_balance | served + ledgered - released = 0 per assay; 83 control gRNAs |
| L1 | provenance_gaps | 8,692,635 documented gaps (one `duration_hours` per record); 0 deferred |
| L1 | canonical_gene_names | 1,721 names, one spelling each, all current |
| L2 | uncertainty_sanity | 8,692,635 labeled, no zero dispersion |
| L2 | value_fidelity | every fitness, error, background, guide and ORF equals its cell |
| L2 | normalization_reproduced | offsets recomputed from 77 control gRNAs within 3.7e-16 |
| L2 | mosaic_round_trip | 170 / 170 backgrounds re-expand to their SD12 column |
| L3 | compound_identity, media_compound_identity, media_membership | pass; one medium (SC-Ura) |
| L3 | reference_zero | 0.0 for ATC1, ATC2, CON |
| L3 | environment_documented | ATC1 and ATC2 anhydrotetracycline 0.25 ug/mL, CON none |
| L3 | measurement_typed | relative_fitness / pooled barcode / standard_error / n = 1 |
| L3 | sourced_values | 28 / 28 quotes present in their pinned sources |
| L4 | gene_containment_sgd, current_genome_genes | 1,721 / 1,721 |
| L4 | matrix_subset_agreement | 5,717 / 5,717 SD2 markers in SD12; 0 of 1,039,423 cells disagree |
| L4 | variant_reference | 43,828 / 43,828 SNV and deletion rows carry the R64-4-1 base |
| L4 | guide_targets | 7,963 / 7,963 served guides carry their SD3 ORF |
| L4 | parents_pinned | BY -> `sgd_S288C_R64-4-1_20230830`, 3S -> `hale2024_322134S_variants` |

L0 first ran on the `ExperimentType` union: 10.3 ms per record, about 25 h for the store
(measured on 20,000), so the job was cancelled and L0 now validates each record as the
class its `experiment_type` names, with each distinct background and environment
validated once (193 us per record). The two superseded store copies were moved to
`/scratch/projects/torchcell-deprecated` with `scripts/deprecate.sh`.

### Not done here

- The candidate ledger (`notes/torchcell.candidates.ledger.md`) already has its
  2026.10.11 section and `ledger` appends one per day, so the Hale row appears at the
  next regeneration; the verdict JSON is in `database/candidates/`.
- The rendered notes-tex schedule tables still show the old Hale row text (author, no
  parents); only the table script was corrected.
- Incremental admission is not attempted: KG 4.0 is a full rebuild.

## 2026.10.12 - Released gene_orf and gene_name stored verbatim; no resolver

Owner rule of 2026-10-12, applied through the genome-gating sweep ([[database.genome-gating-sweep.2026.10.12]]: the `target_orf_is_not_a_current_genome_gene` rule, 0 records, and the resolver rewrite row) on branch `fix/genome-gate-reversals-yeast-2`.

- `LINEAGE_COLUMNS` gains `gene_name`. Each `CrisprInterferencePerturbation` stores the lineage table's `gene_orf` as `systematic_gene_name` and its `gene_name` as `perturbed_gene_name`, both verbatim; for the 162,753 gene rows (of 2,923,436 in ATC1; the same ORFs in ATC2 and CON) whose `gene_name` cell is `NA`, the ORF stands in. Before, the name was the genome's canonical standard name and the ORF went through the shared resolver.
- `orf_resolver`, the `smith2006.canonical_common_names` import and the `_resolvers` method are gone; the loader reads no genome. `genome` stays an optional constructor argument because `verify_build` (the runner's `verify_segregant_perturbation_dataset`) still passes `resolve_gene_name` to the shared record rules for the L4 containment row. `verify_segregant_perturbation_streaming` no longer takes `resolve_orf`: the L2 re-read compares the stored ORF with the released `gene_orf` cell byte for byte.
- `preprocess/dropped_records.json` keeps one rule, `control_guide_targets_no_gene` (106,796 records: ATC1 36,064, ATC2 35,257, CON 35,475), the source's own control guides with `NA` under the ORF column.
- Record count unchanged: 8,692,635. Every record's `perturbed_gene_name` can change (released name instead of the genome's), so the store is rebuilt under slurm with `--retire-existing --verify` from a `git clone --shared` snapshot of the branch rebased onto `main` at `f85df69af5` (#1069 landed: genome rules report, never gate), slurm array 4451; the L0-L4 table is appended below when the job finishes (the previous build took 1 h 40 min).

## 2026.11.24 - Per-replicate review: records already one per lineage; control lineages need a genotype shape

Review under the per-replicate policy (#1178), part of #1171 (control replicates collapsed). Branch `fix/per-replicate-hale2024`. No record changes: the store stays 8,692,635.

### What a replicate is (verbatim, `paper.md` sha256 `3ff16870...`, `si/si3.md` sha256 `f31850db...`)

- A record is one double-barcode lineage in one assay: "Each row represents a unique double barcode lineage." (si3.md, Supplemental Data 6). A lineage is one integrated transformant, so it is a biological replicate, and the release is already one value per lineage. Nothing is averaged.
- Three levels of replication, all stored apart: "Roughly 10 distinct barcodes were generated for each gRNA on average" (Results, library construction); "To enable tests for reproducibility, 30 segregants were represented by 2–3 distinct barcodes." (Results); and the assay flasks: "The T0 pool was split into three separate fitness assays: two replicate experimental assays where the gRNA expression was induced with ATC (ATC1 and ATC2, experimental conditions) and one control assay where no ATC was added and gRNA expression was not induced (CON, control condition)." plus "The ATC1, ATC2, and CON assays were grown, harvested, and diluted in parallel, under identical growth conditions." (Methods, Fitness assay). ATC1 and ATC2 are separate records with one `replicate_id` (the double barcode) and their own `screen_id`.
- Measured on the raw mirror (gene rows): (segregant, guide) designs per assay ATC1 881,342 / ATC2 871,778 / CON 877,379, of which 599,300 / 589,006 / 595,950 have more than one lineage, each lineage its own record. No double barcode repeats within an assay (0 of 2,959,500 / 2,892,633 / 2,947,298 rows).
- The `error` column is PyFitSeq's estimation error of one lineage's fitness, a statistic the source releases per lineage, not a dispersion over replicates; it stays on the n = 1 record (policy rule 4).
- No sample withdrawn by the authors is released and then dropped by the loader. The authors' own filters (log-likelihood, T0 reads, <100 guides per segregant, <2 segregants per guide; Methods, Fitness estimation) act before the release.

### Control-gRNA lineages: released replicates, not records, owner decision

- 106,796 lineages (ATC1 36,064, ATC2 35,257, CON 35,475) of 83 control guides, every one a per-lineage released fitness with its own double barcode, over all 170 backgrounds (measured). They are the #1171 item-1 pattern.
- The plan called them "non-targeting". They are not: "91 unique gRNAs targeting intergenic and noncoding regions rather than promoters were also included with the intent of serving as controls." Supplemental Data 3 gives each its target chromosome, strand and genome positions (`chrom`, `strand`, `start_gp`, `end_gp`); all 83 in the lineage tables have them, and the essential-collection control rows of SD3 number exactly 91 (measured).
- The existing schema cannot state their genotype. `PerturbedSegregantGenotype` refuses an empty perturbation list, `SegregantPerturbationExperiment` takes only that genotype, and `CrisprInterferencePerturbation` needs a host systematic gene name. An empty genotype on the segregant (AGENT_NOTES rule 10) applies only when the paper prints no control genotype; here the paper prints the guide and its locus, so an empty genotype would erase it, and it would also hide the integrated dCas9-Mxi1 / URA3 construct that SC-Ura selects for.
- Proposed shape (owner decision, same gap as the Carruthers 2025 controls, #1200): a locus-keyed CRISPRi leaf in the genotype union, e.g. `CrisprInterferenceLocusPerturbation(crispr: CrisprConstruct, chromosome, strand, start, end, assembly_set, target_class: Literal["intergenic", "noncoding"])`, sorted beside the gene-keyed leaves; or a present-construct field on the genotype. With it, the 106,796 lineages become records with `replicate_id` = double barcode, 8,692,635 -> 8,799,431.
- The stored reference 0.0 is the paper's own statistic ("such that the mean fitness of all lineages in CON was exactly zero"): the CON mean over all 2,947,298 released lineages, controls included, is 1.7e-17 (measured), so the reference is not a loader mean and stays.

### Changes

- `hale2024.py`: the control-gRNA docstring paragraph replaced (the old "perturb no gene, so they are never a record" reason is superseded); a replicate-structure paragraph; eight new sourced values (`CONTROL_GUIDES_IN_LIBRARY`, `CONTROL_GUIDE_LOCUS`, `REPLICATE_ASSAYS`, `ASSAYS_IN_PARALLEL`, `SAME_LINEAGES_IN_EVERY_ASSAY`, `SEGREGANT_BARCODES`, `BARCODES_PER_GUIDE`, `REPLICATION_BY_BARCODES`), audited at L3 with the others; `CONTROL_LINEAGES = 106,796`; the ledger rule keeps its name `control_guide_targets_no_gene` with a description that states why the lineages are held out.
- `test_hale2024.py`: two `--data` tests, one auditing every sourced quote against the pinned mirror bytes, one pinning the release's lineage, control and design counts above.
- No change to `runners.py` (`expected_count` stays 8,692,635) or `build_time_projection.py`.
