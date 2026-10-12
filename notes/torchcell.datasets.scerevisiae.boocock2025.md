---
id: 5frujg9cdpla45rel1tnjrn
title: Boocock2025
desc: ''
updated: 1791727604656
created: 1791727604656
---

## 2026.10.11 - Loaded: one-pot single-cell eQTL panels of three Bloom 2019 crosses

Yeast expansion wave 1, row 4. Boocock et al. 2025, eLife, doi 10.7554/eLife.95566 (the candidate table once carried a Nature DOI; the mirror's manifest is authoritative), citation key `boocockSinglecellEQTLMapping2025`. Loader `torchcell/datasets/scerevisiae/boocock2025.py`, dev store `$DATA_ROOT/data/torchcell/boocock2025/`, verifier [[torchcell.verification.segregant_expression]], adapter [[torchcell.adapters.boocock2025_adapter]]. Design inherited from [[torchcell.datamodels.eqtl-data-model]] and [[torchcell.datasets.scerevisiae.bloom2019]].

### What the release is, corrected against the brief and the candidate row

The brief and the candidate row described "27,744 single cells from 393 sequenced BY x RM segregants in two carbon sources (glucose, galactose)". Read against the mirrored paper and Supplementary file 1 Table S4, that conflates two experiments and names a condition that does not exist:

- 393 PREVIOUSLY GENOTYPED BY x RM segregants (Albert 2018), pooled and profiled as 7,124 cells, a median of 17 cells per segregant (Table S4 row `A*`).
- 27,744 cells are a different experiment: DE NOVO BY x RM segregants from a random-spore prep (cross A), each cell its own meiotic product. Two more de novo crosses were profiled the same way: B = YJM145 x YPS163 (44,784 cells) and C = CBS2888 x YJM981 (6,595 cells, reduced by the authors from 14,823 to unique genotypes).
- Every culture is glucose-based. No galactose condition exists; the only galactose in the Methods is the galactose-inducible Cas9 of the GPA1 allele-replacement strains.

The candidate row's text fields were corrected in `experiments/database/scripts/build_candidate_datasets_table.py` and the tables regenerated (only this row's lines moved).

### Records (measured from the built store)

| panel | records | blocks per record (min / median / max) | stored bytes per record (median) |
|---|---|---|---|
| Ap | 393 | 46 / 72 / 118 | 48,990 |
| A | 27,744 | 257 / 736 / 1149 | 51,760 |
| B | 44,784 | 246 / 729 / 1122 | 47,295 |
| 3004 | 6,595 | 167 / 529 / 952 | 34,462 |
| all | 79,516 | | 48,087 (total 3.79 GB) |

One record per pooled segregant (its matched cells summed) and one per de novo cell. The genotype is a `SegregantGenotype`, never `SequenceVariantPerturbation`; the phenotype is the new `SingleCellExpressionPhenotype` (schema block `--- begin boocock2025` in `schema.py`), in the new `SegregantExpressionExperiment` pair, the expression sibling of `SegregantGrowthExperiment`.

### Parents and the genomes tier: no new assembly set

The three crosses are Bloom 2019's crosses A, B and 3004 (the HMM's transition probabilities come from their genetic maps), and each parent is already a pinned assembly:

| Boocock label | Table S1 genotype (verbatim) | assembly |
|---|---|---|
| BY | BY MatA (YLK1879) | `sgd_S288C_R64-4-1_20230830`, `S288C_reference_sequence_R64-4-1_20230830.fsa` |
| RM | RM MatAlpha AMN1-BY ho∆::HphMX flo8∆::NatMX (YLK1950) | `peter2018_1011_assemblies`, member `GENOMES_ASSEMBLED/AAA_6.re.fa` |
| YJM145 | YJM145 MatAlpha ho∆::HphMX flo8∆::NatMX (YLK2436) | same tarball, `AAS_3.re.fa` |
| YPS163 | YPS163 MatA ho∆::HphMX flo8∆::NatMX (YLK2438) | same tarball, `ACD_7.re.fa` |
| CBS2888 | CBS2888 MatA ho∆::KanMX (Box A1 C3) | same tarball, `ABL_5.re.fa` |
| YJM981 | YJM981 MatAlpha ho∆::HphMX (yST191) | same tarball, `ADI_4.re.fa` |

The Peter ids are Bloom 2019's (its Figure 1 source data 1); RM11-1a = AAA is also stated directly by Peter 2018 Table S1 (mirrored `peterGenomeEvolution10112018/si/si3.xls`, sheet Table S1: `RM11-1a | AAA | Zinfandel vineyard | Wine | California, USA`). tc-data serves both sets through the existing genomes route (`torchcell/datasets/server.py`); a tier directory for "BY and RM" would duplicate bytes the tier already pins, so none was deposited. Allele coding: the HMM posterior is P(parent 2) with parent 2 = RM, YPS163, YJM981 (Table S4 `effect_directions`), the same order as Bloom 2019's `cross_genotypes_README`; L4 confirms it at the MAT locus.

### Genotype encoding and its cardinality cost

De novo cell: the released per-cell posterior P(parent 2) at the cross's LD-pruned markers (`segData.RDS` `Gsub`; 2,190 to 3,268 markers) as blocks of constant called parent (posterior > 0.5) AND constant P(called parent) rounded to a 0.05 grid. Every marker re-expands to its exact called parent and a posterior within 0.025 of the release. Pooled segregant: the whole-genome hard calls it was matched to (`gdata_42k.RData`, 1,012 segregants x 42,052 markers, -1 = BY, +1 = RM), run-length encoded at posterior 1.0.

Measured by `experiments/036-dataset-fixes-before-kg-build/scripts/boocock2025_release_inventory.py` (500 cells per de novo panel, seed 0; results `experiments/036-dataset-fixes-before-kg-build/results/boocock2025_release_inventory.json`), the panels and the cost of each encoding:

| panel | cells | transcripts | median UMI | LD-pruned markers | posteriors in (0.05, 0.95) | P(parent 2) at MAT |
|---|---|---|---|---|---|---|
| Ap | 7,124 | 4,995 | 1513.5 | 2,700 | 0.2465 | 0.202 (chrIII:209941) |
| A | 27,744 | 5,435 | 4017.0 | 3,268 | 0.3307 | 0.2114 (chrIII:204897) |
| B | 44,784 | 5,556 | 2130.0 | 2,824 | 0.4705 | 0.6417 (chrIII:201163) |
| 3004 | 6,595 | 4,696 | 1982.0 | 2,190 | 0.4483 | 0.2526 (chrIII:204174) |

| panel | encoding | blocks median | blocks p90 | blocks max | genotype bytes median |
|---|---|---|---|---|---|
| A | called_parent_only | 82.5 | 104.1 | 145 | 3,932 |
| A | 0.1 | 513.5 | 632.1 | 816 | 24,241 |
| A | 0.05 | 724.5 | 883.4 | 1145 | 34,184 |
| A | 0.02 | 1037 | 1259.3 | 1551 | 48,833 |
| A | 0.01 | 1269 | 1544.1 | 1850 | 59,780 |
| B | called_parent_only | 76 | 100 | 154 | 3,626 |
| B | 0.1 | 493.5 | 583.1 | 741 | 23,336 |
| B | 0.05 | 728.5 | 849.2 | 1005 | 34,448 |
| B | 0.02 | 1115 | 1261.5 | 1377 | 52,566 |
| B | 0.01 | 1421 | 1591.1 | 1730 | 67,036 |
| 3004 | called_parent_only | 57 | 74 | 106 | 2,736 |
| 3004 | 0.1 | 355.5 | 447.1 | 635 | 16,806 |
| 3004 | 0.05 | 533 | 667.1 | 864 | 25,195 |
| 3004 | 0.02 | 842.5 | 1027.1 | 1216 | 39,842 |
| 3004 | 0.01 | 1107 | 1302.1 | 1466 | 52,346 |

A pure called-parent run (the Bloom encoding) is about 4 KB a cell but replaces the posterior by its minimum over the run; the 0.05 grid keeps the posterior at a median of 25 to 34 KB a cell (arithmetic on those medians: about 2.7 GB of genotype across the 79,123 cells). The pooled panel's whole-genome mosaics run 46 to 118 blocks (median 72). The posterior is uncertain on a large share of markers (25% to 47% of marker posteriors lie strictly inside (0.05, 0.95)), which is what the grid preserves and a binarized run would hide.

Schema consequence: `SegregantGenotype` refused two adjacent blocks of one parent, a rule written for hard calls. It now refuses only adjacent blocks sharing parent AND posterior; every Bloom 2019 genotype validates as before, and `schema_impact_check` marks `Bloom2019Dataset` stale (validator change, not breaking).

### Phenotype

`umi_count` holds the nonzero counts over the panel's analysis transcripts, the matrix the authors model (`segData.RDS` `Yr`, transcripts with nonzero counts in at least 128 cells; 4,995 / 5,435 / 5,556 / 4,696, Table S4 `n_transcripts`), against a pinned universe (`n_features`, `feature_universe_sha256`). `total_umis` is the release's `nUMI` over every feature (the size factor in the authors' models); `cell_cycle_stage_counts` is the authors' per-cell stage (`cell.cycle.df`, five haploid stages M/G1, G1, G1/S, S, G2/M), the cell-cycle occupancy the growth/reproduction tradeoff is read from; `library_cell_counts` names the 10x library, the authors' batch. The reference of each panel is the panel pseudobulk (every cell summed), an absolute baseline as in Caudal 2024, not a centered zero.

Search for a per-segregant growth readout, done before concluding there is none: the paper's growth measurement is the GPA1 allele-replacement strains in a plate reader ("Growth rate was quantified as the geometric mean rate of growth (GMR)", doublings per hour, YP + 2% glucose, 30 C), engineered BY4741 derivatives, not segregants; the per-segregant growth of these crosses in the release (`data/rr/extracted_average_phenotypes.RData`) is Bloom 2019's, which is already served as `Bloom2019Dataset`. So this dataset adds no growth class; the engineered strains are issue #1005.

### Environment and sourcing (verbatim, `paper.md` sha256 `a2920750...`)

- Pooled panel: "This pool of strains was then transferred to minimal YNB medium $\left( 6 . 7 ~ \mathsf { g } / \right|$ Difco Yeast Nitrogen Base w/o Amino Acids, $2 \%$ glucose) and grown overnight at $3 0 ^ { \circ } \mathsf { C }$ in YPD $2 \%$ bacto peptone, $1 \%$ yeast extract, $2 \%$ glucose)." Stored as `restated(SM, ...)` (identity of library `SM`), 30 C.
- De novo panels: "YNB with $2 \%$ glucose was added to the sorted cells in sheath fluid (PBS, Bio-Rad #12012932) and they were transferred to a shaking incubator and grown at $3 0 ^ { \circ } \mathsf { C }$ ." and "For our experiments with the BY and RM cross (YLK3051), the cells were harvested in $2 \times \mathsf { Y N B }$ medium containing sheath fluid." New library medium `YNB_SHEATH_FLUID_BOOCOCK2025` (every amount an identity without a number: the sheath-fluid volume is not stated).
- Segregant matching: "Segregant identity was determined by picking the previously genotyped segregant with genotypes that were most correlated with the genotype probability vector for a given single cell."
- Analysis transcripts: "Transcripts with non-zero counts in at least 128 cells were used for downstream analyses."
- Duplicates: "This filtering step was only needed for the cross of YJM981 and CBS2888, and the procedure reduced the number of cells from 14,823 to 6595."

Every quote is a `SourcedValue` in `SOURCED` and audited at L3.

### Verification, L0 to L4

`PYTHONPATH=<wt> python -m torchcell.database.build_dataset_lmdb --dataset Boocock2025Dataset --retire-existing --verify` (report `preprocess/verification_report.json`, rendered by the inventory script's `render`):

| level | rule | result | message |
|---|---|---|---|
| L0 | `structural` | pass | 79516/79516 records validate as segregant_expression experiments |
| L1 | `count` | pass | 79516 records (expected 79516) |
| L1 | `panel_counts` | pass | records per panel {'Ap': 393, 'A': 27744, 'B': 44784, '3004': 6595} |
| L1 | `segregant_uniqueness` | pass | 79516 unique (panel, segregant) keys, 0 duplicates |
| L1 | `provenance_gaps` | pass | 0 documented provenance gaps over 0/79516 records; 0 deferred field(s): []; 1270684 undeclared None values over 9 carrier fields (top: Compound.inchi x238548, Compound.chebi_id x238155, Compound.inchikey x158639, Compound.pubchem_cid x158639, Compound.smiles x158639) |
| L1 | `canonical_gene_names` | pass | no gene perturbations to check |
| L2 | `uncertainty_sanity` | pass | 0 labeled uncertainties, none a zero dispersion; 0 records report n_samples >= 2 with no uncertainty |
| L2 | `value_fidelity` | pass | every record's counts, totals and partitions equal the release |
| L2 | `mosaic_round_trip` | pass | every mosaic re-expands to its release (worst de novo posterior error 0.0250, bound 0.025) |
| L2 | `reference_pseudobulk` | pass | each panel's reference is the sum of its cells |
| L3 | `compound_identity` | pass | environment edits: 0 compound references carry a structure identifier; 0 declare a typed gap (0 distinct compounds, unencodable) |
| L3 | `media_compound_identity` | pass | medium components: 79909 compound references carry a structure identifier; 0 declare a typed gap (0 distinct compounds, unencodable) |
| L3 | `media_membership` | pass | 79123 records on a shared MEDIA_LIBRARY medium, 393 on a medium deriving from one (2 distinct media) |
| L3 | `table_s4` | pass | cells, analysis transcripts and median UMI per panel equal Table S4 |
| L3 | `pooled_panel` | pass | 393 segregants, median 17.0 cells, 277 with more than ten; median per-cell agreement 0.9212 on 2641 shared markers (paper 0.925, tolerance 0.01) |
| L3 | `sourced_values` | pass | 34/34 quotes present in the pinned bytes |
| L4 | `parents_pinned` | pass | 6 parents; Peter ids ['AAA', 'AAS', 'ABL', 'ACD', 'ADI'] in the assembly member index; BY is the S288C reference |
| L4 | `mat_orientation` | pass | P(parent 2) at the marker nearest MAT per panel {'Ap': 0.01272264631043257, 'A': 0.21140421704896686, 'B': 0.6417149698417319, '3004': 0.2525628545762396}; the MATa parent is parent 2 only where the value exceeds 0.5 |
| L4 | `marker_reference` | pass | 1.0000 of 2000 sampled markers carry the S288C base as ref (>= 0.95) |
| L4 | `gene_containment_sgd` | pass | 5623/5623 analysis transcripts are S288C reference genes (1.0000, >= 0.99); 6504 spanned genes, all reference: True |

### Raw mirror (`$DATA_ROOT/torchcell-raw/boocockSinglecellEQTLMapping2025/`, 2.0 GB, `manifest.json`)

| path | bytes | sha256 | retrieval |
|---|---|---|---|
| `data/zenodo/data/out/combined/Ap/segData.RDS` | 150,283,799 | `4901cbc87c9b...` | `tar_member` |
| `data/zenodo/data/out/combined/A/segData.RDS` | 770,103,195 | `da7dd8ae659a...` | `tar_member` |
| `data/zenodo/data/out/combined/B/segData.RDS` | 1,060,929,225 | `b45792763e9b...` | `tar_member` |
| `data/zenodo/data/out/combined/3004/segData.RDS` | 117,847,602 | `405f0c4bee8e...` | `tar_member` |
| `data/zenodo/data/out/combined/Ap/segMatch.RDS` | 140,447 | `c40a576d6a52...` | `tar_member` |
| `data/dropbox/gdata_42k.RData` | 11,682,582 | `23554b871d3a...` | `direct_url` (the authors' code README link) |
| `si/si1.xlsx` | 979,712 | `7a0d8104d053...` | `pmc_cloud_object` |
| `code/*.R` (6 files) | | | `direct_url`, joshsbloom/single_cell_eQTL @ 9c5f66a3 |

The Zenodo deposit (record 12695128, doi 10.5281/zenodo.12695127) is one `additional_data_boocock_et_al_2023.tar.gz` of 111,900,721,589 bytes; retrieved 2026-10-11 by parallel range requests, md5 `d625073dbfe53914a7f38fa60bfec447` equal to Zenodo's own checksum, sha256 `ac3c27545462708ea6ac96e7a496a4c41d391041cd3806744920f82c64d57942`. Each member's `RetrievalRecord` is the new `torchcell.literature.retrieve.tar_member` (streams the container, checks its sha256, returns the member), so a rebuild does not need the container on disk. The container itself is not deposited. Raw reads (SRA PRJNA1049497) are not deposited. The Dryad deposit (doi 10.5061/dryad.xgxd254qb, the per-library CellRanger matrices) answers HTTP 401 without a bearer token: issue #1004 (by-hand), not needed by any record.

### Not loaded, each with the reason

| what | size | why |
|---|---|---|
| eQTL, hotspot, cell-cycle occupancy QTL and noise tables (Supplementary files 2 to 7; `LOD_NB_*`, `Betas_NB_*` RDS) | | derived analyses, conditional on method and threshold, no source value to verify: recorded here, not stored as phenotypes |
| F1 diploid allele-specific expression cells | 13,973 cells (5,890 / 2,864 / 5,219) | a diploid hybrid has no `SegregantGenotype`; issue #1005 |
| GPA1 allele-replacement scRNA-seq, growth rates, mating efficiency | 26,859 cells; plate-reader series; 2 mating replicates | engineered BY4741 derivatives, a separate record family; issue #1005 |
| per-library full count matrices (all features) | Dryad, 4.7 GB | walled; issue #1004 |

### Owner decisions taken by recommendation

1. Posterior grid 0.05 (error at most 0.025) rather than a called-parent run (binarizes) or one block per marker (about 3,000 blocks a cell): the measured middle of the cardinality table.
2. The pooled segregant's genotype is its whole-genome hard call, not an average of its cells' posteriors: the cells were matched TO that genotype, and a pooled posterior would be this repo's estimate, not the release's.
3. De novo records are per cell; cells of one spore clone are kept as separate records, as the authors kept them for A and B.
4. Pooled-panel medium = minimal YNB (the sentence also names YPD for the overnight step), flagged.
5. Crosses B and C carry the sheath-fluid medium as a documented representative ("captured as early as practicable" names no medium), flagged.
6. The phenotype universe is the authors' analysis transcripts; `total_umis` is the all-feature `nUMI`.
7. The reference is the panel pseudobulk.
8. RM11-1a resolves to the 1011 panel in the candidate gate (`torchcell/candidates/gates.py`), from Peter 2018 Table S1.

### Schema impact (`scripts/schema_impact_check.py --base origin/main`)

9 changed symbols, 0 breaking: `SegregantExpressionExperiment`, `SegregantExpressionExperimentReference` and `SingleCellExpressionPhenotype` added; the `SegregantGenotype` and `SingleCellExpressionPhenotype` validators; `ExperimentType`, `ExperimentReferenceType`, `PhenotypeType`, `EXPERIMENT_TYPE_MAP` and `EXPERIMENT_REFERENCE_TYPE_MAP` gained members. Stale, rebuilt by the KG 4.0 full build: `Bloom2019Dataset` (via `SegregantGenotype`), `Boocock2025Dataset`, and the 17 loaders whose closure holds `ExperimentType` (Caglar 2017 x3, Gupta 2024, Ishii 2007 x4, Li 2014, Schastnaya 2021, De Siqueira 2025 x4, Lim 2022, Volk 2021 bioscreen, Hoepfner 2014). The supported-query drift the pre-commit hook reports names `ProvenanceGapMixin` and `SampleUnit`, no symbol of this change, and is pre-existing at the base `d88f3865c`.

Candidate gate: `python -m torchcell.candidates gate --row "Boocock 2025" --table yeast --phenotype-class SingleCellExpressionPhenotype --write`: G1 to G5 pass, outcome admissible (`database/candidates/boocockSinglecellEQTLMapping2025.json`).

## 2026.10.11 - One single-cell leaf with Jackson 2020 (#1016), store rebuilt

Owner decision (#1016): one class for Boocock 2025's and Jackson 2020's single-cell UMI counts, unless merging loses data. Measured lossless, so `SingleCellExpressionPhenotype` is now the leaf for both (and for N'Guessan 2025); the field-by-field table, the deciding fields and the cost are in [[torchcell.datamodels.eqtl-data-model]] (2026.10.11, #1016). The experiment pair, the mosaic genotype and the panel-pseudobulk reference are unchanged.

What changed in a Boocock record, measured by `experiments/036-dataset-fixes-before-kg-build/scripts/single_cell_leaf_merge_audit.py`:

- Before the merge (`round-trip` on the retired store `processed.superseded.20261011-220231`): all 79,516 records and 4 references round-trip through the merged field set with no stored value dropped or changed. `total_umis` equals the stored gene counts on 35,156 records and exceeds them on 44,360 (173,191 UMIs in all on transcripts outside the analysis set), which is why the merged class bounds the total from below instead of requiring Jackson's equality.
- After the rebuild (`old-vs-new`, 79,516 records and 4 references paired by key): 0 stored values dropped or changed. Added to every record and reference: `cell_barcodes`, `cell_libraries` and `cell_total_umis` (each summed cell's release barcode, `cell.cycle.df` `named_dataset` library and `barcode.features` `nUMI`), and `marker_umi_count=None` with `MARKER_GAP` (`not_reported_by_primary`). The gap's source is a new sourced value, `SOURCED["alignment_reference"]`: "Sequencing reads were analyzed using Cellranger (version 5.01) using the S2888C reference genome (SGD, R64-2-1)" (paper.md, OCR's "S2888C" kept verbatim): no cassette sequence is named, where Jackson 2020 states adding its two.
- A bare barcode does not name a cell: every panel's libraries reuse barcodes (Ap 7,124 cells and 7,105 distinct barcodes, A 27,744 and 27,713, B 44,784 and 44,603, 3004 6,595 and 6,520; `cells` mode), so the pair (library, barcode) is the cell's identity, as the de novo `segregant_id` already wrote it. The pooled panel's records now say which matched cells they sum, which the pre-merge store did not record.

Rebuilt with `build_dataset_lmdb --dataset Boocock2025Dataset --retire-existing --verify` at `78ec49eb5` as slurm job 3973 (`gilahyper_build_dataset_lmdbs_array.slurm` with this worktree as `TORCHCELL_SRC`): build 2,046 s (the pre-merge build took about 13 min; the per-cell lists make each panel reference 0.44 to 2.16 MB of JSON, re-serialized for every record interned), peak RSS 11.2 GB, every rule passes:

| level | rule | result | message |
|---|---|---|---|
| L0 | `structural` | pass | 79516/79516 records validate as segregant_expression experiments |
| L1 | `count` | pass | 79516 records (expected 79516) |
| L1 | `panel_counts` | pass | records per panel {'Ap': 393, 'A': 27744, 'B': 44784, '3004': 6595} |
| L1 | `segregant_uniqueness` | pass | 79516 unique (panel, segregant) keys, 0 duplicates |
| L1 | `provenance_gaps` | pass | 79516 documented provenance gaps over 79516/79516 records (the marker gap, one each) |
| L1 | `canonical_gene_names` | pass | no gene perturbations to check |
| L2 | `uncertainty_sanity` | pass | 0 labeled uncertainties |
| L2 | `value_fidelity` | pass | every record's counts, totals, partitions and summed cells equal the release; every record gaps marker_umi_count |
| L2 | `mosaic_round_trip` | pass | worst de novo posterior error 0.0250, bound 0.025 |
| L2 | `reference_pseudobulk` | pass | each panel's reference is the sum of its cells (and lists every cell) |
| L3 | `compound_identity`, `media_compound_identity` | pass | 0 environment edits; 79,909 medium compound references identified |
| L3 | `media_membership` | pass | 79,123 records on a library medium, 393 on one deriving from it |
| L3 | `table_s4` | pass | cells, analysis transcripts and median UMI per panel equal Table S4 |
| L3 | `pooled_panel` | pass | 393 segregants, median 17.0 cells, 277 with more than ten; agreement 0.9212 on 2,641 shared markers |
| L3 | `sourced_values` | pass | 35/35 quotes present in the pinned bytes (the alignment-reference quote is new) |
| L4 | `parents_pinned` | pass | 6 parents; BY is the S288C reference |
| L4 | `mat_orientation` | pass | P(parent 2) at MAT {'Ap': 0.0127, 'A': 0.2114, 'B': 0.6417, '3004': 0.2526} |
| L4 | `marker_reference` | pass | 2,000 of 2,000 sampled markers carry the S288C base |
| L4 | `gene_containment_sgd` | pass | 5,623/5,623 analysis transcripts are S288C genes; 6,504 spanned genes |

Schema impact (`scripts/schema_impact_check.py --base origin/main`): 5 changed symbols, all breaking, 3 datasets (`Boocock2025Dataset`, `TfDeletionPseudobulkJackson2020Dataset`, `Nguessan2025Dataset`); none is served, and KG 4.0 is a full rebuild.

## 2026.10.12 - The leaf's measurement_type is pinned to scrnaseq_umi_count (PR #1052), store rebuilt

`SingleCellExpressionPhenotype.measurement_type` is now `Literal["scrnaseq_umi_count"]`: Boocock 2025, Jackson 2020 and N'Guessan 2025 are all 10x Chromium 3' UMI counts (this release: "loaded onto the Chromium device using either the Single Cell $3 ^ { \prime }$ Solution V2 or V3 kit.", `paper.md` sha256 `a2920750e12fb9977f36a687bb9f73aa0382527de84270b44609a3338bbeafe4`, section "Single-cell library preparation"), so one leaf takes one tag. Boocock's records already stored the leaf default, so no value changes; the leaf's contract fingerprint moved, which re-staled the store. Decision and record: [[torchcell.datamodels.eqtl-data-model]] (2026.10.12), `database/schema_changes/2026-10-12-single-cell-umi-one-tag.json`.

Rebuilt with `build_dataset_lmdb --dataset Boocock2025Dataset --retire-existing --verify` at `de33b6b15` (the branch rebased on main `007cd23b0`), slurm job 4314: 79,516 records, 4 references, build 2,265 s, job 54.6 min, peak RSS 11.3 GB, every rule passes.

| level | rule | result |
|---|---|---|
| L0 | `structural` | 79,516 of 79,516 records validate as segregant_expression experiments |
| L1 | `count` | 79,516 (expected 79,516) |
| L1 | `panel_counts` | Ap 393, A 27,744, B 44,784, 3004 6,595 |
| L1 | `segregant_uniqueness` | 79,516 unique (panel, segregant) keys, 0 duplicates |
| L1 | `provenance_gaps` | 79,516 documented gaps (the marker gap, one each) |
| L1 | `canonical_gene_names` | no gene perturbations to check |
| L2 | `uncertainty_sanity` | 0 labeled uncertainties |
| L2 | `value_fidelity` | counts, totals, partitions and summed cells equal the release |
| L2 | `mosaic_round_trip` | worst de novo posterior error 0.0250, bound 0.025 |
| L2 | `reference_pseudobulk` | each panel's reference is the sum of its cells |
| L3 | `compound_identity`, `media_compound_identity` | 0 environment edits; 79,909 medium compound references identified |
| L3 | `media_membership` | 79,123 on a library medium, 393 on one deriving from it |
| L3 | `table_s4` | cells, analysis transcripts and median UMI per panel equal Table S4 |
| L3 | `pooled_panel` | 393 segregants, median 17.0 cells, 277 above ten; agreement 0.9212 on 2,641 markers (paper 0.925, tolerance 0.01) |
| L3 | `sourced_values` | 35 of 35 quotes present in the pinned bytes |
| L4 | `parents_pinned` | 6 parents; BY is the S288C reference |
| L4 | `mat_orientation` | P(parent 2) at MAT: Ap 0.0127, A 0.2114, B 0.6417, 3004 0.2526 |
| L4 | `marker_reference` | 2,000 of 2,000 sampled markers carry the S288C base |
| L4 | `gene_containment_sgd` | 5,623 of 5,623 analysis transcripts are S288C genes; 6,504 spanned |

`old-vs-new` (slurm job 4315, `experiments/036-dataset-fixes-before-kg-build/scripts/single_cell_leaf_merge_audit.py`) against `processed.superseded.20261012-080305`, the store dev-store array 4011 built at `1475b8cf1`: 79,516 records and 4 references paired, **0 stored values dropped or changed**. That baseline predates #1038 and #985 on main, so the fields it lists as added come from those two PRs, none from this change: the #1038 merge fields (`cell_barcodes`, `cell_libraries`, `cell_total_umis`, `marker_umi_count` with its gap) and the #985 `provenance_gaps` on `environment.media`.

Candidate verdict re-gated at the rebased commit: G1 to G5 pass, admissible.
