---
id: dl3271ykf81ncfbfr0nsucj
title: Puddu2019
desc: ''
updated: 1791732588252
created: 1791732588252
---

## 2026.10.11 - Puddu 2019: the realized genome architecture of the knockout collection, loaded

Loader: `torchcell/datasets/scerevisiae/puddu2019.py` (`GenomeArchitecturePuddu2019Dataset`, dev store `data/torchcell/genome_architecture_puddu2019`). Source: Puddu F, Herzog M, Selivanova A, et al., Jackson SP. Nature 2019;573:416-420, doi:10.1038/s41586-019-1549-9, citation key `pudduGenomeArchitectureStability2019` (Zotero item `2GFWMA3B`). Yeast expansion schedule row 7 (`notes-tex/database/database-expansion-100`). Vocabulary: [[torchcell.datamodels.genome_architecture_features]]. Verifier: [[torchcell.verification.genome_architecture]]. Adapter: [[torchcell.adapters.puddu2019_adapter]]. Every count below is measured on the pinned workbook or read out of the built store's `preprocess/release_ledger.json`.

### Why this row is a genotype backbone

Half the served yeast rows are screens OF this collection (Costanzo 2016, Kuzmin 2018/2020, Kemmeren 2014, Messner 2023, Ohya 2005, Hillenmeyer 2008, Hoepfner 2014) and every one of them models a strain as "S288C with one ORF deleted". Puddu sequenced the collection and measured what those strains actually carry. The designed perturbation is the deletion; the realized genome content is what the sequencing found. The Hoepfner 2014 background already records the gap this closes, verbatim from its own typed gap: "The schema has no per-chromosome copy number, so the aneuploidy and the amplification cannot be recorded per strain".

How much detail the served stores would gain, measured gene by gene against this store's 4,408 served genes and against Table 4's own deviation flags (1,683 of those genes carry at least one element deviating from wild type):

| served store | genes | sequenced here | sequenced AND deviating |
|---|---|---|---|
| `smf_costanzo2016` | 5,493 | 4,026 (73.3%) | 1,470 (26.8%) |
| `dmf_costanzo2016` | 5,848 | 4,215 (72.1%) | 1,565 (26.8%) |
| `dmi_costanzo2016` | 5,848 | 4,215 (72.1%) | 1,565 (26.8%) |
| `microarray_kemmeren2014` | 1,484 | 1,353 (91.2%) | 529 (35.6%) |

(Script: `/scratch/tmp/.../scerevisiae-puddu2019/overlap2.py`, read-only over the served `database/data/torchcell/*/preprocess/gene_set.json` and the dev Kemmeren store. 176 released genes carry 3 or more deviating elements; the paper's own count of 151 "genome instability genes" uses its per-strain definition, so these two numbers are not the same statistic.)

### What the release is

One workbook: `si/si3.xlsx` (3,098,936 bytes, sha256 `fa369c18a31c9ec7f0ade8b63f0be6a595821ac360aa72b1e1dcd46ed8994214`), retrieved from the PMC Article Datasets bucket as `EMS83962-supplement-Supp_Tab.xlsx` on 2026-10-11 (`pmc_cloud`), holding Supplementary Tables 1-6. Measured on those bytes:

| sheet | rows | what it is |
|---|---|---|
| Table 1 | 9,014 colonies (+1 stray trailing cell holding a bare `2`) | ENA sample accession per colony, over 4,535 strain ids; 171 of those colonies are not in Table 3 |
| Table 2 | 10 features | the wild-type mean, SD and 3-SD interval, plus the number of genes and colonies outside it |
| Table 3 | 8,843 colonies x 28 numeric columns | the per-colony estimates, no missing cell, no negative value |
| Table 4 | 4,436 genes | a DERIVED per-gene summary listing only values that deviate from wild type |
| Table 5 | 21 strains | the validation experiments' other strains (ts alleles, reporters), not a YKOC screen |
| Table 6 | 8,831 colonies | the barcodes detected in the colony and its barcode-CONFIRMED deletion |

Three internal identities hold exactly on those bytes and are what makes the release self-checking: `ANtot` equals the sum over the 16 chromosomes of `|copy number - 2|` to within 3.1e-15 for all 8,843 colonies; every colony's accession in Table 3 equals Table 1's; and `GCR` is integral everywhere (0 in 8,615 colonies, up to 6).

### The new phenotype and its vocabulary

`GenomeArchitecturePhenotype` (additive block in `schema.py`) holds one sequenced sample's profile keyed by its assay's own symbols, with `GenomeArchitectureExperiment` and `GenomeArchitectureExperimentReference` beside it. The vocabulary is a separate module, `torchcell/datamodels/genome_architecture_features.py`, registering `puddu2019_wgs_coverage` with the 28 column headers of Table 3 and, per feature, the unit basis of its number:

| basis | symbols | what one number is |
|---|---|---|
| `copies_per_haploid_genome` | rDNA, CUP1, Ty1-Ty5 | coverage over the locus / genome-wide median, scaled to the haploid genome |
| `copies_per_cell` | mitochondria, 2-micron | the same ratio times the sample's expected ploidy |
| `chromosome_copies_per_cell` | chr01-chr16 | one chromosome's copy number, 2.0 in an unperturbed diploid |
| `reads_per_million` | Telomeres | telomeric reads per million, an abundance and NOT a length |
| `ploidy_deviation_units` | ANtot | the summed deviation from 2n, in chromosome units |
| `rearrangement_count` | GCR | a non-negative integer count of called events |

This is the `BacterialMorphologyPhenotype` shape with the statistic replaced by the unit basis, for the same reason: the numbers in one record are not in one unit, so a consumer that averages across keys averages quantities the source never claimed were comparable. The class refuses a key outside the named assay, a negative or non-finite value, a non-integral `GCR`, and an `ANtot` that disagrees with the chromosome values beside it.

Nothing in `schema.py` could carry this: grep found no `aneuploid`/`karyotype` field anywhere, `CopyNumberVariantPerturbation` is gene-keyed with `provenance="natural"` and a `strain_id`, and `CalMorphPhenotype` / `BacterialMorphologyPhenotype` are imaging vocabularies. Schema impact (`scripts/schema_impact_check.py --base origin/main`): 9 changed symbols, 9 impacted datasets, **0 breaking**, every one stale only through the `ExperimentType` union gaining a member. KG 4.0 is a full rebuild, so this is admissible as additive.

### Records

8,826 `GenomeArchitectureExperiment` records, one per sequenced knockout colony, over 4,453 strains and 4,408 genes. Colonies are not averaged into strains: two colonies of one knockout are two independently isolated samples whose architectures differ, which is the paper's own unit of analysis (its figures plot each colony). Served strata: 257 strains with 1 colony, 4,093 with 2, 30 with 3, 72 with 4, 1 with 5. Over all 8,835 knockout colonies the strata are 258 / 4,097 / 30 / 72 / 1, which is the Extended Data Fig. 1c legend exactly.

Store: `processed/lmdb` 35 MB, `processed/interned` 32 KB (one environment, one reference, one publication; the genotype stays inline because it is one perturbation).

The genotype is a `BarcodedKanMxDeletionPerturbation` carrying the barcodes Table 6 DETECTED in that colony rather than the collection's expected tags, which is realized detail of its own. Census over the served colonies' UPTAG notes: 5,112 match the reference tag, 1,291 "No tags detected", 1,153 "Heterozygous tag: 1 same as reference, 1 mutated", 1,023 "Tag is different from reference expected tag", 218 "Heterozygous tag: both different to reference", and 29 colonies carry one of the aliasing or mis-attribution notes. A colony with no detected tag carries `barcode=None` with a typed gap quoting the note.

### The reference is the 8 wild-type colonies, and Table 2 is their summary

Table 3 holds 8 wild-type BY4743 colonies under four strain ids (`Del4733_WT-1` ... `Del4736_WT-4`). They are not perturbation records, so they are the reference: `phenotype_reference` carries Table 2's published wild-type mean with `uncertainty_type = sample_sd`, `n_samples = 8`, `sample_unit = colony`, and their per-colony rows go to `preprocess/wild_type_colonies.csv`.

Measured, which is why the reference can be sourced to Table 2 and verified against Table 3: the published mean IS those 8 colonies' mean and the published SD IS their n-denominator (population) SD.

| feature | published mean | mean of the 8 colonies | published SD | n-denominator SD |
|---|---|---|---|---|
| rDNA | 124.99 | 124.9905 | 7.85 | 7.8471 |
| CUP1 | 14.03 | 14.0340 | 0.48 | 0.4789 |
| mitochondria | 17.5 | 17.5041 | 1.6 | 1.6042 |
| 2-micron | 49.42 | 49.4225 | 9.06 | 9.0601 |
| Ty1 | 37.27 | 37.2691 | 0.27 | 0.2656 |
| Ty2 | 11.48 | 11.4752 | 0.24 | 0.2408 |
| Ty3 | 2.01 | 2.0066 | 0.14 | 0.1367 |
| Telomeres | 756.15 | 756.1467 | 43.97 | 43.9704 |
| ANtot | 0.15 | 0.1523 | 0.4 | 0.4030 |
| GCR | 0 | 0.0 | 0 | 0.0 |

Two per-key gaps on the reference, both typed: Ty4 and Ty5 have no Table 2 row although Table 3 releases both per colony; and the published `ANEUP` mean (0.15, SD 0.4) is dropped from the profile because the wild-type colonies themselves deviate slightly from 2n, so storing it beside the expected per-chromosome value of 2.0 would assert that a sum of zero deviations equals 0.15. The 16 chromosomes carry 2.0, sourced to "We define a strain as deviating from diploidy when any of the 16 chromosomes deviates from diploidy".

### ORF history: 24 renamed, 5 retired

The release names each deletion as the collection annotated it around 2000. Through `resolve_gene_name` against R64-4-1, of the 4,437 released names: 4,397 current and self-resolving, 11 features the annotation answers with their own id (a blocked reading frame or pseudogene, which IS the deleted locus), 24 renamed or absorbed into a neighbour, 5 retired. A renamed name is stored under the gene R64 carries now with a `ConstructedOrf` naming the source ORF; because each of those 24 lands on a gene the release ALSO screened separately, every relation is `merged` and the deleted interval is a `deferred_pending_source_review` gap on the SGD locus history. The map is `preprocess/constructed_orfs.json`. This is the Hillenmeyer 2008 rule applied to the same collection.

`perturbed_gene_name` is the systematic name, as in Hillenmeyer, not the release's strain label: the label is a strain designation, in 2 colonies it names a different gene than the barcodes confirmed, and 23 labels resolve to another R64 gene. The label stays verbatim in `construction.strain_accession` (`Del5_UBX7`) and in `preprocess/data.csv`.

### Refused, each with the measurement that refuses it

- **9 colonies (5 strains)** whose released deletion is one of the retired ORFs YAR037W, YAR040C, YAR043C, YCL013W, YCL026C. A record keyed to a name the current genome does not carry joins nothing. In the ledger under `refused`.
- **SNVs and indels**, i.e. the secondary point mutations of these strains. Figure 4 analyses them and the Methods state the calling pipeline, but no supplementary table carries a variant column (Tables 1-6 hold 3, 6, 31, 12, 5 and 9 columns, listed above) and the per-gene browser at `sgv.gurdon.cam.ac.uk` is a live URL, which is retrieval metadata and not a source. So no `SequenceVariantPerturbation` is emitted.
- **The raw reads.** 9,014 ENA samples in study ERP109205; the accession travels on every record instead (`sample_accession`).
- **Table 4 and Table 5.** Table 4 is a derived summary of Table 3, read only for its (systematic, common) pairs and as a cross-check; Table 5's 21 strains are the validation experiments, not a YKOC screen.

### No genomes-tier set, and why

The genomes tier holds sequence data. This release has none: the publisher's three files are an SI guide (`si/si1.docx`, 16,832 bytes, a six-line table of contents), one western-blot image (`si/si2.tif`, 11,697,348 bytes) and the table workbook. The derived calls are 8,843 x 28 = 247,604 numbers, which are records, not reference sequence, so they are served through the dataset route (35 MB of LMDB) and no tier directory is created. The raw mirror is `$DATA_ROOT/torchcell-raw/pudduGenomeArchitectureStability2019/si/si3.xlsx` with the literature mirror's own `pmc_cloud` retrieval record.

### Environment

The measured sample is the liquid culture the DNA came from: 1.8 ml YPAD at 30 C grown to saturation. `CultureEnvironment` with a loader-local liquid YPAD (`base_medium = "YPAD"`, the library object's components in the liquid state), `culture_format.working_volume_ul = 1800`, and a typed gap on `duration_hours` because "grown to saturation for 2-5 days" names no single duration and which colony took which is not released. `EndpointRule` has no member for "until the culture saturated" (`until_control_saturation` is about a vehicle control), so that statement rides on `culture_format.provenance` rather than being forced into the enum; adding the member is a follow-up.

### Sourced values

35 pinned quotes, all re-found in the sha256-pinned bytes by the L3 audit: 23 from `paper.md` (sha256 `ba2a9439...`) and 12 legend cells of the workbook. The ones that decide a stored value:

| value | quote | where |
|---|---|---|
| collection | "The homozygous diploid YKOC was acquired from EUROSCARF in 2001, propagated for two batch expansions (10–20 cell generations) in liquid complete synthetic medium" | Methods, duplication |
| the measured culture | "Two colonies per knockout were inoculated in $1 . 8 \mathrm { m l }$ YPAD and grown to saturation for 2–5 days, before collecting by centrifugation." | Methods, duplication |
| repeat copy number | "Estimates of copy numbers were obtained by dividing the median coverage across the appropriate genomic region (rDNA, chromosome XII:452000–459000; CUP1, chromosome VIII: 212986–213525) by the genome-wide median coverage and multiplying by two (this is because the S288C reference genome contains two copies of the minimal repeat unit)." | Methods, rDNA and CUP1 |
| per cell vs per haploid | "Median coverage was calculated across the following regions, and normalized by the genome-wide median to obtain the estimate of the copy number (in number of copies per haploid genome; in the case of the $2 \mu$ plasmid these estimates were multiplied by the expected ploidy to obtain the number of copies per cell; see ‘Quantification of mtDNA copy number’)." | Methods, Ty and 2u |
| telomere units | "Indirect measure of telomere length (in number of telomeric reads per million)" | Table 3 legend |
| chromosome ploidy | "This was obtained by dividing the coverage across a 10-kb window centred on the centromere by the genome-wide median and multiplying by 2 (assuming largely diploid samples)." | Methods, aneuploidy |
| the 2n reference | "We define a strain as deviating from diploidy when any of the 16 chromosomes deviates from diploidy" | Methods, thresholds |
| rho0 | "$r h o ^ { 0 }$ colonies were defined as having less than one mtDNA copy." | Methods, mtDNA |
| the deleted gene | "Confirmed gene deletion (systematic name)" | Table 6 legend |
| barcodes | "For each sample, the detected barcodes—observed at least 5 times and for a frequency ${ > } 2 0 \%$ —are reported in Supplementary Table 6, and noted if they do not match with the expected barcodes." | Methods, validation |
| ENA | "The primary sequencing data have been deposited in the European Nucleotide Archive in the study ERP109205, under the accession numbers detailed in Supplementary Table 1." | Data availability |
| wild-type SD | "Standard deviation of wild-type samples" | Table 2 legend |

### Verification (L0-L4, `preprocess/verification_report.json`)

| level | rule | result |
|---|---|---|
| L0 | structural | 8,826 records validated |
| L1 | count | observed 8,826, expected 8,826 |
| L1 | sample_uniqueness | 8,826 sequenced samples, one record each |
| L1 | assay_coverage | every record carries all 28 features of `puddu2019_wgs_coverage`, nothing outside it |
| L1 | provenance_gaps | 11,773 documented gaps over 8,826/8,826 records; 1 deferred field (`deleted_span`) |
| L1 | canonical_gene_names | 4,408 systematic names, one spelling each, each current in the genome; 11 are pseudogene loci the genome resolves to themselves |
| L2 | feature_fidelity | 247,128 values finite and inside their basis bound |
| L2 | release_fidelity | 247,128 stored values equal the released cell for their sample |
| L2 | uncertainty_sanity | no zero dispersion |
| L3 | vocabulary_parity | 28 distinct features, every basis bounded, assembly S288c_R64-1-1_Ensembl |
| L3 | provenance_audit | 35 pinned quotes present in the sha256-pinned sources |
| L3 | reference_profile | the reference states its mean, its uncertainty kind, its sample count and a gap per uncovered feature |
| L3 | compound/media identity, media_membership | 17,652 medium components identified; 8,826 records on a medium deriving from `YPAD` |
| L4 | paper_statistics | 303 rho0 colonies (paper: 303) and 0.5214 of served colonies away from 2n (paper: about 52%) |
| L4 | gene_containment_sgd | 1.000 of 4,408 measured genes are SGD S288C R64-4-1 genes |
| L4 | current_genome_genes | every one of the 4,408 names is a gene of the current genome |

The L4 rule is the paper's own arithmetic reproduced from the store: `mitochondria < 1` gives exactly the 303 rho0 colonies the Figure 3c legend states, and 4,607 of the 8,843 released colonies (52.10%) carry a chromosome away from 2.0 against the main text's "about 52%".

### Owner decisions taken by recommendation

- **The 8 wild-type colonies are the reference, not records.** They are BY4743 with no perturbation, Table 2's published statistics are exactly their mean and population SD, and serving them as records would need an empty genotype. Their rows are kept in `preprocess/wild_type_colonies.csv` so nothing is dropped silently.
- **One record per colony, never per strain.** The release's own unit, and two colonies of one knockout genuinely differ.
- **The deleted gene is read per colony from Table 6**, not per strain: 15 strain ids resolve to two different confirmed genes, so a per-strain read would assign one colony the other colony's gene. For the 7 colonies with no Table 6 row the gene comes from Table 4's (systematic, common) pair for the strain label, with both barcode fields gapped.
- **A colony is refused when its released ORF is retired** (9 colonies), rather than stored under a name the genome does not carry.
- **`perturbed_gene_name` is the systematic name** (see ORF history above).
- **The unit basis lives on the feature, in a registered assay**, not in a free-text `units` string, and not as 28 separate schema fields.
- **No `EndpointRule` member was invented** for "grown to saturation"; the statement is carried as provenance instead.

### Follow-ups

- An `EndpointRule` member for a culture grown to its own saturation (this release, and any batch culture read at saturation).
- A chromosome-level realized-genotype entity, if the graph should answer "which strains are aneuploid" from the GENOTYPE side rather than from this phenotype. Today the measured architecture is where that lives, and it is queryable per gene through the perturbation.
- The Hoepfner 2014 background's gap note can now point at this store for the per-chromosome copy number it says the schema cannot hold.

## 2026.10.12 - Stored as released: the genome gate is reversed

The loader no longer consults a genome. Removed: `resolve_released_genes`, `constructed_orf`, `_sgd_history_gap`, `ResolvedGenes`, the `genome=` constructor argument and `_resolver`, the `SCerevisiaeGenome` import, and the `preprocess/constructed_orfs.json` map.

- **Every knockout colony is stored.** `N_SERVED_RECORDS` goes from 8,826 to 8,835. The 9 colonies (5 strains) whose released deletion names an ORF R64 has since retired (YAR037W, YAR040C, YAR043C, YCL013W, YCL026C) are restored under those released names. Measured on the pinned workbook: 8,835 knockout colonies, 4,437 distinct released deletion names, 9 colonies on the 5 retired ORFs. The strata are the Extended Data Fig. 1c legend exactly (258 / 4,097 / 30 / 72 / 1 over 4,458 strains).
- **`systematic_gene_name` is the released name, verbatim**: the colony's Table 6 `Confirmed_deletion`, or for the 7 colonies with no Table 6 row the systematic name Table 4 pairs with the strain label. The 24 released names R64 has renamed or absorbed are stored as released too, with `constructed_orf=None`; the old rewrite touched 49 records over 24 ORFs.
- **`perturbed_gene_name` is Table 4's common name** for that systematic name (release-internal; 933 Table 4 rows pair a name with itself), or the systematic name when Table 4 has no row for it (1 name, YKL075C). Before this change it was the systematic name on every record.
- **Ledger.** `ReleaseLedger` drops `refused`, `n_released_orfs_renamed`, `n_released_orfs_refused` and `released_orfs_refused`, and gains `n_genes_without_table4_row`. The accounting validator is now served + wild type = released (8,835 + 8 = 8,843). `data.csv` carries one systematic-name column (the released one) plus `perturbed_gene_name`.
- **Verifier spec.** `runners.GENOME_ARCHITECTURE_DATASETS["genome_architecture_puddu2019"]["expected_count"]` goes from 8,826 to 8,835. The L1 `canonical_gene_names` and L4 `current_genome_genes` rules still read the genome; on the restored names they report findings (5 retired, 24 renamed), which gate the manifest only until the genome rules become report-only.

L0-L4 after rebuild: (orchestrator fills in after the slurm build)
