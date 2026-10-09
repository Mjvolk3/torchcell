---
id: wtq5bfgw6nv005psli2lkkn
title: Lamoureux2023_public_k12
desc: ''
updated: 1791435765061
created: 1791435765061
---

## 2026.10.08 - Public K-12 loader (rank 3 of the E. coli SI audit)

Source: `torchcell/datasets/ecoli/lamoureux2023_public_k12.py`
(`RnaseqPublicK12Lamoureux2023Dataset`, root `data/torchcell/rnaseq_public_k12_lamoureux2023`).
Tests: `tests/torchcell/datasets/ecoli/test_lamoureux2023_public_k12.py`.
Sibling: [[torchcell.datasets.ecoli.lamoureux2023]] (PRECISE-1K, the same release's other arm).
Audit: [[plan.bacteria-si-phenotype-audit-ecoli]] ranks 3 and 19.

Lamoureux et al. 2023, "A multi-scale expression and regulation knowledge base for
Escherichia coli", Nucleic Acids Res., doi:10.1093/nar/gkad750. Citation key
`lamoureuxMultiscaleExpressionRegulation2023`; `paper.md` sha256
`ca41cceba80fe6af16dfc8ffc671c2b9d0dac5d9eb409956891ba4bbedc147a7`.

The release's second arm: every publicly available *E. coli* K-12 RNA-seq run in the SRA,
curated and reprocessed through the same pipeline. "Finally, the 0.95 minimum replicate
correlation threshold was applied, yielding the final set of 1675 high-quality
publicly-available samples" (Methods, "Compiling the public K-12 dataset"). These are other
labs' experiments, so they are a separate dataset rather than more of PRECISE-1K, and each
record names the SRA experiment accession it came from.

### Built, measured 2026.10.08

`python -m torchcell.database.build_dataset_lmdb --dataset RnaseqPublicK12Lamoureux2023Dataset`:
**240 records** of the 1,675 public rows, 1 reference, gene set 22, 7 s. 140 wild type, 84
single and 16 double deletions over 22 distinct deleted genes; 156 LB and 84 M9; 97
conditions; 20 projects and 20 BioProjects. `torchcell.provenance.build_manifest`:
`rnaseq_public_k12_lamoureux2023` fresh.

| stage | rule | n |
|---|---|---|
| genotype | `strain_not_mg1655` | 669 |
| genotype | `plasmid_borne_construct` | 175 |
| genotype | `point_mutation_allele` | 41 |
| genotype | `allele_described_in_prose` | 36 |
| genotype | `evolved_isolate` | 29 |
| genotype | `phage_infected` | 20 |
| genotype | `background_label_undefined` | 7 |
| genotype | `deletion_symbol_unresolved` | 6 |
| genotype | `partial_gene_edit` | 3 |
| environment | `medium_not_in_library` | 250 |
| environment | `culture_not_batch` | 164 |
| environment | `minimal_medium_source_not_stated` | 24 |
| environment | `ph_not_stated` | 11 |

`strain_not_mg1655` dominates because the curation deliberately kept substrains: "These
profiles come from 134 different projects, including 15 K-12 substrains and 9 distinct
temperatures and pHs." Measured `Strain` cells: MG1655 1,006, BW25113 361, W3110 146,
AR3110 45, MC4100 41, JM109 37, PFM2 18, LE392 9, CV104_pHDB3 3, CV104_pLCV1 3,
`MG1655::φO104` 3, `MG1655::φPA8` 2, K-12 1. Only MG1655 can be written against the
MG1655 assembly pin the expression is keyed to. Four rules are defined and fire zero
times on this release, because the rows they describe are removed by an earlier rule:
`oxygen_regime_not_stated`, `oxygen_regime_transition`,
`oxygen_setpoint_not_expressible` and `temperature_not_stated`.

### Deduplication (the design question this dataset turned on)

These are reprocessed public samples, so the question is whether a record is a second copy
of something. Every number below is measured and asserted at build time; the full ledger is
`preprocess/accession_ledger.json`.

| measurement | value |
|---|---|
| Public K-12 metadata rows | 2,710 (1,675 public + 1,035 `p1k_`) |
| public rows | 1,675 |
| distinct experiment accessions | **1,675** (1,523 SRX, 112 ERX, 40 DRX) |
| distinct run accessions | **1,675** |
| distinct `BioSample` accessions | **1,568** |
| BioSamples carrying several rows | 38, over 145 rows |
| of those, spanning several conditions | 7 |
| count columns identical to another public column | **0** |
| count columns identical to a PRECISE-1K column | **0** |
| distinct BioProjects / PMIDs / GEO series | 89 / 38 / 30 |

**Policy, in code.** The record identity is the experiment accession, and the build refuses a
repeated experiment or run accession. `BioSample` is NOT the key, and the release itself
shows why: 38 BioSamples carry 2 to 12 rows, and 7 of those span different conditions or
strains (`SAMN12285586` covers `hyperpersistence:wt`, `:pth_mut` and `:metG_mut`;
`SAMN10078928` covers `evo_bz:w3110d13` and `:w3110d13_bz`), so the submitter registered one
BioSample for several libraries. Collapsing on it would merge distinct genotypes. The
value-level rule is profile identity: the build hashes every kept count column and refuses a
repeat, within the arm and against the 1,055 columns of `data/precise1k/counts.csv` the
sibling dataset serves. Both come back zero, which is the measured negative result.

**Against our own store: one BioProject overlaps, no record does.** `PRJNA645443` is in this
table (12 `phage_resist` BW25113 RNA-seq samples) and in `PhageRbTnseqMutalik2020Dataset`'s
manifest, where it is recorded as the SRA BioProject holding that paper's raw reads, which
the mirror deliberately does not keep ("the mirror keeps the processed fitness tables the
loader consumes, not the reads"). That dataset serves RB-TnSeq fitness, a different assay
from RNA-seq expression, and the 12 rows are BW25113 and dropped here anyway. None of the 38
PMIDs and none of the 30 GEO series matches any identifier in any `torchcell/datasets/ecoli`
or `torchcell/datasets/pputida` module. `RnaseqCaglar2017Dataset` is REL606, an *E. coli* B
strain the curation discarded: "RNA-seq samples were discarded if the strain was not from a
K-12 strain, if the strain was missing, or if the type of experiment was not actually
RNA-seq." Its GEO series `GSE94117` is not among the 30.

**The `p1k_` arm is excluded by construction.** `public_rows` keeps only the non-`p1k_` rows
and requires each to be an SRA, ENA or DDBJ experiment accession; the public count file
carries no `p1k_` column at all, so the two datasets share no measurement.

### The log2[TPM] matrix is NOT released, measured

The paper says it is: "The $\log _ { 2 } [ \mathrm { T P M } ]$ , raw read count, QC data
files and sample metadata for the high-quality public samples may be found in the data
directory of this project's GitHub repository." The audit listed this as "not checked" and
guessed the matrix was inside `k12_modulome.json.gz`. Checked 2026.10.08, and the guess is
refuted:

- The pinned Zenodo archive has no `data/k12_modulome/log_tpm*.csv` member, and neither does
  the live `SBRG/precise1k` tree at HEAD (GitHub API, recursive listing of 364 blobs).
- All three packaged `IcaData` objects (`k12_modulome.json.gz`,
  `k12_only_p1k_ctrl.json.gz`, `k12_only_proj_ref.json.gz`) carry `X: null` and
  `log_tpm: null`, as do `precise1k.json.gz` and `precise.json.gz`.

So the counts, the MultiQC table and the metadata are released and the expression matrix is
not. `expression_tpm` is therefore computed here from the released counts and the release's
own gene coordinates, the `RnaseqCaglar2017Dataset` convention ("The TPM is computed here:
the paper reports DESeq2-normalized values, not TPM"), and `measurement_type` says so:
`rnaseq_tpm_from_released_counts`.

**It is not the paper's TPM, and the release does not let anyone reproduce that.** Measured
on PRECISE-1K, where both a count matrix and a log2[TPM] matrix are released, this same
derivation does not reproduce the released values: maximum absolute difference 9.48 log2
units, median 0.0137, and 0 of 1,035 samples agree to within 0.01 across all genes. With a
per-sample scale fitted, 88.8% of cells agree to 1e-4 and the implied normalization total is
1.1% above the sum over the released genes, so the pipeline's TPM denominator covered a
larger gene set and a minority of cells came from a different quantification run. That is the
measurement behind the distinct `measurement_type`; passing a computed value off as the
released one would have been the laundering the evidence rules forbid.

### Phenotype and values

`BacterialRNASeqExpressionExperiment` + `RNASeqExpressionPhenotype`, one record per
sequenced library, over the count file's own 4,355 genes (the pre-QC-filter gene set, so the
98 short / low-FPKM genes the expression component dropped are present here).

- `expression_count`: the gene's cell of `data/k12_modulome/counts.csv`. The file carries
  3,125 SRA columns, the pre-QC superset, and the loader subsets it to the 1,675.
- `expression_tpm`: `(count / length) / sum(count / length) * 1e6`, so a record sums to
  exactly one million (asserted per record, and again by the verifier).
- gene length: the span of the gene's row in `data/annotation/gene_info.csv`, the annotation
  featureCounts counted against ("generating read counts using featureCounts (25) with the
  following non-default options: -p -B -C -P -fracOverlap 0.5"). Of the 4,352 genes whose
  b-number resolves to an ASM584v2 locus, 4,313 spans equal the pinned GenBank span and 39
  differ; the build pins that count and writes the list to
  `preprocess/gene_length_divergence.json`. The release's coordinates are used because they
  are the ones the counting used.
- `n_mapped_reads`: the library's featureCounts `Assigned` total from
  `data/k12_modulome/multiqc_stats.tsv`. This is a read, not a derivation, and it is
  verified: it equals the sum of the stored counts for 1,675 of 1,675 samples, and its
  minimum over them, 533,469, clears the paper's own floor, "At least $5 0 0 ~ 0 0 0$ reads
  mapped to coding sequences (CDS) from the reference genome (NC 000913.3)". The sibling
  dataset carries this field as a `deferred_pending_source_review` gap; here it is filled.

Gene keys go through `reconcile_locus_tags` against MG1655: 4,238 current, 114
non_gene_feature, 3 retired and kept as given (`b3036`, `b4223`, `b4590`), one remap, 0
outside the namespace. Resolved 4,352 / 4,355 = 0.99931 against the 0.99 floor.

### Genotype settling

Only MG1655 is kept. Within it, the `Strain Description` edit tokens are classified by an
explicit table (`PUBLIC_EDIT_TOKENS`, 128 distinct tokens measured over the 1,006 MG1655
cells, 59 of them `del_<gene>`) and an unlisted token raises. A row whose tokens are all
`del_<gene>` is a whole-gene deletion set; everything else is dropped with its reason, in a
fixed rule order so a row carrying two kinds of edit is reported under the one that makes it
unwritable first. Four deletion symbols the pinned annotation resolves to no b-number are
dropped rather than guessed: `ihf` names the IhfA/IhfB heterodimer rather than a gene, and
`rrnC`, `rrnD` and `rrnE` name rRNA operons. `del_7prrn` (the seven rRNA operons) is a
deletion that is not one gene.

### Environment, and rank 19

The environment is written with the sibling loader's own encoding, so one condition means
the same thing in both datasets: a loader-local `Media` derived from the `M9` or `LB`
library key, the carbon and nitrogen sources and the pH as `EnvironmentPhysicalPerturbation`,
each supplement compound as a `SmallMoleculePerturbation`. Two parsing differences are
measured rather than assumed: the public `Supplement` cells separate compounds with `;` or
`,`, never with the `+` PRECISE-1K uses (and one name, `2,2'-Dipyridyl`, carries a comma
with no space, so a bare comma is not a separator); and two unit tokens appear that no
PRECISE-1K cell of a parsed column carries, `g/L` and `ng/mL`, the second restated in the
enum's `ug/mL` by exact decimal arithmetic. `parse_amount` grew a keyword-only `extra_units`
for exactly that, which leaves the sibling's behavior unchanged.

**`aerobicity` is loaded (rank 19's loadable half).** The `Electron Acceptor` column
PRECISE-1K reads is blank on every public row, and this table carries an `aerobicity` column
instead. Measured values over the 1,675: `aerobic` 1,135, blank 250, `O2` 206,
`aerobic(30% DO)` 52, `anaerobic` 17, `transition` 15. `O2` is the same token PRECISE-1K's
own `Electron Acceptor` column uses for an aerobic culture; `aerobic(30% DO)` is a
dissolved-oxygen setpoint with no slot on `Environment`, and `transition` is a regime that
changes during the culture, so both are named drop rules rather than being flattened to
`aerobic`.

**`time` is NOT loaded; the gap is recorded instead (rank 19's other half).** The column has
no unit in its header, and `grep -i` over `paper.md` and `si/si1.md` finds no mention of it
at all. Measured on the kept rows, its 13 distinct values mix bare numbers (`12`, `0.25`,
`0.08`, `0`) with an H:MM:SS clock (`12:00:00`, `0:00:30`, `0:02:30`), so neither a unit nor
one format can be sourced. `duration_hours` is a typed `not_reported_by_primary`
`ProvenanceGap` on every record whose note states that measurement. Guessing hours would
have put a 12-hour and a 12-second sample on the same number.

### Reference

One reference for every record, the condition the release centers this dataset on: "After
centering the Public K-12 dataset to the PRECISE-1K control condition". That is the two
`control:wt_glc` samples of PRECISE-1K (wild-type MG1655, M9 + glucose), whose counts come
from the already-mirrored `data/precise1k/counts.csv` and whose environment is settled by the
sibling loader's own `settle_environment`, so the reference environment is the same object
the sibling writes. The reference TPM is the mean of the two controls' computed TPM and its
counts their mean rounded half to even; `n_mapped_reads` is a typed gap there, because the
PRECISE-1K MultiQC table is not a consumed file.

### Raw mirror

Four members are added to the shared raw mirror
`$DATA_ROOT/torchcell-raw/lamoureuxMultiscaleExpressionRegulation2023/`, and two already-
mirrored PRECISE-1K members are read for the reference. `deposit_public_raw_mirror` is
ADDITIVE: it keeps every manifest record it does not itself write, so the sibling's four pins
survive and one `manifest.json` pins all eight members.

| member | sha256 | read for |
|---|---|---|
| `data/k12_modulome/metadata_qc.csv` | `3904fb6d...` | genotype, environment, accessions |
| `data/k12_modulome/counts.csv` | `e60ac837...` | `expression_count` and the TPM |
| `data/k12_modulome/multiqc_stats.tsv` | `6cd04f09...` | `n_mapped_reads` (`Assigned`) |
| `data/annotation/gene_info.csv` | `355de115...` | each gene's `start` and `end` |
| `data/precise1k/counts.csv` | `f6307411...` | the reference's control counts |
| `data/precise1k/metadata_qc.csv` | `68a0c5aa...` | the reference's environment |

The archive re-retrieved on 2026.10.08 reproduced the pinned 278,428,285 bytes and sha256
`7c7008f2c8bcd66aebecbdb97b8c1a0314637e3873ec09e12c26d0ccaaa35172` with
`curl -sL -o precise1k-v1.0.zip https://zenodo.org/api/records/8284223/files/SBRG/precise1k-v1.0.zip/content`.
The `retrieve._UA` 403 the sibling note records is unchanged.

### Verification (2026.10.08)

`verify_rnaseq_dataset` with `replicate_aware=True` plus `shared_rule_results` with the
MG1655 resolver, run over the stored records the way `run_rnaseq` runs them.

| level | check | result |
|---|---|---|
| L0 | structural | PASS, 240 validated |
| L1 | count | PASS, 240 |
| L1 | replicate_groups | PASS, 240 distinct profiles over 52 (genotype, environment) groups, each measuring one gene set |
| L1 | provenance_gaps | PASS, 674 gaps over 240/240 records (duration_hours, pH agent, inchikey) |
| L1 | canonical_gene_names | PASS, 22 |
| L2 | tpm_value_fidelity | PASS, 1,045,200 values |
| L2 | count_value_fidelity | PASS, 1,045,200 counts |
| L2 | uncertainty_sanity | PASS |
| L3 | measurement_type_consistent | PASS, `rnaseq_tpm_from_released_counts` |
| L3 | reference_finite | PASS, 1,045,200 values |
| L3 | compound_identity | PASS, 193 structure identifiers, 140 typed gaps (17 compounds) |
| L3 | media_compound_identity, media_membership | PASS, 240 on a medium deriving from a library key (2 distinct media) |
| L4 | gene_containment_assembly | PASS, 0.999 of 4,355 against ecoli_K12_MG1655_ASM584v2 (>= 0.99) |

`strain_uniqueness` is the wrong rule here for the same structural reason as PRECISE-1K and
putidaPRECISE321 (one row per library, no `strain_id` on a bacterial deletion leaf), so the
registry entry passes `replicate_aware=True` and the group rule runs instead. The 97 release
conditions give 52 distinct (genotype, environment) pairs and 53 distinct condition-cell
tuples; the pairs that pool several conditions pool conditions whose
`CONDITION_COLUMNS` cells are identical and differ only in their project label.

### Open items (named, not guessed)

- The Public K-12 log2[TPM] matrix is not released. If the authors publish it, the stored
  `expression_tpm` becomes a read instead of a derivation and the `measurement_type` changes
  with it; that is a new build, not an edit.
- `approx_OD` (788 rows) and `growth_phase` (732 rows) are blocked: nothing on `Environment`
  or `CultureFormat` holds a harvest optical density or a growth phase.
- `dilution_rate` (92 rows) is blocked by issue #753 point 3, the same chemostat gap the
  sibling records.
- The 669 non-MG1655 rows are the real headroom: BW25113 has an assembly set in the genomes
  tier, so a BW25113-pinned loader is constructible, but the expression is keyed by MG1655
  b-numbers (reads were aligned to NC_000913.3), which is the design question to settle
  first. It is the same question the sibling's note already raises for its own 148 BW25113
  samples.
- The 11 `ph_not_stated` rows would be recoverable if pH became a gappable slot rather than
  a perturbation every sibling record carries.

## 2026.10.09 - This arm contributes no growth rate, measured

The release's `Growth Rate (1/hr)` column is now served as its own dataset
([[torchcell.datasets.ecoli.lamoureux2023_growth]]). This arm contributes NOTHING to it:
all 354 released rate cells are `p1k_*` ids of the PRECISE-1K index, so 0 of this store's
240 built records carry a rate. The audit's rank-11 ceiling of 354 therefore sits entirely
on the PRECISE-1K arm.
