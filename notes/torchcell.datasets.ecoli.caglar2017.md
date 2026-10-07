---
id: rkt0rucs0vzjnvebcvr76iy
title: Caglar2017
desc: ''
updated: 1791382773920
created: 1791382773920
---

## 2026.10.07 - Strain gate: REL606 is E. coli B, raw mirror deposited, no loader

Module: `torchcell/datasets/ecoli/caglar2017.py`. Tests:
`tests/torchcell/datasets/ecoli/test_caglar2017.py`. Row 9 of the fifty in
[[plan.bacteria-ontology-genome]]; skeleton [[torchcell.datasets.bacteria_common]].

Caglar et al. 2017 (Scientific Reports, doi:10.1038/srep45303, citation key
`caglarColiMolecularPhenotype2017`) measured mRNA (RNA-seq), protein (LC-MS/MS) and 13C
central-carbon flux ratios of one wild-type strain across 34 conditions. This branch stops
at the strain gate of the per-dataset checklist (item 3): the strain is not in the genomes
tier, so no record is built, no LMDB exists and no class is registered. The raw data are
deposited, and the module carries the finding as typed objects that a later loader calls
first.

### The strain, quoted

All quotes are verbatim from `paper.md` (sha256 `0878d5e7d49bcea4570aa2db225318a8469f4755645f1ddf73563effe7b3109b`)
or `si/si1.md` (sha256 `1b7b8ed0f8b21c1909568f8a05d6a92d99bffaf851aa474a3a8673ebc2da4bdc`)
of the mirror, and each is a `SourcedValue` that `audit_sourced_value` re-reads (26 of
them, all passing under `--data`).

- Strain (Methods, Cell Growth): `E. coli B REL606 was inoculated from a freezer stock`.
- Same stock throughout (Results): `We grew multiple cultures of E. coli REL606, from the same stock, under a variety of different growth conditions.`;
  Discussion: `used the exact same $E$ . coli genotype throughout`.
- RNA-seq reference (Methods): `we implemented a custom analysis pipeline using the REL606 Escherichia coli B genome (GenBank:NC_012967.1) as the reference sequence`.
- Proteome database (Methods): `Spectra were searched against an $E _ { \ast }$ . coli strain REL606 protein sequence database`.
- Identifier forms (SI, Table S8 description): `Gene id (ECB number for mRNA and YP number for proteins), and corresponding gene name`.

REL606 is an *E. coli* **B** strain. The tier holds K-12 MG1655, K-12 BW25113 and KT2440
(`BacterialReferenceStrain`), and neither K-12 set is REL606's genome. Pinning the records
to MG1655 would assert that an `ECB_` locus is a b-number, the cross-strain inference D5
and D9 refuse. So this row is the "strain whose genome is not in the tier" case of the
plan's section 3c, and it is answered with a typed gap rather than a label.

### Typed objects the module exposes

| name | what it is |
|---|---|
| `strain_pin_finding()` | `StrainPinFinding`: strain `REL606`, lineage `E. coli B`, the tier's strains read from `get_args(BacterialReferenceStrain)`, `pinnable=False`, the gap and the tier addition |
| `require_pinnable_strain()` | the gate a Caglar loader calls before building anything; raises `UnpinnedStrainError` (carrying the finding) until `REL606` is in the schema vocabulary, then returns the finding |
| `STRAIN_GAP` | `ProvenanceGap(field="genome_reference", reason=deferred_pending_source_review)`, `looked_in` the paper, `resolve_with` the GCA assembly directory and the sha256 of its GenBank flat file |
| `REL606_TIER_ADDITION` | `AssemblyTierAddition`: the exact assembly set, members, pattern and edits below |

`deferred_pending_source_review` is the only recoverable member of `ProvenanceGapReason`,
and the gap is recoverable: the paper reports its genome, the tier lacks it. The enum's
docstring describes that member as an unfinished SI comb, so a dedicated member (for
example `reference_assembly_not_in_tier`) would describe this gap more exactly; that is a
`torchcell/verification/sourced.py` change, outside this branch.

### The tier addition that opens the gate

NCBI assembly ASM1798v1, measured on 2026-10-07 from the assembly report: organism
`Escherichia coli B str. REL606 (E. coli)`, GenBank `GCA_000017985.1`, RefSeq
`GCF_000017985.1`, one replicon CP000819.1 / NC_012967.1 of 4,629,812 bp, and
`RefSeq assembly and GenBank assemblies identical: yes`. Every file's md5 matched its
directory's `md5checksums.txt`.

| member | bytes | sha256 |
|---|---|---|
| `GCA_000017985.1_ASM1798v1_genomic.gbff.gz` | 3,239,604 | `aacf2559815f959c9417984ce1632228fd94caeac4b62b7910f714e310542e6b` |
| `GCA_000017985.1_ASM1798v1_genomic.fna.gz` | 1,375,449 | `070a03fc2e2813853d5327608ee3ebcb4b0b2fe7faa239169921b3362b24adfa` |
| `GCA_000017985.1_ASM1798v1_genomic.gff.gz` | 270,806 | `b928f83a99ea3ec7e64137f36490c37aa4585689de1abb884ff9cbaa4e1199d5` |
| `GCA_000017985.1_ASM1798v1_protein.faa.gz` | 889,482 | `40f1748bf2e86f5a43d7a8bb1515a0b3812f66f27f4d7fb9dc62a0c348962663` |
| `GCA_000017985.1_ASM1798v1_feature_table.txt.gz` | 173,353 | `5cba47c018f4a5180eb1af9f06e4b9103837f894a08f05fb5f6f70e8795379ec` |
| `GCA_000017985.1_ASM1798v1_assembly_report.txt` | 1,172 | `51968f440a6497669ad8ccf703c437d5a8055990d2c7e27194b9cc1ffeeda369` |
| `GCF_000017985.1_ASM1798v1_genomic.gbff.gz` | 3,428,153 | `b90a8ab7a8f1e9e736952b6e17017a9cd6bc6567cb11b1ecdc2e7c895695a26b` |
| `GCF_000017985.1_ASM1798v1_genomic.gff.gz` | 433,859 | `27c302a37ac517de79999cc8438c744367e5c12ad4ac60b34f55bfd753214f25` |
| `GCF_000017985.1_ASM1798v1_gene_ontology.gaf.gz` | 158,466 | `4cbd6f5767d0f8651346891af174eaf9fd3353c25b6916fdee1f3d6c399374b1` |

URLs: `https://ftp.ncbi.nlm.nih.gov/genomes/all/GCA/000/017/985/GCA_000017985.1_ASM1798v1/`
and `https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/017/985/GCF_000017985.1_ASM1798v1/`.
The RefSeq GAF is a member because `provision_bacterial_genomes.py` requires every
`_gene_ontology.gaf.gz` a listing names to be one; it has 5,246 rows over 2,188 objects,
all IEA (measured with `zcat | awk` on the downloaded file).

Locus tags, parsed with Biopython from the GenBank flat file: 4,383 gene features, all
`ECB_\d{5}` (4,276), `ECB_t\d{5}` (85) or `ECB_r\d{5}` (22), so the proposed pattern
`^ECB_[rt]?\d{5}$` matches 4,383 of 4,383 and is disjoint from the three deposited
namespaces and from yeast systematic names. 67 of the gene features are pseudogenes. The
RefSeq file retags to `ECB_RS\d+` (4,507 genes, 4,253 carrying an `old_locus_tag`), and
its CDS products are `WP_` proteins; the GenBank file's CDS products are `ACT` proteins.
So, as for BW25113 and KT2440, the GenBank flat file is the one that carries the
identifiers the paper reports.

Edits needed, in the order the bacterial plan used for the first three sets (none made
here; `schema.py`, `bacteria_common.py` and the tier are outside this branch):

1. `torchcell/sequence/genome/registry.py`: `ECOLI_B_REL606 = "ecoli_B_REL606_ASM1798v1"`.
2. `scripts/provision_bacterial_genomes.py`: the REL606 set of the nine members above,
   fetched by `direct_url`, md5-checked, deposited with `deposit_assembly_set`.
3. `torchcell/datamodels/schema.py`: `"REL606"` in `BacterialReferenceStrain`;
   `BACTERIAL_ASSEMBLY_SETS["REL606"]`; the set id in `BacterialAssemblySet`;
   `ASSEMBLY_SET_ACCESSIONS[set] = ("GCA_000017985.1", "GCF_000017985.1")`;
   `"ecoli_b_rel606_locus_tag"` in `BacterialGeneNamespace` with
   `^ECB_[rt]?\d{5}$` in `BACTERIAL_LOCUS_TAG_PATTERNS`. These are Literal and dict
   widenings of bacterial-only symbols; no served yeast dataset imports them, so the
   expectation is 0 of 36 served closures (hypothesis until `schema_impact` is run on
   that change).
4. `torchcell/sequence/genome/ecoli/`: a REL606 genome class beside `EcoliK12Genome`
   (the K-12 classes and the ECK crosswalk do not apply to a B strain).
5. `torchcell/datasets/bacteria_common.py`: REL606 in `HOST_STRAINS["ecoli"]`,
   `STRAIN_GENE_NAMESPACES` and `BACTERIAL_GENOME_CLASSES`.
6. `torchcell/verification/runners.py`: a REL606 gene universe beside
   `_ecoli_k12_gene_set` (4,383 GenBank gene features).

The numbers above (gene features, prefixes, pseudogenes, pattern matches, RefSeq retagging
and the GAF counts) are printed by the module's own `measure` command,
`annotation_summary`, on the downloaded files.

### Data home and replicate structure (sourced for the loader the gate is waiting on)

The paper names its processed data: `final processed data are available as Supplementary Tables S2, S3 and S4.`
(the OCR separates those words with no-break spaces, kept verbatim in the quote). Raw reads
are GEO `GSE94117`, raw spectra PRIDE `PXD005721`, raw GC-MS the Texas Data Repository
(doi:10.18738/T8/UG3TUR); none of the three is a processed table, so none is mirrored, and
the manifest's `si_expected` says so.

| table | PMC object | content (measured on the mirror) |
|---|---|---|
| S1 `tableS1_meta_data.csv` | `srep45303-s2.csv` | 171 samples x 22 columns: carbon source, `Mg_mM`, `Na_mM`, growth phase, batch, `RNA_Data_Freq`, `Protein_Data_Freq`, doubling time |
| S2 `tableS2_mRNA_normalized_raw_data.csv` | `srep45303-s3.csv` | 4,196 `ECB_` genes x 152 samples |
| S3 `tableS3_protein_normalized_raw_data.csv` | `srep45303-s4.csv` | 4,196 `YP_` proteins x 105 samples |
| S4 `tableS4_fluxData.csv` | `srep45303-s5.csv` | 260 rows: 13 branches x salt x concentration x phase, `MeanFluxRatio`, `SDEFluxRatio` |

PMC object `-s<N+1>` is Table S<N> because the SI PDF is `-s1`; the shapes agree with the
SI's `Includes data for 4196 distinct proteins each for 152 samples.` (S2) and `... for 105
samples.` (S3), and the S2 and S3 sample columns are exactly the S1 rows with
`RNA_Data_Freq > 0` (152) and `Protein_Data_Freq > 0` (105).

Replicate structure, each a `SourcedValue`:

- `For each experimental condition, bacteria were grown in three biological replicates.`
  (Fig. 1 legend) and `Each of the three biological replicates was performed on a separate day.`
  (Methods). S2 and S3 are per sample, one column per culture, not condition means, so an
  RNA or protein record would be one sample with `n_samples` 1, or a condition aggregate
  the loader computes and says it computed.
- Technical replicates: S1 columns `number of RNA samples (technical replicates), number of protein samples (technical replicates)`.
  Measured: RNA 1 for all 152 RNA samples; protein 1 for 93 and 2 for 12 of the 105.
- Flux: `For each condition, flux samples were analyzed in triplicate (except one, which was analyzed in duplicate only), and 13 different flux ratios were measured for each sample.`
  and `The flux ratios were then averaged across replicates`; each vial had `three technical replicates of each vial`.
  Which condition had two is not named in the mirror, so the conservative `n_samples` for
  every flux record is 2 by the range rule (lower end), unless the GC-MS deposit resolves it.
  `SDEFluxRatio` is not defined anywhere in the mirror (SD or SE is open).
- Doubling time: `Means and confidence intervals were calculated from three replicate growth curves for all conditions except for gluconate and lactate, which had measurements for only two replicates.`

Normalization of S2 and S3: `we normalized read counts using size-factors calculated via $\mathrm { D E S e q } 2 ^ { 1 7 }$`,
after `All resulting data sets were checked for quality, normalized, and log-transformed.`
The base of the log is not stated in the mirror and the Methods defer to ref 10 (the
glucose-starvation paper, Houser et al. 2015), which is not mirrored. That is a gap for the
`measurement_type` unit until ref 10 or the authors' GitHub pipeline is mirrored. S2's range
is -1.14 to 15.78 and S3's 0.968 to 11.52; S3's floor value 0.968021139095309 repeats for
unobserved proteins (`We set the counts of all unobserved proteins to zero.`).

Environment: `DM500` and `DAVIS_MINIMAL` are already in `MEDIA_LIBRARY` (Lenski 1991 salts
deferral recorded in [[torchcell.datamodels.media]]). The varied factors are quoted on the
module: the carbon swap `supplemented with $0 . 5 { \mathrm { g } } / { \mathrm { L } }$ of the specified compound (glycerol, lactate, or gluconate) instead of glucose.`,
the Mg2+ series from `the concentration of $0 . 8 3 \mathrm { m M }$ that is normally present`
(S1's base level reads 0.8), and the Na+ series from the base `${ \sim } 5 \mathrm { m M N a ^ { + } }$`
of sodium citrate plus added NaCl. S1 levels: Mg 0.005 to 400 mM, Na 5 to 300 mM, carbon
glucose 126, glycerol 27, lactate 6, gluconate 6, `base` 6 (the pilot rows, which have no
RNA or protein data).

### Per-family design for the loader (not built)

- **RNA**: `BacterialRNASeqExpressionExperiment` with `RNASeqExpressionPhenotype`, one
  record per S2 sample, genotype empty (`reference-only`), environment DM500 plus typed
  carbon, Mg2+ and Na+ factors and the growth phase. Identifiers are `ECB_` tags, 4,196 of
  4,196 GenBank locus tags (below), so `reconcile_locus_tags` against a REL606 genome would
  be all `CURRENT`.
- **Protein**: `BacterialProteinAbundanceExperiment`, one record per S3 sample, keyed by the
  `ECB_` tag the NCBI record of each `YP_` accession names.
- **Flux**: S4 holds flux RATIOS (fraction of a metabolite made through a branch, for
  example `OAA from PEP` 0.403), not net fluxes and not metabolite abundances. The plan maps
  this row to `MetabolitePhenotype`, which stores pool sizes, and `FluxPhenotype` stores a
  signed net flux with interval bounds. Neither is a ratio, so the flux arm is a schema
  question for its loader (a ratio phenotype, or `FluxPhenotype` with a ratio
  `measurement_type`), not something to force into either class. S4 also carries
  stationary-phase rows although the Results say `We here analyzed only flux samples taken in exponential phase`.

### Identifier coverage, measured

`python -m torchcell.datasets.ecoli.caglar2017 measure --genbank-gbff <GCA gbff> --refseq-gbff <GCF gbff> --refseq-gaf <GCF gaf>`
reads the deposited raw mirror and the ASM1798v1 files (downloaded to session scratch, not
deposited) and prints `IdentifierCoverage` then `AnnotationSummary`. Output on 2026-10-07:

| measure | value |
|---|---|
| Table S2 ids / of form `ECB_` | 4,196 / 4,196 |
| Table S2 ids that are GenBank (CP000819.1) locus tags | **4,196 of 4,196** |
| Table S2 ids that are a RefSeq `old_locus_tag` | 4,090 |
| Table S3 ids / of form `YP_` | 4,196 / 4,196 |
| Table S3 ids in the current GenBank or RefSeq CDS `protein_id`s | **0** and **0** |
| Table S3 ids whose NCBI protein record names a `/locus_tag` | 4,196 |
| ... whose named tag is a GenBank locus tag | 4,196 |
| ... whose named tag equals the `ECB_` id on the same row of Table S2 | **4,196** |
| paper ids matching a deposited namespace (MG1655, BW25113, KT2440) | 0, 0, 0 |
| Table S2 ids matching the proposed REL606 pattern | 4,196 |

So the two tables are row-aligned gene for gene, which is what the Methods state
(`This resulted in 4196 matching mRNA and protein counts for each sample.`), and the
protein table reaches the GenBank locus tags only through NCBI's records of the retired
`YP_` accessions (`DBSOURCE REFSEQ: accession NC_012967.1`, `/locus_tag="ECB_00001"` for
`YP_003043230.1`). Those records are deposited, so the join no longer depends on NCBI
continuing to serve suppressed RefSeq proteins. Status histogram the loader would report
through `reconcile_locus_tags` on a REL606 genome: not run (no REL606 genome class exists);
the coverage above predicts 4,196 `CURRENT` for both tables, which is a hypothesis until
that genome is built.

### Raw mirror, deposited

`$DATA_ROOT/torchcell-raw/caglarColiMolecularPhenotype2017/`: 46 files plus `manifest.json`,
35 MB.

- `data/srep45303-s2.csv` to `data/srep45303-s5.csv` (Tables S1 to S4), retriever
  `torchcell.literature.retrieve.pmc_cloud_object` with key `PMC5394689.1/<object>`. The
  bytes equal the paper mirror's `si/si2.csv` to `si/si5.csv` (same sha256 pins).
- `ncbi_protein/yp_batch_00.gp` to `yp_batch_41.gp`: GenPept records of Table S3's 4,196
  `YP_` accessions, 100 per batch in table order, retriever
  `torchcell.literature.retrieve.direct_url` on
  `https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=protein&rettype=gp&retmode=text&id=<ids>`.
  The URLs are a function of Table S3's pinned first column (`yp_batch_specs`). A second
  retrieval of batches 0 and 41 was byte-identical. Batches of 100 took 38 to 85 s against
  the retriever's 120 s timeout (200 took about 85 to 95 s, so 100 was chosen).

Command: `python -m torchcell.datasets.ecoli.caglar2017 deposit --staging <dir>`, which runs
`retrieve_raw_files` (each recorded retriever, pin checked; a staged file already holding
its pin is not fetched again) and then `deposit_raw_mirror`. The deposit checks every
staged file, every existing mirror file and an existing manifest before writing anything,
refuses a differing file or a manifest recording other files, and is idempotent: a second
run left the tree hash (`03506b35...`) and the manifest's mtime unchanged.

### Gaps carried forward

1. **`genome_reference`** (`STRAIN_GAP`): blocks every record until the REL606 set is in
   the tier and `REL606` is in the schema vocabulary.
2. **Log base of Tables S2 and S3**: not stated in the mirror; deferred to Houser 2015
   (ref 10), not mirrored. Needed for each family's `measurement_type`.
3. **`SDEFluxRatio`**: undefined in the mirror (SD or SE). And which flux condition had two
   replicates rather than three is not named.
4. **Flux ratios have no phenotype class** (above). A schema question for the flux arm.
5. **`ProvenanceGapReason`** has no member for "the reported genome is not in our tier";
   `deferred_pending_source_review` is used and its docstring is narrower than this use.

### Per-dataset checklist (plan section 4)

1. **Pin the paper.** `paper.md` sha256 `0878d5e7...`, `si/si1.md` `1b7b8ed0...`; 26
   `SourcedValue`s, all passing `audit_sourced_value` under `--data`.
2. **Exact tables and columns.** S1 (sample sheet: `carbonSource`, `Mg_mM`, `Na_mM`,
   `growthPhase`, `RNA_Data_Freq`, `Protein_Data_Freq`), S2 and S3 (one value column per
   sample, first column the identifier), S4 (`MeanFluxRatio`, `SDEFluxRatio`). Nothing is
   consumed yet.
3. **Background strain.** REL606, *E. coli* B (quotes above). Not pinnable; typed gap.
   Stopped here.
4. **Identifiers.** Coverage above. `reconcile_locus_tags` not run (no REL606 genome).
5. **`n_samples` and uncertainty type.** Sourced above: S2 and S3 are single biological
   replicates per column (`n_samples` 1 per sample record); flux means over 3 (one
   condition 2), `SDEFluxRatio` type open.
6. **Media.** `DM500` / `DAVIS_MINIMAL` exist; carbon, Mg2+ and Na+ quoted as varied
   factors. Not applied (no records).
7. **Superset.** Not measured. Hypothesis (unverified): no other row of the fifty carries
   these REL606 samples, since the other *E. coli* rows are K-12; the PRECISE-1K
   compendium (rank 5) is the one to check before a loader lands.
8. **Dendron note.** This note.
9. **LMDB.** None. No class is registered, so `len(dataset)`, drop counts, the build
   manifest and the L0 to L4 verifier do not apply.
