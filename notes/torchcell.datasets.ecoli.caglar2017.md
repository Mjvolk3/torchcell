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

## 2026.10.07 - The strain gate is open

The six edits listed above are made on branch `feat/rel606-genome`
([[torchcell.sequence.genome.ecoli.rel606]], [[scripts.provision_bacterial_genomes]]):
the `ecoli_B_REL606_ASM1798v1` set is deposited (all nine members equal to
`REL606_TIER_ADDITION` in bytes, sha256 and NCBI md5), `REL606` is in
`BacterialReferenceStrain`, and `EcoliBREL606Genome` reads it. This module is unchanged;
its gate now reports `pinnable=True` with no gap, so `require_pinnable_strain()` returns
instead of raising. The tests that pinned the closed gate now pin it by patching the
strain vocabulary back to the three earlier strains.

The hypothesis under "Identifier coverage" is now measured: `reconcile_locus_tags` of
Table S2's 4,196 `ECB_` ids against the REL606 genome gives 4,196 `CURRENT` at the
locus-tag layer, none remapped and none outside the `ecoli_b_rel606_locus_tag`
namespace (`test_every_table_s2_id_is_a_current_rel606_gene`, `--data`). Table S3 (YP_
proteins) was not reconciled here; its route to the same tags is the NCBI-record join
measured above.

Gaps 2 to 5 of the list above are unchanged. Gap 1 (`genome_reference`) is closed: a
record can store `assembly_reference("REL606")`, which pins `GCA_000017985.1`.

## 2026.10.07 - The RNA-seq and proteome loaders

Two registered classes in `torchcell/datasets/ecoli/caglar2017.py`, both wild-type REL606
against `assembly_reference("REL606")` (`ecoli_B_REL606_ASM1798v1`, `GCA_000017985.1`),
`REFERENCE_STRAIN = "REL606"`, genome injected as `ecoli_genome`:

| class | root | source | records | phenotype |
|---|---|---|---|---|
| `RnaseqCaglar2017Dataset` | `data/torchcell/rnaseq_caglar2017` | Tables S1 + S2 | 152 | `RNASeqExpressionPhenotype`, `rnaseq_tpm` |
| `ProteomeCaglar2017Dataset` | `data/torchcell/proteome_caglar2017` | Tables S1 + S3 + 42 NCBI batches | 105 | `ProteinAbundancePhenotype`, `lcmsms_spectral_count_deseq2_size_factor_normalized` |

The flux arm (Table S4) is not loaded; the schema proposal it needs is at the end.

### The log base: back-solved, not quoted

The mirror never states the base. The paper says `All resulting data sets were checked for quality, normalized, and log-transformed.`
and the Methods defer to ref 10, which is not mirrored. What the Methods do state is the
normalization: `we added pseudo-counts of $+ 1$ to all counts before calculating size factors.`
and `We then used those size factors to normalize the original raw counts (i.e., without pseudo-counts).`

`back_solve_counts` settles the base from the released numbers. Every sample's minimum is
the same value (Table S2 `-1.14456265985177`, Table S3 `0.968021139095309`), which is what
a variance-stabilizing transform of a zero count looks like (it depends only on the
normalized count, so zero maps to one value in every sample). DESeq2's parametric VST,
written through that floor `P = 2**floor`, inverts as `q = (2**y - P)**2 / 2**y`. Taking
each sample's smallest nonzero `q` as a count of 1 gives its size factor; every other cell
is then `q` times that factor. Measured on the mirror (`--data` tests pin these):

| | Table S2 (mRNA) | Table S3 (protein) |
|---|---|---|
| features x samples | 4,196 x 152 | 4,196 x 105 |
| zero cells (at the floor) | 185,331 | 203,214 |
| worst distance of a reconstructed count from an integer | 8.3e-9 | 1.1e-10 |
| worst relative gap, factor vs DESeq2's +1 median-of-ratios size factor of the reconstructed counts | 1.5e-14 | 1.6e-14 |
| size factors | 0.210 to 13.87 | 0.499 to 4.98 |
| library size (sum of counts) min / median / max | 3,271 / 177,377 / 5,005,764 | 1,107 / 64,241 / 198,696 |

The same inverse in base e or base 10 leaves cells 0.5 from an integer (max deviation
0.49999 and 0.5), and plain `log2(q + P)` or `ln(q + P)` do too. So the tables are the
base-2 DESeq2 VST of size-factor-normalized integer counts, and the size factors are the
+1-pseudocount ones the Methods describe; for large counts the VST is `log2(q)`, so the
base is 2. That the authors called DESeq2's `varianceStabilizingTransformation` and not a
hand-written function of the same form is inference; the functional form and the integer
counts are measured. The build refuses a table that misses either check
(`COUNT_INTEGER_TOLERANCE`, `SIZE_FACTOR_TOLERANCE`, both 1e-6) and writes
`preprocess/vst_back_solve.json` and `preprocess/log_base_derivation.json` (a
`StatDerivation`, method `back_solve`).

### What is stored

- **RNA-seq.** `expression_count` is the reconstructed integer count of each of the 4,196
  coding genes (`The raw number of reads mapping to each gene were counted using HTSeq`;
  `For RNA, we only analyzed the counts of reads that overlapped annotated protein coding genes, i.e., reads mapping to mRNAs.`).
  `expression_tpm` is computed here: counts per base over the GenBank gene-feature span of
  each locus in the REL606 genome (all 4,196 are single-part, non-pseudo CDS genes, 66 to
  7,152 bp), scaled to sum to one million. The paper reports no TPM; this one is a
  derivation from the reconstructed counts, recorded in `gene_lengths.json`.
  `n_mapped_reads` is a typed gap (`deferred_pending_source_review`, resolve with GEO
  GSE94117): the counts are reads on coding genes, not the library's mapped reads.
- **Proteome.** `protein_abundance` is the reconstructed integer spectral count over the
  sample's size factor, the "Normalized protein counts" the SI names
  (`Supplementary Table S3: Normalized protein counts.`), keyed by the `ECB_` tag each
  `YP_` accession's deposited NCBI record names (`protein_crosswalk.json`). An unobserved
  protein is the 0 the source wrote (`We set the counts of all unobserved proteins to zero.`).
  The VST values themselves are not stored: they depend on a dispersion fit over the whole
  table, and they put an unobserved protein at 0.968.
- **Replicates.** A record is one biological-replicate culture (`For each experimental condition, bacteria were grown in three biological replicates.`),
  so the protein `n_replicates` is 1 and `protein_abundance_se` is `None`. Table S1's
  technical-replicate counts (RNA 1 for all 152; protein 1 for 93 and 2 for 12) are in
  `record_samples.json`; the paper does not say whether two technical runs were summed or
  averaged before the counts, and the size factor absorbs depth either way. The replicates
  of a condition are listed in `replicate_groups.json` (by Table S1 `uniqueCondition`,
  with each record's collection time and batch).
- **QC.** The authors kept every sample, including the two RNA samples they flagged
  (`MURI_091`, `MURI_130`), and so does the loader: no record is dropped, and
  `build_accounting.json` reads `drops_by_reason: {}` for both families. A Table S1 cell
  the loader cannot read raises instead.

### Identifiers

`reconcile_locus_tags` against `EcoliBREL606Genome`, both families:

| | Table S2 genes | Table S3 proteins (after `YP_` to `ECB_`) |
|---|---|---|
| unique names | 4,196 | 4,196 |
| current / renamed / non-gene / retired / ambiguous | 4,196 / 0 / 0 / 0 / 0 | 4,196 / 0 / 0 / 0 / 0 |
| layer: locus tag | 4,196 | 4,196 |
| remapped, outside `ecoli_b_rel606_locus_tag` | 0, 0 | 0, 0 |

`MIN_RESOLVED_FRACTION` is 1.0 for both.

### Environment

Every record is wild-type REL606 (an empty `Genotype`); the conditions are environment
edits read from the sample's Table S1 row:

- carbon: glucose is `DM500`; glycerol, lactate and gluconate are `DAVIS_MINIMAL` plus an
  `EnvironmentPhysicalPerturbation(factor=carbon_source)` at 0.5 g/L (the compound through
  `resolved_compound`; `gluconate` carries the resolver's deferred InChIKey gap);
- Mg2+: a non-base level is a `SmallMoleculePerturbation` of magnesium sulfate at Table
  S1's `Mg_mM`, its `description` saying it replaces the 0.83 mM DM level (Table S1 writes
  the base as 0.8, and the loader refuses a `baseMg` row at another number);
- Na+: NaCl added at `Na_mM - 5` mM (`so $9 5 \mathrm { m M N a C l }$ was added for the $1 0 0 \mathrm { m M N a ^ { + } }$ condition, for example`);
- 37 C, aerobic (orbitally shaken flasks), `duration_hours` = `growthTime_hr`
  (`the growth time at which the sample was collected`).

Records use 57 distinct environments (RNA) and 40 (protein); the perturbation counts
(RNA: 60 Mg, 37 carbon, 16 NaCl; protein: 18, 39, 11) are the Table 1 sums.

**Growth phase has no slot.** `Environment` has no growth-phase field and `PhysicalFactor`
no growth-phase member, so the phase is carried by the collection time, by which
reference a record points at, and by the ledgers. A typed growth phase is a
`schema.py` change (proposal in the PR body), outside this branch.

**Reference.** `The reference conditions always had glucose as carbon source and base $\mathrm { N a ^ { + } }$ and $\mathbf { M } \mathbf { g } ^ { 2 + }$ concentrations.`,
one per phase. Each record points at its phase's glucose, base Mg2+, base Na+ condition
(the late-stationary one applies the same rule to the third phase, which the paper's
DESeq2 contrasts did not use). The reference phenotype averages that condition's samples
(RNA: mean TPM, mean count rounded half to even; protein: mean, standard error, sample
count). Members: RNA 21 / 12 / 6 (exponential / stationary / late stationary), protein
20 / 11 / 6. The reference environment has no `duration_hours`; a typed gap says why (the
pooled samples were collected at several stated times, and the paper sets the phase by
optical density).

**Dataset gene set.** No genotype names a gene, so `ExperimentDataset.compute_gene_set`
would return the empty set, which the base class refuses. Both classes override it to the
4,196 loci their phenotypes are keyed by (`measured_gene_set`).

### Cross-check against the paper's Table 1

Table S1's samples with data reproduce the paper's Table 1 `# samples` column exactly
(`test_the_sample_sheet_reproduces_the_paper_s_table_1_sample_counts`):

| | exponential / stationary / late stationary | glucose / glycerol / lactate / gluconate | low / base / high Mg | base / high Na |
|---|---|---|---|---|
| mRNA | 79 / 63 / 10 | 115 / 25 / 6 / 6 | 36 / 92 / 24 | 136 / 16 |
| protein | 56 / 37 / 12 | 66 / 27 / 6 / 6 | 6 / 87 / 12 | 94 / 11 |

### Build and verification (dev tree)

`python -m torchcell.database.build_dataset_lmdb --dataset <Class>`: RNA-seq 152 records,
gene set 4,196, 3 references, 19 MB; proteome 105, 4,196, 3, 13 MB. Both read `fresh`
under `torchcell.provenance.build_manifest`. A first RNA-seq build failed on the empty
gene set (above); that partial tree is in `/scratch/projects/torchcell-deprecated/2026-10-07_112847__rnaseq_caglar2017/`.

`python -m torchcell.datasets.ecoli.caglar2017 verify --family {rnaseq,proteome}` (the
module's own verifiers; the RNA-seq family verifier's L1 strain uniqueness fails on
replicate-level records by design), both PASS:

- RNA-seq: L0 152 validated; L1 count 152/152, 152 distinct count profiles; L2 637,792
  TPMs finite and non-negative, counts non-negative integers, 152/152 sum to one million;
  L3 one measurement type, 3/3 references finite and summing to one million, assembly pin
  `ecoli_B_REL606_ASM1798v1` / `GCA_000017985.1`, back-solve within tolerance; L4 4,196 of
  4,196 loci are REL606 gene rows.
- Proteome: the protein family verifier (L0 105, L1 count, L2 440,580 finite values, L3
  reference finite and key-matched, one measurement type) plus 105 distinct profiles,
  abundances non-negative, the assembly pin, the back-solve, and L4 4,196 of 4,196.
- Both: 45 of 45 `SourcedValue`s re-read verbatim from the sha256-pinned mirror.

### Checklist (plan section 4)

1. Paper pinned (`paper.md` `0878d5e7...`, `si/si1.md` `1b7b8ed0...`); 45 `SourcedValue`s.
2. Tables and columns: S1 `dataSet`, `experiment`, `growthTime_hr`, `batchNumber`,
   `carbonSource`, `Mg_mM`, `Mg_mM_Levels`, `Na_mM`, `Na_mM_Levels`, `growthPhase`,
   `uniqueCondition`, `RNA_Data_Freq`, `Protein_Data_Freq`; S2 and S3 every sample column;
   the GenPept `/locus_tag` of each `YP_` record.
3. Strain: REL606, pinned (gate open since `feat/rel606-genome`).
4. Identifiers: histogram above, 4,196 CURRENT in both.
5. `n_samples` and uncertainty: one culture per record (protein `n_replicates` 1, no SE);
   reference aggregates state their sample count and SE. Log base back-solved.
6. Media: `DM500` / `DAVIS_MINIMAL` with carbon, Mg2+ and Na+ as typed edits.
7. Superset: measured, none. PRECISE-1K's 1,035 samples are MG1655 582, GMOS 241,
   BW25113 160, DGF-298 26, W3110 26, and no strain description names REL606 or an E. coli
   B strain. The glucose time-course columns are ref 10's samples, processed with the
   rest (`Results from one of these conditions, long-term glucose starvation, have been presented previously10.`);
   ref 10 has no loader.
8. Note: this section.
9. LMDBs built, manifests fresh, L0 to L4 pass.

### Gaps carried forward

1. **Growth phase** has no typed slot (schema proposal in the PR).
2. **Flux ratios** (Table S4) have no phenotype class; proposal in the PR. `SDEFluxRatio`
   is still undefined in the mirror, and which flux condition had two replicates is not
   named.
3. **`n_mapped_reads`**: typed gap; GEO GSE94117 is not mirrored.
4. **Gluconate** has no InChIKey in the compound table (the resolver's deferred gap).
5. **Technical replicates** of the 12 two-run protein samples: summed or averaged is not
   stated.
6. `ProvenanceGapReason` still has no member for a genome missing from the tier (gap 5 of
   the first section; moot now that REL606 is deposited).

## 2026.10.07 - Tables S5 to S14, and why the doubling times are not loaded

The loader pins `SI_TABLES` for S1 to S4 only, and nothing in this note or the loader
addressed Tables S5 to S14. The SI phenotype audits
(`[[plan.bacteria-si-phenotype-audit-ecoli]]`, `[[plan.bacteria-si-phenotype-audit-pputida]]`)
enumerated all fourteen; this section records the Table S5 decision, which is the only one
of the ten that looked loadable.

Measured by
`experiments/036-dataset-fixes-before-kg-build/scripts/caglar2017_doubling_time_loadability.py`,
results under that experiment's `results/`. Both inputs are sha256-verified against the
library mirror manifest before being read:

- `si/si6.csv` = Supplementary Table S5, sha256
  `76411accacbdc28310622cc15289b65ad44937bdc915051bbcfc8c4da1b04c60`, PMC object
  `srep45303-s6.csv`. Not in the raw mirror, which holds `-s2.csv` to `-s5.csv` only.
- `si/si2.csv` = Supplementary Table S1, sha256
  `1486290bf6a340ae64ee20c915435c0a00ff5eede489de1f56b62733b66f8940`, already pinned as
  `SI_TABLES["S1"]`.

### What Table S5 holds

Caption, `si/si1.md:174`, verbatim: "Supplementary Table S5: Doubling time measurements in
exponential phase. Includes the mean, $\pm 9 5 \%$ confidence interval, and $r ^ { 2 }$
from the linear fit to OD600 values."

The file is **per replicate**, not per condition: columns
`name,replicate,doubling.time.minutes,doubling.time.minutes.95m,doubling.time.minutes.95p,r.squared`,
**55 data rows over 19 conditions**, each row carrying its own `r.squared`. Methods, Cell
Growth, verbatim: "Doubling times were calculated as $\log _ { \mathrm { e } } 2$ divided
by the fit slope for each biological replicate separately. Means and confidence intervals
were calculated from three replicate growth curves for all conditions except for gluconate
and lactate, which had measurements for only two replicates." Replicate counts in the file
are 3 for 17 conditions and 2 for exactly `Gluconate.tab` and `Lactate.tab`, so the
loader's existing `DOUBLING_TIME_REPLICATES` quote is verified against the data.

The condition mean and the confidence interval OF that mean are not released as numbers;
they are what Fig. 2 draws. `si6.csv` also holds an internal duplicate:
`MgSO4-2_000.020_mM.tab` replicate 3 and `MgSO4-2_000.040_mM.tab` replicate 3 are identical
in all four numeric columns (64.6719153, 58.70481327, 71.98933404, 0.994667398), so 55 rows
hold 54 distinct fits.

### Why it is not loaded

1. **The interval is asymmetric in 55 of 55 rows.** Upper-to-lower half-width ratio runs
   -26.3920 to 10.1066, median 1.3187; median absolute asymmetry 2.5783 min. `Glycerol.tab`
   replicate 1 reports `95p` = -1027.769034, a negative doubling-time upper bound, which is
   what a slope interval straddling zero becomes under `DT = log_e 2 / slope`.
   `UncertaintyType.ci95` is a single half-width, so recording either side would falsify the
   source; the upper side derives `environment_response_se` = -565.6845.
   `FluxPhenotype` is the in-repo precedent for the lossless shape
   (`net_flux_lower` / `net_flux_upper` / `confidence_level`, `label_statistic_name = None`);
   `EnvironmentResponsePhenotype` has no equivalent.
2. **The absolute value has no home that verifies.** `MeasurementType.growth_rate` covers
   "absolute or normalized growth rate / doubling time", but the environment-response
   verifier's L3 `reference_zero` requires the reference's `environment_response` to be 0,
   and the base condition's doubling time is 53.25 min. Measured FAIL at both grains, plus
   `environment_perturbed` (3 of 19 conditions ARE the unperturbed base condition, run in
   three separate experiments) and `pair_uniqueness` (those 3, plus `MgSO4_000.080_mM`
   against `MgSO4-2_000.080_mM`).
3. **The ratio form would store 11 of 19 conditions.** The `MgSO4_stress_low` series
   releases no base-Mg2+ curve, so 5 conditions have no in-experiment reference, and the
   three released base measurements span 0.2174 log2 (53.2538, 61.9140, 58.3515 min), so
   borrowing one is not neutral. A doubling-time ratio is also a quantity the paper never
   released.

Both missing capabilities are fields on `EnvironmentResponsePhenotype`, which moves the
schema closure of every served environment-response dataset, so the fix is a full KG
rebuild and not an incremental admission. Filed as #776; nothing was loaded.

### Table S1's doubling time is a second fit, not a summary of Table S5

`si2.csv`'s 171 rows carry a doubling time on 165 (the 6 blanks are `pilot_24_hour` and
`pilot_mid-log`), and those 165 rows hold **19 distinct
`(doublingTimeMinutes, .95m, _95p, rSquared)` tuples** that join **1:1** onto Table S5's 19
conditions on `(experiment, carbonSource, Mg_mM, Na_mM)` and cover all 165 rows. So the
sample-level column is one condition-level fit repeated per sample, and Tables S1 and S5
must never both be loaded as phenotypes.

It is not an aggregate of Table S5, however: 0 of 19 values equal the arithmetic mean of
that condition's replicates (median absolute difference 0.5571 min, max 6.5763), 0 of 19
the geometric mean, and 5 of 19 the harmonic mean. Table S1 also carries one `rSquared` per
condition and a much milder asymmetry (half-width ratio 1.0981 to 1.4632). Hypothesis
(untested): Table S1 is one linear fit to the pooled exponential-phase points of all of
that condition's replicate curves, which equals the harmonic mean of the per-replicate
doubling times exactly when the replicates share a time grid. Either way the two tables are
two fits of one OD600 experiment, so the duplication is at the level of the experiment even
though 14 of 19 numbers differ; the per-replicate table is the finer grain.

## 2026.10.09 - #771: each record cites the study that first reported its sample

27 of the 152 mRNA records and 27 of the 105 protein records are Houser 2015's glucose
time course, released again here. Before this, all 257 records asserted Caglar 2017 as
the source, including the 54 whose measurements Houser 2015 published.

**Decision: per-record attribution (option 1 of #771), the Borchert 2024 pattern.** Each
record now stores the `Publication` of the study that FIRST reported its sample, exactly
as `torchcell/datasets/pputida/borchert2024.py` does for the three rows its compendium
subsumes (`SOURCE_STUDIES` + `attribute_sample`). Options 2 (mirror Houser and apply the
superset rule) and 3 (leave it) are not taken: 2 is work that only pays off if Houser is
loaded separately, and 3 leaves a false provenance claim in the store.

No schema class changes, no new field: `publication` is already per record, and
`scripts/schema_impact_check.py --base origin/main` reports **no schema contract
changes**. The experiment content id is `sha256` of the EXPERIMENT dump only, so the ids
of all 257 records are unchanged; what changes is which publication node each record
points at. That is still a changed served record, so it needs the full KG rebuild rather
than incremental admission.

### The split rule, and the three quotes it rests on

The rule is Table S1's own `experiment` column: `glucose_time_course` is Houser 2015's,
everything else is this paper's. All three quotes are re-read from `paper.md` at the
pinned sha256 `0878d5e7d49bcea4570aa2db225318a8469f4755645f1ddf73563effe7b3109b` and
found verbatim (a hash pin does not make a transcription verbatim, #758).

| constant | page | quote |
|---|---|---|
| `HOUSER2015_DEFERRAL` | Results, Experimental design and data collection | "Results from one of these conditions, long-term glucose starvation, have been presented previously10." |
| `HOUSER2015_CITATION` | References, reference 10 | "Houser, J. R. et al. Controlled Measurement and Comparative Analysis of Cellular Components in E. coli Reveals Broad Regulatory Changes in Response to Glucose Starvation. PLOS Comput Biol 11, e1004400 (2015)." |
| `HOUSER2015_DEPOSITS` | Data availability | "accession GSE67402 for the glucose time-course previously published10, accession GSE94117 for all other experiments" |

### Houser 2015 is unmirrored, and the DOI is derived rather than quoted

Measured: no `torchcell-library` key and no `torchcell-raw` key for Houser 2015 exists on
disk, and there is no loader. So nothing in the graph stores those measurements twice and
the paper itself is unread here; the attribution is sourced entirely from Caglar's own
citation of it, which is what #771 asked for in that case.

That citation gives journal, volume and article id but NO DOI. `HOUSER2015_DOI` is
`10.1371/journal.pcbi.1004400`, fixed from the citation because PLOS mints DOIs as
`10.1371/journal.<journal code>.<article number>`, then CHECKED by resolving it on
2026.10.09 (`curl -LH 'Accept: application/vnd.citationstyles.csl+json'
https://doi.org/10.1371/journal.pcbi.1004400`), which returned title, container-title
`PLOS Computational Biology`, volume `11` and page `e1004400` equal to the citation's.
`HOUSER2015_PUBMED_ID` is `26275208`, from an esearch of PubMed on that DOI. A test pins
that the DOI's article number is the article id the quote prints, so a mis-typed DOI
cannot pass silently.

### Measured: the released overlap and the stores after the rebuild

Script: `experiments/036-dataset-fixes-before-kg-build/scripts/caglar2017_houser2015_attribution.py`
Results: `experiments/036-dataset-fixes-before-kg-build/results/caglar2017_houser2015_attribution.json`

Table S1 holds 171 rows over 10 `experiment` values. `glucose_time_course` is 27 samples,
and all 27 are columns of BOTH Table S2 (152 sample columns) and Table S3 (105). A
separate 9 rows, `glucose_time_course (repeated between MURI 97- 105)` (MURI_007 to
MURI_015), are columns of neither table and so are not records of either family.

| dataset | records | cite Caglar 2017 (`10.1038/srep45303`) | cite Houser 2015 (`10.1371/journal.pcbi.1004400`) |
|---|---|---|---|
| `RnaseqCaglar2017Dataset` | 152 | 125 | **27** |
| `ProteomeCaglar2017Dataset` | 105 | 78 | **27** |

Both dev LMDBs were rebuilt with `--retire-existing`: 152 and 105 records before and
after, so the attribution moves no record. Each build also writes
`preprocess/source_study_attribution.json`, holding the rule, both studies' identities
(with `is_mirrored`), the per-study counts and one row per sample, so the attribution is
auditable from the store without re-reading Table S1.

### What this changes about the duplication risk

It does not remove it: if Houser 2015 is ever mirrored and loaded, those 27 plus 27
samples become a real duplication and the superset rule applies, the same shape as #760.
The attribution is what makes that detectable, because a Houser 2015 admission check can
now ask which records already name it.
