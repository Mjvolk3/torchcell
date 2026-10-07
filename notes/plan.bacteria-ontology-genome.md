---
id: 5fxhkll65v9sdllxkmw9wuv
title: Bacteria Ontology Genome
desc: ''
updated: 1791363978715
created: 1791363978715
---

## 2026.10.07 - Plan

Fifty bacterial datasets (35 *E. coli*, 15 *P. putida* KT2440) are being added to a
substrate that serves 51 *S. cerevisiae* datasets and nothing else. The authoritative
list is the first 50 rows of `ranked()` in
`experiments/database/scripts/build_bacteria_candidate_datasets_table.py`; the document
is [[experiments.database.expansion-bacteria]]. This plan settles the data
representation: the genomes tier for three bacterial assemblies, the organism-agnostic
genome abstraction, the pydantic changes and their rebuild consequence, the loader and
raw-mirror pattern, the adapter and KG-build preparation, and the branch order.

Three framing commitments, taken from the owner and from the Vision section of
`CLAUDE.md`, decide most of what follows. The data side is decided on its own terms, with
no modeling input. GenBank is ingested first, with GFF3 plus FASTA as the second route.
A b-number and a `PP_` locus tag must stay distinguishable namespaces, because `Genotype`
derives its content identity from the sorted systematic names of its perturbations.

### Verified

Everything in this section was run on GilaHyper on 2026-10-07. Downloads landed in
`/scratch/tmp/claude-1000/.../scratchpad/dl/` (session scratch, not the tier); nothing was
deposited, no KG build was run, and no Zotero call was made.

**NCBI assemblies.** Both the GenBank (GCA) and RefSeq (GCF) directories of all three
assemblies were listed and fetched with `curl` from
`https://ftp.ncbi.nlm.nih.gov/genomes/all/<GCA|GCF>/<3>/<3>/<3>/<acc>_<asm>/`, and every
downloaded file's md5 matched that directory's `md5checksums.txt`:

| host / strain | GenBank | RefSeq | replicon |
|---|---|---|---|
| *E. coli* K-12 MG1655 | `GCA_000005845.2_ASM584v2` | `GCF_000005845.2_ASM584v2` | U00096.3 / NC_000913.3, 4,641,652 bp |
| *E. coli* K-12 BW25113 | `GCA_000750555.1_ASM75055v1` | `GCF_000750555.1_ASM75055v1` | CP009273.1 / NZ_CP009273.1, 4,631,469 bp |
| *P. putida* KT2440 | `GCA_000007565.2_ASM756v2` | `GCF_000007565.2_ASM756v2` | AE015451.2 / NC_002947.4, 6,181,873 bp |

sha256 of the files this plan proposes to deposit (bytes, then digest):

```
GCA_000005845.2_ASM584v2_genomic.gbff.gz   3401246  50d48e5dc7c6ad18699db4819bc11540d61505cacaa57aedf28ac132f26a7d87
GCA_000005845.2_ASM584v2_genomic.fna.gz    1379898  a8bf0111c936605eb3b8d73d5a5c1dbb64ef39c87553608a6752719995e9d9ba
GCA_000005845.2_ASM584v2_genomic.gff.gz     384317  5be073e97f95efe337d1e5b69cd5af61d8cbee3450f8cb92c3ea722c8cb0af9b
GCA_000005845.2_ASM584v2_protein.faa.gz     901247  900cb656bba2f4fde8dafc43c80c84ddd326c85ef1d10b476e0a49a104db5263
GCA_000005845.2_ASM584v2_feature_table.txt.gz 190280 50ad5af9f85813b68a9a96f7989b588f28eafc5ace492ecbf93213f139da015e
GCA_000005845.2_ASM584v2_assembly_report.txt   1207 b40db266b9b21654c6ba07b8102e6e7451ce41b94539d92e591cc24a46fcc8b9
GCA_000750555.1_ASM75055v1_genomic.gbff.gz 3416102  41a48b1a84ecf0a533bec6041c65b71894448da3612ed93e4cc7ebc300f59ea3
GCA_000750555.1_ASM75055v1_genomic.fna.gz  1377238  bdc393dceb31717b63a7b4a87ebfc73b579ca4b014680aa0e066a5a20f9b50ae
GCA_000750555.1_ASM75055v1_genomic.gff.gz   385835  9f79536c9b3f134d1f773349144e50a97c0b5b5f8824496a7c8f549a734187a1
GCA_000750555.1_ASM75055v1_protein.faa.gz   894262  03bdd3a48e14f35f79a65b40c633b4270917a29ad1f4dd58283edde588f48253
GCA_000750555.1_ASM75055v1_feature_table.txt.gz 200398 796f9e961a2d351f8a5a9314700ae32273bd979867af0663b08d4d15c0a7d9ce
GCA_000750555.1_ASM75055v1_assembly_report.txt 1322 b79737157aadd821bbfb1df33a063174ff1d22ce705bd794dfa96330ef15e4f4
GCA_000007565.2_ASM756v2_genomic.gbff.gz   4361085  dbc3666fda5443db2c95040fd870585c92fbfb609ab2939821131fce40579bb0
GCA_000007565.2_ASM756v2_genomic.fna.gz    1802201  7893b5160c1ff24ab4e8f5faa2861051d455a2d8ded7e9de1f9eea4b99b66664
GCA_000007565.2_ASM756v2_genomic.gff.gz     391311  703bff9433dfd5d4b2e9aaebdccc3555be7790efaf5dba227a590f964c4978d0
GCA_000007565.2_ASM756v2_protein.faa.gz    1189638  914c6cc763fb5f404dbfe9dcd16c3eacec98d4a477a0bec3caf52bc58a24f0a9
GCA_000007565.2_ASM756v2_feature_table.txt.gz 217058 cb0fe25120a1e1506b165e5b3091d4a3d66ea0f5c36a447d634ea9912dc90666
GCA_000007565.2_ASM756v2_assembly_report.txt   1143 8b685bb9a0965e7db6bc5444bef3306cdac8fc0cdf34d704616b265753baa9c4
GCF_000005845.2_ASM584v2_genomic.gbff.gz   3401424  cbcc40a9859312cbcdbeb3df484ee471dfe354c5cd8f28056b48431353b28384
GCF_000005845.2_ASM584v2_genomic.gff.gz     387627  afdf03dc1d06e423d874ee29d9e0df14f5d32baf893f4da9f93aa721eb5f495e
GCF_000750555.1_ASM75055v1_genomic.gbff.gz 3428964  1f28a28be37211bb43ce4bfb520f8acdcd61189de61dd180e5ac2bfc94a6b9bf
GCF_000750555.1_ASM75055v1_genomic.gff.gz   438196  b5d361ed256bf30f5e2522c54239b241cf2787b7b5a03d4ebf2c95734ee3b1bd
GCF_000750555.1_ASM75055v1_gene_ontology.gaf.gz 157150 20798489747e477161a092f33e0b83f4369e70415d05f372d8d40d08d268f0bd
GCF_000007565.2_ASM756v2_genomic.gbff.gz   4472217  f607d834deae4a5f4a8a041ee16dbf6436d9fdd9cf2fd5c9562820325b8d22c0
GCF_000007565.2_ASM756v2_genomic.gff.gz     488028  69e01eb2ee6fdb12db203d0965fc2c4e23d156073378969032a246e31a7914aa
GCF_000007565.2_ASM756v2_gene_ontology.gaf.gz 216610 d3e54d8669aaaec1ba28f29f0ecf8b61b152109cdaddf3e533d114106ffded9a
```

A sha256 recorded here is retrieval evidence for the files fetched on this date, not a
deposited manifest. A deposit re-fetches and re-hashes, and `deposit_assembly_set`
refuses a manifest whose records disagree with the bytes (`registry.py`), so a changed
upstream is detected rather than followed.

**GenBank and RefSeq carry different locus tags, and that decides the identifier.**
Parsed with Biopython from the `.gbff.gz` files above:

| assembly | gene features | locus-tag form | `old_locus_tag` |
|---|---|---|---|
| GCA MG1655 | 4,651 | `b\d{4}`, all 4,651 | none |
| GCF MG1655 | 4,651 | `b\d{4}`, all 4,651 | none |
| GCA BW25113 | 4,490 | `BW25113_\d{4}`, all 4,490 | none |
| GCF BW25113 | 4,519 | `BW25113_RS\d+` | 8,737 qualifier lines |
| GCA KT2440 | 5,786 | `PP_\d{4}` for 5,621; the rest are `PP_16SA`, `PP_23SB`, `PP_mr01`-style RNA tags | none |
| GCF KT2440 | 5,719 | `PP_RS\d+` | 11,123 qualifier lines |

So the GenBank flat file is the file that carries the identifiers the fifty papers
actually report (b-numbers, `PP_` tags), and RefSeq's `_RS` retagging would force every
loader through a crosswalk. MG1655 is the one case where both agree.

**The two *E. coli* backgrounds must not be conflated, and the arithmetic shows why.**
Joining BW25113 to MG1655 by gene name: 4,215 of 4,490 BW25113 genes have a numeric
suffix equal to their MG1655 b-number, 45 differ (`BW25113_4659`/`yabP` against `b0056`;
`BW25113_0372`/`insF1` against `b0299`), and 230 BW25113 gene names are absent from
MG1655. Joining instead by the shared `ECK\d{4}` synonym: 4,435 ECK ids appear in both,
4,423 are 1:1, and of those 4,412 have equal numerics, leaving 11 one-to-one pairs whose
numerics disagree. MG1655 carries 192 ECK ids BW25113 lacks and BW25113 carries 16 MG1655
lacks. A b-number is therefore NOT derivable from a `BW25113_` number by string surgery,
and the ECK synonym is the only published 1:1 key between the two.

**GCA and GCF agree on the sequence and disagree on the annotation.** For each of the
three assemblies, the md5 of the FASTA with headers stripped is identical between GCA and
GCF (MG1655 `482a2b04485ec8c4...`, BW25113 `56b9c257c5be4d7e...`, KT2440
`4bda7ed2d0b626cf...`, first 16 hex each). Only the FASTA header id differs (`U00096.3`
against `NC_000913.3`, and so on). So depositing both annotations costs one extra GFF and
GBFF per host and no sequence duplication.

**GO annotation sources, measured rather than assumed.**

- NCBI ships `_gene_ontology.gaf.gz` under the RefSeq directory for BW25113 (5,182 rows,
  2,163 objects) and KT2440 (8,263 rows, 3,415 objects) and **not** for MG1655 (404 on
  both GCA and GCF). Both are keyed on `WP_` protein accessions, carry an empty synonym
  column, and are **100% IEA**, so neither reaches a locus tag without the assembly's own
  protein-to-locus mapping.
- `https://current.geneontology.org/annotations/gaf/ECOLI-uniprot.gaf.gz` (gaf-version
  2.2, generated 2026-07-28, sha256 `ad338c31d8114ce5579a43be4b8e3b78ab366541968cf76c3a91f82b353a9cdf`,
  952,515 bytes): 56,262 rows, 4,482 objects, **54,533 rows carry a b-number** in the
  synonym column, 4,016 distinct b-numbers, 29 objects carrying more than one, 20 `NOT`
  rows, 37% IEA. This is the one E. coli GAF that reaches MG1655 locus tags directly.
- `ECOLI-mod.gaf.gz` and `ecocyc.gaf.gz` are keyed on EcoCyc frame ids
  (`1-ACYLGLYCEROL-3-P-ACYLTRANSFER-MONOMER`) and contain **zero** b-numbers, so neither
  is usable without an EcoCyc crosswalk we do not mirror. `ecocyc.gaf.gz` is also absent
  from the dated release archive (404), unlike `ECOLI-uniprot.gaf.gz`.
- EBI GOA per-proteome files resolve through `proteome2taxid`:
  `109.P_putida_KT2440.goa` (taxon 160488, 4,502,711 bytes, sha256
  `575731316d9fcb98580dd7e2209a0239c909ada389e42c4052e5a8f7a1069a81`): 25,276 rows,
  3,887 objects, **every row carries a `PP_\d{4}` tag**, 3,912 distinct tags, 10 objects
  with more than one, **99% IEA**. `18.E_coli_MG1655.goa` (taxon 83333, sha256
  `12694eed60b0caf54b5edbce367ca2d47df50c6691a734f27ab9f72249a7f06f`) is the b-number
  equivalent: 4,016 distinct b-numbers, 37% IEA. EBI publishes no dated archive of these
  per-proteome files (`goa/old/` holds only the big model organisms), so a deposited copy
  is the only stable version.
- The Pseudomonas Genome DB is **not scriptable**: `https://www.pseudomonas.com/downloads`
  and a direct KT2440 GO path both return Cloudflare 403 with a challenge page.
- The RefSeq GFFs do carry inline GO: 2,278 rows with `Ontology_term=`/`go_process=` for
  BW25113 and 3,293 for KT2440, and **zero** for MG1655 and for every GCA GFF. This is the
  same shape the yeast genome reads GO from (`Ontology_term` on the GFF feature), so it is
  a real second route for two of the three hosts.
- GO ontology: the dated release `https://release.geneontology.org/2026-08-05/ontology/go-basic.obo`
  returns 200 and is **byte-identical** to `current.geneontology.org` today (sha256
  `b08d45b268b8c24ccb2513dbbbc7d4df9f6521c099b413f79eb31e06e0fa3bcc`, 32,227,785 bytes,
  `data-version: releases/2026-07-26`). `ECOLI-uniprot.gaf.gz` under the same dated prefix
  is also 200 and byte-identical to the current copy.
- UniProt REST answers for both hosts at release 2026_03 (02-September-2026):
  `proteome:UP000000625` (MG1655) 4,403 entries and `proteome:UP000000556` (KT2440) 5,527,
  with `gene_oln` giving `b1109 JW1095` and `PP_4621` respectively plus `xref_refseq`
  `WP_`/`NP_` accessions. This is the crosswalk that turns the NCBI `WP_`-keyed GAFs into
  locus tags, and the cross-check on the GOA files.

**The yeast GO path is unpinned, which the bacterial work should not copy.**
`SCerevisiaeGenome.__attrs_post_init__` downloads `http://current.geneontology.org/ontology/go.obo`
into `go_root` when absent and holds no hash. The copy in use,
`/scratch/projects/torchcell-scratch/data/go/go.obo`, is `data-version: releases/2024-01-17`,
34,420,328 bytes, sha256 `ad2f050219a579bc7cb448dc2fac5378b3b6b38fa857ab0b4e8ded7d270aa9ff`,
dated 2024-03-15, and two experiment scripts already note that the upstream URL 403s.

**Closure-impact measurements (what a schema edit would cost).** Measured with
`torchcell.provenance.schema_deps` over the 36 loader modules behind `dataset_adapter_map`,
by rewriting `schema.py` in memory and recomputing each loader's closure fingerprints:

| candidate edit | served loader modules whose closure moves |
|---|---|
| widen `GenePerturbation.validate_sys_gene_name` body | **35 / 36** (symbol `GenePerturbation`) |
| add a defaulted `gene_namespace` field to `GenePerturbation` | **35 / 36** |
| rename `systematic_gene_name` on `GenePerturbation` | **35 / 36** |
| add a field to `ReferenceGenome` | **36 / 36** (symbol `ReferenceGenome`) |
| change `Genotype`'s sort validator | **36 / 36** |
| add a field to `Phenotype` or to `Media` | **36 / 36** |
| widen `StrainBackground.reference_strain` Literal, or touch `BackgroundAllele` | **4 / 36** (Hillenmeyer, Hoepfner, Vanacloig, Wildenhain) |
| **new leaf subclass of `DeletionPerturbation` only** | **0 / 36** |
| **new leaf plus widening the `GenePerturbationType` union** | **0 / 36** |
| **new `ReferenceGenome` subclass** | **0 / 36** |
| **new phenotype class** | **0 / 36** |

The union result is not luck: `loader_schema_deps_from_source` collects only names a
loader imports that are `ClassDef`s on the surface, and `GenePerturbationType` is a
module-level assignment, so it is outside the fingerprinted surface. This is the same
mechanism `SegregantGenotype` used, and it is recorded in `schema.py`'s own comment.

**A `ReferenceGenome` subclass silently loses its extra fields in a base-typed slot.**
Constructed `AssemblyReferenceGenome(ReferenceGenome)` with an `assembly_set` field, put
it in `FitnessExperimentReference.genome_reference` (annotated `ReferenceGenome`), and
dumped: the attribute keeps the subclass, `model_dump()` emits only
`{species, strain, ploidy}`, and `model_validate` of that dump returns a plain
`ReferenceGenome`. So a subclass carries an assembly pin ONLY where the field is
re-annotated to it, exactly as `StrainEnvironmentResponseExperimentReference` narrows
`genome_reference` to `StrainReferenceGenome`.

**What the genome surface's callers actually use.** `SCerevisiaeGenome(` appears at 37
call sites in `experiments/006-kuzmin-tmi` alone and across 32 files under
`torchcell/datasets`; the attributes reached are narrow. Library code
(`torchcell/graph`, `torchcell/data`, `torchcell/datasets`, `torchcell/metabolism`,
`torchcell/verification`): `gene_set` (65 reads), `go_dag` (8), `drop_chrmt`,
`drop_empty_go`, `alias_to_systematic`, `fasta_dna`, `chr_to_nc`. Loaders:
`resolve_gene_name` (19), `alias_to_systematic` (12), `gene_set` (9),
`gene_attribute_table` (9), `feature_index` (3), `db` (1). Experiments add
`drop_empty_go` (131), `drop_chrmt` (61), `db[...]`, `go_dag[...]`,
`fasta_dna[...]`. The KG build itself
(`create_kg.py`, `create_scerevisiae_kg_small.py`) only CONSTRUCTS a genome and injects it
into loaders that declare a `genome` parameter; it reads no attribute. Same for
`torchcell/database/build_dataset_lmdb.py` and `torchcell/verification/runners.py`
(which additionally takes the bound `resolve_gene_name` and reads
`resolve(SGD_S288C_R64, ...)` FASTA headers for its L4 gene universe).

### Decisions

Each decision states the recommendation first. Three are the owner's to make and are
marked **OWNER**; the rest follow from the measurements above.

**D1. Identify a bacterial gene by its GenBank locus tag, namespaced, and carry the
namespace in a new field on new leaf classes only.** The identifier is `b\d{4}` for
MG1655, `BW25113_\d{4}` for BW25113, and `PP_\d{4}` (plus the named RNA tags `PP_16SA`,
`PP_23SB`, `PP_mr01`) for KT2440, taken from the GenBank flat file. The namespace is a
required `Literal` field (`gene_namespace`) on the new bacterial perturbation leaves, and
a collision between namespaces is impossible by construction because the three patterns
are disjoint from each other and from the yeast patterns. The namespace field also means
a future host whose tags DO collide (two organisms both using `PP_`) stays
distinguishable without a second rename.

Rationale against the alternatives. Prefixing the string
(`ecoli_k12_mg1655:b0002`) would make the identifier self-describing but would change
what `genotype.systematic_gene_names` holds for every bacterial record and would make
every join to a paper's own table a string operation. A bare locus tag with no namespace
is what the expansion document warns against: `Genotype` content identity is the sorted
systematic names, and so the namespace is load-bearing for node identity.

**D2. OWNER -- `systematic_gene_name` keeps its name.** Renaming it to something
organism-neutral (`gene_id`, `locus_id`) is the honest name, and it is a **full rebuild
plus a 1,100-site edit**: measured 348 occurrences in `torchcell/` across 65 files, 378 in
`tests/`, 391 in `experiments/`, 6 in `database/`, 2 in the BioCypher schema config, and
24 query files under `torchcell/knowledge_graphs/queries`, `supported_queries` and
`experiments/*/queries`. The closure check says the rename moves 35 of 36 served loader
modules, and it also changes the `perturbation` graph node's property name, so every
stored Cypher query and every downstream notebook breaks at once.

Recommended instead: keep `systematic_gene_name` as the field, document in its docstring
that it holds a namespaced systematic identifier whose namespace is given by
`gene_namespace`, and spend the rename budget on the namespace field. The field name then
reads as slightly yeast-flavored prose while the data is correct, which is the cheaper of
the two wrongs. Two sub-options if the owner wants the rename anyway: (a) do it in the
same full rebuild as the bacterial admission, as one commit touching schema, adapters,
schema config and every query, with the Cypher property renamed too; (b) add `gene_id` as
a computed alias property that serializes alongside, which is NOT free (adding a
`computed_field` to `GenePerturbation` moves 35 of 36 closures the same as a field would).

**D3. Yeast-specific names get an explicit yeast prefix where they are yeast-specific,
and the genome base keeps the neutral names.** Concretely: `SYSTEMATIC_GENE_PATTERN` in
`schema.py` becomes `SGD_SYSTEMATIC_GENE_PATTERN`; `_LOCUS_FEATURE_TYPES` in `s288c.py`
becomes `SGD_LOCUS_FEATURE_TYPES` and stays in the scerevisiae subpackage;
`GeneNameStatus`/`GeneNameResolution` stay neutral (they describe resolution, not yeast)
and move to the genome base module; `_sgd_gene_set` in
`torchcell/verification/runners.py` is already correctly named and gains a sibling per
host. `SCerevisiaeGenome` and `SCerevisiaeGene` are already explicit. Renaming the
`schema.py` module constant is free under the closure check only if no class body
references it; `BackgroundAllele` does reference it, which puts that rename in the 4-of-36
bucket (the four chemogenomic loaders), so it rides the same full rebuild rather than
being deferred.

**D4. Deposit GenBank AND GFF3+FASTA for all three assemblies, in one assembly set per
strain, and pin the GO ontology and the GAF inside the same set.** The Vision says GenBank
first, and the measurements say the GenBank flat file is also the only file carrying the
reported locus tags. GFF3 plus FASTA are deposited in the same set because `gffutils`
(the existing `data.db` builder) reads GFF3, not GenBank, so the second route is what
makes the cache generalize with no new dependency. The RefSeq GFF is deposited too for
BW25113 and KT2440 because it is where their inline GO lives. The sequence is deposited
once (the GCA FASTA; GCF is byte-identical).

**D5. Three assembly sets, one per strain, never one per host.** `ecoli_K12_MG1655_ASM584v2`,
`ecoli_K12_BW25113_ASM75055v1`, `pputida_KT2440_ASM756v2`. The expansion document's
`K-12-KO` sequence basis is deliberately strain-ambiguous because the Keio collection is
BW25113 and the knockdown libraries are mostly MG1655; the tier is where that ambiguity
must NOT be reproduced, since the tier is what dereferences a strain to bytes. A row whose
background is not yet settled picks its set when its loader is written, and records which.

**D6. OWNER -- the assembly pin goes on a new `AssemblyReferenceGenome` subclass, not on
`ReferenceGenome`.** Adding `assembly_set` to `ReferenceGenome` moves 36 of 36 served
closures; a new subclass moves 0. The subclass is only honored where a field is
re-annotated to it, so each new bacterial experiment-reference class narrows its
`genome_reference` the way `StrainEnvironmentResponseExperimentReference` already narrows
to `StrainReferenceGenome`. Cost of the subclass route: the 51 served yeast datasets keep
storing `{species, strain, ploidy}` with no assembly pin until some later rebuild adds it,
so "which assembly" stays implicit for yeast and explicit for bacteria. Alternative, if the
owner prefers one uniform shape: put `assembly_set` on `ReferenceGenome` as part of the
full rebuild this program already needs, and set it to `sgd_S288C_R64-4-1_20230830` for
every yeast record. That is one field, one rebuild, and a uniform substrate. The subclass
is recommended only because it keeps the bacterial work independent of how long the yeast
backfill takes.

**D7. One full rebuild, taken deliberately, and the additive list kept minimal anyway.**
The owner expects a full rebuild and the plan agrees, because D3's constant rename and the
`ExperimentReference` family work both touch served closures. The reason to still keep the
change list minimal is that the manifest's admission check is the instrument that tells us
WHICH datasets a change reaches, and a minimal change set means the next bacterial tranche
can go in incrementally (`kg_manifest admit --dataset <Class>`) rather than forcing a
second full rebuild.

**D8. GO for bacteria is deposited, never downloaded at construction time.** Each host's
assembly set carries `go-basic.obo` from the dated release
`https://release.geneontology.org/2026-08-05/ontology/go-basic.obo` (one copy in a shared
`go_releases` set, referenced by id, rather than three copies), plus the host's GAF:
`ECOLI-uniprot.gaf.gz` from the same dated prefix for MG1655, the EBI GOA proteome file
for KT2440, and for BW25113 either the RefSeq GFF's inline terms or MG1655's GAF mapped
through the ECK synonym. No `download_url` call at genome construction, which is the one
behavior of `SCerevisiaeGenome` the bacterial genomes must not inherit.

**D9. BW25113 GO is a documented gap until its own source is settled, not an MG1655
copy.** The honest options are the RefSeq GFF's 2,278 inline rows (IEA-only, keyed on
`BW25113_RS` and crosswalked via `old_locus_tag`) or MG1655's richer GAF mapped through the
4,423 one-to-one ECK pairs, which asserts that a BW25113 gene has its MG1655 orthologue's
annotation. The first is weaker but true of BW25113; the second is stronger and is an
inference. Recommendation: deposit both files, default the loader-facing GO to the RefSeq
GFF terms, and expose the ECK-mapped MG1655 annotation as an explicitly labeled derived
view that no record stores. Decide before the first BW25113 row's GO is consumed.

### 1. Genomes tier: the three bacterial assembly sets

The tier is `$DATA_ROOT/torchcell-genomes/<assembly_set>/`, one flat directory per set
with a pydantic `manifest.json` pinning every file by sha256 and recording its retrieval
([[torchcell.sequence.genome.registry]], [[plan.genomes-tier.2026.09.14]]). Four sets are
added: three strain sets and one shared GO-release set.

New ids in `registry.py` beside `SGD_S288C_R64` and `PETER2018_1011`:

```python
ECOLI_K12_MG1655 = "ecoli_K12_MG1655_ASM584v2"
ECOLI_K12_BW25113 = "ecoli_K12_BW25113_ASM75055v1"
PPUTIDA_KT2440 = "pputida_KT2440_ASM756v2"
GO_RELEASE_20260805 = "go_release_2026-08-05"
```

Members per strain set, with the role each takes from `registry.py`'s role constants.
Every file is fetched by `direct_url` from the NCBI path in the Verified table, so
`RetrievalRecord.params` is just the URL and the recorded command re-runs as-is.

| member | role | why it is in the set |
|---|---|---|
| `<GCA>_genomic.gbff.gz` | `ROLE_ANNOTATION` | **the primary ingest**: locus tags, `gene_synonym` (ECK/JW), `db_xref` (ECOCYC, GeneID), product names |
| `<GCA>_genomic.fna.gz` | `ROLE_SEQUENCE` | the replicon sequence (GCF is byte-identical, so deposited once) |
| `<GCA>_genomic.gff.gz` | `ROLE_ANNOTATION` | the GFF3 route, what `gffutils` builds `data.db` from |
| `<GCA>_protein.faa.gz` | `ROLE_SEQUENCE` | protein sequences, the analogue of `orf_trans_all` |
| `<GCA>_feature_table.txt.gz` | `ROLE_INDEX` | flat locus-tag to coordinate to product index, for a crosswalk without parsing GBFF |
| `<GCA>_assembly_report.txt` | `ROLE_INDEX` | replicon names, GCA-to-GCF accession pairing, assembly level |
| `<GCF>_genomic.gbff.gz` | `ROLE_ANNOTATION` | RefSeq annotation; the `old_locus_tag` crosswalk (BW25113, KT2440) |
| `<GCF>_genomic.gff.gz` | `ROLE_ANNOTATION` | RefSeq GFF3; **the inline GO** for BW25113 and KT2440 |
| `<GCF>_gene_ontology.gaf.gz` | `ROLE_ANNOTATION` | NCBI PGAP GAF; **BW25113 and KT2440 only**, keyed on `WP_` |
| host GAF (see below) | `ROLE_ANNOTATION` | the locus-tag-keyed GO annotation the genome reads |

Host GAF member, per set:

- MG1655: `ECOLI-uniprot.gaf.gz` from
  `https://release.geneontology.org/2026-08-05/annotations/gaf/ECOLI-uniprot.gaf.gz`
  (sha256 `ad338c31d8114ce5579a43be4b8e3b78ab366541968cf76c3a91f82b353a9cdf`). Identifier
  column: b-numbers in the synonym field (column 11), 4,016 distinct.
- KT2440: `109.P_putida_KT2440.goa` from
  `https://ftp.ebi.ac.uk/pub/databases/GO/goa/proteomes/109.P_putida_KT2440.goa`
  (sha256 `575731316d9fcb98580dd7e2209a0239c909ada389e42c4052e5a8f7a1069a81`). Identifier
  column: `PP_\d{4}` in the synonym field, 3,912 distinct, 99% IEA. Because EBI keeps no
  dated archive, the deposited copy IS the version, and the manifest's `source_url`
  is retrieval metadata that will drift.
- BW25113: per D9, both the RefSeq GFF inline terms (already a member) and MG1655's GAF
  (referenced by the MG1655 set id, not copied), with the ECK crosswalk built by the
  loader-facing code, not baked into a file.

The `go_release_2026-08-05` set holds one member, `go-basic.obo` (sha256
`b08d45b268b8c24ccb2513dbbbc7d4df9f6521c099b413f79eb31e06e0fa3bcc`,
`data-version: releases/2026-07-26`), so all three bacterial genomes pin one DAG by id.
This also gives yeast a path off its unpinned 2024-01-17 `go.obo`: a later change points
`SCerevisiaeGenome.go_root` at a tier set instead of a download directory. Changing the
yeast GO release changes which terms survive `remove_deprecated_go_terms`, so that is its
own decision and is NOT part of this program.

Provisioning is a script in the same shape as `scripts/migrate_genomes_tier.py`
([[scripts.migrate_genomes_tier]]), `scripts/provision_bacterial_genomes.py`, with
`--refetch-dir`, `--data-root` and `--dry-run`. It fetches each URL, verifies the md5
against the directory's `md5checksums.txt` where NCBI publishes one, computes sha256 and
byte counts from the files on disk, builds a `GenomeManifest` per set
(`organism`, `strain_or_population`, `source="NCBI"`, `release="ASM584v2"`, and so on),
and calls `deposit_assembly_set`, which refuses to overwrite and refuses a manifest that
disagrees with the bytes. `provenance_complete=True` for all four sets, since every member
has a genuine scriptable retrieval.

Acceptance: `python -m torchcell.sequence.genome.registry` is not a CLI, so the check is
`verify_assembly_set("<set>")` returning a digest per member for all four sets, plus
`resolve("<set>", "<member>")` raising `GenomeIntegrityError` on a deliberately corrupted
byte and `FileNotFoundError` with the rsync hint on a machine without the tier. The tier
lives at `/scratch/projects/torchcell-scratch/torchcell-genomes/` on GilaHyper, so the
canonical-host rsync hint in `registry.py` keeps working unchanged.

### 2. Genome abstraction

Today `SCerevisiaeGenome` is the only concrete `Genome`. It is 2,322 lines, of which the
large majority is the shared `data.db` concurrency machinery: the atomic build and install
(`write_genome_database`, `install_genome_database`, `rebuild_genome_database`,
`migrate_genome_database`), the source record inside the sqlite file
(`GenomeDatabaseSource`, `GenomeDatabaseRecord`, `SOURCE_TABLE`, `RECORD_VERSION`), the
torn-file and hot-journal detection, the dead-build sweep, the private write copy. None of
that is yeast-specific and none of it should be written twice.

Proposed shape, three layers:

1. **`torchcell/sequence/genome/base.py` -- `AnnotatedGenome(Genome)`.** Holds everything
   the three hosts share: an `ASSEMBLY_SET` / `GENOME_VERSION` class pair, resolution of
   its members through `registry.resolve` (never a download), the `genome_root` cache with
   `data.db`, the whole concurrency and record machinery moved verbatim out of `s288c.py`,
   `gene_set` / `compute_gene_set`, `feature_types`, `gene_attribute_table`, `get_seq`,
   `__getitem__`, the GO properties (`go`, `go_subset`, `go_genes`, `go_subset_genes`,
   `go_dag`), `drop_empty_go`, the locus index (`feature_index`), `alias_to_systematic`
   and the layered `resolve_gene_name` with `GeneNameStatus` / `GeneNameResolution`.
   The parts that vary become class-level hooks: the set of gene-like locus feature types,
   the mapping from a GFF seqid to the integer `chromosome` key, the GFF attribute (or
   external GAF) that supplies GO terms, and the FASTA members to parse. `drop_chrmt` does
   NOT move up: a mitochondrion is a eukaryote fact.
2. **`torchcell/sequence/genome/scerevisiae/s288c.py` -- `SCerevisiaeGenome(AnnotatedGenome)`.**
   Keeps the SGD specifics and nothing else: `CHROMOSOMES` with the roman-numeral and
   `chrmt`-at-zero convention, `SGD_LOCUS_FEATURE_TYPES`, the `Ontology_term` GO source,
   `orf_classification` and the five-prime-UTR-intron CDS selection in `SCerevisiaeGene`,
   `drop_chrmt`, and the `data/sgd/genome` default root. Every public name it exports today
   keeps working, so the 37-plus call sites in `experiments/` are untouched.
3. **`torchcell/sequence/genome/ecoli/k12.py` and `torchcell/sequence/genome/pputida/kt2440.py`.**
   `EcoliK12Genome(AnnotatedGenome)` parameterized by strain (MG1655 or BW25113, each
   naming its own assembly set and locus-tag pattern) and `PPutidaKT2440Genome`. A single
   circular replicon means the `chromosome` key is 1 for every gene and `chr_to_nc` has
   one entry; `roman_to_int` is not reached. GO comes from the host's deposited GAF, parsed
   on the identifier column measured above, with the RefSeq-GFF inline route available for
   the two hosts that have it. `resolve_gene_name` gets its bacterial layers in order:
   exact locus tag, then `old_locus_tag`, then the `gene` qualifier (the symbol, `thrA`),
   then `gene_synonym` (ECK, JW), then retired.

Cache generalization. `genome_root` stays the cache root and keeps holding `data.db`; the
file is built from the GFF3 member the subclass pins, and `GenomeDatabaseSource` already
records `assembly_set` plus `gff_filename` plus `gff_sha256`, so a bacterial `data.db`
built against the wrong set is detected by the existing `untrusted_reason` path with no
change. Default roots: `data/ecoli/mg1655/genome`, `data/ecoli/bw25113/genome`,
`data/pputida/kt2440/genome` under `DATA_ROOT`. Each host's `data.db` is a separate
sqlite file in a separate directory, so the `_root_lock` flock and the dead-build sweep
stay per host and parallel loaders on different hosts cannot collide.

Renames (D3), all in this step: `SYSTEMATIC_GENE_PATTERN` to `SGD_SYSTEMATIC_GENE_PATTERN`
in `schema.py`; `_LOCUS_FEATURE_TYPES` to `SGD_LOCUS_FEATURE_TYPES`;
`SGD_GENE_FASTAS` and `_sgd_gene_set` in `verification/runners.py` keep their names and
gain `_ecoli_k12_gene_set` / `_pputida_gene_set` siblings built from each set's
`protein.faa.gz` plus `feature_table.txt.gz`.

Acceptance: `SCerevisiaeGenome(genome_root=..., go_root=..., overwrite=False)` still
yields the same `len(gene_set)` and the same `data.db` content digest as before the
refactor (`database_content_digest`, compared against a pre-refactor run);
`tests/torchcell/sequence/genome/scerevisiae/test_s288c.py` and `test_s288c_synthetic.py`
pass unchanged; new tests build each bacterial genome from the tier and assert the gene
counts measured above (4,651 MG1655 / 4,490 BW25113 / 5,786 KT2440 gene features from the
GenBank route) and a round trip through `resolve_gene_name` for a locus tag, a symbol, an
ECK synonym and a retired name per host.

### 3. Schema changes, each with its rebuild consequence

The expansion document's two blockers are confirmed and a third is added by this plan's own
reading: the perturbation leaves, the assembly pin, and the reference-genome narrowing per
experiment family. The classification below uses the admission vocabulary of
`torchcell/knowledge_graphs/kg_manifest.py` ([[torchcell.knowledge_graphs.incremental-admission]]):
**additive** means no served dataset's closure fingerprints move and no existing graph
class changes, so `admit` can say ADMISSIBLE; **closure-changing** means a served
dataset's records would serialize differently today and the honest answer is the full
rebuild.

**3a. Bacterial perturbation leaves (additive, measured 0 of 36).** New concrete leaves,
each a subclass of the existing axis leaf, each carrying a required
`gene_namespace: Literal[...]` and each overriding `validate_sys_gene_name` with its own
namespace pattern:

| leaf | parent | covers which of the fifty |
|---|---|---|
| `BacterialDeletionPerturbation` | `DeletionPerturbation` | the 9 `K-12-KO` rows (Keio and the chemical-genomics screens) and the Keio-derived `K-12-KO x KO` pairs |
| `TransposonInsertionPerturbation` | `PresenceAbsencePerturbation` (`state="absent"`, SO:0001218 `transgenic_insertion`) | the 10 `K-12+transposon` / `KT2440+transposon` rows; carries the barcode and the mapped insertion coordinate, so the genotype is realized rather than designed |
| `BacterialCrisprInterferencePerturbation` | `CrisprInterferencePerturbation` | the 10 `+guide` rows; reuses the existing `CrisprConstruct` unchanged |
| `PromoterReplacementPerturbation` | `ExpressionModulationPerturbation` | the `KT2440+promoter` row and the promoter arms of the combinatorial designs |
| `HeterologousPathwayPerturbation` | `GeneAdditionPerturbation` | the 6 `engineered-chassis` production rows; `GeneAddition` already relaxes the name validator for heterologous genes, so this leaf adds the cassette and copy context, not a new validator |

Each leaf gets a unique `perturbation_type` Literal (the existing
`test_perturbation_type_defaults_are_unique` enforces this) and is appended to the
module-level `GenePerturbationType` union, which is outside the fingerprinted surface.
`test_identity_is_curie_or_systematic` already admits a CURIE or a yeast systematic name,
so it needs its regex extended to the three bacterial patterns; that is a test change, not
a schema one.

What this does NOT do: it leaves `GenePerturbation.validate_sys_gene_name` untouched, so a
bacterial name still fails on a yeast leaf. That is the desired behavior. Widening the base
validator would admit `b0002` into `SgaKanMxDeletionPerturbation` and would move 35 of 36
served closures for no benefit.

**3b. The assembly pin (additive as a subclass, 36 of 36 as a field).** New
`AssemblyReferenceGenome(ReferenceGenome)` with `assembly_set: str` and
`assembly_accession: str` (the `GCA_...`/`GCF_...` pair from the set's assembly report),
validated against `registry`'s known set ids at construction. Per D6 and the serialization
measurement, it is honored only where a field is re-annotated to it, so each bacterial
`ExperimentReference` subclass narrows `genome_reference: AssemblyReferenceGenome`. A
bacterial record then states, in the record, which pinned bytes its genotype is written
against, which is what "sequence-level genotype fidelity" requires and what the yeast
records currently leave implicit.

**3c. Bacterial experiment families (additive; new classes move 0 of 36).** The fifty need
no new phenotype class for 46 of their 50 rows, because the analog column maps them onto
record types that already exist. Mapped from the `analog` and `schema_need` literals:

| phenotype class | reuse or new | rows it serves (count of the fifty) |
|---|---|---|
| `FitnessPhenotype` | reuse | transposon and RB-TnSeq gene fitness, Keio growth, CRISPRi knockdown fitness: the 10 transposon rows, 7 CRISPR-screen rows, the mismatch-CRISPRi row |
| `EnvironmentResponsePhenotype` | reuse (`AssayType.pooled_competitive_growth_barcode`, `MeasurementType.log2_ratio`/`z_score`) | the 4 chemical-genomics rows, the 3 tolerance rows |
| `GeneInteractionPhenotype` | reuse | the 4 genetic-interaction rows (Butland, Typas, Babu, Rachwalski) |
| `GeneEssentialityPhenotype` | reuse | Goodall 2018 TraDIS essentiality calls |
| `ProteinAbundancePhenotype` | reuse | Schmidt 2016, Mori 2021, Banerjee 2025, the Carruthers paired proteome |
| `MetabolitePhenotype` | reuse | Fuhrer 2017, Rapp 2026, Schastnaya 2021 |
| `RNASeqExpressionPhenotype` | reuse | PRECISE-1K, putidaPRECISE321, Caglar 2017, Choe 2019 |
| **`ProductTiterPhenotype`** | **new** | the 8 production-campaign rows plus the 2 combinatorial designs: titer with units, yield and productivity when released, and the product identity as a typed `Compound` |
| `VisualScorePhenotype` | reuse | the Ozaydin-analog rows whose readout is a scored plate rather than a measured titer |
| **`ProteinTurnoverPhenotype`** | **new** | Gupta 2024 per-protein degradation and turnover rates |
| **`FluxPhenotype`** | **new** | Li 2021 fitted net flux for 198 reactions with confidence bounds, and Ishii 2007's 13C flux arm |

`ProductTiterPhenotype` is the one genuinely new record type the engineering half of the
list needs, and it is the one the Vision cares about most, since it is the phenotype side
of inverse strain design. It carries a `Compound` for the product (so it joins the
compound-identity layer the chemogenomic datasets already use), the value with a typed
unit, the uncertainty with its `UncertaintyType`, `n_samples` with its `SampleUnit`, and
the fermentation context on the environment rather than on itself. Each new phenotype gets
its `Experiment` / `ExperimentReference` pair in the same commit, with
`genome_reference: AssemblyReferenceGenome`, and the pairs are appended to
`PhenotypeType`, `ExperimentType`, `ExperimentReferenceType`, `EXPERIMENT_TYPE_MAP` and
`EXPERIMENT_REFERENCE_TYPE_MAP`, all module-level assignments outside the surface.

Ten of the fifty carry a non-empty `schema_need` and are deliberately NOT solved here.
Each is a per-row design question that its loader raises: a guide array carrying several
targets at a graded induction level (Tian 2019), a genotype combining a cataloged deletion
with a guide knockdown (Rachwalski 2024), a titer vector over product identity (Wang
2022), a population allele-frequency genotype against an undeposited reduced-genome
reference (Choe 2019), an evolved variant list against a non-parent reference (Niu 2019),
a fitness record whose genotype is a gene rather than a clone (Royet 2025, a non-barcoded
pool), a fitted flux distribution with per-value intervals (Li 2021), and the
markerless-deletion-plus-cassette stack given by primer rather than coordinate (Yang 2019).
Four of these rows are also `blocked` for released-data reasons, so their schema question
does not block the first tranche.

**3d. Media the fifty need (closure-changing: `Media` is in 36 of 36 closures).** The
expansion table's `env` strings name conditions rather than recipes (`89 conditions`,
`nitrogen sources`, `3 pH levels`), so the actual recipe list comes from each paper's
Methods as its loader is written. What is settled now is the shape. `Media` can express
all of it unchanged: `is_synthetic=True` for M9 and MOPS, `False` for LB,
`base_medium` naming a `MEDIA_LIBRARY` key, components through `resolved_compound` so they
join the existing compound layer, and a carbon source either as a component (when it is
part of the recipe) or as an `EnvironmentPhysicalPerturbation` with
`PhysicalFactor.carbon_source` (when it is the variable). `ConcentrationUnit` already
carries `g_per_l`, `percent_w_v`, molar and `pH`. So the bacterial media work is **library
additions, not schema changes**: `LB`, `LB_AGAR`, `M9`, `M9_GLUCOSE`, `MOPS_MINIMAL`
(Neidhardt), and the per-paper derived media, each with a sourced quote and sha256 from the
mirrored paper, each with `base_medium` naming another library key so `_check_library`
passes at import.

This is nonetheless the single most expensive change in the list, because `media.py` is in
`VALUE_SURFACE_RELPATHS`: the admission gate hashes it file by file, and any edit blocks an
increment with a value-drift finding (acknowledgeable with `--ack-value-drift`). Adding a
new medium does not move an existing medium's content-addressed node id, so the
acknowledgment is honest here; the reason recorded should say exactly that, naming the keys
added and asserting that no existing key's component list changed. A test that pins the
`media_identity` digest of each pre-existing `MEDIA_LIBRARY` entry across the change is
what makes that assertion checkable rather than asserted.

**3e. `StrainBackground` and the bacterial equivalent (4 of 36 if widened).**
`StrainBackground.reference_strain` is `Literal["S288C"]` and `BackgroundAllele` validates
against the yeast pattern, so neither admits a bacterial background as written. Widening
the Literal moves 4 of 36 closures (Hillenmeyer, Hoepfner, Vanacloig, Wildenhain). The
bacterial list does need the concept: BW25113 carries defined lesions
(`rrnB3 lacZ4787 hsdR514 (araBAD)567 (rhaBAD)568 rph-1`, verbatim from the GenBank source
feature's `/note`), which is exactly what `StrainBackground` is for, and the production
chassis rows carry stacked deletions plus heterologous cassettes.

Recommendation: do NOT widen the yeast classes. Add `BacterialStrainBackground` and
`BacterialBackgroundAllele` siblings (new classes, 0 of 36) whose `reference_strain` is a
`Literal` over the three assembly-set strain names and whose allele validator takes the
namespace pattern. The duplication is real and is the price of not moving four served
closures for a yeast-shaped field; the two families can be unified later, in a rebuild
whose purpose is that unification. BW25113's background is sourced from the deposited
GenBank flat file, which is a sha256-pinned artifact in the tier, so the allele list has
provenance without a paper quote.

**3f. Minimal full-rebuild list.** Taking D7's deliberate rebuild, these are the changes
that actually require it, and nothing else should ride along:

1. `media.py` library additions (value surface; blocks any increment).
2. `SYSTEMATIC_GENE_PATTERN` to `SGD_SYSTEMATIC_GENE_PATTERN` (reaches `BackgroundAllele`,
   so 4 of 36).
3. The `perturbation` graph node gaining a `gene_namespace` property, if the owner wants
   the namespace queryable in Cypher rather than only inside the record blob. This is a
   change to an EXISTING served graph class, which incremental import cannot repair, so it
   is the one graph-schema item that forces the rebuild by itself. The alternative is a new
   `bacterial perturbation` node class (additive) that carries the namespace, with the
   existing `perturbation` class untouched; recommended, and it keeps item 3 off this list.

Everything in 3a, 3b, 3c and 3e is additive. If the owner takes the recommendation in item
3 and defers item 2 to a later cleanup, the only full-rebuild trigger left is `media.py`,
and even that is acknowledgeable. That is the honest minimum, and the owner can still
choose the rebuild for cleanliness.

Acceptance for the whole step:
`PYTHONPATH=<wt> python -m pytest tests/torchcell/datamodels -x` green (the ontology
invariants are the real gate: single-rooted hierarchy, acyclic composition, Liskov in the
union slot, unique discriminators, round-trip fidelity, well-formed SO ids); then
`python -m torchcell.provenance.schema_impact` reporting zero affected served loaders for
the additive commits; then, per host,
`DeletionPerturbation(systematic_gene_name="b0002", ...)` still raising while
`BacterialDeletionPerturbation(systematic_gene_name="b0002", gene_namespace="ecoli_k12_bnumber", ...)`
validates and round-trips through `Genotype`.

### 4. Loader and raw-mirror pattern for bacteria

New packages `torchcell/datasets/ecoli/` and `torchcell/datasets/pputida/`, mirroring
`torchcell/datasets/scerevisiae/` exactly: one module per paper, named
`<firstauthor><year>.py`, each decorated with `@register_dataset` and subclassing
`ExperimentDataset`. `torchcell/datasets/__init__.py` imports them and
`build_dataset_lmdb.resolve_dataset_class` / `kg_manifest` gain the two package imports
beside their existing `import torchcell.datasets.scerevisiae` registry-population lines
(two one-line edits, both already commented as doing exactly that).

Raw mirror, unchanged convention: `$DATA_ROOT/torchcell-raw/<citation_key>/` with
`manifest.json` as a `torchcell.literature.manifest.Manifest`, files as `ArtifactRecord`
with `role=ROLE_RAW_DATA` and a `RetrievalRecord` naming one of the
`RetrievalMethod` members (`direct_url`, `zenodo`, `pmc_oa_api`, `manual_browser`). The
Vanacloig loader is the reference implementation: module-level `CITATION_KEY`,
`DATA_URL`, `DATA_SHA256`, `PAPER_MD_SHA256`, a `deposit_raw_mirror(...)` that is
idempotent by sha256 and refuses a differing file rather than overwriting, a
`load_manifest` / `manifest_sha256` pair, and `verify_raw_files` / `check_manifest_pin` /
`link_verified` from `torchcell.data.experiment_dataset` wiring the pins into `download`.
Two bacterial specifics change the retrieval mix: several rows live in SRA/ENA BioProjects
and figshare rather than publisher SI, and four rows' only data location is a portal behind
a challenge (the expansion document's access findings), which means
`RetrievalMethod.manual_browser` with a written human recipe and the deposited bytes as the
authority. One *P. putida* campaign is seven separate proteomics accessions, so its
`deposit_raw_mirror` enumerates all seven or the mirror holds a seventh of the data.

**Shared bacterial loader skeleton**, `torchcell/datasets/bacteria_common.py`:

- `bacterial_genome(host, strain, data_root)` returning the right
  `AnnotatedGenome` subclass from a cached root, the analogue of
  `scerevisiae/gene_name_reconcile.default_genome`.
- `reconcile_locus_tags(genome, names, *, label)` with the retain-all policy of
  `reconcile_systematic_names`: a source name that resolves to a current locus tag is
  remapped unless the tag is already claimed by another record in the same dataset, and a
  retired or ambiguous name is kept verbatim with its status logged. Bacterial resolution
  order per section 2: locus tag, `old_locus_tag`, gene symbol, `gene_synonym` (ECK/JW),
  retired.
- `AssemblyReferenceGenome` factory per strain, so a loader cannot mistype the set id.
- `bnumber_namespace` / `pputida_namespace` constants and the three compiled patterns, so
  the namespace Literal values exist in one place.
- An `eck_crosswalk(mg1655_genome, bw25113_genome)` builder returning the 4,423 one-to-one
  ECK pairs with the 11 numeric disagreements flagged, used only where a paper reports one
  background's identifiers for an experiment done in the other. Any use of it is recorded
  on the record as a derived mapping, never silently.

**Per-dataset checklist** (the bacterial form of the "Adding Datasets (Modular)" rules in
`CLAUDE.md`), to be copied into each loader's PR body:

1. **Pin the paper.** `citation_key` plus the mirrored `paper.md` sha256 from that key's
   `manifest.json` on `tc-lit`; every sourced value quotes a verbatim substring of those
   bytes.
2. **Identify the exact SI table and the exact column.** Name the file, the sheet or table
   number, and the column the loader consumes. Different columns carry different replicate
   structures; the statistic stored is the one belonging to the consumed column.
3. **Settle the background strain before anything else.** MG1655 or BW25113 for an *E.
   coli* row (the expansion document's `K-12` label is deliberately ambiguous and is NOT
   the answer), which assembly set it pins, and the quote that establishes it. A row whose
   paper does not say gets a typed `ProvenanceGap`, not a guess.
4. **Map every reported identifier to the pinned assembly's locus tags** through
   `reconcile_locus_tags`, and report the status histogram: how many CURRENT, how many
   resolved from a symbol or an ECK synonym, how many retired and kept, how many ambiguous.
   A row whose identifiers resolve below a stated threshold stops and reports rather than
   dropping records.
5. **Source `n_samples` and the uncertainty TYPE with a verbatim quote.** Combing the full
   Methods and the SI column descriptions, searching every synonym (`replicate`, `colony`,
   `biological replicate`, `bootstrap`, `standard deviation`, `n =`), and following
   deferrals into the cited method paper (RB-TnSeq rows will defer to Wetmore 2015, which is
   itself row 2, so that paper is mirrored anyway). On a range with no per-record column,
   apply the documented resolution order: back-solve from a companion statistic, else the
   conservative lower end, and say which rule was used.
6. **Media and environment from the Methods**, through the `MEDIA_LIBRARY` additions of
   section 3d, with the carbon source as a component or as a typed physical factor
   depending on whether it is fixed or varied.
7. **Deduplicate against the supersets.** The expansion document names two: the dense *P.
   putida* 4,732-gene-by-332-sample fitness matrix is a superset of the other KT2440
   transposon rows, and the *E. coli* compendium is a superset of the method paper's own
   *E. coli* experiments. The superset is loaded; the subsumed rows become provenance
   records naming which source experiments the matrix covers, not separate loaders.
8. **Write the dataset's Dendron note** with the sourcing decisions, the dropped-record
   counts and the identifier histogram, as the yeast loaders do
   ([[torchcell.datasets.scerevisiae.vanacloig2022]] is the model).
9. **Build the LMDB in the dev tree and report the manifest**:
   `python -m torchcell.database.build_dataset_lmdb --dataset <Class>`, which writes
   `preprocess/build_manifest.json`; the KG build's freshness gate reads exactly that file.

Acceptance per loader: the LMDB builds, `len(dataset)` matches the record count the PR
states, the build manifest reads fresh under
`python -m torchcell.provenance.build_manifest`, and the family verifier passes L0 to L4.

### 5. Adapters and the KG build

**Adapters are a thin per-dataset file plus a conf enable-list, and the bacterial ones are
the same shape.** `CellAdapter` already registers `genome` among its node methods and
builds the genome node as the sha256 of `data.reference.genome_reference.model_dump()`
with `species` and `strain` as queryable properties. Because an `AssemblyReferenceGenome`
is dumped in full where the field is narrowed to it (section 3b), a bacterial genome node
gets a distinct id from a yeast one for free, and per-host genome nodes separate without a
new method. What is needed:

1. **A new graph node class per new record type**, in
   `biocypher/config/torchcell_schema_config.yaml`: `product titer phenotype`,
   `protein turnover phenotype`, `flux phenotype`, and (per 3f item 3)
   `bacterial perturbation` with `is_a: genotype` carrying
   `systematic_gene_name`, `perturbed_gene_name`, `perturbation_type`, `description`,
   `gene_namespace` and `strain_id`. New classes are additive under the admission gate;
   changing `perturbation` would not be. Each needs an `is_a` that resolves in BioCypher's
   pinned Biolink head, or the ontology build fails before the first node is written (the
   `crispr construct` comment in that file records this exact failure).
2. **Two new properties on the `genome` node**, `assembly_set` and `assembly_accession`.
   This IS a change to an existing served class, so if the owner wants them queryable it
   rides the full rebuild; the serialized record carries them either way, so deferring
   costs only Cypher convenience.
3. **`CellAdapter` node methods for the new phenotypes**, each following
   `_fitness_phenotype_node`'s shape exactly: id is sha256 of the phenotype's
   `model_dump()`, properties are the queryable scalars only, no `serialized_data`. A NEW
   method is additive under the adapter-drift check; editing an existing method that a
   served conf enables is not, so the bacterial work adds methods and touches none.
4. **One adapter module and one conf yaml per bacterial dataset**, copied from
   `vanacloig2022_adapter.py` and its conf. The conf is where a dataset says which node and
   edge methods it enables, and the admission gate checks that an adapter enables only
   phenotype methods the graph schema declares, because BioCypher drops undeclared classes
   silently.
5. **`dataset_adapter_map` entries** and the `torchcell/adapters/__init__.py` re-exports.
6. **A KG config listing the bacterial datasets**, `torchcell/knowledge_graphs/conf/kg_bacteria.yaml`
   for a bacteria-only rehearsal build, and the fifty appended to `kg_uncapped.yaml` for the
   live rebuild.
7. **The genome-injection branch in the two build entry points.** `create_kg.py` and
   `create_scerevisiae_kg_small.py` construct ONE `SCerevisiaeGenome` and inject it into any
   loader declaring a `genome` parameter. A bacterial loader declaring `genome` would be
   handed a yeast genome. Fix: inject by parameter NAME per host
   (`ecoli_genome`, `pputida_genome`), constructed lazily so a yeast-only build never
   touches the bacterial tier. The same three-line pattern repeats in
   `torchcell/database/build_dataset_lmdb.py` and `torchcell/verification/runners.py`, and
   all three must change together or a dev LMDB build silently resolves bacterial names
   against S288C. This is the highest-risk edit in the program and it is cheap: it is the
   kind of wrong-path bug `[[experiments.010-kuzmin-tmi.false-torchmetrics-bug-bc-wrong-dataset-path]]`
   records.
8. **Verification runners per bacterial family**, with `_ecoli_k12_gene_set` /
   `_pputida_gene_set` feeding the L4 containment rule and each host's
   `resolve_gene_name` feeding the canonical-name rule. `run_all` gains the new runners.
   The L4 rule as written fails a record keyed to a name the current genome does not carry,
   which is the behavior we want per host and would be nonsense cross-host, so the gene
   universe must be selected by the record's own reference genome.

**Rebuild preparation checklist**, in order, nothing in it touching the served store:

1. Every bacterial dataset has a dev-tree LMDB with a fresh `build_manifest.json`, and
   every one of the 51 existing dev stores still reads fresh. The live-rebuild slurm script
   fails before it starts otherwise, and it names the stale ones.
2. `python -m torchcell.knowledge_graphs.kg_manifest --manifest <m> admit --dataset <Class>
   --data-root <dev DATA_ROOT>` run per bacterial dataset, with the output recorded. Expect
   a value-drift block from `media.py` (3d); the acknowledgment reason names the added keys
   and asserts no existing key changed.
3. `cmp` of the worktree's `torchcell_schema_config.yaml` against the build tree's copy,
   which the slurm script does itself and fails on.
4. Adapter configs checked against the schema config: every enabled phenotype method has a
   declared class.
5. A bacteria-only rehearsal generation (`kg_bacteria.yaml`) to catch BioCypher ontology
   failures and undeclared-class drops before the multi-hour full build.
6. Only then `sbatch database/slurm/scripts/gilahyper_live_rebuild-slurm_docker.slurm` with
   an explicit `--cpus-per-task=64`, per
   [[database.slurm.scripts.gilahyper_live_rebuild-slurm_docker]]. The script builds beside
   the served store, validates, swaps, and keeps the old store as `data.superseded.<ts>`.
   **This plan does not run it and no step below runs it without the owner.**

### 6. Ordered implementation steps

Every step is one worktree branch created with `/setup-worktree <branch>`, worked in
`~/Documents/projects/torchcell.worktrees/<branch>/`, opened as a PR and landed through
`/enqueue-merge`. Steps 1 and 2 are the only serial pair; after step 4 lands, the loaders
fan out. `PYTHONPATH=<worktree>` and `~/miniconda3/envs/torchcell/bin/python` throughout.
No step runs a KG build.

**Step 1 -- `feat/bacterial-genomes-tier`.** Parallel with step 2 (they share no file).

- Scope: the four assembly sets of section 1. New ids in
  `torchcell/sequence/genome/registry.py`; new `scripts/provision_bacterial_genomes.py`
  modeled on `scripts/migrate_genomes_tier.py`; the deposit run on GilaHyper.
- Files: `torchcell/sequence/genome/registry.py`,
  `scripts/provision_bacterial_genomes.py`,
  `tests/torchcell/sequence/genome/test_registry.py`,
  `notes/scripts.provision_bacterial_genomes.md`.
- Acceptance: `python scripts/provision_bacterial_genomes.py --refetch-dir <scratch> --dry-run`
  printing the four manifests with the sha256 values in the Verified table; then the real
  run; then for each set `verify_assembly_set("<set>")` returning one digest per member; a
  test asserting `resolve` raises `GenomeIntegrityError` on a corrupted copy and
  `FileNotFoundError` naming the rsync command when the set is absent; and
  `pytest tests/torchcell/sequence/genome/test_registry.py -x`.
- The step **re-fetches and re-hashes**; it does not trust the digests in this note. A
  mismatch is reported as upstream drift, with the new digest recorded, and does not
  silently proceed.

**Step 2 -- `refactor/genome-organism-agnostic`.** Parallel with step 1.

- Scope: section 2 layers 1 and 2 only. Extract `AnnotatedGenome` into
  `torchcell/sequence/genome/base.py`, reduce `s288c.py` to the SGD specifics, apply the
  D3 renames. **No bacterial subclass in this step**, so the diff is a pure refactor whose
  acceptance is byte-level equivalence.
- Files: new `torchcell/sequence/genome/base.py`;
  `torchcell/sequence/genome/scerevisiae/s288c.py`, `.../scerevisiae/__init__.py`,
  `torchcell/sequence/genome/__init__.py`, `torchcell/sequence/__init__.py`;
  `torchcell/datamodels/schema.py` (the constant rename only);
  `torchcell/verification/runners.py` (names only);
  `tests/torchcell/sequence/genome/scerevisiae/test_s288c.py`,
  `test_s288c_synthetic.py`; `notes/torchcell.sequence.genome.base.md`.
- Acceptance: `database_content_digest` of a freshly built `data.db` equal before and
  after; `len(genome.gene_set)` unchanged; the two s288c test modules green unchanged;
  `pytest tests/torchcell/sequence tests/torchcell/datamodels -x`; `ruff` and `mypy` clean
  on the touched files; and a grep showing no new import of
  `torchcell.sequence.genome.scerevisiae` from `base.py`.
- Note for the implementer: the concurrency machinery's comments document real incidents
  (torn `data.db`, kept journals, pre-2026.10.01 writers). Move it verbatim. Do not
  simplify it while moving it.

**Step 3 -- `feat/bacterial-genome-classes`.** After steps 1 and 2.

- Scope: `torchcell/sequence/genome/ecoli/k12.py` (MG1655 and BW25113) and
  `torchcell/sequence/genome/pputida/kt2440.py`, each with its GenBank-first ingest, its
  GFF3 route, its GAF-backed GO, and its bacterial `resolve_gene_name` layer order.
- Files: the two new subpackages with `__init__.py`; `torchcell/sequence/genome/base.py`
  (hooks only, if a gap appears); new tests
  `tests/torchcell/sequence/genome/ecoli/test_k12.py`,
  `tests/torchcell/sequence/genome/pputida/test_kt2440.py`; two Dendron notes.
- Acceptance: each genome builds from the tier with no network call (assert by running with
  the network-touching helper monkeypatched to raise); gene-feature counts 4,651 / 4,490 /
  5,786 from the GenBank route; GO term counts reported against the measured GAF coverage
  (4,016 b-numbers for MG1655, 3,912 `PP_` tags for KT2440) and a BW25113 figure consistent
  with D9's chosen source; `resolve_gene_name` returning CURRENT for `b0002`, RENAMED for
  `thrA`, RENAMED for `ECK0002`, RETIRED for a fabricated tag, per host; and a test that
  the ECK crosswalk finds 4,423 one-to-one pairs with 11 numeric disagreements, so the
  measurement is pinned rather than remembered.

**Step 4 -- `feat/bacterial-schema`.** After step 3; nothing downstream starts before it
lands.

- Scope: sections 3a, 3b, 3c and 3e. The five perturbation leaves,
  `AssemblyReferenceGenome`, the three new phenotypes with their experiment pairs, the
  bacterial background siblings, the union and map appends.
- Files: `torchcell/datamodels/schema.py`; `torchcell/datamodels/__init__.py`;
  `tests/torchcell/datamodels/test_ontology_invariants.py` (the CURIE regex, the leaf
  factory), `test_schema.py`, `test_identity.py`;
  `notes/torchcell.datamodels.bacterial-perturbation-ontology.md`.
- Acceptance: `pytest tests/torchcell/datamodels -x` green; the schema-impact report
  showing **zero** affected served loaders (this is the measurement that proves the step
  stayed additive, and it must be in the PR body); a round trip per new leaf through
  `Genotype` and through its experiment pair; and the yeast leaves still rejecting
  bacterial names.
- Deliberately NOT in this step: `media.py` (step 5) and the graph schema (step 6), so
  that the one commit the whole program depends on carries no value-surface or
  graph-schema drift.

**Step 5 -- `feat/bacterial-media-library`.** Parallel with step 6, after step 4.

- Scope: section 3d. `LB`, `LB_AGAR`, `M9`, `M9_GLUCOSE`, `MOPS_MINIMAL` and the
  first-tranche derived media, each sourced from a mirrored paper with a quote and sha256,
  each with `base_medium` naming a library key.
- Files: `torchcell/datamodels/media.py`; `tests/torchcell/datamodels/test_media*.py`;
  `notes/torchcell.datamodels.media.md` (append a dated section).
- Acceptance: `_check_library()` passing at import (it runs there, so an import is the
  test); every new single-substance component carrying an InChIKey or ChEBI through
  `resolved_compound`, or an explicit `intrinsically_undefined` for LB's tryptone and yeast
  extract; a new test pinning the `media_identity` digest of every pre-existing
  `MEDIA_LIBRARY` key, so the value-drift acknowledgment in the admission step is backed
  by a check; `pytest tests/torchcell/datamodels -x`.

**Step 6 -- `feat/bacterial-graph-classes`.** Parallel with step 5, after step 4.

- Scope: section 5 items 1 and 3: the new BioCypher node classes and the new `CellAdapter`
  node methods, with `bacterial perturbation` as a NEW class so `perturbation` is untouched.
- Files: `biocypher/config/torchcell_schema_config.yaml`;
  `torchcell/adapters/cell_adapter.py` (new methods only);
  `tests/torchcell/datamodels/test_ontology_all_trees.py` and the ontology-coherence check
  that reads emitted labels statically; `tests/torchcell/adapters/`.
- Acceptance: the ontology-coherence test green (it reads labels from the source, so a
  literal label mismatch fails there); every new class resolving an `is_a` in the pinned
  Biolink head; `pytest tests/torchcell/adapters tests/torchcell/datamodels -x`; and a
  `kg_manifest` dry comparison showing the graph-schema diff as ADDED classes only, never
  CHANGED.

**Step 7 -- `feat/bacterial-loader-skeleton`.** After step 4; parallel with 5 and 6.

- Scope: section 4's packages and `bacteria_common.py`, plus the host-aware genome
  injection of section 5 item 7 in all three entry points, plus the registry-population
  imports.
- Files: new `torchcell/datasets/ecoli/__init__.py`,
  `torchcell/datasets/pputida/__init__.py`, `torchcell/datasets/bacteria_common.py`;
  `torchcell/datasets/__init__.py`; `torchcell/knowledge_graphs/create_kg.py`,
  `create_scerevisiae_kg_small.py`; `torchcell/database/build_dataset_lmdb.py`;
  `torchcell/verification/runners.py`; `tests/torchcell/datasets/test_bacteria_common.py`.
- Acceptance: a test asserting that a loader declaring `ecoli_genome` receives an
  `EcoliK12Genome` and that one declaring `genome` still receives `SCerevisiaeGenome`, for
  each of the three injection sites; `reconcile_locus_tags` tested on a synthetic frame
  containing a current tag, a symbol, an ECK synonym, a collision and a retired name;
  `pytest tests/torchcell/datasets tests/torchcell/database -x`.

**Step 8 -- the first tranche of loaders, fanned out.** After steps 4, 5, 6 and 7 land.
Twenty background agents, one branch each, `feat/<host>-<firstauthor><year>`, each
implementing one row against the checklist in section 4 and its yeast analog as the
template. They share no file, so they are safely parallel; each one's PR body carries the
checklist with its answers.

| rank | host | paper | class | sequence basis | yeast analog (the template loader) | status | new module |
|---|---|---|---|---|---|---|---|
| 1 | E. coli | Fuhrer 2017 | Metabolome / flux | K-12-KO | Mulleder 2016 amino-acid metabolome | candidate | `ecoli/fuhrer2017.py` |
| 2 | E. coli | Wetmore 2015 | Transposon fitness | K-12+transposon | Hillenmeyer 2008 HIP/HOP | candidate | `ecoli/wetmore2015.py` |
| 3 | E. coli | Mutalik 2020 | Transposon fitness | K-12+transposon | Hillenmeyer 2008 HIP/HOP | candidate | `ecoli/mutalik2020.py` |
| 4 | E. coli | Tong 2020 | Fitness / chemical genomics | K-12-KO | Smith 2006 chemogenomic | candidate | `ecoli/tong2020.py` |
| 5 | E. coli | PRECISE-1K | Transcriptome | K-12-KO | Kemmeren 2014 deletion transcriptome | candidate | `ecoli/precise1k.py` |
| 6 | P. putida | Carruthers 2025 | Production campaign | KT2440+guide | Lian 2019 CRISPR-AID | candidate | `pputida/carruthers2025.py` |
| 7 | P. putida | Lim 2022 putidaPRECISE321 | Transcriptome | reference-only | Caudal 2024 pan-transcriptome | aggregation | `pputida/lim2022.py` |
| 8 | P. putida | Borchert 2024 fModules | Transposon fitness | KT2440+transposon | Hillenmeyer 2008 HIP/HOP | aggregation | `pputida/borchert2024.py` |
| 9 | E. coli | Caglar 2017 | Multi-omics campaign | reference-only | Zelezniak 2018 proteome and metabolome | candidate | `ecoli/caglar2017.py` |
| 10 | E. coli | Nichols 2011 | Fitness / chemical genomics | K-12-KO | Hillenmeyer 2008 HIP/HOP | candidate | `ecoli/nichols2011.py` |
| 11 | E. coli | Goodall 2018 | Transposon fitness | K-12+transposon | SGD essentiality | candidate | `ecoli/goodall2018.py` |
| 12 | E. coli | Foo 2014 isopentenol tolerance | Production campaign | engineered-chassis | Lopez 2024 isobutanol | candidate | `ecoli/foo2014.py` |
| 13 | P. putida | Yunus 2026 | Production campaign | KT2440+guide | Lian 2019 CRISPR-AID | candidate | `pputida/yunus2026.py` |
| 14 | P. putida | de Siqueira 2025 | Tolerance / robustness | evolved-WGS | Mormino 2022 CRISPRi acetic acid | candidate | `pputida/desiqueira2025.py` |
| 15 | E. coli | Wang 2015 isoprenol tolerance | Tolerance / robustness | K-12-KO | Lopez 2024 isobutanol | candidate | `ecoli/wang2015.py` |
| 16 | P. putida | Lim 2025 isoprenol TALE | Tolerance / robustness | evolved-WGS | Mormino 2022 CRISPRi acetic acid | candidate | `pputida/lim2025.py` |
| 17 | E. coli | Tian 2019 isopentenol CRISPRi | Combinatorial design | engineered-chassis+guide | Lian 2019 CRISPR-AID | blocked | `ecoli/tian2019.py` |
| 18 | P. putida | Kang 2026 isoprenyl acetate | Production campaign | engineered-chassis | Lopez 2024 isobutanol | candidate | `pputida/kang2026.py` |
| 19 | P. putida | Wang 2022 P. putida isoprenoids | Production campaign | engineered-chassis | Ozaydin 2013 beta-carotene | blocked | `pputida/wang2022.py` |
| 20 | P. putida | Menasalvas 2025 | Production campaign | engineered-chassis | Lopez 2024 isobutanol | candidate | `pputida/menasalvas2025.py` |

Four of the twenty need their row-specific schema question answered first (`schema_need`
non-empty for ranks 12, 17 and 19, and rank 14's evolved-clone genotype), and three carry
a `blocked` or `aggregation` status, so those seven agents' first deliverable is a written
finding in their PR, not a loader. The remaining thirteen are retrieval, provenance and
mapping work against an existing record type. Ranks 2 and 3 and ranks 8 and the other
KT2440 transposon rows are the superset pairs of checklist item 7, so rank 3's agent
de-duplicates against rank 2 rather than loading both, and the same for the KT2440 matrix.

**Step 9 -- `feat/bacterial-adapters-tranche-1`.** After step 8's loaders land. One
adapter module plus one conf yaml per landed loader, the `dataset_adapter_map` entries, the
`__init__.py` re-exports, `kg_bacteria.yaml`, and the per-family verification runners.
Acceptance: `pytest tests/torchcell/adapters tests/torchcell/verification -x`; each
adapter's conf enabling only declared classes; each dataset's verifier passing L0 to L4
against its dev LMDB.

**Step 10 -- `chore/bacterial-rebuild-preparation`.** After step 9. The checklist at the
end of section 5, items 1 to 5 only, with every `admit` output recorded in the PR and in a
dated section of this note. The rebuild itself (item 6) is the owner's call and is not part
of any branch.

**What is explicitly out of scope for this plan.** The second tranche of thirty loaders
(same shape as step 8, after the first tranche is served). Pinning the yeast `go.obo` into
the tier. Unifying the yeast and bacterial `StrainBackground` families. Backfilling
`assembly_set` onto the 51 served yeast datasets. The `systematic_gene_name` rename, unless
D2 is decided the other way, in which case it folds into step 4 and the full rebuild
becomes mandatory rather than chosen.

### Open questions for the owner

1. **D2** -- does `systematic_gene_name` keep its name? Recommendation: yes, with the
   namespace carried in a sibling field.
2. **D6** -- assembly pin as a new subclass (0 of 36 closures, yeast stays implicit) or as
   a field on `ReferenceGenome` (36 of 36, one uniform substrate, rides the rebuild)?
   Recommendation: the subclass.
3. **D9** -- BW25113 GO from its own IEA-only RefSeq GFF terms, or from MG1655's richer
   GAF through the 4,423 ECK pairs? Recommendation: the former as stored, the latter as a
   labeled derived view.
4. **Section 5 item 2** -- are `assembly_set` and `assembly_accession` wanted as queryable
   `genome` node properties now (forcing the full rebuild) or left inside the serialized
   record for now?
