---
id: 7j7ofnbwvxs2yjxtu6pkzpa
title: K12
desc: ''
updated: 1791374709347
created: 1791374709347
---


## 2026.10.07 - E. coli K-12 MG1655 and BW25113 genomes

Step 3 of [[plan.bacteria-ontology-genome]]. Module:
`torchcell/sequence/genome/ecoli/k12.py`; shared bacterial layer:
[[torchcell.sequence.genome.bacterial]]; organism-agnostic base:
[[torchcell.sequence.genome.base]]; tier: [[scripts.provision_bacterial_genomes]]. Tests:
[[tests.torchcell.sequence.genome.ecoli.test_k12]].

### Design

- `EcoliK12Genome` is parameterized by strain. Each strain is a pydantic
  `BacterialAssembly` (`MG1655_ASSEMBLY`, `BW25113_ASSEMBLY`) naming its assembly set,
  GenBank and RefSeq assembly names, replicon, locus-tag pattern, GO source and default
  cache root. A concrete class per strain binds it (`EcoliK12MG1655Genome`,
  `EcoliK12BW25113Genome`); `EcoliK12Genome.for_strain("BW25113", genome_root=...)`
  picks one by name. One class per strain keeps `ASSEMBLY_SET` a class-level fact,
  which `AnnotatedGenome.database_untrusted_reason` and `release_files` read without an
  instance; a `strain=` init field would have needed those classmethods rewritten.
- GenBank first: the GCA `_genomic.gbff.gz` gives the loci (locus tags, `/gene`,
  `/gene_synonym` split on `;`, `/old_locus_tag`, `/db_xref`, products, protein ids);
  the GCA `_genomic.gff.gz` builds the shared `data.db` (recorded with assembly set,
  GFF member and sha256, so a cache built from another set is detected); construction
  checks the database's gene and pseudogene features carry exactly the GenBank tags.
- The gene set is the non-pseudo GenBank loci. Pseudogenes are valid loci and resolve
  `NON_GENE_FEATURE`, the same semantics as an SGD `blocked_reading_frame`.
- One circular replicon: chromosome key 1 for every gene (`U00096.3` for MG1655,
  `CP009273.1` for BW25113); any other seqid is refused.
- Nothing is downloaded at construction. GO comes from deposited files and the DAG is
  `go-basic.obo` of the `go_release_2026-08-05` set.

### Measured counts (GilaHyper, 2026-10-07, from the deposited tier)

| | MG1655 | BW25113 |
|---|---|---|
| GenBank gene features | 4,651 | 4,490 |
| gene set (non-pseudo) | 4,506 | 4,303 |
| pseudogenes | 145 | 187 (18 joined around an insertion) |
| coding CDS = proteins re-keyed to locus tags | 4,290 | 4,131 |
| CDS whose translation differs from the protein | 3 (fdnG, fdoG, fdhF: Sec) | 3 (same genes) |
| GO route | `ECOLI-uniprot.gaf.gz`, column 11 | RefSeq GFF `Ontology_term` via `old_locus_tag` (D9) |
| GO rows read / NOT rows excluded / rows without an identifier | 56,262 / 20 / 1,729 | 2,278 / 0 / 11 |
| identifiers reached / not a locus of the annotation | 4,014 / 52 | 2,250 / 0 |
| genes with GO / pseudogenes with GO | 3,902 / 60 | 2,188 / 62 |
| distinct GO terms over the gene set | 4,082 | 1,632 |

The plan's "4,016 distinct b-numbers" was a substring count. Reading column 11 as
tokens (split on `|`, then on `/`, each token fully matching `b\d{4}`) gives 4,015 over
all rows and 4,014 after the 20 NOT rows. The one difference from 4,016 is `b0691`, which
the substring match read out of `b0691.1`; a Blattner `.1` name is a different gene
(inserted between two numbered genes), so reading it as `b0691` would be wrong. Splitting
on `/` (`b1239/b1240`, UniProt's join of several ordered locus names) adds 23 b-numbers,
all of which are historical numbers absent from GCA_000005845.2, so no gene gains or
loses GO from it. Every stored term is live in go-basic 2026-07-26
(`remove_deprecated_go_terms` removes nothing).

MG1655's GenBank file carries ECK synonyms but no JW numbers (0 of 8,405 synonym values),
so a JW number resolves only on BW25113 (4,334 JW synonym values there).

### D9: BW25113 GO

Stored default: BW25113's own RefSeq annotation, the 2,278 `Ontology_term` rows of
`GCF_000750555.1_ASM75055v1_genomic.gff.gz` mapped to GenBank tags through each RefSeq
gene's `old_locus_tag` (IEA only). The alternative, MG1655's GAF through the ECK
synonym, is an inference across strains and is exposed only as
`EcoliK12BW25113Genome.go_annotations_via_mg1655_eck()`, a `DerivedGoAnnotation` labeled
`view="mg1655_gaf_via_eck"` that no genome stores: 4,423 one-to-one pairs, 3,877
BW25113 loci reached, 3,799 of them genes, 4,066 terms. Calling it leaves
`go_annotations` unchanged (tested).

### ECK crosswalk (pinned by test)

`eck_crosswalk(mg1655, bw25113)` over every locus: 4,435 ECK ids in both strains,
4,423 one-to-one, 12 not one-to-one, 192 only in MG1655, 16 only in BW25113, and 11
one-to-one pairs whose numerics disagree: ECK0018 b0018/BW25113_4412, ECK0057
b0056/BW25113_4659, ECK0281 b0282/BW25113_4694, ECK1313 b1318/BW25113_4524, ECK1536
b1543/BW25113_4600, ECK2646 b2649/BW25113_2650, ECK2853 b2855/BW25113_2856, ECK2858
b2862/BW25113_2863, ECK3674 b3682/BW25113_3683, ECK4096 b4583/BW25113_4104, ECK4329
b4584/BW25113_4339. These match the plan's measurement exactly.

### resolve_gene_name round trips (tier)

| input | MG1655 | BW25113 |
|---|---|---|
| `b0002` | CURRENT b0002 | RETIRED (another namespace) |
| `BW25113_0002` | RETIRED | CURRENT |
| `thrA` | RENAMED b0002 (gene symbol) | RENAMED BW25113_0002 |
| `ECK0002` | RENAMED b0002 (gene synonym) | RENAMED BW25113_0002 |
| `JW0001` | RETIRED | RENAMED BW25113_0002 |
| `BW25113_RS00010` | not applicable | RENAMED BW25113_0002 (RefSeq locus tag) |
| `b9999` / `BW25113_9999` | RETIRED | RETIRED |

### Cache roots

Default roots (relative, like `data/sgd/genome`): `data/ecoli/mg1655/genome` and
`data/ecoli/bw25113/genome`. Seeded on GilaHyper under
`/scratch/projects/torchcell-scratch/data/ecoli/{mg1655,bw25113}/genome` with
`overwrite=True` on 2026-10-07; each `data.db` records its GCA GFF (sha256 prefixes
`5be073e97f95efe3` and `9f79536c9b3f134d`) and `database_untrusted_reason` returns None.
Build time is about 1.5 s per genome, reopen about 1 s. The tier tests build into pytest
temporary roots, never these.

### No-network guarantee

No bacterial module imports a download helper. The tests construct every genome (synthetic
and tier) with `socket.socket.connect`, `socket.create_connection`, `socket.getaddrinfo`,
`urllib.request.urlopen`/`urlretrieve`, `requests` and `torch_geometric`'s
`download_url` patched to raise (`_bacterial_fixtures.forbid_network`).
