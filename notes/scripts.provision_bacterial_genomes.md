---
id: wy43fr1eat8hj9y1rbyca9h
title: Provision_bacterial_genomes
desc: ''
updated: 1791371103033
created: 1791371103033
---


## 2026.10.07 - Bacterial assembly sets deposited

`scripts/provision_bacterial_genomes.py` seeds the genomes tier with the four sets of
[[plan.bacteria-ontology-genome]] section 1: one per strain (*E. coli* K-12 MG1655,
*E. coli* K-12 BW25113, *P. putida* KT2440) and one shared GO ontology release. Registry ids
and the tier contract: [[torchcell.sequence.genome.registry]]. Same shape as
[[scripts.migrate_genomes_tier]]: fetch, check, build a pydantic `GenomeManifest` per set,
copy into the tier, `deposit_assembly_set`, `verify_assembly_set`.

Every member is fetched by the `torchcell.literature.retrieve.direct_url` retriever (the
function named in each `RetrievalRecord`) into `--refetch-dir/<assembly_set>/`, with a
`<file>.retrieval.json` sidecar holding the fetch's `RetrievalRecord`. A second run reuses
a fetched file only when its bytes still match that sidecar, so the dry run and the
deposit share one recorded fetch. Before anything reaches the tier the script checks
every NCBI member's md5 against its directory's `md5checksums.txt`, refuses a listing that
names a `_gene_ontology.gaf.gz` the set does not carry, and compares bytes and sha256 with
the digests the plan measured. A difference there is upstream drift: the run prints the
new digest and stops unless the file is named with `--accept-drift`, in which case the
manifest notes record both digests.

### Run on GilaHyper, 2026-10-07

The refetch dir was a fresh, empty directory in the session scratch; the deposit copy in
the tier is the durable one. Python ran isolated (`-I`, worktree put on `sys.path`
explicitly) because the fetched files are untrusted bytes:

```bash
WT=~/Documents/projects/torchcell.worktrees/feat/bacterial-genomes-tier
R=<session scratchpad>/genomes-refetch
python -I -c "import sys,runpy; sys.path.insert(0,'$WT'); \
  sys.argv=['provision', '--refetch-dir', '$R', '--dry-run']; \
  runpy.run_path('$WT/scripts/provision_bacterial_genomes.py', run_name='__main__')"
# then the same without --dry-run
```

- Dry run (09:55 to 09:57 UTC): 29 members and 6 `md5checksums.txt` listings fetched, all
  HTTP 200.
- **Drift: none.** All 29 members equal the plan's Verified digests in bytes and sha256.
- **md5: 26 of 26 NCBI members match** their directory's `md5checksums.txt` (8 for
  MG1655, 9 for BW25113, 9 for KT2440). The three non-NCBI files (`ECOLI-uniprot.gaf.gz`,
  `109.P_putida_KT2440.goa`, `go-basic.obo`) have no published checksum; sha256 is their
  only anchor.
- Deposit: no file re-fetched (sidecars matched), four manifests written, then
  `verify_assembly_set` re-hashed every member: 9 + 9 + 10 + 1 = 29 digests, all equal to
  the table below. Every manifest is `provenance_complete: true`.
- Tier total 71,169,015 bytes of members (10,999,761 / 10,299,467 / 17,642,002 /
  32,227,785).

Headers read from the deposited files themselves (recorded in each manifest's notes):

| file | header |
|---|---|
| `ECOLI-uniprot.gaf.gz` | `!gaf-version: 2.2`, `!generated-by: UniProt`, `!date-generated: 2026-07-28 19:40` |
| `109.P_putida_KT2440.goa` | `!gaf-version: 2.2`, `!generated-by: UniProt`, `!date-generated: 2026-07-28` |
| `go-basic.obo` | `data-version: releases/2026-07-26` |

Replicons, read from the three GCA `_assembly_report.txt` members: MG1655 `U00096.3` /
`NC_000913.3`, 4,641,652 bp; BW25113 `CP009273.1` / `NZ_CP009273.1`, 4,631,469 bp; KT2440
`AE015451.2` / `NC_002947.4`, 6,181,873 bp.

### Members, as measured

### ecoli_K12_MG1655_ASM584v2 (9 members, 10,999,761 bytes)

| member | role | bytes | sha256 | md5 vs NCBI |
|---|---|---|---|---|
| `GCA_000005845.2_ASM584v2_genomic.gbff.gz` | annotation | 3,401,246 | `50d48e5dc7c6ad18699db4819bc11540d61505cacaa57aedf28ac132f26a7d87` | `aa6aa3c65ea561201b606943bdc89f36` match |
| `GCA_000005845.2_ASM584v2_genomic.fna.gz` | sequence | 1,379,898 | `a8bf0111c936605eb3b8d73d5a5c1dbb64ef39c87553608a6752719995e9d9ba` | `7e69874199f23fd21b060dc0b2b72321` match |
| `GCA_000005845.2_ASM584v2_genomic.gff.gz` | annotation | 384,317 | `5be073e97f95efe337d1e5b69cd5af61d8cbee3450f8cb92c3ea722c8cb0af9b` | `a9636a4e6c974711f2950d31d6200f81` match |
| `GCA_000005845.2_ASM584v2_protein.faa.gz` | sequence | 901,247 | `900cb656bba2f4fde8dafc43c80c84ddd326c85ef1d10b476e0a49a104db5263` | `5dc32e1ffb79c813b79c8f356a8fc739` match |
| `GCA_000005845.2_ASM584v2_feature_table.txt.gz` | index | 190,280 | `50ad5af9f85813b68a9a96f7989b588f28eafc5ace492ecbf93213f139da015e` | `0b278507718a42f4d6ec569c02f80f1b` match |
| `GCA_000005845.2_ASM584v2_assembly_report.txt` | index | 1,207 | `b40db266b9b21654c6ba07b8102e6e7451ce41b94539d92e591cc24a46fcc8b9` | `178dd83e5751f91ff7ae70c7716cd5bb` match |
| `GCF_000005845.2_ASM584v2_genomic.gbff.gz` | annotation | 3,401,424 | `cbcc40a9859312cbcdbeb3df484ee471dfe354c5cd8f28056b48431353b28384` | `3a48563f7ebe3e89912f9280cfb7e1ec` match |
| `GCF_000005845.2_ASM584v2_genomic.gff.gz` | annotation | 387,627 | `afdf03dc1d06e423d874ee29d9e0df14f5d32baf893f4da9f93aa721eb5f495e` | `0f52ffc94af5ddf544ff89cc6f546b0c` match |
| `ECOLI-uniprot.gaf.gz` | annotation | 952,515 | `ad338c31d8114ce5579a43be4b8e3b78ab366541968cf76c3a91f82b353a9cdf` | none published |

### ecoli_K12_BW25113_ASM75055v1 (9 members, 10,299,467 bytes)

| member | role | bytes | sha256 | md5 vs NCBI |
|---|---|---|---|---|
| `GCA_000750555.1_ASM75055v1_genomic.gbff.gz` | annotation | 3,416,102 | `41a48b1a84ecf0a533bec6041c65b71894448da3612ed93e4cc7ebc300f59ea3` | `ee5b08815644cbbc72928137c8b24800` match |
| `GCA_000750555.1_ASM75055v1_genomic.fna.gz` | sequence | 1,377,238 | `bdc393dceb31717b63a7b4a87ebfc73b579ca4b014680aa0e066a5a20f9b50ae` | `3fe88314fac8a0dfd3bbd3ec1504532a` match |
| `GCA_000750555.1_ASM75055v1_genomic.gff.gz` | annotation | 385,835 | `9f79536c9b3f134d1f773349144e50a97c0b5b5f8824496a7c8f549a734187a1` | `f456f91024667d5731b0591a543ea23d` match |
| `GCA_000750555.1_ASM75055v1_protein.faa.gz` | sequence | 894,262 | `03bdd3a48e14f35f79a65b40c633b4270917a29ad1f4dd58283edde588f48253` | `9cf96157a7f634239351f7551c912420` match |
| `GCA_000750555.1_ASM75055v1_feature_table.txt.gz` | index | 200,398 | `796f9e961a2d351f8a5a9314700ae32273bd979867af0663b08d4d15c0a7d9ce` | `bfb423a73ec215fae2ae0d9b189d44ef` match |
| `GCA_000750555.1_ASM75055v1_assembly_report.txt` | index | 1,322 | `b79737157aadd821bbfb1df33a063174ff1d22ce705bd794dfa96330ef15e4f4` | `d7dfbe94d7de62c13110379d716640d8` match |
| `GCF_000750555.1_ASM75055v1_genomic.gbff.gz` | annotation | 3,428,964 | `1f28a28be37211bb43ce4bfb520f8acdcd61189de61dd180e5ac2bfc94a6b9bf` | `c6bdc659aec4a7cad60b9484d7d1f652` match |
| `GCF_000750555.1_ASM75055v1_genomic.gff.gz` | annotation | 438,196 | `b5d361ed256bf30f5e2522c54239b241cf2787b7b5a03d4ebf2c95734ee3b1bd` | `082dda2cbb0b99f2bf081a36a884b3df` match |
| `GCF_000750555.1_ASM75055v1_gene_ontology.gaf.gz` | annotation | 157,150 | `20798489747e477161a092f33e0b83f4369e70415d05f372d8d40d08d268f0bd` | `564a508f7a23f0f60ae31681e2a6e1f6` match |

### pputida_KT2440_ASM756v2 (10 members, 17,642,002 bytes)

| member | role | bytes | sha256 | md5 vs NCBI |
|---|---|---|---|---|
| `GCA_000007565.2_ASM756v2_genomic.gbff.gz` | annotation | 4,361,085 | `dbc3666fda5443db2c95040fd870585c92fbfb609ab2939821131fce40579bb0` | `e84e38fd3d7d5a2ebab1db5bf5f458c9` match |
| `GCA_000007565.2_ASM756v2_genomic.fna.gz` | sequence | 1,802,201 | `7893b5160c1ff24ab4e8f5faa2861051d455a2d8ded7e9de1f9eea4b99b66664` | `8e634e2559849e40059c0518e8b35f9d` match |
| `GCA_000007565.2_ASM756v2_genomic.gff.gz` | annotation | 391,311 | `703bff9433dfd5d4b2e9aaebdccc3555be7790efaf5dba227a590f964c4978d0` | `b2f8b0ac6f32be9b3aa4834b15057500` match |
| `GCA_000007565.2_ASM756v2_protein.faa.gz` | sequence | 1,189,638 | `914c6cc763fb5f404dbfe9dcd16c3eacec98d4a477a0bec3caf52bc58a24f0a9` | `657be6d1b352b4f73a55689b36ba4f5a` match |
| `GCA_000007565.2_ASM756v2_feature_table.txt.gz` | index | 217,058 | `cb0fe25120a1e1506b165e5b3091d4a3d66ea0f5c36a447d634ea9912dc90666` | `aaf681e975254a509cb3ba86dfd7bf9d` match |
| `GCA_000007565.2_ASM756v2_assembly_report.txt` | index | 1,143 | `8b685bb9a0965e7db6bc5444bef3306cdac8fc0cdf34d704616b265753baa9c4` | `ecad2f97b3121362c90045456a247218` match |
| `GCF_000007565.2_ASM756v2_genomic.gbff.gz` | annotation | 4,472,217 | `f607d834deae4a5f4a8a041ee16dbf6436d9fdd9cf2fd5c9562820325b8d22c0` | `30a1a7383f973189beb44748684d22ac` match |
| `GCF_000007565.2_ASM756v2_genomic.gff.gz` | annotation | 488,028 | `69e01eb2ee6fdb12db203d0965fc2c4e23d156073378969032a246e31a7914aa` | `f0d37cd3e020899062bba7b69cd024b3` match |
| `GCF_000007565.2_ASM756v2_gene_ontology.gaf.gz` | annotation | 216,610 | `d3e54d8669aaaec1ba28f29f0ecf8b61b152109cdaddf3e533d114106ffded9a` | `aeed2f6f025301da6a9f94a2448f8b76` match |
| `109.P_putida_KT2440.goa` | annotation | 4,502,711 | `575731316d9fcb98580dd7e2209a0239c909ada389e42c4052e5a8f7a1069a81` | none published |

### go_release_2026-08-05 (1 members, 32,227,785 bytes)

| member | role | bytes | sha256 | md5 vs NCBI |
|---|---|---|---|---|
| `go-basic.obo` | ontology | 32,227,785 | `b08d45b268b8c24ccb2513dbbbc7d4df9f6521c099b413f79eb31e06e0fa3bcc` | none published |

The md5 listings themselves (sha256 of the `md5checksums.txt` fetched with the members):

| directory | sha256 of `md5checksums.txt` |
|---|---|
| `GCA_000005845.2_ASM584v2` | `28d8812e38d61c60f1afa64823460648b2683a7d4069088b07f653220f3fb960` |
| `GCF_000005845.2_ASM584v2` | `433d33545422cffdb762278ecc30256c3c45e8bff9f7859f103ba33f57bc6337` |
| `GCA_000750555.1_ASM75055v1` | `07e72ac9318d4804087adbe5b7c52659205c1fc7a5aa149a0917f774d2d2a599` |
| `GCF_000750555.1_ASM75055v1` | `9a864ec5a066fd1ae7fc9a7bdaf62119d2138a4504e67c050d1748a06e214c25` |
| `GCA_000007565.2_ASM756v2` | `0045f1d827b90d46fb2c4a89897efc7793c52a9fdc29600938307ac9b7515868` |
| `GCF_000007565.2_ASM756v2` | `e423e07ec2648f9124cb187c98aa3312e8f00533c9275dc242078e6ad3097a19` |

### Every retrieval URL

```
# ecoli_K12_MG1655_ASM584v2
https://ftp.ncbi.nlm.nih.gov/genomes/all/GCA/000/005/845/GCA_000005845.2_ASM584v2/GCA_000005845.2_ASM584v2_genomic.gbff.gz
https://ftp.ncbi.nlm.nih.gov/genomes/all/GCA/000/005/845/GCA_000005845.2_ASM584v2/GCA_000005845.2_ASM584v2_genomic.fna.gz
https://ftp.ncbi.nlm.nih.gov/genomes/all/GCA/000/005/845/GCA_000005845.2_ASM584v2/GCA_000005845.2_ASM584v2_genomic.gff.gz
https://ftp.ncbi.nlm.nih.gov/genomes/all/GCA/000/005/845/GCA_000005845.2_ASM584v2/GCA_000005845.2_ASM584v2_protein.faa.gz
https://ftp.ncbi.nlm.nih.gov/genomes/all/GCA/000/005/845/GCA_000005845.2_ASM584v2/GCA_000005845.2_ASM584v2_feature_table.txt.gz
https://ftp.ncbi.nlm.nih.gov/genomes/all/GCA/000/005/845/GCA_000005845.2_ASM584v2/GCA_000005845.2_ASM584v2_assembly_report.txt
https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/005/845/GCF_000005845.2_ASM584v2/GCF_000005845.2_ASM584v2_genomic.gbff.gz
https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/005/845/GCF_000005845.2_ASM584v2/GCF_000005845.2_ASM584v2_genomic.gff.gz
https://release.geneontology.org/2026-08-05/annotations/gaf/ECOLI-uniprot.gaf.gz
# ecoli_K12_BW25113_ASM75055v1
https://ftp.ncbi.nlm.nih.gov/genomes/all/GCA/000/750/555/GCA_000750555.1_ASM75055v1/GCA_000750555.1_ASM75055v1_genomic.gbff.gz
https://ftp.ncbi.nlm.nih.gov/genomes/all/GCA/000/750/555/GCA_000750555.1_ASM75055v1/GCA_000750555.1_ASM75055v1_genomic.fna.gz
https://ftp.ncbi.nlm.nih.gov/genomes/all/GCA/000/750/555/GCA_000750555.1_ASM75055v1/GCA_000750555.1_ASM75055v1_genomic.gff.gz
https://ftp.ncbi.nlm.nih.gov/genomes/all/GCA/000/750/555/GCA_000750555.1_ASM75055v1/GCA_000750555.1_ASM75055v1_protein.faa.gz
https://ftp.ncbi.nlm.nih.gov/genomes/all/GCA/000/750/555/GCA_000750555.1_ASM75055v1/GCA_000750555.1_ASM75055v1_feature_table.txt.gz
https://ftp.ncbi.nlm.nih.gov/genomes/all/GCA/000/750/555/GCA_000750555.1_ASM75055v1/GCA_000750555.1_ASM75055v1_assembly_report.txt
https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/750/555/GCF_000750555.1_ASM75055v1/GCF_000750555.1_ASM75055v1_genomic.gbff.gz
https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/750/555/GCF_000750555.1_ASM75055v1/GCF_000750555.1_ASM75055v1_genomic.gff.gz
https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/750/555/GCF_000750555.1_ASM75055v1/GCF_000750555.1_ASM75055v1_gene_ontology.gaf.gz
# pputida_KT2440_ASM756v2
https://ftp.ncbi.nlm.nih.gov/genomes/all/GCA/000/007/565/GCA_000007565.2_ASM756v2/GCA_000007565.2_ASM756v2_genomic.gbff.gz
https://ftp.ncbi.nlm.nih.gov/genomes/all/GCA/000/007/565/GCA_000007565.2_ASM756v2/GCA_000007565.2_ASM756v2_genomic.fna.gz
https://ftp.ncbi.nlm.nih.gov/genomes/all/GCA/000/007/565/GCA_000007565.2_ASM756v2/GCA_000007565.2_ASM756v2_genomic.gff.gz
https://ftp.ncbi.nlm.nih.gov/genomes/all/GCA/000/007/565/GCA_000007565.2_ASM756v2/GCA_000007565.2_ASM756v2_protein.faa.gz
https://ftp.ncbi.nlm.nih.gov/genomes/all/GCA/000/007/565/GCA_000007565.2_ASM756v2/GCA_000007565.2_ASM756v2_feature_table.txt.gz
https://ftp.ncbi.nlm.nih.gov/genomes/all/GCA/000/007/565/GCA_000007565.2_ASM756v2/GCA_000007565.2_ASM756v2_assembly_report.txt
https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/007/565/GCF_000007565.2_ASM756v2/GCF_000007565.2_ASM756v2_genomic.gbff.gz
https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/007/565/GCF_000007565.2_ASM756v2/GCF_000007565.2_ASM756v2_genomic.gff.gz
https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/007/565/GCF_000007565.2_ASM756v2/GCF_000007565.2_ASM756v2_gene_ontology.gaf.gz
https://ftp.ebi.ac.uk/pub/databases/GO/goa/proteomes/109.P_putida_KT2440.goa
# go_release_2026-08-05
https://release.geneontology.org/2026-08-05/ontology/go-basic.obo
```

### BW25113 GO (plan D9)

The BW25113 set carries only GO that is about BW25113 itself: the RefSeq GFF's inline
`Ontology_term` rows (`GCF_000750555.1_ASM75055v1_genomic.gff.gz`) and NCBI's `WP_`-keyed
GAF (`GCF_000750555.1_ASM75055v1_gene_ontology.gaf.gz`), both IEA-only per the plan.
MG1655's richer `ECOLI-uniprot.gaf.gz` is NOT copied into it: it is referenced by set id
`ecoli_K12_MG1655_ASM584v2`, because mapping it to BW25113 through the ECK synonym asserts
that each BW25113 gene has its MG1655 orthologue's annotation, which is an inference, not
an annotation of the strain. Which source a BW25113 loader reads is still open and must be
decided before the first BW25113 row's GO is consumed. The reference is recorded as text
in the BW25113 manifest's `notes` (the `GenomeManifest` model has no structured
cross-set field), and the same holds for the three strain sets' pin of
`go_release_2026-08-05`.

### Seeding another machine

The canonical tier is on GilaHyper. A machine running BW25113 or KT2440 code also needs
the GO release set, and BW25113 needs MG1655 for the referenced GAF, so seed all four:

```bash
rsync -a gilahyper:/scratch/projects/torchcell-scratch/torchcell-genomes/ecoli_K12_MG1655_ASM584v2/ $DATA_ROOT/torchcell-genomes/ecoli_K12_MG1655_ASM584v2/
rsync -a gilahyper:/scratch/projects/torchcell-scratch/torchcell-genomes/ecoli_K12_BW25113_ASM75055v1/ $DATA_ROOT/torchcell-genomes/ecoli_K12_BW25113_ASM75055v1/
rsync -a gilahyper:/scratch/projects/torchcell-scratch/torchcell-genomes/pputida_KT2440_ASM756v2/ $DATA_ROOT/torchcell-genomes/pputida_KT2440_ASM756v2/
rsync -a gilahyper:/scratch/projects/torchcell-scratch/torchcell-genomes/go_release_2026-08-05/ $DATA_ROOT/torchcell-genomes/go_release_2026-08-05/
```

then confirm the bytes on the receiving machine:

```bash
python -c "from torchcell.sequence.genome import registry as r; [print(s, len(r.verify_assembly_set(s))) for s in (r.ECOLI_K12_MG1655, r.ECOLI_K12_BW25113, r.PPUTIDA_KT2440, r.GO_RELEASE_20260805)]"
```

which prints 9, 9, 10 and 1. The weekly `scripts/backup_mirrors_to_bulk.sh` copies the
whole tier to `/bulk/torchcell-genomes/` on Sundays, so no extra backup step is needed.
