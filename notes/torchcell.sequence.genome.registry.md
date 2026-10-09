---
id: yplo77fypz7lvl7qvl5to7l
title: Registry
desc: ''
updated: 1789364446771
created: 1789364446771
---

## 2026.09.14 - The genomes tier and its registry

`$DATA_ROOT/torchcell-genomes/<assembly_set>/` is the one on-disk home for sequence-level
reference data that the graph only points at. Each assembly set is a flat directory with a
pydantic `GenomeManifest` (`manifest.json`) that names the organism, the source release,
and every file with its sha256 and, where the bytes were reproduced from the source URL,
its `RetrievalRecord`. The per-file record is the literature mirror's `ArtifactRecord`; the
top level is a sibling of the literature `Manifest` because a genome belongs to an
organism and a release, not to a paper. Plan: [[plan.genomes-tier.2026.09.14]].

The module is the sole path authority for the loaders in scope: `resolve(assembly_set,
filename)` returns an absolute path after checking the manifest lists the file and, by
default, that its sha256 matches; `verify_assembly_set` walks a whole set for the backup
and bit-rot check; `deposit_assembly_set` writes a manifest only for files already in place
whose bytes match the records, and refuses an existing manifest. There is no fallback to
`data/sgd/genome` or the literature mirror: a machine without the tier fails at `resolve`
with the rsync that seeds it. `SENTINEL_ASSEMBLY_SETS` maps the Bloom 2019 BY parent's
stored sentinel string to `sgd_S288C_R64-4-1_20230830` so stored records never change.

Assembly sets deposited 2026-09-14 (`scripts/migrate_genomes_tier.py`,
[[scripts.migrate_genomes_tier]]):

| set | files | provenance |
|---|---|---|
| `sgd_S288C_R64-4-1_20230830` | the SGD tgz (21,142,292 B, sha256 `987e7e32...48de1`) plus its eight extracted members | container re-fetched from the SGD archive 2026-09-14; every member equals the file `data/sgd/genome/` had held, so the retrieval record is genuine |
| `peter2018_1011_assemblies` | `1011Assemblies.tar.gz` (3,999,623,201 B, sha256 `53540d09...a9b4da`), the pangenome ORF FASTA, the reference-genes tarball, the presence and copy-number matrices, and the derived member index | all five re-fetched from `1002genomes.u-strasbg.fr` 2026-09-14 and equal to the library key's copies; the index is recorded as derived from the tarball |

Measured on GilaHyper: `SCerevisiaeGenome(overwrite=False)` constructs in 0.3 s through
the registry against 0.1 s on the legacy paths (`scratchpad/genomes-tier/smoke.py`), the
difference being the sha256 of the four resolved files on every construction; gene set
6,607 and the ORF plus RNA gene universe 7,146 are unchanged.

Open items carried in the plan note: the `ReferenceGenome` assembly pointer waits for the
full-rebuild window; the `overwrite=True` default and a DDP-safe build-when-absent; an NCBI
assembly set for `s288c_ncbi.py` and `s288c_gb.py`; tier sync to Delta, IGB and the Mac;
deprecation of the legacy S288C copy once nothing reads it.

## 2026.10.07 - Bacterial assembly sets deposited

Four ids added beside `SGD_S288C_R64` and `PETER2018_1011`, each the name of a tier
directory deposited the same day by `scripts/provision_bacterial_genomes.py`
([[scripts.provision_bacterial_genomes]]); plan [[plan.bacteria-ontology-genome]] section 1
and decisions D4, D5, D8, D9.

| constant | assembly set | members | what it holds |
|---|---|---|---|
| `ECOLI_K12_MG1655` | `ecoli_K12_MG1655_ASM584v2` | 9 | GCA_000005845.2 (GBFF, FASTA, GFF3, proteins, feature table, assembly report) + GCF GBFF and GFF3 + GO Consortium `ECOLI-uniprot.gaf.gz` |
| `ECOLI_K12_BW25113` | `ecoli_K12_BW25113_ASM75055v1` | 9 | GCA_000750555.1 (same six) + GCF GBFF, GFF3 and NCBI `_gene_ontology.gaf.gz` |
| `PPUTIDA_KT2440` | `pputida_KT2440_ASM756v2` | 10 | GCA_000007565.2 (same six) + GCF GBFF, GFF3 and NCBI GAF + EBI GOA `109.P_putida_KT2440.goa` |
| `GO_RELEASE_20260805` | `go_release_2026-08-05` | 1 | `go-basic.obo` (`data-version: releases/2026-07-26`) |

One set per strain, never per host (D5): a `BW25113_` number is not an MG1655 b-number.
The sequence is deposited once per strain, from GCA, because the GCF FASTA is the same
sequence under a different header. The three strain sets pin the GO ontology by the
`go_release_2026-08-05` id rather than carrying a copy each, and BW25113 references
MG1655's GAF by the `ecoli_K12_MG1655_ASM584v2` id rather than copying it (D9). Both
references are recorded as text in the manifests' `notes`; `GenomeManifest` has no
structured cross-set field.

New role constant `ROLE_ONTOLOGY = "ontology"` for `go-basic.obo`, which is an ontology
release and not a gene annotation. `ROLE_INDEX`'s comment now also covers the NCBI feature
table and assembly report, which the plan assigns that role.

Measured on GilaHyper: all 29 members re-fetched and equal to the plan's Verified digests
(no upstream drift), all 26 NCBI members equal to NCBI's published md5, and
`verify_assembly_set` returns 9 / 9 / 10 / 1 digests. Tests:
`tests/torchcell/sequence/genome/test_registry.py` pins the four ids and, on a tmp tier per
id, `resolve` and `verify_assembly_set`, a `GenomeIntegrityError` naming both digests on a
corrupted copy, and the exact rsync hint for an absent set (27 passed).

## 2026.10.09 - `ECOLI_K12_W3110` added, deposited but not yet readable

`ECOLI_K12_W3110 = "ecoli_K12_W3110_ASM1024v1"` joins the five bacterial ids, for row 41
of the bacterial schedule (Teteneva 2024, host W3110). GCA_000010245.1 /
GCF_000010245.2, replicon AP009048.1 / NC_007779.1, 4,646,332 bp, ten members deposited
and `verify_assembly_set` returning all ten.

The constant's docstring carries the one thing a caller must know before resolving the
set: its GenBank member has 4,444 gene features and no `locus_tag` at all, so the
GenBank-first ingest refuses it and no genome class reads this set yet. The id is in NO
schema vocabulary (`BacterialAssemblySet`, `BACTERIAL_ASSEMBLY_SETS` and
`ASSEMBLY_SET_ACCESSIONS` are unchanged), so `schema_impact_check` reports no contract
change and nothing served is staled. Measurements and the open route decision:
[[torchcell.datasets.ecoli.teteneva2024]].

`tests/torchcell/sequence/genome/test_registry.py` now pins the fifth bacterial id and
runs the tmp-tier `resolve` / `verify_assembly_set` / corruption / absent-set parametrization
over it as well (30 passed).
