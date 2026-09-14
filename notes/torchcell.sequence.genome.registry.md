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
