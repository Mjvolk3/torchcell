---
id: fyfihty9yhhgk8uwyz96eox
title: Provision_pkudriavzevii_genome
desc: ''
updated: 1792445975640
created: 1792445975640
---

## 2026.10.19 - First deposit of `pkudriavzevii_SD108_v1`

`scripts/provision_pkudriavzevii_genome.py` deposits the group's curated *Pichia
kudriavzevii* (*Issatchenkia orientalis*) SD108 genome v1 into the genomes tier as
`pkudriavzevii_SD108_v1` (`registry.PKUDRIAVZEVII_SD108`). Source: the private GitHub
repository `Aurosish/I.orientalis_SD108_Genome_Reconstruction` at commit
`8ca6a106809d6b708d4ba621e3834649dd122720`. Read by
[[torchcell.sequence.genome.pkudriavzevii.sd108]].

### Retrieval

Every member carries a `RetrievalRecord` with `method=git_repository` (new member, last in
`RetrievalMethod`), `retriever=torchcell.literature.retrieve.git_repository` and
`params={repo, commit, path_in_repo}`. The retriever runs:

```bash
gh repo clone Aurosish/I.orientalis_SD108_Genome_Reconstruction <dir> -- --depth 1
git -C <dir> fetch --depth 1 origin 8ca6a106809d6b708d4ba621e3834649dd122720
git -C <dir> checkout --detach 8ca6a106809d6b708d4ba621e3834649dd122720
git -C <dir> show 8ca6a106809d6b708d4ba621e3834649dd122720:<path_in_repo>
```

The repository is private, so the clone needs the owner's `gh` authentication; without it
the deposited bytes are the rebuild source. Re-running the recorded retriever for
`NOMENCLATURE.md` on 2026-10-19 reproduced the pinned sha256, so `provenance_complete=True`.

### Integrity

Two anchors, both checked before the deposit: `md5sum -c` of the repository's own
`Assembly_Build/final/MD5SUMS` (19 files, all OK) plus the md5 of every member that listing
names; and every member's git blob id against the pinned commit's tree (covers the members
`MD5SUMS` does not list: `MD5SUMS` itself, the JL09 map and the haplotype-2 files).

### Deposited members (15)

Flat layout, every member under its repository basename; no haplotype-2 basename collides
with a primary one, so none is renamed.

| member | role | bytes | sha256 | path in repository |
|---|---|--:|---|---|
| `SD108_genome_v1.fasta` | sequence | 11,004,873 | `adb8dd254e3adb6167c40fe4be77c98db4f4a479cb3aa5ad10414e3b9cbf8a4d` | `Assembly_Build/final/` |
| `SD108_genome_v1.fasta.fai` | index | 229 | `4478111f87317aef72c82c75843d47a3580877cb200e36107c47c2a5c3a61220` | `Assembly_Build/final/` |
| `SD108_genome_v1.sources.tsv` | index | 2,153 | `80536686fffbe6492792d019db470a7f8f34a797703bb1b4850c80bc889bb781` | `Assembly_Build/final/` |
| `SD108_genome_v1.gff3` | annotation | 4,883,873 | `1c3c658dafd15cbb79924f91ec0bdb58c06987190961643fc5f39de53e51428c` | `Assembly_Build/final/annotation/` |
| `proteins.faa` | sequence | 2,772,771 | `8c3e0d460416fae377858abe2f72300005bed570b49edfac6b06a74bceddbbae` | `Assembly_Build/final/annotation/` |
| `cds.fna` | sequence | 8,049,091 | `5b2c9f87a166f56e3ec574796beb3676a0873825af006357857b8f26acfa927b` | `Assembly_Build/final/annotation/` |
| `transcripts.fna` | sequence | 10,908,976 | `73fd70f472286586e46a84e2c230472a297a76cf42d17d8b2755e1573ab47030` | `Assembly_Build/final/annotation/` |
| `gene_xref.tsv` | annotation | 2,000,621 | `36f85346c0867b7fa7284ac57cf00d0f699789265d0bcfdbebca4fe85bb3924b` | `Assembly_Build/final/annotation/` |
| `NOMENCLATURE.md` | annotation | 7,132 | `5318e083d67c69a56aec8790f069d00c4c821052ef446fba88ea99d2ab63512c` | `Assembly_Build/final/annotation/` |
| `MD5SUMS` | index | 1,251 | `4cb25a3a8728ceaffe143348acc366aa1e9b23b5efbd9f761672f1232fea1c0e` | `Assembly_Build/final/` |
| `SD108_to_NCBI_JL09_map.csv` | annotation | 727,770 | `0eac36653a5fb831a35f1c6fcdb4ace706f037e25550b81edac3f2846d76ac51` | `ID_mappings/` |
| `SD108_genome_v1_hap2.fasta` | sequence | 10,921,468 | `61ab16b42214ec23fa909388860910dfe5694fde9a61ac77f3e05956da1f790a` | `Haplotype2_Build/final/` |
| `SD108_genome_v1_hap2.gff3` | annotation | 5,381,262 | `4d56682afd4a0fe654249b530d13e6a2354056c73b9a018f384342fb4e69dd3b` | `Haplotype2_Build/final/annotation_hap2/` |
| `proteins_hap2.faa` | sequence | 2,807,132 | `30d196f356efe4392b59467ab3515e708b17bd124a41c1a40250f764be602823` | `Haplotype2_Build/final/annotation_hap2/` |
| `cds_hap2.fna` | sequence | 7,904,306 | `6ba2f47554121c83b5d0036c5ce42f270417c69db164eea1308bbe5c84f20bf8` | `Haplotype2_Build/final/annotation_hap2/` |

Not deposited (the set holds what the genome class and the haplotype-2 copy need): the GTF,
the heterozygosity VCF/BED tracks, the polishing tracks, the phased haplotype-2 VCF and
`proteins_mosaic_real_copies.faa`. A later revision can add any of them with
`registry.extend_assembly_set`.

### Run

```bash
PYTHONPATH=$WT python scripts/provision_pkudriavzevii_genome.py \
    --refetch-dir /scratch/projects/torchcell-scratch/tmp/044-inputs/refetch \
    --data-root /scratch/projects/torchcell-scratch
```

Deposited and verified on GilaHyper at
`/scratch/projects/torchcell-scratch/torchcell-genomes/pkudriavzevii_SD108_v1/`.
