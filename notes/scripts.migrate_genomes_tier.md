---
id: lbiz9pgxdj2fjodvwh9p8lg
title: Migrate_genomes_tier
desc: ''
updated: 1789364454196
created: 1789364454196
---

## 2026.09.14 - First deposit: SGD S288C R64-4-1 and Peter 2018

Run once on GilaHyper with `--refetch-dir` pointing at the bytes fetched earlier in the
session (S288C tgz from the SGD archive; the five Peter files from the Strasbourg server).
Nothing was moved or deleted: the legacy `data/sgd/genome/` release directory and the
library key's `data/` are untouched and no symlink was left anywhere. Printed output:

```
(a) S288C_reference_genome_R64-4-1_20230830.tgz: 21142292 bytes, sha256 987e7e324dee8e97368a27a34aa351537a4fa9342a43eaec82f2993486c48de1
member                                         on-disk sha256  fetched sha256  result
NotFeature_R64-4-1_20230830.fasta              8fdf73f0bc73    8fdf73f0bc73    MATCH 
S288C_reference_sequence_R64-4-1_20230830.fsa  dbf065ffc3f5    dbf065ffc3f5    MATCH 
gene_association_R64-4-1_20230830.sgd          69c2009f15c4    69c2009f15c4    MATCH 
orf_coding_all_R64-4-1_20230830.fasta          ebed8e5714da    ebed8e5714da    MATCH 
orf_trans_all_R64-4-1_20230830.fasta           9438eb58e029    9438eb58e029    MATCH 
other_features_genomic_R64-4-1_20230830.fasta  56d40840e85e    56d40840e85e    MATCH 
rna_coding_R64-4-1_20230830.fasta              32ca65953212    32ca65953212    MATCH 
saccharomyces_cerevisiae_R64-4-1_20230830.gff  64f61e315308    64f61e315308    MATCH 
file                                               library sha256  fetched sha256  result
1011Assemblies.tar.gz                              53540d095958    53540d095958    MATCH 
allORFs_pangenome.fasta.gz                         22417aa8c9f8    22417aa8c9f8    MATCH 
allReferenceGenesWithSNPsAndIndelsInferred.tar.gz  b5400b89499f    b5400b89499f    MATCH 
genesMatrix_PresenceAbsence.tab.gz                 63b62d6761c3    63b62d6761c3    MATCH 
genesMatrix_CopyNumber.tab.gz                      d22fe0a20249    d22fe0a20249    MATCH 
(c) copying into the tier
file                                           bytes     nlink owner    sha256 vs source
S288C_reference_genome_R64-4-1_20230830.tgz    21142292  1 michaelvolk  SAME            
gene_association_R64-4-1_20230830.sgd          43685870  1 michaelvolk  SAME            
NotFeature_R64-4-1_20230830.fasta              3595201   1 michaelvolk  SAME            
orf_coding_all_R64-4-1_20230830.fasta          11651150  1 michaelvolk  SAME            
orf_trans_all_R64-4-1_20230830.fasta           5499107   1 michaelvolk  SAME            
other_features_genomic_R64-4-1_20230830.fasta  819179    1 michaelvolk  SAME            
rna_coding_R64-4-1_20230830.fasta              205549    1 michaelvolk  SAME            
S288C_reference_sequence_R64-4-1_20230830.fsa  12361395  1 michaelvolk  SAME            
saccharomyces_cerevisiae_R64-4-1_20230830.gff  20086638  1 michaelvolk  SAME            
file                                               bytes       nlink owner    sha256 vs source
1011Assemblies.tar.gz                              3999623201  1 michaelvolk  SAME            
allORFs_pangenome.fasta.gz                         3290188     1 michaelvolk  SAME            
allReferenceGenesWithSNPsAndIndelsInferred.tar.gz  160823369   1 michaelvolk  SAME            
genesMatrix_PresenceAbsence.tab.gz                 416189      1 michaelvolk  SAME            
genesMatrix_CopyNumber.tab.gz                      1002976     1 michaelvolk  SAME            
1011Assemblies.tar.gz.member_index.tsv             34150       1 michaelvolk  SAME            
(d) depositing manifests
/scratch/projects/torchcell-scratch/torchcell-genomes/sgd_S288C_R64-4-1_20230830/manifest.json
/scratch/projects/torchcell-scratch/torchcell-genomes/peter2018_1011_assemblies/manifest.json
verified sgd_S288C_R64-4-1_20230830: 9 files
verified peter2018_1011_assemblies: 6 files
```

Both manifests carry `provenance_complete: true` because every file reproduced from its
source URL. The weekly backup (`scripts/backup_mirrors_to_bulk.sh`) copied the tier to
`/bulk/torchcell-genomes/` the same day: 17 files, 4,284,249,415 bytes, the tarball's
sha256 verified on the copy, zero symlinks. Registry: [[torchcell.sequence.genome.registry]].
