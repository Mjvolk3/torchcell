---
id: rwnl9xgfktcvc2zw8b0i3iq
title: Sd108
desc: ''
updated: 1792445983830
created: 1792445983830
---

## 2026.10.19 - SD108 v1 genome class

`PKudriavzeviiSD108Genome` reads the genomes-tier set `pkudriavzevii_SD108_v1`
([[scripts.provision_pkudriavzevii_genome]]): the group's curated *Pichia kudriavzevii*
(*Issatchenkia orientalis*) SD108 assembly, one haploid copy of the diploid strain. Seven
sequences, 10.87 Mb: `SD108_chr1`..`SD108_chr5` (keys 1-5), the circular `SD108_mito`
(key 0, the SGD convention) and the unplaced `SD108_rDNA_unit` (key 6, one past the five
chromosomes). Lengths are checked against the deposited `.fai` at construction.

### Route

GenBank first, then GFF3 + FASTA is the repository stance; this release ships no GenBank
flat file, so it is read the SGD way: the GFF3 into `data.db` through gffutils, the genome
FASTA, and the release's `proteins.faa` and `cds.fna`. The `data.db` cache and pickling
contract are `AnnotatedGenome`'s ([[torchcell.sequence.genome.base]]).

### Identifier rules

Gene ids are the assembly's own, `SD108_X<seq><type><number>`
(`SD108_GENE_ID_PATTERN = SD108_X[1-5MU][GTR]\d{4}[A-Z]?`): `<seq>` is 1-5, `M` or `U`;
`<type>` is `G` protein-coding, `T` tRNA, `R` rRNA. The optional letter is the
nomenclature's rule for a gene inserted in a later version; v1 has none. Measured: the
pattern matches all 5,376 gene ids of the GFF3 and none of its mRNA, tRNA, rRNA, exon or
CDS ids. Haplotype-2 alleles (`_H2`) are not read by this class.

*S. cerevisiae* names are a crosswalk, never an identity (the rules of `CenPk1137DGenome`
apply verbatim): `gene.sc_crosswalk` is a read-only `ScerevisiaeNameCrosswalk` from
`gene_xref.tsv` (`sc_gene`, `sc_systematic`, `sc_name_check`, identity, coverages,
e-value, `sc_best_hit`), and `genome["URA3"]` is None. 3,471 genes are named: 3,193
`confirmed`, 233 `best_hit_weak`, 45 `paralog`. The GFF3's `sc_gene` must equal
`gene_xref.tsv`'s; three names carry a literal comma (`ADE5,7`, `ARG5,6`, `DUR1,2`), which
GFF3 reserves as a value separator, so they are rejoined before the comparison.

### Coordinates

3,069 of the 5,159 protein-coding gene rows span UTRs beyond the CDS. As in the SGD
reader, a protein-coding gene's `start`/`end`/`seq` are its coding span (first to last
CDS base), so windows sit on the start and stop codons; the gene row is kept as
`locus_start`/`locus_end`. Measured: for all 4,919 single-CDS genes, `gene.seq` equals the
`cds.fna` record exactly. tRNA and rRNA genes use their gene row.

### Proteins vs genes

5,376 gene rows = 5,159 protein-coding (5,144 nuclear + 15 mitochondrial) + 199 tRNA + 18
rRNA. `proteins.faa` and `cds.fna` each hold 5,159 records, keyed `<gene>.t1`, exactly one
per protein-coding gene (the mitochondrial ones translated with code 3). No record maps to
a non-coding gene and no coding gene lacks one; the 217-record difference from the gene
count is the tRNA and rRNA genes. Construction refuses any departure from this.

### GO

Stated absent: `go_annotations == {}`, `gene.go is None`, and `genome.go_source` is a
`GoAbsence` record naming the set, the members inspected (`SD108_genome_v1.gff3`,
`gene_xref.tsv`) and the reason (no GO release is keyed to SD108 ids; S. cerevisiae GO is
not borrowed through `sc_gene`).

### Construct

```python
from torchcell.sequence.genome.pkudriavzevii import PKudriavzeviiSD108Genome

genome = PKudriavzeviiSD108Genome(
    genome_root=osp.join(DATA_ROOT, "data/pkudriavzevii/sd108/genome")
)
ura3 = genome["SD108_X2G1933"]  # chr2:1,175,606-1,176,394 (-), sc_gene URA3
```

`overwrite=False` is the default; a missing `genome_root` needs `overwrite=True` once.

### Not yet a record-pinnable assembly

`pkudriavzevii_SD108_v1` is in no schema vocabulary: it has no NCBI accession
(`ASSEMBLY_SET_ACCESSIONS` requires a `GCA_` one) and no `GeneIdNamespace` member, so
`tests/torchcell/datamodels/test_schema.py` excludes `PKUDRIAVZEVII_SD108` from its
registry-equals-vocabulary check with that reason. Admitting it to `AssemblySet` is a
schema change owed by the first loader that writes records against SD108. It is listed in
`torchcell.candidates.gates.READABLE_ASSEMBLY_SETS`.

Tests: `tests/torchcell/sequence/genome/pkudriavzevii/test_sd108.py` (26 hermetic, 10
data-gated with `--data`).
