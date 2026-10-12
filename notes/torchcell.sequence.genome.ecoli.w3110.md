---
id: b0669pwyxu1uhvpib76tta9
title: W3110
desc: ''
updated: 1791818003976
created: 1791818003976
---

## 2026.10.12 - W3110 read RefSeq first (#910)

`EcoliK12W3110Genome` (`torchcell/sequence/genome/ecoli/w3110.py`) reads the genomes-tier
set `ecoli_K12_W3110_ASM1024v1` through a `RefSeqPrimaryAssembly`
(`torchcell/sequence/genome/bacterial.py`), because the GenBank member
(GCA_000010245.1, the 2006 DDBJ/NIG annotation) carries 4,444 gene features and no
`locus_tag`, which `read_genbank` refuses by name.

| member | role |
|---|---|
| `GCF_000010245.2_ASM1024v1_genomic.gbff.gz` | the locus universe (4,531 `Y75_RS` loci: 4,346 genes, 185 pseudogenes) and the proteins (`/translation`) |
| `GCF_000010245.2_ASM1024v1_genomic.gff.gz` | `data.db`, and GO: the `Ontology_term` rows keyed by their own locus tag (route `refseq_gff_ontology_term_primary`; 2,298 rows, 2,201 genes) |
| `GCA_000010245.1_ASM1024v1_genomic.fna.gz` | the DNA, record AP009048.1, refused unless byte-identical to NC_007779.1 of the flat file, then keyed `NC_007779.1` |
| `GCF_000010245.2_ASM1024v1_assembly_report.txt` | the accession pair (`read_assembly_report`); the GCA report still names GCF_000010245.1 |

`assembly_reference("W3110")` names the RefSeq accession GCF_000010245.2 (the records'
reference sequence is the one the authors mapped to). The verification runners read
the RefSeq loci as the W3110 gene universe (`_refseq_primary_gene_set`: the GFF gene
and pseudogene tags, refused unless equal to the flat file's). Counts measured on the
tier by `tests/torchcell/sequence/genome/ecoli/test_w3110.py` (`--data`).
