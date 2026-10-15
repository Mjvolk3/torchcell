---
id: pmzqwqaq4nm3nqehev4cppk
title: Measure_gene_id_namespaces
desc: ''
updated: 1792055888711
created: 1792055888711
---

## 2026.10.15 - The gene identifier namespaces the served stores carry (P1, ontology-unify)

`experiments/database/scripts/measure_gene_id_namespaces.py` opens every served dev store of release `2026.10.06-4b293d34` read-only (51 stores, `stream_records`), classifies every stored `systematic_gene_name` by form, and measures the locus-tag forms of the two deposited yeast assembly sets. It is the measurement behind `GeneIdNamespace` and `GENE_ID_NAMESPACE_PATTERNS` in `torchcell/datamodels/schema.py` (constitution principle 5: the vocabulary is measured from the stored records, not assumed) and is cited by the schema-change record `database/schema_changes/2026-10-15-gene-id-namespace-on-gene-perturbation.json`.

Results: `experiments/database/results/gene_id_namespace_vocabulary.json` and `.md` (one row per dataset, perturbation type and identifier form, with examples; the reference strain per store; the yeast locus-tag measurement).

What it found, 130,685,237 identifiers over 51 stores:

| form | identifiers | where |
|---|---|---|
| SGD S288C systematic (`YAL001C`, `Q0010`, `YNCA0001W`) | 130,537,788 | every pinned yeast leaf in all 51 stores |
| Peter 2018 pangenome column id (`13-EC1118_1N26_0012g`) | 79,016 | `CaudalPanTranscriptome2024Dataset`, `natural_gene_presence` |
| non-host identifiers on relaxed leaves | 68,433 | released symbols on `released_gene_name` (Cachera, Mota, Auesukaree, Xue, Yoshida, Mormino, both SynLethDB), heterologous symbols and construct names on `gene_addition` (Cachera, Ozaydin, Mormino), standard names on Caudal's `natural_loss_of_function` |
| bacterial locus tags, CEN.PK tags, CURIEs | 0 | none served; the bacterial spaces and CEN.PK113-7D enter the vocabulary from the genomes registry |

Yeast locus-tag forms from the tier: all 6,171 locus tags of the deposited CEN.PK113-7D GenBank (`GCA_002571405.2`) match `^SCEN_(?:[A-P]\d{5}|[QR]\d+|[A-R]trna\d{1,2}[A-Z]?)$` (5,865 chromosomal, 21 mitochondrial `SCEN_Q`/`SCEN_R`, 285 tRNA); 7,147 of the 7,148 gene-class features of the SGD R64-4-1 GFF match the S288C pattern (the exception, `YNCI0012`, carries no strand letter and was never stored).

Every served reference strain is S288C-derived (BY4741, BY4742, BY4743, FY4, S288C), which is the evidence for the P1 default rule: a leaf whose name validator pins the S288C form defaults `gene_namespace` to S288C; a relaxed leaf states it per record.
