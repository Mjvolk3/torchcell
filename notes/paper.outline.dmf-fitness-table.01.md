---
id: jyuzqq2npnj94gqmx1rtbw3
title: '01'
desc: ''
updated: 1708523963718
created: 1708523953078
---
| model               | nodes features $(\mathcal{N})$    | edge features $(\mathcal{E})$ | mae | mdae | pearson | spearman | $r^2$ | $\mu$(1.0-1.1) | $\sigma$(1.0-1.1) | $\mu$(1.1-1.2) | $\sigma$(1.1-1.2) |
|:--------------------|:----------------------------------|:------------------------------|:----|:-----|:--------|:---------|:------|:---------------|:------------------|:---------------|:------------------|
| DCell               | One Hot Genes                     | -                             |     |      |         |          |       |                |                   |                |                   |
| Set Net             | One Hot Genes                     | -                             |     |      |         |          |       |                |                   |                |                   |
| Set Net             | Codon Frequency                   | -                             |     |      |         |          |       |                |                   |                |                   |
| Set Net             | Fungal-UTR-Transformer Embeddings | -                             |     |      |         |          |       |                |                   |                |                   |
| Set Net             | Nucleotide-Transformer Embeddings | -                             |     |      |         |          |       |                |                   |                |                   |
| Set Net             | Prot T5                           | -                             |     |      |         |          |       |                |                   |                |                   |
| Self-Attention Pool | One Hot Genes                     | PPI                           |     |      |         |          |       |                |                   |                |                   |
| Self-Attention Pool | Codon Frequency                   | PPI                           |     |      |         |          |       |                |                   |                |                   |
| Self-Attention Pool | Fungal-UTR-Transformer Embeddings | PPI                           |     |      |         |          |       |                |                   |                |                   |
| Self-Attention Pool | Nucleotide-Transformer Embeddings | PPI                           |     |      |         |          |       |                |                   |                |                   |
| Self-Attention Pool | Prot T5                           | PPI                           |     |      |         |          |       |                |                   |                |                   |
| DiffPool            | One Hot Genes                     | PPI                           |     |      |         |          |       |                |                   |                |                   |
| DiffPool            | Codon Frequency                   | PPI                           |     |      |         |          |       |                |                   |                |                   |
| DiffPool            | Fungal-UTR-Transformer Embeddings | PPI                           |     |      |         |          |       |                |                   |                |                   |
| DiffPool            | Nucleotide-Transformer Embeddings | PPI                           |     |      |         |          |       |                |                   |                |                   |
| DiffPool            | Prot T5                           | PPI                           |     |      |         |          |       |                |                   |                |                   |
