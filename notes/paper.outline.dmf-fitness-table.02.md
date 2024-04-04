---
id: ko5841vxc8kq4rw9xq0z1cf
title: '02'
desc: ''
updated: 1710710998014
created: 1709585393267
---

```mermaid
graph LR
    Deep_Set --> Models
    Self_Attention-Deep_Set --> Models
    Node_Features --> Self_Attention-Deep_Set
    Node_Features --> Self_Attention-DiffPool
    Node_Features --> Self_Attention-SAG
    Self_Attention-DiffPool --> Models
    Self_Attention-SAG --> Models
    SAG --> Models
    DiffPool --> Models
    One_Hot_Gene_Embedding --> Node_Features
    Codon_Frequency --> Node_Features
    ESM2 --> Node_Features
    CaLM_Embedding --> Node_Features
    FUDT_Embedding --> Node_Features
    NT_Embedding --> Node_Features
    ProtT5_Embedding --> Node_Features
    Node_Features --> Deep_Set
    Node_Features --> SAG
    Node_Features --> DiffPool
    Edge_Features --> SAG
    Edge_Features --> DiffPool
    PPI --> Edge_Features 
    Reg --> Edge_Features 
    GO --o Node_Features
    GO --x Edge_Features
    GO --> DCell
    One_Hot_Gene_Embedding --> DCell
    DCell --> Models
    Models --> Fitness
    Models --> Interactions
    Interactions --> metrics
    Fitness --> metrics 
    metrics --> mae
    metrics --> mdae
    metrics --> pearson
    metrics --> spearman
    metrics --> r_squared
    metrics --> μ_important
    metrics --> σ_important
    DCell -.-> Interpretable((Interpretable))
    SAG -.-> Interpretable((Interpretable))
    DiffPool -.-> Interpretable((Interpretable))
    Self_Attention-DiffPool -.-> Interpretable((Interpretable))
    Self_Attention-SAG -.-> Interpretable((Interpretable))
    Self_Attention-Deep_Set -.-> Interpretable((Interpretable))
```

  | id | model                      | nodes features $(\mathcal{N})$    | edge features $(\mathcal{E})$ | mae | mdae | pearson | spearman | $r^2$ | $\mu$(1.0-1.1) | $\sigma$(1.0-1.1) | $\mu$(1.1-1.2) | $\sigma$(1.1-1.2) |
  |:---|:---------------------------|:----------------------------------|:------------------------------|:----|:-----|:--------|:---------|:------|:---------------|:------------------|:---------------|:------------------|
  |    | DCell                      | One Hot Genes                     | -                             |     |      |         |          |       |                |                   |                |                   |
  |    | Set Net                    | One Hot Genes                     | -                             |     |      |         |          |       |                |                   |                |                   |
  |    | Set Net                    | Codon Frequency                   | -                             |     |      |         |          |       |                |                   |                |                   |
  |    | Set Net                    | CALM                              | -                             |     |      |         |          |       |                |                   |                |                   |
  |    | Set Net                    | Fungal-UTR-Transformer Embeddings | -                             |     |      |         |          |       |                |                   |                |                   |
  |    | Set Net                    | Nucleotide-Transformer Embeddings | -                             |     |      |         |          |       |                |                   |                |                   |
  |    | Set Net                    | Prot T5                           | -                             |     |      |         |          |       |                |                   |                |                   |
  |    | Set Transformer- SAG       | One Hot Genes                     | -                             |     |      |         |          |       |                |                   |                |                   |
  |    | Set Transformer - SAG      | Codon Frequency                   | -                             |     |      |         |          |       |                |                   |                |                   |
  |    | Set Transformer - SAG      | CALM                              | -                             |     |      |         |          |       |                |                   |                |                   |
  |    | Set Transformer - SAG      | Fungal-UTR-Transformer Embeddings | -                             |     |      |         |          |       |                |                   |                |                   |
  |    | Set Transformer - SAG      | Nucleotide-Transformer Embeddings | -                             |     |      |         |          |       |                |                   |                |                   |
  |    | Set Transformer - SAG      | Prot T5                           | -                             |     |      |         |          |       |                |                   |                |                   |
  |    | Set Transformer - DiffPool | One Hot Genes                     | -                             |     |      |         |          |       |                |                   |                |                   |
  |    | Set Transformer - DiffPool | Codon Frequency                   | -                             |     |      |         |          |       |                |                   |                |                   |
  |    | Set Transformer - DiffPool | CALM                              | -                             |     |      |         |          |       |                |                   |                |                   |
  |    | Set Transformer - DiffPool | Fungal-UTR-Transformer Embeddings | -                             |     |      |         |          |       |                |                   |                |                   |
  |    | Set Transformer - DiffPool | Nucleotide-Transformer Embeddings | -                             |     |      |         |          |       |                |                   |                |                   |
  |    | Set Transformer - DiffPool | Prot T5                           | -                             |     |      |         |          |       |                |                   |                |                   |
  |    | SAG                        | One Hot Genes                     | PPI                           |     |      |         |          |       |                |                   |                |                   |
  |    | SAG                        | Codon Frequency                   | PPI                           |     |      |         |          |       |                |                   |                |                   |
  |    | SAG                        | CALM                              | PPI                           |     |      |         |          |       |                |                   |                |                   |
  |    | SAG                        | Fungal-UTR-Transformer Embeddings | PPI                           |     |      |         |          |       |                |                   |                |                   |
  |    | SAG                        | Nucleotide-Transformer Embeddings | PPI                           |     |      |         |          |       |                |                   |                |                   |
  |    | SAG                        | Prot T5                           | PPI                           |     |      |         |          |       |                |                   |                |                   |
  |    | DiffPool                   | One Hot Genes                     | PPI                           |     |      |         |          |       |                |                   |                |                   |
  |    | DiffPool                   | Codon Frequency                   | PPI                           |     |      |         |          |       |                |                   |                |                   |
  |    | DiffPool                   | CALM                              | PPI                           |     |      |         |          |       |                |                   |                |                   |
  |    | DiffPool                   | Fungal-UTR-Transformer Embeddings | PPI                           |     |      |         |          |       |                |                   |                |                   |
  |    | DiffPool                   | Nucleotide-Transformer Embeddings | PPI                           |     |      |         |          |       |                |                   |                |                   |
  |    | DiffPool                   | Prot T5                           | PPI                           |     |      |         |          |       |                |                   |                |                   |
  |    | SAG                        | One Hot Genes                     | Reg                           |     |      |         |          |       |                |                   |                |                   |
  |    | SAG                        | Codon Frequency                   | Reg                           |     |      |         |          |       |                |                   |                |                   |
  |    | SAG                        | CALM                              | Reg                           |     |      |         |          |       |                |                   |                |                   |
  |    | SAG                        | Fungal-UTR-Transformer Embeddings | Reg                           |     |      |         |          |       |                |                   |                |                   |
  |    | SAG                        | Nucleotide-Transformer Embeddings | Reg                           |     |      |         |          |       |                |                   |                |                   |
  |    | SAG                        | Prot T5                           | Reg                           |     |      |         |          |       |                |                   |                |                   |
  |    | DiffPool                   | One Hot Genes                     | Reg                           |     |      |         |          |       |                |                   |                |                   |
  |    | DiffPool                   | Codon Frequency                   | Reg                           |     |      |         |          |       |                |                   |                |                   |
  |    | DiffPool                   | CALM                              | Reg                           |     |      |         |          |       |                |                   |                |                   |
  |    | DiffPool                   | Fungal-UTR-Transformer Embeddings | Reg                           |     |      |         |          |       |                |                   |                |                   |
  |    | DiffPool                   | Nucleotide-Transformer Embeddings | Reg                           |     |      |         |          |       |                |                   |                |                   |
  |    | DiffPool                   | Prot T5                           | Reg                           |     |      |         |          |       |                |                   |                |                   |
