---
id: xr5jpql93ejlm8c632konq7
title: Data Flow
desc: ''
updated: 1792450005669
created: 1792450005669
---

## 2026.10.19 - From the raw store to the batch the model consumes

Top to bottom: the 043 raw store, the four genotype classes and how each becomes a marked
gene set over the shared index of $|G| = 6{,}607$ genes, the per-dataset label tensors
scattered onto that index with a measured-gene mask, the four tokens, and the batch. Every
number is from [[experiments.043-expression-proteome-morphology.scripts.data_shapes_043]]
(the full pass over 173,617 records, slurm job 5902; per-record ranges are min to max, the
segregant marking counts are from the three-records-per-dataset debugger run of
[[experiments.043-expression-proteome-morphology.scripts.dataset_043]]). The marking rules
are the definite ones of the dataset module: a deletion marks its gene; an isolate marks
every gene with a sequence variant, a predicted loss of function or a natural absence; a
segregant marks a gene when its covering block calls the non-reference parent with posterior
at least 0.9. Colors follow the worked example in
[[torchcell.models.equivariant_cell_graph_transformer.mermaid.type-i-ii]]: purple = the
store and the batch, orange = genotype to marked genes, red = labels, yellow = tokens.

```mermaid
%%{init: {'theme':'base','themeVariables':{'background':'#F5EEDD','clusterBkg':'#F5EEDD','clusterBorder':'#E0D6BE','lineColor':'#B7AC93','fontFamily':'Arial'}, 'flowchart': {'nodeSpacing': 14, 'rankSpacing': 30, 'subGraphTitleMargin': {'top': 8, 'bottom': 14}}}}%%
graph TD
  Store["$$\begin{gathered}\text{043 raw store: LMDB, keys } \text{data\_}i,\ i \in \{0, \ldots, N-1\}\\\ 15\ \text{datasets},\ N = 173{,}617\ \text{records}\\\ \text{record} = (\text{experiment},\ \text{experiment\_reference})\\\ \text{experiment} = (\text{genotype},\ \text{environment},\ \text{phenotype})\end{gathered}$$"]

  subgraph Genotypes["$$\text{Genotype} \to \text{marked gene set over } G,\ |G| = 6{,}607$$"]
    direction LR
    Del["$$\begin{gathered}\text{Deletion set } S \subset G,\ |S| = 1\\\ \tau = \mathrm{del}\\\ \text{Kemmeren } 1{,}482 \quad \text{Hu } 262\\\ \text{Jackson } 784 \quad \text{Nadal-Ribelles } 6{,}118\\\ \text{Messner } 4{,}624 \quad \text{Zelezniak } 97\\\ \text{(Ohya } 4{,}678\text{, hold-out)}\end{gathered}$$"]
    Dbl["$$\begin{gathered}\text{Double deletion } |S| = 2\\\ \tau = (\mathrm{del}, \mathrm{del})\\\ \text{Sameith Dm } 72\\\ \text{(Sameith Sm } 82 = \text{Kemmeren arrays, dropped)}\end{gathered}$$"]
    Iso["$$\begin{gathered}\text{Natural isolate (Peter 2018 panel)}\\\ \text{Caudal } 943 \quad \text{Muenzner } 795 + 55\\\ 0 \text{ to } 7{,}025\ \text{perturbations per record (median } 5{,}650)\\\ a \in \{0,1\}^{|G|}:\ \text{variant } 53 \text{ to } 5{,}840,\ \text{lof } 12 \text{ to } 940,\ \text{absent } 43 \text{ to } 453\\\ \text{accessory presence } 34 \text{ to } 352\ \text{pangenome ORFs, no node}\\\ \text{strain token} = \text{isolate name}\end{gathered}$$"]
    Seg["$$\begin{gathered}\text{Segregant: haplotype blocks } b = (c, s, e, p, \pi)\\\ \text{Boocock } 79{,}516\ (\text{3 crosses}) \quad \text{N'Guessan } 2{,}951\ (\text{BY} \times \text{RM})\\\ 46 \text{ to } 2{,}225\ \text{blocks per record},\ \pi \in \{0.5, 0.55, \ldots, 1\}\\\ a_g = \mathbb{1}\big[p_{b(g)} \neq \text{ref} \wedge \pi_{b(g)} \geq 0.9\big]\\\ \text{marked } 1{,}415 \text{ to } 3{,}158\ \text{(6 records)},\ \text{gap or low } \pi:\ 854 \text{ to } 3{,}397\\\ \text{parent token} \in \{\text{BY}, \text{RM}, \text{YJM145}, \text{YPS163}, \text{CBS2888}, \text{YJM981}\}\end{gathered}$$"]
    Prom["$$\begin{gathered}\text{Promoter variant (Vaishnav)}\\\ 71{,}158\ \text{records}\\\ p \in \Sigma^{110},\ \Sigma = \{A, C, G, T\}\\\ \text{variable 80-mer at } 18..97\\\ \text{scaffold pTpA, reporter YFP (no node)}\\\ S = \varnothing\end{gathered}$$"]
  end

  subgraph Labels["$$\text{Phenotype} \to y \in \mathbb{R}^{n_d} \text{ on } G,\ \text{mask } m \in \{0,1\}^{|G|}$$"]
    direction LR
    L2["$$\begin{gathered}\log_2 \text{ ratio to wild type}\\\ \text{Kemmeren, Sameith Dm: } n_d = 6{,}169\\\ (6{,}127 \in G)\\\ \text{Nadal-Ribelles: } 4{,}223 \text{ to } 6{,}313\\\ \text{standard name} \to \text{ORF}:\ 4{,}034 \text{ to } 6{,}016 \in G\end{gathered}$$"]
    Sc["$$\begin{gathered}\text{Error-model score (Hu)}\\\ n_d = 5{,}921 \text{ to } 6{,}429\\\ 5{,}390 \text{ to } 5{,}878 \in G\end{gathered}$$"]
    Umi["$$\begin{gathered}\text{UMI counts (zeros omitted)}\\\ \text{Jackson: } 298 \text{ to } 5{,}561\\\ \text{Boocock: } 182 \text{ to } 4{,}758\\\ \text{N'Guessan: } 475 \text{ to } 5{,}111\end{gathered}$$"]
    Tpm["$$\begin{gathered}\text{TPM (Caudal)}\\\ n_d = 5{,}922 \text{ to } 6{,}120\\\ 5{,}878 \text{ to } 6{,}062 \in G\end{gathered}$$"]
    Prot["$$\begin{gathered}\text{Protein abundance}\\\ \text{Messner: } 1{,}441 \text{ to } 1{,}849,\ \text{UniProt} \to \text{ORF}\\\ \text{Zelezniak: } 726\ (\log \text{ signal})\\\ \text{Muenzner: } 1{,}336 \text{ to } 1{,}571\end{gathered}$$"]
    Turn["$$\begin{gathered}\text{Degradation rate, h}^{-1}\\\ \text{Muenzner turnover}\\\ n_d = 919 \text{ to } 1{,}177\end{gathered}$$"]
    Sort["$$\begin{gathered}\text{Sort-seq expression (Vaishnav)}\\\ y \in \mathbb{R},\ \text{bin index } 0.32 \text{ to } 17.0\end{gathered}$$"]
    Morph["$$\begin{gathered}\text{CalMorph (Ohya, hold-out)}\\\ y \in \mathbb{R}^{281},\ \text{global, not on } G\end{gathered}$$"]
  end

  subgraph Tokens["$$\text{Tokens per record}$$"]
    direction LR
    Env["$$\begin{gathered}\text{environment } e\\\ \mathrm{sha256}(\text{environment JSON})\\\ \text{Jackson 11, Nadal 2, Boocock 2, else 1}\end{gathered}$$"]
    Typ["$$\begin{gathered}\text{experiment type } t\ (\text{phenotype class})\\\ \text{microarray, microarray score, single-cell,}\\\ \text{pseudobulk, RNA-seq, protein abundance,}\\\ \text{protein turnover, pooled sort-seq, calmorph}\end{gathered}$$"]
    Ds["$$\begin{gathered}\text{dataset id } d \in \{0, \ldots, 12\}\\\ \text{scale } \sigma:\ \log_2,\ \text{score},\ \text{UMI},\ \text{TPM},\\\ \text{linear},\ \log,\ \text{rate},\ \text{bin index},\ \text{calmorph}\end{gathered}$$"]
  end

  Batch["$$\begin{gathered}\text{Batch of } B\ \text{records}\\\ X \in \mathbb{R}^{|G| \times d_0}\ \text{node features, shared}\\\ \text{gene\_index} \in \mathbb{Z}^{K},\ \text{pert\_type} \in \{\mathrm{del}, \mathrm{var}, \mathrm{lof}, \mathrm{abs}\}^{K},\ \text{gene\_batch} \in \{0..B-1\}^{K}\\\ \text{allele\_parent} \in \mathbb{Z}^{K},\ \text{promoter\_seq}:\ B\ \text{strings or none}\\\ Y \in \mathbb{R}^{B \times |G|},\ M = \neg\,\mathrm{isnan}(Y),\ y_{\text{scalar}} \in \mathbb{R}^{B},\ Y_{\text{global}} \in \mathbb{R}^{B \times 281}\\\ e, t, d, \sigma \in \mathbb{Z}^{B}\end{gathered}$$"]

  Store --> Genotypes
  Store --> Labels
  Store --> Tokens
  Genotypes --> Batch
  Labels --> Batch
  Tokens --> Batch

  classDef store fill:#E1D5E7,stroke:#846592,stroke-width:2px
  classDef geno fill:#FFE6CC,stroke:#BD8800,stroke-width:2px
  classDef label fill:#F8CECC,stroke:#A24A46,stroke-width:2px
  classDef token fill:#FFF2CC,stroke:#BCA04C,stroke-width:2px
  class Store,Batch store
  class Del,Dbl,Iso,Seg,Prom geno
  class L2,Sc,Umi,Tpm,Prot,Turn,Sort,Morph label
  class Env,Typ,Ds token
```

Render (PDF, outlined SVG and PNG land in `notes/assets/pdf-output/` under this note's name):

```bash
bash notes/assets/publish/scripts/mermaid_pdf.sh notes/experiments.043-expression-proteome-morphology.mermaid.data-flow.md "#F5EEDD"
```

![](./assets/pdf-output/experiments.043-expression-proteome-morphology.mermaid.data-flow.svg)
