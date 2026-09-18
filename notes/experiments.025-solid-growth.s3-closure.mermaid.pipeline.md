---
id: 1fswu3zes2lnofa8djklkeq
title: Pipeline
desc: ''
updated: 1789719768350
created: 1789719768350
---

## 2026.09.18 - How the S3 closure pool is built, and where a join can go wrong

Figure a of [[experiments.025-solid-growth.s3-closure]]. Rendered with
`bash notes/assets/publish/scripts/mermaid_pdf.sh notes/experiments.025-solid-growth.s3-closure.mermaid.pipeline.md`
to `notes/assets/pdf-output/experiments.025-solid-growth.s3-closure.mermaid.pipeline.pdf`, which
`make figures` in `notes-tex/025-s3-closure/` copies to `figures/s3_closure_pipeline.pdf`.
Same palette rule as [[torchcell.models.equivariant_cell_graph_transformer.mermaid.type-i-ii]]:
purple sources, orange query, red processing, yellow training pool; hazards are red-bordered notes
numbered H1 to H5 and discussed in the document.

```mermaid
%%{init: {'theme':'base','themeVariables':{'background':'#F5EEDD','clusterBkg':'#F5EEDD','clusterBorder':'#E0D6BE','lineColor':'#B7AC93','fontSize':'13px'}}}%%
graph TD
  subgraph Sources["$$\text{Sources (each a screen with its own controls)}$$"]
    direction LR
    C16["$$\begin{gathered}\text{Costanzo 2016}\\\ \text{smf}_{26/30},\ \text{dmf},\ \varepsilon,\ p\\\ f_a\text{: 350 control screens}\\\ 12.2\text{M double rows}\end{gathered}$$"]
    K18["$$\begin{gathered}\text{Kuzmin 2018}\\\ \text{smf, dmf, tmf},\ \varepsilon,\ \tau,\ p\\\ f_a \text{ from Costanzo 2016}\\\ 501\text{k rows}\end{gathered}$$"]
    K20["$$\begin{gathered}\text{Kuzmin 2020}\\\ \text{smf, dmf, tmf},\ \varepsilon,\ \tau,\ p\\\ f_a \text{ from Costanzo 2016}\\\ 1.4\text{M rows}\end{gathered}$$"]
    SGD["$$\begin{gathered}\text{SGD essentiality}\\\ \text{essential} \Rightarrow f = 0\\\ \text{no measurement}\end{gathered}$$"]
    SL["$$\begin{gathered}\text{SynthLethDB}\\\ \text{synthetic lethal} \Rightarrow f_{ab} = 0\\\ \text{no measurement}\end{gathered}$$"]
  end

  Query["$$\begin{gathered}\text{Served knowledge graph}\\\ \texttt{001\_all\_solid\_growth.cql}\\\ \text{solid medium, S288C gene set}\end{gathered}$$"]

  subgraph Build["$$\text{Build (query.py)}$$"]
    direction TB
    Conv["$$\begin{gathered}\text{CompositeFitnessConverter}\\\ \text{essential / lethal} \to \text{fitness } 0\end{gathered}$$"]
    Dedup["$$\begin{gathered}\text{MeanExperimentDeduplicator}\\\ \text{key = type + sorted genes}\\\ f \leftarrow \text{mean},\ \sigma \leftarrow \text{RMS}\\\ \varepsilon,\tau \leftarrow \text{mean},\ p \leftarrow t\text{-test over the } k \text{ values}\end{gathered}$$"]
    Agg["$$\begin{gathered}\text{GenotypeAggregator}\\\ \text{one record per genotype}\\\ 13{,}525{,}071 \text{ records}\end{gathered}$$"]
  end

  subgraph Pool["$$\text{S3 closure pool, 1,121,645 records}$$"]
    direction LR
    S1["$$\begin{gathered}\text{singles}\\\ \text{all } 5{,}694\end{gathered}$$"]
    S2["$$\begin{gathered}\text{doubles}\\\ \text{pair inside a triple}\\\ 739{,}219\end{gathered}$$"]
    S3["$$\begin{gathered}\text{triples}\\\ \text{all } 376{,}732\\\ \text{010 random split pinned}\end{gathered}$$"]
  end

  Train["$$\begin{gathered}\text{Trainer}\\\ \text{fitness + interaction per order}\\\ \text{val / test: pinned triples only}\end{gathered}$$"]

  H1["$$\begin{gathered}\textbf{H1}\ \text{essential } 0 \text{ averaged with a}\\\ \text{measured allele: YER048W-A}\\\ 0.904 \to 0.452\end{gathered}$$"]
  H2["$$\begin{gathered}\textbf{H2}\ \text{SynthLethDB } 0 \text{ beside a}\\\ \text{measured single: two values}\\\ \text{in one record (masked)}\end{gathered}$$"]
  H3["$$\begin{gathered}\textbf{H3}\ \varepsilon \text{ is defined inside one screen};\\\ \text{the build's } f_a \text{ is a mean}\\\ \text{over every screen}\end{gathered}$$"]
  H4["$$\begin{gathered}\textbf{H4}\ p \text{ of a merged record is a}\\\ t\text{-test with } k-1 \text{ d.f.};\\\ \text{the source } p \text{ is discarded}\end{gathered}$$"]
  H5["$$\begin{gathered}\textbf{H5}\ \text{source scoring set NaN}\\\ \text{fitness to } 1.0 \text{ before } \tau;\\\ \text{not invertible}\end{gathered}$$"]

  C16 --> Query
  K18 --> Query
  K20 --> Query
  SGD --> Query
  SL --> Query
  Query --> Conv --> Dedup --> Agg
  Agg --> S1
  Agg --> S2
  Agg --> S3
  S1 --> Train
  S2 --> Train
  S3 --> Train
  Conv -.- H1
  Conv -.- H2
  Dedup -.- H3
  Dedup -.- H4
  K18 -.- H5

  classDef source fill:#E1D5E7,stroke:#846592,color:#000
  classDef query fill:#FFE6CC,stroke:#BD8800,color:#000
  classDef proc fill:#F8CECC,stroke:#A24A46,color:#000
  classDef pool fill:#FFF2CC,stroke:#BCA04C,color:#000
  classDef hazard fill:#FFFFFF,stroke:#A24A46,stroke-width:1.5px,stroke-dasharray:4 2,color:#000
  class C16,K18,K20,SGD,SL source
  class Query query
  class Conv,Dedup,Agg proc
  class S1,S2,S3,Train pool
  class H1,H2,H3,H4,H5 hazard
```
