---
id: 247j6dzilufd33mkn02dyts
title: Label Policy
desc: ''
updated: 1789733843685
created: 1789733843685
---

## 2026.09.18 - The 025 build stages as they are, and the label-policy alternative

Measured on `$DATA_ROOT/data/torchcell/experiments/025-solid-growth/001-full-build/` (LMDB entry
counts by `lmdb.stat`, sizes by `du`). The served knowledge graph is NOT rebuilt by any of this; a
build is one Cypher query over it followed by the three stages of
`paper/nature-biotech/figures/neo4j_cell-conversion-deduplication-aggregation.pdf`. Left: the
current pipeline, each stage a full LMDB copy. Right, as built on 2026-09-19 (slurm 2400, [[experiments.029-solid-growth-ko]]): the 029 build, the alternative discussed in
[[experiments.025-solid-growth.s3-closure]], which keeps the query, the conversion and the
genotype aggregation and drops only the mean-merge, moving the choice of value to a versioned
label policy applied at read time, the same way the S-subsets and the pinned splits are already
index artifacts over the processed LMDB.

```mermaid
%%{init: {'theme':'base','themeVariables':{'background':'#F5EEDD','clusterBkg':'#F5EEDD','clusterBorder':'#E0D6BE','lineColor':'#B7AC93','fontSize':'13px'}}}%%
graph TD
  KG["$$\begin{gathered}\text{Served knowledge graph (Neo4j)}\\\ \text{not rebuilt; queried}\end{gathered}$$"]

  subgraph Now["$$\text{025 build as it is: 3.2 TB on disk, five copies}$$"]
    direction TB
    R["$$\begin{gathered}\texttt{raw/}\\\ 43{,}819{,}983 \text{ experiments}\\\ 839 \text{ GB}\end{gathered}$$"]
    C["$$\begin{gathered}\texttt{conversion/}\\\ \text{essential, lethal} \to \text{fitness } 0\\\ 43{,}819{,}983 \text{ experiments}\\\ 837 \text{ GB}\end{gathered}$$"]
    D["$$\begin{gathered}\texttt{deduplication/}\\\ \text{same type + same gene set} \to \text{ONE mean}\\\ f \leftarrow \text{mean},\ \sigma \leftarrow \text{RMS},\ p \leftarrow t\text{-test}\\\ 27{,}040{,}733 \text{ entries}\\\ 517 \text{ GB}\end{gathered}$$"]
    A["$$\begin{gathered}\texttt{aggregation/}\\\ \text{group by genotype}\\\ 13{,}525{,}071 \text{ records}\\\ 517 \text{ GB}\end{gathered}$$"]
    P["$$\begin{gathered}\texttt{processed/}\\\ 13{,}525{,}071 \text{ records} + \text{label\_df}\\\ 517 \text{ GB}\end{gathered}$$"]
    IDX["$$\begin{gathered}\text{index artifacts (already)}\\\ \text{S0 / S2 / S3 subsets, pinned R split,}\\\ \text{Q split: lists of record indices}\end{gathered}$$"]
  end

  subgraph Alt["$$\text{029 build 001: deletions only, no mean-merge, policy at read}$$"]
    direction TB
    R2["$$\begin{gathered}\texttt{raw/}\\\ \text{deletion alleles only, both temperatures}\\\ 26{,}796{,}499 \text{ experiments}\\\ 615 \text{ GB}\end{gathered}$$"]
    C2["$$\begin{gathered}\texttt{conversion/}\\\ \text{essential, lethal} \to \text{fitness } 0\\\ \text{kept as its OWN entry}\\\ 15{,}137 \text{ converted, rest passed through}\\\ 614 \text{ GB}\end{gathered}$$"]
    A2["$$\begin{gathered}\texttt{aggregation/}\\\ \text{group by genotype}\\\ \text{every source entry kept:}\\\ \text{screen, temperature, marker, } \sigma, p\\\ 9{,}297{,}912 \text{ records}\\\ 547 \text{ GB}\end{gathered}$$"]
    POL["$$\begin{gathered}\text{LabelPolicy (pydantic, hashed; to write)}\\\ \text{precedence: Kuzmin} \succ \text{Costanzo 30 C} \succ \text{26 C}\\\ \text{0 only if no measurement}\\\ p \text{ from the chosen entry}\end{gathered}$$"]
    LBL["$$\begin{gathered}\text{label artifact per policy}\\\ \text{one value per label per record}\\\ \text{+ subsets and splits as now}\end{gathered}$$"]
  end

  KG --> R --> C --> D --> A --> P
  P --> IDX
  KG --> R2 --> C2 --> A2
  A2 --> POL --> LBL

  L1["$$\begin{gathered}\text{lost here: source values,}\\\ \text{screen, temperature, allele,}\\\ \text{every source } p\end{gathered}$$"]
  D -.- L1

  classDef kg fill:#FFE6CC,stroke:#BD8800,color:#000
  classDef stage fill:#F8CECC,stroke:#A24A46,color:#000
  classDef keep fill:#E1D5E7,stroke:#846592,color:#000
  classDef pol fill:#FFF2CC,stroke:#BCA04C,color:#000
  classDef loss fill:#FFFFFF,stroke:#A24A46,stroke-width:1.5px,stroke-dasharray:4 2,color:#000
  class KG kg
  class R,C,D,A,P stage
  class R2,C2,A2 keep
  class POL,LBL,IDX pol
  class L1 loss
```
