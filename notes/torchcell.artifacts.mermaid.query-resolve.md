---
id: re5u8hkx0ygmy7wybj3oik5
title: Query Resolve
desc: ''
updated: 1791393965389
created: 1791393965389
---

## 2026.10.07 - A query against Neo4j, and how the off-graph sequence bytes arrive

Rendered by `bash notes/assets/publish/scripts/mermaid_pdf.sh notes/torchcell.artifacts.mermaid.query-resolve.md` into `notes/assets/pdf-output/torchcell.artifacts.mermaid.query-resolve.{pdf,svg,png}`. The graph holds only the pointer (an `ArtifactRef`, sha256-pinned), so the query result is complete when Neo4j answers; the bytes arrive through one resolver in one order, local tier, then cache, then tc-data, verified at every step ([[plan.artifact-tier.2026.10.07]], [[torchcell.artifacts]]).

```mermaid
%%{init: {"theme": "base", "themeVariables": {"fontFamily": "Arial", "fontSize": "13px"}}}%%
sequenceDiagram
    participant U as user code<br/>Neo4jQueryRaw / Neo4jCellDataset
    participant N as Neo4j<br/>KG release (latest)
    participant R as resolver<br/>torchcell.artifacts
    participant L as local tier<br/>DATA_ROOT/torchcell-genomes
    participant T as tc-data on Radiant<br/>tiers on Taiga

    U->>N: Cypher query (Caudal isolates)
    N-->>U: records: expression vectors inline,<br/>genotype perturbations with sequence_ref<br/>tc://genomes/peter2018.../tar#YAL002W.fasta#token

    Note over U,R: process(): gate, once per distinct ref
    U->>R: check(ref), manifests only, no download
    R->>L: manifest lists path + sha256?
    alt local tier present
        L-->>R: yes
    else no local tier
        R->>T: GET /genomes/set/manifest
        T-->>R: manifest row
    end
    R-->>U: resolvable, else UnresolvableArtifactError and no store is written

    Note over U,T: later, when a model needs the bytes
    U->>R: materialize(ref)
    alt local tier present
        R->>L: read file, verify sha256
        L-->>R: path
    else no local tier
        R->>T: GET /genomes/set/artifact/path (X-API-Key, Range)
        T-->>R: bytes + X-Artifact-SHA256
        R->>R: verify sha256, write artifact-cache/sha256/file
    end
    R-->>U: local path, open member YAL002W.fasta#token
```
