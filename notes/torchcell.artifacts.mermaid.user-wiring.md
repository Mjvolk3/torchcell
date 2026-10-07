---
id: 6kk64u109g7f5fp5f6vkh41
title: User Wiring
desc: ''
updated: 1791394613880
created: 1791394613880
---

## 2026.10.07 - The user's view: one query, two stores behind it

Rendered by `bash notes/assets/publish/scripts/mermaid_pdf.sh notes/torchcell.artifacts.mermaid.user-wiring.md white` into `notes/assets/pdf-output/torchcell.artifacts.mermaid.user-wiring.{pdf,svg,png}`. The graph answers with records that carry pointers instead of the heavy data; touching a pointer makes the same library fetch the bytes from the store and verify the hash. The detailed sequence is [[torchcell.artifacts.mermaid.query-resolve]].

```mermaid
%%{init: {"theme": "base", "themeVariables": {"fontFamily": "Arial", "fontSize": "14px"}, "flowchart": {"nodeSpacing": 40, "rankSpacing": 60}}}%%
flowchart LR
    classDef user fill:#FFF2CC,stroke:#BCA04C,color:#1F1D1A
    classDef kg fill:#DAE8FC,stroke:#5B7AA6,color:#1F1D1A
    classDef store fill:#E1D5E7,stroke:#846592,color:#1F1D1A

    U["you<br/>torchcell query"]:::user
    N["Neo4j<br/>knowledge graph"]:::kg
    S["sequence / transcriptome store<br/>tc-data on Radiant, files on Taiga"]:::store

    U -->|Cypher| N
    N -->|"records + pointers<br/>tc://... + sha256"| U
    U -->|pointer| S
    S -->|"bytes, verified"| U
```
