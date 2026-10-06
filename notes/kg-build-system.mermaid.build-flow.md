---
id: yzyc8jh8lm4kfqtonpk2ivx
title: Build Flow
desc: ''
updated: 1791326509839
created: 1791326509839
---

## 2026.10.06 - The build, release and serving system

Rendered by `bash notes/assets/publish/scripts/mermaid_pdf.sh notes/kg-build-system.mermaid.build-flow.md` into `notes/assets/pdf-output/kg-build-system.mermaid.build-flow.pdf`, then placed by `make diagrams` in `notes-tex/kg-build-system/figures/`. Colors follow the draw.io palette: yellow = a file in git, orange = a step on the build host, purple = the release artifact and its copies, blue = a serving host or the client, red = a gate that refuses a mismatched pair.

```mermaid
%%{init: {"theme": "base", "themeVariables": {"fontFamily": "Arial", "fontSize": "13px"}, "flowchart": {"nodeSpacing": 30, "rankSpacing": 24, "subGraphTitleMargin": {"top": 4, "bottom": 14}}}}%%
flowchart TB
  classDef src fill:#FFF2CC,stroke:#BCA04C,color:#1F1D1A
  classDef step fill:#FFE6CC,stroke:#BD8800,color:#1F1D1A
  classDef art fill:#E1D5E7,stroke:#846592,color:#1F1D1A
  classDef host fill:#DAE8FC,stroke:#5B7AA6,color:#1F1D1A
  classDef gate fill:#F8CECC,stroke:#A24A46,color:#1F1D1A
  classDef pad fill:none,stroke:none,color:#F4F6FA

  code["git: loaders, schema, adapters"]:::src
  rel["REL or DB commit on main<br/>semantic-release: bump commit,<br/>tag vX.Y.Z, wheel on PyPI"]:::src
  dev["dev LMDBs on the build host<br/>DATA_ROOT/data/torchcell/*<br/>build_manifest fresh"]:::src
  tag{{"build gate<br/>commit carries vX.Y.Z?"}}:::gate

  subgraph build [build host, GilaHyper today, replaceable]
    pad[" "]:::pad
    gen["live rebuild under slurm<br/>BioCypher CSVs, neo4j-admin import,<br/>validate, swap into tc-neo4j-readonly,<br/>stamp release id + KgRelease node + aliases"]:::step
    back["kg_release.sh backup, archive<br/>release directory on /bulk/kg-releases:<br/>.backup, kg_manifest.json,<br/>release.json, SHA256SUMS"]:::art
    gh["GilaHyper tc-neo4j-readonly<br/>serves the new store<br/>KG 3.0, paired v1.6.2"]:::host
    pad ~~~ gen
    gen --> back
    gen -- served --> gh
  end

  snap["database/releases/release.json<br/>+ closures.json, committed as DB(kg)"]:::src
  page["compatibility.md: pairs table<br/>make check in CI"]:::src

  code --> rel --> tag --> gen
  dev --> gen
  gen -- "snapshot written into the checkout" --> snap --> page
```
