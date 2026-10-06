---
id: uf3na3do03ihw895r8hiu9e
title: Serve Flow
desc: ''
updated: 1791326823579
created: 1791326823579
---

## 2026.10.06 - From release artifact to a client

Rendered by `bash notes/assets/publish/scripts/mermaid_pdf.sh notes/kg-build-system.mermaid.serve-flow.md` into `notes/assets/pdf-output/kg-build-system.mermaid.serve-flow.pdf`, then placed by `make diagrams` in `notes-tex/kg-build-system/figures/`. Same palette as [[kg-build-system.mermaid.build-flow]]: purple = the artifact and its copies, orange = a step, blue = a serving host or the client, red = a gate.

```mermaid
%%{init: {"theme": "base", "themeVariables": {"fontFamily": "Arial", "fontSize": "13px"}, "flowchart": {"nodeSpacing": 34, "rankSpacing": 18}}}%%
flowchart TB
  classDef src fill:#FFF2CC,stroke:#BCA04C,color:#1F1D1A
  classDef step fill:#FFE6CC,stroke:#BD8800,color:#1F1D1A
  classDef art fill:#E1D5E7,stroke:#846592,color:#1F1D1A
  classDef host fill:#DAE8FC,stroke:#5B7AA6,color:#1F1D1A
  classDef gate fill:#F8CECC,stroke:#A24A46,color:#1F1D1A

  art["release directory on the build host<br/>.backup, kg_manifest.json, release.json, SHA256SUMS"]:::art
  taiga["Taiga archive, the home of the DB<br/>/mnt/zhao5/.../kg-releases/release/, sha256 verified"]:::art
  deploy["kg_release.sh deploy release, on the serving host<br/>verify, neo4j-admin restore into kg-release, CREATE DATABASE,<br/>check the node against release.json, alias latest"]:::step
  gh["GilaHyper tc-neo4j-readonly<br/>KG 3.0, paired v1.6.2"]:::host
  rd["Radiant tc-neo4j: needs a block volume for /data<br/>Neo4j does not run a store on NFS"]:::host
  sync{{"ops.sh sync: same release on every host?<br/>exit 1 on DIVERGED or an unpaired store"}}:::gate
  pair{{"client gate, require_paired: installed torchcell<br/>reproduces every served closure?"}}:::gate
  client["client: Neo4jQueryRaw, TORCHCELL_KG_VERSION<br/>latest, pinned, a release id or major.minor"]:::host
  refuse["IncompatibleReleaseError<br/>install the paired package"]:::gate

  art -- "kg_release.sh ship, rsync" --> taiga --> deploy
  deploy --> rd
  deploy -. "build-only host" .-> gh
  gh --> sync
  rd --> sync
  gh --> pair
  rd --> pair
  pair -- yes --> client
  pair -- no --> refuse
```
