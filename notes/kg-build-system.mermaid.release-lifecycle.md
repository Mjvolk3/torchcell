---
id: g5cza5skfiefw0lmd539hpd
title: Release Lifecycle
desc: ''
updated: 1791326517518
created: 1791326517518
---

## 2026.10.06 - The life of one release

Rendered by `bash notes/assets/publish/scripts/mermaid_pdf.sh notes/kg-build-system.mermaid.release-lifecycle.md` into `notes/assets/pdf-output/kg-build-system.mermaid.release-lifecycle.pdf`, then placed by `make diagrams` in `notes-tex/kg-build-system/figures/`. Same palette as the build-flow diagram; the red diamonds are the four gates of [[versioning]] (2026.10.06 section).

```mermaid
%%{init: {"theme": "base", "themeVariables": {"fontFamily": "Arial", "fontSize": "13px"}, "flowchart": {"nodeSpacing": 34, "rankSpacing": 20}}}%%
flowchart TB
  classDef src fill:#FFF2CC,stroke:#BCA04C,color:#1F1D1A
  classDef step fill:#FFE6CC,stroke:#BD8800,color:#1F1D1A
  classDef art fill:#E1D5E7,stroke:#846592,color:#1F1D1A
  classDef host fill:#DAE8FC,stroke:#5B7AA6,color:#1F1D1A
  classDef gate fill:#F8CECC,stroke:#A24A46,color:#1F1D1A

  land["land the loader and<br/>schema work on main"]:::src
  cut["REL or DB commit<br/>bump commit + tag vX.Y.Z<br/>wheel on PyPI"]:::src
  g1{{"gate 1, build<br/>commit tagged, checkout on it?"}}:::gate
  built["store built and stamped<br/>release id, KG version,<br/>torchcell_version, torchcell_tag"]:::step
  snap["snapshot committed<br/>DB(kg): release ..., patch bump"]:::src
  g4{{"gate 4, page<br/>paired tag reads every closure?"}}:::gate
  pagepdf["compatibility.md<br/>pairs table + matrix, CI make check"]:::src
  stop4["page refuses to render, CI fails"]:::gate
  arch["backup, archive, ship<br/>release directory on Taiga"]:::art
  dep["deploy on a serving host<br/>restore, create, check, alias"]:::step
  g3{{"gate 3, sync<br/>every host on the same release?"}}:::gate
  cli["client connects<br/>TORCHCELL_KG_VERSION"]:::host
  g2{{"gate 2, client<br/>installed schema reproduces<br/>every served closure?"}}:::gate
  read["records read under the<br/>contract they were written with"]:::host
  refuse["IncompatibleReleaseError<br/>names the paired package"]:::gate

  land --> cut --> g1 -- yes --> built --> snap --> g4 -- yes --> pagepdf
  g1 -- no --> stop1["refused before the build"]:::gate
  g4 -- no --> stop4
  built --> arch --> dep --> g3
  cli --> g2 -- yes --> read
  g2 -- no --> refuse
```
