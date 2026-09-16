---
id: 7syljo8wddjwenuj04a5ka3
title: Paper Build
desc: ''
updated: 1789524133364
created: 1789524133364
---

## 2026.09.15 - What `make` builds for the manuscript

Rendered by `bash notes/assets/publish/scripts/mermaid_pdf.sh notes/publishing-system.mermaid.paper-build.md` into `notes/assets/pdf-output/publishing-system.mermaid.paper-build.pdf`, then placed by `make diagrams` in `notes-tex/publishing-system/figures/`. Colors follow the draw.io palette: yellow = a source file, orange = a make target, purple = a built PDF, blue = outside the repo, red = a publish step.

```mermaid
%%{init: {"theme": "base", "themeVariables": {"fontFamily": "Arial", "fontSize": "13px"}, "flowchart": {"nodeSpacing": 28, "rankSpacing": 34}}}%%
flowchart TB
  classDef src fill:#FFF2CC,stroke:#BCA04C,color:#1F1D1A
  classDef tgt fill:#FFE6CC,stroke:#BD8800,color:#1F1D1A
  classDef out fill:#E1D5E7,stroke:#846592,color:#1F1D1A
  classDef ext fill:#DAE8FC,stroke:#5B7AA6,color:#1F1D1A
  classDef pub fill:#F8CECC,stroke:#A24A46,color:#1F1D1A

  subgraph inputs [sources]
    direction LR
    content["content.tex<br/>one shared body"]:::src
    wrappers["submission.tex<br/>editing.tex<br/>twocolumn.tex<br/>thin wrappers"]:::src
    drawio["notes/assets/drawio/<br/>NAME.drawio.svg"]:::src
    group["Zotero group<br/>collection paper<br/>served by tc-lit"]:::ext
  end

  figures["make figures<br/>headless draw.io export,<br/>crop-to-ink, size gate"]:::tgt
  bib["make bib-pull"]:::tgt
  figpdf["figures/NAME.pdf"]:::src
  refs["references.bib"]:::src

  paper["make paper<br/>Tectonic, one view per wrapper,<br/>SOURCE_DATE_EPOCH pinned"]:::tgt
  figproto["make figproto"]:::tgt

  sub["submission.pdf<br/>journal view, single column"]:::out
  edit["editing.pdf<br/>drafting view, status chips,<br/>word budgets"]:::out
  two["twocolumn.pdf<br/>published-like double column"]:::out
  proto["figure-proto.pdf<br/>true-scale sizing canvas"]:::out

  publish["make publish"]:::tgt
  sync["sync-overleaf.sh"]:::tgt
  zot["Zotero: torchcell / paper / nature-biotech<br/>editing.pdf as a hashed version"]:::pub
  overleaf["Overleaf, the shared copy<br/>submission.tex as main.tex"]:::pub

  drawio --> figures --> figpdf
  group --> bib --> refs
  content --> paper
  wrappers --> paper
  figpdf --> paper
  refs --> paper
  paper --> sub
  paper --> edit
  paper --> two
  figproto --> proto
  edit --> publish --> zot
  sub --> sync --> overleaf
```
