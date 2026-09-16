---
id: tn6ozzdqnvtzipoiz01ir3u
title: Notes Tex Build
desc: ''
updated: 1789524140802
created: 1789524140802
---

## 2026.09.15 - What `make` builds for a typeset note

Rendered by `bash notes/assets/publish/scripts/mermaid_pdf.sh notes/publishing-system.mermaid.notes-tex-build.md` into `notes/assets/pdf-output/publishing-system.mermaid.notes-tex-build.pdf`, then placed by `make diagrams` in `notes-tex/publishing-system/figures/`. Same colors as the manuscript diagram; gray = the gate, which builds nothing.

```mermaid
%%{init: {"theme": "base", "themeVariables": {"fontFamily": "Arial", "fontSize": "13px"}, "flowchart": {"nodeSpacing": 28, "rankSpacing": 34}}}%%
flowchart TB
  classDef src fill:#FFF2CC,stroke:#BCA04C,color:#1F1D1A
  classDef tgt fill:#FFE6CC,stroke:#BD8800,color:#1F1D1A
  classDef out fill:#E1D5E7,stroke:#846592,color:#1F1D1A
  classDef ext fill:#DAE8FC,stroke:#5B7AA6,color:#1F1D1A
  classDef pub fill:#F8CECC,stroke:#A24A46,color:#1F1D1A
  classDef gate fill:#F5F5F5,stroke:#666666,color:#1F1D1A

  script["experiments/ID/scripts/NAME.py<br/>the only source of figures,<br/>tables and numbers"]:::ext
  svg["notes/assets/images/SLUG/<br/>NAME.svg, true-size panels"]:::src
  tables["tables/NAME.tex<br/>generated, never hand-edited"]:::src
  sections["DOC.tex and sections/*.tex<br/>the prose"]:::src
  coll["Zotero group and personal<br/>collections named DOC,<br/>served by tc-lit"]:::ext

  plots["make plots<br/>rsvg-convert at a measured zoom"]:::tgt
  bib["make bib-pull"]:::tgt
  figpdf["figures/NAME.pdf"]:::src
  refs["references.bib"]:::src

  build["make<br/>Tectonic, tcdoc.sty,<br/>SOURCE_DATE_EPOCH pinned"]:::tgt
  clean["make clean-view"]:::tgt
  check["make check<br/>widths, provenance, citations,<br/>spelling, overfull boxes"]:::gate

  pdf["DOC.pdf<br/>draft view, status chips"]:::out
  cpdf["DOC-clean.pdf<br/>share view, chips hidden"]:::out
  report["gate report, exit 0 or 1"]:::gate

  zotpy["zotero_publish.py DOC"]:::tgt
  zot["Zotero: torchcell / notes-tex / DOC<br/>each build as a hashed version"]:::pub

  script --> svg --> plots --> figpdf
  script --> tables
  coll --> bib --> refs
  sections --> build
  tables --> build
  figpdf --> build
  refs --> build
  build --> pdf
  sections --> clean --> cpdf
  pdf --> check --> report
  pdf --> zotpy --> zot
```
