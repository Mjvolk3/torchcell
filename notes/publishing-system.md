---
id: e7yzw4syg43an2do189si2q
title: Publishing System
desc: ''
updated: 1789524335316
created: 1789524335316
---

## 2026.09.15 - The build and publishing system, typeset

`notes-tex/publishing-system/` is the typeset account of how the manuscript and the typeset notes are built, checked and published: the two Makefiles and what each target produces, where figures, tables and bibliographies come from, the `make check` gate, and how a build reaches Zotero as a timestamped, content-hashed version in a collection derived from the directory path. Its two figures are the mermaid diagrams in [[publishing-system.mermaid.paper-build]] and [[publishing-system.mermaid.notes-tex-build]], rendered by `make diagrams` through `mermaid_pdf.sh` and scaled from mmdc's 600 pt page to the 182 mm text block with Ghostscript. Build with `make`, gate with `make check`.
