---
id: dejx5loei5i11jqkob4ynem
title: Tcdoc
desc: ''
updated: 1791255600192
created: 1791255600192
---

## 2026.10.05 - `\sourcetext{}` marks text copied from a source record

Bibliographic text copied from a source record (a citation's authors, title and journal) must keep its source spelling, so it needs a way to be marked that is not a quotation mark. `\sourcetext{#1}` is an identity macro; its only effect is that `make check` ([[notes-tex.common.check_doc]]) skips its argument when checking spelling. Use it for generated citations, never for prose.
