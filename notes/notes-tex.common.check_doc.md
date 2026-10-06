---
id: 8qgkkjsydxbsa4ygzdgekrp
title: Check_doc
desc: ''
updated: 1791255596669
created: 1791255596670
---

## 2026.10.05 - Source text is exempt from the spelling gate

A generated bibliography carries titles, journal names and surnames exactly as PubMed records them, and Americanizing any of them would falsify the citation: "Mech Ageing Dev" is a journal and Storey is an author. The gate already skipped verbatim quotes (``...'') and `%%` comments; it now also skips the argument of `\sourcetext{...}`, brace-matched because citation text holds accent macros. The macro itself is defined in [[notes-tex.common.tcdoc]] and sets nothing. First used by `experiments/015-spell/scripts/spell_publication_tables.py` for [[experiments.015-spell.publications]].
