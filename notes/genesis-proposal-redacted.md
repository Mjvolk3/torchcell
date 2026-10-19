---
id: khe1y6y3yq2vu3eq3fuiqnu
title: Genesis Proposal Redacted
desc: ''
updated: 1792391912960
created: 1792391912960
---

## 2026.10.19 - Redacted transcription of the Genesis Phase I proposal

The typeset document is `notes-tex/genesis/genesis-proposal-redacted/`, a transcription of the Phase I proposal submitted 2026-04-23 to the Genesis Mission RFA under focus area 2-B (genotype to phenotype), project title "Foundation Models for Metabolic Engineering". The submitted file is NOT committed; it is the only copy of the administrative data.

- Source: `Genesis Proposal 4-23-2026.docx`, sha256 `d59f9065b8895f70b997667e3034f903426caf75ebd3e7fda755b38313ec6fea`, read with `pandoc -t markdown`; the two embedded images (`word/media/image1.png`, `image2.png`) were converted to `figures/genesis-fig1.pdf` and `figures/genesis-fig2.pdf` unchanged.
- Redacted, each replaced by a gray `[redacted: ...]` marker: the lead PI's name, telephone and email; the street and postal addresses; the RFA number; the names of the two partner-lab PIs; the three named industry partners in the Phase II paragraph; and every Year 1 budget figure, including the total.
- Kept as submitted: institutions (UIUC, LBNL, ANL) and facilities (NCSA, iBioFAB, NSF iBioFoundry, Aurora, Equinox, Solstice), the scientific text, every performance number and threshold, Table 1 (the seven go/no-go metrics, the same ones in [[genesis-mission-consortium.resource-request-template]]), the figure, the URLs, and the bibliography with its author lists. Group names in the body ("the X group") became institutional roles so the PI names do not leak through them.
- House-style edits only: the em-dashes in Aim 2 became commas, and the underlined sentence in Aim 1 is italic because an underline cannot break across lines; the text says so in place.
- Build: `make -C notes-tex/genesis/genesis-proposal-redacted` then `make check` (clean, 0 errors, 0 warnings on 2026-10-19). No `ZOTERO_COLLECTION`: the bibliography is a numbered list matching the submission's superscripts, so `references.bib` is an empty placeholder.
