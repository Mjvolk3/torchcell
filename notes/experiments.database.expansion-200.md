---
id: sv5i77ihozhpf8wlh6zq6pq
title: Expansion 200
desc: ''
updated: 1792388588645
created: 1792388588645
---

## 2026.10.19 - Adding 200 yeast datasets: the ranking after KG 4.0

Working note for `notes-tex/database/database-expansion-200/`, the successor of
[[experiments.database.expansion-100]]. The typeset document is the deliverable; this note
holds the decisions behind it.

### What changed from the previous document

- **The served base is read, not typed.** `build_candidate_datasets_table.py` now reads the
  newest `database/releases/*.json` snapshot for the yeast count (70 in
  `2026.10.17-4295f7ca`, KG 4.0) and its release id. The goal is 200 additions, 70 to 270.
- **Thirteen candidate rows are served and leave the ranking** (`status="built"`, each with a
  `REMOVAL_REASON` naming the served class): Boocock 2025, De Chiara 2022 (three classes),
  Hale 2024, Parsons 2006, Piotrowski 2017, Puddu 2019, Hackett 2020, Peter 2018, Hu 2007,
  N'Guessan 2025, Vaishnav 2022, Muenzner 2024 (two classes), Jackson 2020. The other six of
  KG 4.0's nineteen additions (Messner 2021, Volk 2021, Yoshikawa 2009 and the extra
  De Chiara and Muenzner classes) were never candidate rows; the candidate "Yoshikawa 2011
  (deletion + overexpression phenome)" is a different paper from the served Yoshikawa 2009
  ethanol and NaCl screen and stays a candidate.
- **The 19 served names join `SUPPORTED_PARTNERS`**, spelled as `build_supported_datasets_table.py`
  spells them, and the 13 synergy entries that named one of the served rows as a candidate
  partner now point at the supported name with `partner_status="supported"`.
- **The pool is 154 rows, 46 short of 200.** `main()` records the shortfall instead of failing
  (`shortfall` in `candidate_datasets.json`; the candidates caption says it), and with the
  whole pool inside the cut no pin can bind.
- **The previous pass is the committed ranking.** `previous_ranked()` reads the rank order
  from `notes-tex/database/database-expansion-100/tables/candidates.tex` rather than
  re-deriving an unbanded order, so the moves table is against what the previous document
  printed (83 moves: 13 removals, 8 promotions into the new band, the rest displacement).
  The expansion-100 document is frozen at version 15; its Makefile says so and the script
  no longer writes there.

### The new first band: state across genomes

Owner, 2026-10-19: "anything that would help with establishing state of proteome and
transcriptome with different genomes is something I want to focus on." The band
`state across genomes` (letter G) replaces `next build`, whose three rows are served. It
holds the rows that measure a transcriptome or proteome on strains whose genomes differ by
sequence: Albert 2018, Jakobson 2025, Teyssonniere 2024, Grossbach 2022, Smith 2008
(promoted from the scale band at rank 112), Skelly 2013, McManus 2014, Foss 2007, ranks 1
to 8. The document's first section lays out what the served graph holds on that axis
(Caudal 2024, Muenzner 2024, Boocock 2025 and N'Guessan 2025 are the only served states on
sequenced genomes; Hale, Bloom, Peter, De Chiara and Puddu fix the genotype side without a
state), what the eight add, and the three gaps no row in the pool fills: a proteome on a
condition axis across isolates, a translation layer on isolates, and a single-cell proteome
on any genome axis.

The per-isolate gene-content figures in that section (median 78 accessory ORFs present,
126 reference ORFs absent, about 5,000 sequence variants and 360 natural loss-of-function
alleles per isolate; 1,614 distinct accessory ORFs and 1,118 distinct absent reference ORFs
over the panel) come from
[[experiments.database.scripts.isolate_gene_content_vs_s288c]], measured on the built
Caudal 2024 and Muenzner 2024 dev stores on 2026-10-19.

### Final sixty, by the numbers

8 G + 23 P + 14 M fill ranks 1 to 45; the first five scale rows (de Boer 2020, Lee 2014,
Nguyen Ba 2022, Galardini 2019, Dutta 2026) complete the fifty; the extra ten are scale.
The sixty name 89 joins, 55 to a served dataset, on 53 rows. Confidence: 34 sourced, 26
recall of the sixty; 65 and 89 over the pool. Six of the eight state rows are recall rows
with no deposit identifier fetched, so gating them is the first task of the next build
week. `make check` clean; the PDF is `notes-tex/database/database-expansion-200/database-expansion-200.pdf`.
