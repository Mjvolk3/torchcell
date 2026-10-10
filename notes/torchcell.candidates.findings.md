---
id: 291t9ymzf3a95pvskyghink
title: Findings
desc: ''
updated: 1791617777897
created: 1791617777897
---

## 2026.10.10 - Recorded G1 judgments

G1 cannot tell an aggregation from a transcription off row fields alone. Five rows were already read, and their settled-row modules hold the quotes, so each finding is composed from those modules' `SourcedValue` constants (the evidence a verdict carries is the evidence the module tests audit).

| row | kind | evidence | AggregationRecord |
|---|---|---|---|
| D2Cell 2026 | transcription | `li2024.RE_MODEL`, `CORPUS`, `TEXT_ONLY` | none |
| MCF2Chem 2023 | transcription | `cai2023.EXTRACTION_FROM_REVIEWS`, `PER_RECORD_REFERENCE_RULE` | none |
| CeCaFDB flux compendium | aggregation | `zhang2015.LUMPED_SPLIT`, `RELATIVE_FLUX` | 33 source studies (distinct DOIs of `zhang2015.WORKBOOKS`), `derived_by_aggregator`, mirrored and net-new unmeasured |
| Lim 2022 putidaPRECISE321 | aggregation | `lim2022.COMPENDIUM_STRUCTURE`, `COLLECTION` | 21 projects (the paper's count), `re_measured`, mirrored and net-new unmeasured |
| Borchert 2024 fModules | aggregation | Q_COMPENDIUM and Q_FB_SHARE of `borchert2024`, bound to its pinned `paper.md` | 4 source studies (`SOURCE_STUDIES`, the compendium itself included), all 4 with a mirrored library key, `re_measured` |

Not yet recorded: Oyetunde 2019 (an aggregation row with no settled module) and the served yeast aggregation SynthLethDB, whose source-study count must be measured from the stored records. Both belong to the backfill (piece 3). Lim 2022's 21 is a project count from the paper, not a count of distinct source publications; the backfill measures the latter from `preprocess/sample_ledger.json`.

## 2026.10.10 - Borchert 2024 now names five source studies

After the rebase onto main `ecbd943ac`, `borchert2024.SOURCE_STUDIES` holds five studies (Rand 2017, `randMetabolicPathwayCatabolizing2017`, was added on main), so the finding reads `n_source_studies=5`. Checked on GilaHyper the same day: four of the five have a literature-mirror directory and Rand 2017 has a raw-mirror directory, so `n_sources_mirrored=5` still holds under the field's definition (own paper or release in a mirror). The count is read off the module, so it moves with it; the test pin moved to 5.
