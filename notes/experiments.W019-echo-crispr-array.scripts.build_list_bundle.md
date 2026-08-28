---
id: mf997ts1hr7m376vgnkufcw
title: build_list_bundle
desc: 'One zip for handoff: the W019 bench sheet as a PDF plus its five tables as CSVs, with a per-file sha256 manifest.'
updated: 1787679086494
created: 1787679086494
---

## 2026.08.25 - One zip: the sheet to read and the tables to work from

A reviewed PDF is not what someone at the bench needs. They need the same tables as
files they can sort, filter and paste into a picklist, and they need to be able to tell
that the two forms agree. This bundles both.

```bash
~/miniconda3/envs/torchcell/bin/python \
  experiments/W019-echo-crispr-array/scripts/build_list_bundle.py \
  --copy-to /tmp/screenshots/W019-echo-crispr-array
```

Contents, seven files:

| in the zip | from | what it is |
| --- | --- | --- |
| `w019-strain-build-list.pdf` | `notes-tex/w019-strain-build-list/w019-strain-build-list-clean.pdf` | the bench sheet, share view |
| `csv/t1-existing-singles.csv` | `results/build_list_tables/` | the 12 singles in hand, ours against Costanzo 2016 |
| `csv/t2-existing-doubles.csv` | same | the 13 doubles in hand |
| `csv/t3-new-doubles.csv` | same | D01 to D25, the doubles to construct |
| `csv/t4-new-triples.csv` | same | T01 to T20, the triples to construct |
| `csv/t5-plate.csv` | same | what goes on the measurement plate |
| `MANIFEST.txt` | generated | per-file sha256, source path, and the commit |

Four decisions worth keeping:

- **The clean view, not the draft.** The plain build carries section-status chips, which are
  an internal editing signal and say nothing to a reader outside the group.
- **CSVs are copied, never re-derived.** `build_list_tables.py` renders each table once
  from one list of records into both `.tex` and `.csv`, which is what makes the typeset
  table and the data file unable to disagree. Re-deriving here would reopen exactly that
  gap. A missing input is a hard stop, not a partial zip.
- **Deterministic.** Every member is stamped with a fixed mtime, so an unchanged bundle
  rebuilds to the same bytes and the same sha256. That is what lets a hash identify a
  handoff, the same rule the rest of the repo uses for provenance.
- **The zip is not committed.** Every byte in it is already tracked, so the archive would
  be a second binary copy that goes stale as soon as either half changes. It is
  gitignored and rebuilt on demand, which also keeps the manifest's commit line honest:
  a dirty tree is recorded as `(dirty)` rather than passed off as the commit it names.

First build: sha256 `0b7c456e3ed1d4f54802f7874266f46a3dbf670736c9016f5412b9c1617779c2`,
77,680 bytes, from commit `303709a0e2b0`. Delivered to
`/tmp/screenshots/W019-echo-crispr-array/w019-strain-build-list.zip`.

The picklist is deliberately not in the bundle. Table 5 fixes how many strains the plate
carries and says the well assignment comes from the picklist, which is not settled; see
[[experiments.W019-echo-crispr-array.build-list]].
