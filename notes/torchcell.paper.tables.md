---
id: 9kiyntomkp172ya55xssj83
title: Tables
desc: ''
updated: 1783476221688
created: 1783476221688
---

## 2026.07.07 - Reusable paper-table generation (kill md→LaTeX hand-transcription)

Paper tables were being hand-copied from markdown notes into LaTeX, which drifts
silently and hid errors (a whole dataset table had been transcribed by eye). This
module makes the pattern reliable and reusable: define a table once as data, then
render the SAME object to both markdown and LaTeX, and compute any data-derived
column directly from the source of truth rather than typing numbers in.

The motivating column is *Signal (gzip)* -- a Kolmogorov-complexity proxy equal to
the gzip size of every record's serialized phenotype. It exists so a reader grasps
a dataset's information content (breadth × depth) in one number. The hard part is
scale: the recipe in the note accumulated one big in-memory blob before compressing,
which is impossible for the 20.7M-record / 40+ GB Costanzo LMDBs, so those cells sat
at "pending". `stream_gzip_signal` feeds records one at a time into a
`zlib.compressobj`, giving byte-identical output with bounded memory -- so every
dataset gets a real value in a single pass (Costanzo dmf/dmi = 128 / 155 MB).

### What's reusable here

- `stream_gzip_signal` -- streaming gzip size over an LMDB cursor; `extract` is
  pluggable so future tables can measure a different per-record payload.
- `SignalCache` -- pydantic, JSON-backed, keyed by the LMDB's `data.mdb` mtime+size,
  so reruns only recompute datasets that were actually rebuilt.
- `PaperTable` (`Column`/`Row`) -- one object, `.to_markdown()` (sections → H3) and
  `.to_latex()` (booktabs; sections → bold `\multicolumn`); `tex_escape`,
  `human_bytes`, `read_frontmatter` (preserve dendron identity when regenerating a note).

First consumer: `[[paper.nature-biotech.scripts.generate_datasets_table]]`. Tests:
`tests/torchcell/paper/test_tables.py` (incl. streaming == non-streaming
`gzip.compress` byte-parity).
