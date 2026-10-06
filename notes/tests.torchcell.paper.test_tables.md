---
id: 5132b9elazvjwku32i93nyf
title: Test_tables
desc: ''
updated: 1791270273216
created: 1791270273216
---

## 2026.10.06 - Phase 21 lane D tests

Phase 21 (lane D): exact full renderings of `PaperTable.to_latex` (header comment, sections in first-seen order including a `None` section, `\addlinespace` placement, bold empty cell, footer, footnote) and `to_markdown`; the unsectioned LaTeX layout; `instance_bytes`; the progress log of `stream_gzip_signal` under a fake clock (`[ds] 2 records (2/s, 1s)`); `read_first_record`; the interaction roles of `phenotype_descriptor`; the `read_frontmatter` fallbacks; and a stale cache fingerprint recomputing.

Finding: `scientific` takes the exponent before rounding the mantissa, so 99,999 renders `10.0×10⁴` (and 9.96 renders `10.0×10⁰`) instead of `1.0×10⁵` (tables.py:107-108).

### Audit 2 notes applied

- Reach (audit 2): `scientific` cells reach paper/nature-biotech/sections/datasets_table.tex through experiments/database/scripts/render_supported_datasets_table.py; no current value has a 10.0 mantissa.
