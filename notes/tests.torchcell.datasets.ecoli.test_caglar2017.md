---
id: xfi6yahzlqinncrss1k7g5w
title: Test_caglar2017
desc: ''
updated: 1791382782083
created: 1791382782083
---

## 2026.10.07 - Strain gate, tier addition and raw mirror

Tests for [[torchcell.datasets.ecoli.caglar2017]]. Seventeen synthetic tests: the gate
refuses with the typed `genome_reference` gap and opens once the strain is in the
vocabulary; the nine ASM1798v1 members and the proposed `^ECB_[rt]?\d{5}$` pattern
(disjoint from the three deposited namespaces and from yeast names); the PMC and efetch
retrieval records; the deposit (exact manifest, idempotent second run, refusals of
off-pin staged bytes, of a differing mirror file and of a manifest recording other files,
each leaving nothing written); `retrieve_raw_files` with a stand-in retriever; the GenPept
and GenBank parsers; `identifier_coverage` and `annotation_summary` on hand-built files
with every count asserted.

Four `--data` tests, skipped unless both mirrors are under the exported `DATA_ROOT`: all 26
quotes audit; the raw mirror holds exactly the 46 pinned files; the table shapes equal the
SI's statements (4,196 x 152, 4,196 x 105, 13 branches, 171 samples); every Table S3 protein
resolves through its NCBI record to an `ECB_` tag, 4,196 of them on the same row as Table
S2's id.
