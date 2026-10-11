---
id: n3u0lvmi3l87qjn9iap4bcd
title: Dechiara2022_reporting_summary_text
desc: ''
updated: 1791727352725
created: 1791727352725
---

## 2026.10.11 - Why the Reporting Summary is rendered with pdftotext

The De Chiara 2022 Reporting Summary (`si/si2.pdf`, MOESM2, sha256 `d51bc60064c2514b3a1d1487eefdf274256423b9cd1df051863fd4d67cd11af7`) is the only mirrored text that states the replicate design of the growth phenome ("Automatic growth curve were conducted in 4 replicates, median values were taken for each isolates.") and the haploid exclusion of the sporulation statistics. MinerU's OCR (`si/si2.md`) cut that study-design table to fragments, but the PDF has a text layer. The script renders it with `pdftotext -layout` (poppler 21.01.0) to `si/si2.pdftotext.txt` (sha256 `ce323f8d2b0c2e59f4fef4551794599b350eb372ef247652aea814fef5075757`, 10,084 bytes) beside the PDF and records it in the key's `manifest.json` with a `ProcessingRecord` naming the tool, version, flags and input sha256. The loader's quotes from it audit against those bytes: [[torchcell.datasets.scerevisiae.dechiara2022]].
