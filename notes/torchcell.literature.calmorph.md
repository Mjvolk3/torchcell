---
id: 9mfpec23laebe72u2770ezn
title: Calmorph
desc: ''
updated: 1783563699661
created: 1783563699661
---

## 2026.07.08 - Replace a hand-cleaned Excel with a rebuildable extraction recipe

This module exists to eliminate a provenance dead-end: the CalMorph 501-parameter schema was previously built from a manually prepared `SI_1_parameters.xlsx` that had been hand-cleaned from a poor Mathpix OCR of the Ohya 2005 SI -- an artifact nobody could rebuild or trust. This is the extraction-backed replacement: the same table exists as a born-digital PDF in the Zotero library (DOI `10.1073/pnas.0509436102`), so this recipe reads its exact text layer and reconstructs all 501 `ID -> Description` rows deterministically, with the old manual build kept only as verification ground truth. It is the concrete proof-of-concept for the born-digital-first stance.

- `extract_calmorph_parameters`: refuses to run unless [[torchcell.literature.extract]] classifies the PDF as born-digital -- it trusts the text layer and does NOT silently fall back to OCR (scans are an upstream routing decision).
- `parse_calmorph_table`: anchors each row on the running `No.` integer + the CalMorph id regex, then takes the first column-gap-delimited field as the description, so sparse nuclear-stage cells and multi-token Definitions parse unambiguously.
- Every value traces to a hash-pinnable source PDF + a testable parser -- exactly the "more rigorous than the source" reconstruction the project aims for.
