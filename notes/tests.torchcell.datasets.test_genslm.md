---
id: xhxi0agwj818ru4j80m60hh
title: Test_genslm
desc: ''
updated: 1791416326348
created: 1791416326348
---

## 2026.10.07 - Coverage

Six tests on the `embedding_genome` conftest stub and an in-test genome with an RNA gene, with
`GenSLMDataset.initialize_model` patched to a stand-in mapping each sequence to
`[len, #A, #C, #G, #T]`: window strings per gene, batch order and size, the `no_cds.json`
record and `KeyError` for a left-out RNA gene, 6,144 nt truncation, reload without rebuilding
the backbone, and the invalid model name. See [[torchcell.datasets.genslm]].
