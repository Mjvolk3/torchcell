---
id: vl95zahgs4zlx08elqksndk
title: Nucleotide_transformer
desc: ''
updated: 1790813178196
created: 1790813178196
---

## 2026.09.30 - load_model honors its argument; pooled shape (issue #543)

- `load_model(model_name)` passed the module constant `MODEL_NAME` to both `from_pretrained` calls and the download check; it now downloads (into the same cache directory) and loads the named Hub id. `_check_and_download_model` takes the name, defaulting to `MODEL_NAME`.
- `embed(..., mean_embedding=True)` returned `[1, batch, dim]` from a stray `unsqueeze(0)`; it now returns `[batch, dim]`. The only live caller, [[torchcell.datasets.nucleotide_transformer]], was adjusted so its stored layout is unchanged.

Tests: [[tests.torchcell.models.test_nucleotide_transformer]].
