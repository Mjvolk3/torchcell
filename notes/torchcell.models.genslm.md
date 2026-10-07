---
id: 2raoi10h8xh3m4qqyzs6nbi
title: Genslm
desc: ''
updated: 1791416295364
created: 1791416295364
---

## 2026.10.07 - GenSLM codon language model wrapper

`torchcell/models/genslm.py` wraps the GenSLM foundation models of
[[zvyaginGenSLMsGenomescaleLanguage2023]] (mirror key `zvyaginGenSLMsGenomescaleLanguage2023`,
captured 2026.10.07) for embedding coding sequences.

**What the model encodes.** The paper states the inputs are "nucleotide sequences, encoded at
the codon level (every three nucleotide represents a codon; hence the 20 natural amino acid
language is described by 64 codons)" and that the foundation models were pretrained on ">110
million unique prokaryotic gene sequences from BV-BRC" (Sec. 5.1.2). So the model sees a
coding sequence only, in frame from the start codon, with a context of 2,048 codons (6,144 nt,
Table 1 MSL). No promoter, UTR or intergenic sequence is ever tokenized.

**Vendored assets, not the `genslm` package.** The package pins `pytorch-lightning 1.6.5`, a
`transformers` fork and `pydantic 1`, which conflict with this environment, and its loader is
fifty lines over `transformers`. `genslm_assets/` therefore carries the MIT-licensed tokenizer
(`codon_wordlevel_69vocab.json`: 64 codons plus `[UNK] [CLS] [SEP] [PAD] [MASK]`, no
normalizer, whitespace pre-tokenizer) and the four GPT-NeoX configs, with `provenance.json`
pinning the upstream commit (`6622c47e`) and every sha256; the test
[[tests.torchcell.models.test_models_genslm]] re-hashes them.

| model id | hidden | layers | checkpoint file (Globus) |
|---|---|---|---|
| `genslm_25M_patric` | 512 | 8 | `patric_25m_epoch01-val_loss_0.57_bias_removed.pt` |
| `genslm_250M_patric` | 1840 | 12 | `patric_250m_epoch00_val_loss_0.48_attention_removed.pt` |
| `genslm_2.5B_patric` | 3840 | 28 | `patric_2.5b_epoch00_val_los_0.29_bias_removed.pt` |
| `genslm_25B_patric` | 8196 | 64 | `model-epoch00-val_loss0.70-v2.pt` |

**Weights.** Released only through the authors' Globus endpoint
(`25918ad0-2a4e-4f37-bcfc-8183b19c3150`); not on Hugging Face (the only `genslm` hit there is a
third-party PepMLM fine-tune). `$DATA_ROOT/models/genslm/manifest.json` records each checkpoint's
retrieval (`RetrievalMethod.globus`, added for this) and sha256, written by
[[scripts.genslm_fetch_weights]]; the loader verifies the hash before reading the file and
refuses a checkpoint with a missing parameter or an unexpected key (only the derived rotary and
causal-mask buffers may be absent, which is what the upstream `strict=False` load was for).

**Embedding.** Last hidden layer. `mean_embedding=True` averages over the real codon
positions only; the README's recipe averages `hidden_states[-1]` over the padded length, which
dilutes a short gene by the batch's padding. Tokenization refuses a CDS whose length is not a
multiple of 3 and any non-ACGT character (the word-level vocabulary would map that codon to
`[UNK]`, which is not a measurement of the gene).

Status 2026.10.07: code and tests landed; no checkpoint on disk yet (Globus login is a by-hand
step), so no real embedding has been computed.
