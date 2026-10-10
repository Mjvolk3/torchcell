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

## 2026.10.10 - First real loads: 25M and 250M checkpoints

Both checkpoints were transferred from the Globus collection (`/models/25M/`, `/models/250M/`,
the 2023-05-03 re-release, not `legacy/`) and pinned in `$DATA_ROOT/models/genslm/manifest.json`
(sha256 `a21652c0...` and `b21a49e9...`, Globus task ids recorded).

**What the released checkpoints look like.** Lightning + DeepSpeed checkpoints (`state_dict`
beside `ds_config`, `hyper_parameters`, `global_step`, ...), parameters in float32. Every
parameter name matches `GPTNeoXForCausalLM` built from the vendored config: 100 of 100 (25M),
148 of 148 (250M). The only extra keys are per-layer `attention.masked_bias` (`-inf`) and
`attention.rotary_emb.inv_freq`, buffers that transformers 4.21 registered and 4.57 no longer
does; the stored rotary table is fp16-rounded (max relative difference 4e-4 from the recomputed
one), so recomputing is the correct choice. The key policy was therefore turned around from the
first draft: derived buffers may be absent or extra, and a missing or unexpected parameter still
refuses. This is what the upstream `strict=False` load silently covered.

**Sanity on real genes** (scratch probe, next-codon loss with `labels=input_ids`, KT2440 CDS):

| gene | codons | 25M perplexity | 250M perplexity |
|---|---|---|---|
| PP_0001 | 291 | 31.0 | 10.6 |
| PP_0002 | 264 | 18.6 | 7.1 |
| PP_1000 | 337 | 17.5 | 3.9 |
| PP_1000, codons shuffled | 337 | 49.1 | 53.4 |

Chance is 64. The shuffled-codon control keeps composition and destroys order, so the gap
is the model reading sequence, not codon usage. Embeddings are finite, mean norm 23 (25M) and
40 (250M) over the two probe sequences.
