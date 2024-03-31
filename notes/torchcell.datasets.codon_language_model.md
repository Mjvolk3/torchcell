---
id: 3gko92s7mz1c8pj0tnrfq3i
title: Codon_language_model
desc: ''
updated: 1711909924803
created: 1711856034988
---

## 2024.03.30 - CaLM Model Input Description

> The model described in the document you provided, referred to as CaLM (Codon Adaptation Language Model), was trained on a dataset consisting of 9 million non-redundant and diverse cDNA sequences. It is capable of handling sequences trimmed to a maximum size of 1024 tokens, a number empirically found to enable efficient learning while preserving computational efficiency. This maximum sequence length of 1024 tokens is critical for understanding the maximum length of DNA sequence that can be effectively used with this model. Given that each codon consists of three nucleotides and a token in this model represents a codon, the maximum length of a DNA sequence that can be used in this model is 3072 nucleotides (1024 tokens * 3 nucleotides per token)

#ChatGPT

## 2024.03.31 - Overcoming Semaphore Error when Processing CalM Dataset

If we try to add all of the embeddings to list for all genes in the gene set we get as semaphore error at about gene 180. To over come this we write the data to disk in chunks, then at the at the end we put everything into standard format with with a collate then overwrite the last `calm.pt`. This is visually displeasing and a bit hard to track what is going on but there is a reason for the madness. Of course any better way to overcome the semaphore error is welcome.
