---
id: 3gko92s7mz1c8pj0tnrfq3i
title: Codon_language_model
desc: ''
updated: 1711856060199
created: 1711856034988
---
> The model described in the document you provided, referred to as CaLM (Codon Adaptation Language Model), was trained on a dataset consisting of 9 million non-redundant and diverse cDNA sequences. It is capable of handling sequences trimmed to a maximum size of 1024 tokens, a number empirically found to enable efficient learning while preserving computational efficiency. This maximum sequence length of 1024 tokens is critical for understanding the maximum length of DNA sequence that can be effectively used with this model. Given that each codon consists of three nucleotides and a token in this model represents a codon, the maximum length of a DNA sequence that can be used in this model is 3072 nucleotides (1024 tokens * 3 nucleotides per token)

#ChatGPT