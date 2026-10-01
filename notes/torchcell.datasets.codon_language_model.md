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

## 2026.09.30 - Embed and store the spliced CDS; chunk file (issue #543)

- CaLM tokenizes codons (`calm.sequence.CodonSequence`), so it expects a CDS. Every gene now embeds its CDS FASTA record truncated to the first 3,072 nt (in frame from the start codon), and `dna_windows["calm"]` stores exactly that string. Previously a gene whose locus was at most 3,072 nt embedded the CDS but stored `window(len(cds))` on the locus (for a spliced gene, genomic sequence that was not embedded), and a longer locus embedded `window(3072)` of the locus, which for a spliced gene includes intron sequence. `MODEL_TO_WINDOW["calm"]` is now `("cds", 3072)`; the dead `is_max_size` flag is gone.
- Chunks go to `processed/calm.partial.pt`; a chunk file left by an interrupted build is removed with a logged warning and the build restarts at the first gene, instead of blocking the next construction.

A rebuilt store differs on disk: `dna_windows["calm"]` is a string for every gene (was a `DnaWindowResult`), and the embedding differs for spliced genes whose locus exceeds 3,072 nt. Tests: [[tests.torchcell.datasets.test_codon_language_model]].
