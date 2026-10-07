---
id: 09oienz02d3dk7gphhmtmrv
title: Genslm
desc: ''
updated: 1791416303122
created: 1791416303122
---

## 2026.10.07 - GenSLM embedding dataset

`GenSLMDataset(BaseEmbeddingDataset)` in `torchcell/datasets/genslm.py`, the codon-level sibling
of [[torchcell.datasets.codon_language_model]] (CaLM) built on [[torchcell.models.genslm]].

**Contract.** Each gene embeds its spliced CDS (`gene.cds`), in frame from the start codon,
truncated to the first 6,144 nt (2,048 codons, the model context) when longer; `dna_windows`
stores exactly the string embedded and `embeddings[model_name]` the masked mean of the last
hidden layer. The genome is any object with `gene_set` and `__getitem__` whose genes expose
`cds`: `SCerevisiaeGenome` and every `BacterialGenome` (KT2440, MG1655, BW25113, REL606)
qualify, so one class serves the bacterial hosts the model was pretrained on and yeast (out of
distribution for a prokaryotic model; CaLM is the yeast codon model).

**RNA genes.** A gene whose `cds` is `None` has no codon representation and is left out of the
store; its id goes to `processed/<model_name>.no_cds.json`. Measured on KT2440
(`PPutidaKT2440Genome`, dev cache, 2026.10.07): 5,729 genes, 5,564 with a CDS (every length a
multiple of 3), 165 without (16S, 23S, 5S rRNA and tRNA loci), 8 CDS longer than 6,144 nt
(truncated).

**Build.** The backbone loads inside `process()` only when the store is absent, so a store on
disk opens on an offline node without the checkpoint. Batches pad to the longest sequence
(`batch_size` sequences per forward). The `__main__` block builds KT2440 at
`$DATA_ROOT/data/pputida/kt2440/genslm_embedding`.

Status 2026.10.07: tests [[tests.torchcell.datasets.test_genslm]] pass with a faked backbone;
no store has been built because the checkpoints are not yet fetched (see
[[scripts.genslm_fetch_weights]]). Not yet registered in `NodeEmbeddingBuilder`, whose root
paths are yeast-specific.
