---
id: tlaznu45ttiavhuqifsee7q
title: Poc_isolate_fudt_feasibility
desc: ''
updated: 1784705236846
created: 1784705236846
---

## 2026.07.22 - WS9b Per-Isolate FUDT Feasibility (CDS -> Assembly Alignment)

WS9 delivered per-isolate ESM2 from the Peter gene-keyed CDS store but STUBBED FUDT
(SpeciesLM 5'/3' promoter+terminator windows) because that store holds only the CDS body,
not the flanking genomic windows FUDT needs (5' = 1000 bp upstream + start codon = 1003 bp;
3' = stop codon + 297 bp downstream = 300 bp). This PoC proves those flanks can be recovered
per-isolate by aligning each isolate's own CDS back to its own de-novo assembly
(`1011Assemblies.tar.gz`) and slicing the flank from the located contig.

- Script: `experiments/embeddings/poc_isolate_fudt_feasibility.py`
- Wired path: `experiments/embeddings/compute_isolate_embeddings.py::compute_isolate_fudt`
  (replaces the old `NotImplementedError` stub; run via `--with-fudt`).
- Results: `experiments/embeddings/results/poc_isolate_fudt_feasibility.json`
- Aligner: BLASTN (BLAST+ 2.17.0). Not on the base `torchcell` PATH; resolved via
  `$TC_BLASTN` / `$TC_MAKEBLASTDB` (used `biomni_e1` env here).

### Stores (mirror, sha256-pinned by the loaders)

- CDS: `$DATA_ROOT/torchcell-library/peterGenomeEvolution10112018/data/allReferenceGenesWithSNPsAndIndelsInferred.tar.gz`
  (6015 gene-keyed FASTAs, 1011 records each; isolate key = `SACE_<KEY>_<GENE>` or `<KEY>_<GENE>`).
- Assembly: `.../1011Assemblies.tar.gz` (1011 de-novo FASTAs, `GENOMES_ASSEMBLED/<KEY>[_N].re.fa`,
  N50 ~130 kb, ~4900 contigs/isolate). Isolate key -> assembly member is cached to
  `<tar>.member_index.tsv`; 1010/1011 CDS keys map (only `CRL` unmatched).

### Findings (3 isolates x first 40 systematic genes)

| isolate | mapped | mean %id | mean cov | full-len 100% | flank-trunc rate | 3' unembeddable |
|---------|--------|----------|----------|---------------|------------------|-----------------|
| AAA | 40/40 | 99.92 | 0.997 | 38/40 | 2.5% | 0 |
| AAB | 40/40 | 99.92 | 0.997 | 38/40 | 2.5% | 0 |
| AAC | 40/40 | 99.60 | 0.973 | 5/40  | 6.25% | 1 |

- **Alignment is clean/reliable.** CDS->own-assembly BLASTN is ~100% identity, full-length,
  e=0, with a huge unique-hit bitscore margin (median ~2340 over the runner-up) for AAA/AAB.
- **Contig-end / insufficient-flank rate is low** (2.5-6.25% of windows truncated). A
  truncated 5' window still embeds (SpeciesLM upstream pads undersize); only a 3' window
  < 12 bp is un-embeddable (1 case in AAC) -> reference-fallback.
- **Sample caveat (worst case):** the first 40 systematic genes are the chr-I LEFT-ARM
  SUBTELOMERE (YAL...), enriched for paralog/repeat families (PAU/seripauperins etc). That
  is why AAC shows 22 tied runner-up hits (margin 0) + low full-length count -- ambiguous
  placement in a repeat region. A genome-wide random gene sample would show far fewer ties.
  Production must apply a coverage + unique-margin filter and reference-fallback ambiguous
  genes (`allow_reference_fallback` hook + `_reference_window`; needs a real
  `SCerevisiaeGenome` wired for the fallback).
- **Round-trip confirmed:** sliced windows embed with `FungalUpDownTransformer` to
  `(1, 768)`; the wired CLI (`--with-fudt --skip-esm2`) writes `fudt_species_upstream.pt` /
  `fudt_species_downstream.pt` as collated `(data, slices)` datasets (20 records, embeddings
  keyed `species_upstream`/`species_downstream`, `fudt_source="assembly"`), same shape as
  `FungalUpDownTransformerDataset`.

### Compute cost + verdict

- FUDT embedding dominates: upstream ~6 it/s, downstream ~19 it/s (small SpeciesLM, CPU) ->
  ~22 min/isolate for 6015 genes. BLAST is negligible (~seconds/isolate). makeblastdb ~0.15s.
- Full 943 isolates: ~350 h single-thread, but embarrassingly parallel across isolates
  (and GPU-acceleratable) -> ~1-2 days on a modest worker pool. Much cheaper than the WS9
  ESM2 pass (650M model).
- **Verdict: per-isolate FUDT is FEASIBLE for the conference.** Remaining work: (1) validate
  mapping quality on a genome-wide (non-subtelomeric) gene sample and finalize
  coverage/margin thresholds; (2) wire the `SCerevisiaeGenome` reference-fallback for
  ambiguous/edge genes; (3) provision BLAST+ in the run env; (4) extract all 1011 assemblies
  once, then batch the run with per-isolate parallelism.
