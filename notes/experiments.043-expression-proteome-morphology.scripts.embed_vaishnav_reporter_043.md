---
id: 5i3hvmb6xm76uucevd3ssld
title: Embed_vaishnav_reporter_043
desc: ''
updated: 1792637963406
created: 1792637963406
---

## 2026.10.21 - Vaishnav windows and the reporter into the allele tables

Script: `experiments/043-expression-proteome-morphology/scripts/embed_vaishnav_reporter_043.py`; GPU launcher [[experiments.043-expression-proteome-morphology.scripts.gh_embed_vaishnav_reporter_043]].

The reporter token of the primer (Eq. input-reporter) reads FUDT upstream on the 1,003 bp window ending at the reporter start codon, ProtT5 on the reporter protein and CaLM on its CDS. All of them go into the same `alleles/<encoder>.h5` tables as the isolate alleles ([[experiments.043-expression-proteome-morphology.scripts.compute_isolate_composite]]), keyed by the sha256 of the string each encoder reads, so a designed promoter and a native promoter share one function and one table. The batch path, the S288C reference check and fp16 storage are imported from `compute_isolate_composite.py`, which is not edited (job 6106 reads it at run time).

Plan and state:

- `--stage inputs` (CPU, ran 2026-10-21 in the session, 1.1 s): rebuilt all 71,158 upstream windows from the Addgene 127546 contig with `vaishnav_promoter_window.py`'s functions; every sha256 equals `results/vaishnav_promoter_windows.parquet`. One distinct downstream window. Reporter (SnapGene label `yeGFP`, depositor name yEVenus): 717 bp CDS, 239 aa protein with the trailing `*` (the ProtT5 convention), CaLM input = the CDS (under 3,072 nt). Written to `$DATA_ROOT/data/scerevisiae/isolate_embeddings/peter2018_1011/sequences/vaishnav2022_inputs.parquet` (71,161 rows: 71,158 FUDT upstream, 1 each for the other three encoders) and `vaishnav2022_inputs.json`.
- `--stage embed` + `--stage verify` (GPU): job 6115, submitted `--dependency=afterany:6106` so only one 043 GPU job holds a card and no two jobs append to one h5 file. Writes `results/embed_vaishnav_reporter_043.json`. Not run yet at the time of writing.
- A CPU smoke check of the batch path on two inputs (FUDT upstream and CaLM) returned (2, 768) and (1, 768); no reference check was run on CPU.

Keys the representation resolver uses (`representation_043.py`): the upstream window sha256 per `promoter_variant_id`, the constant downstream window `bc9d41ec...`, protein `04f4103e...`, CDS/CaLM `ad9d2ad1...`.
