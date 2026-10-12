---
id: l81sqbk73a5qc4wn8ctkpwm
title: Thermo_accession_sweep
desc: ''
updated: 1789109621531
created: 1789109621531
---

## 2026.09.11 - What this script is

Changes the stopping rule of the metabolite audit: try EVERY accession a metabolite carries
and accept the first that yields a finite formation energy, instead of the first that
resolves to a cache record at all. A compound whose heavy-atom composition disagrees with
the GEM formula is rejected rather than priced.

The candidate list comes from the eQuilibrator recomputation parquet, a fixed upstream
artifact the sweep never writes, so the sweep is idempotent: it is itself a pipeline stage,
and asking the pipeline what is unpriced would drop everything the sweep had priced on the
previous run.

Result: 12 of 530 candidates priced, including all three copies of hydrogen sulfide, whose
MetaNetX accession resolves to a compound with no energy while its BiGG and ChEBI accessions
resolve to `[SH-]` at 11.57 kJ/mol. Those three rows were blocking 9 reactions.

Findings: [[experiments.026-metabolism-flux.pka-method-selection]].
