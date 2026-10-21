---
id: pcpsm668o1jlcyfljrpwszs
title: Pputida Cell State
desc: ''
updated: 1792613140569
created: 1792613140569
---

## 2026.10.21 - Survey of the served P. putida alcohol and cell-state data

Working notes for the typeset document at
`notes-tex/database/pputida-cell-state/`, built from
`experiments/database/scripts/build_pputida_cell_state_table.py`.

### The question

Which of the 32 served *P. putida* datasets in KG 4.0 carry an alcohol growth or
tolerance arm, or a perturbation of cell state, and what does each one's
`genotype x environment -> phenotype` record look like. Then: which single
non-Carruthers dataset helps predict the last DBTL round of the Carruthers isoprenol
campaign.

### Findings

- **No served *P. putida* isobutanol data.** Zero hits across the 15 loader modules
  under `torchcell/datasets/pputida/`. The two isobutanol datasets in the release are
  Lopez 2024, which is *S. cerevisiae*. The Fitness Browser deposit carries no
  isobutanol experiment for any organism.
- **Fifteen rows qualify**, and the two transposon compendia hold 99.5 percent of the
  records. Scale is a cost estimate, not a ranking: the engineering rows with single
  digits of records carry the phenotype the strain-design work targets.
- **Catabolism is not tolerance.** The compendium alcohol arms feed the alcohol as the
  sole carbon source at 10 mM. Lim 2022 and Lim 2025 apply one as a stressor. A query
  filtering on the compound alone merges the two.
- **`IsoprenolTiterYunus2026` is the transfer dataset.** Same chassis IY1452, same
  CRISPRi modality, Carruthers is a co-author, and all but one of its CRISPRi targets
  are also Carruthers targets. Same product, unit, quantification method, replicate
  structure and duration. Conditions differ (24 C in M9-NREL with 20 g/L glucose
  against 30 C in plain M9 salts), so it is a held-out condition and a reproducibility
  bound on the label, not extra training rows for absolute titer.
- **Blocker: the DBTL cycle label is not in the served graph.** A permissive regex
  over the full serialization of every record in all four Carruthers stores finds no
  DBTL token, and `screen_id` is empty throughout. The loader's record identity is
  per (construct, DBTL cycle), so the cycle is implied and never materialized.
  Recoverable from the raw mirror's `DBTL_Cycle` column by joining on the construct
  name. Worth filing as a loader gap.

### Probe notes, for anyone rerunning it

- LMDB values are **pickled**, not JSON. Reading one as UTF-8 fails on the pickle
  protocol byte.
- `Environment`, media and compounds are **interned** in a sibling
  `processed/interned/` LMDB; a record carries `{"$ref": <sha256>, "name": <hint>}`.
  Splice before touching any field.
- Distinct environments are counted **by value over a full scan**, not off the
  interned table: that table holds every interned constant and nothing guarantees an
  entry is still referenced. Costs a scan of 2.7 million records; the result is
  cached in `experiments/database/results/pputida_cell_state_probe.json`.
- `genome_reference` lives on `rec["reference"]`, not on the experiment. The
  experiment-level field is null in all fifteen datasets, by design.
- Every row pins the same assembly, `GCA_000007565.2`, but the **strain varies**:
  KT2440 for the screens, IY1449b, IPL400 and CJ522 for the production and tolerance
  work. The engineered backgrounds carry their cassettes on the genotype, not in the
  assembly.

### Corrections made while surveying

- The Fitness Browser row does **not** add a stress panel or methanol arms over the
  compendium. Those samples are all `VillalobosEscobedo23`, which the loader refuses
  because every sample is on R2A and the media library has no entry for it.
- The Borchert 2024 **reactor arms are in no served dataset**. All 22 bioreactor and
  adaptation samples are dropped by `reactor_process_not_representable`.
- The dissolved-oxygen axis does **not** need a schema extension. `Environment`
  already carries `dissolved_oxygen` as a physical factor, plus `duration_hours`,
  `duration_generations` and `dilution_rate_per_hour`. Only the solvent overlay has
  no representation, which affects 3 of the 22 dropped samples.
