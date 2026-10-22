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
- **Seventeen of the 32 served datasets carry isoprenol, in four roles.** Matched on
  the InChIKey `CPJRRXSHAYUTGL`, not the name. Five measure it as a product
  (Carruthers 502, Yunus 125, Kang 7 of 26, Banerjee 2024 20, de Siqueira 10); two
  measure growth **on** it as the sole carbon source (Borchert 2024 9,464 and Fitness
  Browser 9,556 records at 10 mM); three carry it as a stressor (Lim 2025); seven
  carry the pathway on the genotype while measuring something else.
- **The carbon-source arm was the miss.** It is the largest isoprenol block by three
  orders of magnitude and it names the catabolism genes, which is the deletion
  target for stopping a producer from eating its own product. Easy to overlook
  because a cell-state survey filters it out and a product-name search finds it only
  if you also search the environment, not just the phenotype.
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

## 2026.10.21 - Gene-level correlates of the Carruthers titer effect

Generated by `--probe-correlates` in
`experiments/database/scripts/build_pputida_cell_state_table.py`, rendered to
`tables/correlates.tex` (Tables 5 and 6 of the PDF), discussed in Sec. 6.4.

- **The effect.** Carruthers single-knockdown titer, 123 strains over 121 targets. Joined
  to the raw mirror's `DBTL_Cycle` by target set, 119 are DBTL0 and 4 match no cycle or
  several, so it is a within-round estimate. Reliability 0.985: mean released SE
  7.0 ug/mL against a strain-mean SD of 73.7 ug/mL. A secondary additive ridge fit over
  all 502 strains has no cycle term and is a robustness check only.
- **No served gene-level readout tracks it.** Spearman over shared targets, with a
  best-of-many permutation null (2,000 shuffles, seed 0) for datasets offering many
  vectors.
  - Yunus 2026 titer: rho -0.10, p 0.28, n 120. Top-quartile overlap 8 of 30 against 7.5
    by chance. Median titer 858 against 163 ug/mL. Its knockdown depth: rho +0.05.
  - Compendia growth on isoprenol (set16IT008, set16IT009): rho -0.06 to +0.01, n 77.
  - Best of 234 / 310 compendium experiments: rho +0.28, best-of p 0.76 / 0.80.
  - Best of 321 PRECISE321 samples: rho +0.25, best-of p 0.09.
  - Lim 2025 proteome fold change: rho +0.27, best-of p 0.028 over 4 lineages; +0.18
    against the additive fit. A lead, not a result.
- **Weak prior, does not stand.** Share of compendium experiments with |fitness| > 1:
  rho +0.27 / +0.24, nominal p 0.02 / 0.03, uncorrected, the two compendia share most
  experiments, and it falls below +0.10 against the additive fit. Hypothesis
  (untested): knocking down a growth-limiting gene diverts carbon to product.
- **Correction to the section above.** Carruthers and Yunus do NOT share a chassis. The
  served records pin IY1449b (Carruthers) and IY1452 (Yunus), and the Yunus loader
  refuses to borrow the IY1449b genotype for IY1452. The two-campaign table now carries
  a generated chassis row. So "Yunus is the transfer dataset" is withdrawn: it overlaps
  most, and its ranking does not transfer.
