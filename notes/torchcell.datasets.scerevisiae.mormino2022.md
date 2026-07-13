---
id: fh5h3nghpjiozvmzt7ccd5n
title: Mormino2022
desc: ''
updated: 1783922212215
created: 1783922212215
---

## 2026.07.13 - Mormino 2022 CRISPRi acetic-acid biosensor screen (12 strains)

`CrispriMormino2022Dataset` -- first CRISPR **interference** dataset (Lian was the first
CRISPR dataset overall). 12 records = Table 1 "Properties of isolated strains".

- **Genotype**: `CrisprInterferencePerturbation(target, effector="dCas9-Mxi1", guide=None)`
  (guides live upstream in the Smith 2017 CRISPRi library; not released here), BY4742.
- **Environment**: acetic acid 50 mM, pH 3.5, 30 C, aerobic (screen/biosensor condition).
- **Phenotype**: `EnvironmentResponsePhenotype` categorical -- Haa1-biosensor RFP `+`
  (enhanced -> more sensitive) -> `sensitive` (QCR8/TIF34/MSN5/PAP1/COX10/TRA1); `=` ->
  `no_effect` (NDC1/CBP2/UBA2/RPS30B/HSH49/LCB1). Reference = CC23 control (`no_effect`).

**Deliberately NOT built** (documented, not guessed): genome-wide enrichment is figure-only
(bar charts) -> not ingested; Table 1 Growth column ambiguous -> not stored; `n_samples` None
(qualitative summary call); temperature 30 C = standard (not stated for these isolates) --
flagged for review. Source = sha256-pinned mirror `paper.pdf` (388f8e92...), Table 1 embedded
literal (no SI data file released). L0-L4 all pass (12 records, containment 1.000). Scout that
established figure-only status: memory `[[remaining-datasets-blocked-status]]`.
