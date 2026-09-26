---
id: dj6vf37mbtz9xcic101oewu
title: Avalos Data Request Draft
desc: ''
updated: 1790411954010
created: 1790411954010
---

## 2026.09.26 - Draft email to the Avalos lab

**Status: DRAFT, not sent.** Review and edit before sending. Recipients and any prior
correspondence context need to be filled in by hand. The ask is narrow on purpose: the
per-strain numbers behind figures whose genotypes are already released, not a
collaboration or a data dump.

Who to address, from the dissertation and the biosensor paper: Jose L. Avalos (PI,
Princeton, Chemical and Biological Engineering), Jose de Jesus Montano Lopez (dissertation
author), Leopoldo Duran (co-author on the biosensor paper). Confirm current affiliations
before sending, since the dissertation author has likely moved on from Princeton.

### Subject

Request: per-strain isobutanol values behind the 2024 dissertation figures

### Body

Dear Professor Avalos,

I am a PhD student in Huimin Zhao's group at the University of Illinois Urbana-Champaign.
I am building TorchCell, an open database and modeling framework that stores yeast
experiments as typed genotype-by-environment-to-phenotype records, each one traceable to a
hash-pinned source file and a verbatim quote from the paper or supplement it came from. It
currently holds 51 datasets covering roughly 100 million records.

Two of those datasets come from Jose de Jesus Montano Lopez's 2024 dissertation. We ingested
the genome-wide branched-chain-amino-acid biosensor screen of the knockout collection,
Supplementary Tables S2 and S3, as 4,554 single-deletion records plus the 224 validated
re-screened strains, with the fold change stored as a median-GFP ratio against the
same-plate wild type. We cite the 2022 Nature Communications biosensor paper for the
construct and methods. The screen has been valuable, and it is the only isobutanol-related
genotype-phenotype data in our store.

I am writing about the part of that work we cannot use yet. We are training a model to
predict isobutanol tolerance from gene perturbations, and the strongest available test of
such a model is not another screen but engineered strains whose measured behavior is known
in advance. Your dissertation contains exactly that, and in two forms whose genotypes are
fully specified in Supplementary Table 1 while the measured values appear only in figures:

1. The 23 strains combining a single mitochondrial-morphology deletion (MDM36, MDM35,
   MDM32, TOM7, FIS1, DNM1, MGM1, FZO1, NUM1, MMM1 and others) with the five-cassette
   mitochondrial isobutanol pathway, and their isobutanol titers. These appear against
   mitochondria number and volume in Supplementary Figures 3 to 5.

2. The gln3 gcn4 and gln3 gnp1 double deletions and their tolerance factors, alongside the
   corresponding single deletions. The statement that the double is not additive over the
   singles is the specific claim we would most like to test a model against, because
   predicting it requires an interaction rather than a ranking.

Would you be willing to share the per-strain numbers behind those figures, in whatever form
they already exist, a spreadsheet or a plain table with one row per strain? We do not need
raw instrument files, replicate-level traces, or anything unpublished beyond these values.
Mean and standard deviation per strain per condition, with the replicate count and the
units, would be enough.

On how we would handle it. Every value we store records its source, so the provenance would
name the dissertation and, for these numbers, a direct communication from your group, with
the date. We can hold the records private and excluded from the public graph for as long as
you prefer, including indefinitely, and we would not publish the values themselves. If you
would rather these numbers stay with your group entirely, an equally useful alternative is
for us to send you our model's predictions for those strains first, in writing and
timestamped, and for someone in your group to tell us only whether the predicted ordering
and the predicted sub-additivity are right. That keeps your data with you and still gives
us the test.

I would also welcome a pointer if any of this work has since been published, or is in
preparation, in a form that releases the values. We would rather cite and ingest a paper
than a dissertation, and our records currently carry a flag noting that the dissertation has
no DOI.

Thank you for considering this, and for the biosensor screen, which is already doing work in
our database.

With best regards,

Michael Volk
PhD candidate, Zhao Laboratory
Department of Chemical and Biomolecular Engineering
University of Illinois Urbana-Champaign

### Notes for the sender

- **Verify before sending:** that the dissertation chapters have not since been published
  with released data. A literature check is running; if a paper exists with a Source Data
  file, retrieve that instead and drop points 1 and 2 of the ask.
- **Affiliation line** is written from the repository's own records. Correct it if the
  department name or title is wrong.
- **The blinded-prediction offer in the fifth paragraph is the fallback that costs them
  least.** Keep it. It converts a data request into a collaboration invitation and it is
  genuinely as useful for validation purposes.
- **Do not attach anything.** If they ask what the database looks like, the served Neo4j
  browser link in the repository README is the right follow-up.
- Consider whether to copy Huimin Zhao before sending, since the framing leads with the
  group affiliation.
