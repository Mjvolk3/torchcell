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

I am writing about three sets of values we cannot use yet. We are training a model to
predict isobutanol tolerance and production from gene perturbations, and the strongest test
of such a model is not another screen but engineered strains whose measured behavior is
already known. Your group's work contains exactly that. In each case below the genotypes are
released and machine-readable while the measured values appear only in figures.

1. **The 2022 Nature Communications biosensor paper (10.1038/s41467-021-27852-x).** Its Data
   Availability statement and its figure captions say source data are provided as a Source
   Data file, but no such file appears with the article or in the PubMed Central package for
   PMC8755756, and the three deposited supplements are all PDFs. The Supplementary
   Information does give allele-level genotypes for the Ilv6p and Leu4p variant sets, the
   Ll_IlvD variants, and the FACS-isolated high producers with their cassette copy numbers.
   If the Source Data file exists and simply was not deposited, that would be the single
   most useful thing you could send: roughly fifty typed genotypes with paired isobutanol
   and isopentanol measurements.

2. **The mitochondrial morphology strains**, which I understand from the 2025 review in
   IJMS are in review as Montano Lopez et al. The 23 strains combining one
   mitochondrial-morphology deletion (MDM36, MDM35, MDM32, TOM7, FIS1, DNM1, MGM1, FZO1,
   NUM1, MMM1 and others) with the five-cassette mitochondrial isobutanol pathway, and their
   titers. We are glad to wait for publication, and equally glad to receive them now under
   any embargo you prefer.

3. **The gln3 gcn4 and gln3 gnp1 double deletions** and their tolerance factors, alongside
   the corresponding single deletions. That the double is not additive over the singles is
   the specific claim we would most like to test a model against, because predicting it
   requires an interaction rather than a ranking.

Would you be willing to share the per-strain numbers behind any of these, in whatever form
they already exist, a spreadsheet or a plain table with one row per strain? We do not need
raw instrument files or replicate-level traces. Mean and standard deviation per strain per
condition, with the replicate count and the units, would be enough.

On how we would handle it. Every value we store records its source, so the provenance would
name the dissertation and, for these numbers, a direct communication from your group, with
the date. We can hold the records private and excluded from the public graph for as long as
you prefer, including indefinitely, and we would not publish the values themselves. If you
would rather these numbers stay with your group entirely, an equally useful alternative is
for us to send you our model's predictions for those strains first, in writing and
timestamped, and for someone in your group to tell us only whether the predicted ordering
and the predicted sub-additivity are right. That keeps your data with you and still gives
us the test.

One more note, offered as a small thank you rather than a request. Your 2019 Cell Systems
paper with Kuroda and colleagues released its full initial-screen table, and it is the best
quantitative isobutanol tolerance dataset we have found anywhere: tolerance factors for
about 4,380 deletion strains with the underlying optical densities at zero and 1.4 percent
isobutanol. We are ingesting it. Anything at that level of release, for any of the work
above, is immediately usable by us and by anyone else building on it.

Thank you for considering this, and for the biosensor screen, which is already doing work in
our database.

With best regards,

Michael Volk
PhD candidate, Zhao Laboratory
Department of Chemical and Biomolecular Engineering
University of Illinois Urbana-Champaign

### Notes for the sender

- **Publication status is now checked** (2026.09.26, exhaustive over PubMed and Europe PMC
  preprints). The GLN3 tolerance chapter is published as Kuroda et al. Cell Systems 2019,
  PMID 31734159, and its screen table IS released. The `SPT10` chapter and the
  double-deletion plus evolution chapter are unpublished with no preprint and no sequence
  deposit. The mitochondrial morphology chapter is in review, cited as reference 17 of
  Kichuk and Avalos, IJMS 2025;26:2152. So point 2 of the ask should acknowledge the paper
  in review, which the current draft does.
- **Point 1 is the strongest item and the most legitimate**, because the 2022 paper committed
  in writing to releasing a Source Data file that was never deposited. Lead with it.
- **Affiliation line** is written from the repository's own records. Correct it if the
  department name or title is wrong.
- **The blinded-prediction offer in the fifth paragraph is the fallback that costs them
  least.** Keep it. It converts a data request into a collaboration invitation and it is
  genuinely as useful for validation purposes.
- **Do not attach anything.** If they ask what the database looks like, the served Neo4j
  browser link in the repository README is the right follow-up.
- Consider whether to copy Huimin Zhao before sending, since the framing leads with the
  group affiliation.
