---
id: jffic0557nr6xau9wzpbtud
title: spell-publications
desc: ''
updated: 1791255603711
created: 1791255603711
---

## 2026.10.05

- [x] Typeset list of all 603 SPELL publications with DOI, PubMed, PMC and GEO links, the 48 studies that add a perturbed genotype beyond Kemmeren and Sameith ranked first, and the 88 studies with no GEO accession set apart; published to Zotero [[experiments.015-spell.publications]]
- [x] Manifest of the SPELL archive with DOIs resolved from PubMed, so each paper can be retrieved and run through the dataset addition process [[experiments.015-spell.scripts.spell_publication_manifest]]
- [x] Deletion coverage of SPELL estimated from column headers: 685 deleted genes, 788 genotypes, only 90 in neither Kemmeren nor Sameith, so SPELL adds replication rather than coverage [[experiments.015-spell.scripts.spell_knockout_coverage]]
- [x] Costanzo 2016 and Kuzmin 2020 temperature-sensitive and DAmP allele lists, with their reuse in SPELL headers (8 alleles, 9 studies) [[experiments.015-spell.scripts.spell_allele_reuse]]
- [x] Retrieved everything SGD publishes in its expression download area with a sha256 provenance manifest, closing the gap that the original tarball was downloaded with no record [[experiments.015-spell.scripts.spell_archive_retrieve]]
- [x] `make check` skips `\sourcetext{}` so generated citations keep their source spelling [[notes-tex.common.check_doc]]
- [ ] Assign the inclusion status (probabilistic genome representation, "not possible for now") per SPELL study and add the allele, imputation and provenance findings to the notes-tex document [[experiments.015-spell.publications]]
- [ ] Verify the Costanzo strain table against the loader's copy on gilahyper and cover Kuzmin 2018 in the allele list [[experiments.015-spell.scripts.spell_allele_reuse]]
- [x] YEASTRACT+ 2022 regulation file traced to its 1,671 source papers with DOIs, cross-checked against SPELL by PMID (84 shared; Reimand 2010 is a reanalysis of Hu 2007), and typeset so the papers worth tracing to source data can be chosen [[experiments.037-yeastract.publications]]
