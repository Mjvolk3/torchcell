---
id: 3yyjavfxon5ad96i98j2xlf
title: Embed_media_components
desc: ''
updated: 1790481666187
created: 1790481666187
---

## 2026.09.26 - Media components embedded, and tunicamycin stays a typed gap

Closes the two molecules-not-captured gaps. Built by a background agent; route and numbers
verified in its report.

**Route.** Which media were served comes from the distinct `media_name` and `ref_media_name`
values of the flattened parquets, so a medium appearing only as a reference still counts. What
each medium contains comes from the `Media` constants in `torchcell/datamodels/media.py`,
indexed by name. The loaders pass those constants in by reference, so the LMDB objects are
those objects. The join is exact: all 23 served media names matched the 54-entry library index
with none unmatched.

**Coverage.** 39 distinct components across the three datasets, 35 embeddable. All 12 encoders
embedded the 37-key union with zero failures, including Uni-Mol, which rejects 11 ionic salts
in the dosed set but nothing here. The four non-embeddable components are honest definition
classes rather than failures: the SynH3 hydrolysate base and commercial yeast nitrogen base are
`composition_deferred` sub-mixes, and peptone and yeast extract are `intrinsically_undefined`
biological digests that no recipe pins.

The media axis is nearly disjoint from the dosed axis. It shares only 3 of 37 InChIKeys with
the 343 dosed compounds, those being 4-aminobenzoic acid, acetamide and sodium acetate, so this
is new chemical space rather than a re-embedding.

**One curation gap surfaced.** Cellobiose, a SynBase dropout, carries a typed `inchikey` gap
because it appears in no curation input list. That is a missing table row, not a property of
the substance, and it is actionable.

**Tunicamycin stays uncaptured, and that is the sourced answer.** It appears in 22,944 of the
2,698,797 heterozygous records and is the only compound with no InChIKey in any of the three
datasets. Its identity record carries `resolution_status: RESOLVED_MIXTURE` and
`chebi_id: CHEBI:29699`, which ChEBI defines as a mixture of at least 10 homologues, and
PubChem returns no CID for the name. The mirrored paper and its supplement never mention
tunicamycin, the raw column header gives only name and dose, and no homologue is curated
anywhere. Representing it by a single homologue or by a homologue mean would require structures
that exist in no source on disk, so both options mean inventing an identifier. The recommended
change is to promote it from "not a compound" to a counted `resolved_mixture` reason row in the
dosed-compound accounting, so it is a visible typed gap rather than a printed aside. What would
settle it is a methods or vendor statement giving the catalog number of the preparation used in
the screens, which the mirror does not hold.

Result files: `results/media_components.csv`, `results/media_component_coverage.csv`,
`results/embeddings_media/` (npz per encoder, gitignored, plus `failures.json`)
