---
id: 55160mu9r2ontn9xzxty4b7
title: Smith2016
desc: ''
updated: 1789274048457
created: 1789274048457
---

## 2026.09.12 - Serve-50 build: the vendor-code drop, SC-Ura, 20 doublings

`CrispriChemgenSmith2016Dataset` -- per-guide dCas9-Mxi1 CRISPRi chemical-genetic fitness,
one record per (pool x guide x drug-condition) row of Additional file 10. This is the note's
first entry; the loader existed since 2026-07-12 but had no note and no test file.

The 2026-07-12 build failed L0 structural on all 14,463 records (`Media` gained a required
`is_synthetic` after they were serialized), so a rebuild was mandatory and everything below
rides on it. Nothing was served, so nothing is updated in place.

### Records dropped: 7,410 of 14,463 (51.2%)

Ten of the nineteen drug labels are vendor catalog ids whose structure the primary never
released: `1181-0519`, `4130-1276` and `0KPI-0099` (ChemDiv), `9121982`, `9125678`,
`7312221`, `CBF-666774`, `6630449` and `9150499` (ChemBridge), `ST016598` (TimTec).
Additional file 8 gives the vendor and the dose but no SMILES, CAS or structure for any of
them, so `resolve_compound_identity` returns `PROPRIETARY` with no InChIKey, ChEBI id or
PubChem CID. Per the Hoepfner precedent those records are dropped rather than served under a
bare name, and the rule plus the per-compound counts are written to
`preprocess/dropped_records.json`. **Kept: 7,053.**

The other three retention rules fire zero times today and are kept as live gates: rows with
no `A`/`var(A)` (0 -- Additional file 10 is complete), unresolved target ORFs (0 -- all 20
are current R64) and guides with no released spacer (0 -- all 977 join Additional file 4).

`NSC-180973` is the one label that could have been lost and was not. Additional file 8
releases the alias `NSC-180973 (tamoxifen)`, and the pinned compound identity table carries
that synonym onto the tamoxifen row (`NKANXQFJJICGDU-QPLCGJKRSA-N`, CID 2733526), so its 456
records are kept. The old loader passed `known_proprietary=True` for every non-DMSO drug,
which would have mislabelled eight ordinary public compounds' gaps as terminal; the flag is
gone, because the table's own per-row `resolution_status` is authoritative.

### Encoding changes

- **Medium**: the shared `media.SC_URA` object replaces `Media(name="SCM-Ura")`. Smith
  releases no SCM-Ura recipe, so the library object's own component gaps are the honest
  state, and using it joins every other SC-Ura dataset.
- **`assay_type = pooled_competitive_growth_barcode`**, sourced from the Methods sentence
  that makes the spacer a barcode ("the short specificity-determining regions of gRNAs ...
  can act as unique identifiers of individual strains").
- **`duration_generations = 20.0`** ("This amounted to approximately 20 culture doublings
  from the beginning of the experiment"). Exposure duration is part of environment identity
  and the old build left it None. `duration_hours` is a typed `ProvenanceGap`: the cultures
  were held in log phase by repeated dilution and no wall time is reported.
- **Common names** now come from the genome's own standard name for the resolved ORF, via
  the shared `resolve_gene_name`.
- The record grain and the `library_pool` discriminator are unchanged: 272 (guide, drug,
  concentration) triples appear twice as the pool pair (`broad_tiling`,
  `gene_tiling_20bp`), and those are two independent pooled measurements, not replicates.

### `n_samples = 1` is a decision, not an oversight

The review flagged it as contradicting the released replicate structure (eight replicate
experiments for 1% DMSO, three for 20 uM fluconazole). It is kept at 1 deliberately. The
stored `var(A)` is the variance of the SINGLE released estimate: for those two conditions
the paper has ALREADY inverse-variance-combined the replicates into it ("We combined the R
replicates ... to obtain the variance"). `derive_se` computes `sqrt(var/n)`, so recording 8
would shrink the stored SE by `sqrt(8)` against a variance that is already the combined
estimate's. The replicate structure is recorded instead in the module-level
`REPLICATE_STRUCTURE` SourcedValue, where it cannot corrupt arithmetic.

### Two things deliberately NOT asserted

- **The ATc dose.** ATc induction is what turns the CRISPRi genotype on, but the pooled
  screen's Methods say only "grown +/- a drug listed in Additional file 8, and +/- ATc". The
  one released number, 250 ng/mL, is from the qPCR section and Additional file 5's
  sample-metadata header, neither of which is this screen. The induced-vs-uninduced
  definition rides in the phenotype `units` string and in the uninduced (-ATc) reference.
  `Environment` has no inducer field, so there is nothing to attach a `ProvenanceGap` to;
  this is flagged here rather than typed.
- **The solvent.** "Drugs were dissolved in DMSO" sits in the individual-strain growth-assay
  section and the pooled section does not restate it, so no `Solvent` is asserted on the
  pooled drugs. The 1% DMSO vehicle control is served as a compound in its own right.

The ~20%-growth-inhibition dose rule IS released (Additional file 8 ReadMe, sha256
`a22ef00ace51...`: "These concentrations were selected to inhibit growth of a wild-type
strain by ~20%"), but `DoseBasis` has no `IC20` member and adding one changes a class in all
36 served dataset closures, so the rule is recorded here and in the loader rather than in
`Concentration.basis`.

### Provenance

Raw mirror `$DATA_ROOT/torchcell-raw/smithQuantitativeCRISPRInterference2016/` holds exactly
the two files the loader consumes, `si/si_data/13059_2016_900_MOESM10_ESM.xlsx` (Additional
file 10, sha256 `02962e51...`) and `si/si_data/13059_2016_900_MOESM4_ESM.xlsx` (Additional
file 4, sha256 `e5eb4e3c...`), each with a `springer_esm` retrieval record. Both URLs were
re-retrieved on 2026-09-12 and reproduced the pinned hashes exactly, so the retrieval chain
is live rather than aspirational.

### Verifier result (verbatim, scratch runner with this entry's parameters)

```
crispri_chemgen_smith2016: PASS
  [ok] L0 structural: 7053 records validated
  [ok] L1 count: observed 7053, expected 7053
  [ok] L1 pair_uniqueness: 7053 unique (study, strain, condition) records, one each
  [ok] L1 provenance_gaps: 7053 documented provenance gaps over 7053/7053 records; 0 deferred field(s)
  [ok] L1 canonical_gene_names: 20 systematic names, one canonical spelling each, each current in the genome
  [ok] L2 value_fidelity: 7053 values checked
  [ok] L2 se_nonnegative: 7053 values checked
  [ok] L2 uncertainty_sanity: 7053 labelled uncertainties, none a zero dispersion
  [ok] L3 measurement_type_consistent: single measurement_type: MeasurementType.log2_ratio
  [ok] L3 reference_zero: numeric rule: reference response == 0 for all 7053 records
  [ok] L3 environment_perturbed: all 7053 experiments carry an environmental edit
  [ok] L3 compound_identity: 7053 compound references carry a structure identifier
  [ok] L3 media_compound_identity: 218643 compound references carry a structure identifier
  [ok] L3 media_membership: 7053 records on a shared MEDIA_LIBRARY medium (1 distinct media)
  [ok] L4 gene_containment_sgd: 1.000 of 20 measured genes are S288C reference genes
  [ok] L4 current_genome_genes: every one of the 20 measured systematic names is current
```

### Open flags

- The class was NOT in `torchcell/datasets/scerevisiae/__init__.py` when this group was
  reviewed, which made `kg_manifest admit --dataset CrispriChemgenSmith2016Dataset` raise
  `KeyError` at `kg_manifest.py:489`. Registration is the orchestrator's file, not this
  agent's; it has to land before admission can run.
- Resolving the 10 vendor catalog codes through a pinned PubChem SID lookup would recover
  7,410 records. ChemDiv / ChemBridge / TimTec ids ARE carried by PubChem as SUBSTANCE
  records, so it is a real retrieval rather than a guess, and it is the single highest-yield
  compound-table item in this group.
