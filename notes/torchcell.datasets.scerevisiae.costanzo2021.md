---
id: x3g73i9gyfuoupd0ktr0tue
title: Costanzo2021
desc: ''
updated: 1789272444649
created: 1789272444649
---

## 2026.09.12 - Serve-50 pass: shared SGA medium, shared gene resolver, typed gaps

Rebuilt `EnvChemgenCostanzo2021Dataset` (`torchcell/datasets/scerevisiae/costanzo2021.py`)
so the condition-SGA panel can be admitted to the served graph. The July build failed L0 on
100% of its records (it predates `Media.is_synthetic`, `ProvenanceGapMixin` and
`resolved_compound`), so a rebuild was mandatory regardless of the content fixes below.

### What changed

- **Medium.** `Media(name="SGA final selection medium (synthetic, agar)")` was an invented
  free-text label with no components: it joined nothing, and `media_to_bounds` could not
  resolve a single exchange from it. The 13 small-molecule conditions now use
  `torchcell.datamodels.media.SGA_DM_SELECTION` verbatim, the SAME object served
  `SmfCostanzo2016Dataset` uses, so the media node is identical and the SGA family joins.
- **Galactose is a medium, not an added compound.** The paper calls it "an alternative
  carbon source", which is a replacement; the old encoding asserted glucose AND galactose in
  one flask and, because the adapter projects no compound for an
  `EnvironmentPhysicalPerturbation`, lost the compound outside `serialized_data`. The
  condition is now `media.SGA_DM_SELECTION_GALACTOSE` with no perturbation.
- **Gene resolution moved to the shared resolver.** The loader validated ORFs against raw
  SGD FASTA headers. That test both dropped 18 strains it should have kept (old systematic
  names SGD RENAMED) and kept 10 it should have dropped (blocked reading frames and
  pseudogenes: valid R64 features, not genes). Every `Systematic Name` now goes through
  `SCerevisiaeGenome.resolve_gene_name`.
- **Typed axes.** `assay_type=AssayType.colony_size_array` (sourced); `units` collapsed back
  to the single shared `MEASUREMENT_UNITS` (it carried six different strings because the
  SI dose anomalies had been appended to it); `Environment.duration_hours` carries
  `ProvenanceGap(deferred_pending_source_review)`.
- **Reference strain** `"SGA reference (BY4741-derived deletion / TS array background)"`
  (a repo-unique free-text value) -> `"S288C"`, matching served Costanzo 2016 and Kuzmin.
- **Every metadata number is a `SourcedValue`** with a verbatim quote + the mirrored file's
  sha256: `_N_SAMPLES`, `_TEMPERATURE`, `_MATCHED_DESIGN`, `_ASSAY`, `_READOUT_DEFINITION`,
  `_VARIANCE_NOTE`, `_NEUTRAL_QUERY`, `_ALTERNATIVE_CARBON`, and one per condition dose.

### Drop rule and counts

Rule: a source ORF is kept only when `resolve_gene_name` returns CURRENT or RENAMED.

| resolver status | ORFs | records |
|---|---|---|
| CURRENT (kept) | 4,396 | |
| RENAMED (kept, stored under the current systematic name) | 18 | +252 vs the old build |
| NON_GENE_FEATURE (dropped) | 12 | 168 |
| RETIRED (dropped) | 3 | 42 |

4,414 kept strains x 14 conditions minus empty cells = **61,430 records** (was 61,318).
The accounting is written to `dropped_records.json` beside `processed/`.

The 12 non-gene features are YDR134C, YER108C, YER109C, YFL056C, YIL167W, YIL168W, YIL170W,
YIR043C, YLL016W, YLL017W, YOL153C, YOR031W; the 3 retired are YAR037W, YAR040C, YAR043C.

### Tunicamycin is KEPT, correcting the review

The review expected 4,390 tunicamycin records to be dropped for having no resolvable
structure. Measured against the current curated table, `resolved_compound("tunicamycin")`
returns `RESOLVED_MIXTURE` with `chebi_id="CHEBI:29699"` and no InChIKey. The retention rule
is "carries SOME structure identifier" (`CompoundIdentityResolution.identified` accepts
InChIKey OR ChEBI OR CID), and the L3 `compound_identity` verifier applies the same rule, so
the records stay. All 14 condition compounds are identified; 0 records are dropped on the
compound rule.

### 26 C is a derivation, and it is labelled as one

The only temperature sentence in the mirrored `paper.md` is: *"Colony size measurements of
SGA deletion and TS array mutant strains were based on an average of three replicate control
screens conducted per each of 14 test conditions as well as the reference condition at 26
C."* The clause closes a list that ends with the reference condition; the sentence does not
separately state the 14 test screens' temperature. 26.0 is applied to all 15 conditions
because one screen's three array copies share an incubator (*"every double-mutant array
generated from a singlequery SGA screen was copied three times. One copy was grown in the
standard SGA reference condition, whereas the two other copies were each grown in different
conditional media"*) and because the TS array needs a permissive temperature. That is a
derivation, recorded in `_TEMPERATURE.note`, in the loader docstring and in the verifier
`method` string. It also matches served Kuzmin 2018's `Temperature(value=26)`, so the
temperature node joins.

### YPR089W / YPR090W: a merged pair that must not collapse

The array screened YPR089W (`dma5081`) and YPR090W (`dma5080`) as two strains with two
measurements; SGD has since merged YPR090W into YPR089W. Keying both to YPR089W with the
resolved name in `perturbed_gene_name` produced 14 duplicate (strain, condition) triples and
failed L1. `perturbed_gene_name` therefore keeps the SHEET's own name for a strain with no
common name, which is what keeps the two apart. The cost is that YPR089W carries two
spellings.

### Provenance chain

- Data File S1 `Costanzo et al_Data File S1_Conditions_Strains_Fitness.xlsx`, sha256
  `f6c313de416ce8cc6ae87e2020b4389bd4adeb07cdb6a438aecaf1e45e6228ad`, deposited to
  `$DATA_ROOT/torchcell-raw/costanzoEnvironmentalRobustnessGlobal2021/data/` with a
  `manifest.json`.
- Methods quotes anchor to the library mirror's `paper.md`, sha256
  `ba22973ed0c53c00c37bcfb9f659d3b0373c451a3f7633158afae274035559fb`. Every quote in the
  loader was verified to be an exact substring of that file.

### Verifier result (verbatim, 2026.09.12)

```
env_chemgen_costanzo2021: FAIL
  [ok] L0 structural: 61430 records validated
  [ok] L1 count: observed 61430, expected 61430
  [ok] L1 pair_uniqueness: 61430 unique (study, strain, condition) records, one each
  [ok] L1 provenance_gaps: 127258 documented provenance gaps over 61430/61430 records; 1 deferred field(s): ['duration_hours']
  [XX] L1 canonical_gene_names: 147 genes carry more than one common-name spelling (5466 records); 0 systematic names are not the genome's current name; 780 common names resolve to another gene
  [ok] L2 value_fidelity: 61430 values checked
  [ok] L3 measurement_type_consistent: single measurement_type: differential_fitness
  [ok] L3 reference_zero: numeric rule: reference response == 0 for all 61430 records
  [ok] L3 compound_identity: 57039 compound references carry a structure identifier
  [ok] L3 environment_perturbed: all 61430 experiments carry an environmental edit
  [ok] L3 media_membership: 61430 records on a shared MEDIA_LIBRARY medium (2 distinct media)
  [ok] L4 gene_containment_sgd: 1.000 of 4158 measured genes are S288C reference genes
  [ok] L4 current_genome_genes: every one of the 4158 measured systematic names is a gene of the current genome
```

### Open flags

- **L1 `canonical_gene_names` fails, and it is a shared-rule gap, not this dataset's
  defect.** The rule requires one common-name spelling per systematic name and requires that
  spelling to resolve back to it. An SGA TS ALLELIC SERIES violates both by construction:
  one ORF carries up to 18 alleles, and `resolve_gene_name("act1-101")` returns RETIRED
  `ACT1-101`, not `YFL039C`. The allele cannot move out of `perturbed_gene_name` either,
  because `_genotype_signature` keys L1 uniqueness on it and 18 ACT1 strains would become 18
  duplicates. Measured on the SERVED `smf_costanzo2016` build, the same rule reports
  `669 genes carry more than one common-name spelling (4432 records); 1979 common names
  resolve to another gene` (examples `tfc3-g349e`, `cys3_damp`), so every SGA allele/DAmP
  dataset fails it. The fix belongs in `verification/common.py` +
  `verification/environment_response.py`: either exempt allele-bearing perturbations from
  the spelling rule, or put the SGA `strain_id` into `_genotype_signature` so the allele can
  leave `perturbed_gene_name`.
- **Solvent is unknown.** Row 1 of the Conditions sheet reads "Reference condition +
  solvent", so a vehicle was used, but the Science supplementary PDF that would name it is
  not mirrored. `SmallMoleculePerturbation` is not a `ProvenanceGapMixin`, so the absence
  cannot be typed; depositing the Science SI is the follow-up.
- **Three SI doses are chemically implausible** and are stored verbatim with the flag in
  their `SourcedValue.note`: bortezomib "1300 mM", actinomycin D "20 mM", geldanamycin
  "10 mM". Two bare fractions are read as percent: galactose "0.02" -> 2% w/v, MMS "0.0001"
  -> 0.01% v/v. Neither basis nor unit is in the source; both readings are notes, not quotes.
- **The released sheet is 57 deletion strains short of the paper's text** (3,647 deletion
  rows vs "3704 viable deletion mutant strains"). A source discrepancy, not a loader bug.
- **The neutral natMX query marker is not in the genotype**, matching served Costanzo 2016.

### Files

- `torchcell/datasets/scerevisiae/costanzo2021.py`
- `torchcell/adapters/costanzo2021_adapter.py`, `torchcell/adapters/conf/costanzo2021_adapter.yaml`
- `tests/torchcell/datasets/scerevisiae/test_costanzo2021.py`, `tests/torchcell/adapters/test_costanzo2021_adapter.py`

## 2026.09.12 - Data File S1 now carries a typed manual_browser retrieval record

`RetrievalMethod.manual_browser` exists, so the raw mirror's `manifest.json` records a typed `RetrievalRecord` (method `manual_browser`, `SCIENCE_SI_URL` as the source, the `MANUAL_RECIPE` as the `retrieval_command` parameter, and the sha256 of the deposited bytes) instead of `retrieval=None`, and the manifest is `provenance_complete=True`. With allele-bearing perturbation types now exempt from the spelling and round-trip checks in `verification/common.py`, `L1 canonical_gene_names` passes and the dataset is `env_chemgen_costanzo2021: PASS`.
