---
id: e7kukrfh295qa787sn2rvzi
title: Mota2024
desc: ''
updated: 1789272459497
created: 1789272459497
---

## 2026.09.12 - Serve-50 pass: pH as a typed edit, the ordinal grade, one gene under two names

Rebuilt `EnvChemgenMota2024Dataset` (`torchcell/datasets/scerevisiae/mota2024.py`). The July
build failed L0 on 100% of its 1,273 records (its first error on every record was
`Experiment.environment.media.is_synthetic Field required`, input `{'name': 'YPD, pH 4.5',
'state': 'solid'}`), so a rebuild was mandatory.

### What changed

- **Medium and pH.** `Media(name="YPD, pH 4.5", state="solid", is_synthetic=False)` was free
  text with zero components; its node hash `9bd652b78792a9f9` joined nothing. The paper
  states the recipe AND the acid: *"in liquid YPD medium containing, 20 g/L glucose ...
  10 g/L yeast extract and 20 g/L peptone ... acidified with HCl until pH 4.5. Solid media
  were prepared by addition of 20 g/L agar."* The medium is now the shared
  `media.YPD_AGAR` (`base_medium="YPD"`), and the acidification is a typed
  `EnvironmentPhysicalPerturbation(factor=ph, magnitude=Concentration(4.5, pH),
  agent=resolved_compound("hydrochloric acid"))` on BOTH the treated and the reference
  environment. pH is never a medium name.
- **Ordinality is stored instead of erased.** `measurement_type` was `categorical` with
  `environment_response=None`, which threw away the order and left L2 value fidelity
  checking nothing. The source scale `0 < + < ++` is exactly `MeasurementType.ordinal`:
  the rank rides on `environment_response` (0.0 / 1.0 / 2.0), the shared call on `category`
  (`no_change` / `reduced` / `severely_reduced`), and the source's own symbol on
  `category_label` (`"0"` / `"+"` / `"++"`). The reference is the parental strain, which the
  paper scores `0` = *"an absence of a detectable susceptibility phenotype"*, so its rank is
  0.0 and L3 reference_zero now holds NUMERICALLY over 1,270 records.
- **Gene resolution.** `if _SYSTEMATIC_RE.match(gene): return gene` accepted any token that
  merely LOOKED systematic, which stored four stale systematic names on seven records (L4
  read 0.993, above its 0.90 floor, so it passed). All four are RENAMED, not retired:
  YGR272C -> YGR271C-A, YJL021C -> YJL020C, YML010W-A -> YML009W-B,
  YML013C-A -> YML012C-A. Every token now goes through `resolve_gene_name`.
- **`assay_type=AssayType.spot_dilution`**, with a note that the collection screen itself is
  a single-density 96-pin stamp (the serial dilutions chose the doses).
- **`n_samples` / `sample_unit` are TYPED GAPS**
  (`ProvenanceGap(not_reported_by_primary)`). The paper's *">= 3 independent experiments"*
  statements attach to the CFU-viability and intracellular-pH assays; the disruptome screen
  states no replicate count. This was previously a silent `None` documented only in a
  docstring.
- **Every metadata number is a `SourcedValue`** quoting the mirrored `paper.md`:
  `_BACKGROUND`, `_MEDIUM`, `_PH`, `_TEMPERATURE`, `_CONCENTRATIONS`, `_SCREEN_DESIGN`,
  `_ASSAY`, `_DURATION`, `_SCORING`, `_FIGURE_S2`.

### One gene under two names: EFG1 and YGR272C

All three acid tables list EFG1 and YGR272C as separate rows, and the SI's own annotation of
YGR272C says they are one gene: *"YGR272C was originally annotated as an independent ORF,
but as a result of a sequence change, it was merged with an adjacent ORF into a single
reading frame, designated YGR271C-A"*. Correct resolution maps both to YGR271C-A, merging
them, which is why the record count falls by 3 rather than staying at 1,273.

### Dedup and drop rules, and the counts

`_DEDUP_RULE`: one record per (resolved systematic ORF, acid); where two source rows claim
one gene the MORE SEVERE grade wins (`++` > `+`) and a tie is broken by the
lexicographically smallest source token. Where two DIFFERENT tokens claim one gene the
genome's canonical common name is stored in `perturbed_gene_name` instead of either token,
so the gene carries one spelling across the three acids (here: `EFG1`).

`DROP_RULE`: a token whose resolver status is neither CURRENT nor RENAMED is dropped. Six
tokens are genuinely RETIRED in R64-4-1: REF1, RLM2, SBR2 (all three acids), ILM2 (butyric),
VPS236 (butyric, octanoic), SIW15 (octanoic) -- 13 records. Likely SI typos (VPS236->VPS36?,
SIW15->SIW14?, ILM2->ILM1?, RLM2->RLM1?), a flagged follow-up; they are never guessed.

| acid | raw susceptible rows | records |
|---|---|---|
| acetic | 377 | 372 |
| butyric | 422 | 415 |
| octanoic | 490 | 483 |
| total | 1,289 | **1,270** |

1,289 - 3 RNR4 duplicates - 3 EFG1/YGR272C merges - 13 retired-token drops = 1,270 (was
1,273). The raw counts reproduce the paper's own totals exactly (*"377 deletion mutants were
found to be more susceptible ... 46 ... 331 ... butyric ... 422 ... 51 ... 371 ... octanoic
... 490 ... 53 ... 437"*). The accounting is written to `dropped_records.json` beside
`processed/`.

### The duration rule, written down

The Methods give a RANGE for when the acid plates were photographed: *"Photographs were
taken after 24 h of incubation for control plates (YPD medium) or 36-48 h in the presence of
the acids."* The SCORING definition this dataset stores is anchored at a point: *"(++) if no
growth was observed after 48 h of incubation"*. `duration_hours` is the time at which the
STORED call was made, so the rule is **48.0**, and the 36-48 h photograph window plus the
24 h control reading are recorded in `_DURATION.note`.

### Provenance chain

- The three BMC supplementary spreadsheets (Additional files 1-3 = Tables S1/S2/S3) are
  deposited at `$DATA_ROOT/torchcell-raw/motaSharedMoreSpecific2024/si/` with a
  `manifest.json` recording `RetrievalMethod.springer_esm`, the exact
  `torchcell.literature.retrieve.springer_esm` retriever and its URL, and the per-file
  sha256. Re-running the retrieval on 2026-09-12 produced all three bit-identically, which
  is recorded as each record's `SourceCheck`. `download()` now prefers the mirror and only
  falls to the CDN when the mirror lacks the file, so a rebuild does not depend on a live
  URL surviving.
- Methods quotes anchor to the library mirror's `paper.md`, sha256
  `a19769f757fd912139551f39736dd2b67581cb03f83a7a9b9385e28516b1f1b6`. Every quote in the
  loader was verified to be an exact substring of that file.

### Verifier result (verbatim, 2026.09.12)

```
env_chemgen_mota2024: PASS
  [ok] L0 structural: 1270 records validated
  [ok] L1 count: observed 1270, expected 1270
  [ok] L1 pair_uniqueness: 1270 unique (study, strain, condition) records, one each
  [ok] L1 canonical_gene_names: 600 systematic names, one canonical spelling each, each current in the genome
  [ok] L2 value_fidelity: 1270 values checked
  [ok] L3 measurement_type_consistent: single measurement_type: ordinal
  [ok] L3 reference_zero: numeric rule: reference response == 0 for all 1270 records
  [ok] L3 compound_identity: 2540 compound references carry a structure identifier
  [ok] L3 media_membership: 1270 records on a shared MEDIA_LIBRARY medium (1 distinct media)
  [ok] L4 gene_containment_sgd: 1.000 of 600 measured genes are S288C reference genes
  [ok] L4 current_genome_genes: every one of the 600 measured systematic names is a gene of the current genome
```

### Open flags

- **The pH perturbation's `factor`, `magnitude` and `agent` are NOT projected columns.**
  `CellAdapter._environment_perturbation_node_from` reads `compound` and `concentration`
  only, so an `EnvironmentPhysicalPerturbation` carries its identity in `serialized_data`
  alone. Projecting them means editing a method a served dataset uses, which is adapter
  drift; no served dataset emits an `EnvironmentPhysicalPerturbation`, so no served node's
  properties would change, but the change is a shared-layer decision and not this dataset's
  to make. The perturbation serializes correctly either way.
- **The octanoic dose is written "0.3 mM" in the Methods and "0.30 mM" in the Figure S2
  caption.** The same number; stored as 0.30.
- **This dataset does NOT join the served YPD datasets at the media-node level** (they emit
  their own inline `Media(name="YPD")`), but it joins Auesukaree 2009 and Bloom 2019 at
  `base_medium == "YPD"`.

### Files

- `torchcell/datasets/scerevisiae/mota2024.py`
- `torchcell/adapters/mota2024_adapter.py`, `torchcell/adapters/conf/mota2024_adapter.yaml`
- `tests/torchcell/datasets/scerevisiae/test_mota2024.py`, `tests/torchcell/adapters/test_mota2024_adapter.py`
