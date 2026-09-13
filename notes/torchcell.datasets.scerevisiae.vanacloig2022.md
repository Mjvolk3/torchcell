---
id: ucmnrz3nbu3jszxwu3vmwmx
title: Vanacloig2022
desc: ''
updated: 1789272399098
created: 1789272399098
---

## 2026.09.12 - Serve-50 rebuild: raw mirror, batch-matched control, retention rules

Loader: `torchcell/datasets/scerevisiae/vanacloig2022.py`. Adapter:
`torchcell/adapters/vanacloig2022_adapter.py` +
`torchcell/adapters/conf/env_chemgen_vanacloig2022_adapter.yaml`. Tests:
`tests/torchcell/datasets/scerevisiae/test_vanacloig2022.py`,
`tests/torchcell/adapters/test_vanacloig2022_adapter.py`.

### Why the previous build could not be served

The stored LMDB failed L0 on 100% of its records (`Media.is_synthetic Field required`,
the medium predated that field becoming required), and every value in it was either
name-only or silently defaulted: a free-text `Media(name="SynBase")` that joined nothing,
41 of 45 compounds with no structure identifier, the DMSO vehicle control served as an
inhibitor at IC30, a reference environment containing the very compound it was the control
for, 4,854 records shipping `sample_sd = 0` manufactured by the CPM pseudocount, and a
`download()` that depended on a live NCBI FTP URL with no mirror behind it.

### Raw mirror and provenance chain

`$DATA_ROOT/torchcell-raw/vanacloig-pedrosComparativeChemicalGenomic2022/` now holds the
one file the loader consumes, `data/GSE186866_ChemGenomics_Raw_Counts_matrix.txt.gz`
(sha256 `e29eb027...`), with a `manifest.json` recording `source_url`, the
`torchcell.literature.retrieve.direct_url` retriever, its params and `retrieved_at`
(2026-09-13). `download()` reads the mirror and verifies that sha256; the GEO URL is
retrieval metadata that `deposit_raw_mirror` re-runs, never a build dependency. Table S1
(per-compound IC30 molar values, per-compound DMSO pairing) and Dataset2_mclust_cdt are
recorded in `si_expected` as NOT mirrored: academic.oup.com is not scriptable.

Every environment and phenotype number is a module-level `SourcedValue` carrying a verbatim
quote plus the sha256 of `paper.md`
(`0b5d938b54b8424fa08203a4357bc8f7c7dfae3fbe1a6d07d422848b92f37ba3`). The 14 of them audit
clean through `audit_sourced_value`, which the loader test asserts: temperature 30 C,
48 h (two 24 h periods), 6.5 doublings, pH 5.0, biological triplicate, the IC30 basis, the
Benomyl and MMS doses, the assay type, the collection label, the up-tag barcode statement,
the paired-control design and the vehicle-control statement.

### What the stored number is

The paper's published values are edgeR TMM + glmQLF paired logFCs, which need R/edgeR and
the unmirrored Table S1, so the loader recomputes the paper's DEFINED quantity from the raw
counts instead: per-sample CPM, then per gene
`log2((CPM_treated_rep + 1) / (CPM_control + 1))`, response = mean of the 3 replicates,
uncertainty = their sample SD. The control is now the mean of the **same CG00n batch's**
inhibitor-free control columns (the paper's paired design), pooled over all 16 control
columns for MMS only, which is the one retained compound the paper analyzed unpaired. The
`units` string records which control each record used, so the two are distinguishable in
the record itself. Nothing mirrored can check the stored number against a published one:
L2 checks range and finiteness, never agreement.

### Retention rules and counts

Source grid 45 compounds x 3,651 library rows = 164,295. Kept **143,218**. Rules and counts
(machine-readable in `preprocess/dropped_records.json`):

| rule | scope | records | items |
|---|---|---|---|
| `vehicle_control_served_as_a_treatment` | compound | 3,651 | DMSO |
| `compound_without_a_structure_identifier` | compound | 10,953 | MBO, QUADRIS1, QUADRIS2 |
| `row_is_not_a_barcoded_orf_or_carries_no_counts` | library row | 164 | 2 all-NaN, 2 background |
| `orf_is_not_a_current_genome_gene` | library row | 902 | 22 retired / non-gene ORFs |
| `orf_is_a_legacy_spelling_of_another_library_orf` | library row | 697 | 17 ORFs |
| `all_three_replicate_counts_are_zero` | cell | 4,710 | measured per compound |

MBO is dropped because the primary contradicts itself: the Abbreviations block reads
"MBO : 2-Methyl-3-butyn-2-ol" (CID 8258) and the Introduction reads
"2-methyl-3-buten-2-ol (MBO)" (CID 8257). QUADRIS is a commercial suspension at two
arbitrarily selected concentrations, not a chemical species. Both reasons are recorded in
`compound_identity_inputs/vanacloig2022.txt`.

The legacy-spelling rule matters for a barcoded pool: 17 ORFs resolve to an ORF the SAME
library also carries under its current name, so remapping them would merge two physically
distinct barcoded strains into one record key. Dropping the legacy row keeps the current
one; the loader raises if any duplicate survives.

### Encoding

- **Medium**: the shared `MEDIA_LIBRARY["SYNBASE"]` object (SynH3- minus acetamide, sodium
  acetate and cellobiose, with MSG replacing ammonium sulfate), so it joins at its
  `base_medium` SynH3-. pH 5.0 rides as
  `EnvironmentPhysicalPerturbation(factor=ph, magnitude=5.0 pH)` because `Media` has no pH
  field and `Media` sits in 36 served closures, so adding one is a full rebuild.
- **Compounds**: `resolved_compound` against the pinned identity table. All 41 retained
  tokens resolve to a canonical PubChem name and an InChIKey, so the compound entity joins
  across datasets. Dose is `Concentration(basis=IC30)` for 39 of them (Table S1 holds the
  molar values), `value=10.0 ug/mL, basis=fixed` for Benomyl, and `basis=fixed` with no
  value for MMS: the primary writes "0.01%" with no v/v or w/v and defers to Piotrowski
  2017, which is not mirrored, so the unit would be a guess.
- **Genotype**: `BarcodedKanMxDeletionPerturbation` carrying the row's UPTAG barcode and
  the collection label, plus the three constant 3DeltaAlpha background deletions
  (`NatMx` PDR1, `MarkerDeletion` PDR3/KlURA3 and SNQ2/KlLEU2). `perturbed_gene_name` is
  the GENOME's standard name, not the source `std_name`, so one gene is one node.
- **Reference**: `environment_reference` is now the inhibitor-free SynBase environment
  (pH only) rather than a copy of the treated environment, and `genome_reference.strain` is
  S288C rather than 3DeltaAlpha, since the three background deletions are already carried
  as perturbations.

The normalization is deliberately independent of those rules: the CPM library size is
summed over EVERY released barcode (NaN rows contributing no reads) BEFORE any retention
rule runs, so a later change to the gene-name policy cannot silently move a stored value.

10 of the 3,608 retained library rows are all-zero in every one of the 41 kept compounds,
so 3,598 genes appear in the served records (YBR212W, YGR052W, YGR176W, YJL126W, YLR120C,
YML004C, YNR033W, YOR384W, YPR004C, YPR030W).

### Verifier result (verbatim)

`python scripts/verify_vanwild.py env_chemgen_vanacloig2022` in the scratchpad, which is
`run_environment_response`'s body with this dataset's entry; report written to
`preprocess/verification_report.json`.

```
env_chemgen_vanacloig2022: PASS
  [ok] L0 structural: 143218 records validated
  [ok] L1 count: observed 143218, expected 143218
  [ok] L1 pair_uniqueness: 143218 unique (study, strain, condition) records, one each
  [ok] L1 provenance_gaps: 143218 documented provenance gaps over 143218/143218 records; 1 deferred field(s): ['inchikey']; 2777106 undeclared None values over 8 carrier fields (top: Compound.inchi x859308, Compound.chebi_id x772054, Compound.pubchem_cid x286436, Compound.smiles x286436, Compound.inchikey x143218)
  [ok] L1 canonical_gene_names: 3598 systematic names, one canonical spelling each, each current in the genome
  [ok] L2 value_fidelity: 143218 values checked
  [ok] L2 se_nonnegative: 143218 values checked
  [ok] L2 uncertainty_sanity: 143218 labelled uncertainties, none a zero dispersion; 0 records report n_samples >= 2 with no uncertainty
  [ok] L3 measurement_type_consistent: single measurement_type: <MeasurementType.log2_ratio: 'log2_ratio'>
  [ok] L3 reference_zero: numeric rule: reference response == 0 for all 143218 records
  [ok] L3 environment_perturbed: all 143218 experiments carry an environmental edit (perturbation, non-baseline temperature, or non-baseline media; baseline temp=30.0, media='SynBase (SynH3- minus acetamide/sodium acetate/cellobiose, MSG for ammonium sulfate)')
  [ok] L3 compound_identity: environment edits: 143218 compound references carry a structure identifier; 0 declare a typed gap (0 distinct compounds, unencodable)
  [ok] L3 media_compound_identity: medium components: 143218 compound references carry a structure identifier; 0 declare a typed gap (0 distinct compounds, unencodable)
  [ok] L3 media_membership: 143218 records on a shared MEDIA_LIBRARY medium, 0 on a medium deriving from one (1 distinct media)
  [ok] L4 gene_containment_sgd: 1.000 of 3598 measured genes are S288C reference genes (>= 0.9)
  [ok] L4 current_genome_genes: every one of the 3598 measured systematic names is a gene of the current genome
```

The previous build failed L0 on 164,115 of 164,115 records and reported "no provenance
gaps ... (fully sourced)", which was a false reassurance: it gapped nothing because its
unsourced values were silent Nones. The one deferred gap here is the `inchikey` of
SynBase's composition-deferred SynH3- base component, which is a shared `media.py` object,
and it names a real worklist item (Zhang 2019 is not mirrored).

### Open flags

- The kanMX marker of the library deletion is NOT named by this paper. The primary defers
  to Andrusiak 2012 and Piotrowski 2017, neither mirrored. The
  `BarcodedKanMxDeletionPerturbation` leaf therefore asserts a marker the primary does not
  state; the `collection` field records what the primary DOES say. Closing this means
  mirroring Piotrowski 2017.
- `Concentration` is not a `ProvenanceGapMixin` carrier, so the 39 missing IC30 molar
  values and the absent per-compound `Solvent` (Table S1 says which compounds were
  DMSO-delivered) are untyped absences rather than declared gaps.
- 57 genes have a pooled control CPM of 0 and 72 to 113 genes per CG batch have a
  batch-matched control CPM of 0, so their log2 ratio has a pseudocount denominator. These
  are NOT dropped (the treated value was measured); the rule set only drops cells with no
  treated measurement at all.
- The compound count still does not reconcile: the paper says "each of 34 inhibitory
  chemicals", the matrix has 45 columns, and 41 are served. Table S1 would settle it.
