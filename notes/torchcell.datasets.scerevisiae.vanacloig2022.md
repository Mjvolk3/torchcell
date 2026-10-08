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
  `EnvironmentPhysicalPerturbation(factor=ph, magnitude=5.0 pH, agent=hydrochloric acid)`
  because `Media` has no pH field and `Media` sits in 36 served closures, so adding one is
  a full rebuild. The agent is the acid the same sentence names ("adjusted to pH 5.0 with
  HCl"), so the factor's realizing species is a joined compound entity rather than a silent
  None, and the adapter projects it onto `factor` / `compound_name` / `concentration_value`.
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
  [ok] L1 provenance_gaps: 286436 documented provenance gaps over 143218/143218 records; 2 deferred field(s): ['inchikey', 'solvent']; 3063542 undeclared None values over 8 carrier fields (top: Compound.inchi x1002526, Compound.chebi_id x915272, Compound.pubchem_cid x286436, Compound.smiles x286436, Compound.inchikey x143218)
  [ok] L1 canonical_gene_names: 3598 systematic names, one canonical spelling each, each current in the genome
  [ok] L2 value_fidelity: 143218 values checked
  [ok] L2 se_nonnegative: 143218 values checked
  [ok] L2 uncertainty_sanity: 143218 labelled uncertainties, none a zero dispersion; 0 records report n_samples >= 2 with no uncertainty
  [ok] L3 measurement_type_consistent: single measurement_type: <MeasurementType.log2_ratio: 'log2_ratio'>
  [ok] L3 reference_zero: numeric rule: reference response == 0 for all 143218 records
  [ok] L3 environment_perturbed: all 143218 experiments carry an environmental edit (perturbation, non-baseline temperature, or non-baseline media; baseline temp=30.0, media='SynBase (SynH3- minus acetamide/sodium acetate/cellobiose, MSG for ammonium sulfate)')
  [ok] L3 compound_identity: environment edits: 286436 compound references carry a structure identifier; 0 declare a typed gap (0 distinct compounds, unencodable)
  [ok] L3 media_compound_identity: medium components: 143218 compound references carry a structure identifier; 0 declare a typed gap (0 distinct compounds, unencodable)
  [ok] L3 media_membership: 143218 records on a shared MEDIA_LIBRARY medium, 0 on a medium deriving from one (1 distinct media)
  [ok] L4 gene_containment_sgd: 1.000 of 3598 measured genes are S288C reference genes (>= 0.9)
  [ok] L4 current_genome_genes: every one of the 3598 measured systematic names is a gene of the current genome
```

The previous build failed L0 on 164,115 of 164,115 records and reported "no provenance
gaps ... (fully sourced)", which was a false reassurance: it gapped nothing because its
unsourced values were silent Nones.

### Which fields carry gaps, and which absence is carried some other way

286,436 gaps = exactly 2 per record, and the deferred worklist now names both:

- **`SmallMoleculePerturbation.solvent`**, one per record, `deferred_pending_source_review`
  with `resolve_with` naming Table S1. This is possible because `EnvironmentPerturbation`
  became a `ProvenanceGapMixin` carrier; before that it was a silent None. The vehicle is
  UNKNOWN rather than absent: the primary says water-insoluble compounds went in at 1% v/v
  DMSO but names which ones only in Table S1.
- **`Compound.inchikey`** of SynBase's composition-deferred SynH3- base component, one per
  record, from the shared `media.py` object. Zhang 2019 is not mirrored.

The absent per-compound **IC30 molar value is deliberately NOT a gap**. A gap is legal only
on a field that is `None`, and `SmallMoleculePerturbation.concentration` is never None: an
IC30 or fixed basis is always known, and `basis` is the schema's own documented mechanism
for a dose set to a target without a released number. The only way to type the molar
value's absence would be to make `Concentration` a gap carrier, and `Concentration` sits in
all 36 served closures, so that is a full-rebuild change rather than something to slip in
here. `L3 compound_identity` now counts 286,436 compound references (up from 143,218)
because the pH factor's HCl agent is a second identified compound on every record.

### Open flags

- The kanMX marker of the library deletion is NOT named by this paper. The primary defers
  to Andrusiak 2012 and Piotrowski 2017, neither mirrored. The
  `BarcodedKanMxDeletionPerturbation` leaf therefore asserts a marker the primary does not
  state; the `collection` field records what the primary DOES say. Closing this means
  mirroring Piotrowski 2017.
- `Concentration` is still not a `ProvenanceGapMixin` carrier, so the 39 missing IC30 molar
  values remain carried by `basis=IC30` rather than by a declared gap (see above). The
  per-compound solvent IS now a typed gap.
- 57 genes have a pooled control CPM of 0 and 72 to 113 genes per CG batch have a
  batch-matched control CPM of 0, so their log2 ratio has a pseudocount denominator. These
  are NOT dropped (the treated value was measured); the rule set only drops cells with no
  treated measurement at all.
- The compound count still does not reconcile: the paper says "each of 34 inhibitory
  chemicals", the matrix has 45 columns, and 41 are served. Table S1 would settle it.

## 2026.09.30 - Raw sha256 pin verified at build time

Issue #518 (sweep); the whole sweep is in [[torchcell.data.experiment_dataset]] (2026.09.30). Before: download-only check: yes; PyG skips `download()` when `raw/` is populated, so a file placed or edited in `raw/` built unchecked; copy before check: no (symlink); refused deposit leaving a directory: no.

Now `process()` starts with `verify_raw_files(self.raw_dir, ...)` against `DATA_SHA256`, before any record is read, and raises `RawSha256MismatchError` ("sha256 mismatch for <file>: expected <pin>, observed <digest>") with no store written. `download()` stages files through the shared `copy_verified` / `write_verified` / `link_verified` helpers, which hash before writing, so a refusal leaves nothing in `raw/`. Records built from a verified raw file are unchanged. Test: `test_a_raw_file_off_the_pin_is_refused_at_build_time` (or the renamed former Finding test) in the paired test file.

## 2026.09.30 - Module constant is the one pin at download (issue #561)

Before this change, `download()` verified the mirror bytes against the digest recorded in the raw-mirror `manifest.json`, and `process()` verified them against the module constant. Because `deposit_raw_mirror` writes the manifest from the constant, the two were equal by construction, but the loader still carried two pins. Now `download()` verifies the bytes against the module constant, and `check_manifest_pin` refuses a manifest that records any other digest, raising `ManifestPinMismatchError` named by path with both digests. The manifest stays the retrieval record. Built records are unchanged. Tests: `test_download_refuses_a_manifest_digest_off_the_module_pin` in [[tests.torchcell.datasets.scerevisiae.test_raw_pins]].

## 2026.10.02 - Ingestion audit fixes: TMM, Fig 1B conditions, DMSO, SGA background, MBO (#501, #500)

Issue #501 (ingestion audit, seven ranked findings) and #500 (library strain background). Loader `torchcell/datasets/scerevisiae/vanacloig2022.py`; identity row `torchcell/datamodels/compound_identity_inputs/vanacloig2022.txt` + `compound_identity_table.json` (re-pinned); medium `torchcell/datamodels/media.py` (`SYNBASE`); verifier entry `torchcell/verification/runners.py`. Measurement: `experiments/036-dataset-fixes-before-kg-build/scripts/vanacloig2022_ingestion.py`, outputs `experiments/036-dataset-fixes-before-kg-build/results/vanacloig2022_ingestion.json` and `vanacloig2022_ingestion_per_condition.csv`, run on the dev store (before, built 2026-09-13) vs a full scratch build of this branch (after).

### 1. Normalization: TMM replaces library-size CPM

The paper's quantity is TMM-normalized ("using TMM normalization and glmQLFit comparing paired treatment to control samples", paper.md line 66, now `NORMALIZATION`). `tmm_factors` ports edgeR 3.26.8 `calcNormFactors(method="TMM")` with its defaults: `logratioTrim=0.3`, `sumTrim=0.05`, `doWeighting=TRUE`, `Acutoff=-1e10`, all-zero rows removed, reference = the sample whose 75th-percentile count fraction is closest to the mean (first on a tie), factors scaled to geometric mean 1. Factors are computed per condition over its 3 replicates and the control columns it is paired with (same-batch Controls; all 16 for MMS), over every complete row of the matrix, against the library size summed over every released barcode. Ratio per replicate: `log2((TMM-CPM_rep + 1) / (mean TMM-CPM of the paired controls + 1))`.

- Port check (scratch, not committed): factors for CV, NAO, EtOH, MMS, DMSO, Benomyl, FeruloylAmide equal edgeR 4.4.2 `calcNormFactors` on the same sample sets to max |diff| 4.9e-15. edgeR 4.4.2's TMM code reads the same as 3.26.8's (reference choice and `.calcFactorTMM`). The test `test_tmm_factors_equal_edger_calc_norm_factors` pins a 24 x 4 matrix to edgeR 4.4.2's printed factors.
- Pseudocount: 1 CPM, a loader choice the paper does not state, recorded as `PSEUDOCOUNT_GAP` (`not_reported_by_primary`) and in the `units` string. The low-count flag stays the stored SD/SE (no new field).

Measured, per-condition median response (before CPM, after TMM) and median SD:

| condition | median before | median after | median SD before | median SD after | Spearman after vs before |
|---|---|---|---|---|---|
| crystal violet | -2.773 | -0.062 | 1.033 | 0.569 | 0.971 |
| nonylacridine orange | -0.408 | 0.025 | 0.568 | 0.318 | 0.990 |
| ethanol | -0.305 | -0.104 | 0.325 | 0.310 | 0.999 |
| acetosyringone | -0.163 | -0.023 | 0.264 | 0.260 | 0.999 |
| gamma-valerolactone | -0.178 | -0.062 | 0.295 | 0.301 | 1.000 |
| ferulamide | -0.143 | -0.027 | 0.283 | 0.285 | 0.999 |

Across the 32 conditions both stores serve, Spearman after vs before: median 0.99993, minimum 0.971 (crystal violet); outside the six the largest |median shift| is 0.099. Against the audit's edgeR reconstruction (`--edger-tsv` pointed at the #501 audit's scratch `edger_logfc.tsv`, not committed): median response minus median edgeR logFC was -2.812 (CV), -0.454 (NAO), -0.235 (EtOH) before and is -0.101 (CV), -0.020 (NAO), -0.033 (EtOH) after; the largest |offset| over the 34 served conditions is 0.101 (CV). Spearman vs edgeR is unchanged by normalization (median 0.975 over conditions in both stores, minimum p-coumaric acid 0.79), as expected for a per-condition scale factor.

### 2. The 11 tokens Fig 1B does not list are dropped

Rule `compound_not_reported_by_the_paper`: a matrix token not among the 34 Fig 1B conditions (`FIG_1B_TOKENS`; caption quote "before and after exposure to one of 34 different inhibitors", bar labels read from `images/8355ec...jpg`, image sha256 `f00ee21b...64cf`). 34 = 32 previously served + DMSO + MBO. Dropped tokens: 24Dimethylimidazole, 2Methylimidazole, 45Methylimidazole, CaffeicAcid, LevulinicAcid, Mycobutanil, SodiumAcetate, SodiumButyrate, SodiumGlyoxylate, QUADRIS1, QUADRIS2 (the QUADRIS pair moves here from the unidentified rule). Replicate agreement the #501 audit measured for the nine previously served ones (audit `analysis2.py`: mean pairwise replicate Pearson / reliability = 1 - mean(SE^2)/Var(mean); not re-measured here): SodiumGlyoxylate -0.157 / -0.91, SodiumButyrate -0.101 / -0.43, 45Methylimidazole 0.016 / 0.04, 2Methylimidazole 0.030 / 0.08, LevulinicAcid 0.083 / 0.17, SodiumAcetate 0.229 / 0.37, CaffeicAcid 0.235 / 0.46, 24Dimethylimidazole 0.292 / 0.55, Mycobutanil 0.356 / 0.56 (median reliability 0.17 vs 0.77 for the 32 published). Decision the user may reverse: drop (the paper does not report them) rather than serve with a flag.

### 3. DMSO is a served condition

DMSO is served as `SmallMoleculePerturbation(dimethyl sulfoxide, 1.0 percent_v/v, basis fixed)` (`DMSO_DOSE`, "the final concentration of DMSO in SynBase medium was 1% (v/v)"), paired against the same-batch Control columns, which the Results call "the paired SynBase medium control" (paper.md line 103, `PAIRED_SYNBASE_CONTROL`). Its `solvent` is None with no gap (it is the vehicle itself). Which inhibitors DMSO delivered is still only in Table S1 (not mirrored), so every inhibitor keeps its `solvent` gap; no inhibitor's environment names DMSO. Measured Pearson of each condition's response profile with the DMSO profile (after store): ferulamide 0.641, acetosyringone 0.563, acetovanillone 0.560, 4'-hydroxyacetophenone 0.461, acetamide 0.415, coumaroyl amide 0.403; median over the other 27 conditions 0.114. Hypothesis (untested): those six are the DMSO-delivered compounds; acetamide is water-miscible, so its correlation may not mean DMSO delivery.

### 4. Strain background (#500): SGA MATa progeny, one perturbation per record

Records move to `StrainEnvironmentResponseExperiment` / `...Reference`. The reference genome is `StrainReferenceGenome(strain="3DeltaAlpha SGA MATa progeny (Y13206 x MATa xxxΔ::kanMX array)", ploidy="haploid", background=library_background())`:

- MATa (Piotrowski 2017 paper.md line 204, "to select for the MATa meiotic progeny"); parents Y13206 and the MATa xxxΔ::kanMX array.
- Sourced alleles: can1Δ::STE2pr-Sp_his5 and lyp1Δ (Piotrowski line 204 query quote + Ohnuki 2022 Y8835 genotype), pdr1Δ::natMX, pdr3Δ::KlURA3, snq2Δ::KlLEU2 (Piotrowski query quote + the final G418/NAT/-Ura/-Leu selection sentence).
- Pending source review (`BRACHMANN_1998`): his3Δ1, leu2Δ0, ura3Δ0, met15Δ0. Only the query lineage states them (Ohnuki: Y13206 "ura3Δ0 met15Δ", parent Y8835 "ura3Δ0:: natMX4 ... met15Δ0", which disagree), and no mirrored source states the array's genotype, which a segregant's unselected allele depends on.
- Each genotype now holds ONE `BarcodedKanMxDeletionPerturbation` with `cassette="kanMX"` (Piotrowski "MATa xxxΔ::kanMX"), measured 4 -> 1 perturbations per record on all records.
- New rule `orf_is_a_selected_background_locus`: a library row screening a locus the SGA selections fix (PDR1, PDR3, SNQ2, CAN1, LYP1) contradicts the background; PDR3 and SNQ2 rows were already dropped, CAN1 (YEL063C) and LYP1 (YNL268W) rows are new drops (68 records).
- Benomyl dose is now 34.4 uM (`BENOMYL_MOLAR`, Piotrowski paper.md lines 218/208/25; Vanacloig's "10 ug/mL as previously published"); consistency check, not a source: 10 / 290.32 g/mol = 34.44 uM.
- Environment is a `CultureEnvironment`: 24-well plates (Falcon), 1,500 uL, static (0 rpm), inoculum OD600 0.1, `fixed_duration`; `pre_culture` gapped (Piotrowski 2015, not mirrored) and `auxotroph_supplements` gapped (Zhang 2019, not mirrored).

### 5. MBO adjudicated to 2-methyl-3-buten-2-ol

Rule `MBO_IDENTITY_RULE`: the in-text definition where the experiment uses the compound outranks the glossary. Both lines are sourced values: `MBO_ABBREVIATION` ("MBO : 2-Methyl-3-butyn-2-ol", line 31, not adopted) and `MBO_IDENTITY` ("biofuel endproducts (ethanol, isobutanol and 2-methyl-3-buten-2-ol (MBO))", line 103), plus `MBO_IS_A_BIOFUEL` (line 131). The identity table's MBO row is now RESOLVED: CID 8257, InChIKey HNVRRHSXBLFLIG-UHFFFAOYSA-N, SMILES CC(C)(C=C)O, ChEBI CHEBI:132752 (PubChem PUG REST, 2026-10-02; RDKit derives the same key from the SMILES). The conflict and the rule are recorded as comment lines on the identity input row: a RESOLVED `CompoundIdentityRecord` has no note field. The row was replaced in place as the curator would emit `MBO | query=2-methyl-3-buten-2-ol`, not by a full re-curation (that would restamp `retrieved_at` on all 5,498 rows); table sha256 `e8f97bdf...7f50`. MBO adds 3,482 records; its median after-store SD is 0.227.

### 6, 7. Legacy ORFs, identity details

- The 17 legacy-spelling strains stay dropped; each is now a typed `LegacyOrfStrain` in `dropped_records.json` (`legacy_orf_strains`) carrying `ConstructedOrf(source_systematic_name, relation=None, deleted_span=None)` with both fields gapped pending SGD locus history (e.g. YGL046W -> YGL045W).
- `SYNBASE.dropouts` now lists ammonium sulfate ("ammonium sulfate was replaced with 1 g/L monosodium glutamate"). Cross-dataset joins of salts vs free acids (sodium acetate vs acetic acid) are a note, not a loader change; sodium acetate is no longer served.

### Counts (scratch build, `preprocess/dropped_records.json`)

Source 45 tokens x 3,651 rows = 164,295; kept **118,662** (was 143,218); 34 conditions (was 41); 3,587 genes (was 3,598).

| rule | records | items |
|---|---|---|
| `compound_not_reported_by_the_paper` | 40,161 | 11 tokens |
| `compound_without_a_structure_identifier` | 0 | |
| `row_is_not_a_barcoded_orf_or_carries_no_counts` | 68 | 2 all-NaN rows |
| `orf_is_a_selected_background_locus` | 136 | YBL005W, YDR011W, YEL063C, YNL268W |
| `orf_is_not_a_current_genome_gene` | 748 | 22 ORFs |
| `orf_is_a_legacy_spelling_of_another_library_orf` | 578 | 17 ORFs |
| `all_three_replicate_counts_are_zero` | 3,942 | cells |

### Open

- Which inhibitors DMSO delivered (Table S1, academic.oup.com, not scriptable) and whether any Control column is SynBase + DMSO (GEO GSE186866 SOFT / series matrix, scriptable, not mirrored; needs a retrieval go-ahead).
- The four BY auxotrophies stay pending until Brachmann 1998 or the Piotrowski 2017 strain table (Supplementary, not mirrored) is mirrored.
- Pre-existing on main, not caused here: `tests/torchcell/data/test_neo4j_query_raw_single_pass.py::test_cached_environment_is_safe_to_pass_unvalidated` asserts every experiment class's `environment` is exactly `Environment`, which `StrainEnvironmentResponseExperiment` (#507) is not; the neo4j single-pass query path may therefore hand a cached plain `Environment` to a strain-resolved record.
- Experiment 033's cell table keys on `"environment_response"`; it must add `"strain_environment_response"` to keep Vanacloig records.

## 2026.10.07 - Table S1 doses and vehicles

Issue #764. Table S1 was the open item above ("Which inhibitors DMSO delivered"); it is now mirrored and every served condition's dose and vehicle is sourced from it.

### Mirror chain (key `vanacloig-pedrosComparativeChemicalGenomic2022`)

| file | sha256 | how |
|---|---|---|
| `si/si1.zip` | `707e3edecaecc3e17b553e248f5b836219a24912a0a1536a03942fd4e95e29eb` | `scripts/lit_capture_si.py`, `pmc_cloud` route: PMC Article Datasets object `PMC9508847.1/foac036_supplemental_files.zip` |
| `si/si2.pdf` | `2712cfdb92569c014a933307973353ec8da0218ba00326d165999f8c0ef96d85` | zip member `Table_S1.pdf`, stored by `capture_si.store_zip_member` (`--zip-member si/si1.zip:Table_S1.pdf`); retrieval `retrieve.zip_member` with `url`, `member`, `container_sha256` = the zip's |
| `si/si2.md` | `bad5b060bda2f50470e6a72c4005d48a9ecad8a9fa837a5bd2008307cc35e5d2` | MinerU 2.7.6, pipeline backend, `device_mode=cpu`, 200 dpi (`si/si2_ocr_provenance.json`) |

Table S1's header row: Chemical Additive | IC30 Concentration | Dissolved in DMSO? | CAS Number or reference | Vendor/Source | Catalog #. It lists 48 chemicals; the loader serves the 34 of `FIG_1B_TOKENS`.

### Dose and vehicle rules

`TABLE_S1_DOSES` holds one `SourcedValue` per Fig 1B token (`_table_s1`, `source_uri="si/si2.md"`, `page="Table S1"`), whose quote is that compound's row exactly as MinerU wrote it (HTML table row). The IC30 cell and the DMSO flag are parsed from the quote, so no number is typed separately. The mirror audit test checks every row.

- mM, uM and ug/mL rows (27 conditions): `Concentration(value, unit, basis=IC30)`, e.g. Furfural 8 mM, 2,2'-Dipyridyl 18 ug/mL, CV 15 uM.
- Percent rows stay basis-only, `value=None`: MBO 1.50%, Ethanol 4%, Isobutanol 0.75%, GVL 1.5% (IC30), MMS 0.01% (fixed). Table S1 writes no v/v or w/v and the schema has no basis-free percent unit; the percent rides in each row's `note`. A unit decision is open in #764.
- GVL: the OCR leaves its IC30 cell empty and puts `1.5%` (with the catalog number 140795000) on the next row, the "OTHER COMPOUNDS" header; both rows are quoted (`TABLE_S1_DOSES["GVL"]`, `GVL_TABLE_S1_DISPLACED`).
- Benomyl stays 34.4 uM fixed (Piotrowski 2017); Table S1 confirms the Methods' 10 ug/mL.
- DMSO stays 1.0 percent_v/v fixed (the #501 finding). Table S1 lists DMSO at 2.50% under the IC30 column, which conflicts with the Methods' "final concentration of DMSO in SynBase medium was 1% (v/v)". Left for review in #764.
- Vehicle: "Dissolved in DMSO? Yes" (17 compounds: the phenolics, both amides, and azelaic acid) -> `Solvent(name="DMSO", percent=1.0, compound=dimethyl sulfoxide)`, the percent from the `VEHICLE_CONTROL` sentence; "No" -> `solvent=None` (dissolved directly). `_solvent_gap` and the pending-source `TABLE_S1` provenance are retired; no record carries a solvent gap now.
- OCR quirks kept verbatim in the quotes: "EMIM-CI", "BMIM-CI" (Cl read as CI), "MethyIglyoxal".
- MBO: Table S1 names the chemical "2-Methyl-3-buten-2-ol (MBO)", agreeing with the Results definition already adopted (`MBO_IDENTITY_RULE` updated).

The commit message of the loader change says "29 conditions" carry a numeric IC30; the count is 27.

### Dev rebuild and record check

Old build moved aside to `processed.superseded.20261007-185707` and `preprocess.superseded.20261007-185707`; rebuilt with `python -m torchcell.database.build_dataset_lmdb --dataset EnvChemgenVanacloig2022Dataset`: 118,662 records in 84 s, 3,587 genes (unchanged). `python -m torchcell.provenance.build_manifest` reads the new build fresh (its two STALE `env_chemgen_vanacloig2022` lines are the older `.deprecated-2026-09-13-pre-gaps` and `.superseded-2026-09-12-libsize` sibling directories). One record per condition, `environment.perturbations[0]`:

| token | concentration | solvent | gaps |
|---|---|---|---|
| Furfural | 8.0 mM, IC30 | None | [] |
| Vanillin | 5.0 mM, IC30 | DMSO, 1.0 (dimethyl sulfoxide, IAZDPXIOMUYVGZ-UHFFFAOYSA-N) | [] |
| EtOH | None, IC30 | None | [] |
| 22Dipyridyl | 18.0 ug/mL, IC30 | None | [] |
| DMSO | 1.0 percent_v/v, fixed | None | [] |

### Served store

The served KG still carries the old Vanacloig records (no IC30 values, solvent gaps) until the next FULL rebuild: every Vanacloig record's environment changed, and changed records cannot go through incremental admission.

The raw-mirror manifest's `si_expected` entry (written by `deposit_raw_mirror`) still says Table S1 is not mirrored; that record describes the raw mirror's deposit and was left unchanged.
