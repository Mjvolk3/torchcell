---
id: 2s7rymd6wo03wq2opoq6r1r
title: Parsons2006
desc: ''
updated: 1791746748302
created: 1791746748302
---

## 2026.10.11 - Loaded: the chemical-genetic compendium from its own Table S7

Yeast Zotero audit rank 2 ([[database.yeast-zotero-collection-audit.2026.10.11]], schedule row 60, Zotero `3PRLBH7Y`). Parsons et al. 2006, Cell 126:611 (doi 10.1016/j.cell.2006.06.040, PMID 16901791) screened the pooled MATa haploid deletion collection, read out by its barcodes on Agilent arrays, against 82 conditions (75 compounds, 7 crude extracts). Loader `torchcell/datasets/scerevisiae/parsons2006.py`, class `EnvChemgenParsons2006Dataset`, store `$DATA_ROOT/data/torchcell/env_chemgen_parsons2006/` (built at `1a4a9381a`, clean tree).

### Records (measured from the built store)

| quantity | count |
|---|---|
| Table S7 cells (4,111 strains x 82 conditions) | 337,102 |
| released ratios (non-blank) | 329,254 |
| served records | **291,673** |
| strains served (4,111 minus 14 non-gene ORFs) | 4,097, on 4,096 genes |
| conditions served (82 minus 9 without a structure) | 73 |
| records on YPD / SC / SC lacking lysine | 279,497 / 8,133 / 4,043 |
| references (one per medium) | 3 |

Dropped (`preprocess/dropped_records.json`, rules applied in this order, each cell counted once):

| rule | records | items |
|---|---|---|
| `ratio_cell_blank_in_release` | 7,848 | the release's two-space blanks (both tags removed as noise-dominated) |
| `orf_is_not_a_current_gene` | 1,144 | 14 ORFs R64 calls `non_gene_feature`: YCL074W, YCL075W, YDR134C, YER109C, YFL056C, YIL167W, YIL168W, YIL170W, YIR043C, YIR044C, YLL016W, YLL017W, YOL153C, YOR031W |
| `compound_without_a_structure_identifier` | 36,437 | the 7 crude extracts, agelasine E, stichloroside |

Not released as ratios (counted in the same file, no record possible): Table S4 lists 4,232 strains, 121 more than Table S7, and the build checks they are exactly Table S2's 121 multidrug-sensitive strains ("Prior to clustering, we processed the data by removing a set of 121 genes whose corresponding deletion mutant displayed statistically significant multidrug sensitivity (Table S2)."); they carry 9,875 p-values and no ratio. 6 Table S7 cells have a p-value and a blank ratio; 0 ratios lack a p-value. The audit's 329,254 is confirmed; the brief's "4,111 strains x 82 compound conditions" holds, with 9 of the 82 having no compound structure.

### The sign: stored as released, positive = sensitive

Table S7 is log2(CONTROL / TREATED). Verbatim (`paper.md`, sha256 `dfc222ac...`): "In each experiment, ${ \mathsf { l o g } } _ { 2 }$ ratios of the control divided by sample intensities were computed for each tag." and "the combined average ${ \mathsf { l o g } } _ { 2 }$ ratio (control/experiment) of both barcodes (uptag and downtag) corresponding to each strain is greater than 0.5, meaning that the drug treatment leads to at least a 1.4-fold depletion of the deletion mutant relative to the control". The SI (`si/si1.md`, sha256 `36fa1487...`): "The ratio of control signal/treated sample signal corresponding to each barcode was averaged for two fluor-reversed chips and reported as a $\log _ { 2 }$ ratio."

The value is stored unnegated on `environment_response` with `measurement_type=log2_ratio` and the orientation in `units` ("POSITIVE = the deletion strain is depleted in the drug-treated pool relative to the 1% solvent-control pool (sensitive)"), the Hillenmeyer 2008 HET precedent. This puts Parsons in the positive-is-sick group with both Hillenmeyer arms, against Vanacloig, Hoepfner and Wildenhain; a pooled target must orient it first. Measured ([[experiments.database.scripts.parsons2006_release_checks]]): of the 5,550 cells whose Table S4 -log10 p exceeds 5, 5,521 (99.5%) carry a positive ratio, so the bytes agree with the stated orientation. `MeasurementType.log2_ratio`'s docstring says treatment/control; the conflict (shared with Hillenmeyer HET) is issue #1031, which proposes a typed orientation for the KG 4.0 schema pass.

Tags: a strain with one non-functional tag (Table S5) is, per the Methods, "averaged with the most probable value for a ${ \mathsf { l o g } } _ { 2 }$ ratio measurement (approximately zero)". Measured on Table S7 (same script), the 180 such strains have a value SD of 0.530 against 0.471 for the rest (ratio 1.13); a halved value would put the ratio below 0.71 whatever the correlation between tags, so Table S7 is not halved for them. Hypothesis (untested): the halving belongs to the clustering input only. The stored value is the released one either way.

### Phenotype

- `environment_response`: the Table S7 value verbatim.
- `environment_response_p_value`: 10^-x of the same Table S4 cell, "a t test score and a corresponding p value were calculated based on the ${ \mathsf { l o g } } _ { 2 }$ ratio measurements of growth of that strain in the presence of that particular drug, with respect to all other measurements obtained for that strain under all 82 conditions"; the SI caps it ("$- \log _ { 1 0 }$ (p-values) ${ > } 3 0 0$ are set to 300"), so a released 300 is stored as 1e-300. The p-value's tags are combined by variance weight, not by the plain mean of Table S7; recorded in `units`.
- `n_samples = 2`, `sample_unit = technical_replicate`: the two fluor-reversed chips of one pooled culture. No replicate screen per condition is released, and no dispersion: `environment_response_uncertainty` and `_se` carry `not_reported_by_primary` gaps.
- `assay_type = pooled_competitive_growth_barcode`.

### Genotype and background

- One `BarcodedKanMxDeletionPerturbation` per record: `collection` "MATa haploid deletion mutants (Research Genetics)" ("MATa haploid deletion mutants $( \sim 4 8 0 0 )$ were obtained from Research Genetics"), `cassette` kanMX4 (Giaever 2014, which also says the YKO strains were "directly made available to the yeast community by Research Genetics, SGTC, and ATCC"), barcode and downtag gapped pending Giaever 2002.
- Renamed ORFs keep their strain under the current gene with a `ConstructedOrf` (Hillenmeyer's `constructed_orf`): YDL038C under YDL039C (`merged`, since the release also carries YDL039C directly) and YGR272C under YGR271C-A (relation gapped).
- Reference genome: `StrainReferenceGenome` BY4741. The SI strain table states "BY4741 MATa ura3∆0 leu2∆0 his3∆1 met15∆0", which sources each allele (each with a `deleted_span` gap pending Brachmann 1998); the paper never says the pool is BY4741, so the background's `provenance` is a `deferred_pending_source_review` gap naming Giaever 2002 (the Hoepfner 2014 precedent).
- Measured (same script): 24 of the 4,111 Table S7 ORFs are in the SGD essentiality set (`data/torchcell/gene_essentiality_sgd/preprocess/gene_set.json`), among them autophagy genes (ATG8 YBL078C, ATG14 YBR128C); the release is the viable haploid collection, so they are served as deletions like every other strain, flagged here for review.
- Table S9's 174 multidrug-resistant strains (166 in Table S7) are kept: the paper explains them as slow growers ("the control pool undergoes more population doublings than the drug-treated (growth-inhibited) pool"), which the record cannot flag.

### Environment

- Medium: liquid YPD (`restated(YPD_LIQUID, ...)`, the SI's culture sentence); SC for sulfometuron methyl and brefeldin A and SC minus L-lysine for thialysine ("Sulfometuron methyl and brefeldin A were screened in synthetic complete media $( 0 . 6 7 \%$ yeast nitrogen base without amino acids, $0 . 2 \%$ amino acid add back) and thialysine was screened in synthetic complete media lacking lysine."). The SC sentence names no carbon source, so the shared SC's 20 g/L glucose is the library's, not Parsons'.
- 30 C, aerobic (shaken), inoculum OD600 0.025, pre-culture `frozen_stock`, duration 38 h ("At 14 hours (T1) ODs were recorded and cultures where the growth of the drug-treated culture was inhibited by $20 \mathrm { - } 3 0 \%$ compared to the solvent control were diluted back to $\mathrm { O D } _ { 6 0 0 } = 0 . 0 2 5$ and incubated for a further 24 hours (T2) with fresh compound."). Vessel, working volume ("5 or 10 mls") and shaking speed are gaps. The 20 to 30% inhibition acceptance rule has no `DoseBasis` member; it is documented, and the dose is Table S1's released value.
- Compound: one `SmallMoleculePerturbation` per condition. Identity is the PubChem compound the Table S1 CAS number resolves to, through `compound_identity_inputs/parsons2006.txt`: 33 new table rows, 11 labels joined to existing rows as synonyms (the diff against the previous table: 56 keys added, 0 renames, 0 keys removed, 0 identifier changes). So the weighed salt is the compound (harmine hydrochloride, not harmine; the thialysine hydrochloride of CAS 4099-35-8, not the SGA media's thialysine row). Cisplatin joins the existing `azane;dichloroplatinum` row, whose InChIKey equals the CAS compound's. Alamethicin (CID, "A mixture of alamethicin analogs") and oligomycin (ChEBI:84314, no PubChem compound for CAS 1404-19-9) are mixtures. Basiliskamide is basiliskamide A from the Methods. One served loader passes a label that now resolves: Girgis 2009's "doxycycline hyclate" gains CID 54705095 (CI caught it; its test, docstring and note are updated), so the Girgis 2009 dev store needs a rebuild before KG 4.0, which no schema fingerprint flags. A grep of the 56 added keys over the loaders, `media.py` and the input lists finds no other.
- Dose: Table S1 "Inhibitory Concentration" ("Inhibitory concentration refers to the screening concentration."), exact decimal conversion; 5.5 pM rapamycin and 71 pM tunicamycin are stated as 0.0055 and 0.071 nM (no pM unit); MMS's cell is the number 4e-05 under the format `0.000%`, stored as 0.004 basis-free `percent`. Coniine and 2-hydrazinoethanol are released as "5% (v/v)" and stored so.
- Vehicle: the Methods solvent of each compound at the 1% of the solvent control (DMSO by default; water for 18; ethanol for artemisinin, brefeldin A and phenanthroline; methanol for wortmannin; 0.9% sodium chloride solution for cisplatin). Water and the saline are plain vehicles (`compound=None`): the compound table has no water row.
- Reference: the solvent-control pool in each medium, log2 ratio 0.

### Verification, L0 to L4

`ENVIRONMENT_RESPONSE_DATASETS["env_chemgen_parsons2006"]` (streamed), run alone through `torchcell.verification.runners.run_environment_response` on the store built at `1a4a9381a` (report `preprocess/verification_report.json`; about 95 minutes, almost all of it L0 schema validation at about 16 ms a record). All rows pass:

| level | rule | result | message |
|---|---|---|---|
| L0 | `structural` | pass | 291673 records validated |
| L1 | `count` | pass | observed 291673, expected 291673 |
| L1 | `pair_uniqueness` | pass | 291673 unique (study, strain, condition) records, one each |
| L2 | `value_fidelity` | pass | 291673 values checked |
| L2 | `se_nonnegative` | pass | 0 values checked |
| L2 | `interval_orientation` | pass | 0 of 0 stored intervals do not bracket their value, as declared |
| L3 | `measurement_type_consistent` | pass | single measurement_type: 'log2_ratio' |
| L3 | `reference_zero` | pass | numeric rule: reference response == 0 for all 291673 records |
| L3 | `environment_perturbed` | pass | all 291673 experiments carry an environmental edit (perturbation, non-baseline temperature, or non-baseline media; baseline temp=30.0, media='YPD (yeast extract / peptone / dextrose), liquid') |
| L1 | `provenance_gaps` | pass | 2345593 documented provenance gaps over 291673/291673 records; 4 deferred field(s): ['barcode', 'deleted_span', 'downtag_barcode', 'relation']; 10784828 undeclared None values over 26 carrier fields (top: Compound.inchi x1736747, Compound.chebi_id x1225924, Compound.smiles x570990, Compound.pubchem_cid x567183, Compound.inchikey x558994) |
| L1 | `canonical_gene_names` | pass | 4096 systematic names, one canonical spelling each, each current in the genome |
| L2 | `uncertainty_sanity` | pass | 0 labeled uncertainties, none a zero dispersion; 291673 records report n_samples >= 2 with no uncertainty |
| L3 | `compound_identity` | pass | environment edits: 508624 compound references carry a structure identifier; 0 declare a typed gap (0 distinct compounds, unencodable) |
| L3 | `media_compound_identity` | pass | medium components: 665086 compound references carry a structure identifier; 0 declare a typed gap (0 distinct compounds, unencodable) |
| L3 | `media_membership` | pass | 0 records on a shared MEDIA_LIBRARY medium, 291673 on a medium deriving from one (3 distinct media) |
| L4 | `gene_containment_sgd` | pass | 1.000 of 4096 measured genes are S288C reference genes (>= 0.9) |
| L4 | `current_genome_genes` | pass | every one of the 4096 measured systematic names is a gene of the current genome |

The `uncertainty_sanity` line is expected: every record states `n_samples = 2` (the two chips) and the release carries no dispersion, which the records gap explicitly.

### Raw mirror (`$DATA_ROOT/torchcell-raw/parsonsExploringModeofActionBioactive2006/`, `manifest.json`)

| path | table | bytes | sha256 |
|---|---|---|---|
| `data/mmc2.xls` | S1, compounds, CAS, doses | 45,056 | `959f14d2dab418a9b1dd1125e302b2e28c1de372d8d7ee249d369d30fb518353` |
| `data/mmc3.xls` | S2, multidrug-sensitive strains | 27,648 | `766140a410419b19456235db2e4909e56b24e3523bcc0f6f7aa698ae13eb6870` |
| `data/mmc5.xls` | S4, -log10 p | 5,171,712 | `0446eae63b5b08af7a1a94fa3b8d42d70f1c75b55a60e0aa6e03fea189753a01` |
| `data/mmc8.xls` | S7, log2 ratios | 6,239,232 | `564e94c963424687edc24ed23b6e3c6882b70c246a2524057cf1dc5f393c848f` |

Each was deposited from the literature mirror's bytes (`si/si2.xls`, `si3`, `si5`, `si8`), with the Elsevier retrieval (`torchcell.literature.retrieve.elsevier_mmc`, PII S0092867406009068) recorded. The Supplemental Data PDF was OCR'd into the literature mirror for the quotes (`scripts/lit_capture_si.py parsonsExploringModeofActionBioactive2006 --ocr-recorded`, MinerU 2.7.6, `si/si1.md`). No genomes-tier deposit: the host is the S288C reference.

### Duplication with YeastPhenome (#1000), L4 cross-check

[[experiments.database.scripts.parsons2006_yeastphenome_crosscheck]]: the yeastphenome dev store holds 0 Parsons rows. Against yp-data v1.0's `parsons_boone_2006_value.txt`, 291,457 of 291,530 joined cells are the exact negation of the stored ratio (Pearson -0.999993); the other 73 are all YDL039C. Against its NPV (`_valuez.txt`), Pearson -0.917, Spearman -0.974. So yp-data flips Table S7's sign, and PMID 16901791 has to stay off `SCREENS`.

### Owner decisions taken by recommendation

1. Stored the released log2(control/treated) unnegated, orientation in `units` (Hillenmeyer HET precedent), rather than negating to the `log2_ratio` docstring's treatment/control; the conflict is #1031.
2. Dropped the 7 crude extracts, agelasine E and stichloroside under `compound_without_a_structure_identifier`, the rule Wildenhain, Hoepfner, Hillenmeyer and Vanacloig apply, rather than serving name-only compounds. 36,437 records.
3. Compound identity from the Table S1 CAS number (the salt weighed), not from the Table S7 column name (often the parent). Measured: for 8 conditions the column name resolves to a different existing row than the CAS number does, 7 parent-versus-salt (amiodarone, chlorpromazine, desipramine, dyclonine, emetine, harmine, pentamidine) and usnic acid (unspecified stereo versus the (R)-(+) form Table S1 names), so those 8 nodes do not join a parent-keyed record of another dataset.
4. `n_samples = 2` technical replicates (two fluor-reversed chips) rather than 1 screen.
5. One reference per medium, not per vehicle: the reference record carries the medium, the vehicle sits on the compound edit.
6. The 24 SGD-essential ORFs are served as deletions.

### Schema impact

`scripts/schema_impact_check.py --base origin/main`: "No schema contract changes vs origin/main." The compound-table rows are not in the schema closure; they add keys only (the diff above).

Candidate gate: `python -m torchcell.candidates gate --citation-key parsonsExploringModeofActionBioactive2006 --table yeast --phenotype-class EnvironmentResponsePhenotype --write` at `1a4a9381a`: G1 to G5 pass, outcome admissible (G3 now reads library + raw, 27 files); the audit's evidence quotes were kept. `candidates audit`: ok 79. The G4-value overlap of the release against this store's own records (291,457 cells, every column matched at max abs diff 0) reads as "subsumed" by design, so it is not folded into the verdict.
