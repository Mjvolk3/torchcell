---
id: 0s6gui1z01a87158vovj3e3
title: Wildenhain2015
desc: ''
updated: 1789272406505
created: 1789272406505
---

## 2026.09.12 - Serve-50 rebuild: AID description mirrored, screens counted, category mapped

Loader: `torchcell/datasets/scerevisiae/wildenhain2015.py`. Adapter:
`torchcell/adapters/wildenhain2015_adapter.py` +
`torchcell/adapters/conf/env_chemgen_wildenhain2015_adapter.yaml`. Tests:
`tests/torchcell/datasets/scerevisiae/test_wildenhain2015.py`,
`tests/torchcell/adapters/test_wildenhain2015_adapter.py`.

### Why the previous build could not be served

It failed L0 on 100% of its records (`Media.is_synthetic Field required`). Beyond that:
the medium was a free-text `Media(name="synthetic complete (SC), 2% glucose")` that did not
join the shipped `SC`; `Temperature(value=30.0)` came from a PubChem AID description that
was in no mirror and carried no hash; `n_samples = 2 x released rows` counted OD reads of
re-exported duplicate rows; no uncertainty was stored at all; 16 ORFs were spelled two ways
(`TOR1` and `Tor1`), splitting the perturbation node; the released Inactive / Active /
Inconclusive curation verdict was ingested silently; and `download()` fetched a 151 MB FTP
container whose own sha256 was never pinned.

### Raw mirror and provenance chain

`$DATA_ROOT/torchcell-raw/wildenhainPredictionSynergismChemicalGenetic2015/` holds the two
files the loader consumes:

- `data/1159580.csv.gz`, sha256 `c461c679...`, retrieved with
  `torchcell.literature.retrieve.zip_member` from the NCBI FTP range archive
  `1159001_1160000.zip`, whose OWN sha256
  (`d1fd5dc2bf7c526ad9845e0a14ae9981256fb820aaf4228b48a3ba0724ee59b0`, measured 2026-09-13)
  is asserted before the member is read, so a re-packed upstream archive fails loudly.
- `data/aid_1159580_description.json`, sha256 `23c5f8c5...`, the PUG-REST assay
  description. This is the artifact that turns the screening protocol from an unsourced
  default into a sourced value.

`download()` links both out of the mirror and verifies each sha256.

The AID description is the ONLY source for several values: "All strains were grown and
screened in synthetic complete (SC) medium with 2% glucose." (the paper's own medium
sentence has "All fungal species" as its subject, which in context follows the fungal
PATHOGEN isolates), and "Plates were incubated at 30 C without shaking for approximately
18 h or until culture saturation was achieved for the solvent controls." Grepping the
paper for a 30 C statement returns nothing; its only temperature is 37 C, for HeLa/HEK
cells. The 13 module-level `SourcedValue` constants audit clean against the mirror each
one names (AID description in the raw mirror, `paper.md` sha256 `f46409eb...` in the
library mirror), which the loader test asserts.

### Screens, and the measured correction to the previous build

The export re-emits the same datapoint under two gene-symbol spellings (`MDH1` and `mdh1`):
of the 46,195 cells with more than one released row, 33,483 contain byte-identical
duplicates. Two measurements settle how to count:

- within a cell, rows sharing a `z_score` share every other data column too (they differ
  only in the `sym` casing), and
- deduplicating on the full datapoint identity and deduplicating on the z string give the
  IDENTICAL partition, with zero multi-datapoint cells sharing a z.

So the z string is the datapoint key. After deduplication the screens-per-cell histogram is
{1: 412,368, 2: 14,579, 3: 1,617, 4: 7, 5: 2}, i.e. 16,205 multi-screen cells, not the
46,195 the raw row count suggests. `n_samples` is now the number of contributing SCREENS
with `sample_unit=screen` (the AID says the z is computed from "normalized average reads
per screen", so the technical-duplicate OD pair is already inside one z), the uncertainty
is the sample SD across screens, and a single-screen cell carries a typed `ProvenanceGap`
on `environment_response_uncertainty` and `environment_response_se` instead of a silent
None. A sample SD of exactly 0 is impossible by construction and the loader raises if one
appears.

This also removes the `non replicate` flag's effect on `n_samples`: counting screens rather
than reads is what the flag broke. 2,320 kept cells have every contributing screen flagged
(the two OD reads of each disagreed); they are KEPT and recorded here rather than dropped,
since the release retained them after its own "> 3 MAD" outlier filter.

### Released curation verdict -> ResponseCategory

The `PUBCHEM_ACTIVITY_OUTCOME` and `bioactivity` columns are now typed rather than
discarded, sourced from the AID's own column definition ("sensitive if compound decreases
fitness or resistant if compound increases fitness compared to negative control"):

| released (outcome, bioactivity) | `category` | cells |
|---|---|---|
| Inactive, "" | `no_change` | 381,659 |
| Active, sensitive | `sensitive` | 43,785 |
| Active, resistant | `resistant` | 920 |
| Inconclusive, sensitive / resistant | `not_determined` | 829 |
| screens disagree | `not_determined` | 1,380 |

`category_label` keeps the source words verbatim ("Active / sensitive",
"Active / Inactive / sensitive" when screens disagree), so the mapping is auditable and
the 976 Inconclusive rows are visible instead of silent.

### Retention rules and counts

428,573 (ORF, compound-identity) cells, kept **428,206**. The one rule
(`preprocess/dropped_records.json`): `compound_without_a_structure_identifier` removes the
367 cells of the 5 SID-only compounds (no CID and no SMILES, so no InChIKey, CID or ChEBI
id exists). Nothing else is dropped: all 5,173 CIDs resolve through the pinned identity
table to a canonical PubChem name AND an InChIKey, which is what moves 427,177 records from
"CID + SMILES only" to joinable, and all 242 released ORFs are current R64 genes.

### Encoding

- **Medium**: the shared `MEDIA_LIBRARY["SC"]` object, which now carries its D-glucose at
  20 g/L (= the AID's 2%), so the medium matches a library object exactly instead of being
  free text.
- **Compounds**: `resolved_compound(identity, pubchem_cid=..., smiles=...)`. `Compound.name`
  is the canonical PubChem title (`vanillin`), not the identifier-posing-as-a-name
  `"CID 1183"` the previous build stored. Dose `20 uM`; `Solvent(name="DMSO")` with a typed
  vehicle `Compound` and `percent=None` (the final DMSO fraction is not released, and
  back-computing it from "2 uL of 1 mM stock into 100 uL" would assume a neat-DMSO stock).
- **Genotype**: `BarcodedKanMxDeletionPerturbation` with `collection="Euroscarf deletion
  collection"` and `barcode=None` (the release publishes no barcode).
  `perturbed_gene_name` is the GENOME's standard name, which collapses the `TOR1` / `Tor1`
  split.
- **Reference**: BY4741 at z = 0 in the SAME compound environment. That is now a sourced
  statement rather than an assumption: the z is standardized within a screen, so 0 is that
  screen's own normalized-growth center. The reference `units` says exactly that.
- `Environment.duration_generations` carries a typed `ProvenanceGap`: an 18 h OD growth to
  saturation doses exposure in hours, not doublings.

### Verifier result (verbatim)

Run with the STREAMING gate (`stream: True`), which the runners entry needs for this size;
report written to `preprocess/verification_report.json`. The rebuild also collapses the
stale 42 GB `experiment_reference_index.json` (5,178 references x 428,573 booleans, a
pre-sparse artifact) to 212 MB: the whole dataset tree is 2.0 GB, down from 41 GB.

```
env_chemgen_wildenhain2015: PASS
  [ok] L0 structural: 428206 records validated
  [ok] L1 count: observed 428206, expected 428206
  [ok] L1 pair_uniqueness: 428206 unique (strain, condition) records, one each
  [ok] L1 provenance_gaps: 1252226 documented provenance gaps over 428206/428206 records; 0 deferred field(s): []; 26734849 undeclared None values over 4 carrier fields (top: Compound.inchi x14559004, Compound.chebi_id x11335629, EnvironmentResponsePhenotype.screen_id x428206, EnvironmentResponsePhenotype.environment_response_uncertainty_type x412010)
  [ok] L1 canonical_gene_names: 242 systematic names, one canonical spelling each, each current in the genome
  [ok] L2 value_fidelity: 428206 values checked
  [ok] L2 se_nonnegative: 16196 values checked
  [ok] L2 uncertainty_sanity: 16196 labelled uncertainties, none a zero dispersion; 0 records report n_samples >= 2 with no uncertainty
  [ok] L3 measurement_type_consistent: single measurement_type: <MeasurementType.z_score: 'z_score'>
  [ok] L3 reference_zero: numeric rule: reference response == 0 for all 428206 records
  [ok] L3 environment_perturbed: all 428206 experiments carry an environmental edit (perturbation, non-baseline temperature, or non-baseline media; baseline temp=30.0, media='SC (synthetic complete: YNB + 20 amino acids + uracil + adenine + glucose)')
  [ok] L3 compound_identity: environment edits: 856412 compound references carry a structure identifier; 0 declare a typed gap (0 distinct compounds, unencodable)
  [ok] L3 media_compound_identity: medium components: 13702592 compound references carry a structure identifier; 0 declare a typed gap (0 distinct compounds, unencodable)
  [ok] L3 media_membership: 428206 records on a shared MEDIA_LIBRARY medium, 0 on a medium deriving from one (1 distinct media)
  [ok] L4 gene_containment_sgd: 1.000 of 242 measured genes are S288C reference genes (>= 0.9)
  [ok] L4 current_genome_genes: every one of the 242 measured systematic names is a gene of the current genome
```

Two lines to read carefully. `L2 se_nonnegative: 16196 values checked` is no longer the
vacuous "0 values checked" of the previous build: 16,196 of the 16,205 multi-screen cells
carry a derived SE (the other 9 are cells of the dropped SID-only compounds). And
`L2 uncertainty_sanity` now has something to check at all, because every labelled
dispersion comes from at least two distinct screens.

### Open flags

- The kanMX marker is a property of the Euroscarf MATa collection (Winzeler 1999 /
  Giaever 2002), not something either source names. The `collection` label records what IS
  sourced; the marker assertion is inherited from the leaf class.
- The released `sym == 'wild type'` rows (1,940, plus `wtn01`, `YGL11`, `NNK1` and `TSCII`,
  7,296 non-strain rows in total) are NOT ingested. They are strain measurements, not the
  z-score baseline (their z has mean -2.31, median -0.071, sd 7.82), so ingesting them
  needs a wild-type genotype decision the orchestrator has not made.
- The 195-strain sentinel roster is in Table S3, which cell.com does not serve to a script;
  all 242 released ORFs are served and the `si_expected` manifest entry records the gap.
  Coverage is very unbalanced: 63 strains have fewer than 1,000 cells.
- The `p_value`, `normalized OD average`, `raw OD read 1/2` and `cryptagen` columns are
  still not stored.
