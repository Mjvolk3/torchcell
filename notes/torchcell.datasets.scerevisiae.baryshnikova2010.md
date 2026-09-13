---
id: dd6ag78l42e4kqq8if0h77g
title: Baryshnikova2010
desc: ''
updated: 1789271834379
created: 1789271834379
---

## 2026.09.12 - Provenance rebuild: sourced constants, split temperature, publisher raw mirror

Loader: `torchcell/datasets/scerevisiae/baryshnikova2010.py`. Adapter:
`torchcell/adapters/baryshnikova2010_adapter.py` +
`torchcell/adapters/conf/smf_baryshnikova2010_adapter.yaml`. Dev build:
`$DATA_ROOT/data/torchcell/smf_baryshnikova2010`, 5,993 records, verifier PASS at L0-L4.

### What changed

The loader carried zero provenance objects: `n_samples = 80`, `30 C`, `BY4741`,
`bootstrap_se` and the WT reference of 1.0 were bare module constants with prose comments,
the raw file had no recorded retrieval, and the verifier's record-count oracle was fitted
to a previous build. Every one of those is now either a `SourcedValue` pinned to a
sha256-verified mirrored artifact with a verbatim quote, or a typed `ProvenanceGap`.

- 13 `SourcedValue` constants, exported as `SOURCED_VALUES` and re-audited by
  `test_every_sourced_value_audits_against_its_pinned_artifact` (the audit re-opens each
  artifact, checks the sha256 still matches and that the quote is still present).
- Three mirrored papers carry them: this paper (`paper.md`, `si/si1.md`), Tong and Boone
  2006 (`yantongSyntheticGeneticArray2006/paper.txt`) and Costanzo 2016
  (`costanzoGlobalGeneticInteraction2016/si/si1.md`).

### The array side / query side split, and why `n_samples = 80` shrank

SI Supplementary Table 2 describes two measurement designs in one cell. The ARRAY side
crosses a neutral natMX query against the nonessential deletion collection and is the only
side with a released screen count, verbatim:

> Thus, colony sizes derived from these control screens (after applying standard normalization procedures) reflect array strain single mutant fitness. Fitness measurements were based on 80 control screens.

The QUERY side ("Query strain single mutant fitness was obtained in a similar manner.")
states no screen count. The released ids say which side a row is on: `_tsq` and `_DAmP`
are the SGA QUERY strain labels in the same lab's nomenclature, which the Costanzo 2016
SI spells out for both collections. So 80 is stored for the 4,606 kept deletion rows and
the 1,387 kept query-side rows (1,081 DAmP + 306 TS) carry
`ProvenanceGap(field="n_samples", reason=not_reported_by_primary)` instead of the
extrapolated 80.

Correction to the serve-50 review on one point of reasoning, not of outcome: the review
called the query-side attribution its own inference. The main text actually reads

> We measured colony sizes for 4,635 viable deletion mutants and 1,388 temperature-sensitive or hypomorphic alleles of essential genes after SGA analysis using a neutral (control) query mutation

which is the array-side phrasing applied to all 6,023 rows. The SI's two-sided table plus
the `_tsq` / `_damp` labels are the precise statement, so the split stands, but the main
text is looser than the SI and the note records that rather than hiding it.

### Temperature is split 30 C / 26 C, and the split was back-solved

Baryshnikova 2010 states no SGA incubation temperature anywhere; the Online Methods defer
("as described previously1,7") to Baryshnikova et al. Methods Enzymol. 470 and Costanzo et
al. Science 327, NEITHER of which is mirrored. The two mirrored papers of the same lab
describing the same SGA system fix it:

- deletions and DAmP -> 30 C. Tong and Boone 2006 step 17 pins onto the (SD/MSG)
  -His/Arg/Lys +canavanine/thialysine/G418/clonNAT plate (which is `SGA_DM_SELECTION`) and
  step 18 is "Incubate the kanR/natR-selection plates at 30°C for 2 d." Corroborated by
  Costanzo 2016: "Double mutant selection plates involving a nonessential deletion mutant
  query strain and the DMA were incubated at 30 °C."
- TS alleles -> 26 C. Costanzo 2016: all SGA selection steps involving a TS allele run at a
  permissive 22 C except the final haploid double-mutant selection at a semipermissive
  26 C, and that holds for a TS allele used "either as a query or an array mutant".

Because this is a cross-paper deferral, it was BACK-SOLVED rather than assumed. 291 of the
306 TS rows share their exact `strain_id` with a Costanzo 2016 SMF row, and Costanzo
releases SMF at both 26 C and 30 C:

| Baryshnikova TS SMF vs | n | Pearson r | mean absolute difference |
|---|---|---|---|
| Costanzo 26 C column | 291 | 0.8777 | 0.0577 |
| Costanzo 30 C column | 291 | 0.7105 | 0.1401 |

In the same Costanzo release the 26 C and 30 C columns are EXACTLY equal for every
deletion, `_sn` and `_damp` strain (3,876/3,876, 3,845/3,845, 733/733) and never equal for
a TS strain (0/792 `_tsa`, 0/961 `_tsq`). Only TS alleles are temperature-resolved at all,
which is what makes 30 C right for the non-TS rows and 26 C right for the TS rows.
Measured by `scratchpad/serve50/scripts` against
`data/torchcell/smf_costanzo2016/raw/strain_ids_and_single_mutant_fitness.xlsx`.

Consequence for the graph, measured on the rebuilt store (sha256 of `model_dump`, the way
`CellAdapter` computes node ids): 2 distinct `environment` nodes, 2 `temperature` nodes,
1 `media` node and 2 `experiment reference` nodes across the 5,993 records, instead of one
of each. The two references are the 30 C one (shared by the deletion and DAmP rows, whose
reference phenotypes are identical) and the 26 C one.

| kind | records | stored `n_samples` | temperature |
|---|---|---|---|
| `sga_kanmx_deletion` | 4,606 | 80 | 30.0 C |
| `damp` | 1,081 | gap | 30.0 C |
| `temperature_sensitive_allele` | 306 | gap | 26.0 C |

### Raw mirror: the publisher's Supplementary Data 1 replaces an unprovenanced text export

The loader used to read `torchcell-library/.../data/S1_SMF_standard_100209.txt`, a file
whose retrieval is recorded nowhere (it is absent from that key's `manifest.json`). It now
reads the publisher's Supplementary Data 1 workbook, sheet `S1_SMF_standard_100209`,
deposited at `$DATA_ROOT/torchcell-raw/baryshnikovaQuantitativeAnalysisFitness2010/` with
a `manifest.json` recording `source_url`, retriever, params, sha256 and `retrieved_at`.
The Springer ESM MediaObject was located by hashing candidates until one matched the
mirrored bytes: MOESM168 is Supplementary Data 1 (sha256 `086bfadf...`), MOESM167 is the SI
PDF, MOESM169 / MOESM170 are Supplementary Data 2 / 3 and MOESM171 is the software zip. The
retrieval was then run for real through
`torchcell.literature.retrieve.springer_esm` and re-hashed before deposit.

Measured correction to the old docstring: it claimed the `.txt` and the `.xls` "differ by
up to 0.031". They do not. Sorted on (id, fitness, se) the two files are the same data,
every fitness value bit-identical and the SEs agreeing to 1.7e-18; they differ only in ROW
ORDER. The 0.031 was an artifact of joining on the duplicated `YDL227C` id, whose two rows
(1.000155 and 1.031582) cross-join.

### Drop rule and the count oracle

`DROP_RULE`: a released allele whose ORF token does not resolve to a gene of the current
R64 annotation (a retired, merged or dubious ORF) is dropped. The 30 such ids are frozen in
`_UNRESOLVABLE` with their kind, `process()` raises if the build's drop set differs, and
`EXPECTED_RECORDS = 6023 - len(_UNRESOLVABLE) = 5993` is now anchored to the RELEASE rather
than fitted to whatever genome build is on disk. Counts are written to
`preprocess/dropped_records.json` with the rule text.

| | released | dropped | kept |
|---|---|---|---|
| kanMX deletion | 4,635 | 29 | 4,606 |
| DAmP | 1,082 | 1 | 1,081 |
| TS allele | 306 | 0 | 306 |
| total | 6,023 | 30 | 5,993 |

### Other corrections

- The docstring said the environment was "Media(YEPD, solid)" while the code used
  `SGA_DM_SELECTION`. The code was right; the docstring now matches it.
- BY4741 was attributed to a sentence that does not support it. The only place the mirror
  names the array background is the SI's plate-border-control row ("isogenic to BY4741");
  the main text's BY4741 sentence is scoped to the serial-dilution, actin-staining and
  RIM101 follow-up strains. `_SV_BACKGROUND_STRAIN` quotes the border-control sentence and
  its `note` records the scope limit.
- The reference phenotype no longer claims `n_samples = 80`. The reference fitness of 1.0
  is the mode-normalization convention, not an averaged wild-type measurement, so both
  `n_samples` and `sample_unit` are typed gaps on every reference.

### Verifier

Run from a scratch script with the same parameters `run_fitness` uses (genome resolver +
SGD gene set + `MIN_RNASEQ_GENE_CONTAINMENT`):

```
n records 5993
smf_baryshnikova2010: PASS
  [ok] L0 structural: 5993 records validated
  [ok] L1 count: observed 5993, expected 5993
  [ok] L1 pair_uniqueness: 5993 unique (strain, environment) records, one each
  [ok] L1 provenance_gaps: 7380 documented provenance gaps over 5993/5993 records; 0 deferred field(s): []; 185783 undeclared None values over 8 carrier fields (top: Compound.inchi x71916, Compound.chebi_id x53937, Compound.smiles x17979, Compound.inchikey x11986, Compound.pubchem_cid x11986)
  [ok] L1 canonical_gene_names: 5436 systematic names, one canonical spelling each, each current in the genome
  [ok] L2 value_fidelity: 5993 values checked
  [ok] L2 se_nonnegative: 5993 values checked
  [ok] L2 uncertainty_sanity: 5993 labelled uncertainties, none a zero dispersion; 0 records report n_samples >= 2 with no uncertainty
  [ok] L3 reference_one: reference fitness == 1.0 for all 5993 records
  [ok] L3 compound_identity: environment edits: 0 compound references carry a structure identifier; 0 declare a typed gap (0 distinct compounds, unencodable)
  [ok] L3 media_compound_identity: medium components: 41951 compound references carry a structure identifier; 0 declare a typed gap (0 distinct compounds, unencodable)
  [ok] L3 media_membership: 5993 records on a shared MEDIA_LIBRARY medium, 0 on a medium deriving from one (1 distinct media)
  [ok] L4 gene_containment_sgd: 1.000 of 5436 measured genes are S288C reference genes (>= 0.9)
  [ok] L4 current_genome_genes: every one of the 5436 measured systematic names is a gene of the current genome
```

The 7,380 documented gaps are 1,387 query-side `n_samples` gaps plus the 5,993 `inchikey`
gaps the shared medium's agar component carries (agar is a `RESOLVED_MIXTURE`: ChEBI:2509,
no single-molecule InChIKey). `media_compound_identity` FAILED on the pre-rebuild store
(8 name-only medium components x 5,993 references) and passes now, because the rebuild
picked up the retyped media library.

### Open flags

- **The medium's VALUE changed when the library was retyped**, so this dataset's `media`,
  `temperature` and `environment` node ids no longer coincide with the SERVED Costanzo 2016
  SMF ones the review measured as shared. `kg_manifest` does not fingerprint
  `torchcell/datamodels/media.py`, so the admission gate cannot see this. Orchestrator
  decision, not a per-dataset one.
- **TS allele names are NOT imported (option A).** `perturbed_gene_name` stays the
  systematic ORF and `strain_id` is the cross-dataset strain key. Deriving the allele names
  (`cdc24-11`, `mcm2-1`, ...) for the 291 TS strains shared with Costanzo 2016 would make
  those perturbation nodes coincide with served nodes, but the Costanzo strain table lives
  only in the dev tree (`data/torchcell/smf_costanzo2016/raw/`), not in a raw mirror, so a
  `SourcedValue` pointing at it could not be audited and 15 of 306 would keep the ORF
  anyway. Revisit once that table is deposited.
- **Two genome hubs for one SGA system.** Costanzo and Kuzmin carry `S288C`; this dataset,
  O Duibhir, Ohya and da Silveira carry `BY4741`. Unchanged here; flagged for the schema
  layer.
- **The DAmP 30 C is the weakest link in the temperature chain.** The 26 C exception is
  explicitly scoped to TS alleles and DAmP is not a TS allele, so DAmP falls to the default
  SGA 30 C. It could not be back-solved: Costanzo's `_damp` rows carry the same SMF in both
  temperature columns (733/733 exactly equal), so the release gives no discriminating
  signal either way.
