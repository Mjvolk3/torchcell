---
id: 12u4ixbhhu24gywz8m43e8r
title: Fuhrer2017
desc: ''
updated: 1791383119041
created: 1791383119041
---

## 2026.10.07 - Loader, sourcing decisions and the first build

`torchcell/datasets/ecoli/fuhrer2017.py`, class `MetabolomeFuhrer2017Dataset`
(`REFERENCE_STRAIN = "BW25113"`, takes `ecoli_genome`), row 1 of
[[plan.bacteria-ontology-genome]] section 6. Template: [[torchcell.datasets.scerevisiae.mulleder2016]]
for the metabolome shape, [[torchcell.datasets.scerevisiae.vanacloig2022]] for the raw
mirror wiring. Shared API: [[torchcell.datasets.bacteria_common]]. Every number below is
read from the build's own ledgers in `$DATA_ROOT/data/torchcell/metabolome_fuhrer2017/preprocess/`
(`dropped_records.json`, `identifier_reconciliation.json`, `ion_alignment.json`,
`zscore_scale.json`, `verification_report.json`), written by the loader and by
`fuhrer2017.run_verification`.

### Where the data is

Fuhrer et al. 2017, Mol Syst Biol 13:907, doi 10.15252/msb.20167150, citation key
`fuhrerGenomewideLandscapeGene2017`, `paper.md` sha256
`ec87736bda4d30fa4c5eafa49a5d033bfff4f1ee3d641bbeb4c7f9e5d55f055b`.

The publisher SI does not carry the per-strain metabolome: Table EV1A (`si/si2.xlsx`) is
the strain list with growth rates, Table EV1B the ion list, EV2 an annotation summary,
EV3 (`si4.zip`) the orphan-gene predictions, EV5 the annotation AUCs. The paper puts the
matrices in BioStudies: `Raw data and modified $z$ -scores for positive and negative mode (tabseparated and excel files) can be downloaded from https://www.eb i.ac.uk/biostudies/, accession code: S-BSST5.`
S-BSST5 is a plain HTTPS tree, so every file is `direct_url`.

Raw mirror `$DATA_ROOT/torchcell-raw/fuhrerGenomewideLandscapeGene2017/` (written by
`deposit_raw_mirror`, idempotent by sha256, refuses a differing file), exactly the files
the loader reads:

| file | source | sha256 | read for |
|---|---|---|---|
| `S-BSST5.json` | S-BSST5 study record | `65e4cb0e...` | the design attribute and the file descriptions (quotes) |
| `zscore_neg.tsv` | S-BSST5 | `ecb93f2c...` | 3,169 ions x 3,807 columns |
| `zscore_pos.tsv` | S-BSST5 | `6059a555...` | 4,365 ions x 3,807 columns |
| `sample_id_zscore.xls` | S-BSST5 | `511fc0d0...` | the z-score column names |
| `sample_id_all.xls` | S-BSST5 | `b420f178...` | the raw matrices' column names, to count replicates |
| `neg_ionMz.xls`, `pos_ionMz.xls` | S-BSST5 | `e3722fec...`, `a9aeb65d...` | each row's m/z |
| `si2.xlsx` (Table EV1) | PMC bucket, `pmc_cloud` (the literature mirror's capture) | `cda4dc21...` | EV1A name to JW id and Blattner id; EV1B ion index and m/z |

Not mirrored, because nothing reads them: the raw intensity matrices
(`rawdata_*_all.tsv`, 219 MB and 300 MB) and the KEGG annotation files. Recorded in the
manifest's `si_expected`.

### Strain: BW25113, by deferral

Fuhrer never names the background: `Escherichia coli wild-type and 4,320 deletion mutants (Table EV1) from the KEIO knockout collection (Baba et al, 2006)`.
Baba 2006 (`babaConstructionEscherichiaColi2006/paper.md`, sha256 `ca71475b...`) does:
`The Keio collection is comprised of 3985 deletions in duplicate (7970 total) of E. coli K-12 strain BW25113 (Datsenko and Wanner, 2000)`.
Records pin `assembly_reference("BW25113")` = `ecoli_K12_BW25113_ASM75055v1`,
`GCA_000750555.1`. Cassette, also from Baba:
`Open-reading frame coding regions were replaced with a kanamycin cassette flanked by FLP recognition target sites`.

### Identifiers: the JW id, not the deposit's gene name

The z-score columns are labelled by gene name only. On the BW25113 GenBank annotation 11
of the kept names do not resolve to the locus of their own strain's JW id: five resolve
to another locus (`ecpD` JW0136 is BW25113_0140 but the name now means BW25113_0290;
`rarA`, `rbn`, `ykgB`, `zupT`), four are ambiguous (`pin`, `rffT`, `spr`, `ygaD`), two
retired (`tiaE`, `yiaI`). So the name is used only as the join key to Table EV1A, and the
Keio JW id (the strain's own accession, which BW25113's GenBank file carries as a
`gene_synonym`) goes through `reconcile_locus_tags`. Stored per record:
`systematic_gene_name` = the BW25113 locus tag, `perturbed_gene_name` = that locus's own
GenBank symbol (only if it resolves back to the tag), `construction.strain_accession` = the
JW id, `collection = "KEIO knockout collection"` (Fuhrer's wording), `cassette` (Baba's
wording).

JW reconciliation on BW25113 (3,781 single-entry strains):

| status | count | layer |
|---|---|---|
| renamed (gene) | 3,620 | gene synonym |
| non_gene_feature (pseudogene) | 115 | gene synonym |
| retired | 46 | not found |
| ambiguous / collision / case-insensitive | 0 | |

Resolved fraction 3,735 / 3,781 = 0.988, above the stated stop threshold of 0.95
(`MIN_RESOLVED_FRACTION`; BW25113's GenBank file carries 4,334 JW synonyms, so a fraction
below 0.95 would mean the wrong file or strain). The 46 retired JW ids are dropped, not
mapped. Each dropped item in `dropped_records.json` says what its deposit name resolves
to and whether a kept record already holds that locus:

| the deposit name resolves to | dropped JW ids |
|---|---|
| a pseudogene locus a kept record already holds | 29 |
| a pseudogene locus no record holds | 6 (`insO`, `ybfQ`, `yedM`, `yjgW`, `yjiV`, `ykgN`) |
| a gene locus no record holds | 3 (`hokC`, `ymfJ`, `ymgG`) |
| nothing (retired) | 8 |

The 29 are Keio strains with their own b-numbers whose names the BW25113 annotation puts
on the same locus as a kept strain (`yaiU` with `yaiT` on BW25113_4580; `ygaR`, `yqaC`,
`yqaD` with `ygaQ` on BW25113_4462), so mapping them by name would give two distinct Keio
deletions one genotype. The other 9 with a name match could be placed by name, but this
loader does not use names as identity (they disagree with the JW id for 11 kept strains,
above). The MG1655 b-number to BW25113 path through the ECK crosswalk was not used either:
the plan requires such a derived mapping to be recorded on the record, and
`BacterialDeletionPerturbation` has no field for it.

### Records and drops

3,807 deposit columns = 3,806 strain names + `wt`. The paper says
`resulting in a final data set with 3,807 mutants`; the deposit's 3,807 includes `wt`.

| rule | records | why |
|---|---|---|
| `pooled_keio_entries` | 25 | the name matches 2 or 3 Table EV1A entries and `sample_id_all.xls` carries 4 columns per entry under it (8 or 12), so the released value summarizes several Keio strains: 20 of different loci (`ilvG` b3767 + b3768, `molR` three loci, ...), `dcuC` two JW ids of one locus, and 4 where one entry is `Eliminated; wrong primers` (`dsbB`, `flgM`, `kptA`, `ymcC`) |
| `jw_id_not_on_a_bw25113_locus` | 46 | above |
| kept | **3,735** | |

Every single-entry strain has exactly 4 raw columns; the build refuses otherwise.

### Phenotype, n and the uncertainty type

`BacterialMetaboliteExperiment` / `BacterialMetaboliteExperimentReference` (the
assembly-pinned pair), phenotype `MetabolitePhenotype`, `measurement_type =
"fia_tof_ms_ion_modified_z_score"`, keys `neg_0001` ... `neg_3169`, `pos_0001` ...
`pos_4365` = mode plus 1-based row = Table EV1B's `Ion Index`. `preprocess/ions.csv`
gives each key's m/z in the deposit and in EV1B. Alignment check (`ion_alignment.json`):
equal counts (3,169 / 4,365, the paper's `3,169 and 4,365 distinct mass-tocharge`), and
the nearest deposit m/z of each EV1B ion is its own row for 3,168 of 3,169 negative and
4,365 of 4,365 positive ions; the negative files differ by median 5.8 mDa (max 10.2), the
positive by 0.9 mDa (max 1.3).

**What the number is.** The paper's definition is per sample: `where $i$ and $j$ denote ion and samples, respectively, and median as well as standard deviation (std) refers to all intensities of ion i in the entire dataset.`
Measured on the deposit (`zscore_scale.json`, enforced at build time): every ion of the
released matrix has median exactly 0 and sample SD 1.0000 (0.9999985 to 1.0000016) over
its 3,807 columns, `wt` included. The stored score is therefore standardized over exactly
the released columns, unitless, and not centered on the wild type.

**Replicate summary, a recorded conflict.** Paper: `Modified $z$ -scores referring to technical and biological replicates are summarized into a unique median modified $z$ -score`.
Deposit (`S-BSST5.json`): `Zscore transformed data matrix of negative ionization mode, replicates are averaged, rows correspond to ions, columns to genes`.
Median or mean cannot be checked: per-replicate scores are not released. Not resolved.

**n_samples.** `n_replicates` = 2 independent clones per strain on every ion. Methods:
`For each knockout mutant, two different clones contained on separate plates in the library were separately processed on different days.`
Checklist (`si/si6.md`, sha256 `58a5a9b0...`): `Two biological replicates (independent clones) were extracted and measured with two technicalreplicates.`
Deposit: `2 Biological and 2 technical replicates` and `4 columns per gene from technical and biological duplicates`.
No range, so no resolution rule was needed; the value is also counted per strain
(`sample_id_all.xls` columns / 2 technical injections = 4 / 2), and technical duplicates
are not counted because they are injections of one extract (`analyzed in technical duplicates`).

**Uncertainty type: none released, a typed gap.** `metabolite_level_se = None` with
`ProvenanceGap(field="metabolite_level_se", reason=not_reported_by_primary)` on every
phenotype. Looked in: the full Methods, the checklist, and every S-BSST5 file
description. The only stated spread is dataset-level and is not stored on records:
`Ninety-nine percent of variability between biological replicates was estimated to be smaller than a $z$ -score of 2.765 (Fig EV2A).`
The raw intensities cannot be re-normalized into per-replicate z-scores (plate drift,
low-pass and harvest-OD LOWESS corrections with unreleased inputs).

**Reference.** The deposit's measured `wt` column, same scale and medium, with
`n_replicates` = 96 on every ion: `wt` has 192 columns in `sample_id_all.xls`, divided by
the two technical injections. This is a back-solve from the released column labels plus
the stated technical duplicates; the paper states no wild-type replicate count.

### Environment

`Environment(media=M9_GLUCOSE_CASEIN_FUHRER2017, temperature=37 C, aerobicity="aerobic")`,
the [[torchcell.datamodels.media]] key assigned to row 1. Temperature and culture:
`Culture volumes of $1 \ \mathrm { m l }$ were incubated in 96-deep well plates at $3 7 ^ { \circ } \mathrm { C }$ with shaking at $3 0 0 ~ \mathrm { { r p m } }$ .`
Harvest by phase (`all samples were harvested during mid-exponential growth phase`), so
`duration_hours` stays None. Nothing varies across records.

### Build and verification

`PYTHONPATH=<wt> python -m torchcell.database.build_dataset_lmdb --dataset MetabolomeFuhrer2017Dataset`:
3,735 records, 1 reference group, 83 s; LMDB 745 MB plus a 236 KB interned env (the `wt`
reference is stored once). Peak RSS 2.8 GB, measured on the first build, which differed
only in the drop-ledger wording and was retired to `/scratch/projects/torchcell-deprecated/`. `python -m torchcell.provenance.build_manifest`
reads `metabolome_fuhrer2017` as fresh.

`fuhrer2017.run_verification()` (the metabolite family verifier of
`torchcell/verification/metabolite.py`, `reference_centered=False`, plus a BW25113 L4 and the
provenance audits): PASS. L0 3,735 records validate; L1 count 3,735 and one record per
strain; L2 28,139,490 finite values; L3 reference finite and key-matched, one
measurement type, 18 of 18 sourced values verbatim at their pinned sha256; L4 3,735 of
3,735 deleted loci are BW25113 GenBank gene rows.

### Open, and what is a gap

- `metabolite_level_se`: typed `ProvenanceGap`, terminal (not released).
- Median vs mean replicate summary: recorded conflict, not resolvable from released files.
- 46 JW ids not on a BW25113 locus: dropped. 29 cannot be placed without merging two
  strains; for the other 17, a rescue through the ECK crosswalk (or by name for 9) needs a
  derived-mapping field on the deletion leaf first.
- 25 pooled names: dropped; recoverable only from the raw intensity matrices, which carry
  the unpooled columns but not the normalization inputs.
- Culture format (1 ml, 96-deep-well, 300 rpm) is not stored: the bacterial experiment
  pair declares `environment: Environment`, so a `CultureEnvironment` would serialize as
  its base class.
- `target_metabolite_ids` is None: ions carry ambiguous putative annotations (EV1B), and
  linkage to iML1515 is not decided.
- Not registered in `torchcell/verification/runners.py`'s `METABOLITE_DATASETS` (outside this
  branch's files); `run_verification` here runs the same verifier.
