---
id: vd6g7ysa3ng7k5e53h5ye9s
title: Mutalik2020
desc: ''
updated: 1791382900592
created: 1791382900592
---

## 2026.10.07 - Rank 3 of the fifty: the phage-resistance RB-TnSeq row

Mutalik et al. 2020, "High-throughput mapping of the phage resistance landscape in E.
coli" (PLoS Biology, doi:10.1371/journal.pbio.3000877, citation key
`mutalikHighthroughputMappingPhage2020`). Source:
`torchcell/datasets/ecoli/mutalik2020.py`. Plan: [[plan.bacteria-ontology-genome]]
section 4; skeleton: [[torchcell.datasets.bacteria_common]]; schema layer:
[[torchcell.datamodels.bacterial-perturbation-ontology]].

**What landed: the raw mirror, the sourcing and the environment axis. What did not: the
dataset class.** The environment of every record in this row IS a phage challenge, and
the schema has no typed environment perturbation for a phage. Writing the records under
an existing leaf would mislabel the agent, and dropping the phage from the environment
would collapse 68 challenges onto one environment identity, so the loader stops here and
the needed addition is stated below. Everything that does not depend on it is finished,
measured and committed.

### Superset decision: load this row, de-duplicate nothing

The experiments are NOT in the Fitness Browser compendium, so there is no double copy to
avoid. Measured 2026-10-07:

| source | Keio (BW25113) experiments it carries | Mutalik's sets 16 / 19 / 28 / 30 |
|---|---|---|
| Fitness Browser, July 2026 archive (`db.StrainFitness.Keio`, figshare 10.6084/m9.figshare.32865896) | 168 columns from sets 1, 2, 5, 6, 52 | none |
| Fitness Browser, February 2024 archive (figshare 10.6084/m9.figshare.25236931) | 168 columns from the same five sets | none |
| Price 2018 `expsUsed` (`genomics.lbl.gov/supplemental/bigfit/html/Keio/`) | 225 rows, sets 1, 2, 6 | none |
| Price 2018 `fit_quality.tab`, the 162 successful `Keio` samples that subsume Wetmore 2015 | 162 names, sets 1, 2, 6 | none |

The de-duplication is by experiment NAME, not by set prefix: the intersection of these 99
experiment names with the 162 successful `Keio` samples of the Price 2018 compendium (the
set that subsumes rank 2, see the subsumption record on `feat/ecoli-wetmore2015`) is
empty. The 162 are read from the compendium's own `fit_quality.tab` where `u` is TRUE,
which equals the 162 columns of `fit_logratios_good.tab`; the live Fitness Browser
(`fit.genomics.lbl.gov`) returns 403 to scripts, so the compendium release page is the
route that works.

The row shares the LIBRARY with Wetmore 2015 and Price 2018 (one pool, KEIO_ML9, 152,018
barcoded insertions) and shares none of the EXPERIMENTS. That is the case the superset
rule exists for: the pool is a join key, not a duplicate.

### Deposited raw mirror

`$DATA_ROOT/torchcell-raw/mutalikHighthroughputMappingPhage2020/`, written by
`deposit_raw_mirror`, five files, each with a `RetrievalRecord` that re-runs through an
existing `torchcell.literature.retrieve` helper.

| path | bytes | sha256 (first 16) | retrieval |
|---|---|---|---|
| `data/S1_Table_RB-TnSeq_K12.xlsx` | 4,126,084 | `7db056cf673e0d32` | `pmc_cloud_object(key="PMC7553319.1/pbio.3000877.s009.xlsx")` |
| `data/S13_Table_MOI.xlsx` | 34,539 | `86221e234567b2e8` | `pmc_cloud_object(key="PMC7553319.1/pbio.3000877.s021.xlsx")` |
| `data/figshare/Keio_exps_used.tab` | 9,982 | `f9e78846ef9a85e5` | `direct_url("https://ndownloader.figshare.com/files/21662430")` |
| `data/figshare/README.txt` | 922 | `ebf050b56ed1d6f9` | `direct_url(".../files/21662427")` |
| `data/figshare/RBTnSeq.tar.gz` | 915,077,703 | `f760df837a6324a1` | `direct_url(".../files/20305338")` |

The PMC-cloud retrieval of the two SI tables reproduces the library mirror's `si/si9.xlsx`
and `si/si21.xlsx` byte for byte, so the raw mirror and the literature mirror agree on the
same bytes by two independent routes. The tarball's md5 matches figshare's own
(`8cd059e83ff7d6692255e37d028f2bde`); sha256 is the canonical anchor. Seventeen members
of it are pinned individually in `TARBALL_MEMBERS` and read through
`read_tarball_member`, which refuses a member whose bytes moved. One caveat for the
follow-up: a `.tar.gz` has no random access, so each member read streams the whole 915 MB
(about 13 s per member here); the loader should extract the members it needs into
`raw_dir` once.

### The environment axis, exactly as the release states it

`read_experiment_axis()` over `Keio_exps_used.tab` (the paper's own statement of which
experiments its analysis used): **99 experiments = 21 time-zero start samples + 10
no-phage controls + 68 phage challenges across 14 phages** (186, CEV1, CEV2, LZ4, N4, P1,
P2, T2, T3, T4, T5, T6, T7, lambda cI857), in three FEBA analysis sets
(`Keio_ML9_set16_set19`, `Keio_ML9_set28_set29`, `Keio_ML9_set30`).

The 68 matches the paper ("In total, we performed 68 RB-TnSeq assays across 14 phages at
varying multiplicity of infection (MOI) and 9 no-phage control assays"). The control count
does not: the released list carries **ten**, the extra one being `set16IT014`, run in
plain LB rather than LB plus SM buffer. Reported as the file holds it, not reconciled to
the text.

Released-versus-used, measured both directions:

- the S1 Table has 68 experiment columns (65 challenges + 3 controls), all of whose values
  are bit-identical to the figshare `fit_logratios.tab` (max absolute difference
  4.4e-16 over 250,960 cells) once keyed on the `setNNITNNN` experiment name;
- two S1 columns are NOT in the paper's used list (`set30IT059` P2 solid, `set30IT061`
  CEV1 solid): released but excluded from the analysis;
- 33 used experiments have no S1 column: the 21 start samples (which have no fitness
  column by construction), 7 controls and 5 challenges (`set16IT013`, `set16IT019`,
  `set16IT026`, `set16IT032`, `set16IT038`). All 5 do have fitness values in the figshare
  tables.

So the loader's selection rule is `Keio_exps_used.tab` and its values come from the
figshare tables, with the S1 Table as the published cross-check.

Per-experiment environment metadata lives in each analysis set's `exps` member: `Media`
(`LB_plus_SM_buffer`, `LB`, `LB_agar`), `Temperature` 37, `Liquid v. solid`, `Shaking`
(`orbital`, `200 rpm`), `Growth Method` (`48 well microplate; Tecan Infinite F200`,
`plate`, `flask`, `droplets`), `Condition_1` = `<phage>_phage` with `Units_1` = `MOI`, and
`Condition_2` = `Kan` at 50 ug/ml. Note a third culture format the paper's prose does not
describe: three `set28` experiments are `Group = "phage; droplets"`.

### MOI adjudication: S13 Table, not the S1 column labels

The Methods designate the authority: "Phage plaque-forming units/ ml and MOIs used in each
experiment are listed in the S13 Table." S13 releases the MOI as a formula over its own
inputs rather than as a number, so `read_moi_table` validates the formula shapes and
evaluates `MOI = pfu/ml * 0.35 mL * dilution / (0.04 OD * 0.35 mL * 8e8 cfu/mL)`,
resolving the dilution chain. Measured:

- S13 covers 67 of the 68 challenges. `set16IT019` has no S13 row; its MOI is taken from
  its own description (T2 at 0.01875 MOI, the same dose as `set16IT012`) and
  `Assay.moi_source` says so.
- The S1 Table column labels disagree with S13 for **11** experiments, by a factor of 10
  to 100 (`set16IT073` lambda cI857: S1 says 4.84375, S13 computes 484.375;
  `set30IT064` N4: S1 says 19.375, S13 computes 1937.5). The figshare experiment
  descriptions agree with S13 in all 11, so the error is in the S1 headers.
- S13 and the experiment descriptions agree for 64 of 67 and differ for 3: `set16IT051`
  (the description is rounded and a factor of 10 low) and the two P2 rows, whose
  `Concentration_1` carries the DILUTION (`0.1`, `0.01`) rather than the MOI (0.5625,
  0.05625) because their descriptions read "P2 dilution 10-1".

### Identifiers: BW25113 biology labeled in MG1655 b-numbers

The assayed strain is BW25113 ("We used a previously constructed E. coli K-12 BW25113
RB-TnSeq library [64]") and the released identifiers are MG1655 b-numbers. The mapping
reference in the figshare release (`g/Keio/genome.fna`) is a single scaffold of
**4,639,675 bp**, which is neither deposited assembly: ASM584v2 (U00096.3) is 4,641,652 bp
and ASM75055v1 (CP009273.1) is 4,631,469 bp. So FEBA mapped this pool against an older
MG1655 assembly, which is why its `sysName` is a b-number.

`audit_identifiers` scores three routes. Two are the naive ones, over the 4,610 rows of
the FEBA gene table (4,497 distinct b-numbers, 4,488 distinct symbol fallbacks):

| mapping | resolved | statuses | layers |
|---|---|---|---|
| b-numbers against MG1655 ASM584v2 | **0.998** | 4,346 current, 141 non-gene feature, 1 renamed, 8 retired, 1 ambiguous | 4,484 locus tag, 5 gene synonym, 8 not found |
| FEBA gene symbols against BW25113 ASM75055v1 | **0.961** | 4,130 renamed, 184 non-gene feature, 168 retired, 6 ambiguous | 3,776 gene symbol, 544 gene synonym, 168 not found |

The BW25113 symbol shortfall is interpretable: the 168 unresolved names are the 60-odd
`ins*` IS-element genes, the 114 genes with no FEBA symbol (which fall back to a b-number
BW25113 does not carry), and `araA`, `araB`, `rhaA`, `rhaB` -- genes BW25113 has deleted.
Six symbols are ambiguous in BW25113 (`pin`, `radC`, `rffT`, `spr`, `thiJ`, `ygaD`) and
ten collide, and all of those are kept as given by the retain-all policy.

**The third route is the one the records take: the ECK synonym join (`eck_route`).** Both
GenBank annotations carry `ECK\d{4}` synonyms, and `eck_crosswalk` returns 4,423 ECK ids
carried by exactly one locus in each strain (11 of them with disagreeing numbers).
Measured over the 3,716 genes that actually carry a fitness value: **3,697 map, 0.9949**,
19 do not (`b1172` ymgG, `b1370` insH-5, `b2863` ygeQ, `b3683` glvC, `b4104` phnE,
`b4294` insA-7, `b4339` yjiP, `b4416` rybA, `b4486` yjiV, `b4499` yehH, `b4524` ycjV,
`b4576` insB-7, `b4587` insN, `b4600` ydfJ, `b4659` yabP, `b4693` aaaE, `b4694` yagP,
`b4695` ykgT, `b4700` sokE), and exactly one of the mapped genes has disagreeing
numerics: `b0018` to `BW25113_4412`. That single case is the proof that no string surgery
relates the namespaces, and the reason the join rather than a numeric rewrite is the
route. The ECK route is also the one the Wetmore 2015 row settled on, measured there at
99.45% on the compendium's gene set.

A mapped record is written against BW25113 with a `BW25113_` locus tag, and the mapping
is DERIVED: a b-number the paper released became a BW25113 tag by way of a shared ECK
accession in two annotations the paper did not use. Plan section 4 requires that to be
recorded on the record, and `TransposonInsertionPerturbation` has no slot for it (see the
blockers below).

The released fitness table is itself BW25113-consistent, which is the measurement that
settles what the data IS: `araA` (b0062), `araB` (b0063), `rhaA` (b3903) and `rhaB`
(b3904) carry **no row**, while `lacZ` (b0344), `hsdR` (b4350) and `rph` (b3643), whose
BW25113 lesions leave the gene in place, do. The numbers are BW25113's; only the labels
are MG1655's.

**The pin is BW25113** (`assembly_reference("BW25113")`, assembly set
`ecoli_K12_BW25113_ASM75055v1`, GenBank `GCA_000750555.1`), because that is the strain
the assay was run in and the strain whose gene content the released table reflects. The
MG1655 alternative keeps the identifiers verbatim at the price of misstating the strain,
which is the one thing plan D5 says the tier must never do.

### Phenotype class: EnvironmentResponsePhenotype, not FitnessPhenotype

The readout is "the normalized log2 change in the abundance of mutants in that gene", a
SIGNED log2 ratio. Of the 250,960 released (gene, experiment) cells in the S1 Table,
**106,536 (42.45%) are negative** (minimum -8.06, maximum 18.22, median 0.106, 815 cells
at or above the paper's hit threshold of 5). `FitnessPhenotype.validate_fitness` clamps
every non-positive value to 0.0, so that class would zero out 42% of the matrix. The
right pair is the one plan section 3c already added:
`BacterialEnvironmentResponseExperiment` with
`EnvironmentResponsePhenotype(measurement_type=log2_ratio,
assay_type=pooled_competitive_growth_barcode)`. The plan's section 3c table maps
"transposon and RB-TnSeq gene fitness" onto `FitnessPhenotype`; for a log2-ratio readout
that mapping is wrong, and this row is the counterexample.

### Uncertainty and n_samples, both sourced

- **Uncertainty TYPE: `standard_error`, used as-is.** The release carries a per-record
  estimated standard error (`fit_standard_error_obs.tab`, one column per experiment,
  same gene rows as the fitness table) beside a per-record t-like statistic
  (`fit_t.tab`). The paper's own filter writes the relationship out: "we required that
  fit $\ge 5 ; \mathsf { t } \ge 5 ;$ ; standard error $\dot { \bf \varphi } = \bf { f i t
  } / t \le 2$". It is an SE of the gene fitness estimate, so it is stored as-is and
  never divided by sqrt(n) again. The S1 Table releases `fit`, `t` and `se` only for the
  354 filtered hits (`Keio_hits_fit5`); the full per-record SE exists only in the figshare
  tarball, which is one reason the tarball is deposited.
- **n_samples: 16, the deferral.** One sample is an independent insertion strain
  ("The fitness value of each gene is the weighted average of the fitness of its
  strains."). The per-gene count is not a released column, and the method paper gives it
  for this exact library: Wetmore 2015 Table 1, column *Escherichia coli* BW25113
  (KEIO_ML9), "Median no. of strains per genec</td><td>16</td>", footnote c "Includes only
  genes for which we report fitness estimates and only strains that were used to make
  those estimates." Mirrored at `wetmoreRapidQuantificationMutant2015/paper.md`, sha256
  `ca3e7ef2...`. This is a library median, NOT a per-record count; the per-record count is
  derivable from the deposited pool and `strain_fit.tab` and is left to the loader. For
  comparison the paper states the BL21 figure directly ("Twelve independent strains were
  used to compute fitness for the typical protein-coding gene.").

### Media: an open gap, deliberately not filled

The assay medium is LB, and the paper defers the recipe to its reference 96 (Bertani),
which is not mirrored: "we recovered a frozen aliquot of the E. coli K-12 RB-TnSeq library
in lysogeny broth (LB [96]) to mid-log phase". [[torchcell.datamodels.media]] already
records this row as unsourced for that reason. Borrowing `LB` (Miller, 10 g/L NaCl from
Menasalvas and Schmidt) or `LB_LENNOX` (5 g/L NaCl, which is what Wetmore 2015 and Price
2018's own media tables state) would assert a recipe Mutalik never gave, so the medium
stays a gap until either Bertani is mirrored or the owner accepts the same-lab `LB_LENNOX`
reading. The rest of the environment is fully sourced: 2X LB diluted 1:1 with phage in SM
buffer, 37 C, 48-well microplate at 700 uL per well, 8 h with orbital shaking and OD600
every 15 min for the planktonic format; LB agar plus kanamycin, overnight at 37 C, for the
solid format; kanamycin at 50 ug/ml throughout; SM buffer supplemented with 10 mM calcium
chloride and magnesium sulfate.

### What blocks the loader

**1. There is no typed environment perturbation for a phage.**
`EnvironmentPerturbationType` is `SmallMoleculePerturbation |
EnvironmentPhysicalPerturbation | BiologicPerturbation`. A small molecule is keyed by
InChIKey; `PhysicalFactor` is pH, osmolarity, carbon source, nitrogen source, ionic
strength, nutrient dropout, radiation; `BiologicAgentClass` is peptide, protein, antibody,
toxin. A virion is none of them, and nothing in `torchcell/` mentions a phage today. The
dose is also unexpressible: a multiplicity of infection is a particle-to-cell ratio, and
`ConcentrationUnit` and `DoseBasis` have no member for it. The addition this row needs is
written in the PR body (a new leaf plus its `identity.py` projection plus a graph node
class), and it is additive under the admission gate by the same argument the five
perturbation leaves used.

**2. The perturbation leaf cannot say its identifier was remapped.** The records carry
BW25113 locus tags derived from released b-numbers through the ECK join, and
`TransposonInsertionPerturbation` has `gene_namespace`, `barcode`,
`insertion_position`, `insertion_strand`, `transposon` and `library_pool` but no slot for
the source name or the mapping route. Plan section 4 requires the derived mapping to be
recorded on the record; the PR body names the two optional fields that do it. Nothing
served imports that leaf yet, so adding them is additive today (reasoning, not a fresh
measurement: `schema_deps` fingerprints the classes a loader imports, and no landed
loader imports this leaf).

**3. `SampleUnit` has no insertion-strain member.** The replicate unit of an RB-TnSeq gene
fitness value is an independent insertion strain in a pooled library, which is neither a
colony, a screen, a biological replicate, a technical replicate nor "pooled". Because the
released uncertainty is already an SE, `n_samples` and `sample_unit` are optional on
`EnvironmentResponsePhenotype`, so this one does not block a build; it limits what the
record can say about its own replication.

### The build the follow-up will run

Selection rule `Keio_exps_used.tab`, values from the figshare tables, one record per
(gene, experiment): **287,815 records** over the paper's own analysis set (3,686 genes x 58
experiments in `set16_set19`, 3,714 x 9 in `set28_set29`, 3,691 x 11 in `set30`), of which
250,894 are phage challenges and 36,921 are no-phage controls, over 3,716 distinct genes.
No record is dropped for a naming reason; the retain-all policy keeps every unresolved
identifier as given, and the identifier audit above is the report.

Out of scope here, each named rather than silently omitted: the Dub-seq gain-of-function
arm (S5 Table and figshare 10.6084/m9.figshare.11838879.v2 -- a multicopy plasmid-borne
genomic fragment is not one of the five bacterial perturbation leaves, and its gene scores
are a non-negative-least-squares attribution over fragments), the MG1655 CRISPRi arm (S4
Table and figshare 10.6084/m9.figshare.11859216.v3 -- a different library and strain, and
the row for it is Smith 2016's analog), the BL21 arm (S8 and S9 Tables -- BL21-DE3 is not a
deposited assembly set, so a BL21 record has nothing to pin), and SRA BioProject
PRJNA645443 (raw reads; the mirror keeps the processed tables the loader consumes).

### Reproducing every number here

```bash
cd ~/Documents/projects/torchcell && set -a && source .env && set +a
PYTHONPATH=$PWD python -m pytest tests/torchcell/datasets/ecoli/test_mutalik2020.py -q --data --slow
```

54 passed on 2026-10-07 (37 hermetic, 12 mirror-gated, 5 tarball-gated, 2 of them
also needing the two deposited K-12 assembly sets). The hermetic tier builds a synthetic
mirror in `tmp_path` (a five-file stand-in deposit, a real small `.tar.gz`, an S13
workbook written with the sheet's own formula strings), re-pins `RAW_ARTIFACTS` and
`TARBALL_MEMBERS` to it by monkeypatch, and patches the ECK crosswalk and the resolver,
so every reader and every refusal runs on the CI runner with no `$DATA_ROOT`; it alone
covers 100% of the module's changed lines under `diff-cover`. The identifier
histograms and the ECK route come from `audit_identifiers(mg1655, bw25113,
measured=measured_gene_ids())`, the axis counts from `read_experiment_axis()`, the doses
from `read_moi_table()`.

## 2026.10.07 - The loader lands: 286,344 records on PhagePerturbation

The blocker recorded above is gone. `PhagePerturbation` landed on main (schema.py around
line 3128), so the environment of a phage challenge is now typed and the dataset class is
registered: `PhageRbTnseqMutalik2020Dataset` in
`torchcell/datasets/ecoli/mutalik2020.py`, root `data/torchcell/phage_rbtnseq_mutalik2020`.
Everything below is measured on the deposited release, not projected.

### The build

```bash
cd ~/Documents/projects/torchcell.worktrees/feat/ecoli-mutalik2020-loader
set -a && source .env && set +a
PYTHONPATH=$PWD python -m torchcell.database.build_dataset_lmdb \
  --dataset PhageRbTnseqMutalik2020Dataset
```

| quantity | value |
|---|---|
| records | **286,344** |
| gene-set size (distinct BW25113 locus tags) | **3,697** |
| references (one per record-bearing experiment) | **78** |
| wall time | **159 s** |
| `build_manifest` status | `fresh` |

No previous store existed, so nothing was retired for the first build. A partial first
attempt (a `ProvenanceGap` whose `field` named no model field, corrected before the real
run) had its `processed/` and `preprocess/` moved to
`/scratch/projects/torchcell-deprecated/2026-10-07_164926__{processed,preprocess}/`, and
both directories were confirmed gone before the rebuild; `raw/` was kept, so the 915 MB
tarball was streamed once.

Per analysis set, mapped genes times record-bearing experiments:

| analysis set | genes mapped | experiments | records |
|---|---|---|---|
| `Keio_ML9_set16_set19` | 3,667 | 58 | 212,686 |
| `Keio_ML9_set28_set29` | 3,695 | 9 | 33,255 |
| `Keio_ML9_set30` | 3,673 | 11 | 40,403 |
| total | 3,697 (union) | 78 | **286,344** |

By experiment kind: 249,612 phage-challenge records (68 experiments) and 36,732 no-phage
control records (10 experiments). By assay format: 245,941 planktonic and 40,403
solid-agar; every solid assay is in `set30` and `set30` holds nothing else.

This is 1,471 records fewer than the 287,815 the earlier section projected, because that
projection kept every released gene. See the retention ledger.

### Retention ledger, with the arithmetic

`preprocess/dropped_records.json`. Source records are the cells of every record-bearing
used experiment; the 21 start samples are not source records at all, because the release
gives a time-zero sample no fitness column.

| | records |
|---|---|
| source (78 experiments x their set's gene rows) | 287,815 |
| kept | 286,344 |
| dropped | **1,471** |

| rule | scope | records | items |
|---|---|---|---|
| `no_one_to_one_eck_pair` | gene | 1,471 | 19 b-numbers: `b1172 b1370 b2863 b3683 b4104 b4294 b4339 b4416 b4486 b4499 b4524 b4576 b4587 b4600 b4659 b4693 b4694 b4695 b4700` |
| `medium_has_no_media_library_base` | experiment | 0 | none |

`1471 = 19 genes x (58 + 9 + 11 experiments)`, which closes exactly: the build raises if
the rules do not total the difference.

**The ECK drop is the Price 2018 rule, applied to the same release.** The records pin
BW25113 and a gene with no one-to-one ECK pair has no storable identifier in the namespace
that pin declares, so it is dropped and named rather than stored under an MG1655 b-number.
That is a deliberate change from the "retain-all" wording of the earlier section: the
retain-all policy is about not silently renaming, and here the honest alternative to a
rename is a counted drop. 3,697 of 3,716 place (0.9949), and
`MIN_ECK_ROUTE_FRACTION = 0.99` stops the build rather than dropping more. One mapped gene
has disagreeing numerics, `b0018 -> BW25113_4412` through `ECK0018`, which is why the join
rather than a numeric rewrite is the route. Every record says the mapping was derived:
`identifier_mapping = DerivedIdentifierMapping(source_identifier="b0018",
route="eck_crosswalk")` on the `TransposonInsertionPerturbation`.

**No assay was dropped for its medium.** The rule exists and is exercised by a test; the
release names exactly three media labels and all three have an object.

### The experiment ledger: selection is not a record drop

`preprocess/assay_ledger.json`.

| | count |
|---|---|
| used experiments (`Keio_exps_used.tab`) | 99 |
| start samples (no fitness column by construction) | 21 |
| record-bearing | 78 |
| dropped for medium | 0 |
| released fitness columns across the three sets | 190 |

So 190 columns are released and 78 carry records: the selection rule is the paper's own
used list, not the release's column inventory. By medium: `LB_plus_SM_buffer` 66,
`LB_agar` 11, `LB` 1. By format: liquid 67, solid 11. All 14 phages appear, 68 challenges
in total.

**The release's own quality flag is FALSE for 68 of the 78 kept experiments**, and that is
not a finding about this build. The paper states it directly: under the strong positive
selection of a phage assay "our standard quality metrics reported earlier [64] were not
suitable". That is why the selection rule is `Keio_exps_used.tab` and not `u`; the flag is
reported in the ledger so the choice is visible rather than implicit.

### The MOI: bound per experiment, gapped when absent

Each challenge's environment carries exactly one `PhagePerturbation`, and the dose is
`multiplicity_of_infection`, a dimensionless ratio in its own field rather than a
`Concentration`. Measured over the 68 challenges: **67 doses come from the S13 Table** (the
authority the Methods designate) and **1 from the experiment's own released description**
(`set16IT019`, T2 at 0.01875 MOI, which S13 has no row for), as `Assay.moi_source` records.

Where a source states no MOI the perturbation carries a typed
`ProvenanceGap(field="multiplicity_of_infection",
reason=not_reported_by_primary)` whose `looked_in` names the S13 sheet and the experiment,
and whose note says the dose is never borrowed from a neighbouring assay of the same
phage. **No released challenge needs that gap** (68 of 68 have a dose), so the path is
covered by a test rather than by the build; it is the refusal that keeps a future assay
from being given its neighbour's dose.

The other typed slots, each sourced or gapped:

- `genome_type = "dsDNA"` for all 14, from "We sourced 14 diverse E. coli phages with
  dsDNA genomes, belonging to Myoviridae, Podoviridae, and Siphoviridae families (within
  the order Caudovirales)".
- `genome_accession` per phage from the **S12 Table** (`si/si20.xlsx` in the paper mirror,
  sha256 `f3e9a797...`), 11 of 14 determined: T2 `MH751506.1`, T3 `NC_003298.1`, T4
  `AF158101.6`, T5 `NC_005859.1`, T6 `MH550421.1`, T7 `NC_001604.1`, N4 `EF056009.1`, P1
  `NC_005856.1`, P2 `AF063097.1`, 186 `NC_001317.1`, lambda cI857 `NC_001416.1`. CEV1,
  CEV2 and LZ4 are `None` because the sheet itself says "Not determined". Two names are
  spelled differently in the two tables and the mapping is recorded in the constant's
  note: S13's `P1` is S12's `P1vir` (the Methods: "a strictly virulent strain of P1 phage
  (P1vir)") and S13's `lambda cI857` is S12's `lambda c1857`.
- `host_of_propagation`: `E. coli BW25113` for 12 phages, `E. coli C` for P2 and
  `E. coli W3350` for N4, from one Methods sentence.
- `titer_pfu_per_ml` is computed for the planktonic format only, as
  `pfu/ml * dilution / 2`: S13 releases the stock titer and the dilution, and the Methods
  state 350 uL of diluted phage into 350 uL of 2X library for 700 uL per well. For the
  solid format the final volume is never stated (the planktonic paragraph gives the two
  volumes; the solid paragraph says only "the mixture"), so it stays `None` there. It is
  also `None` for `set16IT019`, which has no S13 row and so no released stock titer.
- `family` is a typed gap on every phage: the three families are named for the panel
  COLLECTIVELY and neither the paper nor the S12 Table assigns one to a phage.
- `ncbi_taxid` is `None`: no source names a taxon id.

**The 10 no-phage controls carry no phage perturbation at all.** The paper: "We also set
up control 'no-phage' competitive mutant fitness assays wherein we replaced phages with
simply the phage dilution buffer." A zero-MOI phage would assert a challenge that did not
happen; the buffer is in the medium instead, which is what makes a control's environment
the honest reference for the challenges run beside it.

### Where the assay format landed

`BacterialEnvironmentResponseExperiment.environment` is `Environment`, not
`CultureEnvironment`, so a vessel and a shaking speed cannot be stored without editing
`schema.py`. The planktonic / solid-agar distinction therefore lands in three places a
plain `Environment` can hold, and the 68 challenges keep 68 environment identities:

1. **The `Media` object and its `state`.** Three dataset-local objects, every one with
   `base_medium="LB"` (a `MEDIA_LIBRARY` key) and no invented amounts:
   `MUTALIK2020_LB_SM_BUFFER` (liquid; LB plus the Teknova SM buffer as one
   `composition_deferred` line, plus calcium chloride and magnesium sulfate with no
   concentration because the stated 10 mM is the BUFFER's and the assay dilutes it),
   `MUTALIK2020_LB_AGAR_KAN` (solid; plus agar with no percentage and kanamycin at
   50 ug/mL as a `selection_agent`), and `MUTALIK2020_LB` (liquid, the one plain-LB
   control). The release's `Media` label selects the object through `MEDIA_BY_LABEL`.
2. **`duration_hours`**: 8.0 for the planktonic format ("for 8 hr"), and a typed
   `ProvenanceGap` for the solid format, whose incubation the Methods state only as
   "overnight".
3. **The phenotype's `units` and `screen_id`.** `units` names the format in prose;
   `screen_id` is `Keio:<expName>`, the Price 2018 idiom on the same FEBA namespace, which
   is what keeps the same phage at the same MOI in two formats, and two replicate assays
   of one dose, L1-distinct.

`process()` checks the hardcoded media against the released row rather than trusting
them: a `Media` whose `state` disagrees with `Liquid v. solid`, a solid assay without
`Kan` at 50 ug/ml, or a non-agar assay WITH a second condition each raise.

### Media: why three local objects and not a MEDIA_LIBRARY edit

`torchcell/datamodels/media.py` is a fingerprinted value surface, so an edit there marks
served stores stale. The earlier section called the medium an open gap because Mutalik
defers the LB recipe to its reference 96 (Bertani), which is not mirrored. The loader
resolves that the way the Rousset 2018 branch resolves the same situation: a dataset-local
`Media` with `base_medium="LB"` and components that carry NO amounts. Nothing false is
asserted, the join works at the `LB` base, and `media.py` is untouched. The needed
addition, if the owner wants amounts, is a Bertani-sourced `LB` entry or an explicit
acceptance of the same-lab `LB_LENNOX` reading (5 g/L NaCl, which is what Wetmore 2015 and
Price 2018's own media tables state); neither is assumed here.

### n_samples and sample_unit are typed gaps, not the library median

A correction to the earlier section's reading. The gene fitness value is "the weighted
average of the fitness of its strains", so one sample is an independent insertion strain,
and Wetmore 2015 Table 1 gives the library MEDIAN as 16 for KEIO_ML9. A median over genes
is not a per-record count, so it is NOT stored: `n_samples` and `sample_unit` are
`ProvenanceGap`s with `looked_in` on the released per-gene tables (which carry no
strain-count column) and `resolve_with` on the release's own `strain_fit.tab` plus
`g/Keio/pool`. This is the same decision the Price 2018 loader made on the same library.
The stored uncertainty is already an SE of the estimate, so no `n` is needed to derive it:
`environment_response_uncertainty_type = standard_error` and `environment_response_se` is
the same number.

The released t-like statistic (`fit_t.tab`) is neither the measurement nor an uncertainty
and `EnvironmentResponsePhenotype` has no slot for a test statistic, so that member is not
among the twelve the loader extracts.

### raw/: twelve members extracted once, not re-streamed per read

The earlier section's caveat is fixed. `download()` links the S13 Table and
`Keio_exps_used.tab` into `raw/` and extracts the twelve per-analysis-set members
(`exps`, `fit_logratios.tab`, `fit_standard_error_obs.tab`, `fit_quality.tab`) in ONE
streaming pass over the 915 MB tarball, verifying each member's sha256 before it is
written. `verify_raw_files` re-checks all fourteen at the start of `process()`, so an edit
to `raw/` after download still cannot reach a record.

### Verification

`verify_build()` carries the environment-response L0-L4 gate itself, because
`torchcell.verification.runners.run_environment_response` is yeast-only today (the same
reason tong2020, borchert2024 and menasalvas2025 carry their own). It uses the STREAMING
verifier: 286,344 records with a component-level medium on each environment are not
materialized. The gene universe and resolver are BW25113's, every GenBank locus included.
Five SUPPLEMENTARY rows are appended: the analysis-set, experiment-kind and assay-format
censuses (`experiment_census`), `stored_tags_are_loci_of_the_pinned_assembly` and
`gene_set_size`. The report is written to `preprocess/verification_report.json`.

One limitation found while wiring it, worth recording rather than working around: the
shared `_condition_signature` of `torchcell/verification/environment_response.py` reads a
perturbation's `compound.name` / `agent.name` and its `concentration` / `magnitude`, none
of which a `PhagePerturbation` has, so a phage contributes only
`perturbation_type="phage"` to the condition key. L1 uniqueness still holds here because
`screen_id` is per-experiment and joins the signature, but a dataset that relied on the
phage NAME or the MOI to separate two conditions would not be separated by that rule
today. Not fixed here: `runners.py` and its helpers are being changed on
`feat/titer-verification-runners`.

### The BioCypher adapter, and the one-class rule

`torchcell/adapters/mutalik2020_adapter.py` with
`torchcell/adapters/conf/phage_rbtnseq_mutalik2020_adapter.yaml`, plus the
`dataset_adapter_map` entry and the `torchcell/adapters/__init__.py` re-export.
`kg_bacteria.yaml` names the class, so the bacterial rehearsal generation covers it.

**The conf enables `phage perturbation` and NOT `environment perturbation`.** Those two
node classes are mutually exclusive by construction: the served
`_environment_perturbation_node` does not filter phages out, so enabling both writes every
phage twice, under two labels on one content id. Both
`environment perturbation to environment` EDGE methods stay enabled, because a phage node's
id is the same composition projection the other environment-side nodes use.

**Nothing is lost by that choice, measured.** The only non-phage item on any Mutalik
environment would be the kanamycin of the solid-agar assays, and that is a plate
ingredient rather than a dosed perturbation: the release's own metadata carries it as the
`LB_agar` recipe's `Condition_2`, and the Methods call the plates "LB agar supplemented
with kanamycin plates". It is a `MediaComponent` with `role=selection_agent` for that
reason and not to satisfy the adapter. Across all 194 released experiments `Condition_2`
is only ever empty or `Kan`, and among the 78 kept experiments only the 11 `LB_agar` rows
carry it, so the choice costs 0 records and 0 perturbation nodes.

The shared adapter harness learned the phage shape rather than being bypassed:
`Shape` gained a `phage` flag that emits the phage node pair in the
environment-perturbation slot and refuses a shape with both; `EDGE_ENDPOINTS` and
`NODE_LINK` now let the two environment-perturbation edges address a phage node; and the
dev-store check asserts `("phage perturbation" in labels) is shape.phage` while exempting
`environment perturbation (chunked)` from the "left-off families emit nothing" converse
for a phage conf only, with the reason written in place.

### Tests

```bash
cd ~/Documents/projects/torchcell.worktrees/feat/ecoli-mutalik2020-loader
set -a && source .env && set +a
PYTHONPATH=$PWD python -m pytest tests/torchcell/datasets/ecoli/test_mutalik2020.py -q --data --slow
PYTHONPATH=$PWD python -m pytest tests/torchcell/adapters/test_mutalik2020_adapter.py -q --data
```

94 passed on 2026-10-07 for the dataset file (40 new, of which the loader tier is
hermetic) and 6 for the adapter. The new hermetic tier builds a SECOND synthetic mirror in
`tmp_path`, independent of the sourcing tier's: a real small `.tar.gz` carrying twelve
members (three `exps` files, three fitness tables, three SE tables, three quality tables),
an S13 workbook in the sheet's own formula strings, and the synthetic MG1655 + BW25113
assemblies from `tests/torchcell/sequence/genome/_bacterial_fixtures.py`, so the whole
build -- download, extraction, ECK route, records, LMDB, reports -- runs on a CI runner
with no `$DATA_ROOT` and no network. Diff coverage of the changed lines under
`torchcell/`: **92%** (`diff-cover --compare-branch=origin/main`), against the 80% gate.
