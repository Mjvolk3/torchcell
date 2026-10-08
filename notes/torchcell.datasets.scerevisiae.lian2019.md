---
id: iqhtiqrvf77x9ojrqzbjk2g
title: Lian2019
desc: ''
updated: 1783919988175
created: 1783919988175
---

## 2026.07.13 - Lian 2019 MAGIC CRISPRa/i/d furfural screen (per-guide, SRA-reprocessed)

`CrisprMagicLian2019Dataset` -- the FIRST dataset on the AXIS-4 CRISPR expression-modulation
ontology (`[[plan.torchcell-crispr-expression-perturbation.2026.07.12]]`). 266,415
(guide x round) `EnvironmentResponseExperiment` records.

### Why the phenotype had to be reprocessed from raw NGS

Lian released only the DESIGNED guide library (Supp Data 1-3, design scores) + the guide
reference (Supp Data 4). The furfural per-guide ENRICHMENT (Fig 2a/c/e) is **not in any
supplement** (excluded from the Source Data file), and the in-house Carl-Schultz re-analysis
we had was collapsed to per-GENE (modality lost). Modality lives at guide resolution
(orthogonal Cas), so a true CRISPRa/i/d dataset required per-GUIDE counts -> reprocess SRA.
Full trail: memory `[[lian2019-magic-data-availability]]`.

### Reprocessing pipeline (scripts in the library mirror `.../lianMulti.../data/`)

1. `download.sh` -- 21 SRA runs (PRJNA504483): 3 rounds x {before/after} x triplicate +
   ecLibA/I/D plasmid baselines (2 runs pulled direct from ENA after prefetch 404).
2. `count_guides.py` -- barcode = read[27:70] (43bp activation) | read[27:71] (44bp
   interference/deletion), forward, offset-27 (empirically scanned); exact-match to the
   100,493-guide reference (sha256 4e3f225a...). Mapping 74-78% furfural / 58-74% plasmid.
   Barcode collisions (6,666 guides) are ALL same-gene+same-modality -> safe to collapse.
3. `enrichment.py` -- CPM(+1)/library; per round per replicate log2(after/before); mean+-SD
   over 3 triplicates. **Validated vs paper hits: PDR1i r3 rank 1, SLX5i r1 rank 1, SAP30d
   r1 rank 2.** Round 2 is the paper's noisy/synergistic round (RCF1a rank137, NAT1a mid).
4. `guide_enrichment_final.tsv` (sha256 f9af849f...) = the loader's pinned input.

### Record model

- **genotype (per-guide)**: library member as `CrisprActivation/Interference/Deletion`
  (target gene + spacer + effector dLbCas12a-VP/dSpCas9-RD1152/SaCas9) PLUS the round's
  integrated background (R2 +SIZ1i=YDR409W; R3 +SIZ1i+NAT1a=YDL040C, guide unspecified).
  So R1 = 1-pert, R2 = 2-pert, R3 = 3-pert mixed-modality combos.
- **environment**: furfural 5/10/15 mM (by round), SED/G418 liquid, 30 C, aerobic.
- **phenotype**: `EnvironmentResponsePhenotype` log2_ratio, response = mean log2FC,
  uncertainty = SD (sample_sd, n=3 -> SE=SD/sqrt(3)); reference = no-enrichment (log2FC 0).

### Drops (documented, counted)

300 random controls + 16 source-corrupted-gene guides (Excel date/serial artifact identical
in reference AND library -> unrecoverable) + 2,633 unresolved-gene guides (165 ncRNA/rDNA
genes absent from the ORF genome); 26,169 (guide,round) undetected; 48 guides skipped in the
round where they target their own background.

### Verification note (shared code touched)

L1 strain identity (`_genotype_signature`) was extended to be GUIDE-AWARE: sibling guides of
one (gene, mode) are distinct strains (like a TS-allele series), keyed by
`crispr.guide_sequence`. `background_genes` is EMPTY for this dataset -- the per-round
background is genuine genotype content, not a constant to subtract.

## 2026.09.12 - Serve-50 rebuild: SED-URA/G418, and the CRISPRd guide/donor split

Rebuilt for admission. The 2026-07-12 build failed L0 structural on all 266,415 records
(`Media` gained a required `is_synthetic` after they were serialized) and carried two
factual errors: the wrong medium on every record, and a donor fragment stored as a guide
spacer on 62,793 of them. Nothing was served, so nothing is updated in place.

### Error 1: the medium was SED/G418, the screen ran in SED-URA/G418

`lian2019.py` stored `Media(name="SED/G418", state="liquid")` for all 266,415 records.
Methods, verbatim:

> "iMAGIC screening of furfural tolerance. The iMAGIC libraries in triplicates were
> inoculated into 50 mL SED-URA/G418 medium with or without furfural in a 250 mL baffled
> flask."

and the same paragraph shows where SED/G418 belongs:

> "For the individually constructed strains, a single colony was pre-cultured in 2 mL
> SED-URA/G418 (plasmid-bearing strains) or SED/G418 (integrated strains) medium"

The enrichment comes from the plasmid-borne pooled library, so the medium is SED-URA/G418.
This is not a naming nit: SED/G418 lacks the uracil dropout that selects the guide plasmid,
so the old record described a different selection regime from the one that produced the
data. The build now uses the shared `media.SED_URA_G418` object, whose components (YNB w/o
AA 0.17%, monosodium L-glutamate 0.1%, CSM-URA 0.077%, D-glucose 2%, G418 200 ug/mL,
`dropouts=[uracil]`) are all sourced from one Methods sentence. The G418 line reuses the
exact `Compound` spelling the SGA media use, so the two G418-selected families share one
selection-agent identity.

### Error 2: a 44 nt donor fragment stored as a guide spacer (62,793 records)

The old build put the reprocessing's amplicon BARCODE -- the first 44 nt of the 121 nt
design cassette -- into `crispr.guide_sequence`, where the schema documents a "~20 nt"
spacer, and left `CrisprDeletionPerturbation.donor_sequence` None for all of them. The paper
gives the layout but not the offset:

> "To enable genome-scale gene disruption, the homologous recombination donor was integrated
> to the 5'-end of the targeting sequences."

and:

> "Homology-directed repair resulted in the deletion of 28 bp nucleotides in the coding
> sequences, including both the targeting sequences and the protospacer adjacent motif
> sequences"

So the boundary was MEASURED against S288C rather than assumed:

1. Split each 121 nt cassette into two 50 nt halves plus a tail and map the halves. For 193
   of 200 sampled designs whose halves both map uniquely, the gap between arm 1's end and
   arm 2's start is **exactly 28 bp** in 193/193, and the design's last 21 nt starts exactly
   at that gap (offset 0 in 191/193; the two outliers are multi-hit spacers).
2. For **all 24,706 non-control designs**, the last 21 nt is present in the genome with a
   canonical SaCas9 `NNGRRT` PAM immediately 3' of it. 0 in-genome-but-no-PAM, 0
   not-in-genome.

The cassette is therefore `donor (100 nt, two 50 nt arms flanking the 28 bp deletion) +
spacer (21 nt)`. The build stores the last 21 nt as `crispr.guide_sequence` and the leading
100 nt as `donor_sequence`. CRISPRa (23 nt) and CRISPRi (20 nt) already released their true
spacer and are unchanged.

The split is read from the PUBLISHED Supplementary Data 3
(`41467_2019_13621_MOESM5_ESM.xlsx`, sha256 `737074a7...`), not from the lab design copy.
Measured: its `Sequence` column is element-for-element identical to the archived lab file
(and the same holds for Supplementary Data 1 and 2 against the CRISPRa and CRISPRi lab
files), and its first 44 nt reproduce the enrichment table's `d` barcode for all 24,806
rows. That last identity is the positional join, and the loader ASSERTS it at build time
rather than trusting row order.

### The new drop rule: 318 records on ambiguous strain identity

Switching from the 44 nt barcode to the 21 nt spacer changes what makes two designs
distinguishable. Measured on the rebuilt keys, 150 groups of CRISPRd designs resolve to the
same gene AND the same 21 nt spacer, covering 318 records. These are multicopy loci (tRNA
genes, paralogs) where the spacer maps to more than one site; the designs differ only in
their donor arms.

The stored record grain is `(gene, mode, spacer, pool)` and the verifier's
`_genotype_signature` reads `crispr.guide_sequence` and `crispr.library_pool` but NOT
`donor_sequence`, so serving both would be two records claiming to be one strain and L1
`pair_uniqueness` would fail. They are dropped and counted. Adding `donor_sequence` to
`_identity` in `torchcell/verification/environment_response.py` restores all 318 with no
other effect -- no other dataset stores a donor -- and that is the requested shared-layer
change.

### Other fixes

- **`assay_type = pooled_competitive_growth_barcode`** (was None on all records).
- **Exposure duration** is a typed `ProvenanceGap` on BOTH `duration_hours` and
  `duration_generations`: the cultures were harvested at mid-log ("1 OD of the mid-log phase
  growing cells ... were collected") and neither a wall time nor a doubling count is
  reported. The old build left both silently None.
- **Common names** now come from the genome's standard name for the resolved ORF, through
  the shared `resolve_gene_name` (standard-before-alias). That resolver is stricter than the
  old ad-hoc alias lookup, which is why 2,670 guides are now unresolved against 2,633
  before: 37 more names are AMBIGUOUS or non-gene features and are dropped rather than
  first-matched.

### bAID stays the reference strain (a decision, not a default)

bAID is BY4742 plus an integrated pAID6 carrying the three Cas effector cassettes:

> "The CRISPR-AID strain (bAID) was constructed by integrating PmeI-digested pAID6 into the
> genome of BY4742 and selection for G418 resistance."

The integration LOCUS is not stated anywhere in the release, so a gene-keyed
`GeneAdditionPerturbation` for those cassettes cannot be sourced, and the three effectors
are already carried per record on `CrisprConstruct.effector`. The cost is recorded rather
than hidden: the strain string `bAID` does not join the BY4742 datasets. (Contrast Mormino
2022, whose biosensor cassette IS given an integration locus and so DOES enter the
genotype.)

### Retention accounting

Source 100,493 guides x 3 rounds = 301,479 cells.

| rule | records |
|---|---|
| `random_negative_control_guide` (300 guides) | 900 |
| `source_corrupted_gene_name` (16 guides) | 48 |
| `target_gene_is_not_a_current_genome_gene` (2,670 guides, 168 distinct names) | 8,010 |
| `guide_round_has_no_enrichment_value` | 26,169 |
| `guide_targets_its_own_round_background` | 48 |
| `ambiguous_strain_identity_shared_gene_and_spacer` (150 groups) | 318 |
| **kept** | **265,986** |

### Provenance

Raw mirror `$DATA_ROOT/torchcell-raw/lianMultifunctionalGenomewideCRISPR2019/` now holds
exactly the two files the loader consumes: `data/guide_enrichment_final.tsv` (DERIVED, with
a `ProcessingRecord` naming the versioned `experiments/016-lian-magic-reprocess` pipeline,
its barcode windows, its CPM(+1) normalization, its validation against the paper's hits, and
the sha256 of the Supplementary Data 4 reference it mapped against) and
`si/si_data/41467_2019_13621_MOESM5_ESM.xlsx` (a `springer_esm` retrieval, re-retrieved and
sha256-verified 2026-09-12). Before this, neither the derived table nor any of its inputs
appeared in any manifest.

### Verifier result (verbatim, streaming, scratch runner with this entry's parameters)

```
crispr_magic_lian2019: PASS
  [ok] L0 structural: 265986 records validated
  [ok] L1 count: observed 265986, expected 265986
  [ok] L1 pair_uniqueness: 265986 unique (strain, condition) records, one each
  [ok] L1 provenance_gaps: 531972 documented provenance gaps over 265986/265986 records; 0 deferred field(s)
  [ok] L1 canonical_gene_names: 5057 systematic names, one canonical spelling each, each current in the genome
  [ok] L2 value_fidelity: 265986 values checked
  [ok] L2 se_nonnegative: 265986 values checked
  [ok] L2 uncertainty_sanity: 265986 labelled uncertainties, none a zero dispersion
  [ok] L3 measurement_type_consistent: single measurement_type: MeasurementType.log2_ratio
  [ok] L3 reference_zero: numeric rule: reference response == 0 for all 265986 records
  [ok] L3 environment_perturbed: all 265986 experiments carry an environmental edit
       (baseline temp=30.0, media='SED-URA + 200 ug/mL G418')
  [ok] L3 compound_identity: 265986 compound references carry a structure identifier
  [ok] L3 media_compound_identity: 797958 compound references carry a structure identifier
  [ok] L3 media_membership: 265986 records on a shared MEDIA_LIBRARY medium (1 distinct media)
  [ok] L4 gene_containment_sgd: 1.000 of 5057 measured genes are S288C reference genes
  [ok] L4 current_genome_genes: every one of the 5057 measured systematic names is current
```

The verifier entry should carry `stream: True`: the eager path materializes all 265,986
records and takes roughly 20 minutes wall clock, the streaming path is single-pass.

### Open flags

- Plasmid-borne vs integrated is still unrepresentable. In rounds 2 and 3 the foreground
  guide is on a URA3 plasmid while the SIZ1i / NAT1a backgrounds are genomically integrated,
  and `CrisprConstruct` has only nullable `effector_plasmid_uri`/`_sha256` pointers. A
  `delivery: Literal["plasmid","integrated"] | None` field on `CrisprConstruct` would close
  it and is additive (the class is in no served closure).
- Designed-vs-realized is unmarked on all 265,986 pooled-library CRISPR genotypes. That is
  the documented deferral in memory `designed-vs-realized-perturbation-material-entity`, not
  something this build measured.

## 2026.09.13 - The 318 ambiguous-identity records are kept; donor_sequence joins the strain key

The 2026.09.12 build dropped 318 records in 150 groups where two CRISPRd designs resolve to
the same gene AND the same 21 nt spacer, because the verifier's genotype signature could not
tell them apart. `donor_sequence` now joins that signature, so those designs are distinct
strains and the drop rule is gone. Rebuilt count: **266,304** records (265,986 + 318, the
arithmetic holds exactly).

Retention is now six rules, none of them identity-based:

| rule | records |
|---|---|
| `random_negative_control_guide` (300 guides) | 900 |
| `source_corrupted_gene_name` (16 guides) | 48 |
| `target_gene_is_not_a_current_genome_gene` (2,670 guides, 168 distinct names) | 8,010 |
| `guide_round_has_no_enrichment_value` | 26,169 |
| `guide_targets_its_own_round_background` | 48 |
| **kept** | **266,304** |

### The identity fix reads one level too deep, measured

`_genotype_signature._identity` in `torchcell/verification/environment_response.py` now adds
`crispr["donor_sequence"]` to the key. That field does not exist there: `donor_sequence` is a
field of `CrisprDeletionPerturbation`, one level ABOVE `crispr`, and `CrisprConstruct` is a
`ModelStrict` that has no such slot. The clause is therefore never taken.

Measured on the rebuilt 266,304-record LMDB
(`scratchpad/serve50/smith_lian_group/measure_donor_identity.py`, which monkeypatches a local
copy of `_identity` rather than editing the shared file):

```
as-applied : 266304 records -> 266136 unique keys; 168 duplicate records in 150 groups
corrected  : 266304 records -> 266304 unique keys; 0 duplicate records in 0 groups
```

The one-line correction is to read the field off the perturbation, outside the `crispr`
block:

```python
if p.get("donor_sequence") is not None:
    ident = ident + (p["donor_sequence"],)
```

Until that lands, L1 `pair_uniqueness` reports 168 duplicate records for this dataset. The
BUILD is correct either way: the 318 records are real, independently measured designs, and
the collision is purely in the verifier's key.

### Build hazard worth recording

Clearing `processed/` alone is not enough to rebuild this dataset. `experiment_reference_index`
is cached in `preprocess/experiment_reference_index.json` and is loaded whenever that file
exists, so a rebuild that changes the record COUNT reuses the previous index and the
`post_process` coverage assertion fails with "Each item in the dataset must be covered by
exactly one reference" (here: 265,986 covered indices against 266,304 records). Move
`preprocess/experiment_reference_index.json` and `preprocess/build_manifest.json` aside along
with `processed/`.

## 2026.09.30 - Raw sha256 pin verified at build time

Issue #518 (sweep); the whole sweep is in [[torchcell.data.experiment_dataset]] (2026.09.30). Before: download-only check: yes; PyG skips `download()` when `raw/` is populated, so a file placed or edited in `raw/` built unchecked; copy before check: no; refused deposit leaving a directory: partial.

Now `process()` starts with `verify_raw_files(self.raw_dir, ...)` against `TSV_SHA256`, `DESIGN_D_SHA256`, before any record is read, and raises `RawSha256MismatchError` ("sha256 mismatch for <file>: expected <pin>, observed <digest>") with no store written. `download()` stages files through the shared `copy_verified` / `write_verified` / `link_verified` helpers, which hash before writing, so a refusal leaves nothing in `raw/`. Records built from a verified raw file are unchanged. Test: `test_a_raw_file_off_the_pin_is_refused_at_build_time` (or the renamed former Finding test) in the paired test file.

## 2026.09.30 - Module constant is the one pin at download (issue #561)

Before this change, `download()` verified the mirror bytes against the digest recorded in the raw-mirror `manifest.json`, and `process()` verified them against the module constant. Because `deposit_raw_mirror` writes the manifest from the constant, the two were equal by construction, but the loader still carried two pins. Now `download()` verifies the bytes against the module constant, and `check_manifest_pin` refuses a manifest that records any other digest, raising `ManifestPinMismatchError` named by path with both digests. The manifest stays the retrieval record. Built records are unchanged. Tests: `test_download_refuses_a_manifest_digest_off_the_module_pin` in [[tests.torchcell.datasets.scerevisiae.test_raw_pins]].

## 2026.10.07 - Typed strain background, and the integration locus IS stated

### The corrected locus claim

The previous build's module docstring said of the bAID host that "the integration LOCUS is
not stated anywhere in the release". That is wrong. Supplementary Table 11, "Strains
constructed in this study", in the mirrored SI OCR
(`si/si1.md`, sha256 `b2bcfe2e672674438216472e3e06903c93d4ee54cd8b6fd9b5f964ad2a3d32db`)
carries the strain table, verbatim:

```text
BY4742 | MATα his3∆1 leu2∆0 lys2∆0 ura3∆0
bAID   | BY4742-Delta::KanMX-[dLbCpf1-VP]-[Csy4]-[dSpCas9-RD1152]-[SaCas9]
R1     | bAID-X3::SIZ1i
R2     | bAID-X3::SIZ1i-X4::NAT1a
R3     | bAID-X3::SIZ1i-X4::NAT1a-XI1::PDR1i
```

So bAID is BY4742 with the CRISPR-AID cassette integrated at the **Delta** site (the Ty1
delta repeat family), carrying KanMX plus the four orthogonal effector cassettes, and each
round strain adds a gRNA cassette at a named pre-selected locus. The landing pads are named
in the Results ("integrate heterologous cassettes into X2, X3, X4, XI1, XI2, XI3, XII1,
XII2, XII4, and XII5 loci"), and the integrations are marker-less ("in a CRISPR-assisted
and marker-less manner").

The round-to-host mapping is sourced from the SI figure legends, not inferred: round 2 was
screened in R1 ("the second round iMAGIC screening identified targets when integrated into
the X4 locus of R1 strain (SIZ1i)", Supplementary Fig 4) and round 3 in R2 ("the third
round iMAGIC screening identified targets when integrated into the XI1 locus of the R2
strain (SIZ1i-NAT1a)", Supplementary Fig 6; the paper's Results says the same, "we used the
NAT1a and SIZ1i-integrated strain (R2) as the parent strain for the third round"). R3 is the
product of round 3 and is not itself screened here.

### What the records look like now

The dataset moved to the strain-resolved family, the same shape vanacloig2022 uses:
`StrainEnvironmentResponseExperiment` / `StrainEnvironmentResponseExperimentReference`,
`StrainReferenceGenome`, `CultureEnvironment`.

- **The reference genome carries a typed `StrainBackground` per round** (`round_background`):
  BY4742's four auxotrophies as `BackgroundAllele`s, plus the host's cassettes as the new
  `IntegratedCassette` entries. The strain string is the SI's own genotype column (`bAID`,
  `bAID-X3::SIZ1i`, `bAID-X3::SIZ1i-X4::NAT1a`) rather than its terse `R1` / `R2` label, so
  the join key says what the strain is.
- **`locus` is the site verbatim and `locus_systematic_gene_name` stays None** for every
  cassette here: a Ty1 delta repeat family and an intergenic landing pad are not R64 ORFs,
  and giving one a systematic name would be an invention.
- **The round's cassettes left the `Genotype`.** They are constant across every record of
  that round and shared with the round's reference strain, which is exactly the line the
  schema draws between a background and a genotype, so each record's `Genotype` now holds
  the ONE screened guide. A round-2 record is still the strain in the tube: the background
  states what else it carries, and at which locus. Consequence to know when reading older
  analyses: perturbation count per record is now 1 in every round, where it used to be
  1/2/3 by round. Record count is unchanged, because every drop rule is unchanged (the
  `guide_targets_its_own_round_background` rule still fires on the round's background ORFs,
  which `ROUND_BACKGROUND` still records alongside each cassette's designation and locus).
- **The culture protocol is typed**: 50 mL in a shaken 250 mL baffled flask at 250 rpm.
  `endpoint` is a typed `ProvenanceGap`, not a value: the cultures were read at mid-log
  phase, and `EndpointRule`'s three members are a fixed duration, a fixed generation count
  and control saturation, so picking one would assert a rule the paper did not use.
  `inoculum_od600` is likewise gapped, because the OD 0.05 the Methods state belongs to the
  individually constructed validation strains in culture tubes, not to the pooled screening
  flasks these records come from. `pre_culture` is gapped (the library-construction
  paragraph's "cultured in 50 mL SED-URA/G418 medium for ~2 days" fits no
  `PreCultureSource` member, and labeling it `overnight_culture` would assert a time the
  source contradicts), and so is `auxotroph_supplements` (SED-URA's CSM-URA complements
  His/Leu/Lys inside the medium, and uracil is deliberately withheld to select the guide
  plasmid).
- **The BY4742 auxotrophies stay asserted with a pending-review gap** naming Brachmann
  1998, even though the SI states the genotype string. The SI says WHICH alleles the strain
  carries; it does not say how any of them was made, and `STANDARD_ALLELES` reads each
  designation as a specific `AlleleEdit` (`his3Δ1` partial, the three `Δ0` alleles full)
  that only Brachmann 1998 states.

### One sharp edge worth remembering

A `SourcedValue` whose `value` is a dict with INTEGER keys does not survive the record's
JSON round trip: the stored copy comes back with string keys, so the freshly built
reference and the stored one compare unequal and the synthetic build test fails with a diff
deep inside the background's provenance. `ROUND_STRAIN_GENOTYPES` is therefore a LIST in
round order, with `ROUND_STRAIN` as the derived round-keyed lookup for code. Any
`SourcedValue` that is embedded in a record (rather than only read at build time, like
`FURFURAL_MM`) has to be JSON-stable.

## 2026.10.08 - Correction: the round edits stay in the genotype, bAID is a shared helper

The move described in the section above went too far in one respect and not far enough in
another, and both are corrected in this build. The corrected locus claim stands, and so
does everything else that section records (the strain-resolved family, the typed
`CultureEnvironment`, the `IntegratedCassette`, the JSON-stability lesson).

**The round's accumulated edits go back INTO the record's `Genotype`.** Round-2 records
carry 2 perturbations and round-3 records carry 3, the screened guide plus the accumulated
CRISPR perturbations, exactly as `crispr_perturbation(...)` built them before the move. A
round-2 record is the 2-perturbation strain in the tube, and the genotype is where that
belongs. `ROUND_BACKGROUND` is back to `(ORF, common name, mode)` entries, which is also
what the `guide_targets_its_own_round_background` drop rule reads. The SI's integration
loci (SIZ1i at X3, NAT1a at X4), the marker-less integration sentence, the pre-selected
landing pads and the round-to-host legends are kept as a documentation comment on
`ROUND_BACKGROUND` rather than as reference-background content, so the sourcing is still
in the module and can be promoted later if the loci become record content.

**Every round's reference uses ONE typed background named `bAID`**: BY4742's four
auxotrophies plus the single Delta-site `IntegratedCassette`
`Delta::KanMX-[dLbCpf1-VP]-[Csy4]-[dSpCas9-RD1152]-[SaCas9]`, marker KanMX. The per-round
strain strings (`bAID-X3::SIZ1i`, `bAID-X3::SIZ1i-X4::NAT1a`) and `ROUND_STRAIN_GENOTYPES`
are gone; the strain is `bAID` for all three rounds.

**The background is now a shared helper, `strain_background.baid_background()`**, with
`BAID_STRAIN = "bAID"`, because the in-house Bioscreen dataset
(`torchcell/datasets/private_torchcell/`, coming next) was run on the same strain under
the thesis name **BY4742-iAID6** (pAID6 integrated into BY4742, G418 selection). Both
datasets therefore join on one typed background rather than on two spellings of one
strain. The helper carries its own provenance: `BAID_PARENT_GENOTYPE` and `BAID_GENOTYPE`
quote Supplementary Table 11 against the pinned `si/si1.md`, and `BAID_CONSTRUCTION`
quotes the paper's construction sentence against the pinned `paper.md`. The loader binds
them as `HOST`, `HOST_GENOTYPE` and `HOST_PARENT_GENOTYPE`, which keeps them in the
module's own `SourcedValue` audit, so the quotes are still verified against the mirror by
`test_every_sourced_value_is_backed_by_its_verbatim_quote`.

Tests: `test_the_reference_genome_is_the_one_shared_baid_background` in
[[tests.torchcell.datasets.scerevisiae.test_lian2019]] (one background, one cassette), the
1/2/3 perturbation counts in
[[tests.torchcell.datasets.scerevisiae.test_lian2019_synthetic]], and
`test_baid_background_is_by4742_plus_the_delta_site_crispr_aid_cassette` plus
`test_baid_background_round_trips_through_json_unchanged` in
[[tests.torchcell.datamodels.test_strain_background]].
