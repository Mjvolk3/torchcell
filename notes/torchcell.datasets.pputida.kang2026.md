---
id: 1fk26m31r24q2pxu7ei8nig
title: Kang2026
desc: ''
updated: 1791392699475
created: 1791392699475
---

## 2026.10.07 - Kang 2026 isoprenyl acetate loader

`torchcell/datasets/pputida/kang2026.py`, one dataset class
`IsoprenylAcetateTiterKang2026Dataset` (`ProductTiterExperiment` /
`ProductTiterExperimentReference`), row 18 of the bacterial expansion list. Template:
[[torchcell.datasets.pputida.carruthers2025]]. Skeleton:
[[torchcell.datasets.bacteria_common]].

Kang et al. 2026, "Multi-layered metabolic remodeling of Pseudomonas putida for
efficient conversion of lignocellulosic sugars to the precursors of advanced aviation
fuel", Metab. Eng. Commun., doi:10.1016/j.mec.2026.e00274, citation key
`kangMultilayeredMetabolicRemodeling2026`.

### Pins

| artifact | where | sha256 |
|---|---|---|
| `paper.md` (MinerU OCR) | `$DATA_ROOT/torchcell-library/kangMultilayeredMetabolicRemodeling2026/` | `894aee24472194d22c33ecc0a31994660b19bde41e4aadee2cb0080f0c016c63` |
| `si/si1.docx` (publisher `mmc1.docx`) | `$DATA_ROOT/torchcell-raw/kangMultilayeredMetabolicRemodeling2026/` | `c7d4567fae037c7392e39b855cc71f83b7b4a5d96699c3f18b991a25c57f09e0` |
| Carruthers 2025 `paper.md` (deferred-to) | `$DATA_ROOT/torchcell-library/carruthersAutomationMachineLearning2025/` | `ca9a8a2593d2ae3ab3bacfb767e797ece2bdaa0228a4f798f1c38e1af73ef88d` |

The SI document is the only released data file. Its retrieval is scriptable and recorded:
`RetrievalMethod.pmc_cloud` on
`https://pmc-oa-opendata.s3.amazonaws.com/PMC12996797.1/mmc1.docx`, through
`torchcell.literature.retrieve.pmc_cloud_object`. The loader reads it with a
standard-library `.docx` table reader (`read_si_tables`, `si_table`): zipfile plus
ElementTree over `word/document.xml`, no new dependency, cell text normalized to single
spaces so a quote is matchable.

Every sourced value in the module quotes one of those three files verbatim. 87 quotes
checked; the data-gated test
`test_every_sourced_quote_is_verbatim_in_its_pinned_mirror` re-reads all of them.

### The three titer columns, and which one is primary

1. **Table 1, `Titer (mg/L); culture conditions`** -- the primary column: 7 strains, each
   the flask maximum under its own stated conditions. It lives in the article body, not
   in any released data file, so it is carried as sourced constants quoting the pinned
   `paper.md` and `verify_paper_quotes()` hashes those bytes and re-reads all 15 Table 1
   quotes before a single value is used.
2. **Table S4, `IPA titer (mg/L)`** -- the 5-member AAT panel in the PIPA background, 5 mL
   tube cultures at 48 h.
3. **Table S9, `Isoprenyl acetate, aqueous / organic / off-gas (mg/L)`** -- the fed-batch
   time course of the final strain. The stored titer is the SUM of the three phases.

19 records: 7 + 5 + 7. Nothing is dropped.

| source | strain | titer (mg/L) | culture | time (h) |
|---|---|---|---|---|
| Table 1 | PIPA-AAT1 | 414 | flask | gap |
| Table 1 | PIPA-E3 | 724 | flask | gap |
| Table 1 | PIPAxyl-AAT1 | 181 | flask | gap |
| Table 1 | PIPAxyl-E3 | 383 | flask | gap |
| Table 1 | PIPAxyl-E3-K3 | 599 | flask | 144 |
| Table 1 | PIPAxyl-E3-O15 | 819 | flask | gap |
| Table 1 | PIPAxyl-E3-K3-O15 | 1500 | flask | gap |
| Table S4 | PIPA-AAT1 (ATF1) | 405 | tube | 48 |
| Table S4 | PIPA-AAT2 (ATF2) | 157 | tube | 48 |
| Table S4 | PIPA-AAT3 (SAAT) | 22 | tube | 48 |
| Table S4 | PIPA-AAT4 (AAT) | 119 | tube | 48 |
| Table S4 | PIPA-AAT5 (CAT) | 15 | tube | 48 |
| Table S9 | PIPAxyl-E3-K3-O15 | 98.4 | bioreactor | 21.4 |
| Table S9 | PIPAxyl-E3-K3-O15 | 1053.2 | bioreactor | 45.4 |
| Table S9 | PIPAxyl-E3-K3-O15 | 1445.1 | bioreactor | 69.4 |
| Table S9 | PIPAxyl-E3-K3-O15 | 1683.7 | bioreactor | 93.4 |
| Table S9 | PIPAxyl-E3-K3-O15 | 1606.2 | bioreactor | 117.4 |
| Table S9 | PIPAxyl-E3-K3-O15 | 1909.4 | bioreactor | 141.4 |
| Table S9 | PIPAxyl-E3-K3-O15 | 1435.1 | bioreactor | 165.4 |

Five Table 1 strains carry no titer and are therefore not records: `PIPAxyl`,
`PIPAxyl-O2`, `PIPAxyl-O6`, `PIPAxyl-O7`, and the integration variant whose name the OCR
renders `014` (`PIPAxyl-E3-O14`). They are written to
`preprocess/strains_without_a_titer.csv`.

The two statements for PIPA-AAT1 are kept side by side rather than reconciled: Table 1's
414 mg/L is a flask maximum and Table S4's 405 mg/L is a 48 h tube endpoint. The loader
asserts the tube number is the lower of the two, so a change in either column that
inverted them would stop the build.

### The chassis genotype, as modeled

The reference of every record is **PIPA**, a `BacterialStrainBackground` on an
`AssemblyReferenceGenome` pinned to `pputida_KT2440_ASM756v2` / `GCA_000007565.2`.
`parents = ["KT2440"]`; the parent's own genotype is quoted from Table S2, "Wild-type
mt-2 derivative lacking the TOL plasmid pWW0".

`genotype_statement` is Table S2 verbatim: "P. putida KT2440 ΔphaABC ΔmvaB ΔhbdH ΔldhA
Δ86kb (4,536,184-4,627,926; PP_4023-PP_4092)". Table 1 states the same strain WITH a
locus tag per gene: "P. putida KT2440 ΔphaABC (PP_5003-5005) ΔmvaB (PP_3540) ∆hbdH
(PP_3073) ∆ldhA (PP_1649) ∆86kb (4,536,1844,627,926; PP_4023-PP_4092)" (the OCR loses the
hyphen in the coordinate range, which is why Table S2 is the source for the span).

Typed alleles: 70 in all, 6 named individually and 64 carrying the Δ86kb
`deleted_span` (63 full deletions plus one truncation).

| locus | symbol | designation | edit | how the locus was established |
|---|---|---|---|---|
| PP_5003 | phaA | ΔphaABC | full_deletion | Table 1, and the annotation resolves `phaA` to it |
| PP_5004 | phaB | ΔphaABC | full_deletion | Table 1, and the annotation resolves `phaB` to it |
| PP_5005 | phaC | ΔphaABC | full_deletion | Table 1's range `PP_5003-5005` ONLY; `phaC` is RETIRED against this assembly, which carries PP_5005 as `phaC-II` |
| PP_3540 | mvaB | ΔmvaB | full_deletion | Table 1, annotation agrees |
| PP_3073 | hbdH | ΔhbdH | full_deletion | Table 1, annotation agrees |
| PP_1649 | ldhA | ΔldhA | full_deletion | Table 1, annotation agrees |
| 63 loci | annotation symbol or tag | Δ86kb (4,536,184-4,627,926; PP_4023-PP_4092) | full_deletion | both statements of the Δ86kb cell |
| PP_4023 | (none) | same | partial_deletion | the span starts inside this gene |

Where the paper and the annotation both name a locus the two must AGREE; the builder
refuses rather than preferring one. For `phaC` the symbol must still be RETIRED, or the
reason recorded for PP_5005 is no longer true and the build stops.

**What this paper settles that Carruthers 2025 could not.** The mirrored Carruthers
campaign records `phaC` as an unmappable gap on its sibling chassis IY1449b (the symbol
resolves to nothing on this assembly). Kang's `ΔphaABC (PP_5003-5005)` places it at
PP_5005. The same is true of `crc`: the symbol is RETIRED and PP_5292 carries no symbol
in the annotation, and Table 1 writes `Δcrc (PP_5292)`.

### The Δ86kb deletion states itself three ways, and they disagree

One cell gives a label, a coordinate span and a locus-tag range. Measured on
GCA_000007565.2:

| statement | what it says | measured |
|---|---|---|
| label | `Δ86kb` | the coordinate span is **91,743 bp**, not 86 kb |
| coordinates | 4,536,184-4,627,926 | **67** annotated loci lie entirely inside it; `PP_4023` (4,535,579..4,538,407) straddles the left edge |
| locus-tag range | PP_4023-PP_4092 | **64** annotated tags; 6 of the 70 numbers are not annotated genes here |

The typed alleles are the **63** loci both statements agree on, plus `PP_4023` as a
`partial_deletion`. The four loci inside the span that the tag range does not name
(`PP_5652`, `PP_5653`, `PP_5654`, `PP_mr45` -- later-added annotations whose tags sit out
of coordinate order) and the six unannotated tag numbers (4026, 4028, 4062, 4086, 4087,
4088) are written to `preprocess/span_disagreement.csv` rather than typed. Nothing is
chosen by preference: every statement is kept and the intersection is the conservative
set.

### Perturbations on top of PIPA

Everything above the background is a perturbation in `Genotype`, the Carruthers pattern.

| cassette | localization | genes (verbatim tokens) | promoter | organism source |
|---|---|---|---|---|
| `pIY670` | episomal_plasmid | MvaSef, MvaEef, MKmm, PMDHKQ, AphA | PBAD (first two), Ptrc1-O (rest) | part-token suffix for `ef`/`mm`; Carruthers 2025 for PMD; Kang's own prose for AphA |
| `pAAT1`-`pAAT5` | episomal_plasmid | ATF1, ATF2, SAAT, AAT, CAT | PnagAa | Methods 2.1, with each accession |
| `pXyl` | chromosomal_integration at PP_1444 | xylE, xylA, xylB, talB, tktA | PxylE* (xylE), Ptac (rest) | Results 3.4, "codon-optimized E. coli genes" |
| `pO15` | chromosomal_integration at PP_2876 | acsEc, xfpkBa | Pm | Table 1 / Methods 2.1 |

`xylA` and `xylB` share the construct token `xylAB`, which is how the source writes the
two adjacent genes; `CassetteGene.construct_token` carries that, and every other token
must appear verbatim in its cassette's quoted construct string.

Deletions, each a `BacterialDeletionPerturbation`: PP_1127, PP_3812, PP_4218 (the three
esterases), PP_1444 (`gcd`, the xylose-cassette integration site), PP_5292 (`crc`),
PP_1021 (`hexR`), PP_2876 (`ampC`, the acetyl-CoA-cassette integration site). The gene
name is the source's own symbol where Table 1 gives one, otherwise the annotation's
(PP_1127 is `estC`), otherwise the tag.

**The pIY670 organism deferral.** Kang's Table 1 cites pIY670 to Banerjee et al. 2024,
which is not mirrored. The organism of `MvaSef`, `MvaEef` and `MKmm` is read from the
part token's own suffix in the quoted plasmid description (`ef` = *Enterococcus
faecalis*, `mm` = *Methanosarcina mazei*), with the reading stated in the
`SourcedValue.note`. `PMDHKQ` carries no suffix and no Kang statement at all, so it is
sourced by deferral to the mirrored Carruthers 2025 paper, which carries the same pIY670
and names the decarboxylase's organism: "promiscuous mevalonate decarboxylase (PMD\*)
from S. cerevisiae". The provenance chain is the citation.

### n_samples and the uncertainty type

**Sourced, every figure caption, verbatim:** "Error bars indicate the standard deviation
of biological triplicates." (Figs. 3, 4, 5, 6 and Figs. S2-S5, S8); Fig. 2 and Figs. S1,
S6, S7, S9, S10 write the same thing as "Error bars represent the standard deviation of
biological triplicates."

So `n_samples = 3`, `sample_unit = biological_replicate`, and the uncertainty TYPE is a
sample SD (`SE = SD / sqrt(3)` if the numbers existed).

**The uncertainty NUMBER is not released anywhere in the mirror.** This paper has no
source-data file, no per-replicate table and no SD column in any SI table; the replicate
spread appears only as error bars in the figures. `ProductTiterPhenotype` forbids an
unlabelled uncertainty (the number and its type are both set or both None), so
`titer_uncertainty` and `titer_uncertainty_type` are typed `ProvenanceGap`s with reason
`not_reported_by_primary`, each carrying the design quote in its note. This is "measured
and null" in the strict sense: nothing is fetchable, so the gap is terminal rather than
deferred.

**The fed-batch rows additionally gap `n_samples` and `sample_unit`.** Methods 2.8
describes a single run in a 2 L DASGIP bioreactor and never states a replicate count for
it; the triplicate design is stated for the tube and flask figures only.

`product_yield` is released once, 0.067 g/g, for the fed-batch run's maximum titer, and is
carried on that one record; `productivity` is never released and is gapped on every
record.

### Environment

The three Kang media are already sourced in [[torchcell.datamodels.media]] and are all
carbon-source free, so each sugar is an
`EnvironmentPhysicalPerturbation(factor=carbon_source)` naming its molecule, and every
inducer, overlay and supplement is a `SmallMoleculePerturbation`.

| condition | media object | sugars (g/L) | additions |
|---|---|---|---|
| `tube_m9_glc20` | `M9_NREL_KANG2026` | Glc 20 | Ara 2 g/L, SA 125 uM, dodecane 20% v/v; 48 h |
| `flask_m9_glc20` | `M9_NREL_KANG2026` | Glc 20 | same, duration gapped |
| `flask_m9hn_glcxyl_1010` | `M9_NREL_HIGH_N_KANG2026` | Glc 10, Xyl 10 | Ara 2 g/L, SA 125 uM, dodecane 20% |
| `flask_m9hn_glcxyl_1010_pulse` | `M9_NREL_HIGH_N_KANG2026` | Glc 10, Xyl 10 | same, 144 h, 48 h pulse |
| `flask_mops_ye_glcxyl_1010_pulse` | `M9_MOPS_KANG2026` | Glc 10, Xyl 10 | YE 1 g/L, (NH4)2SO4 40 mM, Ara 2 g/L, SA 62.5 uM, 3MB 250 uM, Durasyn 164 20% |
| `flask_mops_ye_glcxyl_137_pulse` | `M9_MOPS_KANG2026` | Glc 13, Xyl 7 | same |
| `fedbatch_mops_ye_glcxyl_1337` | `M9_MOPS_KANG2026` | Glc 13.3, Xyl 6.7 | YE 1 g/L, Ara 2 g/L, SA 62.5 uM, 3MB 250 uM, Durasyn 164 20% |

Temperature is 30 C, sourced from Methods 2.4's seed and adaptation steps and again
explicitly from Methods 2.8's bioreactor; the production culture's own temperature is
never restated and no other temperature appears anywhere in the Methods.

Three things the slot cannot hold, recorded instead of asserted:

- `duration_hours` is a typed gap on 6 of the 7 Table 1 records. The primary column is a
  MAXIMUM over a flask time course sampled every 24 h and the source states the time for
  one strain only (PIPAxyl-E3-K3, 144 h).
- the vessel and working volume. `ProductTiterExperiment.environment` is annotated
  `Environment`, not `CultureEnvironment`, so pydantic would serialize them away. They
  are in the module's `CULTURE_FORMATS` (50 mL tube / 5 mL, 250 mL unbaffled flask /
  50 mL, 2 L DASGIP / 1.2 L).
- the 48 h sugar + nitrogen pulse four conditions use. No `Environment` slot holds a
  mid-culture feed event, so the flag is per record in `preprocess/titer_rows.csv`.

The condition quotes themselves are written to `preprocess/condition_provenance.json`,
one list per condition, so the environment a record carries is auditable quote by quote.

Also recorded rather than asserted: Methods 2.8 calls the fed-batch medium "modified
M9-MOPS" and never defines it. "Modified M9" is defined for the NREL-type M9 only
(ammonium sulfate raised to 40 mM). The served medium is therefore the plain
`M9_MOPS_KANG2026` object and no raised ammonium level is asserted for that condition.

### References

Four, one per family, each the baseline the paper's own fold-changes divide by. The
`genome_reference` is the PIPA pin in all four.

| key | strain | titer (mg/L) | the quote that makes it the denominator |
|---|---|---|---|
| `flask_pipa` | PIPA-AAT1 | 414 | "titers increasing from 414 mg/" ... "to 724 mg/L, a 1.7-fold increase" |
| `flask_pipaxyl` | PIPAxyl-AAT1 | 181 | "a 2.1-fold increase over PIPAxyl-AAT1"; "from 181 mg/L in the unoptimized PIPAxyl strain to 1.5 g/L in shake flasks" |
| `tube_pipa` | PIPA-AAT1 | 405 | "the PIPA-AAT1 strain ... achieved the highest isoprenyl acetate production with a titer of 405 mg/L" |
| `fedbatch` | PIPAxyl-E3-K3-O15 | 1500 | "1.5 g/L in shake flasks and 1.9 g/L under fed-batch cultivation" |

The baselines are also records of their own families, because they are released strain
measurements rather than control cultures. That is the one place this loader differs from
Carruthers 2025, whose per-cycle controls are the reference and are NOT records.

### Identifier histogram

13 distinct locus tags: the 7 engineered deletions plus the 6 chassis genes Table 1
places. Against `pputida_KT2440_ASM756v2`:

| status | n | layer | n |
|---|---|---|---|
| current | 13 | locus tag | 13 |
| renamed | 0 | old locus tag | 0 |
| non_gene_feature | 0 | RefSeq locus tag | 0 |
| retired | 0 | gene symbol | 0 |
| ambiguous | 0 | gene synonym | 0 |
| | | not found | 0 |

0 remapped, 0 kept on collision, 0 retired kept, 0 ambiguous kept, 0 outside
`pputida_kt2440_locus_tag`. `MIN_RESOLVED_FRACTION = 1.0`, and the build additionally
refuses any tag the reconciliation would REMAP: these are all current standard tags, so a
remap means the annotation moved.

The Δ86kb alleles are not reconciled: their tags come from the assembly's own annotation
by coordinate, not from a source name.

### Build

```bash
python -m torchcell.database.build_dataset_lmdb --dataset IsoprenylAcetateTiterKang2026Dataset
```

```
BUILT IsoprenylAcetateTiterKang2026Dataset: 19 records at
$DATA_ROOT/data/torchcell/isoprenyl_acetate_titer_kang2026 in 2s; gene_set size 24;
references 4
```

`gene_set` is 24: the 7 engineered deletion loci plus the 17 heterologous gene tokens the
four cassettes contribute. The build manifest reads `fresh` with empty drift under
`python -m torchcell.provenance.build_manifest`. No KG build was run and nothing under
`$DATA_ROOT/database/` was touched.

`preprocess/` holds `titer_rows.csv` (the per-record ledger: source, column, strain,
condition, titer, time, n_samples, yield, pulse flag, the three fed-batch phases, quote),
`strains_without_a_titer.csv`, `span_disagreement.csv`, `condition_provenance.json`,
`build_accounting.json`, `build_manifest.json` and `verification_report.json`.

### Verification, L0 to L4

Run from the module's own runner, `kang2026.verify_build(dataset_root, data_root)`, which
writes `preprocess/verification_report.json`. There is no `run_product_titer` runner in
`torchcell/verification/runners.py` yet; that belongs to step 6 of
[[plan.bacteria-ontology-genome]].

```
IsoprenylAcetateTiterKang2026Dataset: PASS
  [ok] L0 structural: 19 records validated
  [ok] L1 count: observed 19, expected 19
  [ok] L2 value_fidelity: 19 values checked
  [ok] L2 cross_method: 19 pairs agree within 0.0
  [ok] L3 titer_unit_is_the_sources_mg_per_l_as_ug_per_ml
  [ok] L3 every_uncertainty_is_a_typed_gap_not_a_guess
  [ok] L3 every_genotype_carries_pIY670_and_one_aat
  [ok] L3 the_reference_is_the_pinned_PIPA_background
  [ok] L4 fed_batch_titer_vs_table_s9_phases: 7 overlapping entities agree within 1e-09
  [ok] L4 table1_titer_vs_results_prose: 6 overlapping entities agree within 1e-09
```

The two L4 levels are real cross-source joins:

- **fed-batch**: the 7 stored titers are re-derived from the deposited Table S9 bytes by
  the same three-phase sum.
- **Table 1**: the 6 stored titers that the Results prose also states are joined to that
  prose, a different place in the same pinned mirror from the table they were read out
  of, with each quote asserted verbatim first. PIPAxyl-E3-K3-O15 is excluded because the
  prose writes its titer as 1.5 g/L rather than 1500 mg/L.

The L2 cross_method compares each record's serialized perturbation count to the count
derived from the declarative strain table, matching records to strains by genotype rather
than by position: LMDB keys are strings, so the store iterates "10" before "2" and a
positional join silently pairs the wrong rows.

Three build-time oracles additionally have to hold on the pinned bytes, or the build
stops before the LMDB is opened:

1. Table S9's released `Off-gas fraction (%)` equals off-gas divided by the three-phase
   sum, for all 7 rows within 0.05 percentage points. That is what proves the sum is the
   total the authors used.
2. Table S9's released `Total sugar (g/L)` equals glucose + xylose within 0.15 g/L. The
   tolerance is three independent one-decimal roundings; one real row needs it (93.4 h,
   2.1 + 3.2 = 5.3 against a released 5.2).
3. The maximum three-phase sum, 1909.4 mg/L at 141.4 h, is the 1.9 g/L the Results state.
   The source calls that titer "final" although the 165.4 h end point is lower
   (1435.1 mg/L); it is the campaign maximum, consistent with Table 1's own scoring rule.

The released off-gas fraction at 165.4 h is 51.9%, which is the "52% of the total
isoprenyl acetate was recovered from the off-gas trap" the Results state.

### Tests

`tests/torchcell/datasets/pputida/test_kang2026.py`: 75 tests that run everywhere plus 6
`@pytest.mark.data` tests. `pytest ... --data` with `DATA_ROOT` set: 81 passed.

The hermetic end-to-end test builds the loader over a synthetic KT2440 assembly (the real
`PPutidaKT2440Genome` class over a short synthetic replicon carrying the 13 loci, with
the network refused), a synthetic SI `.docx` and a synthetic OCR mirror, under
`tmp_path`. The synthetic SI reproduces the released Table S4 and Table S9 numbers
exactly, because the oracles those tables feed are joins onto OTHER statements of the
same measurement; a fixture with invented numbers would exercise the plumbing while
retiring the check. The synthetic loci sit at low coordinates, so the Δ86kb span types
nothing there; that arithmetic is pinned by a separate test against a stub genome placed
at the real coordinates.

Diff coverage on `origin/main`: **96.4%** on `kang2026.py`, 100% on the package
`__init__.py` (704 diff lines, 25 missing).

### Open items for the owner

1. `ConcentrationUnit` has no `mg/L`. Every titer here is stored verbatim as `ug/mL`
   (1 mg/L == 1 ug/mL exactly, so no arithmetic touches a source value). Adding
   `mg_per_l` is additive but the enum is in every served closure, so it is a rebuild
   decision.
2. `ProductTiterExperiment.environment` should narrow to `CultureEnvironment`, or the
   vessel, working volume and shaking of a fermentation cannot be stored. Same open item
   as [[torchcell.datasets.pputida.carruthers2025]].
3. `isoprenyl acetate` needs a `compound_identity_table.json` row. The module records
   `ISOPRENYL_ACETATE_INCHIKEY = "OCUAPVNNQFAQSM-UHFFFAOYSA-N"`, DERIVED from the
   structure SMILES `CC(=C)CCOC(C)=O` with `rdkit.Chem.inchi.MolToInchiKey` (rdkit
   2026.03.6, 2026-10-07) and deliberately NOT asserted onto the `Compound`: that table
   is PubChem-sourced through the committed input lists and curating a row is a human
   act. Same for `L-arabinose`, `salicylic acid`, `m-toluic acid`, `D-xylose`, `dodecane`,
   `Durasyn 164`, `ammonium sulfate` and `yeast extract`, each of which comes back with an
   `inchikey` gap today.
4. A `run_product_titer` verification runner, so these levels run from `run_all` rather
   than from this module.
5. An adapter module, a conf yaml and a `product titer phenotype` graph node class before
   this dataset can be admitted to the served graph (step 5 of
   [[plan.bacteria-ontology-genome]]).
6. `Environment` has no slot for a mid-culture feed event (the 48 h pulse) and no slot for
   a continuous feed (the fed-batch 266.7 g/L glucose + 133.3 g/L xylose solution). Both
   are carried in `preprocess/` today.

## 2026.10.08 - Table S9's released aqueous isoprenol: read, ledgered, declined as a record

`SI_TABLE9_COLUMNS` asserted `Isoprenol, aqueous (mg/L)` while `FedBatchRow` declared no
field for it, so the value was parsed past and never read. That is now fixed:
`FedBatchRow.isoprenol_aqueous_mg_per_l` reads it, `preprocess/titer_rows.csv` carries it
per record beside the three ester phases, `_assert_fed_batch_oracles` checks it, and a new
verification level `L4 ledgered_aqueous_isoprenol_vs_table_s9` re-reads the deposited docx
and joins the ledger back to it.

The seven released values, verbatim from the pinned `si/si1.docx` (sha256
`c7d4567fae037c7392e39b855cc71f83b7b4a5d96699c3f18b991a25c57f09e0`), Table S9 captioned
"Table S9. Time-course analysis of sugar consumption and isoprenyl acetate partitioning in
fed-batch fermentation":

| time (h) | 21.4 | 45.4 | 69.4 | 93.4 | 117.4 | 141.4 | 165.4 |
|---|---|---|---|---|---|---|---|
| isoprenol, aqueous (mg/L) | 190.7 | 197.0 | 175.0 | 231.0 | 234.7 | 307.8 | 254.2 |

### Why they are NOT seven records

`ProductTiterExperimentReference` requires a `phenotype_reference` and
`ProductTiterPhenotype.titer` is a required float, so an isoprenol record needs an
isoprenol REFERENCE titer. Measured on the pinned mirror: Table S9 is the only place any
isoprenol number is released at all. Table S1 is a physicochemical property comparison,
Table S4's only titer column is `IPA titer (mg/L)` (the ester), Tables S2, S3, S5 to S8
release no titer, and Fig. 2b's isoprenol bars have no companion table. So there is no
second isoprenol measurement anywhere in this paper to be the denominator.

The one isoprenol baseline the paper states is second-hand, and the module carries it as
`ISOPRENOL_BASELINE_SECOND_HAND` for the record rather than as a reference, quote verbatim
from Results 3.1 of the pinned `paper.md`:

> In the previously reported engineered P. putida KT2440 background, this strain supported
> isoprenol titers of up to 762 mg/L in shake flasks and $3 . 5 ~ \mathrm { g } / \mathrm
> { L }$ in fed-batch cultures (Banerjee et al., 2024).

Three things disqualify it as this run's reference. The measurement is Banerjee 2024's, not
this study's; the strain is the background PIPA was ADAPTED FROM rather than PIPA itself
("Our base strain, PIPA, incorporated the GSMM-guided deletions of six genes ... from that
study"); and no medium, sugar load, vessel or replicate count travels with either number.
Banerjee 2024 is also not in the literature mirror, so the chain cannot be closed from our
own documentation. Writing it as `phenotype_reference` would state a measurement nobody
made in this study.

The sibling [[torchcell.datasets.pputida.yunus2026]] loader refuses its entire titer family
for exactly this shape of gap ("no control titer is stated anywhere, which is why no
ProductTiter family is built"), and that precedent is followed here. The decline, with its
reason, is `ISOPRENOL_NOT_A_RECORD` and is written into
`preprocess/build_accounting.json`'s notes, so it is auditable from the built tree rather
than only from this note.

### What would make them records

Any ONE of: (a) Banerjee 2024 mirrored, so its own fed-batch isoprenol titer for the named
strain becomes a sourced reference with its own conditions and replicate design; (b) a
released isoprenol number for PIPA or any Kang strain under a stated condition, which would
be the denominator; or (c) a schema decision that a `ProductTiterExperimentReference` may
carry a typed-absent titer, which is a served-closure change and not a loader decision.

### No other Table S9 column changed

The stored titer is still the sum of the three isoprenyl acetate phases, the record count
is still 19 (7 Table 1 + 5 Table S4 + 7 Table S9), and the two existing build oracles are
untouched. The two added isoprenol oracles are: the column is positive at every sampled
time (measured 175.0 to 307.8 mg/L, so it is never a blank or zero placeholder), and its
maximum is below the maximum three-phase ester sum (307.8 against 1909.4), which is what
makes the ester and not the precursor this run's product.

### Still not stored, unchanged from the first build

The residual-sugar columns (`Glucose (g/L)`, `Xylose (g/L)`, 7 time points each) stay out
of `MetabolitePhenotype`. Two independent blockers, both measured. `MetabolitePhenotype`
has no units field, so `g/L` would sit inside the free-text `measurement_type`; and
`n_replicates: dict[str, int]` is required with a `>= 1` validator, so it cannot be a
`ProvenanceGap` (a gap must name a field that is `None`). The replicate count for this run
is not stated ANYWHERE in the mirror: `replicate` occurs once in `paper.md` and that hit is
"To replicate this composition, glucose and xylose are commonly added in a 2:1 ratio",
every other figure caption says "Error bars indicate the standard deviation of biological
triplicates" while Fig. 8's caption says only "Cultivations were performed with 1 L medium
and 200 mL overlay with sampling approximately every 12 h", and the Table S9 caption states
no replicate count. So `n = 1` would be an assumption, not a reading, which is why
`titer_phenotype` already gaps `n_samples` and `sample_unit` for the fed-batch rows.

## 2026.10.09 - This row has no called variants at all, measured

Issue #731 landed the three called-variant leaves, and this row was on the list they were
expected to extend. It does not extend, for the plainest possible reason: **this paper
performs no resequencing and releases no variant table.** Measured on the pinned bytes
(`si/si1.docx`, sha256 `c7d4567fae037c7392e39b855cc71f83b7b4a5d96699c3f18b991a25c57f09e0`;
`paper.md`, sha256 `894aee24472194d22c33ecc0a31994660b19bde41e4aadee2cb0080f0c016c63`):

- All nine SI tables enumerated from `word/document.xml`. **None** carries a position, a
  reference or alternate base, a mutation type, an amino-acid change, a frequency or an
  effect annotation. Table S5 is `No., Locus tag, Predicted enzyme function, Functional
  class`; Table S9 is the time course; nothing is a call table.
- `paper.md` returns zero hits for `resequenc`, `WGS`, `SNP` and `polymorphism`.
  `mutation` occurs twice, both the engineered `Y20F` substitution in the AAT, and `ALE`
  occurs twice, both citing OTHER papers (Lim 2021, Elmore 2020).
- The only deposit is PRIDE `PXD067010` (proteomics). There is no BioProject, BioSample
  or SRA accession anywhere.

The two sequence-level facts this paper does state are engineering, not calls, and both
already have a home: `Y20F` travels as `CassetteGene.variant` on the AAT cassette, and
the ALE-identified single adenosine insertion at -10 of `xylE`'s native promoter is the
asterisk in `PxylE*`, carried as `CassetteGene.promoter`. That insertion is the one place
in this loader where a sequence variant is flattened into a name, and the leaves cannot
take it either: it sits in an introduced cassette, and the release states its position
only as "-10" relative to a promoter, never as a coordinate on a replicon, which
`BacterialVariantCall.reference_sequence` plus `position_start` require. Deriving one
would mean inventing a coordinate for a construct whose sequence the JBEI registry
supplies "upon request" and which is therefore not in the mirror.

The record count is unchanged at 19, and this row's refusals remain the ones already
recorded: the aqueous-isoprenol column of Table S9 (read, oracle-checked and carried in
`preprocess/titer_rows.csv`, and NOT a `ProductTiterExperiment`) and the five strains
Table 1 gives no titer.
## 2026.10.09 - Table S6 stays refused after #770, because the identifier route is the blocker, not the phenotype class

Issue #770 names Kang 2026's Supplementary Table S6 as one of five places a protein-level
fold change is released and the schema could not hold it. `ProteinFoldChangePhenotype`
landed in this wave and holds it. The table is still NOT loaded, for a different reason
that was measured here rather than assumed: its keys cannot be resolved to loci of the
pinned assembly.

Measured by
[[protein_fold_change_refusals_kang_lim|experiments.036-dataset-fixes-before-kg-build.scripts.protein_fold_change_refusals_kang_lim]]
over `si/si1.docx` (sha256
`c7d4567fae037c7392e39b855cc71f83b7b4a5d96699c3f18b991a25c57f09e0`), read through this
module's own `si_table(path, 6)`:

| fact | measurement |
|---|---|
| caption, verbatim | "Table S6. List of top 20 accessory genes upregulated and downregulated by sgRNA targeting PP_4854." |
| columns | `Protein Group`, `Protein Names`, `Protein`, `Protein Description`, `Fold Change`, `Log2 (Fold Change)`, `p-Value (Equal Variance)`, `(-Log10 (p-Value))`, `Category`, `Rank` |
| rows | 41 including the header, so 40 data rows, 40 distinct keys |
| key form | a UniProt accession in `Protein Group` (`Q88HX1`, `Q88DG9`, `Q88C64`) |
| significance | the table is NOT filtered: 29 of 40 rows have `p-Value (Equal Variance)` below 0.05 |
| UniProtKB cross-references on the pinned assembly | **0** in `GCA_000007565.2_ASM756v2_genomic.gbff.gz` |

The last row is the blocker. `DerivedIdentifierRoute` gained a `uniprot_db_xref` member in
this wave, and that member is the route Gupta 2024 resolves 3,225 of 3,262 proteins
through, but it reads the annotation's own
`/db_xref="UniProtKB/Swiss-Prot:<acc>"` entries and the pinned *P. putida* KT2440 GenBank
carries none. For contrast, the same measurement on the pinned MG1655 assembly
(`GCA_000005845.2_ASM584v2_genomic.gbff.gz`) counts 4,281. So the route that unblocks the E. coli
proteomics releases is measurably unavailable for this assembly, and a 40-key dict of
UniProt accessions has no path to locus tags through anything this repo pins as part of
the genome.

The genomes tier does hold `109.P_putida_KT2440.goa`, a UniProt GAF whose synonym column
carries PP_ tags, and it reaches 29 of the 40 accessions (the other 11 are absent from the
GOA entirely, mostly "Uncharacterized protein", and one of the 11 is `Q99ZW2`, Cas9
itself, which is heterologous and has no host locus). Using it would be a THIRD identifier
route neither #770 nor #753 asks for, it would cap the resolved fraction at 0.72 on a
40-key table, and the rank-1 upregulated row is Cas9 at a fold change of 209, which is a
presence/absence artifact of a reference strain that lacks the CRISPRi system rather than
a measured induction. One record of 40 keys is not worth a new route decided in passing.

Recorded as refused and counted, which is the Wang 2018 treatment #770 names as the honest
alternative. The contrast itself is sourced and ready for whoever takes the route
question up, verbatim from the Figure S1 caption: "(a) Volcano plot analysis displaying
differentially expressed proteins in PIPA-D16 (targeting PP_4854) relative to the control
strain PIPA-C (lacking the CRISPRi/dCas9 system)."
