---
id: 2dnb9iuhp2grqunahy9i1rn
title: Banerjee2024
desc: ''
updated: 1791837455709
created: 1791837455709
---

## 2026.10.12 - Banerjee 2024 isoprenol titer loader

`torchcell/datasets/pputida/banerjee2024.py`, one dataset class
`IsoprenolTiterBanerjee2024Dataset` (`ProductTiterExperiment` /
`ProductTiterExperimentReference`), row 39 of the bacterial expansion list, GitHub #788
item 4. Template: [[torchcell.datasets.pputida.kang2026]]. Skeleton:
[[torchcell.datasets.bacteria_common]].

Banerjee et al. 2024, "Genome-scale and pathway engineering for the sustainable aviation
fuel precursor isoprenol production in Pseudomonas putida", Metabolic Engineering,
doi:10.1016/j.ymben.2024.02.004, citation key
`banerjeeGenomescalePathwayEngineering2024`, group-library item `SGZWENVM`.

### Pins

| artifact | where | sha256 |
|---|---|---|
| `paper.md` (MinerU OCR) | `$DATA_ROOT/torchcell-library/banerjeeGenomescalePathwayEngineering2024/` | `6b5b9e8647721569d846af92e5c0059ef9c4bb9eca66a418e8891a7a18d5c18c` |
| `paper.pdf` (publisher, via Zotero) | same | `e7354f36eeffcab351087452c47e3828cefb9092ddde2abcb2bb0f909935596c` |
| `si/si1.pdf` (publisher `mmc1.pdf`) | `$DATA_ROOT/torchcell-raw/banerjeeGenomescalePathwayEngineering2024/` | `c09fe2f914846b3a2dee12d058d62ebb6041c52eb42d4b17b6b028c185ef0108` |
| `si/si1.md` (MinerU OCR of that PDF) | same | `c0313af4bdbd443e7c61ad539ad3f101c88872f28993ac2ad98c154feb4d7ae0` |
| Carruthers 2025 `paper.md` (deferred-to, PMD organism) | `$DATA_ROOT/torchcell-library/carruthersAutomationMachineLearning2025/` | `ca9a8a2593d2ae3ab3bacfb767e797ece2bdaa0228a4f798f1c38e1af73ef88d` |

42 quotes, every one re-read verbatim from the mirror it cites by `verify_quotes()`; the
data-gated test `test_every_sourced_quote_is_verbatim_in_its_pinned_mirror` runs the same
audit, and an L4 level runs it again over the built store.

### The ESM was captured first, and it holds no measurement table

`scripts/lit_capture_si.py` resolved the Elsevier ESM route and captured all six files;
the two the loader consumes are deposited in the raw mirror with `direct_url` retrieval
records. What each file is, measured rather than assumed
(`experiments/database/scripts/banerjee2024_release_checks.py`):

| ESM file | local | what it is |
|---|---|---|
| `mmc1.pdf` | `si/si1.pdf` | Supplementary Information: Figs. S1-S7, Table S1 (plasmids), Table S2 (strains + JBEI ids), Table S3 (the M9 recipe) |
| `mmc2.xlsx` | `si/si2.xlsx` | Supplementary File 1: EFM results, cMCS run details, cMCS results |
| `mmc3.txt` | `si/si3.txt` | Supplementary File 2: the iJN1462 IPP-bypass COBRA model, as JSON |
| `mmc4.xlsx` | `si/si4.xlsx` | Supplementary File 3: OptKnock, OptKnock_constrained, OptForce, Frequency |
| `mmc5.txt` | `si/si5.txt` | Supplementary File 4: the central-metabolism model, as JSON |
| `mmc6.pdf` | `si/si6.pdf` | Supplementary File 5: the combined-score notebook |

**Not one of them releases a per-strain measurement.** The release-checks script walks
every worksheet of both workbooks (2,795 rows) looking for a measurement header and a
campaign strain id in the same sheet. Four sheets carry a measurement WORD and all four
are model constraints: `Glucose` as a cMCS input column holding the 6.3 mmol/gDCW/h
uptake bound, and `Glucose uptake rate 6.0 mmol/gDW/hr` as an OptKnock setting. **No
sheet names a single `XW**` or `IY****` strain**, which is the decisive signal: a
released per-strain table would have to name the strains it is about. The paper's own
data availability reads "Data will be made available on request".

So every measured series in this paper is a figure panel (Figs. 3b, 4b, 4c, 5, 6a, 8a,
8b, 9a, 9b and Supplementary Figs. 1, 2, 4, 5) and the stored titers are the ones the
ARTICLE BODY states numerically, each carried as a sourced constant quoting the pinned
`paper.md`. Nothing is read off a bar.

### The twenty records, in five condition families

| family | n | medium / vessel | reference |
|---|---|---|---|
| EZ rich, pXW1 knockout rounds | 6 | EZ rich + 20 g/L glucose, 5 mL in a 50-mL test tube, 30 C, 200 rpm, 48 h | XW11, 104 mg/L |
| EZ rich, plasmid + phosphatase panel | 4 | same | XW11, 104 mg/L |
| M9 minimal | 3 | M9 (Table S3) + 20 g/L glucose, same tube format | IY1452, 762 mg/L |
| fed-batch | 4 | modified M9 (20 mM NH4Cl), 1 L in a 2 L bioreactor, 25 C | IY1245, 0.5 g/L |
| sorghum hydrolysate | 3 | modified M9 (10 mM NH4Cl), 5 mL culture tube, 30 C, 72 h | IY1452, 841 mg/L |

| strain | genotype above KT2440 | plasmid | titer | time (h) | family |
|---|---|---|---|---|---|
| XW12 | ΔphaABC ΔmvaB | pXW1 | 164 mg/L | 48 | ezrich_knockouts |
| XW15 | ΔphaABC ΔmvaB ΔhbdH | pXW1 | 241 mg/L | 48 | ezrich_knockouts |
| XW16 | + ΔgltA | pXW1 | 19 mg/L | 48 | ezrich_knockouts |
| XW17 | ΔphaABC ΔmvaB ΔhbdH ΔaceA | pXW1 | 12 mg/L | 48 | ezrich_knockouts |
| XW18 | ΔphaABC ΔmvaB ΔhbdH ΔgntZ | pXW1 | 13 mg/L | 48 | ezrich_knockouts |
| XW19 | ΔphaABC ΔmvaB ΔhbdH ΔaceA ΔgntZ | pXW1 | 114 mg/L | 48 | ezrich_knockouts |
| IY782 | ΔphaABC | pIY603 | 80 mg/L | 48 | ezrich_pathway |
| IY721 | ΔphaABC | pIY554 | 395 mg/L | 48 | ezrich_pathway |
| IY846 | ΔphaABC ΔmvaB ΔhbdH | pIY554 | 536 mg/L | 48 | ezrich_pathway |
| IY939 | ΔphaABC ΔmvaB ΔhbdH | pIY670 | 1,111 mg/L | 48 | ezrich_pathway |
| IY1452 | ΔphaABC ΔmvaB ΔhbdH ΔldhA ΔPP_2675 | pIY670 | 762 mg/L | gap | m9_minimal |
| IY1252 | ΔphaABC ΔmvaB ΔhbdH ΔaceA | pIY670 | 816 mg/L | 48 | m9_minimal |
| IY1252 | same | pIY670 | 259 mg/L | 72 | m9_minimal |
| IY1245 | wild type | pIY670 | 0.5 +- 0.1 g/L | 72 | fedbatch |
| IY1262 | ΔphaABC ΔmvaB ΔhbdH ΔldhA | pIY670 | 2.3 +- 0.3 g/L | 96 | fedbatch |
| IY1452 | ΔphaABC ΔmvaB ΔhbdH ΔldhA ΔPP_2675 | pIY670 | 3.5 +- 0.3 g/L | 96 | fedbatch |
| IY1485 | IY1452 + ΔgacA | pIY670 | 2.4 +- 0.3 g/L | 96 | fedbatch |
| IY1452 | same as above | pIY670 | 841 mg/L | 72 | hydrolysate (glucose only) |
| IY1452 | same | pIY670 | 409 mg/L | 72 | hydrolysate 5% |
| IY1452 | same | pIY670 | 432 mg/L | 72 | hydrolysate 10% |

Each titer is stored on the unit its own statement released: the mg/L statements as
`ug/mL` (1 mg/L is exactly 1 ug/mL, so no arithmetic touches a source value) and the
fed-batch g/L statements as `g/L`. 16 records are `ug/mL`, 4 are `g/L`.

**Only the fed-batch family releases an uncertainty.** Those four carry
`titer_uncertainty` with `sample_sd` over the biological triplicates Fig. 8's caption
states. The other sixteen carry a typed gap on BOTH `titer_uncertainty` and
`titer_uncertainty_type`, because their spread exists only as an error bar and the
schema forbids a type without a value.

### Five build oracles recompute the paper's own fold-changes

The titers are constants, so a transcription slip would be invisible without an
independent relation. Each oracle recomputes a fold-change the article states in words
from the stored values, and the tolerance is the precision the SENTENCE carries, not a
fitted slack:

| stated | recomputed | tolerance |
|---|---|---|
| "increased 1.6-fold compared with the starter strain XW11" | 164/104 = 1.577 | 5% |
| "a 2.3-fold increase from the XW11 strain" | 241/104 = 2.317 | 5% |
| "1.5-fold increase from the XW12 strain" | 241/164 = 1.470 | 5% |
| "improved the isoprenol production by 1.3-fold when compared to strain IY721" | 536/395 = 1.357 | 5% |
| Conclusions: "a 10-fold improvement vs. the starting strain" | 1111/104 = 10.683 | 10%, one significant figure |
| "reduced by about 70% - 259 mg/L at 72 h" | (816-259)/816 = 68.3% | 3 points |

Plus two shape oracles: the fed-batch maximum is 3.5 g/L (the abstract's own number),
and an uncertainty appears on the fed-batch family and nowhere else. Every oracle has a
paired refusal test that perturbs one constant and asserts the build stops.

### gacA has no released locus tag, and the deferral contradicts itself

Banerjee names the gene and gives no tag: "the gacA gene was deleted in the 5 genes
knockout strain (IY1452) as previously reported (Eng et al., 2021)".
`BacterialDeletionPerturbation.systematic_gene_name` requires a KT2440 locus tag and
refuses a bare symbol, so the edit cannot be typed from what Banerjee released. Following
the deferral does not close it either, because the mirrored P. putida papers that DO
print a tag for that gene **disagree**:

| paper | `paper.md` sha256 | what it says |
|---|---|---|
| Borchert 2023 | `e156d3b2...` | "gacA (PP_1650) and gacS (PP_4099)" |
| Schmidt 2022 | `acc5c8bf...` | "gacS (PP_1650) and gacA (PP_4099)" |
| Thompson 2020 | `389d0d6c...` | "gacS and gacA (PP_1650 and PP_4099)" |

Two readings give PP_4099 and one gives PP_1650. Taking the majority would be this
loader resolving an identity across sources, which is exactly what the 2026-10-12 rule
forbids. So IY1485's ΔgacA is recorded in `GENOTYPE_NOT_TYPED` with that evidence and
carries no typed perturbation; **its four fed-batch titers are unaffected**. The visible
consequence is that IY1485 and IY1452 now hold identical genotypes and are separated by
their titers, which the loader states rather than hides.

### The M9-minimal reference is a substitution, and it is flagged

Four of five families have a reference the paper names as the comparison. The M9-minimal
family does not: its stated denominator is "over 4-fold higher than WT (IY1245)" and
IY1245's M9 titer is never released. Reconstructing 762/4 is forbidden, so the stored
reference is IY1452's own 762 mg/L in the same condition, which keeps all three released
M9 values as records rather than refusing them for want of a baseline. The substitution
is on the reference's `note`, in `preprocess/build_accounting.json`, and in the PR.
**This is the one judgment call in the loader and it wants the owner's read.**

### XW11 is a reference and not a record

The EZ-rich denominator is XW11 at 104 mg/L, which Banerjee ATTRIBUTES to Wang et al.
2022: "This strain has shown the highest isoprenol level in P. putida (104 mg/L, XW11
strain) ... in our previous work (Wang et al., 2022)". It is the denominator every
EZ-rich fold-change uses, so it is the family's `phenotype_reference`, and it is not one
of this dataset's records because the measurement is Wang 2022's.

### The EZ-rich records carry no inducer

Methods 2.5 states the inducer as a DISJUNCTION, "0.5 mM IPTG or arabinose (2%)", and
never says which strain received which. The plasmid descriptions name both a lacI and an
araC regulator across the panel, so assigning one per strain would be inference. The
statement is kept in `EZRICH_INDUCER_NOT_ASSIGNED` and in
`preprocess/condition_provenance.json`. The M9, fed-batch and hydrolysate families state
"2% arabinose" directly and carry it.

### Four media, because the paper states four recipes

Added to `torchcell/datamodels/media.py` as a delimited block:

| key | what |
|---|---|
| `EZ_RICH_TEKNOVA` | a new base: the vendor defined medium, composition deferred, on the Foo 2014 `M9_DIFCO` pattern |
| `EZ_RICH_BANERJEE2024` | that base + the stated 25 ug/mL kanamycin |
| `M9_BANERJEE2024` | Supplementary Table 3's recipe at the 1X amounts its own "10X M9 Salts \| 100 mL" row gives; 10 mM (NH4)2SO4 |
| `M9_MODIFIED_FEDBATCH_BANERJEE2024` | Methods 2.7: 20 mM NH4Cl, trace metals at an OPEN GAP (the stock composition is printed, the dose never is) |
| `M9_MODIFIED_TUBE_BANERJEE2024` | Methods 2.8: 50 uL of 1 M NH4Cl in 5 mL, which is 10 mM, and trace metals at a stated 0.05% v/v |

The two "modified M9" variants are separate objects because their nitrogen differs by a
factor of two; collapsing them would assert an amount neither statement makes. The M9
salts agree across Table S3 and Methods 2.7 (6.8 / 3.0 / 0.5 g/L), which is what makes
the 10X dilution a reading rather than an assumption. The SI OCR drops subscripts, so the
quoted stock row reads "NaHPO4" and "KHPO4"; the quote keeps that verbatim and the
component note records it.

A paper-internal mismatch is recorded rather than corrected: Methods 2.5 points at
"Supplementary Table 2" for the M9 recipe, but Table S2 is the STRAIN table and the
recipe is Table S3, captioned "Minimal M9 medium recipe".

### What the release carries that no record holds

10 released values, each counted in `not_stored()` and in
`preprocess/build_accounting.json`:

| quantity | n | why not stored |
|---|---|---|
| fed-batch OD600 maxima with their SDs (25.5+-0.7, 20.5+-3.0, 21.0+-3.4, 17.9+-0.2) | 4 | no phenotype class and no `MeasurementType` member holds an optical density (#776 item 3); the Borchert 2023 precedent. The growth DURATION is stored, as `duration_hours` |
| XW17 residual glucose, 6 g/L at 48 h | 1 | `MetabolitePhenotype` would hold the level, but `MetaboliteExperimentReference` requires a reference level and the only comparison the sentence makes is qualitative ("the XW19 strain depleted glucose after 48 h"); storing 0.0 for "depleted" would read a number out of prose that states none |
| the XW13 / XW14 titer range, "14-15 mg/L" | 2 | one range for two strains; splitting it would be this loader's arithmetic |
| growth rates 0.22 and 0.26 h^-1 | 2 | stated for two strain GROUPS, not per strain |
| IY1485's ΔgacA as a typed perturbation | 1 | the tag contradiction above |
| every per-strain titer, OD600 and glucose SERIES | not knowable | bars and curves with no companion table; reading a value off a bar is forbidden, so the count of unreleased values cannot be stated from our documentation |

### Build

```bash
sbatch --export=ALL,DATASET_CLASS=IsoprenolTiterBanerjee2024Dataset \
    database/slurm/scripts/gilahyper_build_dataset_lmdb.slurm
```

Dev tree only (`$DATA_ROOT/data/torchcell/isoprenol_titer_banerjee2024`), built with
`--retire-existing --verify`.

### Candidate gate

`database/candidates/banerjeeGenomescalePathwayEngineering2024.json`, all five gates:

| gate | outcome | reason |
|---|---|---|
| G1 | pass | primary: `status='candidate'`, `klass='Production campaign'`, no aggregation marker |
| G2 | pass | `assembly_set`: KT2440 -> `pputida_KT2440_ASM756v2` resolves and reads |
| G3 | pass | 30 files in library+raw, 2,795 worksheet rows counted, all intact |
| G4 | pass | no key held by another module (1 mention, Yunus 2026's citation) |
| G5 | pass | `ProductTiterPhenotype` exists; compounds resolve |

Outcome **admissible**; the Zotero item is present. The G3 inventory script is
`experiments/database/scripts/banerjee2024_release_checks.py`, whose results land in
`experiments/database/results/banerjee2024_release_checks.json`.

### Tests

`tests/torchcell/datasets/pputida/test_banerjee2024.py`: 39 synthetic tests that pin
every titer value by value, every oracle and its refusal, the genotype cases the release
forces, the environment and phenotype builders with their typed gaps, and the not-stored
ledger; 4 `@pytest.mark.data` tests that re-read the quotes, pin the deposited manifest
and run L0 to L4 over the built store.

## 2026.10.12 - The M9-minimal reference is a typed gap, not a substitution

The owner's decision on the one judgment call the first build flagged: **the M9-minimal
family's parental titer was never released, so the reference carries a
`ProvenanceGap` on `titer`, and the three records stay.** The substitution the first
build made (IY1452's own 762 mg/L standing in as the family's denominator) is undone.

### Why the substitution was wrong

The paper states this family's baseline only as a ratio, verbatim from the pinned
`paper.md` (sha256 `6b5b9e8647721569d846af92e5c0059ef9c4bb9eca66a418e8891a7a18d5c18c`):

> We observed that the IY1452 strain (∆phaABC ΔmvaB ΔhbdH ∆ldhA ∆PP_2675 with plasmid
> pIY670) had the best isoprenol titer at $7 6 2 ~ \mathrm { m g / L }$ , Fig. 5); over
> 4-fold higher than WT (IY1245).

IY1245's M9 titer appears nowhere in the mirror, and the data availability is "Data will
be made available on request". Four routes existed and three are forbidden:

| route | why it fails |
|---|---|
| reconstruct 762/4 = 190.5 mg/L | a ratio plus a child's value is not a released measurement of the parent |
| substitute IY1452's own 762 mg/L (the first build) | states a parental measurement nobody made, and makes a strain its own control |
| refuse the 3 released M9 titers | deletes measured values, which "the data is the data" forbids |
| **typed gap on the reference's titer** | the route `CLAUDE.md` already prescribes for a required field a source does not fill |

### The schema change it needed

`ProductTiterPhenotype.titer` was a required `float`, so the gap could not be expressed.
It is now `float | None` **with no default**, so the field stays required to be STATED
and only a declared absence may state `None`:

- `_check_titer` refuses a `None` that no `ProvenanceGap` names, so an absent titer is a
  declared absence and never a silent one. For every other dataset the contract is
  unchanged in practice: a `None` could not be expressed before and still cannot be
  expressed silently.
- An absent titer also carries no uncertainty, no standard error and no censoring, since
  each is a statement about a value and no value was released.
- Distinct from the #867 censoring case, which is a value the assay BOUNDED; this is a
  value the source never printed. Both are now in the class docstring.

Record: `database/schema_changes/2026-10-12-product-titer-unreleased-parental-titer-typed-gap.json`,
kind `typed_gap`, **`owner_sign_off.signed = false`** -- a retyped field on a served class
needs the owner's sign-off before this lands.

`scripts/schema_impact_check.py --base origin/main` reports the `ProductTiterPhenotype`
closure as impacted: Brunk 2016, Foo 2014, Banerjee 2024, Carruthers 2025, De Siqueira
2025, Kang 2026, Thompson 2019 and Yunus 2026. Measured, and worth stating plainly:
**the change makes zero ADDITIONAL dev stores stale.** `--list-stale --include-private`
returns 161 classes on this branch against 162 on main (the difference is
`RbTnseqTeteneva2024Dataset`, which main has not rebuilt), so every titer store listed
was already stale before this change. No stored record's bytes change either: every
existing `ProductTiterPhenotype` in every store holds a float, so this widens what the
class admits rather than changing what it means.

### Verification, L0 to L4, after the rebuild

`sbatch` job 4458, `build_dataset_lmdb --retire-existing --verify`, dev tree only:

```
BUILT IsoprenolTiterBanerjee2024Dataset: 20 records; gene_set size 15; references 4
```

| level | result |
|---|---|
| L0 structural | 20 records validated |
| L1 count | observed 20, expected 20 |
| L2 value_fidelity / cross_method | 20 values; 20 pairs agree within 0.0 |
| L3 unit rule | `g/L` and `ug/mL`, no source value scaled |
| L3 uncertainty rule | 4 released SDs (fed-batch); 16 typed gaps on both fields |
| L3 medium / plasmid / assembly | 4 media over 6 conditions; one plasmid per strain; all on GCA_000007565.2 |
| L4 stored_titers_vs_the_build_ledger | 20 entities agree within 1e-09 |
| L4 every_quote_is_verbatim_in_its_pinned_mirror | 42 quotes re-read from the three pinned mirrors |
| L4 released_locus_tags_are_reported_not_gated | 10 released tags; gacA of IY1485 carries none, so no typed perturbation and no gene edge |
| L4 perturbed_gene_containment_assembly | 1.000 of 10 measured genes are loci of `pputida_KT2440_ASM756v2` |

`verification PASSED ... 0 genome-reference finding(s) reported, not gated`. The record
count is unchanged at **20**, and `references 4` still holds: the gapped reference is a
reference object like the other three, with its titer absent rather than guessed.

### gacA stays untyped, and that is final

The second decision: the contradiction between the mirrored papers (PP_4099 in Schmidt
2022 and Thompson 2020, PP_1650 in Borchert 2023) is **modeling-time information and is
never resolved in the loader**. IY1485's ΔgacA keeps no typed perturbation, the evidence
stays in `GACA_NO_TAG` and `GENOTYPE_NOT_TYPED`, and the L4 genome-reference level keeps
reporting it as a finding rather than failing the store. Nothing changed here; it is
recorded so the question is not reopened.
