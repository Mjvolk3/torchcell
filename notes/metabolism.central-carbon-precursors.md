---
id: 0mmy9ia10zan2bl6ls8hi0o
title: Central Carbon Precursors
desc: ''
updated: 1786071385404
created: 1786071385404
---

## 2026.08.06 - The precursor/intermediate list, its provenance, and our coverage of it

The canonical list of central-carbon **precursor metabolites** in yeast, why it is that
list, which of our datasets measure each node **and on how many strains**, and what is
missing. Everything here is read off the mirrored paper or computed from the built LMDBs;
nothing is transcribed from memory.

### Provenance of the list

The list is **12 precursor metabolites of central carbon metabolism**. Two nested sources:

- **Domenzain et al. 2025**, *PNAS* 121, doi:10.1073/pnas.2417322122 -- the "103 valuable
  chemicals" ecFactory paper (Jens Nielsen corresponding author). Mirrored at
  `$DATA_ROOT/torchcell-library/domenzainComputationalBiologyPredicts2025/`. The list is
  stated verbatim in `paper.md:68`: *"turnover rates were calculated for the 12 main
  precursor metabolites in central carbon metabolism"*.
- Domenzain attributes it to **reference 11 = J. Nielsen, "Systems biology of metabolism",
  *Annu. Rev. Biochem.* 86, 245-275 (2017)** (confirmed from the PDF reference list, not
  the OCR -- the OCR of the reference block is garbled).

It is the classical Neidhardt / Stephanopoulos-Aristidou-Nielsen set: the metabolites from
which all biomass building blocks are drawn.

### The 12

`s_NNNN` are Yeast9 species ids, sourced from the Zelezniak loader's
`target_metabolite_ids` (built from `YeastGEM`, never invented). Blank = not measured by us,
so no id was resolved through that path.

| # | Metabolite | Abbrev | Pathway node | Yeast9 |
| --: | --- | --- | --- | --- |
| 1 | D-glucose-6-phosphate | G6P | upper glycolysis | `s_0568` |
| 2 | D-fructose-6-phosphate | F6P | upper glycolysis | `s_0557` |
| 3 | ribose-5-phosphate | R5P | PPP (NADPH / nucleotides) | `s_3768` |
| 4 | erythrose-4-phosphate | E4P | PPP -> shikimate | `s_0551` |
| 5 | glyceraldehyde-3-phosphate | GAP | mid glycolysis | `s_0764` |
| 6 | 3-phosphoglycerate | 3PG | lower glycolysis | `s_0260` |
| 7 | phosphoenolpyruvate | PEP | lower glycolysis -> shikimate | `s_1360` |
| 8 | pyruvate | PYR | glycolysis terminus | `s_1399` |
| 9 | acetyl-CoA | AcCoA | lipid / polyketide / terpene entry | `s_0373` |
| 10 | 2-oxoglutarate (alpha-ketoglutarate) | AKG | TCA; amino-acid N donor | `s_0180` |
| 11 | succinyl-CoA | SucCoA | TCA | -- |
| 12 | oxaloacetate | OAA | TCA | `s_1271` |

### Extensions beyond the 12 that our case studies need

The 12 are the *biomass* precursors. Product-specific chassis nodes sit one step off them
and are NOT in Domenzain's list:

| Metabolite | Why we need it | In the 12? | Measured? |
| --- | --- | :--: | :--: |
| **malonyl-CoA** | extender unit for polyketides + fatty acids; the committed step (ACC1) and the real control point for **TAL** | no | **no** |
| mevalonate / FPP | terpene + carotenoid chassis | no | **no** |
| L-valine | -> isobutanol (Ehrlich) | no | yes (Mulleder; Zelezniak n=17) |
| L-tyrosine | -> betaxanthin via L-DOPA | no | yes (Mulleder; Zelezniak n=17) |
| L-phenylalanine | -> 2-phenylethanol, cinnamate, styrene | no | yes (Mulleder; Zelezniak n=17) |
| ornithine / arginine | -> spermidine, putrescine | no | yes (Zelezniak n=17) |

**TAL (triacetic acid lactone)** is worth stating explicitly because it is a live target and
is **absent from Domenzain's 103 chemicals** (grepped: no hit for `triacetic`, `2-pyrone`,
`TAL`, or `malonyl` anywhere in the paper). Its route is 2-pyrone synthase (2-PS,
*Gerbera hybrida*): **1 acetyl-CoA + 2 malonyl-CoA -> TAL + 2 CO2 + 3 CoA**. So its precursor
chain is pyruvate/citrate -> acetyl-CoA -> **malonyl-CoA**, with fatty-acid synthase
(FAS1/FAS2) as the dominant competing malonyl-CoA sink.

### Our measured coverage -- READ THE STRAIN COUNTS

Only **Zelezniak 2018** (`metabolite_zelezniak2018`) measures central-carbon nodes at all.
Its released matrix is **NOT** a 95 x 50 rectangle: a `dataset` column splits it into three
sub-panels with very different strain coverage (96 x 17, 19 x 21, 17 x 22), so per-node n
varies almost 6-fold. Counted directly off the raw file:

| strains | precursor nodes |
| --: | --- |
| 96 | F6P, R5P, GAP, 3PG, PEP, pyruvate |
| 55 | E4P |
| **19** | **G6P, acetyl-CoA, AKG, OAA** |
| 17 | (amino acids) |
| **0** | **succinyl-CoA, malonyl-CoA** |

So "11 of the 12 are measured" is true about **identities** and misleading about **power**.
**Acetyl-CoA -- the node every lipid / polyketide / TAL argument rests on -- has n = 19.**
Same tier: succinate 19, malate 19, citrate 19, fumarate 18.

The build itself is correct and **ragged**, faithfully mirroring the raw design: all 50
distinct metabolite keys are present across the 95 records (including all 20 amino acids),
with 13-50 metabolites per strain. Do not quote the datasets table's
`Zelezniak 2018 (metabolome) | vector (25)` as a panel size -- 25 is the FIRST RECORD's
length, an artifact of `build_supported_datasets_table.py:171` deriving Shape from
`read_first_record` (see [[paper.results-and-discussion.6.experimental-plan]]).

Nearest usable **proxies** for the unmeasured nodes: malonyl-CoA has no direct measurement,
but its dominant sinks do -- **Xue 2025** free-fatty-acid titers (176 strains x 5 species)
and **da Silveira 2014** lipidomics (127 x 135). Those respond to malonyl-CoA supply and are
a better handle than acetyl-CoA at n = 19.

### Open: the WS16 readout says "13 metabolites" and never enumerates them

Both [[plan.cgt-metabolism-flux-layer.2026.07.26]] (line 357) and the metabolism explainer
define the WS16 precursor-pool readout as $\omega_i(v)$ at **13 metabolites**, log
fold-change vs wild type, "matching Domenzain's reported quantity". Domenzain reports **12**.
The 13th is not named in either note. Resolve before WS16 is scored: either it is
malonyl-CoA (sensible -- it is the one addition our case studies demand) or the count is a
typo for 12. **Not yet determined; do not assume.**

### Direction of change is relative to a GROWING cell

Domenzain's precursor figures are fold-changes of turnover
$\omega_i = \tfrac12\sum_j |S_{ij}v_j|$ between two FBA solutions -- **optimal production
divided by optimal biomass**. `FC > 1` means more flux through that node than a
growth-optimizing cell sends there; `FC < 1` means less. Several nodes go DOWN for
production (notably R5P and AKG) simply because biomass formation falls, so demand for
nucleotides and amino acids falls with it -- not because the product needs less carbon.
R5P is `down` for clusters 1-2 (nucleotide demand) and `up` for cluster 3 terpenes (NADPH
supply): same node, two roles.

Related: [[paper.results-and-discussion.6.experimental-plan]] ·
[[plan.cgt-metabolism-flux-layer.2026.07.26]] ·
[[torchcell.datasets.scerevisiae.zelezniak2018]] · [[metabolism.beta-carotene-and-betaxanthin]]
