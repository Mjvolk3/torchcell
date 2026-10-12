---
id: imvyr5tes9720l1523vkeom
title: Wave8
desc: ''
updated: 1789244775424
created: 1789244775424
---

## 2026.09.12 - Block 1: the verified reaction-energy tables

### What the two routes are

Wave 7's cells priced their reaction energies with `--thermo-source union_external`: the
merge of the shipped metabolite table with our eQuilibrator recomputation, the second side
carrying the written-structure lipids and the external-pKa targets. `verified` reads the
curated tables under `thermo_equilibrator/verified/` instead, pinned by their manifest
sha256, and takes the REACTION values from the verified reaction table rather than
re-summing them over S, because three of the routes behind it are reaction-level facts a
stoichiometric sum cannot reproduce: participants that cancel, a conserved moiety priced
from a group difference, and the shipped reaction column. It is also the only route in
which the accession composition guard has run.

### The table diff, before any cell finished

`compare_thermo_sources.py` (writes `results/thermo_source_diff_union_external_vs_verified.json`).

| quantity | union_external | verified |
|---|---:|---:|
| reactions priced, of 4,131 | 3,779 | 3,801 |
| priced by this route only | 10 | 32 |
| reactions with sigma | 1,755 | 1,751 |

**Exactly five reactions carry a different value, and they are the mispricings the guard
removed:**

| reaction | union_external, kJ/mol | verified, kJ/mol | change |
|---|---:|---:|---:|
| r_0464 | 1061.49 | -7.12 | -1068.60 |
| r_0463 | 875.87 | -7.12 | -882.98 |
| r_0510 | -904.55 | -21.57 | +882.98 |
| r_0511 | 892.52 | 9.53 | -882.98 |
| r_1174 | -243.63 | -58.01 | +185.62 |

A standard reaction energy of 1,061 kJ/mol is not a reaction energy, it is a composition
error, and the verified route replaces all five with values of order 10 kJ/mol.

**None of them touches an amino acid.** Of the 269 yeast-GEM reactions that involve one of
the 19 measured amino acids in any compartment, **0 differ between the two routes**, and 0
of the betaxanthin-precursor reactions differ. So the prediction, recorded here BEFORE the
paired cells finish, is that the amino-acid score will move by seed noise rather than by
the table: any difference has to travel indirectly, through the five repriced reactions
changing the flux the concentration term is coupled to. If the paired difference comes out
larger than the between-seed spread of about 0.01, that indirect route is the thing to
explain, and it would be a finding rather than a confirmation.

The provenance mix also changes shape, which is worth recording because it is what the
verified route buys: union_external prices 3,777 reactions from the eQuilibrator sum and 2
from the shipped reaction column, while verified prices 2,751 from cache accessions, 872
mixed, 122 from cache structures, 19 from cancelled participants, 11 from a literature
table, 9 external-pKa, 9 shipped-table, 2 curated structures, 2 composition surrogates, 2
repeat-unit surrogates and 1 conserved moiety, with 16 curation records attached.

### Cells

`gh_wave8_aa_concver_e200_s270{0,1,2}.slurm`: the wave-7 concentration cell verbatim
(`flux_projection_anchored`, constraint scale 0.1 over a 20-epoch warmup, whole-cell
mixture readout, 1 M window, CRPS, 200 epochs, kcat 97 / 782 / 2,831 / 18) with
`--thermo-source verified`, paired seed for seed against `wave7_aa_conc_e200_s<seed>.json`.
Job 1780 is seed 2700; the other two follow as the 2-cell budget frees.

## 2026.09.12 - Block 2: the fractional volumes are now measured, not assumed

### The source

Wave 7 weighted the whole-cell average `[A]_total = sum_n f_n [A]_n` with a table in which
five of six entries were textbook assumptions, and said so. The measurement exists: Uchida
et al. 2011 imaged *S. cerevisiae* by soft X-ray tomography, which segments organelles by
their linear absorption coefficient rather than by a fluorescent label, and therefore
measures the irregular organelles (vacuoles, mitochondria) that fluorescence cannot.

Retrieved to the raw mirror, never to Zotero:
`$DATA_ROOT/torchcell-raw/uchidaQuantitativeAnalysisYeast2011/paper/pmc_fulltext.xml`,
sha256 `4ad4cee88e942feffffd4c372b8b085f59ea70a303331a8173c9ccf260477aa2`, retrieved from
the PMC author manuscript PMC3404734 (NIHMS253192, the open version of Yeast 28(3):227-236,
doi 10.1002/yea.1834) with

```bash
curl -s 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pmc&id=PMC3404734&retmode=xml' \
    -o paper/pmc_fulltext.xml
```

recorded in that key's `manifest.json` with method `pmc_oa_api`. The verbatim sentence the
numbers come from:

> On average, the nucleus occupied 6-11% of cell volume; the nucleolus occupied 20% of
> nucleus; vacuole(s) occupied 3-14% of cell; and mitochondria and lipid bodies occupied
> 1-2% of cell ( Figure 3 ); therefore, the cytosol, other non-segmented organelles, large
> macromolecular complexes, such as the ribosomes, and the cytoskeleton occupy the
> remaining 70% of cell volume.

### The table, and what is still not sourced

`torchcell/metabolism/parameters.py`, `yeast_compartment_volume_fractions()`, pydantic with
the quote and sha256 attached PER ENTRY rather than to the table.

| compartment | assumed (wave 7) | sourced (wave 8) | reported range | sourced |
|---|---:|---:|---|---|
| cytoplasm | 0.65 | 0.70 | the paper's remainder | yes, as an upper bound |
| vacuole | 0.18 | 0.085 | 3 to 14 percent | yes |
| nucleus | 0.10 | 0.085 | 6 to 11 percent | yes |
| mitochondrion | 0.05 | 0.015 | 1 to 2 percent, jointly with lipid bodies | yes |
| endoplasmic reticulum | 0.015 | 0.015 | not resolved | **no** |
| peroxisome | 0.005 | 0.005 | not resolved | **no** |

Each range is taken at its MIDPOINT and the interval is kept in the record. The
conservative-end rule in CLAUDE.md exists to avoid overstating precision on an uncertainty;
here there is no companion statistic to back-solve from, and taking the low end of every
range would drop the weights far below 1 and silently move that mass into the unmodeled
remainder, so the midpoint is the honest central estimate and the range is published beside
it. The cytosolic 0.70 is the paper's own remainder and is an upper bound: the same 70
percent also holds the cytoskeleton, the ribosomes and every organelle the study did not
segment. The ER and the peroxisome are inside that remainder and are NOT resolved by it, so
they keep their assumed values and their `sourced: false` flag rather than being invented
from the leftover.

**The vacuole is the entry that moves, 0.18 to 0.085, and it is the one that matters**:
arginine, lysine and histidine are largely vacuolar, so their whole-cell average is the
most sensitive to this weight of any of the 19.

### Cells

`--volume-table {assumed,sourced}` on `train_bx.py`, default `assumed` so every wave-7 cell
deserializes and reruns unchanged. `gh_wave8_aa_concvol_e200_s270{0,1,2}.slurm` is the
wave-7 concentration cell with `--volume-table sourced` and nothing else changed, paired
seed for seed against `wave7_aa_conc_e200_s<seed>.json`. Queued behind block 1.
