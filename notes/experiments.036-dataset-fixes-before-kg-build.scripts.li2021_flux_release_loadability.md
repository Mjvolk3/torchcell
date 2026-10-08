---
id: ze9wzj3nygqc3222ys9wio8
title: Li2021_flux_release_loadability
desc: ''
updated: 1791453208385
created: 1791453208385
---

## 2026.10.08 - Settling schedule row 50, Li 2021 mevalonate flux

Script: `experiments/036-dataset-fixes-before-kg-build/scripts/li2021_flux_release_loadability.py`
Results: `results/li2021_flux_release_loadability.json`,
`results/li2021_flux_release_loadability_reactions.csv`

Measured on the sha256-pinned mirror: `si/si1.docx`
(`14250d74f8e7bb60335635d51d406e3da6756df185228a6b861af896fa2f5094`) and `si/si2.xlsx`
(`24c836f8459e5cce4f6a322477a7554e1f7dd16e8bb6b7cdd0ac4890b24c585a`, Additional file 2,
the flux maps). Quotes are verbatim from the pinned `paper.md`
(`7d174c8ef7af0e0c275f6194271d39346c22411c27904803ee16b7caf95b7773`).

### The released quantity is a model FIT, and the paper calls it a simulation

One sentence in Methods is the whole account of how the numbers were produced:

> Metabolic fluxes were estimated by minimizing the residual sum of squares between experimentally measured and model predicted 13C-enrichment using 13C-Flux software obtained from Dr. Wiechert [33].

Reference [33] is Weitzel 2013, so the tool is 13CFLUX2, named only through its citation.
No version, no solver, no weighting scheme, no convergence criterion. The sheet agrees
with the word "fit": its own column headers are `best fit`, `LB90`, `UB90`, and its
diagnostic rows are `Number of fitted measurements :`, `Freedom of flux`, `χ2 90%` and
`SSR :`.

**The interval's procedure is never stated.** The strings `confidence`, `interval`,
`Monte Carlo`, `degrees of freedom` and `SSR` appear nowhere in the paper. The column
headers and the `χ2 90%` row state the LEVEL, 90 percent, so that much is sourced; what
produced the bounds is not.

The paper's only description of the file, in its supplementary list, is:

> Additional file 2. MFA simulated result.

So the one line a reader would look at calls the released fluxes a simulated result, while
Methods describes a least-squares fit. That wording conflict is the sharpest provenance
risk in the paper and has to be recorded with the data, not smoothed over.

What was actually measured, and all of it is in the same file, which is what makes the fit
reproducible:

> 13C-MFA was performed using 100% 1-13C1 glucose as the feeding substrate was added to a concentration of 10 g/L.
>
> The resulting proteinogenic acids were derivatized with N-(tert-butyldimethylsilyl)-N-methyl-trifluoroacetamide containing tert-butyldimethylchlorosilane in acetonitrile at 105 C for 1 h, and then analyzed by a GC-MS [Agilent 7890 A GC and 5975 C Mass Selective Detector (Agilent Technologies, Santa Clara, USA)] equipped with a DB-1column (Agilent Technologies).
>
> The data obtained from GC-MS were corrected by reduction of the natural abundance ratio of C, H, O, N, and Si isotopes [30].

`LABEL_MEASUREMENTS` holds 134 cumomer-constraint rows per strain over 13 amino acids
(ALA, GLY, VAL, LEU, ILE, SER, THR, ASX, GLX, LYS, TYR, HIS, MET), `LABEL_INPUT` holds the
tracer isotopomer distribution, and `Biomass Composition` holds the biomass fractions and
the byproduct yields. Columns G to K of `LABEL_MEASUREMENTS` are entirely empty: the
fitting inputs carry no standard deviation, standard error or weight at all, so the
least-squares objective was unweighted or weighted by an undisclosed constant. No
replicate count appears anywhere in the paper.

### The fitted-flux question, argued both ways

A fitted flux **is** honest to store in an experimental-data ontology, under conditions,
and this schema has already made that decision.

The case against storing it is real. A 13C-MFA net flux is a parameter identified from
data under a declared model, not an observation. Change the reaction network or the
biomass composition and the number moves with no change in any measurement, so it is not a
property of the cell alone. Shelving it beside measured phenotypes invites a consumer to
read it as one, which is exactly the conflation this project's evidence discipline exists
to prevent.

The case for is stronger, on three grounds. First, there is no measured alternative: the
observable is the labeling pattern, and no instrument reads a flux through
phosphoglucose isomerase. Excluding fitted fluxes removes intracellular flux from the
ontology permanently, and intracellular flux is the quantity the metabolic side of this
project is built to predict. Second, the fit is a deterministic function of released
inputs. For this paper the tracer distribution, the labeling measurements, the network
with its atom transitions, the biomass composition and the byproduct yields are all in the
one file, so "where did this come from and how was it done" is answerable from the stored
artifact, which is the provenance standard rather than an exception to it. Third,
`FluxPhenotype` already encodes the distinction in its own docstring, verbatim:

> A flux map is not a measurement of one reaction at a time: it is a FIT of a whole reaction network to labeling data, so the interval is a property of that fit and is stored as an interval rather than as a scalar statistic.

So the resolution is a condition, not a prohibition. A fitted value may be stored when the
record says it is a fit, names the fitting procedure and its inputs, and keeps the interval
an interval. What would be dishonest is the opposite: storing it unlabeled, or promoting
one of its two bounds to "the" statistic, which is the failure `label_statistic_name = None`
exists to block. On that test Li 2021 passes on the inputs and on the interval and
PARTIALLY fails on the procedure, since the bound-generating method is unstated; that is a
recordable gap, not a disqualification.

**Li 2021 is therefore refused on its release, not on its fitted-ness.** That distinction
matters for the schedule: the next 13C-MFA row whose control map is released is loadable
today with no schema change.

### Can an existing class carry it? Yes, entirely, and nothing is missing

`FluxPhenotype`, `FluxExperiment` and `FluxExperimentReference` all exist, are exported
from `torchcell.datamodels`, and have **no consumer**. The `flux phenotype` graph class is
already declared in `biocypher/config/torchcell_schema_config.yaml` and pinned by
`tests/torchcell/adapters/test_bacterial_graph_classes.py`, and `CellAdapter` already
carries `_flux_properties`, `_flux_phenotype_node` and
`_get_flux_phenotype_reference_nodes`. A flux dataset needs no schema change and no new
graph class.

The script builds a real `FluxPhenotype` from the BW-P08 column and it takes the released
data whole: 99 reactions, bounds that equal the best fit on the pinned rows, the signed
negative flux (`vGPI = -0.445077375`) unclamped, `confidence_level = 0.90`, and
`label_statistic_name` None by the class's own design. The four fit diagnostics have no
field and would live in the loader's ledger.

### Why no record can be built anyway

**The reference is missing.** Every `Experiment` is stored with an `ExperimentReference`
whose `phenotype_reference` is a phenotype of the same family, and `FluxPhenotype.net_flux`
must be non-empty (measured: an empty map raises `net_flux cannot be empty`). For an
absolute readout the reference is the parent strain's own measured value, as the landed
Fuhrer 2017 and Mori 2021 loaders both do, and an absolute flux has no definitional zero
the way a log fold change does. The text says the control was analyzed:

> 13C-MFA was performed to detect the metabolic flux distribution in strains BW-P08 BF and BW-P10 BF (which had high MVA titers) and the control strain BW-P BF.

and Fig. 5 prints three rows for those three strains:

> Fig. 5 Metabolic flux diagram of zwf-strengthened EP-bifido strains. The metabolic flux shown for strains from top to bottom are BW-P10 BF, BW-P08 BF, BW-P BF

but Additional file 2 holds maps named BW-P08, BW-P04 and BW-P10. The control's map is
absent and BW-P04's is present without ever being mentioned. There is nothing to put in
the reference, which blocks every record including the one strain whose column is clean.

**One of the three maps is mislabeled by its own content.** The sheets name strains three
ways, and the script keys each `NETWORK` block to a `Biomass Composition` column
independently of the labels, by converting that column's released byproduct yields into
molar ratios against glucose and comparing them with the block's own output fluxes:

| NETWORK block | best match | score | runner-up margin | label implies | agrees |
|---|---|---|---|---|---|
| BW-P08 | PB08 | 0.00034 | 0.396 | PB08 | yes |
| BW-P04 | PB04 | 0.00133 | 0.037 | PB04 | yes |
| BW-P10 | PB04 | 0.00133 | 0.037 | PB10 | **no** |

The BW-P10 block carries BW-P04's measured extracellular constraints. Confirmed a second
way: 50 of 99 reaction rows are bit-identical between the BW-P04 and BW-P10 blocks and
their glucose uptake fluxes are exactly equal, while BW-P08 shares 2 rows with BW-P04 and
0 with BW-P10. Two independent fits of two strains cannot share half their rows. So which
strain the third column describes is unresolved, and it cannot be written as BW-P10 BF.

### Other measurements worth keeping

- **The reaction count is 99, not the 198 the schedule row states.** Column C holds 272
  non-blank cells = 1 header + 99 reaction equations + 99 atom-transition lines. 198 is
  the reaction rows plus their atom-transition rows, which double-counts every reaction.
- **Roughly half of each map is pinned, not fitted**: `LB90 == best fit == UB90` on 46 of
  99 rows for BW-P08, 54 for BW-P04 and 50 for BW-P10. Those are the measured rates and
  the biomass drains. The three fits do not pin the same reactions. The same count run on
  the normalized columns is lower (27 / 39 / 34) only because dividing by the glucose
  uptake reintroduces floating-point spread, so the absolute columns are the ones to test.
- **The `χ2 90%` cell disagrees across strains**, 102.8 for BW-P08 against 102.08 for the
  other two, although all three share 85 degrees of freedom and so must share one
  quantile. No sentence in the paper interprets SSR or this quantile at all.
- **There are exactly three blocks.** Columns A, J, Q and X to AB of `NETWORK` are
  entirely empty; there is no fourth strain.
- The labeling culture's medium, temperature, vessel, steady-state status and sampling
  time are all absent. The only stated condition is 10 g/L [1-13C]glucose and a harvest at
  exponential phase, and the one medium the paper describes is the 20 g/L glucose
  production medium, so it is demonstrably a different condition.
- The measured extracellular rates exist only as unitless yields in a sheet the paper
  never references, and no growth rate is reported anywhere.
