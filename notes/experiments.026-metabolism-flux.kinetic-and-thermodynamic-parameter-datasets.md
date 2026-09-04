---
id: cuwsuihdh54ozcj4pbubrcn
title: Kinetic and Thermodynamic Parameter Datasets
desc: ''
updated: 1788492498003
created: 1788492498003
---

## 2026.09.03 - Model mirrors, a kcat dataset, and eQuilibrator thermodynamics

Three things were built: a standard place for third-party predictor weights with
provenance, one dataset module per kinetic predictor on the node-embedding pattern, and a
recomputation of every yeast-GEM formation energy from eQuilibrator so the thermodynamics
carry uncertainty and transfer to another organism.

### Where models live now

`$DATA_ROOT/models/kinetics/<name>/`, beside the existing `mineru` mirror, each with a
`manifest.json` pinning every file by sha256 and recording the source URL, the retrieval
command, and the git commit. `$DATA_ROOT/models/...` holds the model;
`$DATA_ROOT/data/torchcell/...` holds what the model produced. A build carries the weight
hash that produced it, so a value is attributable to particular bytes.

Generating script: `experiments/026-metabolism-flux/scripts/mirror_kinetic_models.py`.
Pydantic records: `torchcell/data/model_mirror.py`.

| predictor | emits | files | runnable | blocker |
| --- | --- | --- | --- | --- |
| dlkcat | k_cat | 224 | yes | none |
| unikp | k_cat, K_M | 34 | yes | needs a pinned scikit-learn 1.2.2 environment |
| turnup | k_cat | 51 | weights present, not yet run | reaction-SMILES input not assembled |
| eitlem | k_cat, K_M | 45 | no | weights are a 12.3 GB Zenodo archive, not mirrored |
| boost_km | K_M | many | no | ships the UniRep featurizer, NOT a fitted model |
| deepenzyme | k_cat | 969 | no | needs NVIDIA apex plus pinned old numpy |

Two corrections to the earlier readiness audit, both from reading files rather than
filenames. DLKcat **does** ship its trained model, as an extensionless file under
`Results/output/`, so the 18 GB Zenodo download it was queued for was never needed.
Boost_KM does **not** ship a fitted model: its 18 weight files are the UniRep 1900 mLSTM
featurizer, so a K_M prediction there requires refitting from the shipped BRENDA data.

### DLKcat, validated then run

The implementation reproduces the authors' three released example values to four decimal
places. That check caught two defects that would otherwise have been invisible:

- **The output is log2, not log10.** The authors exponentiate with `math.pow(2, ...)`.
  Reading it as log10 inflates every turnover number by a power of about 3.3 and still
  looks like a plausible k_cat.
- **Multi-component SMILES are refused by the model itself.** Its fingerprint pass indexes
  a node list built only from bonded atoms, so a disconnected fragment, such as the lone
  `[Fe+2]` in the heme SMILES yeast-GEM gives for ferricytochrome c, walks off the end.

Result: 7,351 predicted k_cat values over 3,552 of 3,728 catalytic units, 95.3 percent,
median 4.01 1/s, p05 0.22, p95 74.6. The 105 failures are all multi-component SMILES.

This is the number that matters against the previous state: measured k_cat covered 148
units, 4.0 percent, and the other 96 percent ran on a single flat default of 10.3 1/s.

### eQuilibrator thermodynamics, and a validated correctness check

`build_equilibrator_thermo.py` resolves each metabolite by accession, MetaNetX then ChEBI
then KEGG then BiGG, with an InChI fallback, and computes a transformed formation energy
plus the covariance factor from component contribution.

**The table is correct.** Summing S times dfG'o over it reproduces eQuilibrator's own
reaction call to 1e-13 kJ/mol across 40 balanced single-compartment reactions.

| quantity | shipped CSVs | eQuilibrator here |
| --- | --- | --- |
| metabolites covered | 2,389 of 2,806, 85.1 % | 2,099 of 2,806, 74.8 % |
| reactions fully covered | 3,210 of 4,131, 77.7 % | 1,755 of 4,131, 42.5 % |
| uncertainty | none | 785 reactions with real sigma, median 2.30 kJ/mol |
| covariance factor Q | absent | present, rank 590 |
| transfers to another organism | no | yes, structure is enough |

The coverage regression is real and is the price of the uncertainty.

**Correction to the agreement number.** Pooling all 1,391 overlapping reactions gives a
median absolute difference of 5.60 kJ/mol, and that number is too flattering. 840 of them
are transport, which at a uniform pH is exactly zero here by construction because the same
species has the same formation energy on both sides of the membrane. On the 551 chemical
reactions, where the two methods make comparable claims, the median absolute difference is
**10.11 kJ/mol**. That is still within the shipped table's own internal inconsistency of
9.53 kJ/mol, so the conclusion holds, but the honest number is 10.11 rather than 5.60.

The 888 estimable reactions with sigma near zero are transports, where the same compound
appears on both sides and its uncertainty cancels exactly. That is correct rather than a
gap.

### Compartment pH is implemented and NOT trusted

Applying per-compartment pH changes 1,309 of 1,755 reactions by a median 42 kJ/mol, up to
1,062 kJ/mol. Shifts of that size are not physically plausible for single reactions, so
the uniform pH 7.0 build is the trustworthy artifact and the compartment build is
experimental. The pH table itself is textbook organellar values, labeled `assumed`, not
sourced from a mirrored paper. Diagnosing the large shifts is the open item.

### The betaxanthin pathway, and exactly where generalization stops

Of five cassette intermediates, all five match their declared molecular formula against a
PubChem structure lookup, and two resolve to a formation energy:

| compound | dfG'o kJ/mol | sigma | status |
| --- | --- | --- | --- |
| L-DOPA | -157.68 | 4.30 | resolved from structure, no identifier needed |
| dopaquinone | -168.31 | 7.54 | resolved from structure |
| cyclo-DOPA | -- | -- | absent from the cache |
| betanidin | -- | -- | absent from the cache |
| betalamic acid | -- | -- | absent from the cache |

**The blocker for the other three is a commercial license, not a method.** Adding a novel
compound runs a group decomposition that needs pKa estimates, and eQuilibrator sources
those from ChemAxon `cxcalc`. Without it `equilibrator_assets` loads read-only and refuses
to create compounds. So the generalization claim holds for structures the cache already
knows and fails for genuinely new ones, and closing it needs a pKa route that does not go
through ChemAxon.

### UniKP, and the first K_M dataset

UniKP runs in an environment pinned to scikit-learn 1.2.2, invoked as a subprocess with
parquet as the interface, because its released regressors were pickled before 1.3 added
`missing_go_to_left` to the tree node dtype. Its three fitted regressors are not in the
repository at all; they are released separately on HuggingFace and are now mirrored and
pinned alongside it.

| parameter | rows | units | coverage | median | p05 | p95 |
| --- | --- | --- | --- | --- | --- | --- |
| k_cat | 7,351 | 3,552 | 95.3 % | 4.05 1/s | 0.77 | 22.2 |
| K_M | 7,351 | 3,552 | 95.3 % | 0.080 mM | 0.026 | 1.24 |

The K_M table is the first of any kind: the parameter previously existed as an enum member
and a protocol flag with nothing behind it.

### The two predictors agree on the median and disagree per pair

Generating script: `compare_kinetic_predictors.py`.

DLKcat and UniKP land on nearly the same median k_cat, 4.01 against 4.05 1/s, and that
agreement is superficial. Across all 7,351 pairs their per-pair correlation is
**pearson 0.337 on log10, spearman 0.224, with a median absolute disagreement of 0.46
decades**, a factor of about three. Since almost none of these pairs has a measured value,
that spread is the honest uncertainty on any individual predicted turnover number, and it
is much wider than either model's own reported error.

Against measured values, on 19 substrate-matched pairs from the Open Enzyme Database:

| comparison | n | pearson (log10) | spearman | median abs error |
| --- | --- | --- | --- | --- |
| DLKcat k_cat | 19 | 0.734 | 0.785 | 0.31 decades |
| UniKP k_cat | 19 | 0.982 | 0.940 | 0.27 decades |
| UniKP K_M | 19 | 0.619 | 0.663 | 1.00 decades |

**This is not a generalization estimate and must not be read as one.** The Open Enzyme
Database aggregates BRENDA and Sabio-RK, which is what both models trained on, so these
pairs are almost certainly inside both training sets. UniKP's 0.982 is what memorization
looks like. The join is on the (enzyme, substrate) pair with canonicalized SMILES; joining
on enzyme alone compares a prediction for one substrate against a measurement for another,
and doing that drops the same UniKP correlation to 0.822 and the K_M correlation to 0.119.

The K_M row is the one worth attention: **even under those favorable conditions UniKP's
K_M is off by a median factor of ten.** Hypothesis (untested): that is why the saturation
term was left out of the flux layer in the first place, and it argues for treating K_M as
an order-of-magnitude prior rather than a measured constant.

### Protein sequences come from the genome object

Sequences are read from `SCerevisiaeGenome` through `genome[orf].protein.seq`, the same
path `Esm2Dataset` uses, not from the `swissprot.tsv` yeast-GEM bundles. The point is the
derivation chain: everything traces to the S288C reference FASTA plus the GAF, so the
protein a predictor sees is the one the perturbation ontology edits, rather than a claim
by a third party's table compiler.

**All 1,161 GEM genes resolve.** Every one comes back `current` from
`genome.resolve_gene_name`, and every one has a protein sequence, so nothing is dropped by
switching source. Rebuilding DLKcat on genome sequences moved the median k_cat from 4.0125
to 4.0116 1/s, which says the two sources agree on content and differ on provenance.

Two failures worth recording, both from the switch:

- **`orf in genome` is a trap.** The genome defines no `__contains__`, so Python falls
  back to integer indexing and calls `genome[0]`, which raises `FeatureNotFoundError`
  before the real lookup happens. Index directly.
- **`go_root` is a separate argument** defaulting to a relative `data/go`. Left unset it
  misses and the constructor tries to fetch `go.obo` from `current.geneontology.org`,
  which answers 403.

### Embedding is sharded over the four GPUs

ProtT5-XL is a 3B-parameter encoder, so the distinct sequences are split by index across
`cuda:0..3` and merged. Within a shard, sequences are sorted by length and batched, which
is where most of the gain comes from; the per-sequence masked mean keeps the result
identical to unbatched inference. 1,154 distinct sequences finish in under 40 seconds.

The genome is read exactly once, by `--dump-sequences`, and the shards read that dump.
Four processes opening the gffutils SQLite database concurrently raced and made it
briefly unreadable, and the shards need the genome for nothing but a sequence lookup, so
the dependency was removed rather than serialized.

### DeepEnzyme ran, and the audit was wrong about it too

Listed as blocked by NVIDIA apex plus pinned old numpy/rdkit with no weights shipped. All
three were checked and two were false. **apex is listed in requirements.txt and imported
nowhere in the code**, so it was a training-time mixed-precision dependency, and the
network runs unmodified on current torch, numpy and rdkit with no separate environment.
The weights are real but live on figshare rather than in the repository, pointed at by a
comment in the authors' own example script; md5 matched figshare's recorded value, and all
137 checkpoint tensors match the architecture with none missing or unexpected.

Its output is **log2**, the same convention as DLKcat, established three ways: the
example script exponentiates base 2, commented reporting lines in the training code only
make sense if the target is log2, and the released training targets have a p95 that is
600 1/s read as log2 and 1.7e9 1/s read as log10.

Result: 7,300 rows over 3,546 of 3,728 units (95.1 percent), median 4.167 1/s, p05 0.989,
p95 23.89. Against DLKcat on the 7,300 shared pairs, Spearman 0.284.

### A silent failure in DLKcat that DeepEnzyme exposes

DeepEnzyme refuses 51 rows with `unseen_atom_type`. All of them are substrates whose SMILES
contains a wildcard `*`, the placeholder yeast-GEM uses for macromolecules: tRNA species,
an apoprotein, a phosphatidylethanolamine backbone, a histone lysine. DeepEnzyme's atom
vocabulary has no wildcard, so it declines to predict.

**DLKcat predicts on all 51 anyway**, because its featurizer maps an unknown atom to index
0, which is carbon. Those rows carry ordinary-looking turnover numbers, median 4.12 1/s,
for molecules the model cannot represent. This is the same mechanism as the unseen-token
counter already added to the DLKcat module, and it is why that counter exists: the fraction
of a molecule's tokens that were never in training is the only confidence signal DLKcat
exposes, and it is not exposed by default.

A downstream consequence: DLKcat reports 7,351 rows and DeepEnzyme 7,300, and the
difference is not coverage. It is 51 predictions that should not have been made.

### Wu et al.'s Fig 3 per-pair predictions were never released

Their notebook `Yeast-MetaTwin-05.Fig3abcde.ipynb` reads five tables from
`Results/kcat_km_predict/`, named explicitly: `yeast8U_kcat_predict.csv` (DLKcat),
`yeast8U_unikp.csv` (UniKP), `yeast8U_kcat_result_TurNuP.csv` (TurNuP),
`yeast8U_km_km_predict.csv` (Boost_KM). That directory exists in neither released artifact.

- The GitHub repo ships `Results/` with nine subdirectories and none is `kcat_km_predict`.
  Its `source_data/` holds only fig2-a, fig3-g, fig3-j, fig4-b, fig5-b, figs11, figS8, and
  the file called `fig3-g.csv` is EC-number counts, not a Km distribution.
- The 6.6 GB Zenodo archive (10.5281/zenodo.13911783, verified byte-exact at
  6,636,083,053) contains exactly three top-level directories: `Data`,
  `Data_retrosynthesis`, `esm`. Of its 1,021,022 entries, **zero** match kcat, unikp,
  turnup, eitlem or deepenzyme.

The Data Availability statement says data for reproducing all figures is on GitHub and
Zenodo. For Fig 3a-h it is not. Stated as a fact about the release, not a criticism.

Consequence: the underground arm cannot be drawn without rebuilding their
retrobiosynthesis pipeline. What we can do, and did, is the stronger half of a
reproduction: re-run the same published models over the same organism from hash-pinned
mirrors rather than re-plot their numbers.

### All six predictors, run

| predictor / parameter | rows | units | coverage | median | p05 | p95 |
| --- | --- | --- | --- | --- | --- | --- |
| dlkcat / k_cat | 7,351 | 3,552 | 95.3 % | 4.012 1/s | 0.225 | 74.6 |
| unikp / k_cat | 7,351 | 3,552 | 95.3 % | 4.050 1/s | 0.772 | 22.2 |
| eitlem / k_cat | 7,456 | 3,552 | 95.3 % | 3.981 1/s | 0.071 | 108.9 |
| turnup / k_cat | 5,143 | 2,056 | 55.2 % | 10.819 1/s | 1.595 | 64.6 |
| deepenzyme / k_cat | 7,300 | 3,546 | 95.1 % | 4.167 1/s | 0.989 | 23.9 |
| unikp / K_M | 7,351 | 3,552 | 95.3 % | 0.0796 mM | 0.026 | 1.24 |
| eitlem / K_M | 7,456 | 3,552 | 95.3 % | 0.0793 mM | 0.006 | 3.23 |
| boost_km / K_M | still running (refit from BRENDA) | | | | | |

Every one of the five that produced numbers was validated against something the authors
themselves published, and three of those checks caught or ruled out a real defect.

- **DLKcat** reproduces its three released examples to four decimals. Base **log2**.
- **DeepEnzyme** matched all 137 checkpoint tensors with none missing or unexpected, and
  its base was established three ways. Base **log2**. Its apex requirement turned out to
  be a training-time dependency imported nowhere in the code, so no patch was needed.
- **EITLEM** reproduces the README's worked example at **1.3904 against a stated 1.39**,
  which log2 could not produce. Base **log10**.
- **TurNuP** reproduces the authors' tutorial value at **216.114853 1/s** against their
  216.114853, a relative difference of 4.4e-10. Base **log10**.

**The log base is not a convention you can assume**: two of these models use base 2 and two
use base 10, and reading either one wrong yields plausible turnover numbers.

TurNuP's low coverage is structural rather than a failure: it consumes the whole reaction,
and the authors' code invalidates a reaction as soon as any single participant lacks a
structure. 1,147 of 2,540 reactions have at least one such metabolite, almost all
acyl-chain-resolved lipids.

A trap TurNuP nearly walked into and did not: the deployed booster was fit on
`[difference_fp, ESM1b_ts BOS-token]`, but the repository's older comparison notebook feeds
plain mean-pooled ESM-1b into the same booster. That path runs without error and returns
numbers that are not TurNuP's.

### The predictors do not agree with each other

Generating script: `compare_kinetic_predictors.py`. On identical enzyme-substrate pairs,
log10 Pearson and Spearman:

| pair | n | pearson | spearman | median disagreement |
| --- | --- | --- | --- | --- |
| eitlem vs unikp | 7,351 | 0.560 | 0.509 | 0.51 decades |
| turnup vs unikp | 5,038 | 0.458 | 0.433 | 0.44 |
| eitlem vs turnup | 5,143 | 0.361 | 0.330 | 0.64 |
| dlkcat vs unikp | 7,351 | 0.337 | 0.224 | 0.46 |
| dlkcat vs eitlem | 7,351 | 0.248 | 0.169 | 0.70 |
| dlkcat vs turnup | 5,038 | 0.142 | 0.084 | 0.56 |
| deepenzyme vs turnup | 5,035 | 0.111 | 0.094 | 0.51 |
| deepenzyme vs unikp | 7,300 | 0.109 | 0.015 | 0.40 |
| deepenzyme vs dlkcat | 7,300 | 0.270 | 0.284 | 0.40 |
| deepenzyme vs eitlem | 7,300 | 0.090 | 0.025 | 0.68 |
| **K_M**: eitlem vs unikp | 7,351 | 0.463 | 0.406 | 0.50 |

Five predictors agree on the median k_cat to within about 4.0 to 4.2 1/s, TurNuP excepted,
and their **per-pair rank correlation ranges from 0.015 to 0.509**. The typical
disagreement is 0.4 to 0.7 decades, a factor of three to five. Since almost none of these
pairs has a measured value, that spread is the honest uncertainty on any single predicted
turnover number, and it is far wider than any published accuracy figure implies.

Against measured values, on 19 substrate-matched Open Enzyme Database pairs: UniKP 0.982,
EITLEM 0.921, DLKcat 0.734, TurNuP 0.482, DeepEnzyme 0.203. **This is memorization, not
generalization** -- the OED aggregates BRENDA and Sabio-RK, which these models trained on.
The ordering here does not predict the model-versus-model ordering, which is the point.

### Still open

- UniKP K_M over all pairs, which is the first K_M dataset of any kind.
- TurNuP needs reaction SMILES assembled rather than substrate SMILES.
- EITLEM weights, 12.3 GB.
- Boost_KM refit from BRENDA.
- The compartment-pH shifts.

## 2026.09.04 - Boost_KM runs, and the per-gene comparison

### Boost_KM was never blocked on missing weights

The earlier claim that Boost_KM "ships a featurizer rather than a fitted model, so a K_M
prediction requires refitting" is **wrong** and is corrected in section 1 of the notes-tex
document. `AlexanderKroll/KM_prediction` ships `xgboost_model.dat` and
`xgboost_model_full.dat` alongside the UniRep featurizer, and its README says the pretrained
weights can be loaded rather than retrained. Reading the 18 UniRep `.npy` files and stopping
there is what produced the wrong conclusion.

The real obstacle was speed. UniRep is a TensorFlow 1 mLSTM stepping one residue at a time
on CPU; a first attempt occupied roughly 85 cores and had to be killed.

**Wu et al. ran a different model than they cite.** Their Methods give
`AlexanderKroll/KM_prediction`, but `Yeast-MetaTwin-04.KMprediction.ipynb` vendors
`KM_prediction_function`, whose enzyme representation is ESM-1b (1,280 dims) rather than
UniRep (1,900), with regressor `xgboost_model_new_KM_esm1b.dat` at 1,332 features. Kroll's
README states the substitution. The DMPNN substrate checkpoint is byte-identical across the
two repositories (all three files match on sha256), so the already-validated fingerprint
stage transfers unchanged.

Their paper and SI say nothing about how Boost_KM was set up: no training description, no
hyperparameters, no statement that pretrained weights were loaded, no commit. Supplementary
Table 8 is cited as the per-method comparison and is not in our mirror. What makes the run
reproducible is their code release, not their prose.

`run_boost_km_esm1b_pinned.py` runs the ESM-1b variant. ESM-1b embeds all 887 enzymes on one
GPU in about four minutes at load average 2.5.

**Result:** 7,157 rows, 3,468 of 3,728 catalytic units, **93.0% coverage**, median K_M
**0.141 mM**. Failures are 194 `too_many_atoms` (the DMPNN pads to 70 nodes) and 105
`multi_component_smiles`. Base is 10, read from the authors' `KMs = 10**bst.predict(dX)`.

### Extraction hazard worth remembering

`bsdtar` pulling the ESM-1b checkpoint out of the 6.6 GB RAR **exited 0 and wrote
649,783,100 bytes** where the archive listing says 7,828,576,466. Silent truncation. Caught
by the size comparison, confirmed by `torch.load` refusing the file. The bytes finally used
came from the published fair-esm URL and verified to sha256
`0569754efaff7dcb7e068c27367bc73f10afb4b450ea30aac30d9bc60783a8b1`.

### kcat predictors disagree more than kcat varies; K_M predictors do not

`plot_per_gene_predictor_agreement.py` re-keys the comparison from the (enzyme, substrate)
pair to the **gene**, which is the unit the flux layer reads. Each gene's value is the
median over its pairs in log10 space, per predictor.

| quantity | median across-predictor spread | consensus p10-p90 range | ratio | genes |
|---|---|---|---|---|
| k_cat | **1.23 decades** | 0.94 decades | **1.32** | 980 |
| K_M | 0.70 decades | 1.34 decades | 0.52 | 1,150 |

For k_cat, 67% of genes move by more than a full decade between the highest and lowest
predictor, and the ratio above 1 means **choosing a different published predictor moves one
gene's turnover number further than real enzymes differ from one another.** Gene-level
Spearman between k_cat predictors runs 0.04 to 0.48.

For K_M the picture is different: Boost_KM and UniKP agree at gene-level Spearman **0.75**
(pair-level 0.79), higher than any k_cat pair. This is agreement, not accuracy, since almost
none of these pairs has a measured value. Read that way it arrives at Wu et al.'s conclusion
from the other side: they report that underground metabolism separates from known metabolism
on K_M rather than k_cat, and K_M is also the parameter the instruments agree on.

### Betaxanthin: the pathway is the least parameterized part of the model

Panel d of the per-gene figure draws the Btx cassette and its tyrosine supply. **CYP76AD1
and DOD have no reaction in yeast-GEM and therefore no predicted kcat from any of the five
tables.** ARO7 alone spans 2.5 decades across predictors. Combined with the ChemAxon
blocker on the pathway intermediates, the flux module cannot currently constrain the
heterologous steps at all; what it can constrain is native precursor supply.

### Plot standard: math now renders in Arial

`PAPER_RC` gained `mathtext.fontset: custom` with all faces set to Arial and
`mathtext.default: regular`. Without it `font.family` governs only the plain text and every
`$...$` span rendered in DejaVu Sans, so a label like `$\sigma(\Delta_r G'^\circ)$ (kJ
mol$^{-1}$)` read as two type sizes in two faces. Greek, primes and `\circ` all exist in
Arial, so nothing falls back to a missing-glyph box. Any script calling `apply_paper_style()`
picks this up; about 15 scripts elsewhere in the repo still set their own rcParams and carry
math, and have not been swept.
