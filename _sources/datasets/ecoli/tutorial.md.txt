# The E. coli datasets

Four built *Escherichia coli* K-12 datasets load with one constructor call each, from a plain install of torchcell and with no account or key.

## Quick start

```bash
pip install torchcell      # Python 3.13 or newer; or: uv pip install torchcell
```

```python
import os

from torchcell.endpoint_http import DEFAULT_URL

os.environ.setdefault("TC_DATA_URL", DEFAULT_URL)  # download built stores

from torchcell.datasets.ecoli.campos2018 import GrowthRateCampos2018Dataset

growth = GrowthRateCampos2018Dataset(root="data/torchcell/ecoli_growth_rate_campos2018")
print(len(growth))                                 # 3664
print(growth[0]["experiment"]["phenotype"]["fitness"])
```

The constructor downloads the store on first use and opens it from disk after that. Nothing else is needed: no repository checkout, no database, no `DATA_ROOT`. The four downloads together take seconds to a few tens of seconds ({doc}`download`).

This page says what each dataset holds and shows one figure of its values. The full walkthrough, with every record, table and plot, is a notebook:

- {doc}`notebook`: the notebook rendered here, with its outputs and the plotting code collapsed.
- [Open it in Google Colab](https://colab.research.google.com/github/Mjvolk3/torchcell/blob/main/experiments/034-showcase-datasets/notebooks/ecoli_datasets.ipynb) to run it in the browser with nothing installed: the first cell installs `torchcell`.
- {download}`Download ecoli_datasets.ipynb <../../../../experiments/034-showcase-datasets/notebooks/ecoli_datasets.ipynb>` to run it yourself.
- [View it on GitHub](https://github.com/Mjvolk3/torchcell/blob/main/experiments/034-showcase-datasets/notebooks/ecoli_datasets.ipynb).

```{include} _generated/tutorial/datasets_table.md
```

In the figures, red marks values below a reference level and blue marks values above it.

## How each dataset names its genes

Every perturbation in every record names its gene twice. `perturbed_gene_name` is the gene symbol. `systematic_gene_name` is an identifier in the namespace that `gene_namespace` names. The identifier is not the same kind in every dataset: the Campos 2018 stores hold BW25113 locus tags that the loader mapped from the released gene symbol, the Babu 2014 store holds the MG1655 b-numbers the paper released, and the Fuhrer 2017 store holds the Keio JW ids its deposit released. A row whose `perturbation_type` is `released_identifier` holds the identifier exactly as the source released it, not resolved against a genome annotation.

```{include} _generated/tutorial/output_gene_names.md
```

The gene symbol is the one name all four datasets carry, so it is the key to join them on. The table counts the symbols each pair of datasets shares, with each dataset's own count on the diagonal.

```{include} _generated/tutorial/output_gene_overlap.md
```

## Growth rate (Campos 2018)

One record per strain of the Keio collection, the single-gene deletion strains of *E. coli* K-12 BW25113. The label is the strain's maximal growth rate in M9 medium with casamino acids and glucose at 30 °C, as a ratio to the parent strain, so the reference level is 1.0. Each value comes from one growth curve, and the release gives no spread.

```python
from torchcell.datasets.ecoli.campos2018 import GrowthRateCampos2018Dataset

dataset = GrowthRateCampos2018Dataset(root="data/torchcell/ecoli_growth_rate_campos2018")
record = dataset[0]  # a dict: experiment, reference, publication
```

```{figure} _generated/tutorial/ecoli_growth_rate_campos2018.svg
:name: fig-ecoli-growth-rate
:width: 100%

Growth rate of the Keio strains in `GrowthRateCampos2018Dataset`. **a**, strains per 0.02-wide bin of growth rate relative to the parent, slower than the parent in red and faster in blue, with the parent level of 1.0 dashed. **b**, the strains ranked from slowest to fastest, with the five slowest named. **c**, the median (dot) and the interquartile range (bar) of the strains on each Keio plate.
```

## Cell shape and cell cycle (Campos 2018)

The same Keio strains from the same screen, imaged in phase contrast and, for most strains, with a DAPI nucleoid channel. A record summarizes the cells measured for one strain (the paper reports 291 ± 116 cells per strain), and its `n_samples` is that strain's number of cells. The features fall in two groups. The cell-shape features (length, width, area, volume, surface area, perimeter, circularity, aspect ratio, surface-to-volume ratio) are stored as a mean in `morphology` and as a coefficient of variation in `morphology_coefficient_of_variation`. The nucleoid and cell-cycle features (nucleoid area, the timing of constriction and of nucleoid separation, the fraction of cells with two nucleoids) exist only for strains imaged with a nucleoid channel, so a strain without one holds 9 mean features and not 15. Feature names are the symbols the paper gives them:

```{include} _generated/tutorial/output_morphology_features.md
```

```python
from torchcell.datasets.ecoli.campos2018 import MorphologyCampos2018Dataset

dataset = MorphologyCampos2018Dataset(root="data/torchcell/ecoli_morphology_campos2018")
record = dataset[0]  # a dict: experiment, reference, publication
```

```{figure} _generated/tutorial/ecoli_morphology_campos2018.svg
:name: fig-ecoli-morphology
:width: 100%

Cell shape of the Keio strains in `MorphologyCampos2018Dataset`, one point or count per strain. **a**, strains per 0.04 µm bin of mean cell length. **b**, mean cell width against mean cell length. **c**, mean cell area against the growth rate of the same strain from `GrowthRateCampos2018Dataset`, strains slower than the parent in red and faster in blue.
```

## Genetic interactions (Babu 2014)

One record per double mutant. A donor strain carrying one marked deletion is crossed by conjugation with an arrayed recipient strain carrying another, and the double mutant's colony size is measured. The label is the interaction score $S$ of the pair. Butland et al. 2008, the method paper, define it in words: "We then calculated an interaction score (S) to quantify the strength and confidence of genetic interaction determined for each mutant gene pair. Negative $S$ scores correspond to putative aggravating interactions and positive S scores, to putative alleviating interactions." Babu et al. 2014 average eight replicate colony measurements of each pair into one score. The release holds only the high-confidence pairs, which the paper states as

$$
S_{ij} \le -3 \quad (25{,}239 \text{ aggravating pairs}) \qquad \text{or} \qquad S_{ij} \ge +3 \quad (17{,}466 \text{ alleviating pairs}),
$$

42,705 pairs in all, so no stored score lies near zero. Neither paper prints the formula that turns colony sizes into $S$: both refer to Collins et al. 2006 (*Genome Biology* 7, R63) for it.

```python
from torchcell.datasets.ecoli.babu2014 import GeneInteractionBabu2014Dataset

dataset = GeneInteractionBabu2014Dataset(root="data/torchcell/gene_interaction_babu2014")
record = dataset[0]  # a dict: experiment, reference, publication
```

```{figure} _generated/tutorial/gene_interaction_babu2014.svg
:name: fig-ecoli-genetic-interaction
:width: 100%

S-scores in `GeneInteractionBabu2014Dataset`. **a**, gene pairs per unit-wide bin of S-score, negative scores in red and positive scores in blue. The gap around zero is the high-confidence cut. **b**, genes by the number of scored pairs each takes part in, on logarithmically spaced bins. **c**, positive against negative pairs per gene, one point per gene, on axes that are linear from 0 to 1 and logarithmic above.
```

## Metabolome (Fuhrer 2017)

One record per Keio strain, measured by flow-injection time-of-flight mass spectrometry. The label maps each detected ion to a modified z-score, the strain's level of that ion standardized across strains. An ion here is a mass feature, a peak at one mass-to-charge ratio in the negative or the positive ionization mode, keyed `neg_0001`, `pos_0001` and so on. The store does not name a metabolite for an ion: `preprocess/ions.csv` gives each ion's mode and mass-to-charge ratio, and matching an ion to a compound is left to the reader. Every record holds the same ions in the same order, so the labels stack into one strains-by-ions matrix. The first lines printed on loading give the number of ions in each mode and their mass-to-charge range:

```{include} _generated/tutorial/output_metabolome_load.md
```

```python
from torchcell.datasets.ecoli.fuhrer2017 import MetabolomeFuhrer2017Dataset

dataset = MetabolomeFuhrer2017Dataset(root="data/torchcell/metabolome_fuhrer2017")
record = dataset[0]  # a dict: experiment, reference, publication
```

```{figure} _generated/tutorial/metabolome_fuhrer2017.svg
:name: fig-ecoli-metabolome
:width: 100%

Modified z-scores in `MetabolomeFuhrer2017Dataset`. **a**, all strain and ion values per 0.5-wide bin, negative values in red and positive values in blue, on a logarithmic count axis. **b**, strains by the number of ions whose z-score is beyond 5 in absolute value, in bins of 10 ions. **c**, for each ion, the number of strains beyond 5 against the ion's mass-to-charge ratio, colored by ionization mode.
```

## Provenance

- Notebook: `experiments/034-showcase-datasets/notebooks/ecoli_datasets.ipynb`, built by `experiments/034-showcase-datasets/scripts/ecoli_download_tutorial_page.py` from the blocks of `ecoli_download_tutorial.py` and run in a Jupyter kernel.
- The figures and tables on this page are from the same code run as a script, 179 mm wide with 7 pt Arial text and colors from `torchcell.utils.PLOT_PALETTE`.
- The S-score quotes are from Butland et al. 2008 (main text) and Babu et al. 2014 (Protocol S2 and Results), as pinned in `torchcell/datasets/ecoli/babu2014.py`.
- The printed output was byte-identical on a repository checkout and on a new environment installed from PyPI ({doc}`../../guide/install-timings`).
