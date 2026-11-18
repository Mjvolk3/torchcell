# experiments/034-showcase-datasets/scripts/ecoli_download_tutorial.py
# [[experiments.034-showcase-datasets.scripts.ecoli_download_tutorial]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/034-showcase-datasets/scripts/ecoli_download_tutorial
"""Load four built E. coli datasets, print their records and plot their values.

This is the code of two documentation pages. ``datasets/ecoli/tutorial`` shows the
blocks that load, tabulate and plot each dataset; ``datasets/ecoli/download`` shows the
blocks that list the endpoint's archives and check the downloaded records against the
knowledge-graph release. A page shows each block between a ``# [name]`` and a
``# [/name]`` marker and, under it, what that block printed or drew. The script needs an
installed ``torchcell`` and a network route to the public endpoint, nothing else: no
repository checkout, no ``DATA_ROOT``, no key.

Run it in an empty directory::

    python ecoli_download_tutorial.py

It downloads into ``./data/torchcell/<slug>/`` and writes one figure of three panels per
dataset into ``./figures/`` (SVG and PNG). Standard output carries the data only and is
the same on every machine; the seconds each loader took to download, verify, unpack and
open its store go to ``./data/torchcell/ecoli_download_tutorial_timings.json`` and to
standard error, because they are not.

The four datasets:

- ``GrowthRateCampos2018Dataset``: maximal growth rate of 3,664 Keio single-gene deletions.
- ``MorphologyCampos2018Dataset``: cell-shape and cell-cycle features of the same strains.
- ``GeneInteractionBabu2014Dataset``: eSGA S-scores of 42,705 double mutants.
- ``MetabolomeFuhrer2017Dataset``: 7,534 ion z-scores for each of 3,806 Keio strains.
"""

from __future__ import annotations

# [setup]
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from torchcell.datasets.artifact import content_sha256
from torchcell.datasets.client import DatasetClient
from torchcell.datasets.ecoli.babu2014 import GeneInteractionBabu2014Dataset
from torchcell.datasets.ecoli.campos2018 import (
    FEATURE_UNITS,
    GrowthRateCampos2018Dataset,
    MorphologyCampos2018Dataset,
)
from torchcell.datasets.ecoli.fuhrer2017 import MetabolomeFuhrer2017Dataset
from torchcell.endpoint_http import DEFAULT_URL
from torchcell.utils import PANEL_WIDTHS_MM, PLOT_PALETTE, mm_to_in, panel_label

# A loader downloads its built store only when TC_DATA_URL is set. Without it the loader
# builds the store from the publisher's files, which is the developer path.
os.environ.setdefault("TC_DATA_URL", DEFAULT_URL)

DATA = Path("data/torchcell")  # each dataset unpacks into DATA/<slug>/
SECONDS: dict[str, float] = {}  # slug -> seconds to download, verify, unpack and open
# [/setup]

# [style]
ORANGE, RED, PURPLE, YELLOW, BLUE, GRAY = PLOT_PALETTE[:6]
plt.rcParams.update(
    {
        "font.family": ["Arial", "DejaVu Sans"],
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "legend.frameon": False,
        "svg.fonttype": "none",  # text stays text in the SVG
        "svg.hashsalt": "torchcell",  # the same bytes on every run
    }
)
FIGURES = Path("figures")
FIGURES.mkdir(exist_ok=True)


def three_panels() -> tuple[Figure, Any]:
    """One row of three panels, 179 mm wide."""
    return plt.subplots(
        1,
        3,
        figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(54)),
        layout="constrained",
    )


def save(figure: Figure, name: str) -> None:
    """Letter the panels a to c and write figures/<name>.svg and .png."""
    for axis, letter in zip(figure.axes, "abc", strict=True):
        panel_label(axis, letter)
    figure.savefig(FIGURES / f"{name}.svg", metadata={"Date": None})
    figure.savefig(FIGURES / f"{name}.png", dpi=200)
    plt.close(figure)


# [/style]

SECTION = "=" * 12


def section(title: str) -> None:
    """Print the banner the page generator splits standard output on."""
    print(f"{SECTION} {title} {SECTION}")


section("index")
# [index]
client = DatasetClient.from_env()
print("endpoint", client.url)
SLUGS = (
    "ecoli_growth_rate_campos2018",
    "ecoli_morphology_campos2018",
    "gene_interaction_babu2014",
    "metabolome_fuhrer2017",
)
for slug in SLUGS:
    artifact = client.select(slug)  # newest archive the installed major.minor reads
    if artifact is None:
        raise SystemExit(f"{client.url} serves no {slug} archive this torchcell reads")
    print(slug)
    print(f"  loader   {artifact.dataset_class}")
    print(f"  archive  {artifact.archive}  ({artifact.archive_bytes:,} bytes)")
    print(f"  records  {artifact.n_experiments:,}")
    print(f"  release  knowledge graph {artifact.kg_version} ({artifact.kg_release})")
# [/index]

section("growth load")
# [growth_load]
started = time.perf_counter()
growth = GrowthRateCampos2018Dataset(root=str(DATA / "ecoli_growth_rate_campos2018"))
SECONDS["ecoli_growth_rate_campos2018"] = time.perf_counter() - started
print(f"{len(growth):,} records")
# [/growth_load]

section("growth record")
# [growth_record]
record = growth[0]  # a dict with three keys: experiment, reference, publication
deletion = record["experiment"]["genotype"]["perturbations"][0]
environment = record["experiment"]["environment"]
phenotype = record["experiment"]["phenotype"]
print("gene       ", deletion["perturbed_gene_name"], deletion["systematic_gene_name"])
print("strain     ", deletion["collection"], "plate", deletion["construction"]["plate"])
print("medium     ", environment["media"]["name"])
print(
    "temperature",
    environment["temperature"]["value"],
    environment["temperature"]["unit"],
)
print(
    "fitness    ", phenotype["fitness"], "from", phenotype["n_samples"], "growth curve"
)
print("wild type  ", record["reference"]["phenotype_reference"]["fitness"])
print("genome     ", record["reference"]["genome_reference"]["assembly_accession"])
print("source     ", record["publication"]["doi_url"])
# [/growth_record]

section("growth table")
# [growth_table]
rows = []
for r in (growth[i] for i in range(len(growth))):
    deletion = r["experiment"]["genotype"]["perturbations"][0]
    rows.append(
        {
            "gene": deletion["perturbed_gene_name"],
            "locus_tag": deletion["systematic_gene_name"],
            "plate": int(deletion["construction"]["plate"]),
            "fitness": r["experiment"]["phenotype"]["fitness"],
        }
    )
growth_table = pd.DataFrame(rows)
print(growth_table.head(6).to_string(index=False, float_format="{:.4f}".format))
print(
    "fitness min {:.4f}  median {:.4f}  max {:.4f}".format(
        growth_table["fitness"].min(),
        growth_table["fitness"].median(),
        growth_table["fitness"].max(),
    )
)
# [/growth_table]

# [growth_figure]
figure, (left, middle, right) = three_panels()
left.hist(growth_table["fitness"], bins=np.arange(0.10, 1.38, 0.02), color=ORANGE)
left.axvline(1.0, color="black", linewidth=0.5, linestyle="--")
left.set(xlabel="Growth rate relative to parent", ylabel="Strains")

ranked = growth_table.sort_values("fitness").reset_index(drop=True)
middle.plot(ranked.index + 1, ranked["fitness"], color=PURPLE, linewidth=1)
for rank, row in ranked.head(5).iterrows():
    middle.annotate(
        row["gene"],
        (rank + 1, row["fitness"]),
        xytext=(600, 0.15 + 0.11 * rank),
        fontstyle="italic",
        va="center",
        arrowprops={"arrowstyle": "-", "linewidth": 0.3},
    )
middle.set(xlabel="Strain rank", ylabel="Growth rate relative to parent")

plates = growth_table.groupby("plate")["fitness"].quantile([0.25, 0.5, 0.75]).unstack()
right.vlines(plates.index, plates[0.25], plates[0.75], color=BLUE, linewidth=1)
right.plot(plates.index, plates[0.5], "o", color="black", markersize=1.5)
right.axhline(1.0, color="black", linewidth=0.5, linestyle="--")
right.set(xlabel="Keio plate", ylabel="Growth rate relative to parent")
save(figure, "ecoli_growth_rate_campos2018")
# [/growth_figure]

section("typed")
# [typed]
typed = growth.transform_item(record)  # the same record as validated pydantic objects
experiment = typed["experiment"]
print(type(experiment).__name__, type(experiment.phenotype).__name__)
print(
    experiment.genotype.perturbations[0].perturbed_gene_name,
    experiment.phenotype.fitness,
)
# [/typed]

section("morphology load")
# [morphology_load]
started = time.perf_counter()
morphology = MorphologyCampos2018Dataset(root=str(DATA / "ecoli_morphology_campos2018"))
SECONDS["ecoli_morphology_campos2018"] = time.perf_counter() - started
print(f"{len(morphology):,} records")

phenotype = morphology[0]["experiment"]["phenotype"]
print("cells measured", phenotype["n_samples"])
print("features      ", list(phenotype["morphology"]))
# [/morphology_load]

section("morphology table")
# [morphology_table]
morphology_table = pd.DataFrame(
    {
        "gene": r["experiment"]["genotype"]["perturbations"][0]["perturbed_gene_name"],
        **r["experiment"]["phenotype"]["morphology"],
    }
    for r in (morphology[i] for i in range(len(morphology)))
).set_index("gene")
print("table shape   ", morphology_table.shape)
print(
    "features per strain:",
    morphology_table.notna().sum(axis=1).value_counts().to_dict(),
)
unit = FEATURE_UNITS.value  # feature symbol -> unit, as the release states it
print("units         ", {symbol: unit[symbol] for symbol in ("<L>", "<W>", "<A>")})
shown = ["<L>", "<W>", "<A>", "<SA/V>", "rho_CD", "%2N"]
print(morphology_table[shown].head(6).to_string(float_format="{:.4f}".format))
# [/morphology_table]

section("morphology growth")
# [morphology_growth]
# The two Campos 2018 datasets hold the same strains in the same order, so the growth
# rate of each strain sits beside its cell shape.
assert list(morphology_table.index) == list(growth_table["gene"])
shape_growth = morphology_table.assign(fitness=growth_table["fitness"].to_numpy())
rank_correlation = shape_growth["<A>"].rank().corr(shape_growth["fitness"].rank())
print(f"strains {len(shape_growth):,}")
print(f"Spearman correlation of cell area with growth rate {rank_correlation:.3f}")
# [/morphology_growth]

# [morphology_figure]
figure, (left, middle, right) = three_panels()
left.hist(morphology_table["<L>"], bins=np.arange(2.0, 4.92, 0.04), color=ORANGE)
left.set(xlabel=f"Mean cell length <L> ({unit['<L>']})", ylabel="Strains")

middle.scatter(
    morphology_table["<L>"],
    morphology_table["<W>"],
    s=1.5,
    color=PURPLE,
    linewidths=0,
    rasterized=True,
)
middle.set(
    xlabel=f"Mean cell length <L> ({unit['<L>']})",
    ylabel=f"Mean cell width <W> ({unit['<W>']})",
)

right.scatter(
    shape_growth["fitness"],
    shape_growth["<A>"],
    s=1.5,
    color=BLUE,
    linewidths=0,
    rasterized=True,
)
right.set(
    xlabel="Growth rate relative to parent",
    ylabel=f"Mean cell area <A> ({unit['<A>']})",
)
save(figure, "ecoli_morphology_campos2018")
# [/morphology_figure]

section("interaction load")
# [interaction_load]
started = time.perf_counter()
interaction = GeneInteractionBabu2014Dataset(
    root=str(DATA / "gene_interaction_babu2014")
)
SECONDS["gene_interaction_babu2014"] = time.perf_counter() - started
print(f"{len(interaction):,} records")

record = interaction[0]
phenotype = record["experiment"]["phenotype"]
for perturbation in record["experiment"]["genotype"]["perturbations"]:
    print(
        "gene   ",
        perturbation["perturbed_gene_name"],
        perturbation["systematic_gene_name"],
        "|",
        perturbation["collection"],
    )
print(
    "S-score", phenotype["gene_interaction"], "from", phenotype["n_samples"], "colonies"
)
print("medium ", record["experiment"]["environment"]["media"]["name"])
# [/interaction_load]

section("interaction table")
# [interaction_table]
rows = []
for r in (interaction[i] for i in range(len(interaction))):
    first, second = r["experiment"]["genotype"]["perturbations"]
    rows.append(
        {
            "gene_a": first["perturbed_gene_name"],
            "gene_b": second["perturbed_gene_name"],
            "id_a": first["systematic_gene_name"],
            "id_b": second["systematic_gene_name"],
            "s_score": r["experiment"]["phenotype"]["gene_interaction"],
        }
    )
pairs = pd.DataFrame(rows)
print(pairs.head(6).to_string(index=False, float_format="{:.4f}".format))
print("genes in any pair:", len(interaction.gene_set))
print("negative scores:", int((pairs["s_score"] < 0).sum()))
print("positive scores:", int((pairs["s_score"] > 0).sum()))
print("strongest negative pairs:")
print(
    pairs.nsmallest(3, "s_score").to_string(index=False, float_format="{:.4f}".format)
)
# [/interaction_table]

# [interaction_figure]
figure, (left, middle, right) = three_panels()
edges = np.arange(-32, 25, 1)
negative = pairs["s_score"] < 0
left.hist(pairs.loc[negative, "s_score"], bins=edges, color=BLUE, label="negative")
left.hist(pairs.loc[~negative, "s_score"], bins=edges, color=RED, label="positive")
left.set(xlabel="S-score", ylabel="Gene pairs")
left.legend()

# One row per gene: how many of its pairs score below zero and how many above.
per_gene = (
    pd.concat(
        [
            pairs[["id_a", "s_score"]].rename(columns={"id_a": "id"}),
            pairs[["id_b", "s_score"]].rename(columns={"id_b": "id"}),
        ]
    )
    .assign(negative=lambda t: t["s_score"] < 0, positive=lambda t: t["s_score"] > 0)
    .groupby("id")[["negative", "positive"]]
    .sum()
)
degree = per_gene["negative"] + per_gene["positive"]
middle.hist(degree, bins=np.logspace(0, np.log10(degree.max() + 1), 40), color=PURPLE)
middle.set(xscale="log", xlabel="Scored pairs per gene", ylabel="Genes")

right.scatter(
    per_gene["negative"],
    per_gene["positive"],
    s=1.5,
    color=BLUE,
    linewidths=0,
    rasterized=True,
)
right.set(
    xscale="symlog",
    yscale="symlog",
    xlabel="Negative pairs per gene",
    ylabel="Positive pairs per gene",
)
for axis in (right.xaxis, right.yaxis):  # linear from 0 to 1, logarithmic above
    axis.set_ticks([0, 1, 10, 100, 1000], labels=["0", "1", "10", "100", "1,000"])
    axis.minorticks_off()
save(figure, "gene_interaction_babu2014")
# [/interaction_figure]

section("metabolome load")
# [metabolome_load]
started = time.perf_counter()
metabolome = MetabolomeFuhrer2017Dataset(root=str(DATA / "metabolome_fuhrer2017"))
SECONDS["metabolome_fuhrer2017"] = time.perf_counter() - started
print(f"{len(metabolome):,} records")

record = metabolome[0]
levels = record["experiment"]["phenotype"]["metabolite_level"]  # ion key -> z-score
print(
    "strain      ", record["experiment"]["genotype"]["perturbations"][0]["strain_label"]
)
print("ions        ", len(levels))
print("first ions  ", dict(list(levels.items())[:4]))

ions = pd.read_csv(Path(metabolome.preprocess_dir) / "ions.csv")  # ion key -> m/z
print(ions.head(4).to_string(index=False, float_format="{:.5f}".format))
# [/metabolome_load]

section("metabolome matrix")
# [metabolome_matrix]
ion_keys = list(levels)
strains, matrix_rows = [], []
for r in (metabolome[i] for i in range(len(metabolome))):
    perturbations = r["experiment"]["genotype"]["perturbations"]
    strains.append("+".join(p["perturbed_gene_name"] for p in perturbations))
    row_levels = r["experiment"]["phenotype"]["metabolite_level"]
    matrix_rows.append([row_levels[key] for key in ion_keys])
matrix = np.asarray(matrix_rows, dtype=np.float32)  # strains x ions
print("matrix shape", matrix.shape, matrix.dtype)
corner = pd.DataFrame(matrix[:5, :5], index=strains[:5], columns=ion_keys[:5])
print(corner.to_string(float_format="{:.4f}".format))
print("z-scores beyond 5 in absolute value:", int((np.abs(matrix) > 5).sum()))
print("records that name more than one gene:", sum("+" in name for name in strains))
# [/metabolome_matrix]

# [metabolome_figure]
figure, (left, middle, right) = three_panels()
left.hist(matrix.ravel(), bins=np.arange(-11.0, 62.5, 0.5), color=ORANGE)
left.set(yscale="log", xlabel="Modified z-score", ylabel="Strain and ion values")

beyond = np.abs(matrix) > 5
middle.hist(beyond.sum(axis=1), bins=np.arange(0, 580, 10), color=PURPLE)
middle.set(yscale="log", xlabel="Ions beyond 5 per strain", ylabel="Strains")

for mode, color in (("pos", RED), ("neg", BLUE)):
    of_mode = (ions["mode"] == mode).to_numpy()
    right.scatter(
        ions.loc[of_mode, "deposit_mz"],
        beyond.sum(axis=0)[of_mode],
        s=1.5,
        color=color,
        linewidths=0,
        rasterized=True,
        label={"neg": "negative mode", "pos": "positive mode"}[mode],
    )
right.set(xlabel="Mass-to-charge ratio", ylabel="Strains beyond 5 per ion")
right.legend(markerscale=3, loc="upper right")
save(figure, "metabolome_fuhrer2017")
# [/metabolome_figure]

section("verify")


# [verify]
def experiment_id(record: dict[str, Any]) -> str:
    """The id the knowledge graph gives this record: sha256 of its experiment."""
    return hashlib.sha256(json.dumps(record["experiment"]).encode("utf-8")).hexdigest()


loaded = {
    "ecoli_growth_rate_campos2018": growth,
    "ecoli_morphology_campos2018": morphology,
    "gene_interaction_babu2014": interaction,
    "metabolome_fuhrer2017": metabolome,
}
for slug, dataset in loaded.items():
    ids = (experiment_id(dataset[i]) for i in range(len(dataset)))
    digest = content_sha256(ids)  # one hash over every record of the dataset
    served = client.select(slug).content_sha256  # what the release recorded
    print(
        f"{slug}  {digest}  {'matches' if digest == served else 'DIFFERS FROM'} the release"
    )
# [/verify]

timings_path = DATA / "ecoli_download_tutorial_timings.json"
timings_path.write_text(json.dumps(SECONDS, indent=2) + "\n", encoding="utf-8")
for slug, seconds in SECONDS.items():
    print(
        f"[seconds to download, verify, unpack and open] {slug}: {seconds:.2f}",
        file=sys.stderr,
    )
