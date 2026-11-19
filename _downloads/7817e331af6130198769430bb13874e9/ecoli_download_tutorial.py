# experiments/034-showcase-datasets/scripts/ecoli_download_tutorial.py
# [[experiments.034-showcase-datasets.scripts.ecoli_download_tutorial]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/034-showcase-datasets/scripts/ecoli_download_tutorial
"""Load four built E. coli datasets, show their records and plot their values.

This is the source of the notebook ``experiments/034-showcase-datasets/notebooks/
ecoli_datasets.ipynb`` and of the documentation pages under ``datasets/ecoli``. Each block
between a ``# [name]`` and a ``# [/name]`` marker is one notebook cell. The cells that
load and read the data are short on purpose, because they are the interface a reader
copies; the ``style`` cell and the ``*_figure`` cells hold the plotting code and are
collapsed in the notebook and on the page. A cell that ends in a bare expression (a
table) shows that table in a notebook and prints nothing when this file runs as a script.

The script needs an installed ``torchcell`` and a network route to the public endpoint,
nothing else: no repository checkout, no ``DATA_ROOT``, no key. Run it in an empty
directory::

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
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from torchcell.datasets.artifact import content_sha256
from torchcell.datasets.client import DatasetClient
from torchcell.datasets.ecoli.babu2014 import GeneInteractionBabu2014Dataset
from torchcell.datasets.ecoli.campos2018 import (
    FEATURE_UNITS,
    MORPHOLOGY_ASSAY,
    GrowthRateCampos2018Dataset,
    MorphologyCampos2018Dataset,
)
from torchcell.datasets.ecoli.fuhrer2017 import MetabolomeFuhrer2017Dataset
from torchcell.endpoint_http import DEFAULT_URL
from torchcell.sequence.genome.standard import genome_for, standard_genomes

# A loader downloads its built store only when TC_DATA_URL is set. Without it the loader
# builds the store from the publisher's files, which is the developer path.
os.environ.setdefault("TC_DATA_URL", DEFAULT_URL)
os.environ.setdefault("DATA_ROOT", ".")  # genome files go to ./torchcell-genomes/

DATA = Path("data/torchcell")  # each dataset unpacks into DATA/<slug>/
pd.set_option("display.precision", 4)


def brief(value: Any, depth: int = 1) -> Any:
    """A record part with anything nested deeper than ``depth`` replaced by its size."""
    if isinstance(value, dict):
        kept = {k: v for k, v in value.items() if k != "provenance_gaps"}
        if depth < 0:
            return f"<{len(kept)} fields>"
        return {k: brief(v, depth - 1) for k, v in kept.items()}
    if isinstance(value, list):
        return f"<{len(value)} items>" if value else []
    return getattr(value, "value", value)  # an enum member prints as its value


def show(part: dict[str, Any]) -> None:
    """Print one part of a record as indented JSON, two levels deep."""
    print(json.dumps(brief(part), indent=2, default=str))


# [/setup]

import sys  # noqa: E402
import time  # noqa: E402

# [style]
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from torchcell.utils import (  # noqa: E402
    PANEL_WIDTHS_MM,
    PLOT_PALETTE,
    mm_to_in,
    panel_label,
)

ORANGE, RED, PURPLE, YELLOW, BLUE, GRAY = PLOT_PALETTE[:6]
NEGATIVE, POSITIVE = RED, BLUE  # below and above a reference level, in every figure
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
        "figure.dpi": 150,  # size of a figure shown in a notebook
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


# [/style]

SECTION = "=" * 12
SECONDS: dict[str, float] = {}  # slug -> seconds to download, verify, unpack and open


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
started = time.perf_counter()
# [growth_load]
growth = GrowthRateCampos2018Dataset(root=str(DATA / "ecoli_growth_rate_campos2018"))
print(f"{len(growth):,} records")
# [/growth_load]
SECONDS["ecoli_growth_rate_campos2018"] = time.perf_counter() - started

section("growth record")
# [growth_record]
record = growth[0]  # a dict with three keys: experiment, reference, publication
experiment = record["experiment"]
show(experiment["genotype"]["perturbations"][0])
show(experiment["environment"])
show(experiment["phenotype"])
print("reference fitness", record["reference"]["phenotype_reference"]["fitness"])
print("source", record["publication"]["doi_url"])
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
print(
    "fitness min {:.4f}  median {:.4f}  max {:.4f}".format(
        growth_table["fitness"].min(),
        growth_table["fitness"].median(),
        growth_table["fitness"].max(),
    )
)
growth_table.head(6)
# [/growth_table]

# [growth_figure]
figure, (left, middle, right) = three_panels()
edges = np.arange(0.10, 1.38, 0.02)
slower = growth_table["fitness"] < 1.0
left.hist(
    growth_table.loc[slower, "fitness"], bins=edges, color=NEGATIVE, label="slower"
)
left.hist(
    growth_table.loc[~slower, "fitness"], bins=edges, color=POSITIVE, label="faster"
)
left.axvline(1.0, color="black", linewidth=0.5, linestyle="--")
left.set(xlabel="Growth rate relative to parent", ylabel="Strains")
left.legend()

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
right.vlines(plates.index, plates[0.25], plates[0.75], color=ORANGE, linewidth=1)
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
started = time.perf_counter()
# [morphology_load]
morphology = MorphologyCampos2018Dataset(root=str(DATA / "ecoli_morphology_campos2018"))
print(f"{len(morphology):,} records")
show(morphology[0]["experiment"]["phenotype"])
# [/morphology_load]
SECONDS["ecoli_morphology_campos2018"] = time.perf_counter() - started

section("morphology features")
# [morphology_features]
# The assay vocabulary names every feature a record can hold, with its unit.
features = pd.DataFrame(
    feature.model_dump(mode="json") for feature in MORPHOLOGY_ASSAY.features
).drop(columns="note")
print(features.to_string(index=False))
features
# [/morphology_features]

section("morphology table")
# [morphology_table]
morphology_table = pd.DataFrame(
    {
        "gene": r["experiment"]["genotype"]["perturbations"][0]["perturbed_gene_name"],
        **r["experiment"]["phenotype"]["morphology"],
    }
    for r in (morphology[i] for i in range(len(morphology)))
).set_index("gene")
unit = FEATURE_UNITS.value  # feature symbol -> unit, as the release states it
print("table shape   ", morphology_table.shape)
print(
    "features per strain:",
    morphology_table.notna().sum(axis=1).value_counts().to_dict(),
)
print("units         ", {symbol: unit[symbol] for symbol in ("<L>", "<W>", "<A>")})
morphology_table.head(6)
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

slower = shape_growth["fitness"] < 1.0
for mask, color, label in ((slower, NEGATIVE, "slower"), (~slower, POSITIVE, "faster")):
    right.scatter(
        shape_growth.loc[mask, "fitness"],
        shape_growth.loc[mask, "<A>"],
        s=1.5,
        color=color,
        linewidths=0,
        rasterized=True,
        label=label,
    )
right.set(
    xlabel="Growth rate relative to parent",
    ylabel=f"Mean cell area <A> ({unit['<A>']})",
)
right.legend(markerscale=3)
save(figure, "ecoli_morphology_campos2018")
# [/morphology_figure]

section("interaction load")
started = time.perf_counter()
# [interaction_load]
interaction = GeneInteractionBabu2014Dataset(
    root=str(DATA / "gene_interaction_babu2014")
)
print(f"{len(interaction):,} records")
record = interaction[0]
for perturbation in record["experiment"]["genotype"]["perturbations"]:
    print(
        perturbation["perturbed_gene_name"],
        perturbation["systematic_gene_name"],
        "|",
        perturbation["collection"],
    )
show(record["experiment"]["phenotype"])
# [/interaction_load]
SECONDS["gene_interaction_babu2014"] = time.perf_counter() - started

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
print("genes in any pair:", len(interaction.gene_set))
print("negative scores:", int((pairs["s_score"] < 0).sum()))
print("positive scores:", int((pairs["s_score"] > 0).sum()))
pairs.nsmallest(6, "s_score")
# [/interaction_table]

# [interaction_figure]
figure, (left, middle, right) = three_panels()
edges = np.arange(-32, 25, 1)
negative = pairs["s_score"] < 0
left.hist(pairs.loc[negative, "s_score"], bins=edges, color=NEGATIVE, label="negative")
left.hist(pairs.loc[~negative, "s_score"], bins=edges, color=POSITIVE, label="positive")
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
middle.hist(degree, bins=np.logspace(0, np.log10(degree.max() + 1), 40), color=ORANGE)
middle.set(xscale="log", xlabel="Scored pairs per gene", ylabel="Genes")

right.scatter(
    per_gene["negative"],
    per_gene["positive"],
    s=1.5,
    color=PURPLE,
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
started = time.perf_counter()
# [metabolome_load]
metabolome = MetabolomeFuhrer2017Dataset(root=str(DATA / "metabolome_fuhrer2017"))
print(f"{len(metabolome):,} records")
record = metabolome[0]
levels = record["experiment"]["phenotype"]["metabolite_level"]  # ion key -> z-score
print("strain", record["experiment"]["genotype"]["perturbations"][0]["strain_label"])
print("ions", len(levels), "first", dict(list(levels.items())[:4]))
ions = pd.read_csv(Path(metabolome.preprocess_dir) / "ions.csv")  # ion key -> m/z
# The ions are mass features of the two instrument modes, not named metabolites.
print(ions.groupby("mode")["deposit_mz"].agg(["size", "min", "max"]).to_string())
ions.head(4)
# [/metabolome_load]
SECONDS["metabolome_fuhrer2017"] = time.perf_counter() - started

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
print("z-scores beyond 5 in absolute value:", int((np.abs(matrix) > 5).sum()))
print("records that name more than one gene:", sum("+" in name for name in strains))
pd.DataFrame(matrix[:5, :5], index=strains[:5], columns=ion_keys[:5])
# [/metabolome_matrix]

# [metabolome_figure]
figure, (left, middle, right) = three_panels()
values = matrix.ravel()
edges = np.arange(-11.0, 62.5, 0.5)
left.hist(values[values < 0], bins=edges, color=NEGATIVE, label="negative")
left.hist(values[values >= 0], bins=edges, color=POSITIVE, label="positive")
left.set(yscale="log", xlabel="Modified z-score", ylabel="Strain and ion values")
left.legend()

beyond = np.abs(matrix) > 5
middle.hist(beyond.sum(axis=1), bins=np.arange(0, 580, 10), color=PURPLE)
middle.set(yscale="log", xlabel="Ions beyond 5 per strain", ylabel="Strains")

# Ionization mode is a property of the instrument run, not a sign of the value.
for mode, color in (("pos", ORANGE), ("neg", PURPLE)):
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

section("gene names")
# [gene_names]
loaded = {
    "ecoli_growth_rate_campos2018": growth,
    "ecoli_morphology_campos2018": morphology,
    "gene_interaction_babu2014": interaction,
    "metabolome_fuhrer2017": metabolome,
}
# Every perturbation names its gene twice: perturbed_gene_name is the gene symbol and
# systematic_gene_name is the identifier in the namespace gene_namespace names.
rows = []
for slug, dataset in loaded.items():
    for r in (dataset[i] for i in range(len(dataset))):
        for p in r["experiment"]["genotype"]["perturbations"]:
            mapping = p["identifier_mapping"] or {}
            rows.append(
                {
                    "dataset": slug,
                    "symbol": p["perturbed_gene_name"],
                    "identifier": p["systematic_gene_name"],
                    "namespace": p["gene_namespace"],
                    "identifier_kind": p.get("identifier_kind") or "locus identifier",
                    "mapped_from": mapping.get("route") or "released as is",
                    "perturbation_type": p["perturbation_type"],
                }
            )
names = pd.DataFrame(rows)
how = ["dataset", "perturbation_type", "namespace", "identifier_kind", "mapped_from"]
gene_names = (
    names.groupby(how, sort=False)
    .agg(
        perturbations=("symbol", "size"),
        genes=("identifier", "nunique"),
        example_symbol=("symbol", "first"),
        example_identifier=("identifier", "first"),
    )
    .reset_index()
)
print(gene_names.to_string(index=False))
gene_names
# [/gene_names]

section("glance")
# [glance]
# The same rows for every dataset, counted from the loaded records: the shape of the
# labels, what the genotype is and how it was perturbed, and the growth condition.


def plain(value: Any) -> str:
    """A field as text; an enum shows its value."""
    return str(getattr(value, "value", value))


def tally(values: list[Any]) -> str:
    """The distinct values, most frequent first, with counts when there are several."""
    counts = pd.Series([plain(v) for v in values]).value_counts()
    if len(counts) == 1:
        return str(counts.index[0])
    return "; ".join(f"{value} ({n:,})" for value, n in counts.items())


def glance(dataset: Any) -> dict[str, str]:
    """One dataset's row values, in the order every dataset is shown in."""
    experiments = [dataset[i]["experiment"] for i in range(len(dataset))]
    perturbations = [p for e in experiments for p in e["genotype"]["perturbations"]]
    phenotypes = [e["phenotype"] for e in experiments]
    labels = [ph[ph["label_name"]] for ph in phenotypes]
    values = [x for v in labels for x in (v.values() if isinstance(v, dict) else [v])]
    width = max(len(v) if isinstance(v, dict) else 1 for v in labels)
    cells = len(experiments) * width  # a record may lack some of the label's values
    missing = cells - sum(x is not None and x == x for x in values)
    samples = [ph["n_samples"] for ph in phenotypes if ph.get("n_samples") is not None]
    media = [e["environment"]["media"] for e in experiments]
    # A strain is a genotype: the set of perturbed genes, each with how it was perturbed.
    strains = {
        tuple(
            sorted(
                (p["systematic_gene_name"], p["perturbation_type"])
                for p in e["genotype"]["perturbations"]
            )
        )
        for e in experiments
    }
    return {
        "Records": f"{len(experiments):,}",
        "Distinct strains (genotypes)": f"{len(strains):,}",
        "Label array (records x values)": f"{len(experiments):,} x {width:,}",
        "Missing label values": f"{missing:,} of {cells:,}",
        "Experiment type": tally([e["experiment_type"] for e in experiments]),
        "Label field": tally([ph["label_name"] for ph in phenotypes]),
        "Label attaches to": tally([ph["graph_level"] for ph in phenotypes]),
        "Measurements behind one record": (
            (
                f"{min(samples):,}"
                if min(samples) == max(samples)
                else f"{min(samples):,} to {max(samples):,}"
            )
            + " "
            + tally([ph.get("sample_unit") for ph in phenotypes])
            if samples
            else "not recorded"
        ),
        "Genes perturbed per record": tally(
            [len(e["genotype"]["perturbations"]) for e in experiments]
        ),
        "Perturbation type": tally([p["perturbation_type"] for p in perturbations]),
        "Perturbation mechanism": tally(
            [
                p.get("mechanism_so_name") or p.get("released_kind")
                for p in perturbations
            ]
        ),
        "Strain collection": tally([p.get("collection") for p in perturbations]),
        "Distinct genes perturbed": f"{len({p['perturbed_gene_name'] for p in perturbations}):,}",
        "Gene identifier namespace": tally(
            [p["gene_namespace"] for p in perturbations]
        ),
        "Medium": tally([f"{m['base_medium']}, {m['state']}" for m in media]),
        "Temperature": tally(
            [
                f"{e['environment']['temperature']['value']:g} "
                f"{plain(e['environment']['temperature']['unit'])}"
                for e in experiments
            ]
        ),
    }


at_a_glance = pd.DataFrame({slug: glance(dataset) for slug, dataset in loaded.items()})
at_a_glance.index.name = "property"
for line in [
    " | ".join(["property", *at_a_glance.columns]),
    *(" | ".join([name, *row]) for name, row in at_a_glance.iterrows()),
]:
    print(line)
at_a_glance
# [/glance]

section("gene overlap")
# [gene_overlap]
# The gene symbol is the name every dataset carries, so it is the key to join on.
symbols = {slug: set(group["symbol"]) for slug, group in names.groupby("dataset")}
overlap = pd.DataFrame(
    {a: {b: len(symbols[a] & symbols[b]) for b in loaded} for a in loaded}
)
print(overlap.to_string())
overlap
# [/gene_overlap]

section("genome standard")
# [genome_standard]
for row in standard_genomes():  # every genome that loads by its assembly-set id
    print(f"{row.assembly_set:32} {row.organism} {row.strain} -> {row.class_name}")
# [/genome_standard]

section("genome load")
# [genome_load]
# genome_for reads the assembly set the records name, fetches its files (original
# source first, the torchcell mirror when the source fails or its bytes changed, every
# file checked against its sha256) and returns the genome object.
genomes = {slug: genome_for(dataset) for slug, dataset in loaded.items()}
resolved = {}  # slug -> {identifier the dataset stores -> how the genome resolves it}
for slug, dataset in loaded.items():
    genome = genomes[slug]
    resolved[slug] = {name: genome.resolve_gene_name(name) for name in dataset.gene_set}
    status = pd.Series([r.status.value for r in resolved[slug].values()]).value_counts()
    print(slug)
    print(f"  genome        {type(genome).__name__} ({genome.ASSEMBLY_SET})")
    print(f"  genome genes  {len(genome.gene_set):,}")
    print(f"  dataset genes {len(dataset.gene_set):,}: {status.sort_index().to_dict()}")
# [/genome_load]

section("genome record")
# [genome_record]
# One record as model input. The nodes of a cell graph are the genes of the genome; a
# strain is the indices of the nodes it perturbs, and the phenotype is the label.
slug = "gene_interaction_babu2014"
genome = genomes[slug]
nodes = list(genome.gene_set)  # sorted, so a node index is stable
node_index = {gene: i for i, gene in enumerate(nodes)}

experiment = interaction[0]["experiment"]
stored = [p["systematic_gene_name"] for p in experiment["genotype"]["perturbations"]]
targets = [resolved[slug][name].systematic_name for name in stored]
print("stored identifiers ", stored)
print("genome locus tags  ", targets)
print("perturbed nodes    ", [node_index[t] for t in targets], f"of {len(nodes):,}")
print("label              ", experiment["phenotype"]["label_name"])
for target in targets:
    gene = genome[target]  # sequence, protein and annotation of one node
    print(
        f"{gene.id} {gene.symbol}: {len(gene.seq):,} bp, "
        f"protein {len(gene.protein.seq):,} aa, GO terms {len(gene.go)}"
    )
# [/genome_record]

section("verify")


# [verify]
def experiment_id(record: dict[str, Any]) -> str:
    """The id the knowledge graph gives this record: sha256 of its experiment."""
    return hashlib.sha256(json.dumps(record["experiment"]).encode("utf-8")).hexdigest()


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
