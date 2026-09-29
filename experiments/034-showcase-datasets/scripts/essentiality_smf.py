# experiments/034-showcase-datasets/scripts/essentiality_smf.py
# [[experiments.034-showcase-datasets.scripts.essentiality_smf]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/034-showcase-datasets/scripts/essentiality_smf
"""Generate every table, figure and record dump on the essentiality + SMF showcase page.

Reads the two dev-tree LMDBs read-only (``$DATA_ROOT/data/torchcell/gene_essentiality_sgd``
and ``$DATA_ROOT/data/torchcell/smf_costanzo2016``, the stores ``GeneEssentialitySgdDataset``
and ``SmfCostanzo2016Dataset`` build), resolves interned ``$ref`` pointers the way
``ExperimentDataset.get_single_item`` does, and writes:

- MyST fragments to ``docs/source/showcase/_generated/essentiality-smf/``
  (``record_essentiality.md``, ``record_smf.md``, ``summary_tables.md``, ``figures.md``),
  each opening with an HTML comment naming this script and the run timestamp;
- figures as true-size SVG (``torchcell.utils.savefig_true_size_svg``) plus PNG to
  ``$ASSET_IMAGES_DIR/034-showcase-datasets/``, with the SVGs copied into the
  ``_generated`` directory so Sphinx can embed them;
- ``provenance.md``: loader classes, stores, the raw inputs each loader consumed with
  their sha256, and the citation keys of the sources;
- ``experiments/034-showcase-datasets/results/essentiality_smf_summary.json`` with every
  number the fragments print.

Run from the repo root::

    python experiments/034-showcase-datasets/scripts/essentiality_smf.py
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import os.path as osp
import pickle
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import lmdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from matplotlib.ticker import MultipleLocator  # noqa: E402

from torchcell.data.experiment_dataset import resolve_interned  # noqa: E402
from torchcell.datamodels.schema import (  # noqa: E402
    EXPERIMENT_REFERENCE_TYPE_MAP,
    EXPERIMENT_TYPE_MAP,
)
from torchcell.utils import (  # noqa: E402
    PANEL_WIDTHS_MM,
    PLOT_PALETTE,
    apply_paper_style,
    mm_to_in,
    savefig_true_size_svg,
)

SCRIPT = "experiments/034-showcase-datasets/scripts/essentiality_smf.py"
FRAGMENT_DIR = Path("docs/source/showcase/_generated/essentiality-smf")
RESULTS_JSON = Path(
    "experiments/034-showcase-datasets/results/essentiality_smf_summary.json"
)
IMAGE_SUBDIR = "034-showcase-datasets"

ESSENTIALITY_DIR = "data/torchcell/gene_essentiality_sgd"
SMF_DIR = "data/torchcell/smf_costanzo2016"

# The strain classes SmfCostanzo2016Dataset emits, keyed by perturbation_type, in the
# order the tables and the bar chart use.
STRAIN_TYPES = {
    "sga_kanmx_deletion": "deletion (KanMX)",
    "sga_natmx_deletion": "deletion (NatMX)",
    "damp": "DAmP",
    "temperature_sensitive_allele": "TS allele",
}
TEMPERATURES = (26.0, 30.0)

# What the record dumps keep of ``model_dump()``; everything else is cut and the cut is
# marked in the output, per level.
ESSENTIALITY_FIELDS = (
    "experiment.experiment_type",
    "experiment.dataset_name",
    "experiment.genotype.perturbations[0].systematic_gene_name",
    "experiment.genotype.perturbations[0].perturbed_gene_name",
    "experiment.genotype.perturbations[0].perturbation_type",
    "experiment.genotype.perturbations[0].strain_id",
    "experiment.environment.media.name",
    "experiment.environment.media.state",
    "experiment.environment.temperature.value",
    "experiment.phenotype.graph_level",
    "experiment.phenotype.label_name",
    "experiment.phenotype.is_essential",
    "reference.experiment_reference_type",
    "reference.genome_reference.strain",
    "reference.phenotype_reference.is_essential",
    "publication.pubmed_id",
)
SMF_FIELDS = (
    "experiment.experiment_type",
    "experiment.dataset_name",
    "experiment.genotype.perturbations[0].systematic_gene_name",
    "experiment.genotype.perturbations[0].perturbed_gene_name",
    "experiment.genotype.perturbations[0].perturbation_type",
    "experiment.genotype.perturbations[0].strain_id",
    "experiment.environment.media.name",
    "experiment.environment.media.state",
    "experiment.environment.temperature.value",
    "experiment.phenotype.graph_level",
    "experiment.phenotype.label_name",
    "experiment.phenotype.fitness",
    "experiment.phenotype.fitness_std",
    "experiment.phenotype.fitness_uncertainty_type",
    "experiment.phenotype.n_samples",
    "experiment.phenotype.sample_unit",
    "reference.experiment_reference_type",
    "reference.genome_reference.strain",
    "reference.phenotype_reference.fitness",
    "reference.phenotype_reference.fitness_std",
    "publication.doi",
)


# --------------------------------------------------------------------------- reading
def read_records(processed_dir: str) -> list[dict[str, Any]]:
    """Every record of a built ExperimentDataset, interned pointers resolved.

    Mirrors ``ExperimentDataset.get_single_item``: pickled records from ``lmdb``, the
    sibling ``interned`` env (when present) loaded once, ``resolve_interned`` splicing
    each ``$ref`` back. A store with no ``interned`` env is a legacy inline store.
    """
    interned: dict[str, Any] = {}
    interned_dir = osp.join(processed_dir, "interned")
    if osp.isdir(interned_dir):
        env = lmdb.open(interned_dir, readonly=True, lock=False)
        with env.begin() as txn:
            for key, value in txn.cursor():
                interned[key.decode()] = pickle.loads(value)
        env.close()
    env = lmdb.open(osp.join(processed_dir, "lmdb"), readonly=True, lock=False)
    records: list[dict[str, Any]] = []
    with env.begin() as txn:
        n = txn.stat()["entries"]
        for idx in range(n):
            raw = txn.get(f"{idx}".encode())
            assert raw is not None, f"missing key {idx} in {processed_dir}"
            records.append(resolve_interned(pickle.loads(raw), interned))
    env.close()
    return records


def dump_record(record: dict[str, Any], keep: tuple[str, ...]) -> str:
    """Debugger-style tree of a stored record, cut to the dotted paths in ``keep``.

    The experiment and reference are re-validated through the pydantic classes the
    dataset stored them as (``EXPERIMENT_TYPE_MAP`` / ``EXPERIMENT_REFERENCE_TYPE_MAP``)
    and dumped with ``model_dump()``; the publication dict is used as stored. At every
    level, the number of fields cut is printed as a comment.
    """
    experiment = EXPERIMENT_TYPE_MAP[record["experiment"]["experiment_type"]](
        **record["experiment"]
    )
    reference = EXPERIMENT_REFERENCE_TYPE_MAP[
        record["reference"]["experiment_reference_type"]
    ](**record["reference"])
    full = {
        "experiment": experiment.model_dump(),
        "reference": reference.model_dump(),
        "publication": record["publication"],
    }
    kept: dict[str, Any] = {}
    for path in keep:
        node = kept
        for part in path.split("."):
            node = node.setdefault(part, {})
        node["__leaf__"] = _pick(full, path)
    return "\n".join(_render(kept, full, 0))


def _pick(obj: Any, path: str) -> Any:
    node = obj
    for part in path.split("."):
        name, _, index = part.partition("[")
        node = node[name]
        if index:
            node = node[int(index.rstrip("]"))]
    return node


def _resolve(obj: Any, key: str) -> Any:
    name, _, index = key.partition("[")
    node = obj[name]
    return node[int(index.rstrip("]"))] if index else node


def _render(kept: dict[str, Any], full: Any, depth: int) -> list[str]:
    pad = "    " * depth
    lines: list[str] = []
    for key, child in kept.items():
        if "__leaf__" in child:
            value = child["__leaf__"]
            text = value.value if hasattr(value, "value") else value
            lines.append(f"{pad}{key} = {text!r}")
        else:
            node = _resolve(full, key)
            lines.append(f"{pad}{key}:")
            lines.extend(_render(child, node, depth + 1))
    if isinstance(full, dict):
        shown = {k.partition("[")[0] for k in kept}
        cut = len([k for k in full if k not in shown])
        if cut:
            lines.append(f"{pad}# ... {cut} other field(s) cut")
    return lines


# --------------------------------------------------------------------------- stats
def quantiles(values: list[float]) -> dict[str, float]:
    """n, min, q1, median, mean, q3, max of a list."""
    a = np.asarray(values, dtype=float)
    return {
        "n": int(a.size),
        "min": float(a.min()),
        "q1": float(np.percentile(a, 25)),
        "median": float(np.median(a)),
        "mean": float(a.mean()),
        "q3": float(np.percentile(a, 75)),
        "max": float(a.max()),
    }


def smf_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One flat row per SMF record: gene, strain, type, temperature, fitness, std."""
    rows: list[dict[str, Any]] = []
    for r in records:
        e = r["experiment"]
        p = e["genotype"]["perturbations"][0]
        rows.append(
            {
                "gene": p["systematic_gene_name"],
                "strain_id": p["strain_id"],
                "perturbation_type": p["perturbation_type"],
                "temperature": float(e["environment"]["temperature"]["value"]),
                "fitness": float(e["phenotype"]["fitness"]),
                "fitness_std": e["phenotype"]["fitness_std"],
            }
        )
    return rows


def twin_pairs(rows: list[dict[str, Any]]) -> dict[str, int]:
    """Strains present at both temperatures with identical fitness and std (issue #410)."""
    by_strain: dict[str, dict[float, tuple[float, Any]]] = {}
    for r in rows:
        by_strain.setdefault(r["strain_id"], {})[r["temperature"]] = (
            r["fitness"],
            r["fitness_std"],
        )
    identical = 0
    differing = 0
    single = 0
    for temps in by_strain.values():
        if len(temps) < 2:
            single += 1
        elif temps[26.0] == temps[30.0]:
            identical += 1
        else:
            differing += 1
    return {
        "strains": len(by_strain),
        "pairs_identical_at_both_temperatures": identical,
        "pairs_differing": differing,
        "strains_at_one_temperature": single,
    }


# --------------------------------------------------------------------------- figures
def _box(ax: Any) -> None:
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(0.5)


def figure_histogram(rows: list[dict[str, Any]], out_dir: str) -> dict[str, str]:
    """SMF fitness histogram, one panel per temperature, half-width, shared x."""
    apply_paper_style()
    width_in = mm_to_in(PANEL_WIDTHS_MM["half"])
    fig, axes = plt.subplots(
        2, 1, figsize=(width_in, mm_to_in(62)), sharex=True, constrained_layout=True
    )
    xmax = max(r["fitness"] for r in rows)
    upper = float(np.ceil(xmax * 10) / 10)
    bins = np.arange(0.0, upper + 0.02, 0.02)
    for ax, temperature, color in zip(axes, TEMPERATURES, PLOT_PALETTE[:2]):
        values = [r["fitness"] for r in rows if r["temperature"] == temperature]
        ax.hist(values, bins=bins, color=color, edgecolor="black", linewidth=0.3)
        ax.set_ylabel("records")
        ax.text(
            0.02,
            0.92,
            f"{int(temperature)} °C, n = {len(values):,}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=6,
        )
        ax.xaxis.set_major_locator(MultipleLocator(0.2))
        ax.xaxis.set_minor_locator(MultipleLocator(0.1))
        ax.tick_params(which="minor", length=0)
        ax.grid(True, which="both", axis="x", linewidth=0.3, color="0.85")
        ax.set_axisbelow(True)
        _box(ax)
    axes[0].set_xlim(0.0, upper)
    axes[1].set_xlabel("single-mutant fitness (SmfCostanzo2016Dataset)")
    return _save(fig, out_dir, "smf_fitness_histogram")


def figure_strain_types(
    counts: dict[str, dict[str, int]], out_dir: str
) -> dict[str, str]:
    """Records per strain type, grouped by temperature, half-width."""
    apply_paper_style()
    width_in = mm_to_in(PANEL_WIDTHS_MM["half"])
    fig, ax = plt.subplots(figsize=(width_in, mm_to_in(45)), constrained_layout=True)
    labels = list(STRAIN_TYPES.values())
    x = np.arange(len(labels))
    width = 0.38
    for offset, temperature, color in zip(
        (-width / 2, width / 2), TEMPERATURES, PLOT_PALETTE[:2]
    ):
        heights = [counts[str(int(temperature))][t] for t in STRAIN_TYPES]
        ax.bar(
            x + offset,
            heights,
            width,
            color=color,
            edgecolor="black",
            linewidth=0.3,
            label=f"{int(temperature)} °C",
        )
        for xi, h in zip(x + offset, heights):
            ax.text(xi, h, f"{h:,}", ha="center", va="bottom", fontsize=5)
    ax.set_xticks(x, labels)
    ax.set_ylabel("records")
    ax.set_ylim(0, max(max(v.values()) for v in counts.values()) * 1.18)
    ax.legend(loc="upper right", fontsize=5)
    _box(ax)
    return _save(fig, out_dir, "smf_records_by_strain_type")


def _save(fig: Any, out_dir: str, stem: str) -> dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    svg = osp.join(out_dir, f"{stem}.svg")
    png = osp.join(out_dir, f"{stem}.png")
    savefig_true_size_svg(fig, svg)
    fig.savefig(png, dpi=300)
    plt.close(fig)
    return {"svg": svg, "png": png}


# --------------------------------------------------------------------------- fragments
def header(stamp: str) -> str:
    return f"<!-- generated by {SCRIPT} at {stamp}; do not edit -->\n\n"


def record_fragment(
    stamp: str,
    title: str,
    record: dict[str, Any],
    keep: tuple[str, ...],
    index: int,
    source_dir: str,
) -> str:
    """A record dump fragment: the Python that produced it, then the dump."""
    code = (
        f"records = read_records(osp.join(DATA_ROOT, {source_dir!r}, 'processed'))\n"
        f"print(dump_record(records[{index}], keep={title}_FIELDS))\n"
    )
    return (
        header(stamp)
        + f"Record {index} of `{source_dir}/processed/lmdb`, produced by\n\n"
        + "```python\n"
        + inspect.getsource(read_records)
        + "\n"
        + inspect.getsource(dump_record)
        + "\n"
        + code
        + "```\n\n"
        + "```text\n"
        + dump_record(record, keep)
        + "\n```\n"
    )


def _fmt(v: float) -> str:
    return f"{v:.4f}"


def summary_fragment(stamp: str, s: dict[str, Any]) -> str:
    """The summary_tables.md fragment from the summary dict."""
    ess = s["essentiality"]
    smf = s["smf"]
    lines = [header(stamp).rstrip("\n"), ""]
    lines += [
        "**Counts per dataset**",
        "",
        "| dataset | records | unique genes | detail |",
        "|---|---:|---:|---|",
        f"| `GeneEssentialitySgdDataset` | {ess['records']:,} | {ess['unique_genes']:,} | "
        f"{ess['essential_records']:,} records with `is_essential = True`, "
        f"{ess['non_essential_records']:,} with `False`; {ess['publications']:,} distinct "
        f"PubMed ids; {ess['unique_perturbation_types']} |",
        f"| `SmfCostanzo2016Dataset` | {smf['records']:,} | {smf['unique_genes']:,} | "
        + "; ".join(
            f"{smf['records_by_temperature'][t]:,} records at {t} °C"
            for t in ("26", "30")
        )
        + f"; {smf['unique_strains']:,} distinct strain ids |",
        "",
        "**SMF records by strain type and temperature**",
        "",
        "| strain type (`perturbation_type`) | 26 °C | 30 °C | both |",
        "|---|---:|---:|---:|",
    ]
    for key, label in STRAIN_TYPES.items():
        c26 = smf["records_by_temperature_and_type"]["26"][key]
        c30 = smf["records_by_temperature_and_type"]["30"][key]
        lines.append(f"| {label} (`{key}`) | {c26:,} | {c30:,} | {c26 + c30:,} |")
    tw = smf["twins"]
    lines += [
        "",
        f"Issue #410 twins: of {tw['strains']:,} distinct strain ids, "
        f"{tw['pairs_identical_at_both_temperatures']:,} carry a 26 °C and a 30 °C "
        f"record with byte-identical `fitness` and `fitness_std` ({tw['identical_by_type']}), "
        f"{tw['pairs_differing']:,} carry two records that differ ({tw['differing_by_type']}), "
        f"and {tw['strains_at_one_temperature']:,} appear at one temperature only.",
        "",
        "**SMF fitness distribution** (`phenotype.fitness`)",
        "",
        "| subset | n | min | q1 | median | mean | q3 | max |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, q in smf["fitness_stats"].items():
        lines.append(
            f"| {name} | {q['n']:,} | {_fmt(q['min'])} | {_fmt(q['q1'])} | "
            f"{_fmt(q['median'])} | {_fmt(q['mean'])} | {_fmt(q['q3'])} | {_fmt(q['max'])} |"
        )
    ov = s["overlap"]
    lines += [
        "",
        "**SGD essential genes in the Costanzo SMF table**",
        "",
        f"{ov['essential_genes']:,} genes carry an SGD `inviable` record; "
        f"{ov['essential_genes_with_smf']:,} of them have at least one "
        f"`SmfCostanzo2016Dataset` record and {ov['essential_genes_without_smf']:,} have none. "
        f"The {ov['smf_records_of_essential_genes']:,} SMF records of those genes, by strain "
        "type, at 30 °C:",
        "",
        "| strain type | records (30 °C) | genes | min | median | mean | max |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key, label in STRAIN_TYPES.items():
        row = ov["by_type_30"].get(key)
        if row is None:
            lines.append(f"| {label} | 0 | 0 | - | - | - | - |")
        else:
            lines.append(
                f"| {label} | {row['n']:,} | {row['genes']:,} | {_fmt(row['min'])} | "
                f"{_fmt(row['median'])} | {_fmt(row['mean'])} | {_fmt(row['max'])} |"
            )
    conv = s["converter"]
    lines += [
        "",
        "**What `GeneEssentialityToFitnessConverter` emits from the essentiality store**",
        "",
        "| outcome | records |",
        "|---|---:|",
        f"| `is_essential = True` -> `FitnessPhenotype(fitness=0.0)`, reference fitness 1.0 | {conv['fitness_zero_records']:,} |",
        f"| `is_essential = False` -> dropped (`None`) | {conv['dropped_records']:,} |",
        f"| genes that would carry a converted 0 | {conv['genes_with_fitness_zero']:,} |",
        "",
    ]
    return "\n".join(lines)


def figures_fragment(stamp: str, s: dict[str, Any]) -> str:
    """The figures.md fragment: the two SVGs with captions."""
    smf = s["smf"]
    q26 = smf["fitness_stats"]["26 °C"]
    q30 = smf["fitness_stats"]["30 °C"]
    return (
        header(stamp)
        + "```{figure} _generated/essentiality-smf/smf_fitness_histogram.svg\n"
        + ":name: fig-smf-histogram\n"
        + ":width: 88mm\n\n"
        + f"Single-mutant fitness in `SmfCostanzo2016Dataset`, 0.02-wide bins, one panel per "
        f"temperature (26 °C: n = {q26['n']:,}, median {_fmt(q26['median'])}; "
        f"30 °C: n = {q30['n']:,}, median {_fmt(q30['median'])}). Deletion and DAmP "
        "strains contribute the same value to both panels (issue #410), so the panels differ "
        "only through the temperature-sensitive alleles.\n"
        + "```\n\n"
        + "```{figure} _generated/essentiality-smf/smf_records_by_strain_type.svg\n"
        + ":name: fig-smf-strain-types\n"
        + ":width: 88mm\n\n"
        + "Records per strain type and temperature in `SmfCostanzo2016Dataset` "
        f"({smf['records']:,} records, {smf['unique_strains']:,} strain ids).\n"
        + "```\n"
    )


# --------------------------------------------------------------------------- provenance
def sha256_of(path: str) -> str:
    """sha256 of a file, streamed."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def provenance(
    data_root: str, ess_records: list[dict[str, Any]], smf_records: list[dict[str, Any]]
) -> dict[str, Any]:
    """Where each store's inputs came from, as the loaders record them."""
    smf_raw = osp.join(
        data_root, SMF_DIR, "raw", "strain_ids_and_single_mutant_fitness.xlsx"
    )
    sgd_genes = osp.join(data_root, "data/sgd/genome/genes")
    return {
        "essentiality": {
            "loader": "torchcell/datasets/scerevisiae/sgd.py::GeneEssentialitySgdDataset",
            "store": osp.join(ESSENTIALITY_DIR, "processed", "lmdb"),
            "raw": (
                f"SGD locus JSON, one file per gene under $DATA_ROOT/data/sgd/genome/genes "
                f"({len(os.listdir(sgd_genes)):,} files), fetched by "
                "torchcell.graph.sgd.main_get_all_genes; filter: "
                "phenotype_details with mutant_type 'null', strain 'S288C', phenotype "
                "'inviable'. The files carry no sha256 record."
            ),
            "publications": len({r["publication"]["pubmed_id"] for r in ess_records}),
            "citation_keys": ["cherrySGDSaccharomycesGenome1998"],
        },
        "smf": {
            "loader": "torchcell/datasets/scerevisiae/costanzo2016.py::SmfCostanzo2016Dataset",
            "store": osp.join(SMF_DIR, "processed", "lmdb"),
            "raw_file": osp.join(SMF_DIR, "raw", osp.basename(smf_raw)),
            "raw_sha256": sha256_of(smf_raw),
            "source_url": (
                "https://thecellmap.org/costanzo2016/data_files/Raw%20genetic%20interaction"
                "%20datasets:%20Pair-wise%20interaction%20format.zip"
            ),
            "dois": sorted({r["publication"]["doi"] for r in smf_records}),
            "citation_keys": ["costanzoGlobalGeneticInteraction2016"],
        },
    }


def provenance_fragment(stamp: str, prov: dict[str, Any]) -> str:
    """The provenance.md fragment."""
    ess = prov["essentiality"]
    smf = prov["smf"]
    return (
        header(stamp)
        + "| | `GeneEssentialitySgdDataset` | `SmfCostanzo2016Dataset` |\n"
        + "|---|---|---|\n"
        + f"| loader | `{ess['loader']}` | `{smf['loader']}` |\n"
        + f"| dev store | `$DATA_ROOT/{ess['store']}` | `$DATA_ROOT/{smf['store']}` |\n"
        + f"| raw input | {ess['raw']} | `$DATA_ROOT/{smf['raw_file']}`, sha256 "
        f"`{smf['raw_sha256']}`, extracted from `{smf['source_url']}` "
        "(no raw-mirror record under `$DATA_ROOT/torchcell-raw/` yet) |\n"
        + f"| publications | {ess['publications']:,} PubMed ids, one per record's annotation "
        "| DOI "
        + ", ".join(f"`{d}`" for d in smf["dois"])
        + " |\n"
        + f"| citation key | `{ess['citation_keys'][0]}` (SGD itself) "
        f"| `{smf['citation_keys'][0]}` |\n"
    )


# --------------------------------------------------------------------------- main
def main() -> None:
    """Read both stores, compute, plot, write fragments and the results JSON."""
    load_dotenv()
    data_root = os.getenv("DATA_ROOT")
    asset_dir = os.getenv("ASSET_IMAGES_DIR")
    assert data_root is not None, "DATA_ROOT must be set in .env"
    assert asset_dir is not None, "ASSET_IMAGES_DIR must be set in .env"
    stamp = datetime.now(UTC).isoformat(timespec="seconds")
    image_dir = osp.join(asset_dir, IMAGE_SUBDIR)
    FRAGMENT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)

    ess_records = read_records(osp.join(data_root, ESSENTIALITY_DIR, "processed"))
    smf_records = read_records(osp.join(data_root, SMF_DIR, "processed"))
    print(f"essentiality records: {len(ess_records)}; smf records: {len(smf_records)}")

    # ---- essentiality
    ess_genes = [
        r["experiment"]["genotype"]["perturbations"][0]["systematic_gene_name"]
        for r in ess_records
    ]
    ess_flags = [
        bool(r["experiment"]["phenotype"]["is_essential"]) for r in ess_records
    ]
    essential_gene_set = {g for g, f in zip(ess_genes, ess_flags) if f}
    essentiality = {
        "records": len(ess_records),
        "unique_genes": len(set(ess_genes)),
        "essential_records": sum(ess_flags),
        "non_essential_records": len(ess_flags) - sum(ess_flags),
        "publications": len({r["publication"]["pubmed_id"] for r in ess_records}),
        "unique_perturbation_types": ", ".join(
            sorted(
                {
                    "`"
                    + r["experiment"]["genotype"]["perturbations"][0][
                        "perturbation_type"
                    ]
                    + "`"
                    for r in ess_records
                }
            )
        ),
        "environment": {
            "media": ess_records[0]["experiment"]["environment"]["media"]["name"],
            "state": ess_records[0]["experiment"]["environment"]["media"]["state"],
            "temperature": ess_records[0]["experiment"]["environment"]["temperature"][
                "value"
            ],
        },
        "records_per_gene_max": max(
            sum(1 for g in ess_genes if g == gene) for gene in set(ess_genes)
        ),
    }

    # ---- smf
    rows = smf_rows(smf_records)
    by_temp: dict[str, int] = {}
    by_temp_type: dict[str, dict[str, int]] = {
        str(int(t)): {k: 0 for k in STRAIN_TYPES} for t in TEMPERATURES
    }
    for r in rows:
        t = str(int(r["temperature"]))
        by_temp[t] = by_temp.get(t, 0) + 1
        by_temp_type[t][r["perturbation_type"]] += 1
    unknown_types = {r["perturbation_type"] for r in rows} - set(STRAIN_TYPES)
    assert not unknown_types, f"unlisted perturbation types: {unknown_types}"

    tw: dict[str, Any] = dict(twin_pairs(rows))
    by_strain_type = {r["strain_id"]: r["perturbation_type"] for r in rows}
    ident_by_type: dict[str, int] = {}
    diff_by_type: dict[str, int] = {}
    seen: dict[str, dict[float, tuple[float, Any]]] = {}
    for r in rows:
        seen.setdefault(r["strain_id"], {})[r["temperature"]] = (
            r["fitness"],
            r["fitness_std"],
        )
    for sid, temps in seen.items():
        if len(temps) == 2:
            bucket = ident_by_type if temps[26.0] == temps[30.0] else diff_by_type
            label = STRAIN_TYPES[by_strain_type[sid]]
            bucket[label] = bucket.get(label, 0) + 1
    tw["identical_by_type"] = ", ".join(f"{k}: {v:,}" for k, v in ident_by_type.items())
    tw["differing_by_type"] = ", ".join(f"{k}: {v:,}" for k, v in diff_by_type.items())

    fitness_stats = {
        "all records": quantiles([r["fitness"] for r in rows]),
        "26 °C": quantiles([r["fitness"] for r in rows if r["temperature"] == 26.0]),
        "30 °C": quantiles([r["fitness"] for r in rows if r["temperature"] == 30.0]),
    }
    for key, label in STRAIN_TYPES.items():
        fitness_stats[f"{label}, 30 °C"] = quantiles(
            [
                r["fitness"]
                for r in rows
                if r["temperature"] == 30.0 and r["perturbation_type"] == key
            ]
        )
    smf = {
        "records": len(rows),
        "unique_genes": len({r["gene"] for r in rows}),
        "unique_strains": len({r["strain_id"] for r in rows}),
        "records_by_temperature": by_temp,
        "records_by_temperature_and_type": by_temp_type,
        "twins": tw,
        "fitness_stats": fitness_stats,
        "media": smf_records[0]["experiment"]["environment"]["media"]["name"],
    }

    # ---- overlap
    smf_genes = {r["gene"] for r in rows}
    with_smf = essential_gene_set & smf_genes
    ov_rows = [r for r in rows if r["gene"] in essential_gene_set]
    by_type_30: dict[str, dict[str, Any]] = {}
    for key in STRAIN_TYPES:
        sub = [
            r
            for r in ov_rows
            if r["temperature"] == 30.0 and r["perturbation_type"] == key
        ]
        if sub:
            by_type_30[key] = {
                **quantiles([r["fitness"] for r in sub]),
                "genes": len({r["gene"] for r in sub}),
            }
    overlap = {
        "essential_genes": len(essential_gene_set),
        "essential_genes_with_smf": len(with_smf),
        "essential_genes_without_smf": len(essential_gene_set - smf_genes),
        "smf_records_of_essential_genes": len(ov_rows),
        "by_type_30": by_type_30,
    }

    converter = {
        "fitness_zero_records": sum(ess_flags),
        "dropped_records": len(ess_flags) - sum(ess_flags),
        "genes_with_fitness_zero": len(essential_gene_set),
    }

    summary: dict[str, Any] = {
        "script": SCRIPT,
        "generated_utc": stamp,
        "sources": {
            "essentiality": osp.join(data_root, ESSENTIALITY_DIR, "processed", "lmdb"),
            "smf": osp.join(data_root, SMF_DIR, "processed", "lmdb"),
        },
        "essentiality": essentiality,
        "smf": smf,
        "overlap": overlap,
        "converter": converter,
    }

    # ---- figures
    figs = {
        "histogram": figure_histogram(rows, image_dir),
        "strain_types": figure_strain_types(by_temp_type, image_dir),
    }
    for paths in figs.values():
        shutil.copy2(paths["svg"], FRAGMENT_DIR / osp.basename(paths["svg"]))
    summary["figures"] = figs

    # ---- fragments
    ess_index = 0
    smf_index = next(
        i
        for i, r in enumerate(rows)
        if r["perturbation_type"] == "sga_kanmx_deletion" and r["temperature"] == 30.0
    )
    summary["record_indices"] = {"essentiality": ess_index, "smf": smf_index}
    (FRAGMENT_DIR / "record_essentiality.md").write_text(
        record_fragment(
            stamp,
            "ESSENTIALITY",
            ess_records[ess_index],
            ESSENTIALITY_FIELDS,
            ess_index,
            ESSENTIALITY_DIR,
        )
    )
    (FRAGMENT_DIR / "record_smf.md").write_text(
        record_fragment(
            stamp, "SMF", smf_records[smf_index], SMF_FIELDS, smf_index, SMF_DIR
        )
    )
    (FRAGMENT_DIR / "summary_tables.md").write_text(summary_fragment(stamp, summary))
    (FRAGMENT_DIR / "figures.md").write_text(figures_fragment(stamp, summary))
    summary["provenance"] = provenance(data_root, ess_records, smf_records)
    (FRAGMENT_DIR / "provenance.md").write_text(
        provenance_fragment(stamp, summary["provenance"])
    )
    RESULTS_JSON.write_text(json.dumps(summary, indent=2) + "\n")
    print(
        json.dumps(
            {k: summary[k] for k in ("essentiality", "overlap", "converter")}, indent=2
        )
    )
    print(
        json.dumps(
            {
                k: smf[k]
                for k in (
                    "records",
                    "unique_genes",
                    "unique_strains",
                    "records_by_temperature",
                    "twins",
                )
            },
            indent=2,
        )
    )
    print("finished")


if __name__ == "__main__":
    main()
