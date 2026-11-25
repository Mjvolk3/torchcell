# experiments/050-ecoli-benchmark/scripts/website_pages.py
# [[experiments.050-ecoli-benchmark.scripts.website_pages]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/050-ecoli-benchmark/scripts/website_pages
"""Write the E. coli pages of the benchmark website from the committed results.

Reads every ``results/probes/<task>.<feature>.csv`` (``linear_probes.py``),
``results/splits_summary.csv`` (``build_splits.py``), the example scripts in
``examples/benchmark`` and the output ``run_examples.py`` recorded, and writes

- ``results/linear_probes.csv``: every probe run in one table.
- ``experiments/050-ecoli-benchmark/website/benchmark/datasets/ecoli/``: the overview
  page ``index.mdx``, one page per task, and one figure per task (PNG and SVG) at Nature Biotechnology print size, which the
  page shows at most that large (180 mm wide, 170 mm tall) at any window width.

The website branch copies that directory into ``website/docs/benchmark/datasets/ecoli/``.
``notebook_page.py`` writes ``processing.mdx`` into the same directory.

The statistic in every table cell and bar is the test Pearson correlation, averaged
over the targets of the task, at the ridge penalty chosen on validation; a cell is the
mean and sample standard deviation over the seeds of a split family.

    python experiments/050-ecoli-benchmark/scripts/website_pages.py
"""

from __future__ import annotations

import glob
import json
import os.path as osp
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

from torchcell.tasks.ecoli import ECOLI_TASKS  # noqa: E402
from torchcell.utils import (  # noqa: E402
    MAX_HEIGHT_MM,
    PANEL_WIDTHS_MM,
    PLOT_PALETTE,
    mm_to_in,
)

RESULTS = "experiments/050-ecoli-benchmark/results"
OUT = Path("experiments/050-ecoli-benchmark/website/benchmark/datasets/ecoli")
EXAMPLES = Path("examples/benchmark")
GITHUB = "https://github.com/Mjvolk3/torchcell/blob/main"

# Representation: label, what the model reads, the model, numbers per gene.
REPRESENTATIONS: dict[str, tuple[str, str, str, str]] = {
    "random": (
        "Random",
        "nothing",
        "1,280 standard-normal numbers per gene, fixed seed",
        "1,280",
    ),
    "one_hot": (
        "One-hot",
        "gene identity",
        "one indicator per gene of the task",
        "genes in task",
    ),
    "codon_frequency": (
        "Codon frequency",
        "coding sequence",
        "relative frequency of the 64 codons",
        "64",
    ),
    "calm": (
        "CaLM",
        "coding sequence, codons",
        "CaLM (Outeiral and Deane 2024), last layer, mean over codons",
        "768",
    ),
    "genslm_25m": (
        "GenSLM 25M",
        "coding sequence, codons",
        "GenSLM 25M foundation model (Zvyagin 2023), last layer, mean over codons",
        "512",
    ),
    "genslm_250m": (
        "GenSLM 250M",
        "coding sequence, codons",
        "GenSLM 250M foundation model (Zvyagin 2023), last layer, mean over codons",
        "1,024",
    ),
    "nt_v1_2500m": (
        "NT v1 2.5B",
        "coding sequence, 6-mers",
        "Nucleotide Transformer v1, 2.5B multi-species (`InstaDeepAI/nucleotide-transformer-2.5b-multi-species`), last layer, mean over tokens",
        "2,560",
    ),
    "nt_v2_500m": (
        "NT v2 500M",
        "coding sequence, 6-mers",
        "Nucleotide Transformer v2, 500M multi-species (`InstaDeepAI/nucleotide-transformer-v2-500m-multi-species`), last layer, mean over tokens",
        "1,024",
    ),
    "esm2": (
        "ESM-2 650M",
        "protein sequence",
        "ESM-2 650M (`facebook/esm2_t33_650M_UR50D`), last layer, mean over residues",
        "1,280",
    ),
    "prot_t5": (
        "ProtT5-XL",
        "protein sequence",
        "ProtT5-XL-UniRef50 encoder (`Rostlab/prot_t5_xl_uniref50`), mean over residues",
        "1,024",
    ),
}
FAMILIES = {
    "random": "Random",
    "seqid30": "Sequence identity 30 %",
    "seqid50": "Sequence identity 50 %",
    "position": "Chromosome position",
    "gene_holdout": "Gene holdout, random",
    "seqid30_gene_holdout": "Gene holdout, identity 30 %",
    "seqid50_gene_holdout": "Gene holdout, identity 50 %",
    "position_gene_holdout": "Gene holdout, position",
}
# Task: page file, page title, sidebar label, example script, what was measured.
PAGES: dict[str, tuple[str, str, str, str, str]] = {
    "ecoli-growth-rate-campos2018": (
        "growth-rate-campos2018",
        "Growth rate of single-gene deletions (Campos 2018)",
        "Growth rate (Campos 2018)",
        "ecoli_growth_rate_campos2018",
        "Campos and colleagues grew each strain of the Keio collection of *Escherichia coli* K-12 BW25113 single-gene deletions in M9 medium and fit its growth curve. The label is the maximal growth rate of the deletion strain divided by that of the parent strain, one number per strain. A model receives the deleted gene and predicts that number.",
    ),
    "ecoli-morphology-campos2018": (
        "morphology-campos2018",
        "Cell shape and cell cycle of single-gene deletions (Campos 2018)",
        "Cell shape and cell cycle (Campos 2018)",
        "ecoli_morphology_campos2018",
        "The same imaging screen of the Keio collection measured the shape and the cell-cycle state of thousands of cells per strain. The label is 15 numbers per strain: population means of features such as cell length, width, area and, for strains imaged with a nucleoid stain, nucleoid and cell-cycle features. The six nucleoid and cell-cycle features are missing for strains imaged without that stain. A model receives the deleted gene and predicts all 15 features, and each feature is scored on the strains that measured it.",
    ),
    "ecoli-gene-interaction-babu2014": (
        "gene-interaction-babu2014",
        "Genetic interaction score of double mutants (Babu 2014)",
        "Genetic interactions (Babu 2014)",
        "ecoli_gene_interaction_babu2014",
        "Babu and colleagues crossed *Escherichia coli* K-12 single-gene deletion strains pairwise by conjugation (eSGA) and scored the colony size of each double mutant against the expectation from its two single mutants. The label is the S-score of the pair: negative when the double mutant grows worse than expected, positive when it grows better. The released set holds the pairs that passed the paper's significance cut. A model receives the two deleted genes and predicts the S-score.",
    ),
    "ecoli-metabolome-fuhrer2017": (
        "metabolome-fuhrer2017",
        "Metabolome of single-gene deletions (Fuhrer 2017)",
        "Metabolome (Fuhrer 2017)",
        "ecoli_metabolome_fuhrer2017",
        "Fuhrer and colleagues profiled the metabolome of each Keio deletion strain by flow-injection time-of-flight mass spectrometry. The label is 7,534 numbers per strain: the modified z-score of each detected ion (a mass feature of the negative or the positive instrument mode) relative to the other strains. A model receives the deleted gene and predicts all 7,534 z-scores.",
    ),
}
BOARDLESS = {
    "ecoli-metabolome-fuhrer2017": "This task has no board yet. A submission to a board is one row per record and target, and with 7,534 targets one upload is about 5.4 million rows, far above the 16 MB the service accepts. The results below were scored offline with the same splits."
}
SPLIT_NOTES = {
    "random": "Genes (or pairs) assigned at random.",
    "seqid30": "Whole protein clusters at 30 % sequence identity.",
    "seqid50": "Whole protein clusters at 50 % sequence identity.",
    "position": "Whole chromosome blocks (40 contiguous blocks).",
    "gene_holdout": "Genes split at random; a pair follows its genes.",
    "seqid30_gene_holdout": "Genes split by 30 % identity cluster; a pair follows its genes.",
    "seqid50_gene_holdout": "Genes split by 50 % identity cluster; a pair follows its genes.",
    "position_gene_holdout": "Genes split by chromosome block; a pair follows its genes.",
}


def board_url(task: str, family: str) -> str:
    """Site path of the leaderboard with one board of a task selected."""
    return f"/benchmark/leaderboard/?dataset={task}-{family.replace('_', '-')}-s0"


def front_matter(title: str, label: str, description: str) -> list[str]:
    """Docusaurus front matter."""
    return [
        "---",
        f"title: {json.dumps(title)}",
        f"sidebar_label: {json.dumps(label)}",
        f"description: {json.dumps(description)}",
        "---",
        "",
    ]


def table(head: list[str], rows: list[list[str]], left: int = 1) -> list[str]:
    """A markdown table; the first ``left`` columns are left-aligned, the rest right."""
    align = [":--" if i < left else "--:" for i in range(len(head))]
    lines = ["| " + " | ".join(head) + " |", "| " + " | ".join(align) + " |"]
    return lines + ["| " + " | ".join(row) + " |" for row in rows] + [""]


def lighter(color: str, amount: float = 0.55) -> tuple[float, float, float]:
    """A color mixed toward white: the lighter member of a two-level bar."""
    r, g, b = to_rgb(color)
    return (r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount)


def seed_stats(values: pd.Series) -> tuple[float, float, int]:
    """Mean, sample standard deviation and count of the defined values."""
    finite = values.dropna()
    if len(finite) == 0:
        return float("nan"), float("nan"), 0
    return (
        float(finite.mean()),
        float(finite.std(ddof=1)) if len(finite) > 1 else 0.0,
        len(finite),
    )


def result_table(
    runs: pd.DataFrame,
    features: list[str],
    column: str,
    rows: list[tuple[str, pd.DataFrame]],
) -> list[str]:
    """One row per split family, one column per representation, the best mean in bold."""
    body = []
    for label, part in rows:
        stats = [seed_stats(part.loc[part["feature"] == f, column]) for f in features]
        means = [m for m, _, n in stats if n]
        best = max(means) if means else float("nan")
        cells = []
        for mean, std, n in stats:
            if n == 0:
                cells.append("undefined")
                continue
            text = f"{mean:.3f} ± {std:.3f}"
            cells.append(f"**{text}**" if mean == best else text)
        body.append([label, *cells])
    return table(["Split family", *[REPRESENTATIONS[f][0] for f in features]], body)


def figure(task: str, runs: pd.DataFrame, features: list[str]) -> str:
    """One panel per split family, one horizontal bar (or two) per representation.

    Drawn at Nature Biotechnology print size: the full page width (179 mm), at most
    170 mm tall, 7 pt labels and 6 pt tick labels. The page shows it no larger than that.
    """
    families = list(runs["family"].drop_duplicates())
    combines = list(runs["combine"].drop_duplicates())
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 7,
            "xtick.labelsize": 6,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "svg.fonttype": "none",
        }
    )
    ncols = min(len(families), 4) if len(families) <= 4 else 3
    nrows = -(-len(families) // ncols)
    row_mm = 4.2 * len(features) * (1.5 if len(combines) > 1 else 1.0) + 17
    height_mm = min(nrows * row_mm, MAX_HEIGHT_MM)
    fig, grid = plt.subplots(
        nrows,
        ncols,
        figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(height_mm)),
        sharey=True,
        sharex=True,
        squeeze=False,
    )
    axes = list(grid.ravel())
    span = 0.8 / len(combines)
    for axis, family in zip(axes, families):
        part = runs[runs["family"] == family]
        for i, feature in enumerate(features):
            for j, combine in enumerate(combines):
                mean, std, n = seed_stats(
                    part.loc[
                        (part["feature"] == feature) & (part["combine"] == combine),
                        "test_pearson",
                    ]
                )
                if n == 0:
                    continue
                color = PLOT_PALETTE[i]
                axis.barh(
                    i + (j + 0.5) * span - 0.4,
                    mean,
                    span,
                    xerr=std,
                    color=color if j == 0 else lighter(color),
                    edgecolor="black",
                    linewidth=0.4,
                    error_kw={"elinewidth": 0.5, "capsize": 1.2, "capthick": 0.5},
                )
        axis.axvline(0, color="black", linewidth=0.5)
        axis.set_title(FAMILIES[family], pad=3)
        axis.xaxis.set_major_locator(MaxNLocator(nbins=5))
        axis.xaxis.grid(True, linewidth=0.3, color="#cccccc")
        axis.set_axisbelow(True)
        axis.tick_params(width=0.5, length=2, labelbottom=True)
        for spine in axis.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.5)
            spine.set_color("black")
    for axis in axes[len(families) :]:
        axis.axis("off")
    for row in range(nrows):
        left = grid[row, 0]
        left.set_yticks(range(len(features)))
        left.set_yticklabels([REPRESENTATIONS[f][0] for f in features])
    grid[0, 0].invert_yaxis()
    for axis in axes[: len(families)]:
        axis.set_xlabel("Test Pearson r")
    if len(combines) > 1:
        handles = [
            Patch(
                facecolor="#666666",
                edgecolor="black",
                linewidth=0.4,
                label="Two genes concatenated",
            ),
            Patch(
                facecolor=lighter("#666666"),
                edgecolor="black",
                linewidth=0.4,
                label="Two genes summed",
            ),
        ]
        spare = axes[len(families) :]
        if spare:
            spare[0].legend(handles=handles, loc="center", frameon=False)
        else:
            fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(pad=0.6, w_pad=1.0, h_pad=1.2)
    name = f"linear_probes_{PAGES[task][0]}"
    fig.savefig(OUT / f"{name}.png", dpi=300)
    fig.savefig(OUT / f"{name}.svg")
    plt.close(fig)
    return f"{name}.png"


def task_page(
    task: str, runs: pd.DataFrame, sizes: pd.DataFrame, features: list[str]
) -> None:
    """Write one task's page: the task, how to load it, its splits, example, baselines."""
    spec = ECOLI_TASKS[task]
    stem, title, label, example, measured = PAGES[task]
    mine = sizes[sizes["task"] == task]
    group = runs[runs["task"] == task]
    n_samples = int(mine["samples"].iloc[0])
    n_excluded = int(mine["excluded_records"].iloc[0])
    pair = spec.kind == "gene_pair"
    first_split = "gene_holdout_s0" if pair else "seqid30_s0"
    page = front_matter(title, label, spec.n_targets_note)
    page += [measured, ""]
    page += table(
        ["Property", "Value"],
        [
            ["Task name", f"`{task}`"],
            ["Dataset class", f"`{spec.dataset_module}.{spec.dataset_class}`"],
            ["Genome", f"`{spec.assembly_set}`"],
            [
                "One sample is",
                "a pair of deleted genes" if pair else "one deleted gene",
            ],
            ["Label", spec.n_targets_note],
            ["Benchmark samples", f"{n_samples:,}"],
            [
                "Records of the dataset set aside",
                f"{n_excluded:,} (listed with a reason in `task.excluded`)",
            ],
            ["Pinned splits", str(len(mine))],
        ],
        left=2,
    )
    if task not in BOARDLESS:
        links = ", ".join(
            f"[{FAMILIES[f].lower()}]({board_url(task, f)})"
            for f in mine["split"]
            .str.replace(r"_s\d+$", "", regex=True)
            .drop_duplicates()
        )
        page += [f"**Leaderboards for this task:** {links}.", ""]
    page += [
        "## Load the task",
        "",
        "```python",
        "from torchcell.tasks import load_task",
        "",
        f'task = load_task("{task}")',
        f'split = task.split("{first_split}")',
        "",
        "task.genes[split.train]  # locus tags of the deleted genes, shape (n, "
        + ("2)" if pair else "1)"),
        "task.y[split.train]      # labels, one column per target",
    ]
    page += [
        "task.genome              # the genome: sequence, protein and annotation of each gene",
        "split.val                # row indices for model selection",
        'split.unlock_test(reason="final evaluation")  # test row indices, with a warning',
        "```",
        "",
        "A sample is a benchmark sample when every deleted gene resolves to one protein-coding gene of the genome. See the [overview](./index.mdx) for what a task object holds and how the test part is locked.",
        "",
        "## Splits",
        "",
        "Each family has several seeds. The name of a split is the family, `_s` and the seed.",
        "",
    ]
    rows = []
    mine = mine.assign(family=mine["split"].str.replace(r"_s\d+$", "", regex=True))
    for family, part in mine.groupby("family", sort=False):

        def span(column: str) -> str:
            low, high = int(part[column].min()), int(part[column].max())
            return f"{low:,}" if low == high else f"{low:,} to {high:,}"

        rows.append(
            [
                FAMILIES[family],
                f"`{family}_s0` to `{family}_s{len(part) - 1}`",
                SPLIT_NOTES[family],
                span("train"),
                span("val"),
                span("test"),
            ]
        )
    page += table(
        ["Family", "Split names", "What is kept whole", "Train", "Validation", "Test"],
        rows,
        left=3,
    )
    page += [
        "## Example script",
        "",
        f"[`{example}.py`]({GITHUB}/{EXAMPLES.as_posix()}/{example}.py) is a complete run in one file: it loads this task, builds 21 features from the protein of each deleted gene, fits a ridge regression, chooses the penalty on validation and scores the test part once. Replace `gene_features` and `ridge` with your model and keep the rest.",
        "",
        "```bash",
        'pip install "torchcell>=1.8.8"',
        f"python {example}.py",
        "```",
        "",
        "```python",
        (EXAMPLES / f"{example}.py").read_text(encoding="utf-8").rstrip("\n"),
        "```",
        "",
        "Output of one run:",
        "",
        "```text",
        Path(RESULTS, "examples", f"{example}.out")
        .read_text(encoding="utf-8")
        .rstrip("\n"),
        "```",
        "",
        "## Linear-probe baselines",
        "",
        "A frozen representation of each deleted gene with a ridge regression on top, for every split of this task. The representations and the fitting procedure are described in the [overview](./index.mdx#linear-probe-baselines).",
        "",
    ]
    image = figure(task, group, features)
    seeds = "five seeds for random splits and three for every other family"
    caption = (
        f"**Test Pearson correlation of each linear probe on {label}.** Each panel is one split family and each bar is one gene representation. "
        f"A bar is the mean over the seeds of that family ({seeds}) and its whisker is the sample standard deviation over those seeds. "
        "The correlation is computed on the test part at the ridge penalty chosen on validation"
        + (
            ", separately for each target, and then averaged over the targets. "
            if not spec.n_targets_note.startswith("1 target")
            else ". "
        )
        + "Random features carry no biology and mark the level a representation must exceed. "
        + (
            "Dark bars concatenate the representations of the two genes and light bars add them. "
            if pair
            else "A representation with no bar has an undefined correlation: one-hot features predict the training mean for a gene that was not in training. "
        )
        + "Panels to the right of Random hold out genes that are increasingly unlike the training genes or far from them on the chromosome."
    )
    page += [
        f'<img src={{require("./{image}").default}} alt="Linear-probe test Pearson for {label}" style={{{{display: "block", width: "100%", maxWidth: "180mm", maxHeight: "170mm", height: "auto", objectFit: "contain"}}}} />',
        "",
        caption,
        "",
        "The table gives the same numbers: mean ± sample standard deviation of the test Pearson correlation over the seeds of each split family. The highest mean of each row is in bold. `undefined` marks a constant prediction.",
        "",
    ]
    for combine, sub in group.groupby("combine", sort=False):
        if combine != "single":
            page += [
                f"Pair representation: the two genes {'concatenated' if combine == 'concat' else 'summed'} (`{combine}`).",
                "",
            ]
        page += result_table(
            sub,
            features,
            "test_pearson",
            [
                (FAMILIES[f], sub[sub["family"] == f])
                for f in sub["family"].drop_duplicates()
            ],
        )
    if pair:
        page += [
            "### Test pairs by how many genes are new",
            "",
            "Under gene holdout a test pair has either one gene that was in no training pair or two. The table scores the two kinds separately.",
            "",
        ]
        holdout = group[group["family"].str.contains("gene_holdout")]
        for combine, sub in holdout.groupby("combine", sort=False):
            page += [f"Pair representation: `{combine}`.", ""]
            for column, kind in (
                ("test_pearson_one_new", "one gene new"),
                ("test_pearson_both_new", "both genes new"),
            ):
                page += [f"Test pairs with {kind}:", ""]
                page += result_table(
                    sub,
                    features,
                    column,
                    [
                        (FAMILIES[f], sub[sub["family"] == f])
                        for f in sub["family"].drop_duplicates()
                    ],
                )
    page += ["## Boards", ""]
    if task in BOARDLESS:
        page += [BOARDLESS[task], ""]
    else:
        families = list(mine["family"].drop_duplicates())
        page += [
            "The [leaderboard](../../leaderboard.mdx) has one board per split family of this task, on the seed-0 split. The linear probes above are its baseline rows. The other seeds are in the package for your own experiments and have no board.",
            "",
        ]
        page += table(
            ["Open the board", "Board name", "Split"],
            [
                [
                    f"[{FAMILIES[f]}]({board_url(task, f)})",
                    f"`{task}-{f.replace('_', '-')}-s0`",
                    f"`{f}_s0`",
                ]
                for f in families
            ],
            left=3,
        )
        if not spec.n_targets_note.startswith("1 target"):
            page += [
                "The board names a target with letters, digits and `_.:|+-` only, so a feature written `<L>` in the dataset is `mean_L` on the board, `<SA/V>` is `mean_SA_per_V` and `%2N` is `pct_2N`.",
                "",
            ]
    at_edge = int((group["alpha"] == group["alpha"].max()).sum())
    page += [
        "### Reading these numbers",
        "",
        f"- This task has {len(group):,} probe runs, one per split, representation"
        + (" and pair combination" if pair else "")
        + f". In {at_edge:,} of them the penalty chosen on validation is the largest value tried (1,000,000), which means the best linear model in that run is close to predicting the training mean.",
        "- Differences smaller than the standard deviations shown are not resolved by three to five seeds.",
        "- These are frozen, mean-pooled representations with a linear readout. They are a floor for models that are fine-tuned, nonlinear or use the cell graph, and they say nothing about what such models reach.",
        "",
        f"Every run, with validation scores, the chosen penalty and four more metrics, is in [`linear_probes.csv`]({GITHUB}/{RESULTS}/linear_probes.csv).",
        "",
    ]
    (OUT / f"{stem}.mdx").write_text("\n".join(page), encoding="utf-8")


def overview(runs: pd.DataFrame, sizes: pd.DataFrame, features: list[str]) -> None:
    """Write the overview page: tasks, loading, the test lock, splits, the baselines."""
    page = front_matter(
        "E. coli tasks",
        "Overview",
        "Four E. coli benchmark tasks, how to load them, their splits and the baseline method.",
    )
    page += [
        "Four benchmark tasks built from four *Escherichia coli* K-12 datasets. Each asks a model to predict a measured phenotype of a deletion strain from the genes that were deleted. Every task has its own page with its splits, an example script and baseline results.",
        "",
    ]
    for task, (stem, title, _, _, _) in PAGES.items():
        mine = sizes[sizes["task"] == task]
        spec = ECOLI_TASKS[task]
        page.append(
            f"- [{title}](./{stem}.mdx): {int(mine['samples'].iloc[0]):,} samples, {spec.n_targets_note.split(':')[0]}, {len(mine)} splits."
        )
    page += [
        "- [Processing notebook](./processing.mdx): how a dataset becomes a task, step by step.",
        "",
        "## Loading a task",
        "",
        'Install with `pip install "torchcell>=1.8.8"`. One function returns a processed task; its argument is the task name.',
        "",
        "```python",
        "from torchcell.tasks import load_task",
        "",
        'growth = load_task("ecoli-growth-rate-campos2018")',
        'morphology = load_task("ecoli-morphology-campos2018")',
        'interactions = load_task("ecoli-gene-interaction-babu2014")',
        'metabolome = load_task("ecoli-metabolome-fuhrer2017")',
        "```",
        "",
        "`load_task` downloads the built dataset store when `TC_DATA_URL` is set, fetches the genome the records name when it is absent, matches each record to genes of that genome, reads the labels and attaches the pinned splits. Every task object has the same parts:",
        "",
        "```python",
        "task = growth",
        "task.samples        # one row per sample: record index, stored identifiers, locus tags",
        "task.genes          # locus tags of the deleted genes, shape (n, 1) or (n, 2)",
        "task.y              # labels, shape (n, n_targets); nan where a target was not measured",
        "task.targets        # name of each label column",
        "task.genome         # the genome; task.genome[locus_tag].protein.seq is a protein",
        "task.excluded       # records of the dataset that are not samples, each with its reason",
        "task.split_names    # names of the pinned splits",
        "",
        'split = task.split("seqid30_s0")',
        "split.train, split.val                     # row indices into task.genes and task.y",
        "task.score(task.y[split.val], prediction)  # Pearson, Spearman, R2, MSE, MAE",
        "```",
        "",
        "The split files ship inside the package and are checked against a sha256 recorded in the code. The samples built on your machine must equal the samples the split file was written for, row for row, or `load_task` raises.",
        "",
        "Because a script reaches the processed task through this one import, a submitted script can be checked for it: a script that builds its own samples or splits does not call `load_task`, and a script that reads the test part calls `unlock_test`.",
        "",
        "## The test part is locked",
        "",
        "`split.train` and `split.val` are plain index arrays. `split.test` raises `TestSplitLocked`. The test indices come from",
        "",
        "```python",
        'test = split.unlock_test(reason="final evaluation of the model selected on validation")',
        "```",
        "",
        "which requires a reason and emits a `TestSplitWarning` on every call. Fit on train, choose the model and every hyperparameter on validation, and unlock the test part once for the final number.",
        "",
        "## Split families",
        "",
        "Train, validation and test hold about 80 %, 10 % and 10 % of the unit that is split, and a unit is never divided. Each task page lists its own splits and their sizes.",
        "",
        "- **Random.** Genes at random. For pairs, pairs at random, with the two orders of a pair and repeated measurements of it kept together.",
        "- **Sequence identity 30 % and 50 %.** Whole protein clusters. A cluster is a connected component of an all-against-all MMseqs2 search at that identity over 80 % of both sequences, so a test protein has no reported hit to a training protein at or above the threshold.",
        "- **Chromosome position.** Whole chromosome blocks (40 contiguous blocks of equal gene count), so the genes of an operon stay on one side.",
        "- **Gene holdout** (pair task). The genes are split first, at random, by cluster or by block. A pair is test when either gene is a test gene, validation when neither is test and either is validation, and train when both are training genes.",
        "",
        "## Linear-probe baselines",
        "",
        "The first baselines are linear probes: a frozen representation of each deleted gene and a ridge regression on top. They measure how much of a label a linear readout of a gene representation recovers under each kind of split.",
        "",
    ]
    page += table(
        ["Representation", "Reads", "Model and pooling", "Numbers per gene"],
        [
            [
                REPRESENTATIONS[f][0],
                REPRESENTATIONS[f][1],
                REPRESENTATIONS[f][2],
                REPRESENTATIONS[f][3],
            ]
            for f in features
        ],
        left=3,
    )
    page += [
        "A single-gene sample is its gene's representation. A gene-pair sample is the two representations concatenated in stored order or added. Features are standardized with the training mean and standard deviation, and one ridge regression is fit per task, split and representation, on all targets at once. The penalty is the one of eight values from 0.1 to 1,000,000 with the lowest validation mean squared error. The test part is unlocked once per split, after that choice. A sequence longer than a model's window is truncated from its end.",
        "",
        "### Best representation per task and split family",
        "",
        "The representation with the highest mean test Pearson correlation in each split family, with the value for random features beside it. Means and sample standard deviations are over the seeds of the family. For the pair task the two genes are concatenated.",
        "",
    ]
    rows = []
    for task, (stem, _, label, _, _) in PAGES.items():
        group = runs[
            (runs["task"] == task) & (runs["combine"].isin(["single", "concat"]))
        ]
        for family in group["family"].drop_duplicates():
            part = group[group["family"] == family]
            stats = {
                f: seed_stats(part.loc[part["feature"] == f, "test_pearson"])
                for f in features
            }
            defined = {f: s for f, s in stats.items() if s[2]}
            best = max(defined, key=lambda f: defined[f][0])
            rows.append(
                [
                    f"[{label}](./{stem}.mdx)",
                    FAMILIES[family],
                    REPRESENTATIONS[best][0],
                    f"{defined[best][0]:.3f} ± {defined[best][1]:.3f}",
                    f"{stats['random'][0]:.3f} ± {stats['random'][1]:.3f}",
                ]
            )
    page += table(
        [
            "Task",
            "Split family",
            "Best representation",
            "Its test Pearson",
            "Random features",
        ],
        rows,
        left=3,
    )
    (OUT / "index.mdx").write_text("\n".join(page), encoding="utf-8")


def main() -> None:
    """Write the combined run table, the overview page, the task pages and the figures."""
    OUT.mkdir(parents=True, exist_ok=True)
    runs = pd.concat(
        [
            pd.read_csv(p)
            for p in sorted(glob.glob(osp.join(RESULTS, "probes", "*.csv")))
        ],
        ignore_index=True,
    )
    runs.to_csv(osp.join(RESULTS, "linear_probes.csv"), index=False)
    features = [f for f in REPRESENTATIONS if f in set(runs["feature"])]
    sizes = pd.read_csv(osp.join(RESULTS, "splits_summary.csv"))
    overview(runs, sizes, features)
    for task in PAGES:
        task_page(task, runs, sizes, features)
    print(f"{len(runs)} runs, {len(features)} representations; wrote {OUT}")


if __name__ == "__main__":
    main()
