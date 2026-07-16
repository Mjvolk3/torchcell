# experiments/database/scripts/render_supported_datasets_table.py
# [[experiments.database.scripts.render_supported_datasets_table]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/database/scripts/render_supported_datasets_table
r"""STEP 2 of the supported-datasets DB report: render VIEWS off the raw-data
JSON (produced by build_supported_datasets_table.py). Reads only the JSON --
never an LMDB -- so re-rendering is instant.

Views written:
  - paper/nature-biotech/sections/datasets_table.tex   (\input into the SI)
  - notes/paper.supported-datasets-and-databases.md      (readable mirror)
  - <pre-build dir>/datasets_table_preview.pdf           (standalone preview, gitignored)
  - notes/assets/pdf-output/paper.supported-datasets-and-databases.pdf  (committed copy)

Run from the repo root (defaults to the newest pre-build snapshot):
  python experiments/database/scripts/render_supported_datasets_table.py
  python experiments/database/scripts/render_supported_datasets_table.py --data <path/to/supported_datasets.json>
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from torchcell.paper.tables import (
    Cell,
    Column,
    DatasetSignalRecord,
    PaperTable,
    Row,
    read_frontmatter,
    scientific,
)

SCRIPT = Path(__file__).resolve()
REPO = SCRIPT.parents[3]
RESULTS = SCRIPT.parent.parent / "results"
TEX_OUT = REPO / "paper" / "nature-biotech" / "sections" / "datasets_table.tex"
MD_OUT = REPO / "notes" / "paper.supported-datasets-and-databases.md"
PDF_OUT = REPO / "notes" / "assets" / "pdf-output" / "paper.supported-datasets-and-databases.pdf"

SIGNAL_HEADER = "Signal (gzip, bytes)"
COLUMNS = [
    Column(header="Dataset", align="l"),
    Column(header="Genotypes", align="r"),
    Column(header="Env", align="r"),
    Column(header="Instances", align="r"),
    Column(header="Phenotype", align="l"),
    Column(header="Shape", align="l"),
    Column(header="Graph role", align="l"),
    Column(header=SIGNAL_HEADER, align="r"),
]

MD_INTRO = """\
## Supported datasets + databases (paper table)

<!-- GENERATED: raw data from experiments/database/scripts/build_supported_datasets_table.py
(-> results/pre-build/<date>/supported_datasets.json); this note + the paper LaTeX are
VIEWS rendered by render_supported_datasets_table.py. Do not edit tables by hand. -->

Pre-build inventory of the datasets schematized + L0-L4 verified as LMDBs (not yet a
versioned Neo4j DB build). For the reader to grasp the **diversity and scale** of the
training signal; the full backlog is `[[paper.north-star.dataset-triage]]`.

**Columns.** *Genotypes* = distinct perturbed strains/isolates (curated). *Env* = number
of environments. *Instances* = dataset length (total genotype×environment records).
*Shape* = shape of a single phenotype instance (`scalar` / `vector (D)`). *Graph role* =
where the label sits in the cell graph (`global` / `node` / `edge` / `hyperedge` /
`bipartite node`; a digenic interaction is an `edge`, a trigenic one a `hyperedge`).
*Signal (gzip, bytes)* = scientific-notation gzip size of the concatenated stored
instances (the **perturbation** + environment + phenotype of every record) -- a
Kolmogorov-complexity proxy. The perturbation counts each instance's edit off the S288C
reference: a single deletion (a few bytes) or a natural isolate's thousands of gene-presence
entries that amount to a new genome (sequence stays external, referenced by uri+sha256). The
shared reference genome is never counted. Instances, Shape, Graph role, and Signal are
**derived from the built LMDB**, not hand-typed.
"""

MD_TAIL = """\
### In progress (not yet built/verified)

{notbuilt}. See `[[paper.north-star.dataset-triage]]` for the full ~75-candidate backlog.

## Reference databases

Curated resources torchcell reads from or cross-references. Not per-strain datasets.

| Database | Type | What it provides | URL |
| :-- | :-- | :-- | :-- |
| SGD | genome / knowledgebase | reference genome, annotation, phenotype + literature curation | yeastgenome.org |
| **SPELL** | expression compendium (SGD) | search engine over 752 datasets / 15,475 arrays / 576 studies | spell.yeastgenome.org |
| YMDB 2.0 | metabolite database | curated yeast metabolite structures, concentrations, pathways | ymdb.ca |
| YeastNet v3 | functional gene network | probabilistic integrated gene-interaction network | inetbio.org/yeastnet |
| CYCLoPs / LoQAtE | localization + abundance | GFP-collection protein localization/abundance atlases | thecellvision / weizmann |
| TheCellMap.org | genetic-interaction portal | query/download for the Costanzo/Boone global GI network | thecellmap.org |
| Yeast9 GEM | genome-scale metabolic model | consensus stoichiometric reconstruction (metabolite-node IDs) | github SysBioChalmers/yeast-GEM |
| ScRAPdb | pan-omics assembly panel | 142-strain telomere-to-telomere reference panel omics | evomicslab.org/db/ScRAPdb |
| Yeast PeptideAtlas | proteome-observation DB | reprocessed MS peptide/protein observation confidence | peptideatlas.org/builds/yeast |

## Maintenance + provenance

- **Two steps, regenerated never hand-edited.** `build_supported_datasets_table.py` scans
  the LMDBs -> `results/pre-build/<date>/supported_datasets.json` (raw data);
  `render_supported_datasets_table.py` renders this note + the paper LaTeX + a preview PDF
  off that JSON; `plot_supported_datasets_signal.py` draws instances-vs-signal. Per-dataset
  spot check: `python -m torchcell.paper.signal <subpath>`.
- **Signal caveat.** The gzip number includes phenotype metadata (SE, n_replicates,
  measurement_type), so it is a *relative* proxy, comparable across datasets, not exact.
- Backlog + differentiation: `[[paper.north-star.dataset-triage]]`, `[[paper.north-star]]`.
"""


def newest_json() -> Path:
    """The most recent pre-build snapshot's JSON."""
    cands = sorted(RESULTS.glob("pre-build/*/supported_datasets.json"))
    if not cands:
        raise FileNotFoundError("No pre-build snapshot; run build_supported_datasets_table.py first")
    return cands[-1]


def signal_cell(rec: DatasetSignalRecord) -> str | Cell:
    """Signal as a scientific-notation cell, or a marker for not-built/deferred."""
    if not rec.built:
        return "not built"
    if rec.signal_bytes <= 0:
        return "pending"  # built, but signal deferred (e.g. a very large LMDB)
    return scientific(rec.signal_bytes)


def totals_footer(records: list[DatasetSignalRecord]) -> Row:
    """A bold totals row summing ONLY the additive columns.

    Instances (total dataset length) and Signal (total gzip bytes) sum cleanly.
    Genotypes/Env are left blank -- summing strain/condition counts across
    datasets double-counts the same genes/conditions -- and Shape/Graph role/
    Phenotype are categorical, so they get no total either.
    """
    built = [r for r in records if r.built]
    n_built = len(built)
    total_instances = sum(r.instances for r in built)
    total_signal = sum(r.signal_bytes for r in built if r.signal_bytes > 0)
    return Row(bold=True, cells={
        "Dataset": f"Total ({n_built} datasets)",
        "Instances": f"{total_instances:,}",
        SIGNAL_HEADER: scientific(total_signal),
    })


def build_table(records: list[DatasetSignalRecord], sections: list[str]) -> PaperTable:
    """Assemble the ``PaperTable`` from the raw-data records."""
    rows: list[Row] = []
    for section in sections:
        for rec in (x for x in records if x.section == section):
            rows.append(Row(section=section, cells={
                "Dataset": rec.name,
                "Genotypes": rec.genotypes,
                "Env": rec.env,
                "Instances": f"{rec.instances:,}" if rec.built else "—",
                "Phenotype": rec.phenotype,
                "Shape": rec.shape,
                "Graph role": rec.graph_role,
                SIGNAL_HEADER: signal_cell(rec),
            }))
    return PaperTable(columns=COLUMNS, rows=rows, footer=totals_footer(records))


def main() -> None:
    """Render the paper .tex, the note .md, and a preview .pdf off the JSON."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=None, help="supported_datasets.json (default: newest snapshot)")
    args = ap.parse_args()

    data_path = args.data or newest_json()
    payload = json.loads(data_path.read_text())
    records = [DatasetSignalRecord(**d) for d in payload["datasets"]]
    table = build_table(records, payload["sections"])
    notbuilt = " · ".join(payload["not_built"])

    footnote = "In scope but not yet built/verified (no signal): " + "; ".join(payload["not_built"]) + "."
    TEX_OUT.write_text(
        table.to_latex(
            caption=(
                f"Supported datasets ({payload['status']}, {payload['date']}) schematized + L0--L4 "
                "verified in the TorchCell database. \\emph{Instances} is the dataset length (total "
                "genotype$\\times$environment records); \\emph{Shape} the shape of one phenotype "
                "instance; \\emph{Graph role} where the label sits in the cell graph; \\emph{Signal "
                "(gzip, bytes)} a Kolmogorov-complexity proxy -- the gzip size in bytes of the "
                "concatenated per-record stored instances (perturbation + environment + phenotype), "
                "so a natural isolate's genome-defining perturbation set counts alongside its "
                "phenotype; the shared reference genome is never counted."
            ),
            label="tab:supported-datasets",
            footnote=footnote,
            header_comment=(
                "GENERATED off experiments/database/results/pre-build/<date>/supported_datasets.json "
                "by render_supported_datasets_table.py -- do not edit by hand."
            ),
        )
    )
    print(f"Wrote {TEX_OUT.relative_to(REPO)}")

    front = read_frontmatter(MD_OUT, default_title="Supported Datasets and Databases")
    note = "\n".join([front.rstrip("\n"), "", MD_INTRO, table.to_markdown(sectioned=True),
                      "", MD_TAIL.format(notbuilt=notbuilt)])
    MD_OUT.write_text(note.rstrip("\n") + "\n")
    print(f"Wrote {MD_OUT.relative_to(REPO)}")

    n_lines = len(records) + len(payload["sections"])  # data rows + section headers
    _preview_pdf(data_path.parent, n_lines)


def _preview_pdf(out_dir: Path, n_lines: int) -> None:
    r"""Wrap the fragment in a standalone doc and compile a preview PDF (pdflatex).

    A ``table*`` taller than ``\\textheight`` does not paginate -- it runs off the
    bottom of the page silently. So the page HEIGHT is sized to the row count (data
    rows + section headers + caption/header/totals slack); width stays ~A4 landscape.
    This is a "see every row at once" canvas, not the journal layout.
    """
    height_in = 3.0 + n_lines * 0.17  # ~caption+header+totals slack + per-line
    wrapper = out_dir / "_preview.tex"
    wrapper.write_text(
        "\\documentclass[9pt]{article}\n"
        f"\\usepackage[paperwidth=11.7in,paperheight={height_in:.1f}in,margin=0.4in]{{geometry}}\n"
        "\\usepackage[T1]{fontenc}\\usepackage{lmodern}\\usepackage{booktabs}\n"
        "\\usepackage{amsmath,amssymb}\\usepackage{textcomp}\n"
        "\\pagestyle{empty}\\twocolumn\n"
        f"\\begin{{document}}\\input{{{TEX_OUT}}}\\end{{document}}\n"
    )
    for _ in range(2):
        subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
                        "-output-directory", str(out_dir), str(wrapper)],
                       cwd=out_dir, capture_output=True, check=False)
    pdf = out_dir / "_preview.pdf"
    if pdf.exists():
        final = out_dir / "datasets_table_preview.pdf"
        pdf.rename(final)
        print(f"Wrote {final.relative_to(REPO)}")
        # also publish a committed copy next to the note it mirrors, so the table PDF
        # has a stable home instead of only the gitignored pre-build snapshot.
        PDF_OUT.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(final, PDF_OUT)
        print(f"Wrote {PDF_OUT.relative_to(REPO)}")
    else:
        print("(preview PDF not built -- pdflatex missing?)")


if __name__ == "__main__":
    main()
