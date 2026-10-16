# experiments/database/scripts/render_bacteria_datasets_table.py
# [[experiments.database.scripts.render_bacteria_datasets_table]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/database/scripts/render_bacteria_datasets_table
r"""Render the bacteria datasets table and the Signal-controls table off the JSON
written by build_bacteria_datasets_table.py. Reads only the JSON, never an LMDB.

Views written:
  - notes/experiments.database.<view>-datasets-table.md  (generated note; frontmatter kept; view = bacteria | yeast)
  - notes/experiments.database.<view>-datasets-table-expanded.md  (the main table plus a
    Description column, the summary line of each dataset class's docstring), with its own
    .tex fragment and committed PDF beside the ones below
  - <pre-build dir>/<view>_datasets_table.tex                      (both tables, \input-able)
  - <pre-build dir>/<view>_datasets_table_preview.pdf              (standalone preview, gitignored)
  - notes/assets/pdf-output/experiments.database.<view>-datasets-table.pdf  (committed copy)

Run from the repo root (defaults to the newest pre-build snapshot):
  python experiments/database/scripts/render_bacteria_datasets_table.py
  python experiments/database/scripts/render_bacteria_datasets_table.py --data <path/to/ecoli-pputida_datasets.json>
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path

from build_bacteria_datasets_table import DatasetProbeRecord

from torchcell.paper.tables import (
    Cell,
    Column,
    PaperTable,
    Row,
    read_frontmatter,
    scientific,
    tex_escape,
)

SCRIPT = Path(__file__).resolve()
REPO = SCRIPT.parents[3]
RESULTS = SCRIPT.parent.parent / "results"
# Output names follow the organism set of the JSON: the bacteria pair renders to the
# bacteria note, a yeast run to its own note, any other set to its raw joined label.
VIEW_NAME = {"ecoli-pputida": "bacteria", "scerevisiae": "yeast"}


def view_name(organisms: list[str]) -> str:
    """``["ecoli", "pputida"]`` -> ``bacteria``; the raw joined label otherwise."""
    label = "-".join(organisms)
    return VIEW_NAME.get(label, label)


def md_out(name: str, suffix: str = "") -> Path:
    """The generated note for one organism set (``suffix="-expanded"`` for that view)."""
    return REPO / "notes" / f"experiments.database.{name}-datasets-table{suffix}.md"


def pdf_out(name: str, suffix: str = "") -> Path:
    """The committed preview PDF for one organism set."""
    return (
        REPO
        / "notes"
        / "assets"
        / "pdf-output"
        / f"experiments.database.{name}-datasets-table{suffix}.pdf"
    )


BITS_PER_BYTE = 8
SIGNAL = "Signal (gzip, bits)"
ORGANISM_LABEL = {
    "ecoli": "E. coli",
    "pputida": "P. putida",
    "scerevisiae": "S. cerevisiae",
}
# Species names are italic in both targets, in the Host column and inside the
# descriptions, where they are written plainly so one rule italicizes them everywhere.
ITALIC_SPECIES = (
    "S. cerevisiae",
    "E. coli",
    "P. putida",
    "Saccharomyces cerevisiae",
    "Escherichia coli",
    "Pseudomonas putida",
)
_SPECIES_RE = re.compile("|".join(re.escape(s) for s in ITALIC_SPECIES))


def italic_species(text: str) -> Cell:
    r"""A cell whose species names are italic: ``*...*`` in markdown, ``\emph`` in LaTeX."""
    md = _SPECIES_RE.sub(lambda m: f"*{m.group(0)}*", text)
    tex = _SPECIES_RE.sub(lambda m: rf"\emph{{{m.group(0)}}}", tex_escape(text))
    return Cell(md=md, tex=tex)


MAIN_COLUMNS = [
    Column(header="Dataset", align="l"),
    Column(header="Host", align="l"),
    Column(header="Genotypes", align="r"),
    Column(header="Env", align="r"),
    Column(header="Instances", align="r"),
    Column(header="Phenotype", align="l"),
    Column(header="Shape", align="l"),
    Column(header="Graph role", align="l"),
    Column(header=SIGNAL, align="r"),
]

# Wrapping widths for the expanded view's two free-text columns. The characters per
# wrapped line at \footnotesize only size the standalone preview canvas.
DESCRIPTION_WIDTH = "4.6in"
DESCRIPTION_CHARS_PER_LINE = 80
PHENOTYPE_WIDTH = "1.5in"
PHENOTYPE_CHARS_PER_LINE = 26
SEQ_UNIT_WIDTH = "1.5in"
SEQ_UNIT_CHARS_PER_LINE = 26
SEQ_FIT = "Sequence-only fit"
EXPANDED_COLUMNS = [
    MAIN_COLUMNS[0],
    MAIN_COLUMNS[1],
    Column(header="Description", align="l", width=DESCRIPTION_WIDTH),
    MAIN_COLUMNS[2],
    MAIN_COLUMNS[3],
    MAIN_COLUMNS[4],
    Column(header="Phenotype", align="l", width=PHENOTYPE_WIDTH),
    *MAIN_COLUMNS[6:],
    Column(header=SEQ_FIT, align="l"),
    Column(header="Label unit", align="l", width=SEQ_UNIT_WIDTH),
]

# The tiers of build_bacteria_datasets_table.sequence_fit, in table order.
SEQ_FIT_LABEL = {"best": "best", "good": "good", "weak": "weak", "none": "no"}

CONTROL_COLUMNS = [
    Column(header="Dataset", align="l"),
    Column(header="Instances", align="r"),
    Column(header="Numeric leaves", align="r"),
    Column(header=SIGNAL, align="r"),
    Column(header="Structure (bits)", align="r"),
    Column(header="Shuffled (bits)", align="r"),
    Column(header="Noise (bits)", align="r"),
    Column(header="Binary (bits)", align="r"),
    Column(header="Packed 1-bit (bits)", align="r"),
    Column(header="Structure/Signal", align="r"),
    Column(header="Signal/Noise", align="r"),
    Column(header="Packed/Signal", align="r"),
]

MD_INTRO = """\
## {title} datasets: Instances vs Signal (gzip), with the Signal controls

<!-- GENERATED: raw data from experiments/database/scripts/build_bacteria_datasets_table.py
(-> results/pre-build/<date>/<organisms>_datasets.json); this note is a VIEW rendered by
render_bacteria_datasets_table.py. Do not edit tables by hand. -->

Pre-build inventory of the {hosts} datasets built as LMDBs in the dev tree
(the set `build_adapter_map(organisms={{{organisms}}})` serves), a temporary counterpart of
[[paper.supported-datasets-and-databases]] (issue #1098), with the Signal-column controls of
issue #1099 measured in the same pass. Commentary and findings:
[[experiments.database.scripts.build_bacteria_datasets_table]]. The same table with a
description of each dataset: [[experiments.database.{name}-datasets-table-expanded]].

**Columns.** Every column is read from the built LMDB; nothing is hand-typed. *Genotypes* =
distinct stored genotype payloads; *Env* = distinct stored environment payloads; *Instances* =
record count; *Phenotype* = the stored `label_name` (and `measurement_type` when set); *Shape* =
shape of one phenotype instance; *Graph role* = where the label sits in the cell graph;
*Signal (gzip, bits)* = gzip size, in bits, of the concatenated stored instances (perturbation +
environment + phenotype, sorted keys), the yeast table's definition.

**Controls** (second table). *Structure* = the same serialization with every leaf replaced by a
type token (keys, nesting and list lengths kept). *Shuffled* = numeric leaves permuted within
their key path across the dataset (marginals kept, joint structure destroyed). *Noise* = every
digit of every numeric leaf redrawn uniformly at the stored precision (the ceiling for that
structure). *Binary* = numeric leaves z-scored per key path and thresholded at 0, written as
`1`/`0` in place. *Packed 1-bit* = gzip of just those bits, 8 per byte, no structure (the one-bit
floor on value-carried information). Ratios are dimensionless.

**Caveat.** Several bacterial stores intern the environment as a `$ref` hash plus a name rather
than the full media payload, so their environment bytes are a 64-character digest; the yeast
2026-07-17 snapshot predates interning. Signal is a relative proxy, comparable within this
table, not an absolute information content.
"""


EXPANDED_MD_INTRO = """\
## {title} datasets, expanded: the main table with a description of each dataset

<!-- GENERATED: raw data from experiments/database/scripts/build_bacteria_datasets_table.py
(-> results/pre-build/<date>/<organisms>_datasets.json); this note is a VIEW rendered by
render_bacteria_datasets_table.py. Do not edit tables by hand. -->

The table of [[experiments.database.{name}-datasets-table]] with three more columns.

*Description* is two to three sentences per dataset from
`experiments/database/scripts/dataset_descriptions.json`, written from the loader module
and its Dendron note: what was measured and with what technology, what the strains are,
and what one stored record is. The build script refuses to write a snapshot with a
served dataset missing from that file, so a better description is made by editing it and
rerunning both scripts.

*Sequence-only fit* says how well a model that sees only sequence embeddings, with no
cell graph, fits the dataset, and *Label unit* names what one label attaches to once the
graph is dropped. Both are derived by `sequence_fit` in the build script from the
genotype of 400 evenly spaced records per store, never typed by hand. *best* = the label
sits on one gene, one gene pair or triple, or one inserted construct, so a per-gene or
per-pair sequence representation is the whole input. *good* = the label needs the
readout entity crossed with the perturbed gene(s) or with the condition, which a pair of
embeddings still carries. *weak* = a multi-edit strain, where only the heterologous genes
have a sequence to stand on. *no* = a natural genome (a segregant or a wild isolate) or a
whole-strain readout with no single edit to attach a sequence to.

Every other column is read from the built LMDB; definitions and the Signal controls are
in the non-expanded note. Commentary:
[[experiments.database.scripts.build_bacteria_datasets_table]].
"""


def bits(n: int) -> Cell | str:
    return scientific(n * BITS_PER_BYTE) if n > 0 else "pending"


def ratio(num: int, den: int) -> str:
    return f"{num / den:.2f}" if den > 0 else "—"


def phenotype_text(r: DatasetProbeRecord) -> str:
    mt = r.measurement_type
    return r.label_name if mt in ("None", "") else f"{r.label_name} ({mt})"


def short_name(r: DatasetProbeRecord) -> str:
    """``EnvChemgenNichols2011Dataset`` -> ``EnvChemgenNichols2011``."""
    return r.dataset_class.removesuffix("Dataset")


def main_table(records: list[DatasetProbeRecord], sections: list[str]) -> PaperTable:
    rows: list[Row] = []
    for section in sections:
        for r in (x for x in records if x.section == section):
            rows.append(
                Row(
                    section=section,
                    cells={
                        "Dataset": short_name(r),
                        "Host": italic_species(ORGANISM_LABEL[r.organism]),
                        "Genotypes": f"{r.genotypes:,}" if r.probed else "—",
                        "Env": f"{r.environments:,}" if r.probed else "—",
                        "Instances": f"{r.instances:,}",
                        "Phenotype": phenotype_text(r),
                        "Shape": r.shape,
                        "Graph role": r.graph_role,
                        SIGNAL: bits(r.signal_bytes),
                    },
                )
            )
    probed = [r for r in records if r.probed]
    footer = Row(
        bold=True,
        cells={
            "Dataset": f"Total ({len(records)} datasets, {len(probed)} probed)",
            "Instances": f"{sum(r.instances for r in records):,}",
            SIGNAL: scientific(sum(r.signal_bytes for r in probed) * BITS_PER_BYTE),
        },
    )
    return PaperTable(columns=MAIN_COLUMNS, rows=rows, footer=footer)


def breakable_phenotype(r: DatasetProbeRecord) -> Cell:
    """Phenotype text whose LaTeX may break after each underscore.

    A wrapping column can then split a long snake_case measurement type; markdown is
    unchanged.
    """
    text = phenotype_text(r)
    return Cell(md=text, tex=tex_escape(text).replace(r"\_", r"\_\allowbreak{}"))


def expanded_table(
    records: list[DatasetProbeRecord], sections: list[str]
) -> PaperTable:
    """The main table with each row's Description inserted after Host."""
    main_t = main_table(records, sections)
    by_name = {short_name(r): r for r in records}
    rows = [
        row.model_copy(
            update={
                "cells": {
                    **row.cells,
                    "Description": italic_species(
                        by_name[str(row.cells["Dataset"])].description
                    ),
                    "Phenotype": breakable_phenotype(
                        by_name[str(row.cells["Dataset"])]
                    ),
                    SEQ_FIT: SEQ_FIT_LABEL[by_name[str(row.cells["Dataset"])].seq_fit],
                    "Label unit": by_name[str(row.cells["Dataset"])].seq_unit,
                }
            }
        )
        for row in main_t.rows
    ]
    return PaperTable(columns=EXPANDED_COLUMNS, rows=rows, footer=main_t.footer)


SEQ_TALLY_COLUMNS = [
    Column(header="Sequence-only fit", align="l"),
    Column(header="Datasets", align="r"),
    Column(header="Instances", align="r"),
    Column(header="What the tier means", align="l", width="3.4in"),
]

SEQ_TIER_MEANING = {
    "best": "the label sits on one gene, a gene pair or triple, or one inserted construct",
    "good": "readout entity x perturbed gene(s), or gene x condition",
    "weak": "a multi-edit strain, only its heterologous genes carry a sequence",
    "none": "a natural genome, or a whole-strain readout with no single edit",
}


def seq_fit_tally(records: list[DatasetProbeRecord]) -> PaperTable:
    """Datasets and instances per sequence-only fit tier."""
    rows = [
        Row(
            cells={
                "Sequence-only fit": SEQ_FIT_LABEL[tier],
                "Datasets": f"{len(group):,}",
                "Instances": f"{sum(r.instances for r in group):,}",
                "What the tier means": SEQ_TIER_MEANING[tier],
            }
        )
        for tier in ("best", "good", "weak", "none")
        if (group := [r for r in records if r.seq_fit == tier])
    ]
    footer = Row(
        bold=True,
        cells={
            "Sequence-only fit": "Total",
            "Datasets": f"{len(records):,}",
            "Instances": f"{sum(r.instances for r in records):,}",
        },
    )
    return PaperTable(columns=SEQ_TALLY_COLUMNS, rows=rows, footer=footer)


def control_table(records: list[DatasetProbeRecord], sections: list[str]) -> PaperTable:
    rows: list[Row] = []
    for section in sections:
        for r in (x for x in records if x.section == section and x.probed):
            rows.append(
                Row(
                    section=section,
                    cells={
                        "Dataset": short_name(r),
                        "Instances": f"{r.instances:,}",
                        "Numeric leaves": f"{r.numeric_leaves:,}",
                        SIGNAL: bits(r.signal_bytes),
                        "Structure (bits)": bits(r.structure_bytes),
                        "Shuffled (bits)": bits(r.shuffled_bytes),
                        "Noise (bits)": bits(r.noise_bytes),
                        "Binary (bits)": bits(r.binary_bytes),
                        "Packed 1-bit (bits)": bits(r.binary_packed_bytes),
                        "Structure/Signal": ratio(r.structure_bytes, r.signal_bytes),
                        "Signal/Noise": ratio(r.signal_bytes, r.noise_bytes),
                        "Packed/Signal": ratio(r.binary_packed_bytes, r.signal_bytes),
                    },
                )
            )
    probed = [r for r in records if r.probed]
    tot = {
        k: sum(getattr(r, k) for r in probed)
        for k in (
            "signal_bytes",
            "structure_bytes",
            "shuffled_bytes",
            "noise_bytes",
            "binary_bytes",
            "binary_packed_bytes",
        )
    }
    footer = Row(
        bold=True,
        cells={
            "Dataset": f"Total ({len(probed)} probed)",
            "Instances": f"{sum(r.instances for r in probed):,}",
            "Numeric leaves": f"{sum(r.numeric_leaves for r in probed):,}",
            SIGNAL: bits(tot["signal_bytes"]),
            "Structure (bits)": bits(tot["structure_bytes"]),
            "Shuffled (bits)": bits(tot["shuffled_bytes"]),
            "Noise (bits)": bits(tot["noise_bytes"]),
            "Binary (bits)": bits(tot["binary_bytes"]),
            "Packed 1-bit (bits)": bits(tot["binary_packed_bytes"]),
            "Structure/Signal": ratio(tot["structure_bytes"], tot["signal_bytes"]),
            "Signal/Noise": ratio(tot["signal_bytes"], tot["noise_bytes"]),
            "Packed/Signal": ratio(tot["binary_packed_bytes"], tot["signal_bytes"]),
        },
    )
    return PaperTable(columns=CONTROL_COLUMNS, rows=rows, footer=footer)


def newest_json() -> Path:
    cands = sorted(RESULTS.glob("pre-build/*/ecoli-pputida_datasets.json"))
    if not cands:
        raise FileNotFoundError(
            "No pre-build snapshot; run build_bacteria_datasets_table.py first"
        )
    return cands[-1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=None)
    args = ap.parse_args()
    data_path = (args.data or newest_json()).resolve()
    payload = json.loads(data_path.read_text())
    records = [DatasetProbeRecord(**d) for d in payload["datasets"]]
    sections = payload["sections"]
    hosts = " + ".join(ORGANISM_LABEL[o] for o in payload["organisms"])
    name = view_name(payload["organisms"])
    main_t = main_table(records, sections)
    ctrl_t = control_table(records, sections)

    tex = (
        main_t.to_latex(
            caption=(
                f"{hosts} datasets ({payload['status']}, {payload['date']}; {hosts}) built as LMDBs in the "
                "TorchCell dev tree. \\emph{Genotypes} and \\emph{Env} count distinct stored genotype and "
                "environment payloads; \\emph{Instances} is the record count; \\emph{Signal (gzip, bits)} the "
                "gzip size in bits of the concatenated stored instances (perturbation + environment + "
                "phenotype), as in the yeast table."
            ),
            label=f"tab:{name}-datasets",
            header_comment=f"GENERATED off {data_path.relative_to(REPO)} by render_bacteria_datasets_table.py -- do not edit by hand.",
        )
        + "\n"
        + ctrl_t.to_latex(
            caption=(
                "Signal controls, same pass. \\emph{Structure}: leaves replaced by type tokens. "
                "\\emph{Shuffled}: numeric leaves permuted within key path. \\emph{Noise}: digits of numeric "
                "leaves redrawn uniformly at stored precision. \\emph{Binary}: per-path z-score thresholded at "
                "0, written in place. \\emph{Packed 1-bit}: gzip of those bits alone, 8 per byte."
            ),
            label=f"tab:{name}-signal-controls",
            colsep_pt=3,
            size="scriptsize",
        )
    )
    tex_out = data_path.parent / f"{name}_datasets_table.tex"
    tex_out.write_text(tex)
    print(f"Wrote {tex_out.relative_to(REPO)}")

    front = read_frontmatter(
        md_out(name), default_title=f"{name.title()} Datasets Table"
    )
    intro = MD_INTRO.format(
        title=name.title(),
        hosts=hosts,
        organisms=", ".join(payload["organisms"]),
        name=name,
    )
    note = "\n".join(
        [
            front.rstrip("\n"),
            "",
            intro,
            "### Datasets",
            "",
            main_t.to_markdown(sectioned=True, heading_level=4),
            "",
            "### Signal controls",
            "",
            ctrl_t.to_markdown(sectioned=True, heading_level=4),
            "",
            f"Source JSON: `{data_path.relative_to(REPO)}` (probe variant "
            f"`{payload['probe_variant']}`, gzip level {payload['gzip_level']}, seed {payload['seed']}).",
        ]
    )
    md_out(name).write_text(note.rstrip("\n") + "\n")
    print(f"Wrote {md_out(name).relative_to(REPO)}")
    _preview_pdf(
        data_path.parent, tex_out, pdf_out(name), len(records) * 2 + len(sections) * 2
    )
    _render_expanded(data_path, payload, records, sections, hosts, name)


def _render_expanded(
    data_path: Path,
    payload: dict,
    records: list[DatasetProbeRecord],
    sections: list[str],
    hosts: str,
    name: str,
) -> None:
    """The expanded view: note, .tex fragment and committed PDF."""
    missing = [r.dataset_class for r in records if not r.description]
    if missing:
        raise ValueError(
            f"rows without a description; rerun build_bacteria_datasets_table.py: {missing}"
        )
    table = expanded_table(records, sections)
    tex = table.to_latex(
        caption=(
            f"{hosts} datasets ({payload['status']}, {payload['date']}) built as LMDBs in "
            "the TorchCell dev tree, with a description of each. \\emph{Description} is "
            "written from the loader module and its note. \\emph{Sequence-only fit} and "
            "\\emph{Label unit} are derived from the genotype of 400 evenly spaced records "
            "per store: how well a sequence-embedding-only model fits the dataset, and "
            "what one label attaches to without the cell graph. The other columns are as "
            "in the non-expanded table."
        ),
        label=f"tab:{name}-datasets-expanded",
        header_comment=f"GENERATED off {data_path.relative_to(REPO)} by render_bacteria_datasets_table.py -- do not edit by hand.",
    )
    tex_out = data_path.parent / f"{name}_datasets_table_expanded.tex"
    tex_out.write_text(tex)
    print(f"Wrote {tex_out.relative_to(REPO)}")
    suffix = "-expanded"
    front = read_frontmatter(
        md_out(name, suffix), default_title=f"{name.title()} Datasets Table Expanded"
    )
    note = "\n".join(
        [
            front.rstrip("\n"),
            "",
            EXPANDED_MD_INTRO.format(title=name.title(), name=name),
            "### Datasets",
            "",
            table.to_markdown(sectioned=True, heading_level=4),
            "",
            "### Sequence-only fit, summed over the table",
            "",
            seq_fit_tally(records).to_markdown(sectioned=False),
            "",
            f"Source JSON: `{data_path.relative_to(REPO)}`.",
        ]
    )
    md_out(name, suffix).write_text(note.rstrip("\n") + "\n")
    print(f"Wrote {md_out(name, suffix).relative_to(REPO)}")
    n_lines = sum(
        max(
            -(-len(r.description) // DESCRIPTION_CHARS_PER_LINE),
            -(-len(phenotype_text(r)) // PHENOTYPE_CHARS_PER_LINE),
            -(-len(r.seq_unit) // SEQ_UNIT_CHARS_PER_LINE),
        )
        for r in records
    )
    # Measured on the 2026-10-15 expanded render (the wide Description column): the
    # estimate above over-counts lines slightly, so a line costs 0.119 in of canvas.
    _preview_pdf(
        data_path.parent,
        tex_out,
        pdf_out(name, suffix),
        n_lines + len(sections) * 2,
        base_in=2.5,
        line_in=0.119,
        width_in=17.5,
    )


def _preview_pdf(
    out_dir: Path,
    tex_out: Path,
    pdf_path: Path,
    n_lines: int,
    *,
    base_in: float = 4.0,
    line_in: float = 0.17,
    width_in: float = 14.0,
) -> None:
    r"""Standalone pdflatex canvas sized to the row count (a ``table*`` never paginates)."""
    height_in = base_in + n_lines * line_in
    wrapper = out_dir / f"_preview_{tex_out.stem}.tex"
    wrapper.write_text(
        "\\documentclass[9pt]{article}\n"
        f"\\usepackage[paperwidth={width_in:.1f}in,paperheight={height_in:.1f}in,margin=0.4in]{{geometry}}\n"
        "\\usepackage[T1]{fontenc}\\usepackage{lmodern}\\usepackage{booktabs}\n"
        "\\usepackage{amsmath,amssymb}\\usepackage{textcomp}\\usepackage{array}\n"
        "\\pagestyle{empty}\\makeatletter\\setlength{\\@fptop}{0pt}\\makeatother\n"
        f"\\begin{{document}}\\input{{{tex_out}}}\\end{{document}}\n"
    )
    for _ in range(2):
        subprocess.run(
            [
                "pdflatex",
                "-interaction=nonstopmode",
                "-halt-on-error",
                "-output-directory",
                str(out_dir),
                str(wrapper),
            ],
            cwd=out_dir,
            capture_output=True,
            check=False,
        )
    pdf = out_dir / f"_preview_{tex_out.stem}.pdf"
    if not pdf.exists():
        raise RuntimeError("preview PDF not built (pdflatex log in the pre-build dir)")
    final = out_dir / f"{tex_out.stem}_preview.pdf"
    pdf.rename(final)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(final, pdf_path)
    print(f"Wrote {final.relative_to(REPO)}\nWrote {pdf_path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
