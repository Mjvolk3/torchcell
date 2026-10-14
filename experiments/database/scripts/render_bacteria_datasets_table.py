# experiments/database/scripts/render_bacteria_datasets_table.py
# [[experiments.database.scripts.render_bacteria_datasets_table]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/database/scripts/render_bacteria_datasets_table
r"""Render the bacteria datasets table and the Signal-controls table off the JSON
written by build_bacteria_datasets_table.py. Reads only the JSON, never an LMDB.

Views written:
  - notes/experiments.database.<view>-datasets-table.md  (generated note; frontmatter kept; view = bacteria | yeast)
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


def md_out(name: str) -> Path:
    """The generated note for one organism set."""
    return REPO / "notes" / f"experiments.database.{name}-datasets-table.md"


def pdf_out(name: str) -> Path:
    """The committed preview PDF for one organism set."""
    return (
        REPO
        / "notes"
        / "assets"
        / "pdf-output"
        / f"experiments.database.{name}-datasets-table.pdf"
    )


BITS_PER_BYTE = 8
SIGNAL = "Signal (gzip, bits)"
ORGANISM_LABEL = {
    "ecoli": "E. coli",
    "pputida": "P. putida",
    "scerevisiae": "S. cerevisiae",
}

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
[[experiments.database.scripts.build_bacteria_datasets_table]].

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
                        "Host": ORGANISM_LABEL[r.organism],
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
        title=name.title(), hosts=hosts, organisms=", ".join(payload["organisms"])
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


def _preview_pdf(out_dir: Path, tex_out: Path, pdf_path: Path, n_lines: int) -> None:
    r"""Standalone pdflatex canvas sized to the row count (a ``table*`` never paginates)."""
    height_in = 4.0 + n_lines * 0.17
    wrapper = out_dir / f"_preview_{tex_out.stem}.tex"
    wrapper.write_text(
        "\\documentclass[9pt]{article}\n"
        f"\\usepackage[paperwidth=14in,paperheight={height_in:.1f}in,margin=0.4in]{{geometry}}\n"
        "\\usepackage[T1]{fontenc}\\usepackage{lmodern}\\usepackage{booktabs}\n"
        "\\usepackage{amsmath,amssymb}\\usepackage{textcomp}\n"
        "\\pagestyle{empty}\n"
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
