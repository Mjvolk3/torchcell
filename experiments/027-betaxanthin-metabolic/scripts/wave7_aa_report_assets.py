# experiments/027-betaxanthin-metabolic/scripts/wave7_aa_report_assets.py
# [[experiments.027-betaxanthin-metabolic.scripts.wave7_aa_report_assets]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/027-betaxanthin-metabolic/scripts/wave7_aa_report_assets.py

r"""Turn the wave 7 amino-acid round into the two artifacts a notes-tex document needs.

``analyze_wave7_aa.py`` writes the numbers and a timestamped figure. A document cannot
reference either directly: a LaTeX table has to be emitted, and ``make plots`` matches a
figure by a STABLE basename, so a timestamped render is invisible to it. This script does
those two things and nothing else. It computes no statistic of its own.

Emits

``notes-tex/027-metabolic-flux-module/tables/wave7_aa_contrasts.tex``
    One row per paired contrast: the two arms, the seeds paired, the mean difference and
    its across-seed standard deviation.
``notes/assets/images/027-betaxanthin-metabolic/wave7_aa_concentration.{svg,png}``
    The timestamped render named in ``wave7_aa_summary.json``'s ``figure_path``, copied to
    a stable name. The copy is a copy, byte for byte, and the source name and its sha256
    are recorded in the emitted JSON so the stable file can always be traced back to the
    run that drew it.
``experiments/027-betaxanthin-metabolic/results/wave7_aa_report_assets.json``
    What was written, from what, with hashes.

A contrast named in ``CONTRASTS`` and absent from the summary raises, and so does a
contrast whose pair count is zero: a document must not carry a row for a comparison that
was never run.

Run::

    PYTHONPATH=<repo> ~/miniconda3/envs/torchcell/bin/python \
        experiments/027-betaxanthin-metabolic/scripts/wave7_aa_report_assets.py
"""

from __future__ import annotations

import hashlib
import json
import os
import os.path as osp
import shutil
from datetime import UTC, datetime
from typing import cast

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict

load_dotenv()
ASSET_IMAGES_DIR = cast(str, os.getenv("ASSET_IMAGES_DIR"))
EXPERIMENT_ROOT = cast(str, os.getenv("EXPERIMENT_ROOT"))
REPO = osp.dirname(EXPERIMENT_ROOT)
RESULTS = osp.join(EXPERIMENT_ROOT, "027-betaxanthin-metabolic", "results")
IMAGE_DIR = osp.join(ASSET_IMAGES_DIR, "027-betaxanthin-metabolic")
TABLE_DIR = osp.join(REPO, "notes-tex", "027-metabolic-flux-module", "tables")

SUMMARY_JSON = osp.join(RESULTS, "wave7_aa_summary.json")
ASSETS_JSON = osp.join(RESULTS, "wave7_aa_report_assets.json")
STABLE_STEM = "wave7_aa_concentration"


class ContrastRow(BaseModel):
    """The prose half of one row. The numbers come from the summary."""

    model_config = ConfigDict(extra="forbid")

    #: Key in ``wave7_aa_summary.json``'s ``contrasts`` block.
    key: str
    #: What the contrast asks, in the document's words.
    question: str


#: The contrasts the document reports, in the order it reports them. Every one is at the
#: 200-epoch budget, which the summary names as its primary.
CONTRASTS: list[ContrastRow] = [
    ContrastRow(
        key="e200/conc_vs_pooled",
        question="concentration readout against the pooled control",
    ),
    ContrastRow(
        key="e200/conc_nocstr_vs_pooled", question="the same at constraint scale 0"
    ),
    ContrastRow(
        key="e200/turnover_vs_pooled",
        question="wave 6's turnover readout against the same control",
    ),
    ContrastRow(
        key="e200/conc_vs_turnover",
        question="concentration against turnover, matched seeds",
    ),
    ContrastRow(
        key="e200/conc_vs_conc_nocstr",
        question="what the constraint costs the concentration readout",
    ),
    ContrastRow(
        key="e200/pooledq_vs_pooled", question="pinball against CRPS on the control"
    ),
]


class EmittedAsset(BaseModel):
    """One file this script wrote, and what it was written from."""

    model_config = ConfigDict(extra="forbid")

    path: str
    sha256: str
    source_path: str | None = None
    source_sha256: str | None = None


class AssetReport(BaseModel):
    """Everything this run emitted."""

    model_config = ConfigDict(extra="forbid")

    built_at: str
    summary_json: str
    summary_sha256: str
    summary_generated_at: str
    primary_budget: str
    contrasts_emitted: list[str]
    assets: list[EmittedAsset]


def _sha256(path: str) -> str:
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def _read(path: str) -> dict[str, object]:
    with open(path) as handle:
        return cast(dict[str, object], json.load(handle))


def _contrast_table(summary: dict[str, object]) -> tuple[str, list[str]]:
    contrasts = cast(dict[str, dict[str, object]], summary["contrasts"])
    lines = [
        "%% notes-tex/027-metabolic-flux-module/tables/wave7_aa_contrasts.tex",
        "%% AUTO-GENERATED -- do not hand-edit. Regenerate with",
        "%%   PYTHONPATH=. python experiments/027-betaxanthin-metabolic/scripts/"
        "wave7_aa_report_assets.py",
        "%% SOURCE: experiments/027-betaxanthin-metabolic/results/wave7_aa_summary.json,"
        " contrasts",
        "\\begin{table}[H]",
        "  \\centering",
        "  \\footnotesize",
        "  \\caption[Wave 7 amino-acid contrasts]{Every paired contrast of the wave 7"
        " amino-acid round, at the 200-epoch budget. The statistic is the mean per-feature"
        " Spearman over Mulleder's 19 amino acids on the pinned Merzbacher test split, at"
        " the epoch a centered five-epoch moving average of validation Pearson selects,"
        " never a maximum over epochs. A contrast is PAIRED on the seed, so the difference"
        " column is the mean of the per-seed differences and its standard deviation is"
        " across seeds rather than across cells. Two contrasts pair only two seeds because"
        " the turnover arm ran two; at $n = 2$ or $3$ a standard deviation is an estimate"
        " with one or two degrees of freedom, so read the sign and the magnitude and not a"
        " $p$ value.}",
        "  \\label{tab:wave7-aa}",
        "  \\begin{tabular}{@{}p{0.40\\textwidth}llrr@{}}",
        "    \\toprule",
        "    Contrast & Arms & $n$ & Mean difference & s.d. \\\\",
        "    \\midrule",
    ]
    emitted: list[str] = []
    for row in CONTRASTS:
        if row.key not in contrasts:
            raise KeyError(
                f"{row.key!r} is not a contrast in {SUMMARY_JSON}; the document may not "
                "report a comparison the summary does not carry"
            )
        block = contrasts[row.key]
        n_pairs = int(cast(int, block["n_pairs"]))
        if n_pairs == 0:
            raise ValueError(
                f"{row.key!r} pairs no seeds, so it was never run; remove it from "
                "CONTRASTS rather than printing an empty row"
            )
        arms = f"{block['arm_a']} vs {block['arm_b']}".replace("_", "\\_")
        mean = float(cast(float, block["mean_difference"]))
        sd = float(cast(float, block["sd_difference"]))
        # The headline contrast is the one the section is about, so it is the one row a
        # reader should find without reading the others.
        cell = f"{mean:+.4f}"
        if row.key == "e200/conc_vs_pooled":
            cell = f"\\textbf{{{cell}}}"
        lines.append(
            f"    {row.question} & {arms} & {n_pairs} & {cell} & {sd:.4f} \\\\"
        )
        emitted.append(row.key)
    lines += ["    \\bottomrule", "  \\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines), emitted


def main() -> None:
    summary = _read(SUMMARY_JSON)
    body, emitted = _contrast_table(summary)

    assets: list[EmittedAsset] = []
    table_path = osp.join(TABLE_DIR, "wave7_aa_contrasts.tex")
    with open(table_path, "w") as handle:
        handle.write(body)
    assets.append(EmittedAsset(path=table_path, sha256=_sha256(table_path)))
    print(f"wrote {table_path}")

    # The stable-named copy. `make plots` matches figures/NAME.pdf against
    # $ASSET_IMAGES_DIR/<slug>/NAME.svg, so a timestamped render is invisible to it, and
    # renaming the render in place would break the note that embeds the timestamped name.
    figure_svg = cast(str, summary["figure_path"])
    figure_png = f"{osp.splitext(figure_svg)[0]}.png"
    for source in (figure_svg, figure_png):
        if not osp.isfile(source):
            raise FileNotFoundError(
                f"{source} does not exist; rerun analyze_wave7_aa.py rather than "
                "publishing a document against a figure that is not on disk"
            )
        target = osp.join(IMAGE_DIR, f"{STABLE_STEM}{osp.splitext(source)[1]}")
        shutil.copyfile(source, target)
        assets.append(
            EmittedAsset(
                path=target,
                sha256=_sha256(target),
                source_path=source,
                source_sha256=_sha256(source),
            )
        )
        print(f"copied {osp.basename(source)} -> {osp.basename(target)}")

    report = AssetReport(
        built_at=datetime.now(UTC).isoformat(),
        summary_json=SUMMARY_JSON,
        summary_sha256=_sha256(SUMMARY_JSON),
        summary_generated_at=cast(str, summary["generated_at"]),
        primary_budget=cast(str, summary["primary_budget"]),
        contrasts_emitted=emitted,
        assets=assets,
    )
    with open(ASSETS_JSON, "w") as handle:
        handle.write(report.model_dump_json(indent=1))
    print(f"wrote {ASSETS_JSON}")


if __name__ == "__main__":
    main()
