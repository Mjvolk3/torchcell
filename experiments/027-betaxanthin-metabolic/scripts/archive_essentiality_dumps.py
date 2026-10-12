# experiments/027-betaxanthin-metabolic/scripts/archive_essentiality_dumps.py
# [[experiments.027-betaxanthin-metabolic.scripts.archive_essentiality_dumps]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/027-betaxanthin-metabolic/scripts/archive_essentiality_dumps
"""Give every essentiality cell's per-gene prediction dump the name of its result file.

Until 2026-09-16 ``train_bx.py`` named a dump ``<arm>_essentiality_s<seed>.json``, with no
architecture in the name. The wave-11 size ladder ran widths 12, 24, 54 and 90 on the pooled
arm at the same three seeds, so every cell wrote the same three files and the last to finish
won: on 2026-09-16 ``pooled_essentiality_s2700.json`` held the width-12 cell while
``fcl_comparison.py`` read it as the wave-9 width-90 arm. The trainer now names a dump after
the result file, which is unique per cell.

This script is the one-time bridge for cells that ran under the old naming, and it is safe
to rerun. For every essentiality result file it resolves the dump the result recorded in
``test.dump_path``, checks that the dump's ``cell`` is the result's cell on every identity
field, and copies it to ``<TEST_PRED_DIR>/<result stem>.json`` when that file is not there
yet. A dump that fails the identity check is NOT copied: it belongs to a later cell that
overwrote the file, and the result's per-gene predictions are lost. Those cells are listed
under ``lost`` so they can be rerun rather than silently dropped from a comparison.

Run from the worktree root::

    PYTHONPATH=$PWD ~/miniconda3/envs/torchcell/bin/python \
        experiments/027-betaxanthin-metabolic/scripts/archive_essentiality_dumps.py

Writes ``results/essentiality_dump_archive.json``.
"""

from __future__ import annotations

import glob
import json
import os
import os.path as osp
import shutil

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field

load_dotenv()
DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
RESULTS = osp.join(EXPERIMENT_ROOT, "027-betaxanthin-metabolic", "results")
OUT = osp.join(RESULTS, "essentiality_dump_archive.json")
#: Where the essentiality cells dump; the launchers set TC_TEST_PRED_DIR to this directory.
PRED_DIR = osp.join(DATA_ROOT, "test-predictions-027-wave7")
#: The result files of every essentiality cell that trained to its full budget; the
#: two-epoch canaries are left out because nothing reads their predictions.
RESULT_GLOBS = (
    "wave9_ess_*_e80_s*.json",
    "wave11_ess_*_e80_s*.json",
    "wave12_ess_*_e80_s*.json",
)
#: The fields on which a dump must agree with the result that claims it. A missing field
#: on both sides (``preprocessor_layers`` predates wave 11) counts as agreement.
IDENTITY_FIELDS = (
    "head",
    "arm",
    "seed",
    "hidden",
    "layers",
    "attention_heads",
    "preprocessor_layers",
    "epochs",
)


class DumpStatus(BaseModel):
    """What was found for one result file."""

    model_config = ConfigDict(extra="forbid")
    result_file: str
    expected_dump: str
    status: str = Field(
        description="present | archived | lost | no_dump_recorded; `lost` means the file "
        "the result points at now holds a different cell."
    )
    legacy_dump: str | None = None
    legacy_holds: dict[str, object] | None = Field(
        default=None,
        description="Identity of the cell in the legacy file, when it differs.",
    )


class ArchiveReport(BaseModel):
    """Every essentiality result's dump status, and the cells whose dump is gone."""

    model_config = ConfigDict(extra="forbid")
    pred_dir: str
    statuses: list[DumpStatus]
    lost: list[str] = Field(description="Result stems whose per-gene dump is gone.")


def identity(cell: dict[str, object]) -> dict[str, object]:
    return {k: cell.get(k) for k in IDENTITY_FIELDS}


def main() -> None:
    statuses: list[DumpStatus] = []
    files = sorted(f for g in RESULT_GLOBS for f in glob.glob(osp.join(RESULTS, g)))
    assert files, f"no essentiality result files under {RESULTS}."
    for result_file in files:
        with open(result_file) as fh:
            result = json.load(fh)
        cell = result["cell"]
        assert cell["head"] == "essentiality", (
            f"{result_file} is not an essentiality cell."
        )
        stem = osp.splitext(osp.basename(result_file))[0]
        expected = osp.join(PRED_DIR, f"{stem}.json")
        legacy = result["test"].get("dump_path")
        if legacy is None:
            statuses.append(
                DumpStatus(
                    result_file=result_file,
                    expected_dump=expected,
                    status="no_dump_recorded",
                )
            )
            continue
        if osp.isfile(expected):
            with open(expected) as fh:
                held = identity(json.load(fh)["cell"])
            assert held == identity(cell), (
                f"{expected} holds {held}, not {identity(cell)}; a dump named after a result "
                "must be that result's."
            )
            statuses.append(
                DumpStatus(
                    result_file=result_file,
                    expected_dump=expected,
                    status="present",
                    legacy_dump=legacy,
                )
            )
            continue
        if not osp.isfile(legacy):
            statuses.append(
                DumpStatus(
                    result_file=result_file,
                    expected_dump=expected,
                    status="lost",
                    legacy_dump=legacy,
                )
            )
            continue
        with open(legacy) as fh:
            held = identity(json.load(fh)["cell"])
        if held == identity(cell):
            shutil.copy2(legacy, expected)
            statuses.append(
                DumpStatus(
                    result_file=result_file,
                    expected_dump=expected,
                    status="archived",
                    legacy_dump=legacy,
                )
            )
        else:
            statuses.append(
                DumpStatus(
                    result_file=result_file,
                    expected_dump=expected,
                    status="lost",
                    legacy_dump=legacy,
                    legacy_holds=held,
                )
            )

    report = ArchiveReport(
        pred_dir=PRED_DIR,
        statuses=statuses,
        lost=[
            osp.splitext(osp.basename(s.result_file))[0]
            for s in statuses
            if s.status == "lost"
        ],
    )
    with open(OUT, "w") as fh:
        json.dump(report.model_dump(), fh, indent=2)
    for s in statuses:
        line = f"{s.status:17s} {osp.basename(s.result_file)}"
        if s.legacy_holds is not None:
            line += f"  (legacy file holds {s.legacy_holds})"
        print(line)
    print(f"\nlost: {report.lost}\n{OUT}")


if __name__ == "__main__":
    main()
