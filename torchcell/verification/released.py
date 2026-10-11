# torchcell/verification/released
# [[torchcell.verification.released]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/verification/released
"""Sha256-pinned readers of the released tables the #889 family verifiers check against.

A family verifier compares a built store to the table the paper released, so it reads
that table itself rather than trusting the loader that built the store. Every reader
here takes the files a registry entry pins (:class:`ReleasedFile`, a name plus the
sha256 of the bytes the served build consumed), refuses to read a file whose bytes have
drifted, and returns plain arrays or counters, never the loader's frames. The readers
deliberately parse each table the way the paper describes it, by column name, so a
loader that read a neighboring column would disagree with them.

The raw bytes live in each store's own ``raw/`` directory in the dev tree (the
Costanzo 2016, Kuzmin 2018 and Kuzmin 2020 loaders predate the raw mirror and are on
the ``UNPINNED_LOADERS`` debt list), so the pin here is the record of which bytes were
verified. A drifted file is reported as a failing rule by the caller, not raised.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import os.path as osp
from collections import Counter
from collections.abc import Iterable, Sequence

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict

from torchcell.verification.report import sha256_file

#: A one-dimensional float64 array of released or stored values.
FloatArray = npt.NDArray[np.float64]

__all__ = [
    "COSTANZO2016_P_VALUE",
    "COSTANZO2016_SCORE",
    "KUZMIN_COMBINED_TYPE",
    "KUZMIN_FINAL_SCORE",
    "KUZMIN_P_VALUE",
    "InteractionValues",
    "ReleasedFile",
    "drifted_files",
    "interaction_values_from_table",
    "sgd_inviable_null_annotations",
    "sgd_json_digest",
    "sorted_value_mismatches",
]

#: Data File S1 columns 6 and 7 (``costanzoGlobalGeneticInteraction2016/si/si1.md``
#: "Supplementary Data File Descriptions"), as the released tab-delimited header
#: spells them.
COSTANZO2016_SCORE = "Genetic interaction score (ε)"
COSTANZO2016_P_VALUE = "P-value"
#: Kuzmin 2018 Data S1 and Kuzmin 2020 Tables S1/S3 columns 5, 7 and 8 as released.
KUZMIN_COMBINED_TYPE = "Combined mutant type"
KUZMIN_FINAL_SCORE = "Adjusted genetic interaction score (epsilon or tau)"
KUZMIN_P_VALUE = "P-value"


class ReleasedFile(BaseModel):
    """One released file a verifier reads, pinned to the bytes the build consumed."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    sha256: str


def drifted_files(raw_dir: str, files: Sequence[ReleasedFile]) -> dict[str, str]:
    """``{name: observed sha256}`` for every pinned file whose bytes have changed.

    A missing file raises ``FileNotFoundError``: a verifier asked to compare a store to
    a release it cannot see has nothing to report.
    """
    drift: dict[str, str] = {}
    for released in files:
        observed = sha256_file(osp.join(raw_dir, released.name))
        if observed != released.sha256:
            drift[released.name] = observed
    return drift


class InteractionValues(BaseModel):
    """The (score, p-value) pairs of one interaction table, in row order.

    ``p_values`` holds NaN where the release leaves the p-value blank, which is how a
    stored ``None`` is compared with it.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    scores: FloatArray
    p_values: FloatArray

    @property
    def n_rows(self) -> int:
        """The number of released rows these values came from."""
        return int(self.scores.shape[0])


def interaction_values_from_table(
    raw_dir: str,
    files: Sequence[ReleasedFile],
    *,
    score_column: str,
    p_value_column: str,
    combined_type: str | None = None,
    excel_skiprows: int = 1,
) -> InteractionValues:
    """Read the score and p-value columns of every row of the pinned released tables.

    ``.txt`` and ``.tsv`` files are read tab-delimited and ``.xlsx`` files with the
    paper's title row skipped, which is how each release lays its header out. When
    ``combined_type`` is given only the rows whose ``Combined mutant type`` equals it
    are kept (Kuzmin's ``digenic`` / ``trigenic``); Costanzo's Data File S1 has no
    such column and every row is a digenic pair.
    """
    import pandas as pd

    scores: list[FloatArray] = []
    p_values: list[FloatArray] = []
    columns = [score_column, p_value_column]
    if combined_type is not None:
        columns.append(KUZMIN_COMBINED_TYPE)
    for released in files:
        path = osp.join(raw_dir, released.name)
        if released.name.endswith(".xlsx"):
            frame = pd.read_excel(path, skiprows=excel_skiprows, usecols=columns)
        else:
            frame = pd.read_csv(path, sep="\t", usecols=columns)
        if combined_type is not None:
            frame = frame[frame[KUZMIN_COMBINED_TYPE] == combined_type]
        scores.append(frame[score_column].to_numpy(dtype=np.float64))
        p_values.append(frame[p_value_column].to_numpy(dtype=np.float64))
    return InteractionValues(
        scores=np.concatenate(scores), p_values=np.concatenate(p_values)
    )


def sorted_value_mismatches(
    stored: InteractionValues, released: InteractionValues
) -> tuple[int, list[dict[str, float | None]]]:
    """Compare two (score, p-value) multisets; ``(n differing positions, examples)``.

    Both multisets are sorted by (score, p-value) and compared position by position, NaN
    equal to NaN. Equal multisets give 0. Unequal lengths are reported by the caller
    before this is called, so a length mismatch here is a programming error.
    """
    if stored.n_rows != released.n_rows:
        raise ValueError(
            f"multisets differ in size ({stored.n_rows} vs {released.n_rows})"
        )
    order_s = np.lexsort((stored.p_values, stored.scores))
    order_r = np.lexsort((released.p_values, released.scores))
    s_score, s_p = stored.scores[order_s], stored.p_values[order_s]
    r_score, r_p = released.scores[order_r], released.p_values[order_r]
    same = (s_score == r_score) & ((s_p == r_p) | (np.isnan(s_p) & np.isnan(r_p)))
    differ = np.flatnonzero(~same)
    examples = [
        {
            "stored_score": float(s_score[i]),
            "stored_p_value": None if math.isnan(s_p[i]) else float(s_p[i]),
            "released_score": float(r_score[i]),
            "released_p_value": None if math.isnan(r_p[i]) else float(r_p[i]),
        }
        for i in differ[:20]
    ]
    return int(differ.shape[0]), examples


# --------------------------------------------------------------------------- #
# SGD per-gene JSON release (gene essentiality)
# --------------------------------------------------------------------------- #
def sgd_json_digest(genes_dir: str) -> str:
    """One sha256 over every ``<gene>.json`` the SGD gene download wrote.

    The SGD release is not one file but 6,607 per-gene API responses, so the pin is a
    sha256 over the sorted ``"<file name> <file sha256>"`` lines: any changed, added or
    removed file changes it.
    """
    lines = [
        f"{name} {sha256_file(osp.join(genes_dir, name))}"
        for name in sorted(os.listdir(genes_dir))
        if name.endswith(".json")
    ]
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def sgd_inviable_null_annotations(
    genes_dir: str, genes: Iterable[str] | None = None
) -> Counter[tuple[str, str]]:
    """``(systematic name, PubMed id) -> count`` of SGD's inviable null S288C annotations.

    An annotation counts when its ``mutant_type`` is ``null``, its strain is ``S288C``
    and its phenotype is ``inviable``, the three fields SGD's phenotype API states on
    each ``phenotype_details`` entry. ``genes`` narrows the read to those files (all
    ``.json`` files when None).
    """
    names = (
        sorted(f"{gene}.json" for gene in genes)
        if genes is not None
        else sorted(n for n in os.listdir(genes_dir) if n.endswith(".json"))
    )
    annotations: Counter[tuple[str, str]] = Counter()
    for name in names:
        with open(osp.join(genes_dir, name), encoding="utf-8") as handle:
            payload = json.load(handle)
        for detail in payload.get("phenotype_details") or []:
            if (
                detail["mutant_type"] == "null"
                and detail["strain"]["display_name"] == "S288C"
                and detail["phenotype"]["display_name"] == "inviable"
            ):
                annotations[
                    (name[: -len(".json")], str(detail["reference"]["pubmed_id"]))
                ] += 1
    return annotations
