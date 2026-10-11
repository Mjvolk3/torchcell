# tests/torchcell/verification/test_released.py
# [[tests.torchcell.verification.test_released]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/verification/test_released.py
"""The sha256-pinned released-table readers, on small tables written under ``tmp_path``.

A Kuzmin-shaped table holds three rows (digenic -0.2 / 0.01, trigenic 0.3 / 0.5,
digenic 0.1 / blank); a Costanzo-shaped table two rows with no ``Combined mutant
type`` column. The xlsx carries a title row above its header, as the Kuzmin 2020
tables do. Two SGD gene JSONs carry four phenotype annotations, of which two match
(null, S288C, inviable): YAL001C twice from PMID 7, and none from YBR085W, whose only
inviable null entry names strain W303.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from torchcell.verification.released import (
    COSTANZO2016_P_VALUE,
    COSTANZO2016_SCORE,
    KUZMIN_COMBINED_TYPE,
    KUZMIN_FINAL_SCORE,
    KUZMIN_P_VALUE,
    InteractionValues,
    ReleasedFile,
    drifted_files,
    interaction_values_from_table,
    sgd_inviable_null_annotations,
    sgd_json_digest,
    sorted_value_mismatches,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _kuzmin_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Query strain ID": ["q1", "q2", "q3"],
            KUZMIN_COMBINED_TYPE: ["digenic", "trigenic", "digenic"],
            KUZMIN_FINAL_SCORE: [-0.2, 0.3, 0.1],
            KUZMIN_P_VALUE: [0.01, 0.5, np.nan],
        }
    )


def test_drifted_files_names_only_the_changed_file(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("a")
    (tmp_path / "b.txt").write_text("b")
    pins = [
        ReleasedFile(name="a.txt", sha256=_sha(tmp_path / "a.txt")),
        ReleasedFile(name="b.txt", sha256="0" * 64),
    ]
    assert drifted_files(str(tmp_path), pins) == {"b.txt": _sha(tmp_path / "b.txt")}
    with pytest.raises(FileNotFoundError):
        drifted_files(str(tmp_path), [ReleasedFile(name="c.txt", sha256="0")])


def test_a_tab_delimited_table_is_filtered_by_combined_type(tmp_path: Path) -> None:
    _kuzmin_frame().to_csv(tmp_path / "s1.tsv", sep="\t", index=False)
    files = [ReleasedFile(name="s1.tsv", sha256="unused")]
    digenic = interaction_values_from_table(
        str(tmp_path),
        files,
        score_column=KUZMIN_FINAL_SCORE,
        p_value_column=KUZMIN_P_VALUE,
        combined_type="digenic",
    )
    assert digenic.n_rows == 2
    assert digenic.scores.tolist() == [-0.2, 0.1]
    assert digenic.p_values[0] == 0.01
    assert math.isnan(digenic.p_values[1])
    trigenic = interaction_values_from_table(
        str(tmp_path),
        files,
        score_column=KUZMIN_FINAL_SCORE,
        p_value_column=KUZMIN_P_VALUE,
        combined_type="trigenic",
    )
    assert trigenic.scores.tolist() == [0.3]


def test_xlsx_skips_its_title_row_and_files_concatenate_in_order(
    tmp_path: Path,
) -> None:
    with pd.ExcelWriter(tmp_path / "s3.xlsx") as writer:
        pd.DataFrame([["Table S3 title"]]).to_excel(
            writer, index=False, header=False, startrow=0
        )
        _kuzmin_frame().to_excel(writer, index=False, startrow=1)
    _kuzmin_frame().to_csv(tmp_path / "s1.tsv", sep="\t", index=False)
    values = interaction_values_from_table(
        str(tmp_path),
        [
            ReleasedFile(name="s1.tsv", sha256="unused"),
            ReleasedFile(name="s3.xlsx", sha256="unused"),
        ],
        score_column=KUZMIN_FINAL_SCORE,
        p_value_column=KUZMIN_P_VALUE,
        combined_type="digenic",
    )
    assert values.scores.tolist() == [-0.2, 0.1, -0.2, 0.1]


def test_a_costanzo_table_has_no_combined_type_column(tmp_path: Path) -> None:
    pd.DataFrame(
        {
            "Query Strain ID": ["a", "b"],
            COSTANZO2016_SCORE: [0.03, -0.4],
            COSTANZO2016_P_VALUE: [0.1, 0.001],
        }
    ).to_csv(tmp_path / "SGA_ExE.txt", sep="\t", index=False)
    values = interaction_values_from_table(
        str(tmp_path),
        [ReleasedFile(name="SGA_ExE.txt", sha256="unused")],
        score_column=COSTANZO2016_SCORE,
        p_value_column=COSTANZO2016_P_VALUE,
    )
    assert values.scores.tolist() == [0.03, -0.4]
    assert values.p_values.tolist() == [0.1, 0.001]


def _values(scores: list[float], p_values: list[float]) -> InteractionValues:
    return InteractionValues(
        scores=np.array(scores, dtype=np.float64),
        p_values=np.array(p_values, dtype=np.float64),
    )


def test_sorted_value_mismatches_is_order_free_and_nan_equal() -> None:
    stored = _values([0.1, -0.2], [math.nan, 0.01])
    released = _values([-0.2, 0.1], [0.01, math.nan])
    assert sorted_value_mismatches(stored, released) == (0, [])
    changed = _values([-0.2, 0.1], [0.02, math.nan])
    n_differ, examples = sorted_value_mismatches(stored, changed)
    assert n_differ == 1
    assert examples == [
        {
            "stored_score": -0.2,
            "stored_p_value": 0.01,
            "released_score": -0.2,
            "released_p_value": 0.02,
        }
    ]
    blank = _values([-0.2, 0.1], [0.01, 0.5])
    assert sorted_value_mismatches(stored, blank)[1][0]["stored_p_value"] is None
    with pytest.raises(ValueError, match="differ in size"):
        sorted_value_mismatches(stored, _values([0.1], [0.1]))


def _write_gene(directory: Path, gene: str, details: list[dict[str, object]]) -> None:
    (directory / f"{gene}.json").write_text(json.dumps({"phenotype_details": details}))


def _detail(mutant: str, strain: str, phenotype: str, pubmed: int) -> dict[str, object]:
    return {
        "mutant_type": mutant,
        "strain": {"display_name": strain},
        "phenotype": {"display_name": phenotype},
        "reference": {"pubmed_id": pubmed},
    }


def test_sgd_annotations_count_null_s288c_inviable_entries(tmp_path: Path) -> None:
    _write_gene(
        tmp_path,
        "YAL001C",
        [
            _detail("null", "S288C", "inviable", 7),
            _detail("null", "S288C", "inviable", 7),
            _detail("null", "S288C", "viable", 8),
        ],
    )
    _write_gene(tmp_path, "YBR085W", [_detail("null", "W303", "inviable", 9)])
    (tmp_path / "notes.txt").write_text("not a gene")
    assert sgd_inviable_null_annotations(str(tmp_path)) == {("YAL001C", "7"): 2}
    assert sgd_inviable_null_annotations(str(tmp_path), genes=["YBR085W"]) == {}


def test_sgd_json_digest_changes_with_any_gene_file(tmp_path: Path) -> None:
    _write_gene(tmp_path, "YAL001C", [])
    before = sgd_json_digest(str(tmp_path))
    (tmp_path / "notes.txt").write_text("ignored")
    assert sgd_json_digest(str(tmp_path)) == before
    _write_gene(tmp_path, "YBR085W", [])
    assert sgd_json_digest(str(tmp_path)) != before
