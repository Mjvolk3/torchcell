# tests/torchcell/verification/test_natural_isolate_response.py
# [[tests.torchcell.verification.test_natural_isolate_response]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/verification/test_natural_isolate_response.py
"""The natural-isolate verifier's census and provenance rules, on hand-built inputs.

The whole verifier runs on a synthetic two-isolate store in
``tests/torchcell/datasets/scerevisiae/test_peter2018_synthetic.py``. The genotype rules
every natural-isolate store shares (identity, round trip, Table S1, loss-of-function
counts) are tested in ``tests/torchcell/verification/test_natural_isolate_genotype.py``;
this verifier keeps only the census its two L4 containment rules read.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from torchcell.datasets.scerevisiae import peter2018 as p
from torchcell.verification.natural_isolate_response import (
    _GenotypeCensus,
    _l3_conditions_documented,
    _l4_published_md5,
)
from torchcell.verification.report import Level


def test_the_census_keys_presences_by_pangenome_id_and_the_rest_by_name() -> None:
    census = _GenotypeCensus()
    census.add(
        [
            {
                "perturbation_type": "sequence_variant",
                "systematic_gene_name": "YAL001C",
            },
            {
                "perturbation_type": "natural_gene_presence",
                "systematic_gene_name": "2-acc",
                "pangenome_orf_id": "2-acc",
            },
            {"perturbation_type": "deletion", "systematic_gene_name": "YDL227C"},
            {
                "perturbation_type": "natural_loss_of_function",
                "systematic_gene_name": "YAL001C",
            },
        ]
    )
    assert dict(census.type_counts) == {
        "sequence_variant": 1,
        "natural_gene_presence": 1,
        "deletion": 1,
        "natural_loss_of_function": 1,
    }
    assert census.s288c_names == Counter({"YAL001C": 2, "YDL227C": 1})
    assert census.pangenome_ids == Counter({"2-acc": 1})


def test_conditions_documented_fails_on_a_drifted_workbook(tmp_path: Path) -> None:
    workbook = tmp_path / "si3.xls"
    workbook.write_bytes(b"not the workbook")
    conditions = p.build_conditions()
    result = _l3_conditions_documented(
        Counter(dict.fromkeys(conditions, 2)), 3, conditions, workbook
    )
    assert (result.level, result.passed) == (Level.L3, False)
    assert result.message == (
        "3 records on no loader condition; 0 Table 2 rows differ; workbook sha256 "
        "ok=False; 35 conditions observed"
    )


def test_published_md5_names_all_three_digests_on_a_mismatch(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / p.PHENO_NAME).write_bytes(b"x")
    mirror = tmp_path / "mirror"
    (mirror / "data").mkdir(parents=True)
    (mirror / p.MD5_REL).write_text(f"MD5 ({p.PHENO_NAME}) = {'1' * 32}\n")
    result = _l4_published_md5(raw, mirror)
    assert result.passed is False
    assert result.message == (
        "release md5 9dd4e461268c8034f5c8564e155c67a6; md5.txt "
        f"{'1' * 32}; pin {p.PHENO_PUBLISHED_MD5}"
    )
