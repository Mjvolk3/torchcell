# tests/torchcell/verification/test_natural_isolate_response.py
# [[tests.torchcell.verification.test_natural_isolate_response]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/verification/test_natural_isolate_response.py
"""The natural-isolate verifier's genotype and provenance rules, on hand-built inputs.

The whole verifier runs on a synthetic two-isolate store in
``tests/torchcell/datasets/scerevisiae/test_peter2018_synthetic.py``. Here each genotype
rule is driven by a matrix of three pangenome ORFs (``1-YAL001C`` reference,
``2-acc`` accessory, ``3-YBR001W`` reference) and one isolate ``AAA`` with presence
1 1 0 and copy numbers 1, NaN, 1, so its expected genotype is accessory ``2-acc``
present at copy number 1.0 and ``YBR001W`` absent.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from torchcell.datasets.scerevisiae import peter2018 as p
from torchcell.verification.natural_isolate_response import (
    _GenotypeChecks,
    _l3_conditions_documented,
    _l4_published_md5,
)
from torchcell.verification.report import Level

_PANGENOME = p.PangenomeMatrices(
    orf_ids=["1-YAL001C", "2-acc", "3-YBR001W"],
    s288c_names=["YAL001C", None, "YBR001W"],
    s288c_mask=np.array([True, False, True]),
    presence=np.array([[1.0, 1.0, 0.0]]),
    copy_number=np.array([[1.0, np.nan, 1.0]]),
    row_of={"AAA": 0},
)


def _variant(strain: str, gene: str, sha: str = "tar") -> dict[str, Any]:
    return {
        "perturbation_type": "sequence_variant",
        "systematic_gene_name": gene,
        "strain_id": strain,
        "sequence_ref": {"sha256": sha, "member": f"{gene}.fasta#{strain}_{gene}_X"},
    }


def _good() -> list[dict[str, Any]]:
    return [
        _variant("AAA", "YAL001C"),
        {
            "perturbation_type": "natural_gene_presence",
            "systematic_gene_name": "2-acc",
            "pangenome_orf_id": "2-acc",
            "copy_number": 1.0,
            "strain_id": "AAA",
        },
        {
            "perturbation_type": "natural_gene_absence",
            "systematic_gene_name": "YBR001W",
            "strain_id": "AAA",
        },
    ]


def test_a_genotype_matching_its_matrix_row_passes() -> None:
    checks = _GenotypeChecks(_PANGENOME, "tar")
    assert checks.add(_good()) == "AAA"
    assert (
        checks.round_trip_failures,
        checks.pointer_failures,
        checks.multi_strain,
    ) == ([], [], [])
    assert dict(checks.type_counts) == {
        "sequence_variant": 1,
        "natural_gene_presence": 1,
        "natural_gene_absence": 1,
    }
    assert set(checks.s288c_names) == {"YAL001C", "YBR001W"}


def test_each_genotype_defect_is_named() -> None:
    checks = _GenotypeChecks(_PANGENOME, "tar")
    wrong = _good()[:1] + [
        {
            "perturbation_type": "allele",
            "systematic_gene_name": "YCR001W",
            "strain_id": "BBB",
        }
    ]
    wrong[0] = _variant("AAA", "YAL001C", sha="other")
    checks.add(wrong)
    assert checks.multi_strain == ["AAA,BBB"]
    assert checks.round_trip_failures == [
        "AAA: unexpected allele",
        "AAA: 0 present accessory ORFs stored, 1 in the matrix",
        "AAA: 0 absent reference ORFs stored, 1 in the matrix",
    ]
    assert checks.pointer_failures == ["AAA/YAL001C: 'YAL001C.fasta#AAA_YAL001C_X'"]
    checks.add([_variant("ZZZ", "YAL001C")])
    assert checks.round_trip_failures[-1] == "ZZZ: not in the presence matrix"


def test_pointer_failures_are_capped_at_twenty_then_marked() -> None:
    checks = _GenotypeChecks(_PANGENOME, "tar")
    for _ in range(22):
        checks._check_pointer("AAA", _variant("AAA", "YAL001C", sha="other"))
    assert len(checks.pointer_failures) == 22
    assert checks.pointer_failures[20:] == ["...", "..."]


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
