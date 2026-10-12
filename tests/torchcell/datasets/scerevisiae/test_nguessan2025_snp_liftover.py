# tests/torchcell/datasets/scerevisiae/test_nguessan2025_snp_liftover.py
# [[tests.torchcell.datasets.scerevisiae.test_nguessan2025_snp_liftover]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/scerevisiae/test_nguessan2025_snp_liftover.py
"""The modeling-time SNP lift of experiment 036 (#1050), on synthetic references.

The lift lives in ``experiments/036-dataset-fixes-before-kg-build/scripts/
nguessan2025_snp_liftover.py``, not in the N'Guessan 2025 loader, which stores the
released positions. The script is loaded by path.
"""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

from torchcell.datasets.scerevisiae.bloom2019 import Marker

SCRIPT = (
    Path(__file__).resolve().parents[4]
    / "experiments/036-dataset-fixes-before-kg-build/scripts/nguessan2025_snp_liftover.py"
)


@pytest.fixture(scope="module")
def lift() -> ModuleType:
    spec = importlib.util.spec_from_file_location("nguessan2025_snp_liftover", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Registered first, so pydantic resolves the module's postponed annotations.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _dna(n: int, seed: int) -> str:
    return "".join(np.random.default_rng(seed).choice(list("ACGT"), n))


def _snp(chrom: str, position: int, source: Mapping[str, str], i: int) -> Marker:
    base = source[chrom][position - 1]
    alt = "A" if base != "A" else "C"
    return Marker(
        chromosome=chrom, position=position, ref=base, alt=alt, index=i, column=i
    )


def test_every_status_is_assigned_and_no_snp_is_dropped(lift: ModuleType) -> None:
    """Seed 8, flanks 8/16/32 on synthetic references with known edits.

    chrI: S288C has 7 bp inserted at 50 (+7 shift), a base changed inside the second
    SNP's left flank, the third SNP's 17-bp core copied onto chrIV (two hits at 8,
    one at 16), the fourth SNP's 65-bp window copied onto chrIV (two hits at every
    flank). chrII is reverse complemented in S288C. chrIII's only copy is in chrIV.
    """
    a, b, c = _dna(400, 1), _dna(400, 2), _dna(300, 3)
    source = {"chrI": a, "chrII": b, "chrIII": c}
    target_i = list(a[:50] + "GATTACA" + a[50:])
    target_i[198 + 7] = "A" if target_i[198 + 7] != "A" else "C"
    target = {
        "chrI": "".join(target_i),
        "chrII": lift.reverse_complement(b),
        "chrIII": _dna(300, 4),
        "chrIV": _dna(100, 5) + a[292:309] + _dna(50, 6) + a[318:383] + c,
    }
    snps = [
        (0, "snp1", _snp("chrI", 101, source, 0)),
        (1, "snp2", _snp("chrI", 201, source, 1)),
        (2, "snp3", _snp("chrI", 301, source, 2)),
        (3, "snp4", _snp("chrI", 351, source, 3)),
        (4, "snp5", _snp("chrI", 5, source, 4)),
        (5, "snp6", _snp("chrII", 101, source, 5)),
        (6, "snp7", _snp("chrIII", 151, source, 6)),
    ]
    rows = lift.lift_snps(snps, source, target, flanks=(8, 16, 32), seed=8)
    got = [
        (
            r.snp,
            r.status.value,
            r.flank_bp,
            r.hits,
            r.s288c_chromosome,
            r.s288c_position,
            r.s288c_strand,
        )
        for r in rows
    ]
    offset_iv = 100 + 17 + 50 + 65
    assert got == [
        ("snp1", "lifted", 8, 1, "chrI", 108, "+"),
        ("snp2", "no_match", 8, 0, None, None, None),
        ("snp3", "lifted", 16, 1, "chrI", 308, "+"),
        ("snp4", "multiple", 32, 2, None, None, None),
        ("snp5", "edge", 8, 0, None, None, None),
        ("snp6", "lifted", 8, 1, "chrII", 300, "-"),
        ("snp7", "lifted", 8, 1, "chrIV", offset_iv + 151, "+"),
    ]
    assert all(r.s288c_base == r.by for r in rows if r.status == "lifted")
    markers = lift.lifted_markers(rows)
    assert [(m.chromosome, m.position, m.column) for m in markers] == [
        ("chrI", 108, 0),
        ("chrI", 308, 2),
        ("chrII", 300, 5),
        ("chrIV", offset_iv + 151, 6),
    ]
    minus = markers[2]
    assert (minus.ref, minus.alt) == (
        lift.reverse_complement(rows[5].by),
        lift.reverse_complement(rows[5].rm),
    )


def test_the_s288c_base_is_read_independently_of_the_match(lift: ModuleType) -> None:
    a = _dna(200, 7)
    target = {"chrI": a[:100] + ("A" if a[100] != "A" else "C") + a[101:]}
    (row,) = lift.lift_snps(
        [(0, "snp1", _snp("chrI", 101, {"chrI": a}, 0))],
        {"chrI": a},
        target,
        flanks=(8,),
        seed=8,
    )
    assert (row.status, row.s288c_position) == (lift.LiftStatus.lifted, 101)
    assert row.s288c_base != row.by


def test_a_seed_longer_than_a_flank_or_unsorted_flanks_is_refused(
    lift: ModuleType,
) -> None:
    with pytest.raises(ValueError, match="need seed <= min"):
        lift.lift_snps([], {}, {}, flanks=(8, 16), seed=10)
    with pytest.raises(ValueError, match="need seed <= min"):
        lift.lift_snps([], {}, {}, flanks=(16, 8), seed=8)


def test_the_table_writes_every_row_with_empty_cells_for_none(
    lift: ModuleType, tmp_path: Path
) -> None:
    common: dict[str, Any] = {
        "index": 1,
        "snp": "snp2",
        "chromosome": "chrI",
        "position": 100,
        "by": "A",
        "rm": "T",
    }
    rows = [
        lift.LiftedSnp(
            **common,
            status=lift.LiftStatus.lifted,
            flank_bp=25,
            hits=1,
            s288c_chromosome="chrI",
            s288c_position=102,
            s288c_strand="+",
            s288c_base="A",
        ),
        lift.LiftedSnp(
            **common,
            status=lift.LiftStatus.multiple,
            flank_bp=200,
            hits=3,
            s288c_chromosome=None,
            s288c_position=None,
            s288c_strand=None,
            s288c_base=None,
        ),
    ]
    path = tmp_path / "lifted.tsv"
    lift.write_table(rows, path)
    lines = path.read_text().splitlines()
    assert lines[0].split("\t") == list(lift.TABLE_COLUMNS)
    assert lines[1] == "1\tsnp2\tchrI\t100\tA\tT\tlifted\t25\t1\tchrI\t102\t+\tA"
    assert lines[2] == "1\tsnp2\tchrI\t100\tA\tT\tmultiple\t200\t3\t\t\t\t"


def test_the_fasta_readers_name_chromosomes_as_markers_do(
    lift: ModuleType, tmp_path: Path
) -> None:
    by = tmp_path / "by.fasta"
    by.write_text(
        ">2-micron\nTT\n"
        + "".join(f">chr{i:02d}\nacg\n" for i in range(1, 17))
        + ">chr17\nCC\n"
    )
    seqs = lift.by4742_chromosomes(by)
    assert len(seqs) == 16 and seqs["chrXVI"] == "ACG"
    sgd = tmp_path / "s288c.fsa"
    roman = [name.removeprefix("chr") for name in lift._CHROM_ORDER]
    sgd.write_text(
        "".join(f">ref|NC_{i}| [chromosome={r}]\nacgt\n" for i, r in enumerate(roman))
        + ">ref|NC_001224| [location=mitochondrion]\nGG\n"
    )
    assert lift.s288c_chromosomes(sgd)["chrIV"] == "ACGT"
    sgd.write_text(">ref|NC_1| [chromosome=I]\nA\n")
    with pytest.raises(ValueError, match="no nuclear chromosome"):
        lift.s288c_chromosomes(sgd)
