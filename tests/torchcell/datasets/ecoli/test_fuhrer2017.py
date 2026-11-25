# tests/torchcell/datasets/ecoli/test_fuhrer2017.py
# [[tests.torchcell.datasets.ecoli.test_fuhrer2017]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/ecoli/test_fuhrer2017.py
"""The Fuhrer 2017 Keio metabolome loader (``torchcell.datasets.ecoli.fuhrer2017``).

Synthetic tests (run everywhere) build every input in ``tmp_path``. The hermetic build
uses the real ``EcoliK12BW25113Genome`` over the synthetic BW25113 assembly of
``tests/torchcell/sequence/genome/_bacterial_fixtures.py`` (JW4367 on BW25113_0001 thrL,
JW0001 on BW25113_0002 thrA, hokC on BW25113_4412), served through a stubbed ``resolve``
with the network refused, and replaces the loader's ``verify_raw_files`` with a
presence check (the synthetic files cannot carry the real pins; the pins are asserted by
the refusal test and by the data-gated tests). The deposit of the synthetic frame, in
the deposit's alphabetical column order:

    column  name   Table EV1A entries                      raw columns
    1       hokC   JW4367 b0001 (ready)                    4   -> record, JW4367
    2       ilvG   JW3740 b3767, JW3741 b3768 (ready)      8   -> record, pooled: JW3740
                                                               and JW3741, n = 4
    3       thrA   JW0001 b0002 (ready)                    4   -> record, JW0001
    4       thrL   JW8888 b0099 (ready)                    4   -> record, JW8888 (in no
                                                               annotation; a finding)
    5       wt     (none)                                  8   -> reference, n = 4
    6       yaaQ   JW9999 b9999 (ready)                    4   -> record, JW9999 (same)

Each z-score row is a permutation of the standardized 1..6 scores (median 0, SD 1 over
the six columns), so the build's standardization check passes. The raw matrices
(``RAW_LAYOUT``, 36 columns) carry the same names plus ``flhC``, the stand-in for a
withdrawn mutant (4 columns, left out), so the build writes 5 z-score records and 16 clone
records (hokC 2, ilvG 4, thrA 2, thrL 2, wt 4, yaaQ 2): clone ``g`` (0-based over the 18
pairs, flhC's two included) has injections ``_raw(g, r)`` and ``_raw(g, r) + 2`` per ion
row ``r``, so its stored value is ``_raw(g, r) + 1``. No genome is read in the
build (owner rule of 2026-10-12: the data is the data); the L4 row of the verifier
reports which JW ids the synthetic annotation places.

Data-gated tests (``@pytest.mark.data``) read the real raw mirror and the built dev-tree
LMDB under ``$DATA_ROOT`` (they never build it): the manifest pins, the provenance audit
of every sourced value, the record count, one hand-checked record (``thrA``, values read
off ``zscore_neg.tsv`` / ``zscore_pos.tsv`` column 1,915 with ``cut``), and the L0-L4
verifier.
"""

from __future__ import annotations

import json
import os
import os.path as osp
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import openpyxl
import pytest

import torchcell.datasets.ecoli.fuhrer2017 as m
from tests.torchcell.sequence.genome._bacterial_fixtures import (
    BW25113_LOCI,
    forbid_network,
    serve_tier,
    write_assembly,
)
from torchcell.data import ManifestPinMismatchError, RawSha256MismatchError
from torchcell.datamodels.media import M9_GLUCOSE_CASEIN_FUHRER2017
from torchcell.datamodels.schema import (
    Environment,
    ReferenceGenome,
    ReleasedIdentifierPerturbation,
    Temperature,
)
from torchcell.literature.manifest import RetrievalMethod, RetrievalRecord
from torchcell.sequence.genome.ecoli.k12 import BW25113_ASSEMBLY, EcoliK12BW25113Genome
from torchcell.verification.sourced import audit_sourced_value

Z6 = [-1.3363, -0.8018, -0.2673, 0.2673, 0.8018, 1.3363]
NEG_ROWS = [
    [Z6[0], Z6[1], Z6[2], Z6[3], Z6[4], Z6[5]],
    [Z6[5], Z6[4], Z6[3], Z6[2], Z6[1], Z6[0]],
    [Z6[2], Z6[0], Z6[4], Z6[1], Z6[5], Z6[3]],
]
POS_ROWS = [
    [Z6[1], Z6[2], Z6[0], Z6[5], Z6[3], Z6[4]],
    [Z6[3], Z6[5], Z6[1], Z6[0], Z6[2], Z6[4]],
]
COLUMNS = ["hokC", "ilvG", "thrA", "thrL", "wt", "yaaQ"]
RAW_COUNTS = {"hokC": 4, "ilvG": 8, "thrA": 4, "thrL": 4, "wt": 8, "yaaQ": 4}
#: The raw matrices' names in release order: the z-score names plus a withdrawn one.
RAW_LAYOUT = [("flhC", 4), *RAW_COUNTS.items()]
RAW_NAMES = [name for name, n in RAW_LAYOUT for _ in range(n)]
#: Clone index ranges (0-based over the 18 technical pairs) per raw name.
CLONES_OF = {"flhC": [0, 1], "hokC": [2, 3], "ilvG": [4, 5, 6, 7], "thrA": [8, 9]}
CLONES_OF |= {"thrL": [10, 11], "wt": [12, 13, 14, 15], "yaaQ": [16, 17]}


def _raw(clone: int, row: int) -> int:
    """The first injection of clone ``clone`` at ion row ``row`` (neg rows 0-2, then
    pos rows 3-4); the second injection is this plus 2.
    """
    return 1000 + 37 * ((5 * clone + 3 * row) % 13) + 11 * row


EV1A_ROWS: list[tuple[str, str, str, str]] = [
    ("JW4367", "b0001", "hokC", "ready to distribute"),
    ("JW3740", "b3767", "ilvG", "ready to distribute"),
    ("JW3741", "b3768", "ilvG", "ready to distribute"),
    ("JW0001", "b0002", "thrA", "ready to distribute"),
    ("JW8888", "b0099", "thrL", "ready to distribute"),
    ("JW9999", "b9999", "yaaQ", "ready to distribute"),
    ("JW5765", " ", " ", "not current_JW ORF"),
]
NEG_MZ = [100.0, 200.0, 300.0]
POS_MZ = [150.0, 250.0]
EV1B_NEG_MZ = [100.001, 200.001, 300.001]
EV1B_POS_MZ = [150.0005, 250.0005]
REFERENCE = ReferenceGenome(
    species="Escherichia coli",
    strain="BW25113",
    assembly_set="ecoli_K12_BW25113_ASM75055v1",
    assembly_accession="GCA_000750555.1",
)


def _entries() -> list[m.KeioEntry]:
    return [
        m.KeioEntry(jw_id=jw, blattner_id=b, gene_name=g.strip(), delivery_status=s)
        for jw, b, g, s in EV1A_ROWS
    ]


def _write_column(path: Path, values: list[Any]) -> None:
    """A one-column, headerless sheet (xlsx bytes; pandas sniffs the container)."""
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    assert sheet is not None
    for value in values:
        sheet.append([value])
    workbook.save(path)


def _write_table_ev1(path: Path) -> None:
    workbook = openpyxl.Workbook()
    ev1a = workbook.active
    assert ev1a is not None
    ev1a.title = m.TABLE_EV1A_SHEET
    ev1a.append(["title"])
    ev1a.append([None, None, None, None, None, None, "Growth-ratesc"])
    ev1a.append(["blank"])
    ev1a.append(
        [
            "Sample Number",
            "JW ID",
            "Blattner ID",
            "Gene Name",
            "Annotationa",
            "Keio Delivery Statusb",
            "Mean",
            "Standard Deviation",
        ]
    )
    for i, (jw, b, g, s) in enumerate(EV1A_ROWS):
        ev1a.append([i + 1, jw, b, g, "-", s, 0.8, 0.05])
    ev1a.append(["cMean and standard deviations are calculated from duplicate"])
    ev1b = workbook.create_sheet(m.TABLE_EV1B_SHEET)
    ev1b.append(["Ionization Mode", "Ion Index", "m/z", "Annotations"])
    ev1b.append([None, None, None, None])
    for mode, mzs in (("neg", EV1B_NEG_MZ), ("pos", EV1B_POS_MZ)):
        for i, mz in enumerate(mzs):
            ev1b.append([mode, i + 1, mz, "no annotation"])
    workbook.save(path)


def _write_raw(raw: Path) -> None:
    raw.mkdir(parents=True, exist_ok=True)
    (raw / m.STUDY_JSON).write_text(
        json.dumps(
            {
                "accno": "S-BSST5",
                "section": {
                    "attributes": [
                        {"name": "Organism", "value": "Escherichia coli"},
                        {
                            "name": "Experimental design",
                            "value": "2 Biological and 2 technical replicates",
                        },
                    ]
                },
            }
        )
    )
    for name, rows in ((m.ZSCORE_NEG, NEG_ROWS), (m.ZSCORE_POS, POS_ROWS)):
        (raw / name).write_text(
            "".join("\t".join(f"{v:.4f}" for v in row) + "\n" for row in rows)
        )
    for name, ion_rows in ((m.RAW_NEG, range(3)), (m.RAW_POS, range(3, 5))):
        (raw / name).write_text(
            "".join(
                "\t".join(
                    f"{_raw(g, r)}\t{_raw(g, r) + 2}"
                    for g in range(len(RAW_NAMES) // 2)
                )
                + "\n"
                for r in ion_rows
            )
        )
    _write_column(raw / m.SAMPLE_ID_ZSCORE, COLUMNS)
    _write_column(raw / m.SAMPLE_ID_ALL, RAW_NAMES)
    _write_column(raw / m.NEG_ION_MZ, NEG_MZ)
    _write_column(raw / m.POS_ION_MZ, POS_MZ)
    _write_table_ev1(raw / m.TABLE_EV1)


@pytest.fixture
def bw25113(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> EcoliK12BW25113Genome:
    """The real BW25113 class over the synthetic assembly; network refused."""
    files = write_assembly(tmp_path / "tier", BW25113_ASSEMBLY, BW25113_LOCI)
    forbid_network(monkeypatch)
    serve_tier(monkeypatch, files)
    root = tmp_path / "bw25113"
    root.mkdir()
    return EcoliK12BW25113Genome(genome_root=str(root), overwrite=False)


@pytest.fixture
def presence_only_pins(monkeypatch: pytest.MonkeyPatch) -> list[Mapping[str, str]]:
    """Replace the build-time byte check with a presence check that records the pins."""
    calls: list[Mapping[str, str]] = []

    def record(raw_dir: str, pins: Mapping[str, str]) -> None:
        missing = [f for f in pins if not osp.exists(osp.join(raw_dir, f))]
        if missing:
            raise FileNotFoundError(f"pinned raw files absent: {missing}")
        calls.append(dict(pins))

    monkeypatch.setattr(m, "verify_raw_files", record)
    return calls


@pytest.fixture
def synthetic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    bw25113: EcoliK12BW25113Genome,
    presence_only_pins: list[Mapping[str, str]],
) -> Path:
    """A dataset root whose raw/ holds the synthetic deposit, with small ion counts."""
    root = tmp_path / "metabolome_fuhrer2017"
    _write_raw(root / "raw")
    monkeypatch.setattr(m, "EXPECTED_ION_COUNTS", {"neg": 3, "pos": 2})
    monkeypatch.setattr(m, "assembly_reference", lambda strain, **_: REFERENCE)
    monkeypatch.setattr(m, "RAW_NAMES_ANALYZED", len(RAW_LAYOUT))
    monkeypatch.setattr(m, "WITHDRAWN_MUTANTS", ("flhC",))
    monkeypatch.setattr(m, "EXPECTED_ZSCORE_RECORDS", 5)
    monkeypatch.setattr(m, "EXPECTED_CLONE_RECORDS", 16)
    monkeypatch.setattr(m, "EXPECTED_RECORDS", 21)
    return root


def _expected_environment() -> dict[str, Any]:
    return Environment(
        media=M9_GLUCOSE_CASEIN_FUHRER2017, temperature=Temperature(value=37.0)
    ).model_dump()


# --------------------------------------------------------------------------- #
# Pure parsing and selection
# --------------------------------------------------------------------------- #
def test_ion_keys_are_mode_and_one_based_row() -> None:
    assert m.ion_keys(3, 2) == [
        "neg_0001",
        "neg_0002",
        "neg_0003",
        "pos_0001",
        "pos_0002",
    ]
    assert m.ion_key("pos", 4365) == "pos_4365"


def test_select_strains_keeps_every_mutant_column_and_lists_the_pools() -> None:
    """One record per mutant column; the shared name is a pooled record; wt is the
    reference with 8 raw columns = 4 cultures.
    """
    selection = m.select_strains(COLUMNS, _entries(), RAW_COUNTS)
    assert [(s.column, s.deposit_name, s.jw_ids) for s in selection.strains] == [
        (0, "hokC", ["JW4367"]),
        (1, "ilvG", ["JW3740", "JW3741"]),
        (2, "thrA", ["JW0001"]),
        (3, "thrL", ["JW8888"]),
        (5, "yaaQ", ["JW9999"]),
    ]
    assert [s.n_biological for s in selection.strains] == [2, 4, 2, 2, 2]
    assert [s.is_pooled for s in selection.strains] == [
        False,
        True,
        False,
        False,
        False,
    ]
    assert (selection.wt_column, selection.wt_raw_columns) == (4, 8)
    assert selection.wt_n_biological == 4
    assert selection.pooled == [
        "ilvG: JW3740 b3767 (ready to distribute), JW3741 b3768 (ready to "
        "distribute); 2 loci"
    ]
    assert selection.unlisted == []


def test_select_strains_names_a_same_locus_pool_as_one_jw_id() -> None:
    entries = [
        m.KeioEntry(
            jw_id="JW5182", blattner_id="b1185", gene_name="dsbB", delivery_status=s
        )
        for s in ("ready to distribute", "Eliminated; wrong primers")
    ]
    selection = m.select_strains(["dsbB", "wt"], entries, {"dsbB": 8, "wt": 4})
    (strain,) = selection.strains
    assert (strain.jw_ids, strain.n_biological, len(strain.entries)) == (
        ["JW5182"],
        4,
        2,
    )
    assert selection.pooled == [
        "dsbB: JW5182 b1185 (ready to distribute), JW5182 b1185 (Eliminated; wrong "
        "primers); same locus"
    ]
    (perturbation,) = m.deletion_genotype(strain).perturbations
    assert perturbation.systematic_gene_name == "JW5182"


def test_select_strains_stores_an_unlisted_name_and_a_non_ready_entry() -> None:
    entries = [
        m.KeioEntry(
            jw_id="JW0001",
            blattner_id="b0002",
            gene_name="thrA",
            delivery_status="Eliminated; weak growth",
        )
    ]
    selection = m.select_strains(
        ["nope", "thrA", "wt"], entries, {"nope": 4, "thrA": 4, "wt": 4}
    )
    assert selection.unlisted == ["nope"]
    unlisted, listed = selection.strains
    (p,) = m.deletion_genotype(unlisted).perturbations
    assert isinstance(p, ReleasedIdentifierPerturbation)
    assert (p.systematic_gene_name, p.identifier_kind, p.strain_label) == (
        "nope",
        m.DEPOSIT_NAME_KIND,
        "nope",
    )
    (q,) = m.deletion_genotype(listed).perturbations
    assert isinstance(q, ReleasedIdentifierPerturbation)
    assert (q.systematic_gene_name, q.identifier_kind) == ("JW0001", m.IDENTIFIER_KIND)


@pytest.mark.parametrize(
    ("columns", "counts", "error", "message"),
    [
        (
            ["thrA", "wt"],
            {"thrA": 6, "wt": 4},
            m.RawColumnCountError,
            "'thrA': 6 raw columns for 1 Table EV1A entries (expected 4 each)",
        ),
        (["thrA"], {"thrA": 4}, m.UnmappedDepositColumnError, "no 'wt' column"),
        (["thrA", "wt"], {"thrA": 4, "wt": 3}, m.RawColumnCountError, "'wt' has 3"),
    ],
)
def test_select_strains_refuses_unexpected_deposits(
    columns: list[str], counts: dict[str, int], error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message.replace("(", r"\(").replace(")", r"\)")):
        m.select_strains(columns, _entries(), counts)


def test_ion_alignment_counts_rows_nearest_their_own_index() -> None:
    result = m.ion_alignment("neg", NEG_MZ, EV1B_NEG_MZ)
    assert result.nearest_row_agrees == 3
    assert result.agreement == 1.0
    assert result.max_abs_mz_difference == pytest.approx(0.001)
    with pytest.raises(ValueError, match="neg: 3 deposit ions but 2 Table EV1B ions"):
        m.ion_alignment("neg", NEG_MZ, [100.0, 200.0])
    with pytest.raises(ValueError, match="only 1 of 3 Table EV1B ions"):
        m.ion_alignment("neg", NEG_MZ, [300.0, 200.0, 100.0])


def test_zscore_scale_accepts_standardized_rows_and_refuses_others() -> None:
    scale = m.zscore_scale("neg", np.asarray(NEG_ROWS))
    assert scale.max_abs_column_median == 0.0
    assert scale.min_column_sd == pytest.approx(1.0, abs=1e-4)
    with pytest.raises(ValueError, match="neg: released z-scores are not standardized"):
        m.zscore_scale("neg", np.asarray(NEG_ROWS) * 2.0)


def test_read_zscores_refuses_a_wrong_shape_and_a_nan(tmp_path: Path) -> None:
    path = tmp_path / "z.tsv"
    path.write_text("0.1\t0.2\n0.3\tnan\n")
    with pytest.raises(ValueError, match=r"shape \(2, 2\), expected \(2, 3\)"):
        m.read_zscores(path, 2, 3)
    with pytest.raises(m.NonFiniteZScoreError) as err:
        m.read_zscores(path, 2, 2)
    assert "1 non-finite cells, first at row 2, column 2" in str(err.value)


def test_name_blocks_refuses_a_split_name_and_a_straddling_pair() -> None:
    assert m.name_blocks(["a", "a", "b", "b", "b", "b"]) == {"a": (0, 2), "b": (2, 6)}
    with pytest.raises(m.RawColumnCountError, match="'a' appears in two raw-matrix"):
        m.name_blocks(["a", "a", "b", "b", "a", "a"])
    with pytest.raises(m.RawColumnCountError, match=r"'a': raw columns 1..3 are not"):
        m.name_blocks(["a", "a", "a", "b"])


def test_select_clones_pairs_columns_and_leaves_the_withdrawn_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(m, "RAW_NAMES_ANALYZED", len(RAW_LAYOUT))
    monkeypatch.setattr(m, "WITHDRAWN_MUTANTS", ("flhC",))
    selection = m.select_strains(COLUMNS, _entries(), RAW_COUNTS)
    clones = m.select_clones(RAW_NAMES, selection)
    assert [(c.deposit_name, c.replicate_id, c.raw_columns) for c in clones.clones] == [
        ("hokC", "1", (4, 5)),
        ("hokC", "2", (6, 7)),
        ("ilvG", "1", (8, 9)),
        ("ilvG", "2", (10, 11)),
        ("ilvG", "3", (12, 13)),
        ("ilvG", "4", (14, 15)),
        ("thrA", "1", (16, 17)),
        ("thrA", "2", (18, 19)),
        ("thrL", "1", (20, 21)),
        ("thrL", "2", (22, 23)),
        ("wt", "1", (24, 25)),
        ("wt", "2", (26, 27)),
        ("wt", "3", (28, 29)),
        ("wt", "4", (30, 31)),
        ("yaaQ", "1", (32, 33)),
        ("yaaQ", "2", (34, 35)),
    ]
    assert clones.withdrawn == {"flhC": (0, 4)}
    assert [c.replicate_id for c in clones.wt_clones] == ["1", "2", "3", "4"]
    assert len(clones.mutant_clones) == 12
    assert m.deletion_genotype(clones.wt_clones[0]).perturbations == []
    assert [
        p.systematic_gene_name
        for p in m.deletion_genotype(clones.clones[2]).perturbations
    ] == ["JW3740", "JW3741"]


@pytest.mark.parametrize(
    ("raw_names", "withdrawn", "n_names", "message"),
    [
        (RAW_NAMES, ("flhC",), 8, "7 raw-matrix names, the paper states 8"),
        (RAW_NAMES, ("yhfW",), 7, r"raw-only names \['flhC'\], expected the withdrawn"),
        (RAW_NAMES[4:], ("flhC",), 6, r"raw-only names \[\], expected the withdrawn"),
        (
            [n for n in RAW_NAMES if n != "thrL"],
            ("flhC",),
            6,
            r"z-score names absent from the raw matrix: \['thrL'\]",
        ),
    ],
)
def test_select_clones_refuses_an_unexpected_raw_matrix(
    monkeypatch: pytest.MonkeyPatch,
    raw_names: list[str],
    withdrawn: tuple[str, ...],
    n_names: int,
    message: str,
) -> None:
    monkeypatch.setattr(m, "RAW_NAMES_ANALYZED", n_names)
    monkeypatch.setattr(m, "WITHDRAWN_MUTANTS", withdrawn)
    selection = m.select_strains(COLUMNS, _entries(), RAW_COUNTS)
    with pytest.raises(m.RawColumnCountError, match=message):
        m.select_clones(raw_names, selection)


def test_technical_pairing_measures_the_adjacent_split() -> None:
    """Two four-column names over 6 ions: each pair is one profile injected twice."""
    rng = np.random.default_rng(0)
    profiles = rng.uniform(500.0, 5000.0, size=(6, 4))
    matrix = np.repeat(profiles, 2, axis=1) * rng.uniform(0.99, 1.01, size=(6, 8))
    result = m.technical_pairing("neg", matrix, ["a"] * 4 + ["b"] * 4)
    assert (result.n_ions_used, result.n_four_column_names, result.adjacent_best) == (
        6,
        2,
        2,
    )
    assert result.agreement == 1.0
    assert result.within_pair_median_correlation > 0.9
    with pytest.raises(ValueError, match="best for only 0 of 2 four-column names"):
        m.technical_pairing(
            "neg", matrix[:, [0, 2, 1, 3, 4, 6, 5, 7]], ["a"] * 4 + ["b"] * 4
        )


def test_clone_phenotype_is_one_measurement_with_the_technical_pair_stated() -> None:
    phenotype = m.clone_phenotype(["neg_0001", "pos_0001"], [1200.5, -3.0], "2")
    assert phenotype.metabolite_level == {"neg_0001": 1200.5, "pos_0001": -3.0}
    assert phenotype.n_replicates == {"neg_0001": 1, "pos_0001": 1}
    assert (phenotype.replicate_id, phenotype.metabolite_level_se) == ("2", None)
    assert phenotype.measurement_type == "fia_tof_ms_ion_raw_intensity_clone"
    assert phenotype.provenance_gaps == [m.SINGLE_CLONE_SE_GAP]
    assert m.SINGLE_CLONE_SE_GAP.reason.value == "not_applicable_single_measurement"
    assert "n = 2" in (m.SINGLE_CLONE_SE_GAP.note or "")
    with pytest.raises(ValueError, match="2 keys for 1 values"):
        m.clone_phenotype(["neg_0001", "pos_0001"], [1.0], "1")


def test_clone_reference_is_the_wt_mean_with_its_standard_error() -> None:
    wt = np.array([[1.0, 3.0, 5.0], [10.0, 10.0, 10.0]])
    phenotype = m.clone_reference_phenotype(["neg_0001", "pos_0001"], wt)
    assert phenotype.metabolite_level == {"neg_0001": 3.0, "pos_0001": 10.0}
    assert phenotype.metabolite_level_se == pytest.approx(
        {"neg_0001": 2.0 / np.sqrt(3.0), "pos_0001": 0.0}
    )
    assert phenotype.n_replicates == {"neg_0001": 3, "pos_0001": 3}
    assert phenotype.replicate_id is None
    with pytest.raises(ValueError, match="wt profiles"):
        m.clone_reference_phenotype(["neg_0001"], wt)


def test_read_raw_intensities_keeps_negatives_and_refuses_nan(tmp_path: Path) -> None:
    path = tmp_path / "raw.tsv"
    path.write_text("-5\t0\n12\t7\n")
    assert m.read_raw_intensities(path, 2, 2).tolist() == [[-5.0, 0.0], [12.0, 7.0]]
    with pytest.raises(ValueError, match=r"shape \(2, 2\), expected \(3, 2\)"):
        m.read_raw_intensities(path, 3, 2)
    path.write_text("-5\tnan\n12\t7\n")
    with pytest.raises(m.NonFiniteZScoreError, match="non-finite raw intensities"):
        m.read_raw_intensities(path, 2, 2)


def test_the_runner_spec_matches_the_loader() -> None:
    from torchcell.verification.runners import METABOLITE_DATASETS

    spec = METABOLITE_DATASETS["metabolome_fuhrer2017"]
    assert spec["expected_count"] == m.EXPECTED_RECORDS == 11570
    assert spec["protocol_measurement_types"] == m.PROTOCOL_MEASUREMENT_TYPES
    assert spec["root"] == m.DATASET_ROOT_REL


def test_read_study_design(tmp_path: Path) -> None:
    _write_raw(tmp_path)
    assert m.read_study_design(tmp_path / m.STUDY_JSON) == (
        "2 Biological and 2 technical replicates"
    )


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #
def test_metabolite_phenotype_carries_the_clone_count_and_the_se_gap() -> None:
    phenotype = m.metabolite_phenotype(["neg_0001", "pos_0001"], [0.5, -1.25], 2)
    assert phenotype.metabolite_level == {"neg_0001": 0.5, "pos_0001": -1.25}
    assert phenotype.n_replicates == {"neg_0001": 2, "pos_0001": 2}
    assert phenotype.metabolite_level_se is None
    assert phenotype.measurement_type == "fia_tof_ms_ion_modified_z_score"
    assert phenotype.provenance_gaps == [m.METABOLITE_LEVEL_SE_GAP]
    assert phenotype.gapped_fields() == {"metabolite_level_se"}


def test_build_experiment_is_one_keio_deletion_under_its_released_jw_id() -> None:
    entry = m.KeioEntry(
        jw_id="JW0001",
        blattner_id="b0002",
        gene_name="thrA",
        delivery_status="ready to distribute",
    )
    strain = m.KeioStrain(column=2, deposit_name="thrA", entries=[entry], raw_columns=4)
    experiment = m.build_experiment(
        "MetabolomeFuhrer2017Dataset", strain, ["neg_0001"], [1.5], m.environment()
    )
    dumped = experiment.model_dump()
    assert dumped["experiment_type"] == "metabolite"
    assert dumped["environment"] == _expected_environment()
    (perturbation,) = dumped["genotype"]["perturbations"]
    assert {
        k: perturbation[k]
        for k in (
            "systematic_gene_name",
            "perturbed_gene_name",
            "perturbation_type",
            "gene_namespace",
            "strain_label",
            "identifier_kind",
            "released_kind",
            "collection",
            "cassette",
        )
    } == {
        "systematic_gene_name": "JW0001",
        "perturbed_gene_name": "thrA",
        "perturbation_type": "released_identifier",
        "gene_namespace": "ecoli_k12_bw25113_locus_tag",
        "strain_label": "thrA",
        "identifier_kind": "Keio JW id",
        "released_kind": "deletion",
        "collection": "KEIO knockout collection",
        "cassette": "kanamycin cassette flanked by FLP recognition target sites",
    }
    assert dumped["phenotype"]["metabolite_level"] == {"neg_0001": 1.5}
    assert dumped["phenotype"]["n_replicates"] == {"neg_0001": 2}


def test_build_reference_keeps_the_assembly_pin_through_a_dump() -> None:
    reference = m.build_reference(
        "MetabolomeFuhrer2017Dataset",
        REFERENCE,
        ["neg_0001"],
        [0.25],
        96,
        m.environment(),
    )
    dumped = reference.model_dump()
    assert dumped["genome_reference"]["assembly_set"] == "ecoli_K12_BW25113_ASM75055v1"
    assert dumped["genome_reference"]["assembly_accession"] == "GCA_000750555.1"
    assert dumped["phenotype_reference"]["metabolite_level"] == {"neg_0001": 0.25}
    assert dumped["phenotype_reference"]["n_replicates"] == {"neg_0001": 96}


# --------------------------------------------------------------------------- #
# Hermetic build
# --------------------------------------------------------------------------- #
def test_build_five_records_with_the_wt_reference_and_ledgers(
    synthetic: Path, presence_only_pins: list[Mapping[str, str]]
) -> None:
    """Five z-score records, one per mutant column, each under the JW id(s) Table EV1A
    lists for its name: ilvG is a pooled record with two JW ids, thrL and yaaQ carry JW
    ids the synthetic annotation does not have and are records all the same. Then the
    16 clone records (checked in the next test).
    """
    dataset = m.MetabolomeFuhrer2017Dataset(root=str(synthetic))
    assert presence_only_pins == [m.DATA_SHA256]
    assert len(dataset) == 21
    keys = ["neg_0001", "neg_0002", "neg_0003", "pos_0001", "pos_0002"]

    first = dataset[0]["experiment"]
    (hokc,) = first["genotype"]["perturbations"]
    assert (hokc["systematic_gene_name"], hokc["perturbed_gene_name"]) == (
        "JW4367",
        "hokC",
    )
    assert hokc["strain_label"] == "hokC"
    assert first["phenotype"]["metabolite_level"] == dict(
        zip(keys, [Z6[0], Z6[5], Z6[2], Z6[1], Z6[3]], strict=True)
    )
    assert first["phenotype"]["n_replicates"] == dict.fromkeys(keys, 2)
    assert first["environment"] == _expected_environment()

    pooled = dataset[1]["experiment"]
    assert [p["systematic_gene_name"] for p in pooled["genotype"]["perturbations"]] == [
        "JW3740",
        "JW3741",
    ]
    assert {p["strain_label"] for p in pooled["genotype"]["perturbations"]} == {"ilvG"}
    assert pooled["phenotype"]["n_replicates"] == dict.fromkeys(keys, 4)
    assert pooled["phenotype"]["metabolite_level"] == dict(
        zip(keys, [Z6[1], Z6[4], Z6[0], Z6[2], Z6[5]], strict=True)
    )

    second = dataset[2]["experiment"]
    assert second["genotype"]["perturbations"][0]["systematic_gene_name"] == "JW0001"
    assert second["phenotype"]["metabolite_level"] == dict(
        zip(keys, [Z6[2], Z6[3], Z6[4], Z6[0], Z6[1]], strict=True)
    )
    assert [
        dataset[i]["experiment"]["genotype"]["perturbations"][0]["systematic_gene_name"]
        for i in (3, 4)
    ] == ["JW8888", "JW9999"]

    reference = dataset[0]["reference"]
    assert reference["genome_reference"] == REFERENCE.model_dump()
    assert reference["phenotype_reference"]["metabolite_level"] == dict(
        zip(keys, [Z6[4], Z6[1], Z6[5], Z6[3], Z6[2]], strict=True)
    )
    assert reference["phenotype_reference"]["n_replicates"] == dict.fromkeys(keys, 4)
    assert dataset[0]["publication"] == m.PUBLICATION.model_dump()

    preprocess = synthetic / "preprocess"
    drops = json.loads((preprocess / "dropped_records.json").read_text())
    assert (
        drops["deposit_columns"],
        drops["source_records"],
        drops["kept_records"],
        drops["dropped_records"],
        drops["rules"],
    ) == (
        6,
        23,
        21,
        2,
        [
            {
                "rule": "withdrawn_by_authors",
                "description": drops["rules"][0]["description"],
                "n_records": 2,
                "items": ["flhC: raw columns 1..4"],
            }
        ],
    )
    assert drops["rules"][0]["description"].startswith(
        repr(m.SOURCED_VALUES["withdrawn_mutants"].quote)
    )
    assert (
        "ilvG: JW3740 b3767 (ready to distribute), JW3741 b3768 (ready to "
        "distribute); 2 loci"
    ) in drops["notes"]
    assert not (preprocess / "identifier_reconciliation.json").exists()
    ions = (preprocess / "ions.csv").read_text().splitlines()
    assert ions[0] == "key,mode,ion_index,deposit_mz,table_ev1b_mz"
    assert ions[1] == "neg_0001,neg,1,100.0,100.001"
    strains = (preprocess / "strains.csv").read_text().splitlines()
    assert strains[0] == (
        "record,deposit_column,deposit_name,jw_ids,blattner_ids,delivery_statuses,"
        "n_entries,n_biological"
    )
    assert strains[1:] == [
        "0,1,hokC,JW4367,b0001,ready to distribute,1,2",
        "1,2,ilvG,JW3740;JW3741,b3767;b3768,ready to distribute;ready to distribute,"
        "2,4",
        "2,3,thrA,JW0001,b0002,ready to distribute,1,2",
        "3,4,thrL,JW8888,b0099,ready to distribute,1,2",
        "4,6,yaaQ,JW9999,b9999,ready to distribute,1,2",
    ]
    scale = json.loads((preprocess / "zscore_scale.json").read_text())
    assert [s["mode"] for s in scale] == ["neg", "pos"]


def test_build_one_record_per_clone_with_the_wt_mean_reference(synthetic: Path) -> None:
    """Records 5..20 are the clones in release order, the withdrawn flhC left out: each
    is the mean of its two injections, n = 1, its position as replicate_id; the wt
    cultures are records with an empty genotype; the reference is their mean.
    """
    dataset = m.MetabolomeFuhrer2017Dataset(root=str(synthetic))
    keys = ["neg_0001", "neg_0002", "neg_0003", "pos_0001", "pos_0002"]
    expected = [
        (name, str(k + 1), g)
        for name, clones in CLONES_OF.items()
        if name != "flhC"
        for k, g in enumerate(clones)
    ]
    for idx, (name, replicate_id, g) in enumerate(expected, start=5):
        experiment = dataset[idx]["experiment"]
        phenotype = experiment["phenotype"]
        assert phenotype["measurement_type"] == m.RAW_MEASUREMENT_TYPE
        assert phenotype["replicate_id"] == replicate_id
        assert phenotype["metabolite_level"] == {
            key: float(_raw(g, r) + 1) for r, key in enumerate(keys)
        }
        assert phenotype["n_replicates"] == dict.fromkeys(keys, 1)
        assert phenotype["metabolite_level_se"] is None
        (gap,) = phenotype["provenance_gaps"]
        assert (gap["field"], gap["reason"]) == (
            "metabolite_level_se",
            "not_applicable_single_measurement",
        )
        labels = {p["strain_label"] for p in experiment["genotype"]["perturbations"]}
        assert labels == (set() if name == "wt" else {name})
    # the pooled name's four clones each keep both JW ids (the release does not say
    # which entry a clone is)
    ilvg = dataset[7]["experiment"]["genotype"]["perturbations"]
    assert [p["systematic_gene_name"] for p in ilvg] == ["JW3740", "JW3741"]
    # one hand-computed value: hokC clone 1 (g = 2), neg row 1: 1000 + 37 * 10 + 1
    assert dataset[5]["experiment"]["phenotype"]["metabolite_level"]["neg_0001"] == (
        1371.0
    )

    reference = dataset[5]["reference"]["phenotype_reference"]
    wt = np.array([[_raw(g, r) + 1 for g in CLONES_OF["wt"]] for r in range(5)])
    assert reference["measurement_type"] == m.RAW_MEASUREMENT_TYPE
    assert reference["replicate_id"] is None
    assert reference["metabolite_level"] == pytest.approx(
        dict(zip(keys, wt.mean(axis=1), strict=True))
    )
    assert reference["metabolite_level_se"] == pytest.approx(
        dict(zip(keys, wt.std(axis=1, ddof=1) / 2.0, strict=True))
    )
    assert reference["n_replicates"] == dict.fromkeys(keys, 4)
    # the z-score records keep their own reference
    assert dataset[0]["reference"]["phenotype_reference"]["measurement_type"] == (
        m.MEASUREMENT_TYPE
    )

    preprocess = synthetic / "preprocess"
    clones = (preprocess / "clones.csv").read_text().splitlines()
    assert clones[0] == (
        "record,deposit_name,replicate_id,raw_column_first,raw_column_second,jw_ids,"
        "wild_type"
    )
    assert clones[1] == "5,hokC,1,5,6,JW4367,False"
    assert clones[-1] == "20,yaaQ,2,35,36,JW9999,False"
    pairing = json.loads((preprocess / "technical_pairing.json").read_text())
    assert [
        (p["mode"], p["n_four_column_names"], p["adjacent_best"]) for p in pairing
    ] == [("neg", 5, 5), ("pos", 5, 5)]
    agreement = json.loads((preprocess / "raw_zscore_agreement.json").read_text())
    assert agreement["n_strains"] == 5


def test_build_refuses_a_design_other_than_two_by_two(synthetic: Path) -> None:
    study = synthetic / "raw" / m.STUDY_JSON
    study.write_text(study.read_text().replace("2 Biological", "3 Biological"))
    with pytest.raises(ValueError, match="S-BSST5 design '3 Biological"):
        m.MetabolomeFuhrer2017Dataset(root=str(synthetic))


def test_build_refuses_a_column_count_other_than_the_declared_one(
    synthetic: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(m, "EXPECTED_ZSCORE_RECORDS", 6)
    with pytest.raises(
        ValueError, match="5 z-score mutant columns, the module declares 6"
    ):
        m.MetabolomeFuhrer2017Dataset(root=str(synthetic))


def test_build_refuses_a_clone_count_other_than_the_declared_one(
    synthetic: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(m, "EXPECTED_CLONE_RECORDS", 18)
    with pytest.raises(ValueError, match="16 raw clones, the module declares 18"):
        m.MetabolomeFuhrer2017Dataset(root=str(synthetic))


def test_build_refuses_a_swapped_technical_pairing(
    synthetic: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Interleave every four-column block's injections (1, 3, 2, 4): the adjacent
    columns are then two different clones, and the build stops.
    """
    path = synthetic / "raw" / m.RAW_NEG
    rows = [line.split("\t") for line in path.read_text().splitlines()]
    starts = [0, 4, 16, 20, 32]  # flhC, hokC, thrA, thrL, yaaQ
    for row in rows:
        for a in starts:
            row[a + 1], row[a + 2] = row[a + 2], row[a + 1]
    path.write_text("".join("\t".join(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="neg: the adjacent technical pairing is best"):
        m.MetabolomeFuhrer2017Dataset(root=str(synthetic))


def test_process_verifies_the_real_pins(
    synthetic: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With the real check back, the synthetic study JSON is refused by its pin."""
    from torchcell.data import verify_raw_files

    monkeypatch.setattr(m, "verify_raw_files", verify_raw_files)
    with pytest.raises(
        RawSha256MismatchError, match=f"expected {m.DATA_SHA256[m.STUDY_JSON]}"
    ):
        m.MetabolomeFuhrer2017Dataset(root=str(synthetic))


def test_run_verification_reports_the_jw_ids_as_a_finding(
    synthetic: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    bw25113: EcoliK12BW25113Genome,
) -> None:
    root = tmp_path / "data" / "torchcell" / "metabolome_fuhrer2017"
    root.parent.mkdir(parents=True)
    synthetic.rename(root)
    m.MetabolomeFuhrer2017Dataset(root=str(root))
    monkeypatch.setattr(m, "bacterial_genome", lambda *a, **k: bw25113)
    monkeypatch.setattr(m, "SOURCED_VALUES", {})
    report = m.run_verification(str(tmp_path))
    assert report.passed, report.summary()
    finding = report.results[-1]
    assert finding.name == "gene_containment_bw25113_jw_synonyms" and finding.passed
    assert finding.details["n_identifiers"] == 6
    assert finding.details["unplaced"] == ["JW3740", "JW3741", "JW8888", "JW9999"]


# --------------------------------------------------------------------------- #
# Raw mirror
# --------------------------------------------------------------------------- #
def _synthetic_raw_files(
    tmp_path: Path,
) -> tuple[tuple[m.RawFile, ...], dict[str, Path]]:
    import hashlib

    sources: dict[str, Path] = {}
    files: list[m.RawFile] = []
    for name, payload in (("a.tsv", b"1\t2\n"), ("b.xls", b"bytes")):
        path = tmp_path / "download" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        sha = hashlib.sha256(payload).hexdigest()
        url = f"https://example.org/{name}"
        files.append(
            m.RawFile(
                name=name,
                sha256=sha,
                bytes=len(payload),
                description=name,
                retrieval=RetrievalRecord(
                    method=RetrievalMethod.direct_url,
                    source_url=url,
                    retriever="torchcell.literature.retrieve.direct_url",
                    params={"url": url},
                    sha256=sha,
                    retrieved_at="2026-10-07",
                ),
            )
        )
        sources[name] = path
    return tuple(files), sources


def test_deposit_is_idempotent_and_refuses_a_differing_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    files, sources = _synthetic_raw_files(tmp_path)
    monkeypatch.setattr(m, "RAW_FILES", files)
    monkeypatch.setattr(m, "DATA_SHA256", {f.name: f.sha256 for f in files})
    data_root = str(tmp_path / "root")
    root = m.deposit_raw_mirror(sources=sources, data_root=data_root)
    assert root == Path(data_root) / m.RAW_DIR_REL
    m.deposit_raw_mirror(sources=sources, data_root=data_root)
    manifest = m.load_manifest(data_root)
    assert [
        (f.path, f.sha256, f.retrieval.method if f.retrieval else None)
        for f in manifest.files
    ] == [
        ("data/a.tsv", files[0].sha256, RetrievalMethod.direct_url),
        ("data/b.xls", files[1].sha256, RetrievalMethod.direct_url),
    ]
    assert manifest.si_expected == list(m.NOT_MIRRORED)
    assert m.manifest_sha256(manifest, "data/b.xls") == files[1].sha256
    with pytest.raises(KeyError, match="data/c.xls is not in the raw-mirror manifest"):
        m.manifest_sha256(manifest, "data/c.xls")

    (root / "data" / "a.tsv").write_bytes(b"edited")
    with pytest.raises(RuntimeError, match="exists with a different sha256; refusing"):
        m.deposit_raw_mirror(sources=sources, data_root=data_root)
    (root / "data" / "a.tsv").write_bytes(b"1\t2\n")
    sources["b.xls"].write_bytes(b"other")
    with pytest.raises(RuntimeError, match="b.xls sha256 mismatch"):
        m.deposit_raw_mirror(sources=sources, data_root=data_root)
    with pytest.raises(KeyError, match="no source given for"):
        m.deposit_raw_mirror(sources={}, data_root=data_root)


def test_download_links_the_mirror_and_checks_the_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    files, sources = _synthetic_raw_files(tmp_path)
    monkeypatch.setattr(m, "RAW_FILES", files)
    monkeypatch.setattr(m, "DATA_SHA256", {f.name: f.sha256 for f in files})
    data_root = tmp_path / "root"
    m.deposit_raw_mirror(sources=sources, data_root=str(data_root))
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    dataset = object.__new__(m.MetabolomeFuhrer2017Dataset)
    monkeypatch.setattr(
        m.MetabolomeFuhrer2017Dataset, "raw_dir", str(tmp_path / "raw"), raising=False
    )
    dataset.download()
    linked = tmp_path / "raw" / "a.tsv"
    assert linked.is_symlink()
    assert os.readlink(linked) == str(data_root / m.RAW_DIR_REL / "data" / "a.tsv")

    manifest_path = data_root / m.RAW_DIR_REL / "manifest.json"
    manifest_path.write_text(
        manifest_path.read_text().replace(files[0].sha256, "0" * 64)
    )
    with pytest.raises(ManifestPinMismatchError, match="data/a.tsv"):
        dataset.download()


def test_retrieve_runs_the_recorded_retriever_and_verifies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    files, _ = _synthetic_raw_files(tmp_path)
    monkeypatch.setattr(m, "RAW_FILES", files)
    calls: list[RetrievalRecord] = []

    def fake(record: RetrievalRecord) -> bytes:
        calls.append(record)
        return b"1\t2\n" if record.params["url"].endswith("a.tsv") else b"wrong"

    monkeypatch.setattr(m, "run_retriever", fake)
    out = m.retrieve_raw_files(tmp_path / "fetched", ["a.tsv"])
    assert out == {"a.tsv": tmp_path / "fetched" / "a.tsv"}
    assert [c.source_url for c in calls] == ["https://example.org/a.tsv"]
    with pytest.raises(RawSha256MismatchError):
        m.retrieve_raw_files(tmp_path / "fetched", ["b.xls"])
    assert not (tmp_path / "fetched" / "b.xls").exists()


def test_every_consumed_file_is_pinned_once() -> None:
    assert [f.name for f in m.RAW_FILES] == [
        "S-BSST5.json",
        "zscore_neg.tsv",
        "zscore_pos.tsv",
        "sample_id_zscore.xls",
        "sample_id_all.xls",
        "neg_ionMz.xls",
        "pos_ionMz.xls",
        "rawdata_neg_all.tsv",
        "rawdata_pos_all.tsv",
        "si2.xlsx",
    ]
    assert all(
        len(f.sha256) == 64 and f.sha256 == f.retrieval.sha256 for f in m.RAW_FILES
    )
    assert m.RAW_FILES_BY_NAME["si2.xlsx"].retrieval.method == RetrievalMethod.pmc_cloud


def test_sourced_value_roots_split_the_two_mirrors(tmp_path: Path) -> None:
    root = str(tmp_path)
    assert m.sourced_value_root(m.SOURCED_VALUES["design"], root) == (
        tmp_path / "torchcell-raw"
    )
    assert m.sourced_value_root(m.SOURCED_VALUES["temperature_c"], root) == (
        tmp_path / "torchcell-library"
    )


# --------------------------------------------------------------------------- #
# Real data (``--data``): the mirror and the built dev-tree LMDB, never rebuilt
# --------------------------------------------------------------------------- #
def _real_data_root() -> str:
    return os.environ["DATA_ROOT"]


def _built_root() -> str:
    """The built store the data tests read: the dev tree's, or ``FUHRER2017_ROOT``
    (a scratch build of a branch, since the dev stores are shared).
    """
    return os.environ.get(
        "FUHRER2017_ROOT", osp.join(_real_data_root(), m.DATASET_ROOT_REL)
    )


@pytest.mark.data
def test_real_mirror_matches_the_module_pins() -> None:
    from torchcell.data.experiment_dataset import file_sha256

    manifest = m.load_manifest(_real_data_root())
    for raw in m.RAW_FILES:
        assert m.manifest_sha256(manifest, raw.mirror_relpath) == raw.sha256
        path = m.raw_mirror_dir(_real_data_root()) / raw.mirror_relpath
        assert path.stat().st_size == raw.bytes
        assert file_sha256(path) == raw.sha256


@pytest.mark.data
@pytest.mark.parametrize("key", sorted(m.SOURCED_VALUES))
def test_real_sourced_values_are_verbatim(key: str) -> None:
    value = m.SOURCED_VALUES[key]
    result = audit_sourced_value(value, m.sourced_value_root(value, _real_data_root()))
    assert result.passed, result.message


@pytest.mark.data
def test_real_release_has_two_clones_per_strain_and_96_wt_cultures() -> None:
    """The pinned raw matrices' names, read without building: 3,810 names, 7,764 clone
    records after the three withdrawn mutants (6 clones), 3,781 single-entry names with
    clones "1" and "2", 25 pooled names with 4 or 6 clones, 96 wt cultures.
    """
    data = m.raw_mirror_dir(_real_data_root()) / "data"
    names = m.read_name_column(data / m.SAMPLE_ID_ALL)
    selection = m.select_strains(
        m.read_name_column(data / m.SAMPLE_ID_ZSCORE),
        m.read_table_ev1a(data / m.TABLE_EV1),
        Counter(names),
    )
    clones = m.select_clones(names, selection)
    assert len(clones.clones) == m.EXPECTED_CLONE_RECORDS == 7764
    assert len(clones.wt_clones) == 96
    assert {n: (b - a) // 2 for n, (a, b) in clones.withdrawn.items()} == {
        "flhC": 2,
        "hrpB": 2,
        "yhfW": 2,
    }
    per_position = Counter(c.replicate_id for c in clones.mutant_clones)
    assert per_position == {"1": 3806, "2": 3806, "3": 25, "4": 25, "5": 3, "6": 3}
    per_name = Counter(Counter(c.deposit_name for c in clones.mutant_clones).values())
    assert per_name == {2: 3781, 4: 22, 6: 3}


@pytest.mark.data
def test_real_build_counts_and_the_thra_record() -> None:
    """3,806 z-score records, one per mutant column, 25 of them pooled names with two
    or three JW ids, then 7,764 clone records; thrA (JW0001) with values read off the
    deposit by hand.
    """
    from torchcell.verification.runners import load_records

    drops = json.loads(
        Path(_built_root(), "preprocess", "dropped_records.json").read_text()
    )
    assert (
        drops["source_records"],
        drops["kept_records"],
        drops["dropped_records"],
        [r["rule"] for r in drops["rules"]],
    ) == (11576, 11570, 6, ["withdrawn_by_authors"])
    all_records = load_records(_built_root())
    assert len(all_records) == 11570
    by_type = Counter(
        r["experiment"]["phenotype"]["measurement_type"] for r in all_records
    )
    assert by_type == {m.MEASUREMENT_TYPE: 3806, m.RAW_MEASUREMENT_TYPE: 7764}
    # load_records returns LMDB key order ("0", "1", "10", ...), so split by type
    records = [
        r
        for r in all_records
        if r["experiment"]["phenotype"]["measurement_type"] == m.MEASUREMENT_TYPE
    ]
    clone_records = [
        r
        for r in all_records
        if r["experiment"]["phenotype"]["measurement_type"] == m.RAW_MEASUREMENT_TYPE
    ]
    # 25 pooled names: 20 over two or three loci, dcuC over two JW ids of one locus,
    # and 4 that list one JW id twice (one perturbation each)
    assert (
        sum(1 for r in records if len(r["experiment"]["genotype"]["perturbations"]) > 1)
        == 21
    )
    (thra,) = [
        r
        for r in records
        if r["experiment"]["genotype"]["perturbations"][0]["systematic_gene_name"]
        == "JW0001"
    ]
    perturbation = thra["experiment"]["genotype"]["perturbations"][0]
    assert perturbation["perturbed_gene_name"] == "thrA"
    assert perturbation["strain_label"] == "thrA"
    levels = thra["experiment"]["phenotype"]["metabolite_level"]
    assert len(levels) == 3169 + 4365
    assert {k: levels[k] for k in HAND_CHECKED_THRA} == HAND_CHECKED_THRA
    assert set(thra["experiment"]["phenotype"]["n_replicates"].values()) == {2}
    reference = thra["reference"]["phenotype_reference"]
    assert {k: reference["metabolite_level"][k] for k in HAND_CHECKED_WT} == (
        HAND_CHECKED_WT
    )
    assert set(reference["n_replicates"].values()) == {96}

    thra_clones = sorted(
        (
            r
            for r in clone_records
            if [
                p["systematic_gene_name"]
                for p in r["experiment"]["genotype"]["perturbations"]
            ]
            == ["JW0001"]
        ),
        key=lambda r: int(r["experiment"]["phenotype"]["replicate_id"]),
    )
    assert [r["experiment"]["phenotype"]["replicate_id"] for r in thra_clones] == [
        "1",
        "2",
    ]
    for clone, expected in zip(thra_clones, HAND_CHECKED_THRA_CLONES, strict=True):
        levels = clone["experiment"]["phenotype"]["metabolite_level"]
        assert {k: levels[k] for k in expected} == expected
        assert set(clone["experiment"]["phenotype"]["n_replicates"].values()) == {1}
        assert clone["experiment"]["phenotype"]["metabolite_level_se"] is None
    wt_clones = [
        r for r in clone_records if not r["experiment"]["genotype"]["perturbations"]
    ]
    assert len(wt_clones) == 96
    clone_reference = thra_clones[0]["reference"]["phenotype_reference"]
    assert clone_reference["replicate_id"] is None
    assert set(clone_reference["n_replicates"].values()) == {96}
    assert clone_reference["metabolite_level"]["neg_0001"] == pytest.approx(
        np.mean(
            [
                r["experiment"]["phenotype"]["metabolite_level"]["neg_0001"]
                for r in wt_clones
            ]
        )
    )


@pytest.mark.data
@pytest.mark.timeout(1800)  # L0 validates 11,570 records of 7,534 ions (about 5 min)
def test_real_build_passes_l0_to_l4() -> None:
    report = m.run_verification(_real_data_root(), _built_root())
    assert report.passed, report.summary()


#: Read off the deposit by hand (thrA is entry 1,915 and wt entry 2,060 of
#: ``sample_id_zscore.xls``): ``sed -n <row>p zscore_neg.tsv | cut -f 1915`` for rows 1
#: and 3169, the same on ``zscore_pos.tsv`` for rows 1 and 4365, and ``cut -f 2060``.
HAND_CHECKED_THRA: dict[str, float] = {
    "neg_0001": 1.3524,
    "neg_3169": 0.4154,
    "pos_0001": 0.6878,
    "pos_4365": 1.3324,
}
HAND_CHECKED_WT: dict[str, float] = {"neg_0001": -0.1128, "pos_4365": -0.0166}
#: thrA's two clones, read off the raw matrices by hand (thrA is columns 7,709-7,712 of
#: ``sample_id_all.xls``): ``sed -n <row>p rawdata_neg_all.tsv | cut -f 7709-7712`` for
#: rows 1 and 3169 gives 5765 5269 5730 5814 and 196 440 543 542, the same on
#: ``rawdata_pos_all.tsv`` rows 1 and 4365 gives 957 1624 1316 1107 and 799 714 923 999;
#: each clone is the mean of its two adjacent columns.
HAND_CHECKED_THRA_CLONES: list[dict[str, float]] = [
    {"neg_0001": 5517.0, "neg_3169": 318.0, "pos_0001": 1290.5, "pos_4365": 756.5},
    {"neg_0001": 5772.0, "neg_3169": 542.5, "pos_0001": 1211.5, "pos_4365": 961.0},
]
