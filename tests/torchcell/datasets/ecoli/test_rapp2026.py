# tests/torchcell/datasets/ecoli/test_rapp2026.py
# [[tests.torchcell.datasets.ecoli.test_rapp2026]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/ecoli/test_rapp2026.py
"""The Rapp 2026 CRISPRi metabolome loader (``torchcell.datasets.ecoli.rapp2026``).

Synthetic tests (run everywhere) build all five workbooks in ``tmp_path``. The hermetic
build uses the real ``EcoliK12MG1655Genome`` over a synthetic MG1655 assembly derived
from ``tests/torchcell/sequence/genome/_bacterial_fixtures.py``, with ``b0099`` added as
a ``gene_synonym`` of ``b0005`` ``proB`` so the remap rule has something to catch, served
through a stubbed ``resolve`` with the network refused. ``verify_raw_files`` is replaced
by a presence check (the synthetic files cannot carry the real pins; the pins are
asserted by the refusal test and by the data-gated tests). The synthetic screen:

    strain   b-number   outcome
    thrL     b0001      kept
    thrA     b0002      kept
    ghostG   b0099      dropped: a gene_synonym of b0005, so the record would store
                        another locus and no DerivedIdentifierRoute describes that
    oddS     b0000      dropped: no assigned target and no Table S1 sgRNA
    ctrl1    b0000      the reference
    ctrl2    b0000      the reference

Three Table S9 metabolites in two adducts each give six features, one of which is empty
in every column (so the all-or-nothing rule and the empty-row count are exercised), and
``didp`` is the real table's arity anomaly: two ``; ``-separated names under one BiGG id,
which must count as ONE identity.

Data-gated tests (``@pytest.mark.data``) read the real raw mirror and the built dev-tree
LMDB under ``$DATA_ROOT`` (they never build it): the manifest pins, the provenance audit
of every sourced value, the record count, two hand-checked records (``aaeA`` and the
paper's ``ispB`` frdp accumulation, read off ``si5.xlsx`` with ``openpyxl``), the control
reference, and the L0-L4 verifier.
"""

from __future__ import annotations

import json
import os
import os.path as osp
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import openpyxl
import pytest

import torchcell.datasets.ecoli.rapp2026 as m
from tests.torchcell.sequence.genome._bacterial_fixtures import (
    MG1655_GAF,
    MG1655_LOCI,
    forbid_network,
    serve_tier,
    write_assembly,
)
from torchcell.data import ManifestPinMismatchError, RawSha256MismatchError
from torchcell.datamodels.schema import (
    AssemblyReferenceGenome,
    SmallMoleculePerturbation,
)
from torchcell.datasets.bacteria_common import LocusTagResolutionError
from torchcell.literature.manifest import RetrievalMethod, RetrievalRecord
from torchcell.sequence.genome.ecoli.k12 import MG1655_ASSEMBLY, EcoliK12MG1655Genome
from torchcell.verification.sourced import audit_sourced_value

# --------------------------------------------------------------------------- #
# The synthetic screen
# --------------------------------------------------------------------------- #
#: ``b0099`` as a ``gene_synonym`` of ``b0005`` is what the remap rule catches.
SYNTHETIC_LOCI = [
    locus.model_copy(update={"synonyms": ("ECK0005", "b0099")})
    if locus.tag == "b0005"
    else locus
    for locus in MG1655_LOCI
]

GUIDES: list[tuple[str, str, str, str]] = [
    ("thrL", "thrL #1", "b0001", "ACGTACGTACGTACGTACGT"),
    ("thrA", "thrA #2", "b0002", "TTTTCCCCGGGGAAAATTTT"),
    ("proB", "proB #1", "b0005", "GGGGAAAATTTTCCCCGGGG"),
    ("ghostG", "ghostG #1", "b0099", "CCCCTTTTAAAAGGGGCCCC"),
]
#: ``(gene, b-number, plate, well)``; each strain gets replicates R1 and R2 in batch 1.
STRAINS: list[tuple[str, str, str, str]] = [
    ("thrL", "b0001", "1", "A2"),
    ("thrA", "b0002", "1", "A3"),
    ("ghostG", "b0099", "2", "B4"),
    ("oddS", "b0000", "2", "C5"),
    ("ctrl1", "b0000", "1", "A1"),
    ("ctrl2", "b0000", "2", "A1"),
]
#: ``(abbreviation, BiGG, names, KEGG, monoisotopic mass, formula)``.
METABOLITES: list[tuple[str, str, str, str, float, str]] = [
    ("ppal", "ppal", "Propanal", "C00479", 58.0419, "C3H6O"),
    (
        "ac-gcald",
        "ac-gcald",
        "Acetate; Glycolaldehyde",
        "C00033-C00266",
        60.0211,
        "C2H4O2",
    ),
    (
        "didp",
        "didp",
        "DIDP; 2'-deoxyinosine-5'-diphosphate(3-)",
        "C01344",
        412.0291,
        "C10H14N4O10P2",
    ),
]
ADDUCTS = ("[M+H]+", "[M-H]-")
#: Six values of 0.5 and six of 1.5 have median exactly 1 over the twelve samples.
LOW, HIGH = 0.5, 1.5
#: One value per sample column, per feature row, in ``feature_keys()`` order. Row 4
#: (``ac-gcald[M-H]-``) is empty in every column.
PATTERNS: list[list[int] | None] = [
    [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
    [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
    None,
    [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
    [0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0],
]


def sample_columns() -> list[str]:
    """The twelve synthetic sample ids, in Table S4's column order."""
    return [
        f"{gene}_R{replicate}_msSYN{index:03d}_B1"
        for index, (gene, _, _, _) in enumerate(STRAINS)
        for replicate in (1, 2)
    ]


def feature_keys() -> list[str]:
    """The six synthetic Table S4 feature keys, in row order."""
    return [
        f"{abbreviation}{adduct}"
        for abbreviation, _, _, _, _, _ in METABOLITES
        for adduct in ADDUCTS
    ]


def feature_mass(abbreviation: str, adduct: str) -> float:
    """``[M+H]+`` is the monoisotopic mass plus the proton; ``[M-H]-`` minus it."""
    mono = next(row[4] for row in METABOLITES if row[0] == abbreviation)
    return mono + (m.PROTON_MASS if adduct == "[M+H]+" else -m.PROTON_MASS)


def feature_values() -> list[list[float] | None]:
    """The synthetic value matrix, as a list of rows (``None`` = an empty row)."""
    return [
        None if pattern is None else [HIGH if flag else LOW for flag in pattern]
        for pattern in PATTERNS
    ]


def expected_level(gene: str, key: str) -> float:
    """The mean of ``gene``'s two plate values for ``key``."""
    index = [row[0] for row in STRAINS].index(gene)
    row = feature_values()[feature_keys().index(key)]
    assert row is not None
    return float(np.mean(row[2 * index : 2 * index + 2]))


def _write_table_s1(path: Path, guides: Sequence[tuple[str, str, str, str]]) -> None:
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    assert sheet is not None
    sheet.title = m.TABLE_S1_SHEET
    sheet.append(
        ["Gene", "sgRNA Nr.", "b-Nr.", "base pairing region", "Oligo sequence"]
    )
    for gene, sgrna, b_number, spacer in guides:
        sheet.append([gene, sgrna, b_number, spacer, f"gatc{spacer.lower()}gttt"])
    workbook.save(path)


def _write_table_s3(path: Path) -> None:
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    assert sheet is not None
    sheet.title = m.TABLE_S3_SHEET
    sheet.append(
        ["Target gene", "b number", "OD", "Plate ID", "Well", "Replicate", "Sample ID"]
    )
    for column in sample_columns():
        parsed = m.parse_sample_id(column)
        _, b_number, plate, well = next(row for row in STRAINS if row[0] == parsed.gene)
        sheet.append(
            [
                parsed.gene,
                b_number,
                0.5 + 0.1 * parsed.replicate,
                plate,
                well,
                f"R{parsed.replicate}",
                column,
            ]
        )
    workbook.save(path)


def _write_table_s9(path: Path) -> None:
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    assert sheet is not None
    sheet.title = m.TABLE_S9_SHEET
    sheet.append(
        [
            "Abbreviation",
            "BIGG",
            "Metabolite",
            "KEGG",
            "Monoisotopic mass",
            "Neutral Formula",
        ]
    )
    for abbreviation, bigg, names, kegg, mass, formula in METABOLITES:
        sheet.append([abbreviation, bigg, names, kegg, mass, formula])
    workbook.save(path)


def _write_table_s4(
    path: Path, rows: Sequence[list[float] | None] | None = None
) -> None:
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    assert sheet is not None
    sheet.title = m.TABLE_S4_SHEET
    sheet.append(["Abbr", "Metabolite", "Mass", "Kegg", *sample_columns()])
    values = feature_values() if rows is None else rows
    for key, row in zip(feature_keys(), values, strict=True):
        abbreviation, adduct = key.split("[")[0], "[" + key.split("[")[1]
        kegg = next(entry[3] for entry in METABOLITES if entry[0] == abbreviation)
        names = next(entry[2] for entry in METABOLITES if entry[0] == abbreviation)
        sheet.append(
            [
                key,
                f"{names}{adduct}",
                feature_mass(abbreviation, adduct),
                kegg,
                *(row if row is not None else [None] * len(sample_columns())),
            ]
        )
    workbook.save(path)


def _write_table_s5(path: Path, pairs: Sequence[tuple[str, str, float]]) -> None:
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    assert sheet is not None
    sheet.title = m.TABLE_S5_SHEET
    sheet.append(["Gene", "Metabolite Abbreviation", "Mode", "Mean_FC"])
    for gene, abbreviation, mean in pairs:
        index = abbreviation.index("[")
        sheet.append([gene, abbreviation[:index], abbreviation[index:], mean])
    workbook.save(path)


ACCUMULATIONS = [
    ("thrA", "ppal[M+H]+", expected_level("thrA", "ppal[M+H]+")),
    ("thrL", "didp[M-H]-", expected_level("thrL", "didp[M-H]-")),
]


def _write_raw(raw: Path) -> None:
    raw.mkdir(parents=True, exist_ok=True)
    _write_table_s1(raw / m.TABLE_S1, GUIDES)
    _write_table_s3(raw / m.TABLE_S3)
    _write_table_s4(raw / m.TABLE_S4)
    _write_table_s5(raw / m.TABLE_S5, ACCUMULATIONS)
    _write_table_s9(raw / m.TABLE_S9)


REFERENCE = AssemblyReferenceGenome(
    species="Escherichia coli",
    strain=m.HOST_STRAIN,
    assembly_set=m.MG1655_ASSEMBLY_SET,
    assembly_accession="GCA_000005845.2",
    background=m.host_background(),
)


@pytest.fixture
def mg1655(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> EcoliK12MG1655Genome:
    """The real MG1655 class over the synthetic assembly; network refused."""
    files = write_assembly(
        tmp_path / "tier", MG1655_ASSEMBLY, SYNTHETIC_LOCI, gaf_rows=MG1655_GAF
    )
    forbid_network(monkeypatch)
    serve_tier(monkeypatch, files)
    root = tmp_path / "mg1655"
    root.mkdir()
    return EcoliK12MG1655Genome(genome_root=str(root), overwrite=False)


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
def small_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    """The paper's counts, scaled to the synthetic screen."""
    monkeypatch.setattr(m, "LIBRARY_GENES", len(GUIDES))
    monkeypatch.setattr(m, "N_EXTRACTS", len(sample_columns()))
    monkeypatch.setattr(m, "ISOBARIC_METABOLITES", len(METABOLITES))
    monkeypatch.setattr(m, "CONTROL_REPLICATES", 2)


@pytest.fixture
def synthetic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    small_counts: None,
    mg1655: EcoliK12MG1655Genome,
    presence_only_pins: list[Mapping[str, str]],
) -> Path:
    """A dataset root whose ``raw/`` holds the synthetic workbooks."""
    root = tmp_path / "metabolome_rapp2026"
    _write_raw(root / "raw")
    monkeypatch.setattr(m, "assembly_reference", lambda strain, **_: REFERENCE)
    return root


# --------------------------------------------------------------------------- #
# Pure parsing
# --------------------------------------------------------------------------- #
def test_parse_sample_id_reads_gene_replicate_injection_and_batch() -> None:
    parsed = m.parse_sample_id("aaeA_R2_msAV933_B1")
    assert (parsed.gene, parsed.replicate, parsed.injection, parsed.batch) == (
        "aaeA",
        2,
        "msAV933",
        1,
    )
    assert m.parse_sample_id("ctrl10_R1_msAV948_B3").gene == "ctrl10"
    with pytest.raises(ValueError, match="is not a <gene>_R<n>_<injection>_B<n>"):
        m.parse_sample_id("aaeA_R2_AV933_B1")


def test_control_and_target_tokens_are_told_apart() -> None:
    def strain(gene: str, b_number: str) -> m.StrainSamples:
        return m.StrainSamples(
            gene=gene,
            b_number=b_number,
            columns=(0, 1),
            rows=(
                m.SampleRow(
                    sample=m.parse_sample_id(f"{gene}_R1_msSYN001_B1"),
                    b_number=b_number,
                    optical_density=0.5,
                    plate="1",
                    well="A1",
                ),
            ),
        )

    assert strain("ctrl7", "b0000").is_control
    assert not strain("ctrl7", "b0000").has_target
    assert not strain("argR", "b0000").is_control
    assert not strain("argR", "b0000").has_target
    assert strain("aaeA", "b3241").has_target


def test_read_guides_requires_one_acgt_spacer_per_library_gene(
    tmp_path: Path, small_counts: None
) -> None:
    _write_table_s1(tmp_path / m.TABLE_S1, GUIDES)
    guides = m.read_guides(tmp_path / m.TABLE_S1)
    assert set(guides) == {gene for gene, _, _, _ in GUIDES}
    assert guides["thrA"].spacer == "TTTTCCCCGGGGAAAATTTT"
    assert guides["thrA"].sgrna_id == "thrA #2"

    _write_table_s1(tmp_path / "bad_spacer.xlsx", [("x", "x #1", "b0001", "ACGTN")])
    with pytest.raises(ValueError, match="spacer 'ACGTN' is not ACGT"):
        m.read_guides(tmp_path / "bad_spacer.xlsx")

    _write_table_s1(tmp_path / "dup.xlsx", [GUIDES[0], GUIDES[0]])
    with pytest.raises(ValueError, match="Table S1 lists 'thrL' more than once"):
        m.read_guides(tmp_path / "dup.xlsx")

    _write_table_s1(tmp_path / "short.xlsx", GUIDES[:2])
    with pytest.raises(ValueError, match="Table S1 holds 2 genes, the paper states 4"):
        m.read_guides(tmp_path / "short.xlsx")


def test_read_sample_rows_keys_on_the_sample_id(
    tmp_path: Path, small_counts: None
) -> None:
    _write_table_s3(tmp_path / m.TABLE_S3)
    rows = m.read_sample_rows(tmp_path / m.TABLE_S3)
    assert set(rows) == set(sample_columns())
    row = rows["thrA_R1_msSYN001_B1"]
    assert (row.b_number, row.plate, row.well) == ("b0002", "1", "A3")
    assert row.optical_density == pytest.approx(0.6)


def test_read_metabolites_counts_arity_by_id_not_by_name(
    tmp_path: Path, small_counts: None
) -> None:
    """``didp``'s two ``; ``-separated names are ONE metabolite under one BiGG id."""
    _write_table_s9(tmp_path / m.TABLE_S9)
    metabolites = m.read_metabolites(tmp_path / m.TABLE_S9)
    assert metabolites["didp"].names == ("DIDP", "2'-deoxyinosine-5'-diphosphate(3-)")
    assert metabolites["didp"].n_isobaric == 1
    assert metabolites["didp"].bigg_ids == ("didp",)
    assert metabolites["ac-gcald"].n_isobaric == 2
    assert metabolites["ac-gcald"].kegg_ids == ("C00033", "C00266")


def test_read_metabolites_refuses_disagreeing_field_arities(tmp_path: Path) -> None:
    rows = [("a-b", "a-b-c", "A; B", "C1-C2", 10.0, "CH4")]
    path = tmp_path / "bad.xlsx"
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    assert sheet is not None
    sheet.title = m.TABLE_S9_SHEET
    sheet.append(
        [
            "Abbreviation",
            "BIGG",
            "Metabolite",
            "KEGG",
            "Monoisotopic mass",
            "Neutral Formula",
        ]
    )
    for row in rows:
        sheet.append(list(row))
    workbook.save(path)
    with pytest.raises(ValueError, match=r"join \[2, 3\] tokens"):
        m.read_metabolites(path)


def test_read_feature_table_splits_the_adduct_off_every_key(
    tmp_path: Path, small_counts: None
) -> None:
    _write_table_s4(tmp_path / m.TABLE_S4)
    table = m.read_feature_table(tmp_path / m.TABLE_S4)
    assert [feature.key for feature in table.features] == feature_keys()
    assert [feature.abbreviation for feature in table.features][:2] == ["ppal", "ppal"]
    assert [feature.adduct for feature in table.features][:2] == list(ADDUCTS)
    assert [sample.column for sample in table.samples] == sample_columns()
    assert table.matrix.shape == (len(feature_keys()), len(sample_columns()))
    assert table.features[0].row == 1


def test_read_feature_table_refuses_a_keyless_adduct_and_a_repeat(
    tmp_path: Path,
) -> None:
    path = tmp_path / "noadduct.xlsx"
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    assert sheet is not None
    sheet.title = m.TABLE_S4_SHEET
    sheet.append(["Abbr", "Metabolite", "Mass", "Kegg", "thrA_R1_msSYN001_B1"])
    sheet.append(["ppal", "Propanal", 59.0, "C00479", 1.0])
    workbook.save(path)
    with pytest.raises(ValueError, match=r"'ppal' carries no \[M\.\.\] adduct"):
        m.read_feature_table(path)


def test_read_feature_table_refuses_wrong_label_columns(tmp_path: Path) -> None:
    path = tmp_path / "cols.xlsx"
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    assert sheet is not None
    sheet.title = m.TABLE_S4_SHEET
    sheet.append(["Abbr", "Mass", "Kegg", "Metabolite", "thrA_R1_msSYN001_B1"])
    workbook.save(path)
    with pytest.raises(ValueError, match="Table S4 starts with"):
        m.read_feature_table(path)


# --------------------------------------------------------------------------- #
# Identity join and the build-time checks
# --------------------------------------------------------------------------- #
def _identities(tmp_path: Path) -> list[m.FeatureIdentity]:
    """The six synthetic features joined to their Table S9 rows."""
    _write_table_s4(tmp_path / m.TABLE_S4)
    _write_table_s9(tmp_path / m.TABLE_S9)
    table = m.read_feature_table(tmp_path / m.TABLE_S4)
    return m.join_identities(table.features, m.read_metabolites(tmp_path / m.TABLE_S9))


def test_join_identities_carries_one_bigg_id_only_for_a_single_metabolite(
    tmp_path: Path, small_counts: None
) -> None:
    identities = {identity.key: identity for identity in _identities(tmp_path)}
    assert identities["ppal[M+H]+"].target_metabolite_id == "ppal"
    assert identities["didp[M-H]-"].target_metabolite_id == "didp"
    assert identities["ac-gcald[M+H]+"].target_metabolite_id is None
    assert identities["ac-gcald[M+H]+"].bigg_ids == ("ac", "gcald")
    assert identities["ac-gcald[M+H]+"].n_isobaric == 2
    assert identities["ppal[M+H]+"].monoisotopic_mass == 58.0419
    assert identities["ppal[M+H]+"].kegg == "C00479"


def test_identity_ledger_counts_adducts_groups_and_the_merged_candidates(
    tmp_path: Path, small_counts: None
) -> None:
    ledger = m.identity_ledger(_identities(tmp_path))
    assert (ledger.n_features, ledger.n_single_identity, ledger.n_merged_isobaric) == (
        6,
        4,
        2,
    )
    assert (ledger.n_metabolite_groups, ledger.n_single_identity_groups) == (3, 2)
    assert ledger.adduct_histogram == {"[M+H]+": 3, "[M-H]-": 3}
    assert ledger.isobaric_size_histogram == {1: 4, 2: 2}
    assert ledger.merged_candidates == {
        "ac-gcald[M+H]+": ("ac", "gcald"),
        "ac-gcald[M-H]-": ("ac", "gcald"),
    }
    assert ledger.target_metabolite_ids_covered == pytest.approx(4 / 6)


def test_join_identities_refuses_an_unknown_kegg_and_a_wrong_proton_mass(
    tmp_path: Path, small_counts: None
) -> None:
    _write_table_s4(tmp_path / m.TABLE_S4)
    table = m.read_feature_table(tmp_path / m.TABLE_S4)
    good = m.Metabolite(
        abbreviation="ppal",
        bigg="ppal",
        names=("Propanal",),
        kegg="C00479",
        monoisotopic_mass=58.0419,
        neutral_formula="C3H6O",
    )
    with pytest.raises(ValueError, match="which Table S9 does not carry"):
        m.join_identities(table.features, {"ppal": good})
    wrong_kegg = good.model_copy(update={"kegg": "C99999"})
    with pytest.raises(ValueError, match="Table S4 KEGG 'C00479' is not Table S9's"):
        m.join_identities(table.features[:1], {"ppal": wrong_kegg})
    wrong_mass = good.model_copy(update={"monoisotopic_mass": 10.0})
    with pytest.raises(ValueError, match="not the proton mass"):
        m.join_identities(table.features[:1], {"ppal": wrong_mass})
    with pytest.raises(ValueError, match="have no Table S4 feature"):
        m.join_identities(
            table.features[:2],
            {"ppal": good, "other": good.model_copy(update={"abbreviation": "other"})},
        )


def test_populated_features_requires_a_row_to_be_all_or_nothing(
    tmp_path: Path, small_counts: None
) -> None:
    _write_table_s4(tmp_path / m.TABLE_S4)
    table = m.read_feature_table(tmp_path / m.TABLE_S4)
    populated = m.populated_features(table)
    assert populated.n_rows == 6
    assert populated.n_empty_rows == 1
    assert populated.populated_rows == (0, 1, 2, 4, 5)

    partial = feature_values()
    row = partial[0]
    assert row is not None
    partial[0] = [*row[:6], *[None] * 6]  # type: ignore[list-item]  # a deliberately partial row
    _write_table_s4(tmp_path / "partial.xlsx", partial)
    with pytest.raises(ValueError, match="1 partially populated feature rows"):
        m.populated_features(m.read_feature_table(tmp_path / "partial.xlsx"))


def test_batch_normalization_requires_a_unit_median_per_batch(
    tmp_path: Path, small_counts: None
) -> None:
    _write_table_s4(tmp_path / m.TABLE_S4)
    table = m.read_feature_table(tmp_path / m.TABLE_S4)
    rows = m.populated_features(table).populated_rows
    (batch,) = m.batch_normalization(table, rows)
    assert (batch.batch, batch.n_samples, batch.n_features) == (1, 12, 5)
    assert (batch.min_median, batch.max_median) == (1.0, 1.0)

    doubled = [
        None if row is None else [2 * v for v in row] for row in feature_values()
    ]
    _write_table_s4(tmp_path / "doubled.xlsx", doubled)
    shifted = m.read_feature_table(tmp_path / "doubled.xlsx")
    with pytest.raises(ValueError, match="not 1.0, so the released values are not"):
        m.batch_normalization(shifted, m.populated_features(shifted).populated_rows)


def test_check_mean_fold_change_requires_the_paper_s_own_statistic(
    tmp_path: Path, small_counts: None
) -> None:
    _write_table_s4(tmp_path / m.TABLE_S4)
    table = m.read_feature_table(tmp_path / m.TABLE_S4)
    columns = {
        gene: tuple(
            index for index, sample in enumerate(table.samples) if sample.gene == gene
        )
        for gene, _, _, _ in STRAINS
    }
    _write_table_s5(tmp_path / m.TABLE_S5, ACCUMULATIONS)
    result = m.check_mean_fold_change(
        table, m.read_accumulations(tmp_path / m.TABLE_S5), columns
    )
    assert (result.n_pairs, result.n_checked, result.max_abs_difference) == (2, 2, 0.0)

    _write_table_s5(tmp_path / "wrong.xlsx", [("thrA", "ppal[M+H]+", 99.0)])
    with pytest.raises(ValueError, match="differs from Table S5's Mean_FC"):
        m.check_mean_fold_change(
            table, m.read_accumulations(tmp_path / "wrong.xlsx"), columns
        )
    _write_table_s5(tmp_path / "absent.xlsx", [("nope", "ppal[M+H]+", 1.0)])
    with pytest.raises(ValueError, match="is not in Table S4"):
        m.check_mean_fold_change(
            table, m.read_accumulations(tmp_path / "absent.xlsx"), columns
        )


def test_group_samples_requires_two_plates_under_one_b_number(
    tmp_path: Path, small_counts: None
) -> None:
    _write_table_s4(tmp_path / m.TABLE_S4)
    _write_table_s3(tmp_path / m.TABLE_S3)
    table = m.read_feature_table(tmp_path / m.TABLE_S4)
    rows = m.read_sample_rows(tmp_path / m.TABLE_S3)
    strains = m.group_samples(table, rows)
    assert [strain.gene for strain in strains] == sorted(
        gene for gene, _, _, _ in STRAINS
    )
    assert {strain.gene: strain.b_number for strain in strains}["ghostG"] == "b0099"
    assert all(len(strain.columns) == 2 for strain in strains)

    first = next(iter(rows))
    with pytest.raises(ValueError, match="not Table S3's sample ids"):
        m.group_samples(table, {k: v for k, v in rows.items() if k != first})

    crossed = dict(rows)
    crossed[first] = crossed[first].model_copy(update={"b_number": "b9999"})
    with pytest.raises(ValueError, match="carries b-numbers"):
        m.group_samples(table, crossed)


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #
def test_metabolite_phenotype_stores_the_mean_its_se_and_the_known_identities() -> None:
    phenotype = m.metabolite_phenotype(
        ["ppal[M+H]+", "ac-gcald[M-H]-"],
        [[0.5, 1.5], [2.0, 2.0]],
        {"ppal[M+H]+": "ppal"},
    )
    assert phenotype.metabolite_level == {"ppal[M+H]+": 1.0, "ac-gcald[M-H]-": 2.0}
    assert phenotype.metabolite_level_se == {"ppal[M+H]+": 0.5, "ac-gcald[M-H]-": 0.0}
    assert phenotype.n_replicates == {"ppal[M+H]+": 2, "ac-gcald[M-H]-": 2}
    assert phenotype.target_metabolite_ids == {"ppal[M+H]+": "ppal"}
    assert phenotype.measurement_type == (
        "fi_ms_iml1515_feature_fold_change_vs_batch_median_mean_of_2_plates"
    )
    assert phenotype.provenance_gaps == []
    with pytest.raises(ValueError, match="2 keys for 1 replicate lists"):
        m.metabolite_phenotype(["a", "b"], [[1.0, 1.0]], {})


def test_host_background_names_bw25993_as_the_parent_and_gaps_the_construction() -> (
    None
):
    background = m.host_background()
    assert background.name == "YYdCas9"
    assert background.reference_strain == "MG1655"
    assert background.assembly_set == "ecoli_K12_MG1655_ASM584v2"
    assert background.parents == ["BW25993"]
    assert background.genotype_statement == (
        "BW25993 intC:tetR-dcas9-aadA lacY:ypet-cat"
    )
    assert background.alleles == []
    assert background.gapped_fields() == {"construction"}


def test_crispri_genotype_is_one_knockdown_on_the_b_number_namespace() -> None:
    resolved = m.ResolvedStrain(
        strain=m.StrainSamples(
            gene="thrA",
            b_number="b0002",
            columns=(2, 3),
            rows=(
                m.SampleRow(
                    sample=m.parse_sample_id("thrA_R1_msSYN001_B1"),
                    b_number="b0002",
                    optical_density=0.6,
                    plate="1",
                    well="A3",
                ),
            ),
        ),
        locus_tag="b0002",
        symbol="thrA",
        guide=m.Guide(
            gene="thrA",
            sgrna_id="thrA #2",
            b_number="b0002",
            spacer="TTTTCCCCGGGGAAAATTTT",
        ),
    )
    (perturbation,) = m.crispri_genotype(resolved).perturbations
    dumped = perturbation.model_dump()
    assert {
        key: dumped[key]
        for key in (
            "systematic_gene_name",
            "perturbed_gene_name",
            "perturbation_type",
            "gene_namespace",
            "identifier_mapping",
            "state",
            "expression_direction",
            "mechanism_so_id",
        )
    } == {
        "systematic_gene_name": "b0002",
        "perturbed_gene_name": "thrA",
        "perturbation_type": "bacterial_crispr_interference",
        "gene_namespace": "ecoli_k12_mg1655_bnumber",
        "identifier_mapping": None,
        "state": "present",
        "expression_direction": "decreased",
        "mechanism_so_id": "SO:0001998",
    }
    assert dumped["crispr"] == {
        "effector": "dCas9",
        "guide_sequence": "TTTTCCCCGGGGAAAATTTT",
        "n_guides": 1,
        "library_pool": None,
        "effector_plasmid_ref": None,
    }


def test_environment_is_the_induced_m9_glucose_culture() -> None:
    env = m.environment()
    assert env.media.base_medium == "M9"
    assert env.media.state == "liquid"
    assert env.media.is_synthetic
    assert env.media.is_fully_characterized
    assert env.temperature is not None
    assert env.temperature.value == 37.0
    assert env.aerobicity == "aerobic"
    assert env.duration_hours == 6.5
    (inducer,) = env.perturbations
    assert isinstance(inducer, SmallMoleculePerturbation)
    assert inducer.compound.name == "anhydrotetracycline"
    assert (inducer.concentration.value, inducer.concentration.unit) == (200.0, "nM")
    assert inducer.perturbation_type == "small_molecule"
    roles = {
        component.compound.name: component.role for component in env.media.components
    }
    assert roles["D-glucose"] == "carbon_source"
    assert roles["ammonium sulfate"] == "nitrogen_source"
    assert roles["ampicillin"] == "selection_agent"
    glucose = next(c for c in env.media.components if c.compound.name == "D-glucose")
    assert glucose.concentration is not None
    assert (glucose.concentration.value, glucose.concentration.unit) == (5.0, "g/L")


# --------------------------------------------------------------------------- #
# Hermetic build
# --------------------------------------------------------------------------- #
def test_build_two_records_with_the_control_reference_and_ledgers(
    synthetic: Path,
    mg1655: EcoliK12MG1655Genome,
    presence_only_pins: list[Mapping[str, str]],
) -> None:
    """The thrA and thrL strains are kept, in that (gene-sorted) order.

    ghostG's b0099 is a b0005 synonym and oddS has no assigned target, both ledgered;
    the two ctrl strains are the reference.
    """
    dataset = m.MetabolomeRapp2026Dataset(root=str(synthetic), ecoli_genome=mg1655)
    assert presence_only_pins == [m.DATA_SHA256]
    assert len(dataset) == 2
    keys = ["ppal[M+H]+", "ppal[M-H]-", "ac-gcald[M+H]+", "didp[M+H]+", "didp[M-H]-"]

    first = dataset[0]["experiment"]
    (perturbation,) = first["genotype"]["perturbations"]
    assert perturbation["systematic_gene_name"] == "b0002"
    assert perturbation["perturbed_gene_name"] == "thrA"
    assert perturbation["crispr"]["guide_sequence"] == "TTTTCCCCGGGGAAAATTTT"
    assert first["phenotype"]["metabolite_level"] == {
        key: expected_level("thrA", key) for key in keys
    }
    assert first["phenotype"]["n_replicates"] == dict.fromkeys(keys, 2)
    assert first["phenotype"]["target_metabolite_ids"] == {
        "ppal[M+H]+": "ppal",
        "ppal[M-H]-": "ppal",
        "didp[M+H]+": "didp",
        "didp[M-H]-": "didp",
    }
    assert first["environment"] == m.environment().model_dump()

    second = dataset[1]["experiment"]
    assert second["genotype"]["perturbations"][0]["systematic_gene_name"] == "b0001"
    assert second["phenotype"]["metabolite_level"] == {
        key: expected_level("thrL", key) for key in keys
    }

    reference = dataset[0]["reference"]
    assert reference["genome_reference"] == REFERENCE.model_dump()
    assert reference["phenotype_reference"]["n_replicates"] == dict.fromkeys(keys, 4)
    control = reference["phenotype_reference"]["metabolite_level"]
    assert control == {
        key: pytest.approx(
            float(np.mean([expected_level("ctrl1", key), expected_level("ctrl2", key)]))
        )
        for key in keys
    }
    assert dataset[0]["publication"] == m.PUBLICATION.model_dump()

    preprocess = synthetic / "preprocess"
    drops = json.loads((preprocess / "dropped_records.json").read_text())
    assert (
        drops["strain_tokens"],
        drops["source_records"],
        drops["kept_records"],
        drops["dropped_records"],
    ) == (6, 4, 2, 2)
    assert drops["reference_tokens"] == ["ctrl1", "ctrl2"]
    assert [(rule["rule"], rule["items"]) for rule in drops["rules"]] == [
        (
            "no_target_gene_assigned",
            ["oddS (b0000, plate 2 well C5): no Table S1 sgRNA"],
        ),
        (
            "b_number_remapped_by_the_annotation",
            [
                "ghostG (b0099): the annotation carries it as a gene synonym of "
                "current gene b0005, so the record would store b0005"
            ],
        ),
    ]
    identity = json.loads((preprocess / "metabolite_identity.json").read_text())
    assert (identity["n_features"], identity["n_single_identity"]) == (5, 4)
    assert identity["merged_candidates"] == {"ac-gcald[M+H]+": ["ac", "gcald"]}
    normalization = json.loads((preprocess / "batch_normalization.json").read_text())
    assert normalization["n_populated_rows"] == 5
    assert normalization["features"]["n_empty_rows"] == 1
    assert [batch["batch"] for batch in normalization["batches"]] == [1]
    crosscheck = json.loads((preprocess / "mean_fc_crosscheck.json").read_text())
    assert crosscheck["n_checked"] == len(ACCUMULATIONS)
    assert crosscheck["max_abs_difference"] == 0.0
    strains = (preprocess / "strains.csv").read_text().splitlines()
    assert strains[0] == (
        "record,gene,b_number,locus_tag,symbol,sgrna_id,spacer,plate,well,batch,"
        "optical_density"
    )
    assert strains[1] == (
        "0,thrA,b0002,b0002,thrA,thrA #2,TTTTCCCCGGGGAAAATTTT,1,A3,1,0.600000; 0.700000"
    )
    assert strains[2] == (
        "1,thrL,b0001,b0001,thrL,thrL #1,ACGTACGTACGTACGTACGT,1,A2,1,0.600000; 0.700000"
    )
    metabolites = (preprocess / "metabolites.csv").read_text().splitlines()
    assert metabolites[1].startswith("ppal[M+H]+,ppal,[M+H]+,")
    assert metabolites[1].endswith(",C00479,ppal,1,ppal,Propanal")
    ledger = json.loads((preprocess / "identifier_reconciliation.json").read_text())
    assert ledger["reconciliation"]["unique_names"] == 3
    assert ledger["min_resolved_fraction"] == m.MIN_RESOLVED_FRACTION


def test_build_stops_below_the_resolution_threshold(
    synthetic: Path, mg1655: EcoliK12MG1655Genome, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A threshold above what the synthetic screen reaches stops the build."""
    monkeypatch.setattr(m, "MIN_RESOLVED_FRACTION", 1.01)
    with pytest.raises(LocusTagResolutionError, match=r"3 of 3 names \(1\.000\)"):
        m.MetabolomeRapp2026Dataset(root=str(synthetic), ecoli_genome=mg1655)
    assert not (synthetic / "processed" / "lmdb").exists()


def test_build_refuses_a_genome_of_another_strain(
    synthetic: Path, monkeypatch: pytest.MonkeyPatch, mg1655: EcoliK12MG1655Genome
) -> None:
    monkeypatch.setattr(
        m, "BACTERIAL_ASSEMBLY_SETS", {"MG1655": "ecoli_K12_BW25113_ASM75055v1"}
    )
    with pytest.raises(ValueError, match="needs the ecoli_K12_BW25113_ASM75055v1"):
        m.MetabolomeRapp2026Dataset(root=str(synthetic), ecoli_genome=mg1655)


def test_build_refuses_a_control_count_other_than_the_paper_s(
    synthetic: Path, monkeypatch: pytest.MonkeyPatch, mg1655: EcoliK12MG1655Genome
) -> None:
    monkeypatch.setattr(m, "CONTROL_REPLICATES", 15)
    with pytest.raises(ValueError, match="carries 2 control strains, the paper states"):
        m.MetabolomeRapp2026Dataset(root=str(synthetic), ecoli_genome=mg1655)


def test_process_verifies_the_real_pins(
    synthetic: Path, monkeypatch: pytest.MonkeyPatch, mg1655: EcoliK12MG1655Genome
) -> None:
    """With the real check back, the synthetic workbooks are refused by their pins."""
    from torchcell.data import verify_raw_files

    monkeypatch.setattr(m, "verify_raw_files", verify_raw_files)
    with pytest.raises(
        RawSha256MismatchError, match=f"expected {m.DATA_SHA256[m.TABLE_S1]}"
    ):
        m.MetabolomeRapp2026Dataset(root=str(synthetic), ecoli_genome=mg1655)


# --------------------------------------------------------------------------- #
# Raw mirror
# --------------------------------------------------------------------------- #
def _synthetic_raw_files(
    tmp_path: Path,
) -> tuple[tuple[m.RawFile, ...], dict[str, Path]]:
    import hashlib

    sources: dict[str, Path] = {}
    files: list[m.RawFile] = []
    for name, member, payload in (
        ("a.xlsx", "mmc2.xlsx", b"first"),
        ("b.xlsx", "mmc4.xlsx", b"second"),
    ):
        path = tmp_path / "download" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        sha = hashlib.sha256(payload).hexdigest()
        files.append(
            m.RawFile(
                name=name,
                member=member,
                sha256=sha,
                bytes=len(payload),
                description=name,
                retrieval=RetrievalRecord(
                    method=RetrievalMethod.direct_url,
                    source_url=f"https://example.org/{member}",
                    retriever="torchcell.literature.retrieve.elsevier_mmc",
                    params={"pii": m.ELSEVIER_PII, "filename": member},
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
        ("data/a.xlsx", files[0].sha256, RetrievalMethod.direct_url),
        ("data/b.xlsx", files[1].sha256, RetrievalMethod.direct_url),
    ]
    assert manifest.si_expected == list(m.NOT_MIRRORED)
    assert m.manifest_sha256(manifest, "data/b.xlsx") == files[1].sha256
    with pytest.raises(KeyError, match="data/c.xlsx is not in the raw-mirror manifest"):
        m.manifest_sha256(manifest, "data/c.xlsx")

    (root / "data" / "a.xlsx").write_bytes(b"edited")
    with pytest.raises(RuntimeError, match="exists with a different sha256; refusing"):
        m.deposit_raw_mirror(sources=sources, data_root=data_root)
    (root / "data" / "a.xlsx").write_bytes(b"first")
    sources["b.xlsx"].write_bytes(b"other")
    with pytest.raises(RuntimeError, match="b.xlsx sha256 mismatch"):
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
    dataset = object.__new__(m.MetabolomeRapp2026Dataset)
    monkeypatch.setattr(
        m.MetabolomeRapp2026Dataset, "raw_dir", str(tmp_path / "raw"), raising=False
    )
    dataset.download()
    linked = tmp_path / "raw" / "a.xlsx"
    assert linked.is_symlink()
    assert os.readlink(linked) == str(data_root / m.RAW_DIR_REL / "data" / "a.xlsx")

    manifest_path = data_root / m.RAW_DIR_REL / "manifest.json"
    manifest_path.write_text(
        manifest_path.read_text().replace(files[0].sha256, "0" * 64)
    )
    with pytest.raises(ManifestPinMismatchError, match="data/a.xlsx"):
        dataset.download()


def test_retrieve_runs_the_recorded_retriever_and_verifies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    files, _ = _synthetic_raw_files(tmp_path)
    monkeypatch.setattr(m, "RAW_FILES", files)
    calls: list[RetrievalRecord] = []

    def fake(record: RetrievalRecord) -> bytes:
        calls.append(record)
        return b"first" if record.params["filename"] == "mmc2.xlsx" else b"wrong"

    monkeypatch.setattr(m, "run_retriever", fake)
    out = m.retrieve_raw_files(tmp_path / "fetched", ["a.xlsx"])
    assert out == {"a.xlsx": tmp_path / "fetched" / "a.xlsx"}
    assert [call.params["filename"] for call in calls] == ["mmc2.xlsx"]
    with pytest.raises(RawSha256MismatchError):
        m.retrieve_raw_files(tmp_path / "fetched", ["b.xlsx"])
    assert not (tmp_path / "fetched" / "b.xlsx").exists()


def test_every_consumed_file_is_pinned_once_from_the_elsevier_cdn() -> None:
    assert [(f.name, f.member) for f in m.RAW_FILES] == [
        ("si2.xlsx", "mmc2.xlsx"),
        ("si4.xlsx", "mmc4.xlsx"),
        ("si5.xlsx", "mmc5.xlsx"),
        ("si6.xlsx", "mmc6.xlsx"),
        ("si10.xlsx", "mmc10.xlsx"),
    ]
    assert all(
        len(f.sha256) == 64
        and f.sha256 == f.retrieval.sha256
        and f.retrieval.retriever == "torchcell.literature.retrieve.elsevier_mmc"
        and f.retrieval.params == {"pii": m.ELSEVIER_PII, "filename": f.member}
        for f in m.RAW_FILES
    )
    assert m.PUBLICATION.pubmed_id is None
    assert m.PUBLICATION.doi == "10.1016/j.cels.2025.101518"


def test_the_measured_counts_the_module_pins_are_the_paper_s() -> None:
    assert (m.LIBRARY_GENES, m.N_EXTRACTS, m.ISOBARIC_METABOLITES) == (1515, 3026, 802)
    assert (m.N_BIOLOGICAL_REPLICATES, m.CONTROL_REPLICATES) == (2, 15)
    assert (m.TEMPERATURE_C, m.DURATION_HOURS) == (37.0, 6.5)
    assert m.REFERENCE_STRAIN_NAME == m.MetabolomeRapp2026Dataset.REFERENCE_STRAIN
    assert m.MG1655_ASSEMBLY_SET == "ecoli_K12_MG1655_ASM584v2"


# --------------------------------------------------------------------------- #
# Real data (``--data``): the mirror and the built dev-tree LMDB, never rebuilt
# --------------------------------------------------------------------------- #
def _real_data_root() -> str:
    return os.environ["DATA_ROOT"]


def _built_root() -> str:
    return osp.join(_real_data_root(), m.DATASET_ROOT_REL)


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
    result = audit_sourced_value(value, Path(_real_data_root()) / "torchcell-library")
    assert result.passed, result.message


@pytest.mark.data
def test_real_build_counts_and_two_hand_checked_records() -> None:
    """1,496 records (1,513 strain tokens - 15 controls - argR - phnE); aaeA and the
    paper's IspB frdp accumulation are read off ``si5.xlsx`` independently.
    """
    from torchcell.verification.runners import load_records

    drops = json.loads(
        Path(_built_root(), "preprocess", "dropped_records.json").read_text()
    )
    assert (drops["strain_tokens"], drops["source_records"], drops["kept_records"]) == (
        1513,
        1498,
        1496,
    )
    assert [(rule["rule"], rule["n_records"]) for rule in drops["rules"]] == [
        ("no_target_gene_assigned", 1),
        ("b_number_remapped_by_the_annotation", 1),
    ]
    identity = json.loads(
        Path(_built_root(), "preprocess", "metabolite_identity.json").read_text()
    )
    assert (
        identity["n_features"],
        identity["n_single_identity"],
        identity["n_merged_isobaric"],
        identity["n_metabolite_groups"],
    ) == (1321, 1077, 244, 723)
    crosscheck = json.loads(
        Path(_built_root(), "preprocess", "mean_fc_crosscheck.json").read_text()
    )
    assert (crosscheck["n_pairs"], crosscheck["max_abs_difference"]) == (1385, 0.0)

    records = load_records(_built_root())
    assert len(records) == 1496
    by_gene = {
        record["experiment"]["genotype"]["perturbations"][0]["perturbed_gene_name"]: (
            record
        )
        for record in records
    }
    for gene, b_number, expected in (
        ("aaeA", "b3241", HAND_CHECKED_AAEA),
        ("ispB", "b3187", HAND_CHECKED_ISPB),
    ):
        record = by_gene[gene]
        perturbation = record["experiment"]["genotype"]["perturbations"][0]
        assert perturbation["systematic_gene_name"] == b_number
        assert perturbation["gene_namespace"] == "ecoli_k12_mg1655_bnumber"
        assert len(perturbation["crispr"]["guide_sequence"]) == 20
        phenotype = record["experiment"]["phenotype"]
        assert len(phenotype["metabolite_level"]) == 1321
        assert set(phenotype["n_replicates"].values()) == {2}
        assert len(phenotype["target_metabolite_ids"]) == 1077
        for key, (level, se) in expected.items():
            assert phenotype["metabolite_level"][key] == pytest.approx(level)
            assert phenotype["metabolite_level_se"][key] == pytest.approx(se)

    reference = by_gene["aaeA"]["reference"]
    assert reference["genome_reference"]["assembly_set"] == m.MG1655_ASSEMBLY_SET
    assert reference["genome_reference"]["background"]["name"] == "YYdCas9"
    assert set(reference["phenotype_reference"]["n_replicates"].values()) == {30}
    for key, level in HAND_CHECKED_CONTROL.items():
        assert reference["phenotype_reference"]["metabolite_level"][
            key
        ] == pytest.approx(level)


@pytest.mark.data
def test_real_build_passes_l0_to_l4() -> None:
    report = m.run_verification(_real_data_root())
    assert report.passed, report.summary()


@pytest.mark.data
def test_real_table_s4_cells_back_the_hand_checked_values() -> None:
    """Re-read the three oracle rows straight from the workbook with ``openpyxl``."""
    path = m.raw_mirror_dir(_real_data_root()) / f"data/{m.TABLE_S4}"
    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    sheet = workbook[m.TABLE_S4_SHEET]
    rows = sheet.iter_rows(values_only=True)
    header = [str(cell) for cell in next(rows)]
    wanted = set(HAND_CHECKED_AAEA) | set(HAND_CHECKED_ISPB) | set(HAND_CHECKED_CONTROL)
    found: dict[str, Any] = {}
    for row in rows:
        if str(row[0]) in wanted:
            found[str(row[0])] = row
        if len(found) == len(wanted):
            break
    workbook.close()

    def columns(prefix: str) -> list[int]:
        return [i for i, name in enumerate(header) if name.startswith(prefix)]

    for prefix, expected in (
        ("aaeA_", HAND_CHECKED_AAEA),
        ("ispB_", HAND_CHECKED_ISPB),
        ("ctrl", HAND_CHECKED_CONTROL),
    ):
        indices = columns(prefix)
        for key, oracle in expected.items():
            values = np.asarray([found[key][i] for i in indices], dtype=np.float64)
            level = oracle[0] if isinstance(oracle, tuple) else oracle
            assert float(values.mean()) == pytest.approx(level)


#: ``{feature key: (level, se)}`` for ``aaeA``, read off ``si5.xlsx`` columns
#: ``aaeA_R1_msAV932_B1`` / ``aaeA_R2_msAV933_B1`` with ``openpyxl``.
HAND_CHECKED_AAEA: dict[str, tuple[float, float]] = {
    "ppal[M+H]+": (1.0119883541702346, 0.044927784437974516),
    "ppal[M-H]-": (1.0098443072596073, 0.0055788140950918655),
    "frdp[M-H]-": (1.2199111470113086, 0.0777463651050081),
}
#: The same for ``ispB``, whose frdp accumulation is the paper's engineering result.
HAND_CHECKED_ISPB: dict[str, tuple[float, float]] = {
    "ppal[M+H]+": (1.2365701889592966, 0.02095107609750524),
    "frdp[M-H]-": (6.717757135164243, 0.2713381798599892),
}
#: The reference: the mean over the 30 ``ctrlN`` columns.
HAND_CHECKED_CONTROL: dict[str, float] = {
    "ppal[M+H]+": 1.0591982502706005,
    "frdp[M-H]-": 0.9356164908333794,
}


# --------------------------------------------------------------------------- #
# Remaining refusals, the class contract and the CLI surface
# --------------------------------------------------------------------------- #
def test_read_sample_rows_refuses_a_mislabeled_or_repeated_row(tmp_path: Path) -> None:
    def write(rows: Sequence[tuple[str, str, str]]) -> Path:
        path = tmp_path / f"s3_{len(rows)}_{rows[0][0]}.xlsx"
        workbook = openpyxl.Workbook()
        sheet = workbook.active
        assert sheet is not None
        sheet.title = m.TABLE_S3_SHEET
        sheet.append(
            [
                "Target gene",
                "b number",
                "OD",
                "Plate ID",
                "Well",
                "Replicate",
                "Sample ID",
            ]
        )
        for gene, b_number, column in rows:
            sheet.append([gene, b_number, 0.5, "1", "A1", "R1", column])
        workbook.save(path)
        return path

    crossed = write([("other", "b0002", "thrA_R1_msSYN001_B1")])
    with pytest.raises(ValueError, match="target gene 'other' is not the id's gene"):
        m.read_sample_rows(crossed)

    repeated = write(
        [("thrA", "b0002", "thrA_R1_msSYN001_B1")] * 2  # the same sample id twice
    )
    with pytest.raises(ValueError, match="lists 'thrA_R1_msSYN001_B1' more than once"):
        m.read_sample_rows(repeated)

    with pytest.raises(ValueError, match="Table S3 holds 1 samples, the paper states"):
        m.read_sample_rows(write([("thrA", "b0002", "thrA_R2_msSYN001_B1")]))


def test_read_metabolites_refuses_a_repeat_and_a_wrong_count(tmp_path: Path) -> None:
    path = tmp_path / "dup.xlsx"
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    assert sheet is not None
    sheet.title = m.TABLE_S9_SHEET
    sheet.append(
        [
            "Abbreviation",
            "BIGG",
            "Metabolite",
            "KEGG",
            "Monoisotopic mass",
            "Neutral Formula",
        ]
    )
    for _ in range(2):
        sheet.append(["ppal", "ppal", "Propanal", "C00479", 58.0419, "C3H6O"])
    workbook.save(path)
    with pytest.raises(ValueError, match="Table S9 lists 'ppal' more than once"):
        m.read_metabolites(path)

    _write_table_s9(tmp_path / m.TABLE_S9)
    with pytest.raises(ValueError, match="Table S9 holds 3 metabolites, the paper"):
        m.read_metabolites(tmp_path / m.TABLE_S9)


def test_read_feature_table_refuses_a_repeated_key_and_a_wrong_sample_count(
    tmp_path: Path, small_counts: None
) -> None:
    first = feature_values()[0]
    assert first is not None
    path = tmp_path / "repeat.xlsx"
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    assert sheet is not None
    sheet.title = m.TABLE_S4_SHEET
    sheet.append(["Abbr", "Metabolite", "Mass", "Kegg", *sample_columns()])
    for _ in range(2):
        sheet.append(["ppal[M+H]+", "Propanal[M+H]+", 59.04918, "C00479", *first])
    workbook.save(path)
    with pytest.raises(ValueError, match="lists feature 'ppal\\[M\\+H\\]\\+' more"):
        m.read_feature_table(path)

    _write_table_s4(tmp_path / m.TABLE_S4)
    with pytest.raises(ValueError, match="Table S4 holds 12 sample columns, the paper"):
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(m, "N_EXTRACTS", 3026)
            m.read_feature_table(tmp_path / m.TABLE_S4)


def test_resolve_strains_refuses_a_guide_whose_b_number_disagrees(
    synthetic: Path, mg1655: EcoliK12MG1655Genome
) -> None:
    """A Table S1 sgRNA filed under another b-number than Table S3's stops the build."""
    guides = [("thrL", "thrL #1", "b0002", "ACGTACGTACGTACGTACGT"), *GUIDES[1:]]
    _write_table_s1(synthetic / "raw" / m.TABLE_S1, guides)
    with pytest.raises(ValueError, match="Table S1 b-number b0002 is not Table S3's"):
        m.MetabolomeRapp2026Dataset(root=str(synthetic), ecoli_genome=mg1655)


def test_resolve_strains_refuses_a_target_with_no_sgrna(
    synthetic: Path, mg1655: EcoliK12MG1655Genome
) -> None:
    """A strain with a real b-number but no Table S1 row is a table disagreement."""
    _write_table_s1(synthetic / "raw" / m.TABLE_S1, [GUIDES[0], *GUIDES[2:], GUIDES[1]])
    guides = [g for g in GUIDES if g[0] != "thrA"]
    _write_table_s1(
        synthetic / "raw" / m.TABLE_S1,
        [*guides, ("spare", "spare #1", "b0006", "AAAACCCCGGGGTTTTAAAA")],
    )
    with pytest.raises(ValueError, match="'thrA' has b-number b0002 but no Table S1"):
        m.MetabolomeRapp2026Dataset(root=str(synthetic), ecoli_genome=mg1655)


def test_canonical_symbol_keeps_the_tag_when_the_symbol_does_not_round_trip(
    mg1655: EcoliK12MG1655Genome,
) -> None:
    assert m.canonical_symbol(mg1655, "b0002") == "thrA"
    # b0005's symbol is proB, and the synthetic fixture also files b0099 on it, which
    # does not change what proB resolves to.
    assert m.canonical_symbol(mg1655, "b0005") == "proB"
    # A pseudogene row with no symbol keeps its tag.
    assert m.canonical_symbol(mg1655, "b0007") == "insZ"


def test_the_dataset_declares_its_schema_classes_and_raw_files() -> None:
    shell = object.__new__(m.MetabolomeRapp2026Dataset)
    assert shell.experiment_class.__name__ == "BacterialMetaboliteExperiment"
    assert shell.reference_class.__name__ == "BacterialMetaboliteExperimentReference"
    assert shell.raw_file_names == [raw.name for raw in m.RAW_FILES]
    frame = __import__("pandas").DataFrame({"a": [1]})
    assert shell.preprocess_raw(frame) is frame
    with pytest.raises(NotImplementedError):
        shell.create_experiment()


def test_download_refuses_a_mirror_file_the_manifest_promises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    files, sources = _synthetic_raw_files(tmp_path)
    monkeypatch.setattr(m, "RAW_FILES", files)
    monkeypatch.setattr(m, "DATA_SHA256", {f.name: f.sha256 for f in files})
    data_root = tmp_path / "root"
    m.deposit_raw_mirror(sources=sources, data_root=str(data_root))
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    (data_root / m.RAW_DIR_REL / "data" / "a.xlsx").unlink()
    dataset = object.__new__(m.MetabolomeRapp2026Dataset)
    monkeypatch.setattr(
        m.MetabolomeRapp2026Dataset, "raw_dir", str(tmp_path / "raw"), raising=False
    )
    with pytest.raises(RuntimeError) as error:
        dataset.download()
    assert str(error.value) == (
        "required raw artifact missing from mirror: "
        f"{data_root / m.RAW_DIR_REL / 'data' / 'a.xlsx'}"
    )
    assert not (tmp_path / "raw").exists() or not (tmp_path / "raw" / "a.xlsx").exists()


def test_mirror_paths_hang_off_data_root(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATA_ROOT", "/nowhere")
    assert m.raw_mirror_dir() == Path("/nowhere") / m.RAW_DIR_REL
    assert m.library_dir() == Path("/nowhere") / m.LIBRARY_DIR_REL


def test_main_requires_a_download_dir_to_retrieve(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: None)
    with pytest.raises(SystemExit, match="--retrieve needs --download-dir"):
        m.main(["deposit", "--retrieve"])
    with pytest.raises(SystemExit):
        m.main(["nonsense"])
