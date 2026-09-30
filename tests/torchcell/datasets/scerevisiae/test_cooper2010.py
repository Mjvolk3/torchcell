# tests/torchcell/datasets/scerevisiae/test_cooper2010.py
# [[tests.torchcell.datasets.scerevisiae.test_cooper2010]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/scerevisiae/test_cooper2010.py
"""Unit tests for the Cooper 2010 CE-LIF amino-acid loader's record construction.

The loader's surface is how it turns one deposited Table 4 row into a typed metabolite
record, so these tests drive ``read_table4`` / ``retain_rows`` / ``create_experiment`` on
a synthetic Table 4 plus a fake resolver. What is pinned: the header is checked exactly,
missing cells drop the key rather than becoming NaN, the medium is a typed dropout of the
library SC (joinable at ``base_medium``), temperature is a typed gap, the reference is 1.0
per present key, replicate counts are the conservative 1, identifiers are normalized and
recorded, a duplicate identifier keeps its first row and ledgers the second, a RENAMED row
is a distinct strain of the current gene, non-gene and retired rows are dropped with their
status, and every sourced quote is still verbatim in the sha256-pinned mirror.

The second half ("End-to-end build") runs ``process()`` for real under ``tmp_path``:
``DATA_ROOT`` points at ``tmp_path/data_root``, which holds a two-record fake
``gene_essentiality_sgd`` LMDB (YGL180W essential, YBR208C not), and the dataset root's
``raw/`` holds a synthetic eight-row Table 4 (``_E2E_ROWS``). Row ``r`` carries
``base_r + j + 0.5`` in peak column ``j`` (0 to 16, file order), so every value is exact in
binary and a swapped header -> key mapping changes a value:

    row  Sysname     NAME        base  rule
    0    YBR208C     DUR1,2      0     kept (idx 0)
    1    YBR020W     GAL1        100   kept (idx 1); RPS19 "-", gshbiotin "0", Asp "-"
    2    ybr020w     GAL1        200   normalized to YBR020W -> duplicate of row 1
    3    YML048WA-   GSF2        300   normalized to YML048W-A, kept (idx 2)
    4    YAL035C-A   YAL035C-A   400   RENAMED -> YAL034C-B, kept (idx 3)
    5    YCL074W     YCL074W     500   NON_GENE_FEATURE (pseudogene), dropped
    6    BY4742      BY4742      600   parent strain, excluded before the resolver
    7    YGL180W     ATG1        700   kept (idx 4), essentiality-flagged

Kept values: 17 + 15 + 17 + 17 + 17 = 83. The four full rows share one reference (17 keys
at 1.0) and the ragged row has its own (15 keys), so the reference index is
[[0, 2, 3, 4], [1]].
"""

from __future__ import annotations

import hashlib
import json
import os
import os.path as osp
import pickle
from pathlib import Path
from typing import Any

import lmdb
import pytest

from torchcell.datamodels.media import SC
from torchcell.datamodels.schema import (
    Environment,
    Genotype,
    KanMxDeletionPerturbation,
    MetaboliteExperiment,
    MetaboliteExperimentReference,
    MetabolitePhenotype,
    Publication,
    ReferenceGenome,
)
from torchcell.datasets.scerevisiae import cooper2010 as module
from torchcell.datasets.scerevisiae.cooper2010 import (
    COOPER_SC,
    EXPECTED_HEADER,
    LEGEND_SOURCED_VALUES,
    MEASUREMENT_TYPE,
    PEAK_KEYS,
    SOURCED_VALUES,
    AminoAcidCooper2010Dataset,
    normalize_identifier,
    read_table4,
    retain_rows,
)
from torchcell.sequence.genome.scerevisiae.s288c import GeneNameStatus
from torchcell.verification.sourced import ProvenanceGap, ProvenanceGapReason


class _Resolution:
    """The three fields of ``GeneNameResolution`` the loader reads."""

    def __init__(self, status: GeneNameStatus, name: str | None, feature: str | None):
        self.status = status
        self.systematic_name = name
        self.feature_type = feature


_RESOLVER: dict[str, _Resolution] = {
    "YBR208C": _Resolution(GeneNameStatus.CURRENT, "YBR208C", None),
    "YBR020W": _Resolution(GeneNameStatus.CURRENT, "YBR020W", None),
    "YML048W-A": _Resolution(GeneNameStatus.CURRENT, "YML048W-A", None),
    "YLR228C": _Resolution(GeneNameStatus.CURRENT, "YLR228C", None),
    "YAL034C-B": _Resolution(GeneNameStatus.CURRENT, "YAL034C-B", None),
    "YAL035C-A": _Resolution(GeneNameStatus.RENAMED, "YAL034C-B", None),
    "YCL074W": _Resolution(GeneNameStatus.NON_GENE_FEATURE, "YCL074W", "pseudogene"),
    "R0010W": _Resolution(GeneNameStatus.RETIRED, "R0010W", None),
    "YGL180W": _Resolution(GeneNameStatus.CURRENT, "YGL180W", None),
}

_HEADER = "\t".join(EXPECTED_HEADER)


def _row(name: str, gene: str, values: list[str]) -> str:
    assert len(values) == len(PEAK_KEYS)
    return "\t".join([name, gene, *values])


def _full(seed: float) -> list[str]:
    return [f"{seed + i * 0.1:.3f}" for i in range(len(PEAK_KEYS))]


def _table(tmp_path: Path) -> str:
    """A Table 4 with one row per rule: full, ragged, duplicate, malformed, renamed,
    non-gene, retired, and an essentiality-store hit.
    """
    ragged = _full(0.5)
    ragged[0] = "-"  # RPS19 missing
    ragged[2] = "0"  # gshbiotin released zero
    ragged[-1] = "-"  # Asp missing
    rows = [
        _HEADER,
        _row("YBR208C", "DUR1,2", _full(1.0)),
        _row("YBR020W", "GAL1", ragged),
        _row("YBR020W", "GAL1", _full(2.0)),  # duplicate identifier, second row
        _row("YML048WA-", "GSF2", _full(3.0)),  # misplaced suffix
        _row("YLR228 C", "", _full(4.0)),  # internal whitespace, empty NAME
        _row("YAL035C-A", "YAL035C-A", _full(5.0)),  # RENAMED onto YAL034C-B
        _row("YAL034C-B", "YAL034C-B", _full(6.0)),  # the current gene's own row
        _row("YCL074W", "YCL074W", _full(7.0)),  # pseudogene
        _row("R0010W", "R0010W", _full(8.0)),  # retired
        _row("YGL180W", "ATG1", _full(9.0)),  # in the essentiality store, kept
    ]
    path = tmp_path / "SupplementalTable4.txt"
    path.write_text("\n".join(rows) + "\n")
    return str(path)


def _dataset() -> AminoAcidCooper2010Dataset:
    """The loader without its PyG init: these tests never touch disk or an LMDB."""
    dataset = AminoAcidCooper2010Dataset.__new__(AminoAcidCooper2010Dataset)
    dataset.name = "AminoAcidCooper2010Dataset"
    return dataset


def _retained(tmp_path: Path) -> tuple[list[Any], Any]:
    rows = read_table4(_table(tmp_path))
    return retain_rows(
        rows,
        lambda name: _RESOLVER[name],
        frozenset({"YGL180W"}),
        dataset_name="AminoAcidCooper2010Dataset",
    )


# ---- parsing --------------------------------------------------------------------- #
def test_header_is_checked_exactly(tmp_path: Path) -> None:
    path = tmp_path / "bad.txt"
    path.write_text("Sysname\tNAME\tRPS19\targ1\tunknown\n")
    with pytest.raises(RuntimeError, match="unmapped columns \\['unknown'\\]"):
        read_table4(path)


def test_missing_cells_drop_the_key_and_zero_is_a_value(tmp_path: Path) -> None:
    rows = read_table4(_table(tmp_path))
    ragged = rows[1].levels
    assert "lysine_related_peak1" not in ragged and "aspartate" not in ragged
    assert ragged["gshbiotin"] == 0.0
    assert len(rows[0].levels) == len(PEAK_KEYS)


def test_identifier_normalization_is_recorded_verbatim(tmp_path: Path) -> None:
    assert normalize_identifier("YML048WA-") == "YML048W-A"
    assert normalize_identifier("YLR228 C") == "YLR228C"
    assert normalize_identifier("YML009c") == "YML009C"
    assert normalize_identifier("YJL038C ") == "YJL038C"
    _, ledger = _retained(tmp_path)
    assert {(n.source_name, n.normalized_name) for n in ledger.normalizations} == {
        ("YML048WA-", "YML048W-A"),
        ("YLR228 C", "YLR228C"),
    }


# ---- retention ------------------------------------------------------------------- #
def test_non_gene_and_retired_rows_are_dropped_with_their_status(
    tmp_path: Path,
) -> None:
    kept, ledger = _retained(tmp_path)
    assert {row.systematic_gene_name for row in kept} == {
        "YBR208C",
        "YBR020W",
        "YML048W-A",
        "YLR228C",
        "YAL034C-B",
        "YGL180W",
    }
    dropped = {d.source_name: d for d in ledger.dropped_orfs}
    assert set(dropped) == {"YCL074W", "R0010W"}
    assert (dropped["YCL074W"].status, dropped["YCL074W"].feature_type) == (
        "non_gene_feature",
        "pseudogene",
    )
    assert dropped["R0010W"].status == "retired"
    assert dropped["R0010W"].n_values == len(PEAK_KEYS)


def test_renamed_row_is_a_distinct_strain_of_the_current_gene(tmp_path: Path) -> None:
    kept, ledger = _retained(tmp_path)
    assert ledger.renamed_orfs == {"YAL035C-A": "YAL034C-B"}
    by_source = {row.perturbed_gene_name: row.systematic_gene_name for row in kept}
    assert by_source["YAL035C-A"] == "YAL034C-B"
    assert by_source["YAL034C-B"] == "YAL034C-B"


def test_duplicate_identifier_keeps_the_first_row_and_ledgers_the_second(
    tmp_path: Path,
) -> None:
    kept, ledger = _retained(tmp_path)
    gal1 = [row for row in kept if row.systematic_gene_name == "YBR020W"]
    assert len(gal1) == 1 and gal1[0].row_index == 1
    assert len(ledger.duplicate_strain_rows) == 1
    duplicate = ledger.duplicate_strain_rows[0]
    assert (duplicate.row_index, duplicate.kept_row_index) == (2, 1)
    assert duplicate.levels["arginine"] == 2.1  # never averaged into the kept row
    assert ledger.n_duplicate_rows == 1


def test_essentiality_hit_is_flagged_and_kept(tmp_path: Path) -> None:
    kept, ledger = _retained(tmp_path)
    assert "YGL180W" in {row.systematic_gene_name for row in kept}
    assert [f.systematic_gene_name for f in ledger.essentiality_flagged] == ["YGL180W"]
    assert ledger.n_excluded_non_deletion == 0


def test_parent_strain_row_is_excluded(tmp_path: Path) -> None:
    path = tmp_path / "t.txt"
    path.write_text(
        "\n".join([_HEADER, _row("BY4742", "wild type", _full(1.0))]) + "\n"
    )
    kept, ledger = retain_rows(
        read_table4(path), lambda name: _RESOLVER[name], frozenset(), dataset_name="x"
    )
    assert kept == []
    assert ledger.excluded_non_deletion[0].reason.startswith("parent-strain")


def test_ledger_counts_add_up(tmp_path: Path) -> None:
    _, ledger = _retained(tmp_path)
    assert ledger.n_source_rows == 10
    assert (
        ledger.n_kept
        + ledger.n_dropped_orfs
        + ledger.n_duplicate_rows
        + ledger.n_excluded_non_deletion
        == ledger.n_source_rows
    )


# ---- record ---------------------------------------------------------------------- #
def test_record_keys_follow_the_row_and_the_reference_is_one(tmp_path: Path) -> None:
    kept, _ = _retained(tmp_path)
    ragged = next(row for row in kept if row.systematic_gene_name == "YBR020W")
    experiment, reference, publication = _dataset().create_experiment(ragged)
    keys = list(experiment.phenotype.metabolite_level)
    assert len(keys) == len(PEAK_KEYS) - 2
    assert "lysine_related_peak1" not in keys
    assert keys == [k for k in PEAK_KEYS.values() if k in ragged.levels]
    assert experiment.phenotype.metabolite_level["gshbiotin"] == 0.0
    assert experiment.phenotype.n_replicates == dict.fromkeys(keys, 1)
    assert experiment.phenotype.metabolite_level_se is None
    assert experiment.phenotype.measurement_type == MEASUREMENT_TYPE
    assert reference.phenotype_reference.metabolite_level == dict.fromkeys(keys, 1.0)
    assert reference.genome_reference.strain == "BY4741"
    assert publication.pubmed_id == "20610602" and publication.doi == module.DOI


def test_phenotype_declares_its_typed_absences(tmp_path: Path) -> None:
    kept, _ = _retained(tmp_path)
    full = kept[0]
    experiment, _, _ = _dataset().create_experiment(full)
    gaps = {gap.field: gap for gap in experiment.phenotype.provenance_gaps}
    assert set(gaps) == {"metabolite_level_se", "target_metabolite_ids"}
    assert "gshbiotin" in str(gaps["target_metabolite_ids"].note)
    assert experiment.phenotype.target_metabolite_ids is None
    ragged = next(row for row in kept if row.systematic_gene_name == "YML048W-A")
    ragged.levels.pop("gshbiotin")
    experiment, _, _ = _dataset().create_experiment(ragged)
    gaps = {gap.field: gap for gap in experiment.phenotype.provenance_gaps}
    assert set(gaps) == {"metabolite_level_se", "target_metabolite_ids"}
    assert "gshbiotin" not in str(gaps["target_metabolite_ids"].note)


def test_genotype_is_a_kanmx_deletion_keyed_on_the_normalized_source(
    tmp_path: Path,
) -> None:
    kept, _ = _retained(tmp_path)
    renamed = next(row for row in kept if row.perturbed_gene_name == "YAL035C-A")
    experiment, _, _ = _dataset().create_experiment(renamed)
    genotype = experiment.genotype
    assert isinstance(genotype, Genotype)
    perturbation = genotype.perturbations[0]
    assert perturbation.perturbation_type == "kanmx_deletion"
    assert perturbation.systematic_gene_name == "YAL034C-B"
    assert perturbation.perturbed_gene_name == "YAL035C-A"


# ---- environment ----------------------------------------------------------------- #
def test_medium_is_a_typed_dropout_of_the_library_sc() -> None:
    assert COOPER_SC.base_medium == "SC"
    assert COOPER_SC.state == SC.state == "liquid"
    assert [d.name for d in COOPER_SC.dropouts] == [
        "L-alanine",
        "L-asparagine",
        "L-cysteine",
        "L-glutamine",
        "glycine",
        "L-proline",
        "uracil",
    ]
    names = {c.compound.name for c in COOPER_SC.components}
    assert "L-serine" in names and "adenine" in names and "uracil" not in names
    assert COOPER_SC.provenance[0].quote.startswith(
        "Yeast growth was in synthetic complete media (adenine"
    )


def test_temperature_is_a_typed_gap_and_duration_is_sixteen_hours() -> None:
    environment = AminoAcidCooper2010Dataset._environment()
    assert environment.temperature is None
    assert environment.duration_hours == 16.0
    assert [gap.field for gap in environment.provenance_gaps] == ["temperature"]
    assert environment.media == COOPER_SC


# ---- provenance ------------------------------------------------------------------ #
def test_every_sourced_value_quote_is_verbatim_in_the_mirrored_paper() -> None:
    """The audit anchor is only real if the quote is still in the sha256-pinned file."""
    data_root = os.environ.get("DATA_ROOT")
    if data_root is None:
        pytest.skip("DATA_ROOT is not set")
    paper = osp.join(
        data_root, "torchcell-library", module.CITATION_KEY, module.PAPER_MD
    )
    if not osp.exists(paper):
        pytest.skip("the literature mirror is not mounted on this machine")
    text = Path(paper).read_text()
    for key, sourced in SOURCED_VALUES.items():
        assert sourced.quote in text, key
        assert sourced.provenance.sha256 == module.PAPER_MD_SHA256


def test_legend_quote_is_verbatim_in_the_deposited_legends_file() -> None:
    data_root = os.environ.get("DATA_ROOT")
    if data_root is None:
        pytest.skip("DATA_ROOT is not set")
    legends = (
        module.raw_mirror_dir(data_root)
        / module._RAW_FILES[module.LEGENDS_NAME]["relpath"]
    )
    if not legends.exists():
        pytest.skip("the raw mirror is not deposited on this machine")
    text = legends.read_bytes().decode("latin-1")
    sourced = LEGEND_SOURCED_VALUES["table4_legend"]
    assert sourced.quote in text
    assert sourced.provenance.sha256 == module.LEGENDS_SHA256


def test_deposit_refuses_a_hash_mismatch(tmp_path: Path) -> None:
    staged = tmp_path / "deposit"
    staged.mkdir()
    (staged / module.TABLE4_NAME).write_text("Sysname\tNAME\n")
    (staged / module.LEGENDS_NAME).write_bytes(b"x")
    (staged / module.SHA256SUMS_NAME).write_text(
        f"{module.TABLE4_SHA256}  {module.TABLE4_NAME}\n"
        f"{module.LEGENDS_SHA256}  {module.LEGENDS_NAME}\n"
    )
    with pytest.raises(RuntimeError, match="refusing to deposit"):
        module.deposit_raw_mirror(source_dir=staged, data_root=str(tmp_path))


# ---- end-to-end build ------------------------------------------------------------ #
_BASES = (0, 100, 200, 300, 400, 500, 600, 700)
_E2E_ROWS: list[tuple[str, str, int]] = [
    ("YBR208C", "DUR1,2", 0),
    ("YBR020W", "GAL1", 100),
    ("ybr020w", "GAL1", 200),
    ("YML048WA-", "GSF2", 300),
    ("YAL035C-A", "YAL035C-A", 400),
    ("YCL074W", "YCL074W", 500),
    ("BY4742", "BY4742", 600),
    ("YGL180W", "ATG1", 700),
]
_KEYS = [
    "lysine_related_peak1",
    "arginine",
    "gshbiotin",
    "n_acetylornithine",
    "leucine+isoleucine+citrulline",
    "glutamine+valine",
    "methionine+proline",
    "threonine",
    "alanine",
    "serine",
    "asparagine+tyrosine",
    "glycine",
    "lysine_a",
    "ornithine",
    "lysine_b",
    "glutamate",
    "aspartate",
]


def _levels(base: int) -> dict[str, float]:
    return {key: base + j + 0.5 for j, key in enumerate(_KEYS)}


def _ragged_levels() -> dict[str, float]:
    levels = _levels(100)
    del levels["lysine_related_peak1"], levels["aspartate"]
    levels["gshbiotin"] = 0.0
    return levels


def _e2e_table_text() -> str:
    lines = [_HEADER]
    for name, gene, base in _E2E_ROWS:
        cells = [f"{base + j + 0.5}" for j in range(len(_KEYS))]
        if base == 100:
            cells[0], cells[2], cells[16] = "-", "0", "-"
        lines.append("\t".join([name, gene, *cells]))
    return "\n".join(lines) + "\n"


class _Genome:
    """The one method the loader calls; an unknown name is a test bug, so it raises."""

    def resolve_gene_name(self, name: str) -> _Resolution:
        return _RESOLVER[name]


def _put_lmdb(path: Path, items: dict[bytes, Any]) -> None:
    path.mkdir(parents=True)
    env = lmdb.open(str(path), map_size=1 << 22)
    with env.begin(write=True) as txn:
        for key, value in items.items():
            txn.put(key, pickle.dumps(value))
    env.close()


def _essentiality_record(orf: str, essential: bool) -> dict[str, Any]:
    return {
        "experiment": {
            "phenotype": {"is_essential": essential},
            "genotype": {"perturbations": [{"systematic_gene_name": orf}]},
        }
    }


def _write_essentiality_store(data_root: Path) -> None:
    store = data_root / "data" / "torchcell" / "gene_essentiality_sgd" / "processed"
    _put_lmdb(
        store / "lmdb",
        {
            b"0": _essentiality_record("YGL180W", True),
            b"1": _essentiality_record("YBR208C", False),
        },
    )


@pytest.fixture
def built(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> AminoAcidCooper2010Dataset:
    data_root = tmp_path / "data_root"
    _write_essentiality_store(data_root)
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    root = tmp_path / "amino_acid_cooper2010"
    (root / "raw").mkdir(parents=True)
    (root / "raw" / module.TABLE4_NAME).write_text(_e2e_table_text())
    return AminoAcidCooper2010Dataset(root=str(root), genome=_Genome())


_ENVIRONMENT = Environment(
    media=COOPER_SC,
    temperature=None,
    duration_hours=16.0,
    provenance_gaps=[
        ProvenanceGap(
            field="temperature",
            reason=ProvenanceGapReason.not_reported_by_primary,
            note="growth temperature is not stated; the only degree values in the paper "
            "are GC-MS derivatization and oven settings",
        )
    ],
)
_SE_GAP = ProvenanceGap(
    field="metabolite_level_se",
    reason=ProvenanceGapReason.not_reported_by_primary,
    note="duplicates were averaged; no per-strain spread is released",
)


def _expected(
    orf: str, perturbed: str, levels: dict[str, float]
) -> tuple[dict[str, Any], dict[str, Any]]:
    keys = list(levels)
    phenotype = MetabolitePhenotype(
        metabolite_level=levels,
        metabolite_level_se=None,
        n_replicates=dict.fromkeys(keys, 1),
        measurement_type="ce_lif_peak_area_ratio_to_plate_mean",
        target_metabolite_ids=None,
        provenance_gaps=[
            _SE_GAP,
            ProvenanceGap(
                field="target_metabolite_ids",
                reason=ProvenanceGapReason.deferred_pending_source_review,
                note=module._TARGET_IDS_GSHBIOTIN_NOTE,
            ),
        ],
    )
    experiment = MetaboliteExperiment(
        dataset_name="AminoAcidCooper2010Dataset",
        genotype=Genotype(
            perturbations=[
                KanMxDeletionPerturbation(
                    systematic_gene_name=orf, perturbed_gene_name=perturbed
                )
            ]
        ),
        environment=_ENVIRONMENT,
        phenotype=phenotype,
    )
    reference = MetaboliteExperimentReference(
        dataset_name="AminoAcidCooper2010Dataset",
        genome_reference=ReferenceGenome(
            species="Saccharomyces cerevisiae", strain="BY4741"
        ),
        environment_reference=_ENVIRONMENT,
        phenotype_reference=MetabolitePhenotype(
            metabolite_level=dict.fromkeys(keys, 1.0),
            metabolite_level_se=None,
            n_replicates=dict.fromkeys(keys, 1),
            measurement_type="ce_lif_peak_area_ratio_to_plate_mean",
            target_metabolite_ids=None,
        ),
    )
    return experiment.model_dump(), reference.model_dump()


def test_build_stores_five_records_in_kept_row_order(
    built: AminoAcidCooper2010Dataset,
) -> None:
    """Rows 0, 1, 3, 4, 7 become LMDB records 0 to 4, keyed on (current ORF, normalized
    source name); the duplicate, the pseudogene and the parent row are not records.
    """
    assert len(built) == 5
    pairs = []
    for i in range(5):
        (perturbation,) = built[i]["experiment"]["genotype"]["perturbations"]
        pairs.append(
            (perturbation["systematic_gene_name"], perturbation["perturbed_gene_name"])
        )
    assert pairs == [
        ("YBR208C", "YBR208C"),
        ("YBR020W", "YBR020W"),
        ("YML048W-A", "YML048W-A"),
        ("YAL034C-B", "YAL035C-A"),
        ("YGL180W", "YGL180W"),
    ]


def test_full_row_record_maps_every_header_to_its_key(
    built: AminoAcidCooper2010Dataset,
) -> None:
    """Record 0 (row 0, base 0): key j holds j + 0.5 in file order, including the composite
    peaks ``leucine+isoleucine+citrulline``, ``glutamine+valine``, ``methionine+proline``
    and ``asparagine+tyrosine``; the reference is 1.0 on all 17 keys, ``n_replicates`` 1,
    ``measurement_type`` the linear ratio, temperature None with its typed gap.
    """
    experiment, reference = _expected("YBR208C", "YBR208C", _levels(0))
    assert built[0]["experiment"] == experiment
    assert built[0]["reference"] == reference
    level = built[0]["experiment"]["phenotype"]["metabolite_level"]
    assert level["glutamine+valine"] == 5.5
    assert level["aspartate"] == 16.5
    assert (
        built[0]["publication"]
        == Publication(
            pubmed_id="20610602",
            pubmed_url="https://pubmed.ncbi.nlm.nih.gov/20610602/",
            doi="10.1101/gr.105825.110",
            doi_url="https://doi.org/10.1101/gr.105825.110",
        ).model_dump()
    )


def test_ragged_row_record_carries_only_its_present_keys(
    built: AminoAcidCooper2010Dataset,
) -> None:
    """Record 1 (row 1, base 100): 15 keys, no ``lysine_related_peak1`` or ``aspartate``,
    ``gshbiotin`` an exact 0.0 (a released zero, not missing); its reference is 1.0 on the
    same 15 keys, and it keeps the FIRST GAL1 row's values (arginine 101.5, not 201.5).
    """
    experiment, reference = _expected("YBR020W", "YBR020W", _ragged_levels())
    assert built[1]["experiment"] == experiment
    assert built[1]["reference"] == reference
    phenotype = built[1]["experiment"]["phenotype"]
    assert len(phenotype["metabolite_level"]) == 15
    assert phenotype["metabolite_level"]["gshbiotin"] == 0.0
    assert phenotype["metabolite_level"]["arginine"] == 101.5


def test_ledger_json_records_every_rule_exactly(
    built: AminoAcidCooper2010Dataset,
) -> None:
    """``preprocess/dropped_records.json``: 8 source rows, 5 kept (83 values), the
    pseudogene dropped with its 17 values, the lowercase GAL1 row ledgered as a duplicate
    of row 1 with its own values, two normalizations (rows 2 and 3), YAL035C-A renamed,
    BY4742 excluded before the resolver, YGL180W flagged and kept (YBR208C is in the store
    but not essential, so not flagged).
    """
    ledger = json.loads(
        (Path(built.root) / "preprocess" / "dropped_records.json").read_text()
    )
    created_at = ledger.pop("created_at")
    assert created_at.endswith("+00:00")
    assert ledger == {
        "dataset": "AminoAcidCooper2010Dataset",
        "orf_rule": module.ORF_RULE,
        "normalization_rule": module.NORMALIZATION_RULE,
        "duplicate_rule": module.DUPLICATE_RULE,
        "exclusion_rule": module.EXCLUSION_RULE,
        "essentiality_rule": module.ESSENTIALITY_RULE,
        "replicate_rule": module.REPLICATE_RULE,
        "legend_contradiction": module.LEGEND_CONTRADICTION,
        "n_source_rows": 8,
        "n_kept": 5,
        "n_dropped_orfs": 1,
        "n_renamed": 1,
        "n_normalized": 2,
        "n_duplicate_rows": 1,
        "n_excluded_non_deletion": 1,
        "n_essentiality_flagged": 1,
        "n_values_kept": 83,
        "normalizations": [
            {"row_index": 2, "source_name": "ybr020w", "normalized_name": "YBR020W"},
            {
                "row_index": 3,
                "source_name": "YML048WA-",
                "normalized_name": "YML048W-A",
            },
        ],
        "dropped_orfs": [
            {
                "row_index": 5,
                "source_name": "YCL074W",
                "normalized_name": "YCL074W",
                "status": "non_gene_feature",
                "resolved_to": "YCL074W",
                "feature_type": "pseudogene",
                "n_values": 17,
            }
        ],
        "renamed_orfs": {"YAL035C-A": "YAL034C-B"},
        "duplicate_strain_rows": [
            {
                "row_index": 2,
                "source_name": "ybr020w",
                "normalized_name": "YBR020W",
                "systematic_gene_name": "YBR020W",
                "name_cell": "GAL1",
                "kept_row_index": 1,
                "levels": _levels(200),
            }
        ],
        "excluded_non_deletion": [
            {
                "row_index": 6,
                "source_name": "BY4742",
                "name_cell": "BY4742",
                "reason": "parent-strain row, not a deletion",
                "levels": _levels(600),
            }
        ],
        "essentiality_flagged": [
            {"row_index": 7, "systematic_gene_name": "YGL180W", "name_cell": "ATG1"}
        ],
    }


def test_data_csv_lists_kept_rows_with_empty_cells_for_missing_peaks(
    built: AminoAcidCooper2010Dataset,
) -> None:
    """``preprocess/data.csv``: header then one line per kept row in file order; the
    ragged row writes ``""`` for its two missing peaks and ``0.0`` for the released zero.
    """
    lines = (Path(built.root) / "preprocess" / "data.csv").read_text().splitlines()
    assert lines[0] == ",".join(
        [
            "systematic_gene_name",
            "perturbed_gene_name",
            "source_name",
            "name_cell",
            "row_index",
            *_KEYS,
        ]
    )

    def line(
        orf: str, perturbed: str, src: str, name: str, row: int, cells: list[str]
    ) -> str:
        return ",".join([orf, perturbed, src, name, str(row), *cells])

    def full(base: int) -> list[str]:
        return [f"{base + j + 0.5}" for j in range(17)]

    ragged = full(100)
    ragged[0], ragged[2], ragged[16] = "", "0.0", ""
    assert lines[1:] == [
        line("YBR208C", "YBR208C", "YBR208C", '"DUR1,2"', 0, full(0)),
        line("YBR020W", "YBR020W", "YBR020W", "GAL1", 1, ragged),
        line("YML048W-A", "YML048W-A", "YML048WA-", "GSF2", 3, full(300)),
        line("YAL034C-B", "YAL035C-A", "YAL035C-A", "YAL035C-A", 4, full(400)),
        line("YGL180W", "YGL180W", "YGL180W", "ATG1", 7, full(700)),
    ]


def test_side_files_gene_set_reference_index_and_sourced_values(
    built: AminoAcidCooper2010Dataset,
) -> None:
    """``gene_set.json`` is the five current ORFs sorted; the reference index groups the
    four 17-key records apart from the 15-key record; ``sourced_values.json`` holds the
    SOURCED_VALUES and legend keys sorted, then ``peak_notes``, with the n_samples value
    4382 and the legend's sha256 pin.
    """
    preprocess = Path(built.root) / "preprocess"
    assert json.loads((preprocess / "gene_set.json").read_text()) == [
        "YAL034C-B",
        "YBR020W",
        "YBR208C",
        "YGL180W",
        "YML048W-A",
    ]
    index = json.loads((preprocess / "experiment_reference_index.json").read_text())
    assert [entry["member_indices"] for entry in index] == [[0, 2, 3, 4], [1]]
    sourced = json.loads((preprocess / "sourced_values.json").read_text())
    assert list(sourced) == [
        *sorted([*module.SOURCED_VALUES, "table4_legend"]),
        "peak_notes",
    ]
    assert sourced["n_samples"]["value"] == 4382
    assert sourced["table4_legend"]["provenance"]["sha256"] == (
        "ca4983cda4c34318dbc3a05d10edf8e05df11f1560ffc99442df83a6a4399184"
    )
    assert list(sourced["peak_notes"]) == list(PEAK_KEYS)


def test_a_raw_file_placed_in_raw_dir_is_never_sha256_checked(
    built: AminoAcidCooper2010Dataset,
) -> None:
    """Finding: the Table 4 pin is enforced only inside ``download()`` (lines 1102 to
    1107), and PyG calls ``download()`` only when ``raw/`` lacks the file. The synthetic
    table in ``raw/`` does not carry the pinned sha256, and the build consumed it without
    a check. Pinned until ``process()`` verifies the raw file it reads.
    """
    raw = Path(built.raw_dir) / module.TABLE4_NAME
    digest = hashlib.sha256(raw.read_bytes()).hexdigest()
    assert digest != module.TABLE4_SHA256
    assert module.TABLE4_SHA256 == (
        "3c56cd492b4cab51249358e90c7e9731982960eb619e45b8850093e616a471fb"
    )
    assert len(built) == 5


def _mirror_table(data_root: Path) -> Path:
    path = module.raw_mirror_dir(str(data_root)) / "data" / module.TABLE4_NAME
    path.parent.mkdir(parents=True)
    path.write_text(_e2e_table_text())
    return path


def test_download_without_a_mirror_raises_with_the_recipe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No raw file and no mirror: ``FileNotFoundError`` naming the mirror path and the
    manual recipe; there is no network path.
    """
    data_root = tmp_path / "data_root"
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    source = (
        data_root
        / "torchcell-raw"
        / "cooperHighthroughputProfilingAmino2010"
        / "data"
        / "SupplementalTable4.txt"
    )
    with pytest.raises(FileNotFoundError) as excinfo:
        AminoAcidCooper2010Dataset(root=str(tmp_path / "ds"), genome=_Genome())
    assert str(excinfo.value) == (
        f"raw mirror file {source} is absent and the source is not scriptable; "
        f"deposit it with deposit_raw_mirror(). Recipe: {module.MANUAL_RECIPE}"
    )


def test_download_refuses_a_mirror_file_with_the_wrong_sha256(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A mirror file whose bytes are not the pinned Table 4 raises with both digests and
    nothing is linked into ``raw/``.
    """
    data_root = tmp_path / "data_root"
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    source = _mirror_table(data_root)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    with pytest.raises(RuntimeError) as excinfo:
        AminoAcidCooper2010Dataset(root=str(tmp_path / "ds"), genome=_Genome())
    assert str(excinfo.value) == (
        f"raw mirror {source} sha256 mismatch: got {digest}, expected "
        "3c56cd492b4cab51249358e90c7e9731982960eb619e45b8850093e616a471fb"
    )
    assert os.listdir(tmp_path / "ds" / "raw") == []


def test_download_links_a_verified_mirror_file_and_builds_from_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With the pin set to the synthetic table's sha256, ``download()`` symlinks the
    mirror file into ``raw/`` and the build reads it: five records.
    """
    data_root = tmp_path / "data_root"
    _write_essentiality_store(data_root)
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    source = _mirror_table(data_root)
    monkeypatch.setattr(
        module, "TABLE4_SHA256", hashlib.sha256(source.read_bytes()).hexdigest()
    )
    dataset = AminoAcidCooper2010Dataset(root=str(tmp_path / "ds"), genome=_Genome())
    link = tmp_path / "ds" / "raw" / module.TABLE4_NAME
    assert link.is_symlink()
    assert os.readlink(link) == str(source)
    assert len(dataset) == 5


def test_essential_gene_store_resolves_interned_records(tmp_path: Path) -> None:
    """``load_sgd_essential_genes`` keeps only ``is_essential is True`` records, splicing
    an interned ``$ref`` experiment back in: YAL001C (inline, True) and YBR001C (interned,
    True) are returned; YCR001W (False) is not.
    """
    processed = tmp_path / "data" / "torchcell" / "gene_essentiality_sgd" / "processed"
    _put_lmdb(
        processed / "interned",
        {b"exp-ybr001c": _essentiality_record("YBR001C", True)["experiment"]},
    )
    _put_lmdb(
        processed / "lmdb",
        {
            b"0": _essentiality_record("YAL001C", True),
            b"1": {"experiment": {"$ref": "exp-ybr001c"}},
            b"2": _essentiality_record("YCR001W", False),
        },
    )
    assert module.load_sgd_essential_genes(str(tmp_path)) == frozenset(
        {"YAL001C", "YBR001C"}
    )


def test_essential_gene_store_must_exist(tmp_path: Path) -> None:
    """An absent store raises rather than flagging nothing."""
    lmdb_dir = osp.join(
        str(tmp_path), "data/torchcell/gene_essentiality_sgd", "processed", "lmdb"
    )
    with pytest.raises(FileNotFoundError) as excinfo:
        module.load_sgd_essential_genes(str(tmp_path))
    assert str(excinfo.value) == (
        f"{lmdb_dir} is absent; build GeneEssentialitySgdDataset first, the "
        "essentiality flag is measured against it"
    )


def test_read_table4_refuses_a_short_row_and_a_reordered_header(tmp_path: Path) -> None:
    """A row with 3 cells names its 0-based index and the expected 19; the right columns
    in the wrong order are refused with empty unmapped and missing lists.
    """
    short = tmp_path / "short.txt"
    short.write_text(_HEADER + "\nYBR208C\tDUR1,2\t1.0\n")
    with pytest.raises(RuntimeError) as excinfo:
        read_table4(short)
    assert str(excinfo.value) == f"{short} row 0 has 3 cells, expected 19"
    swapped = list(EXPECTED_HEADER)
    swapped[2], swapped[3] = swapped[3], swapped[2]
    reordered = tmp_path / "reordered.txt"
    reordered.write_text("\t".join(swapped) + "\n")
    with pytest.raises(RuntimeError) as excinfo:
        read_table4(reordered)
    assert str(excinfo.value) == (
        f"{reordered} header is not the expected Table 4 header; unmapped columns [], "
        f"missing columns [], order {swapped}"
    )


def test_resolver_current_without_a_systematic_name_is_dropped(tmp_path: Path) -> None:
    """A kept status with ``systematic_name`` None still drops the row (line 957), with
    the status recorded as ``current`` and ``resolved_to`` None.
    """
    path = tmp_path / "t.txt"
    path.write_text("\n".join([_HEADER, _row("YBR208C", "DUR1,2", _full(1.0))]) + "\n")
    kept, ledger = retain_rows(
        read_table4(path),
        lambda name: _Resolution(GeneNameStatus.CURRENT, None, None),
        frozenset(),
        dataset_name="x",
    )
    assert kept == []
    assert [d.model_dump() for d in ledger.dropped_orfs] == [
        {
            "row_index": 0,
            "source_name": "YBR208C",
            "normalized_name": "YBR208C",
            "status": "current",
            "resolved_to": None,
            "feature_type": None,
            "n_values": 17,
        }
    ]


def test_read_sha256sums_skips_blank_lines_and_keeps_spaced_names(
    tmp_path: Path,
) -> None:
    """One split on the first whitespace run: the name keeps its inner space."""
    sums = tmp_path / "SHA256SUMS.txt"
    sums.write_text("aaa  first file.txt\n\n   \nbbb  second.doc\n")
    assert module._read_sha256sums(sums) == {
        "first file.txt": "aaa",
        "second.doc": "bbb",
    }


def _stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, dict[str, str]]:
    """A staged deposit whose two files' digests replace the pinned ones."""
    staged = tmp_path / "deposit"
    staged.mkdir()
    (staged / module.TABLE4_NAME).write_text("table four\n")
    (staged / module.LEGENDS_NAME).write_bytes(b"legend bytes")
    digests = {
        name: hashlib.sha256((staged / name).read_bytes()).hexdigest()
        for name in (module.TABLE4_NAME, module.LEGENDS_NAME)
    }
    (staged / module.SHA256SUMS_NAME).write_text(
        "".join(f"{digest}  {name}\n" for name, digest in digests.items())
    )
    monkeypatch.setattr(
        module,
        "_RAW_FILES",
        {
            name: {"relpath": f"data/{name}", "sha256": digest}
            for name, digest in digests.items()
        },
    )
    return staged, digests


def test_deposit_copies_both_files_and_writes_a_typed_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With the pins set to the staged files' digests the deposit copies both under
    ``data/``, records Table 4 as ``raw_data`` and the legends as ``si_data``, each a
    ``manual_browser`` retrieval carrying the recipe, and ``load_manifest`` reads it back.
    A second deposit of the same bytes is a no-op; a mirror file with other bytes is
    refused.
    """
    staged, digests = _stage(tmp_path, monkeypatch)
    data_root = str(tmp_path / "data_root")
    root = module.deposit_raw_mirror(source_dir=staged, data_root=data_root)
    assert root == Path(data_root) / "torchcell-raw" / module.CITATION_KEY
    assert (root / "data" / module.TABLE4_NAME).read_text() == "table four\n"
    manifest = module.load_manifest(data_root)
    assert manifest.citation_key == "cooperHighthroughputProfilingAmino2010"
    assert manifest.doi == "10.1101/gr.105825.110"
    assert manifest.provenance_complete is True
    assert [
        (f.path, f.role, f.bytes, f.sha256, f.source) for f in manifest.files
    ] == [
        ("data/SupplementalTable4.txt", "raw_data", 11, digests[module.TABLE4_NAME],
         module.DC1_URL),
        ("data/Supplemental_Table_Legends.doc", "si_data", 12,
         digests[module.LEGENDS_NAME], module.DC1_URL),
    ]  # fmt: skip
    retrievals = [f.model_dump(mode="json")["retrieval"] for f in manifest.files]
    assert [
        (r["method"], r["source_url"], r["retriever"], r["retrieved_at"], r["sha256"])
        for r in retrievals
    ] == [
        ("manual_browser", "https://genome.cshlp.org/content/20/9/1288", "manual",
         "2026-09-15", digests[module.TABLE4_NAME]),
        ("manual_browser", "https://genome.cshlp.org/content/20/9/1288", "manual",
         "2026-09-15", digests[module.LEGENDS_NAME]),
    ]  # fmt: skip
    assert retrievals[0]["params"]["retrieval_command"] == module.MANUAL_RECIPE
    module.deposit_raw_mirror(source_dir=staged, data_root=data_root)
    assert (root / "data" / module.TABLE4_NAME).read_text() == "table four\n"
    dest = root / "data" / module.TABLE4_NAME
    dest.write_text("tampered\n")
    tampered = hashlib.sha256(b"tampered\n").hexdigest()
    with pytest.raises(RuntimeError) as excinfo:
        module.deposit_raw_mirror(source_dir=staged, data_root=data_root)
    assert str(excinfo.value) == (
        f"{dest} exists with sha256 {tampered} != pinned "
        f"{digests[module.TABLE4_NAME]}; refusing to overwrite"
    )


def test_deposit_refuses_a_sha256sums_listing_that_disagrees_with_the_pin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The listing is checked against the pin before any file is hashed or copied."""
    staged, digests = _stage(tmp_path, monkeypatch)
    (staged / module.SHA256SUMS_NAME).write_text(
        f"{'0' * 64}  {module.TABLE4_NAME}\n"
        f"{digests[module.LEGENDS_NAME]}  {module.LEGENDS_NAME}\n"
    )
    with pytest.raises(RuntimeError) as excinfo:
        module.deposit_raw_mirror(
            source_dir=staged, data_root=str(tmp_path / "data_root")
        )
    assert str(excinfo.value) == (
        f"SHA256SUMS.txt lists SupplementalTable4.txt as {'0' * 64}, pinned "
        f"{digests[module.TABLE4_NAME]}"
    )
    assert not (tmp_path / "data_root").exists()
