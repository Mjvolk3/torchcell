# tests/torchcell/datasets/scerevisiae/test_hoepfner2014.py
"""Unit tests for the Hoepfner 2014 HIP-HOP loader's record construction.

The loader's whole surface is how it turns one deposited column header into a typed
condition, so these tests drive ``_column_meta`` / ``_environment`` / ``_reference`` /
``_genotype`` directly on a synthetic header plus a synthetic Table S1 mapping. What is
pinned is exactly what the review found broken: the compound is keyed on its CLEAN name
and carries a structure identifier, the screen id is a typed field rather than text inside
that name, the medium is the shared library object on BOTH arms, the vehicle carries its
own identity, the two assays record their own exposure durations, and a compound that
resolves to no identifier takes its column out of the build.
"""

from __future__ import annotations

import os.path as osp
from pathlib import Path
from typing import Any

import pytest

from torchcell.datamodels.media import YPD_LIQUID
from torchcell.datamodels.schema import (
    AssayType,
    ConcentrationUnit,
    DoseBasis,
    MeasurementType,
    SampleUnit,
    SmallMoleculePerturbation,
)
from torchcell.datasets.scerevisiae import hoepfner2014 as module
from torchcell.datasets.scerevisiae.hoepfner2014 import (
    DROP_RULE,
    SOURCED_VALUES,
    EnvChemgenHoepfner2014Dataset,
    _has_identifier,
    load_table_s5_strains,
)
from torchcell.sequence.genome.scerevisiae.s288c import GeneNameStatus

REPO_ROOT = Path(__file__).resolve().parents[4]

# Two real Table S1 rows: amitriptyline resolves, boromycin is the one compound whose
# released SMILES RDKit cannot parse and whose curated row carries no identifier.
AMITRIPTYLINE_SMILES = "CN(C)CCC=C2c1ccccc1CCc3ccccc23"
BOROMYCIN_SMILES = (
    "CC(C)C(N)C(=O)OC(C)C1C/C=C\\CCC(O)C(C)(C)C7CCC(C)C4(OB25OC(C(=O)O1)C3(O2)"
    "OC(CCC3C)C(C)(C)C(O)CCCC6CC(OC(=O)C4O5)C(C)O6)O7"
)

META: dict[str, dict[str, str | None]] = {
    "3": {"common_name": "Amitriptyline", "smiles": AMITRIPTYLINE_SMILES},
    "409": {"common_name": "Boromycin", "smiles": BOROMYCIN_SMILES},
    "777": {"common_name": None, "smiles": AMITRIPTYLINE_SMILES},
    "888": {"common_name": "Unreleased", "smiles": None},
}

HEADER = [
    '"Systematic Name"',
    '"Ad. scores for Exp. 3_50_HIP_0077"',
    '"Ad. scores for Exp. 3_50_HIP_0077 z-score"',
    '"MADL scores for Exp. 3_50_HIP_0091"',
    '"Ad. scores for Exp. 409_1.2_HIP_0077"',
    '"Ad. scores for Exp. 777_10_HIP_0077"',
    '"Ad. scores for Exp. 888_10_HIP_0077"',
    '"Ad. scores for Exp. 3_50_HOP_0077"',
]


class _FakeTxn:
    """A dict-backed stand-in for the interned LMDB write transaction."""

    def __init__(self) -> None:
        self.store: dict[bytes, bytes] = {}

    def get(self, key: bytes) -> bytes | None:
        return self.store.get(key)

    def put(self, key: bytes, value: bytes) -> None:
        self.store[key] = value


def _dataset() -> EnvChemgenHoepfner2014Dataset:
    """The loader without its PyG init: these tests never touch disk or an LMDB."""
    dataset = EnvChemgenHoepfner2014Dataset.__new__(EnvChemgenHoepfner2014Dataset)
    dataset.name = "EnvChemgenHoepfner2014Dataset"
    return dataset


def _columns(assay: str = "HIP") -> tuple[list[Any], list[tuple[int, str]]]:
    return _dataset()._column_meta(HEADER, assay, META, _FakeTxn())


# ---- compound identity and the drop rule ---------------------------------------- #
def test_compound_name_is_clean_and_carries_no_experiment_tag() -> None:
    assert EnvChemgenHoepfner2014Dataset._compound_name("3", META) == "Amitriptyline"
    assert EnvChemgenHoepfner2014Dataset._compound_name("777", META) == "CMB777"
    for cmb in META:
        name = EnvChemgenHoepfner2014Dataset._compound_name(cmb, META)
        assert "[" not in name and "HIP" not in name and "_" not in name


def test_kept_compound_carries_a_structure_identifier() -> None:
    kept, _ = _columns()
    compound = kept[0].env_dump["perturbations"][0]["compound"]
    assert compound["inchikey"] == "KRMDCWKBEZIMAB-UHFFFAOYSA-N"
    assert compound["name"] == "amitriptyline"  # the table's canonical spelling
    assert compound["smiles"] == AMITRIPTYLINE_SMILES


def test_structure_only_proprietary_compound_resolves_from_its_smiles() -> None:
    """A CMB id with no released name still gets an InChIKey from its SMILES."""
    kept, _ = _columns()
    by_cmb = {col.cmb: col for col in kept}
    compound = by_cmb["777"].env_dump["perturbations"][0]["compound"]
    assert compound["name"] == "CMB777"
    assert compound["inchikey"] == "KRMDCWKBEZIMAB-UHFFFAOYSA-N"


def test_unidentifiable_compound_column_is_dropped_by_the_rule() -> None:
    kept, dropped = _columns()
    assert {cmb for _, cmb in dropped} == {"409"}
    assert "409" not in {col.cmb for col in kept}
    assert "structure identifier" in DROP_RULE


def test_compound_without_a_released_smiles_is_the_encodable_filter_not_a_drop() -> (
    None
):
    """CMB888 has no structure at all: it never enters the build and is not a 'drop'."""
    kept, dropped = _columns()
    assert "888" not in {col.cmb for col in kept}
    assert "888" not in {cmb for _, cmb in dropped}


def test_z_score_and_other_assay_columns_are_not_kept() -> None:
    kept, _ = _columns("HIP")
    assert [col.index for col in kept] == [1, 3, 5]
    hop_kept, _ = _columns("HOP")
    assert [col.index for col in hop_kept] == [7]


# ---- screen id ------------------------------------------------------------------ #
def test_screen_id_is_a_typed_field_and_separates_two_screens_of_one_dose() -> None:
    kept, _ = _columns()
    by_study = {col.pheno_base["screen_id"]: col for col in kept if col.cmb == "3"}
    assert set(by_study) == {"0077", "0091"}
    doses = {
        col.env_dump["perturbations"][0]["concentration"]["value"]
        for col in by_study.values()
    }
    assert doses == {
        50.0
    }  # same compound, same dose, two screens: only screen_id parts


def test_n_samples_follows_the_column_prefix() -> None:
    kept, _ = _columns()
    by_study = {col.pheno_base["screen_id"]: col for col in kept if col.cmb == "3"}
    assert by_study["0077"].pheno_base["n_samples"] == 2  # 'Ad.'
    assert by_study["0091"].pheno_base["n_samples"] == 1  # 'MADL'
    assert by_study["0077"].pheno_base["sample_unit"] == SampleUnit.technical_replicate


# ---- environment ---------------------------------------------------------------- #
def test_both_arms_use_the_shared_media_library_object() -> None:
    dataset = _dataset()
    kept, _ = _columns()
    assert kept[0].env_dump["media"] == YPD_LIQUID.model_dump()
    reference = dataset._reference("HIP", "0077")
    assert reference.environment_reference.media == YPD_LIQUID


def test_dose_is_typed_and_the_vehicle_carries_its_own_identity() -> None:
    kept, _ = _columns()
    perturbation = kept[0].env_dump["perturbations"][0]
    assert perturbation["concentration"] == {
        "value": 50.0,
        "unit": ConcentrationUnit.micromolar,
        "basis": DoseBasis.IC30,
    }
    solvent = perturbation["solvent"]
    assert solvent["name"] == "DMSO" and solvent["percent"] == 2.0
    assert solvent["compound"]["inchikey"] == "IAZDPXIOMUYVGZ-UHFFFAOYSA-N"
    assert solvent["compound"]["pubchem_cid"] == 679


def test_hip_duration_is_a_typed_gap_and_hop_states_both_durations() -> None:
    dataset = _dataset()
    hip = dataset._environment("HIP", [])
    assert hip.duration_hours is None
    assert hip.duration_generations == 20.0
    assert [gap.field for gap in hip.provenance_gaps] == ["duration_hours"]
    hop = dataset._environment("HOP", [])
    assert (hop.duration_hours, hop.duration_generations) == (16.0, 5.0)
    assert hop.provenance_gaps == []
    assert hip.temperature is not None and hip.temperature.value == 30.0


# ---- reference ------------------------------------------------------------------ #
def test_reference_is_a_vehicle_control_on_the_joinable_strain_token() -> None:
    reference = _dataset()._reference("HOP", "0077")
    genome = reference.genome_reference
    assert genome.strain == "BY4743" and genome.ploidy == "diploid"
    vehicles = [
        p
        for p in reference.environment_reference.perturbations
        if isinstance(p, SmallMoleculePerturbation)
    ]
    assert len(vehicles) == 1
    vehicle = vehicles[0]
    assert vehicle.compound.name == "dimethyl sulfoxide"
    assert vehicle.concentration.value == 2.0
    assert vehicle.concentration.unit is ConcentrationUnit.percent_v_v
    phenotype = reference.phenotype_reference
    assert phenotype.environment_response == 0.0
    assert phenotype.n_samples == 4  # conservative lower end of "four to eight"
    assert phenotype.screen_id == "0077"
    assert phenotype.assay_type is AssayType.pooled_competitive_growth_barcode


def test_every_phenotype_declares_its_uncertainty_absences() -> None:
    kept, _ = _columns()
    gapped = {gap["field"] for gap in kept[0].pheno_base["provenance_gaps"]}
    assert gapped == {
        "environment_response_se",
        "environment_response_uncertainty",
        "environment_response_uncertainty_type",
    }
    assert kept[0].pheno_base["measurement_type"] == MeasurementType.sensitivity_score
    assert (
        kept[0].pheno_base["assay_type"] is AssayType.pooled_competitive_growth_barcode
    )


# ---- genotype ------------------------------------------------------------------- #
def test_hip_is_a_heterozygous_cnv_and_hop_a_kanmx_deletion() -> None:
    dataset = _dataset()
    hip = dataset._genotype("HIP", "YAL001C").perturbations[0]
    assert hip.perturbation_type == "engineered_copy_number"
    assert (hip.copy_number, hip.reference_copy_number) == (1.0, 2.0)
    assert hip.marker == "KanMX"
    hop = dataset._genotype("HOP", "YAL001C").perturbations[0]
    assert hop.perturbation_type == "kanmx_deletion"


# ---- provenance ----------------------------------------------------------------- #
def test_table_s5_strain_list_is_sha256_pinned() -> None:
    strains = load_table_s5_strains(REPO_ROOT)
    assert len(strains) == 185
    assert sum(row["is_positional"] == "True" for row in strains.values()) == 157
    assert strains["YBR271W"]["mutation"] == "Chromosome XI aneuploidy"


def test_table_s5_csv_hash_mismatch_raises(tmp_path: Path, monkeypatch: Any) -> None:
    fake = tmp_path / module.TABLE_S5_STRAINS_CSV
    fake.parent.mkdir(parents=True)
    fake.write_text("orf,gene\nYAL001C,TFC3\n")
    with pytest.raises(RuntimeError, match="sha256"):
        load_table_s5_strains(tmp_path)


def test_every_sourced_value_quote_is_verbatim_in_the_mirrored_paper() -> None:
    """The audit anchor is only real if the quote is still in the sha256-pinned file."""
    import os

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


def test_identifier_rule_reads_every_identity_field() -> None:
    from torchcell.datamodels.schema import Compound

    assert not _has_identifier(Compound(name="x"))
    assert _has_identifier(Compound(name="x", pubchem_cid=1))
    assert _has_identifier(Compound(name="x", chebi_id="CHEBI:1"))
    assert _has_identifier(Compound(name="x", inchikey="KRMDCWKBEZIMAB-UHFFFAOYSA-N"))


# ---- the ORF retention rule ------------------------------------------------------ #
class _Resolution:
    """The three fields of ``GeneNameResolution`` the loader reads."""

    def __init__(self, status: GeneNameStatus, name: str | None, feature: str | None):
        self.status = status
        self.systematic_name = name
        self.feature_type = feature


_RESOLVER: dict[str, _Resolution] = {
    "YAL001C": _Resolution(GeneNameStatus.CURRENT, "YAL001C", None),
    "YAL034C-B": _Resolution(GeneNameStatus.CURRENT, "YAL034C-B", None),
    "YAL035C-A": _Resolution(GeneNameStatus.RENAMED, "YAL034C-B", None),
    "YCL074W": _Resolution(GeneNameStatus.NON_GENE_FEATURE, "YCL074W", "pseudogene"),
    "R0010W": _Resolution(GeneNameStatus.RETIRED, "R0010W", None),
}


def _matrix(tmp_path: Path) -> str:
    """A HIP matrix with one row per resolver outcome; every kept cell non-empty."""
    rows = [
        "\t".join(HEADER),
        "\t".join(['"YAL001C"'] + ['"0.1"'] * 7),
        "\t".join(['"YAL034C-B"'] + ['"0.2"'] * 7),
        "\t".join(['"YAL035C-A"'] + ['"0.3"'] * 7),
        # column 5 (777_10, a kept HIP column) empty; column 7 is the HOP column
        "\t".join(['"YCL074W"'] + ['"0.4"'] * 4 + ['""'] + ['"0.4"'] * 2),
        "\t".join(['"R0010W"'] + ['"0.5"'] * 7),
    ]
    path = tmp_path / "HIP_scores.txt"
    path.write_text("\n".join(rows) + "\n")
    return str(path)


def _records(tmp_path: Path) -> tuple[list[dict[str, Any]], Any]:
    import pickle

    from pydantic import TypeAdapter

    from torchcell.datamodels.schema import ExperimentType

    columns, dropped = _columns("HIP")
    counts = module._BuildCounts()
    refs = {("HIP", study): {"$ref": study} for study in {c.study for c in columns}}
    records = [
        pickle.loads(value)
        for value in _dataset()._iter_records(
            _matrix(tmp_path),
            "HIP",
            {"YAL001C", "YAL034C-B", "YCL074W", "R0010W"},
            columns,
            dropped,
            refs,
            {"$ref": "pub"},
            counts,
            frozenset({"YAL034C-B"}),
            TypeAdapter(ExperimentType).validate_python,
            lambda name: _RESOLVER[name],
        )
    ]
    return records, counts


def test_non_gene_and_retired_rows_are_dropped_with_their_status(
    tmp_path: Path,
) -> None:
    records, counts = _records(tmp_path)
    kept_rows = {
        r["experiment"]["genotype"]["perturbations"][0]["perturbed_gene_name"]
        for r in records
    }
    assert kept_rows == {"YAL001C", "YAL034C-B", "YAL035C-A"}
    dropped = {d.source_name: d for d in counts.dropped_orfs["HIP"]}
    assert set(dropped) == {"R0010W", "YCL074W"}
    assert (dropped["YCL074W"].status, dropped["YCL074W"].feature_type) == (
        "non_gene_feature",
        "pseudogene",
    )
    assert dropped["R0010W"].status == "retired"
    # the pseudogene row loses its non-empty cells in the kept HIP columns only
    n_kept_columns = len(_columns("HIP")[0])
    assert dropped["R0010W"].n_records == n_kept_columns
    assert dropped["YCL074W"].n_records == n_kept_columns - 1


def test_renamed_row_is_a_distinct_strain_of_the_current_gene(tmp_path: Path) -> None:
    records, counts = _records(tmp_path)
    assert counts.renamed_orfs["HIP"] == {"YAL035C-A": "YAL034C-B"}
    by_source = {
        r["experiment"]["genotype"]["perturbations"][0]["perturbed_gene_name"]: r[
            "experiment"
        ]["genotype"]["perturbations"][0]["systematic_gene_name"]
        for r in records
    }
    assert by_source["YAL035C-A"] == "YAL034C-B"
    assert by_source["YAL034C-B"] == "YAL034C-B"
    # Table S5 names physical strains: the listed gene's own row is flagged, the
    # renamed merged-ORF strain that now maps to that gene is not
    assert counts.flagged == {"YAL034C-B": len(_columns("HIP")[0])}
