# tests/torchcell/datasets/scerevisiae/test_hoepfner2014_synthetic.py
# [[tests.torchcell.datasets.scerevisiae.test_hoepfner2014_synthetic]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/scerevisiae/test_hoepfner2014_synthetic.py
"""Hoepfner 2014 HIP-HOP loader built end to end on synthetic score matrices.

``process()`` reads three things besides ``raw/``: the S288C gene universe from the
genomes tier (``$DATA_ROOT/torchcell-genomes/sgd_S288C_R64-4-1_20230830``, two FASTAs
sha256-pinned by a ``GenomeManifest``), the committed Table S5 strain CSV of experiment
017 (sha256-pinned, read from the worktree), and the injected genome's
``resolve_gene_name``. ``DATA_ROOT`` is pointed into ``tmp_path`` with a five-gene tier
(YAL001C, YAL034C-B, YBR271W, YLR074C, YCL074W in the ORF FASTA; YNCA0001W in the RNA
FASTA) and the resolver is a dict-backed stub returning real ``GeneNameResolution``s.
``Table_S1.xls`` is written with openpyxl under the release's ``.xls`` name; pandas
sniffs the zip signature, so ``_load_compound_meta`` runs unchanged.

Table S1: CMB 3 Amitriptyline (SMILES released, curated: InChIKey
KRMDCWKBEZIMAB-UHFFFAOYSA-N, PubChem 2160, ChEBI:2666), CMB 409 Boromycin (released
SMILES RDKit cannot parse, no curated identifier: the one DROPPED compound), CMB 777
(structure only, no name: resolves from its SMILES as ``CMB777``), CMB 888 Unreleased
(name only, no SMILES: the encodable-only filter, never a drop), plus a ``244, 1818``
multi-id structure row, an empty-id row, a nameless CMB 1234 row and a NaN-SMILES row
that exercise the cell parser.

HIP header columns (index: header): 1 ``Ad. 3_50_HIP_0077`` (kept, n = 2), 2 its z-score
companion (skipped), 3 ``MADL 3_50_HIP_0091`` (kept, n = 1), 4 ``Ad. 409_1.2_HIP_0077``
(dropped compound), 5 ``Ad. 777_10_HIP_0077`` (kept, n = 2), 6 ``Ad. 888_10_HIP_0077``
(filtered), 7 ``Ad. 3_50_HOP_0077`` (other assay, skipped). HOP header: 1
``Ad. 3_50_HOP_0077`` (kept), 2 ``Ad. 409_1.2_HOP_0077`` (dropped compound).

HIP rows (cells at columns 1..7) and what they become:

    YAL001C    0.1 0.11 -1.5 0.9 2.25 0.3 9.9   CURRENT: records 0, 1, 2
    YAL035C-A  0.2 0.21 ""   ""  0.4  0.3 9.9   RENAMED -> YAL034C-B: records 3, 4
    YCL074W    0.5 0.5  0.5  0.5 ""   0.5 9.9   NON_GENE_FEATURE pseudogene: dropped, 2 cells
    YBR271W   -3.0 -3.1 -2.0 -1.0 -0.5 0.3 9.9  CURRENT, Table S5 positional: records 5, 6, 7
    YHR999W    0.7 (all seven)                  CURRENT but not in the FASTA: dropped, 3 cells
    YLR074C    0.6 (line ends after column 1)  CURRENT, Table S5 non-positional: record 8

HOP rows: YAL001C 1.0 0.7 (record 9), YBR271W -0.25 "" (record 10), YAL034C-B 0.33
with no second cell (record 11; the short line exercises the column-bound guard), and
YAL001C again 0.99 "" (record 12; a repeated ORF row is stored, see the Finding test).

Ledger arithmetic: kept HIP 9 + HOP 4 = 13; dropped-compound cells are the non-empty
column-4 cells of KEPT HIP rows (YAL001C 0.9, YBR271W -1.0 = 2) plus the column-2 cell
of kept HOP rows (YAL001C 0.7 = 1), so 3 over 2 columns; the pseudogene row loses its
two non-empty kept-column cells and the FASTA-absent row its three. References are one
per (assay, study): HIP_0077 members [0, 2, 3, 4, 5, 7, 8], HIP_0091 [1, 6], HOP_0077
[9, 10, 11, 12]. Table S5 flags: YBR271W 3 HIP records, YLR074C 1.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import openpyxl
import pytest

from torchcell.data import RawSha256MismatchError
from torchcell.data.experiment_dataset import verify_raw_files
from torchcell.datamodels.media import YPD_LIQUID
from torchcell.datamodels.schema import (
    AssayType,
    Compound,
    Concentration,
    ConcentrationUnit,
    DoseBasis,
    EngineeredCopyNumberPerturbation,
    Environment,
    EnvironmentResponseExperiment,
    EnvironmentResponseExperimentReference,
    EnvironmentResponsePhenotype,
    Genotype,
    KanMxDeletionPerturbation,
    MeasurementType,
    Publication,
    ReferenceGenome,
    SampleUnit,
    SmallMoleculePerturbation,
    Solvent,
    Temperature,
)
from torchcell.datasets.scerevisiae import hoepfner2014 as m
from torchcell.literature.manifest import ArtifactRecord
from torchcell.sequence.genome.registry import GenomeManifest
from torchcell.sequence.genome.scerevisiae.s288c import (
    GeneNameResolution,
    GeneNameStatus,
)
from torchcell.verification.sourced import ProvenanceGap, ProvenanceGapReason

_CITATION_KEY = "hoepfnerHighresolutionChemicalDissection2014"
_DOI = "10.1016/j.micres.2013.11.004"
_DATASET = "EnvChemgenHoepfner2014Dataset"
_AMITRIPTYLINE = "CN(C)CCC=C2c1ccccc1CCc3ccccc23"
_BOROMYCIN = (
    "CC(C)C(N)C(=O)OC(C)C1C/C=C\\CCC(O)C(C)(C)C7CCC(C)C4(OB25OC(C(=O)O1)C3(O2)"
    "OC(CCC3C)C(C)(C)C(O)CCCC6CC(OC(=O)C4O5)C(C)O6)O7"
)

_RESOLUTIONS: dict[str, tuple[GeneNameStatus, str, str | None]] = {
    "YAL001C": (GeneNameStatus.CURRENT, "YAL001C", None),
    "YAL034C-B": (GeneNameStatus.CURRENT, "YAL034C-B", None),
    "YAL035C-A": (GeneNameStatus.RENAMED, "YAL034C-B", None),
    "YBR271W": (GeneNameStatus.CURRENT, "YBR271W", None),
    "YCL074W": (GeneNameStatus.NON_GENE_FEATURE, "YCL074W", "pseudogene"),
    "YHR999W": (GeneNameStatus.CURRENT, "YHR999W", None),
    "YLR074C": (GeneNameStatus.CURRENT, "YLR074C", None),
}


class _StubGenome:
    """Only ``resolve_gene_name`` is read from the injected genome."""

    def resolve_gene_name(self, name: str) -> GeneNameResolution:
        status, systematic, feature = _RESOLUTIONS[name]
        return GeneNameResolution(
            input_name=name,
            status=status,
            systematic_name=systematic,
            feature_type=feature,
        )


_HIP_HEADER = [
    "Systematic Name",
    "Ad. scores for Exp. 3_50_HIP_0077",
    "Ad. scores for Exp. 3_50_HIP_0077 z-score",
    "MADL scores for Exp. 3_50_HIP_0091",
    "Ad. scores for Exp. 409_1.2_HIP_0077",
    "Ad. scores for Exp. 777_10_HIP_0077",
    "Ad. scores for Exp. 888_10_HIP_0077",
    "Ad. scores for Exp. 3_50_HOP_0077",
]
_HIP_ROWS: list[list[str]] = [
    ["YAL001C", "0.1", "0.11", "-1.5", "0.9", "2.25", "0.3", "9.9"],
    ["YAL035C-A", "0.2", "0.21", "", "", "0.4", "0.3", "9.9"],
    ["YCL074W", "0.5", "0.5", "0.5", "0.5", "", "0.5", "9.9"],
    ["YBR271W", "-3.0", "-3.1", "-2.0", "-1.0", "-0.5", "0.3", "9.9"],
    ["YHR999W"] + ["0.7"] * 7,
    ["YLR074C", "0.6"],
]
_HOP_HEADER = [
    "Systematic Name",
    "Ad. scores for Exp. 3_50_HOP_0077",
    "Ad. scores for Exp. 409_1.2_HOP_0077",
]
_HOP_ROWS: list[list[str]] = [
    ["YAL001C", "1.0", "0.7"],
    ["YBR271W", "-0.25", ""],
    ["YAL034C-B", "0.33"],
    ["YAL001C", "0.99", ""],
]


def _quoted_matrix(header: list[str], rows: list[list[str]]) -> str:
    """The deposited format: every cell double-quoted, tab-separated."""
    lines = ["\t".join(f'"{cell}"' for cell in header)]
    lines += ["\t".join(f'"{cell}"' for cell in row) for row in rows]
    return "\n".join(lines) + "\n"


def _write_table_s1(path: Path) -> None:
    workbook = openpyxl.Workbook()
    known = workbook.active
    known.title = "Reference Substances known MoA"
    known.append(["CMB ID", "Common Name", "IC30 (uM)"])
    known.append([3, "Amitriptyline", 50])
    known.append([None, "Nameless", 5])
    known.append([409, "Boromycin", 1.2])
    known.append([888, "Unreleased", 10])
    novel = workbook.create_sheet("Substances novel MoA")
    novel.append(["CMB ID", "Common Name"])
    novel.append(["1234", None])
    novel.append([1235, "Novel compound"])
    structures = workbook.create_sheet("All Structures")
    structures.append(["CMB ID", "SMILE string"])
    structures.append([3, _AMITRIPTYLINE])
    structures.append(["409", _BOROMYCIN])
    structures.append([777, _AMITRIPTYLINE])
    structures.append(["244, 1818", "CC(=O)O"])
    structures.append([555, None])
    workbook.save(path)


def _write_raw(raw: Path) -> None:
    raw.mkdir(parents=True, exist_ok=True)
    (raw / "HIP_scores.txt").write_text(_quoted_matrix(_HIP_HEADER, _HIP_ROWS))
    (raw / "HOP_scores.txt").write_text(_quoted_matrix(_HOP_HEADER, _HOP_ROWS))
    _write_table_s1(raw / "Table_S1.xls")


_ORF_FASTA = (
    ">YAL001C TFC3 SGDID:S000000001\nATGGTA\n"
    ">YAL034C-B MTW1 SGDID:S000028596\nATGACC\n"
    ">YBR271W EFM2 SGDID:S000000475\nATGCAT\n"
    ">YLR074C BUD20 SGDID:S000004064\nATGAAG\n"
    ">YCL074W YCL074W SGDID:S000000579\nATGTTT\n"
)
_RNA_FASTA = ">YNCA0001W tA(UGC)A SGDID:S000006534\nGGGCGT\n"


def _write_genomes_tier(data_root: Path) -> None:
    """The SGD assembly set: two FASTAs pinned by a ``GenomeManifest``."""
    tier = data_root / "torchcell-genomes" / "sgd_S288C_R64-4-1_20230830"
    tier.mkdir(parents=True)
    records = []
    for name, text in (
        ("orf_coding_all_R64-4-1_20230830.fasta", _ORF_FASTA),
        ("rna_coding_R64-4-1_20230830.fasta", _RNA_FASTA),
    ):
        (tier / name).write_text(text)
        records.append(
            ArtifactRecord(
                path=name,
                role="sequence",
                bytes=len(text.encode()),
                sha256=hashlib.sha256(text.encode()).hexdigest(),
            )
        )
    manifest = GenomeManifest(
        assembly_set="sgd_S288C_R64-4-1_20230830",
        organism="Saccharomyces cerevisiae",
        strain_or_population="S288C",
        source="SGD",
        release="R64-4-1_20230830",
        files=records,
        provenance_complete=True,
        created_at="2026-09-27T00:00:00+00:00",
    )
    (tier / "manifest.json").write_text(manifest.model_dump_json(indent=2))


@pytest.fixture
def dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> m.EnvChemgenHoepfner2014Dataset:
    data_root = tmp_path / "data_root"
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    _write_genomes_tier(data_root)
    root = tmp_path / "env_chemgen_hoepfner2014"
    _write_raw(root / "raw")
    return m.EnvChemgenHoepfner2014Dataset(root=str(root), genome=_StubGenome())


_AMITRIPTYLINE_COMPOUND = Compound(
    name="amitriptyline",
    inchikey="KRMDCWKBEZIMAB-UHFFFAOYSA-N",
    smiles=_AMITRIPTYLINE,
    pubchem_cid=2160,
    chebi_id="CHEBI:2666",
)
_CMB777_COMPOUND = Compound(
    name="CMB777", inchikey="KRMDCWKBEZIMAB-UHFFFAOYSA-N", smiles=_AMITRIPTYLINE
)
_DMSO = Compound(
    name="dimethyl sulfoxide",
    inchikey="IAZDPXIOMUYVGZ-UHFFFAOYSA-N",
    smiles="CS(=O)C",
    pubchem_cid=679,
    chebi_id="CHEBI:28262",
)
_HIP_DURATION_GAP = ProvenanceGap(
    field="duration_hours",
    reason=ProvenanceGapReason.not_reported_by_primary,
    note="HIP ran four sequential 16 h incubations to ~20 generations; the paper does not "
    "state which passage's plate was hybridized, so neither 16 nor 64 h can be asserted",
)
_SE_NOTE = (
    "no per-cell uncertainty is released: the replicate t-test p-value is folded into the "
    "adjusted score a_L = min(0.05/p, 1) * s_L"
)
_MEASUREMENT_UNITS = (
    "adjusted MADL sensitivity score = (r_L - med(r_L)) / MAD(r_L) over all pool strains, "
    "r_L = log ratio of treated vs control strain abundance (Hoepfner 2014); negative = "
    "hypersensitive, positive = resistant; 0 = growth equal to control. A companion "
    "gene-wise z-score column is deposited per experiment but not stored here."
)
_REFERENCE_UNITS = (
    "the screen's own no-drug DMSO control samples, the denominator of the MADL log "
    "ratio; a strain growing exactly as its controls scores 0"
)


def _uncertainty_gaps() -> list[ProvenanceGap]:
    return [
        ProvenanceGap(
            field=field,
            reason=ProvenanceGapReason.not_reported_by_primary,
            note=_SE_NOTE,
        )
        for field in (
            "environment_response_se",
            "environment_response_uncertainty",
            "environment_response_uncertainty_type",
        )
    ]


def _environment(assay: str, perturbation: SmallMoleculePerturbation) -> Environment:
    if assay == "HIP":
        return Environment(
            media=YPD_LIQUID,
            temperature=Temperature(value=30.0),
            perturbations=[perturbation],
            aerobicity="aerobic",
            duration_generations=20.0,
            provenance_gaps=[_HIP_DURATION_GAP],
        )
    return Environment(
        media=YPD_LIQUID,
        temperature=Temperature(value=30.0),
        perturbations=[perturbation],
        aerobicity="aerobic",
        duration_hours=16.0,
        duration_generations=5.0,
    )


def _treated(compound: Compound, micromolar: float) -> SmallMoleculePerturbation:
    return SmallMoleculePerturbation(
        compound=compound,
        concentration=Concentration(
            value=micromolar, unit=ConcentrationUnit.micromolar, basis=DoseBasis.IC30
        ),
        solvent=Solvent(name="DMSO", percent=2.0, compound=_DMSO),
    )


def _phenotype(
    score: float, n_samples: int, study: str
) -> EnvironmentResponsePhenotype:
    return EnvironmentResponsePhenotype(
        measurement_type=MeasurementType.sensitivity_score,
        assay_type=AssayType.pooled_competitive_growth_barcode,
        environment_response=score,
        n_samples=n_samples,
        sample_unit=SampleUnit.technical_replicate,
        units=_MEASUREMENT_UNITS,
        screen_id=study,
        provenance_gaps=_uncertainty_gaps(),
    )


def _hip_genotype(orf: str, perturbed: str) -> Genotype:
    return Genotype(
        perturbations=[
            EngineeredCopyNumberPerturbation(
                systematic_gene_name=orf,
                perturbed_gene_name=perturbed,
                copy_number=1,
                reference_copy_number=2,
                marker="KanMX",
            )
        ]
    )


def _reference(assay: str, study: str) -> dict[str, Any]:
    vehicle = SmallMoleculePerturbation(
        compound=_DMSO,
        concentration=Concentration(
            value=2.0, unit=ConcentrationUnit.percent_v_v, basis=DoseBasis.fixed
        ),
    )
    return EnvironmentResponseExperimentReference(
        dataset_name=_DATASET,
        genome_reference=ReferenceGenome(
            species="Saccharomyces cerevisiae", strain="BY4743", ploidy="diploid"
        ),
        environment_reference=_environment(assay, vehicle),
        phenotype_reference=EnvironmentResponsePhenotype(
            measurement_type=MeasurementType.sensitivity_score,
            assay_type=AssayType.pooled_competitive_growth_barcode,
            environment_response=0.0,
            n_samples=4,
            sample_unit=SampleUnit.technical_replicate,
            units=_REFERENCE_UNITS,
            screen_id=study,
            provenance_gaps=_uncertainty_gaps(),
        ),
    ).model_dump()


def _summary(
    dataset: m.EnvChemgenHoepfner2014Dataset,
) -> list[tuple[str, str, str, str, float, float, int, str]]:
    """(leaf, ORF, perturbed name, compound, dose, score, n, study) per record."""
    out = []
    for i in range(len(dataset)):
        experiment = dataset[i]["experiment"]
        (perturbation,) = experiment["genotype"]["perturbations"]
        (treatment,) = experiment["environment"]["perturbations"]
        phenotype = experiment["phenotype"]
        out.append(
            (
                perturbation["perturbation_type"],
                perturbation["systematic_gene_name"],
                perturbation["perturbed_gene_name"],
                treatment["compound"]["name"],
                treatment["concentration"]["value"],
                phenotype["environment_response"],
                phenotype["n_samples"],
                phenotype["screen_id"],
            )
        )
    return out


def test_thirteen_records_stream_hip_then_hop_in_row_then_column_order(
    dataset: m.EnvChemgenHoepfner2014Dataset,
) -> None:
    """Kept HIP columns are header indices 1, 3, 5, so each kept HIP row yields three
    records (fewer where a cell is empty or the line ends), then the HOP rows follow
    with their one kept column; the ``MADL`` column stores n = 1, the ``Ad.`` columns
    n = 2.
    """
    assert len(dataset) == 13
    hip, hop = "engineered_copy_number", "kanmx_deletion"
    assert _summary(dataset) == [
        (hip, "YAL001C", "YAL001C", "amitriptyline", 50.0, 0.1, 2, "0077"),
        (hip, "YAL001C", "YAL001C", "amitriptyline", 50.0, -1.5, 1, "0091"),
        (hip, "YAL001C", "YAL001C", "CMB777", 10.0, 2.25, 2, "0077"),
        (hip, "YAL034C-B", "YAL035C-A", "amitriptyline", 50.0, 0.2, 2, "0077"),
        (hip, "YAL034C-B", "YAL035C-A", "CMB777", 10.0, 0.4, 2, "0077"),
        (hip, "YBR271W", "YBR271W", "amitriptyline", 50.0, -3.0, 2, "0077"),
        (hip, "YBR271W", "YBR271W", "amitriptyline", 50.0, -2.0, 1, "0091"),
        (hip, "YBR271W", "YBR271W", "CMB777", 10.0, -0.5, 2, "0077"),
        (hip, "YLR074C", "YLR074C", "amitriptyline", 50.0, 0.6, 2, "0077"),
        (hop, "YAL001C", "YAL001C", "amitriptyline", 50.0, 1.0, 2, "0077"),
        (hop, "YBR271W", "YBR271W", "amitriptyline", 50.0, -0.25, 2, "0077"),
        (hop, "YAL034C-B", "YAL034C-B", "amitriptyline", 50.0, 0.33, 2, "0077"),
        (hop, "YAL001C", "YAL001C", "amitriptyline", 50.0, 0.99, 2, "0077"),
    ]


def test_hip_record_is_a_heterozygous_cnv_in_ypd_liquid_with_a_dmso_vehicle(
    dataset: m.EnvChemgenHoepfner2014Dataset,
) -> None:
    """Record 0 field by field: YAL001C copy 1 of 2 marked KanMX; YPD liquid, 30 C,
    aerobic, 20 generations with the ``duration_hours`` gap; amitriptyline 50 uM set at
    IC30 in 2 percent DMSO (the vehicle carries its own curated identity); score 0.1
    over 2 technical replicates in study 0077 with the three uncertainty gaps.
    """
    assert (
        dataset[0]["experiment"]
        == EnvironmentResponseExperiment(
            dataset_name=_DATASET,
            genotype=_hip_genotype("YAL001C", "YAL001C"),
            environment=_environment("HIP", _treated(_AMITRIPTYLINE_COMPOUND, 50.0)),
            phenotype=_phenotype(0.1, 2, "0077"),
        ).model_dump()
    )
    assert dataset[0]["reference"] == _reference("HIP", "0077")
    assert (
        dataset[0]["publication"]
        == Publication(doi=_DOI, doi_url=f"https://doi.org/{_DOI}").model_dump()
    )


def test_renamed_row_keeps_the_source_orf_and_a_structure_only_compound_is_cmb_named(
    dataset: m.EnvChemgenHoepfner2014Dataset,
) -> None:
    """Record 4: YAL035C-A is stored under YAL034C-B with the source ORF as
    ``perturbed_gene_name``; CMB 777 has no released name, so its compound is named
    ``CMB777`` with the InChIKey derived from its SMILES and no PubChem or ChEBI id.
    """
    assert (
        dataset[4]["experiment"]
        == EnvironmentResponseExperiment(
            dataset_name=_DATASET,
            genotype=_hip_genotype("YAL034C-B", "YAL035C-A"),
            environment=_environment("HIP", _treated(_CMB777_COMPOUND, 10.0)),
            phenotype=_phenotype(0.4, 2, "0077"),
        ).model_dump()
    )


def test_hop_record_is_a_kanmx_deletion_over_a_16_hour_exposure(
    dataset: m.EnvChemgenHoepfner2014Dataset,
) -> None:
    """Record 9: the HOP arm stores ``KanMxDeletionPerturbation`` and an environment
    with both durations (16 h, 5 generations) and no gap; its reference is the HOP
    control of study 0077.
    """
    assert (
        dataset[9]["experiment"]
        == EnvironmentResponseExperiment(
            dataset_name=_DATASET,
            genotype=Genotype(
                perturbations=[
                    KanMxDeletionPerturbation(
                        systematic_gene_name="YAL001C", perturbed_gene_name="YAL001C"
                    )
                ]
            ),
            environment=_environment("HOP", _treated(_AMITRIPTYLINE_COMPOUND, 50.0)),
            phenotype=_phenotype(1.0, 2, "0077"),
        ).model_dump()
    )
    assert dataset[9]["reference"] == _reference("HOP", "0077")


def test_reference_index_has_one_entry_per_assay_and_study(
    dataset: m.EnvChemgenHoepfner2014Dataset,
) -> None:
    index = json.loads(
        (
            Path(dataset.root) / "preprocess" / "experiment_reference_index.json"
        ).read_text()
    )
    assert [entry["member_indices"] for entry in index] == [
        [0, 2, 3, 4, 5, 7, 8],
        [1, 6],
        [9, 10, 11, 12],
    ]
    assert [entry["reference"] for entry in index] == [
        _reference("HIP", "0077"),
        _reference("HIP", "0091"),
        _reference("HOP", "0077"),
    ]


def test_drop_ledger_measures_the_compound_and_orf_rules(
    dataset: m.EnvChemgenHoepfner2014Dataset,
) -> None:
    """``<root>/dropped_records.json``: Boromycin's two columns cost 3 cells (HIP 2,
    HOP 1) with the RDKit parse failure as the unresolved reason; the pseudogene row
    loses 2 kept-column cells and the FASTA-absent CURRENT row 3; the one RENAMED row is
    listed per assay.
    """
    report = json.loads((Path(dataset.root) / "dropped_records.json").read_text())
    report.pop("created_at")
    assert report == {
        "dataset": _DATASET,
        "rule": (
            "drop every record whose compound carries no structure identifier after "
            "resolution: no curated compound_identity_table row with an InChIKey / ChEBI "
            "id / PubChem CID, and no RDKit-parseable released Table S1 SMILES to derive "
            "an InChIKey from"
        ),
        "orf_rule": (
            "a row is kept only when its 'Systematic Name' resolves through the shared "
            "SCerevisiaeGenome.resolve_gene_name to CURRENT (stored as is) or RENAMED "
            "(stored under the current systematic name, the source ORF kept as "
            "perturbed_gene_name so a merged-ORF strain stays distinct); a "
            "NON_GENE_FEATURE (pseudogene, blocked reading frame, transposable-element "
            "gene) or a RETIRED name drops the row and every one of its cells"
        ),
        "n_kept": 13,
        "n_dropped": 3,
        "kept_by_assay": {"HIP": 9, "HOP": 4},
        "dropped_by_assay": {"HIP": 2, "HOP": 1},
        "dropped_compounds": [
            {
                "cmb_id": "409",
                "source_name": "Boromycin",
                "smiles": _BOROMYCIN,
                "resolution_status": "UNRESOLVED_PUBLIC",
                "unresolved_reason": (
                    "the released Table S1 SMILES does not parse in RDKit 2026.03.6 "
                    "(boron chemistry), so no InChIKey can be derived from the primary "
                    "structure; PubChem carries a Boromycin record (CID 76962270) but "
                    "adopting it would substitute a different structure for the one the "
                    "screen released"
                ),
                "n_columns": 2,
                "n_records": 3,
            }
        ],
        "dropped_orfs": {
            "HIP": [
                {
                    "source_name": "YCL074W",
                    "status": "non_gene_feature",
                    "resolved_to": "YCL074W",
                    "feature_type": "pseudogene",
                    "n_records": 2,
                },
                {
                    "source_name": "YHR999W",
                    "status": "current",
                    "resolved_to": "YHR999W",
                    "feature_type": None,
                    "n_records": 3,
                },
            ],
            "HOP": [],
        },
        "renamed_orfs": {"HIP": {"YAL035C-A": "YAL034C-B"}, "HOP": {}},
    }


def test_table_s5_flag_file_counts_kept_hip_records_per_listed_strain(
    dataset: m.EnvChemgenHoepfner2014Dataset,
) -> None:
    """Two Table S5 strains appear in the HIP matrix: YBR271W (EFM2, cluster CL1,
    positional) with 3 kept records and YLR074C (BUD20, CL1+, not positional) with 1; the
    HOP rows of YBR271W are not flagged. The file pins the source CSV and Table S5 by
    sha256 and carries the paper's cluster paragraph verbatim.
    """
    flags = json.loads(
        (Path(dataset.root) / "table_s5_affected_strains.json").read_text()
    )
    flags.pop("created_at")
    assert flags == {
        "dataset": _DATASET,
        "citation_key": _CITATION_KEY,
        "source_csv": (
            "experiments/017-hoepfner-background-mutations/results/"
            "table_s5_affected_strains.csv"
        ),
        "source_csv_sha256": (
            "05bb74330f7118a8bc565fcbc587c2732daf5785db54b9a29a1aedbaca64bdc1"
        ),
        "table_s5_path": "si/Table_S5.xls",
        "table_s5_sha256": (
            "b123dc3e87fc10d3b4256f449fcd2eb38c91d1779000278af5a1a788356624a2"
        ),
        "paper_quote": (
            "For clusters 1 and 2, the hypersensitive phenotype did not track with any "
            "one mutation, but correlated with an increased sequencing coverage of "
            "chromosome XI suggestive of aneuploidy (Fig. S15). In contrast, Cluster 3 "
            "strains revealed a common point mutation in the WHI2/YOR043w gene resulting "
            "in a premature stop codon which truncates the ORF by $6 0 \\%$ and likely "
            "results in a non-functional protein (Fig. S16A). In support of this, the "
            "original WHI2/whi2 HIP strain significantly correlates with all identified "
            "cluster 3 strains (Fig. S16B). Finally, close analysis of Cluster 4 strain "
            "sequences revealed a discrete 12 kbps region on chromosome V where the "
            "relative coverage was increased by $5 0 \\%$ (Fig. S17). This region "
            "contains 6 annotated chromosomal features, including 3 genes with defined "
            "functions."
        ),
        "policy": (
            "KEPT and FLAGGED, never dropped: the measurement is real, and the paper "
            "reports that the hypersensitivity did not track with any one mutation, so no "
            "per-gene perturbation is invented for the background mutation. The flag is a "
            "property of the physical strain and no served class carries a slot for it, "
            "so it lives here and is joined on systematic_gene_name."
        ),
        "n_strains": 2,
        "n_positional_strains": 1,
        "n_records_flagged": 4,
        "n_positional_records_flagged": 3,
        "strains": [
            {
                "systematic_gene_name": "YBR271W",
                "common_gene_name": "EFM2",
                "cluster": "CL1",
                "is_positional": True,
                "mutation": "Chromosome XI aneuploidy",
                "construction_lab": "Lab 14",
                "n_records": 3,
            },
            {
                "systematic_gene_name": "YLR074C",
                "common_gene_name": "BUD20",
                "cluster": "CL1+",
                "is_positional": False,
                "mutation": "Chromosome XI aneuploidy",
                "construction_lab": "Lab 3",
                "n_records": 1,
            },
        ],
    }


def test_sourced_values_file_and_gene_set(
    dataset: m.EnvChemgenHoepfner2014Dataset,
) -> None:
    """``preprocess/sourced_values.json`` holds the 18 sourced constants in key order;
    ``temperature_c`` is checked literally and the whole file equals the JSON dump of
    ``SOURCED_VALUES``. The gene set is the four current ORFs the kept rows resolve to.
    """
    preprocess = Path(dataset.root) / "preprocess"
    payload = json.loads((preprocess / "sourced_values.json").read_text())
    assert list(payload) == [
        "assay_type",
        "concentration_unit",
        "dose_basis",
        "dose_basis_qualification",
        "hip_collection",
        "hip_duration_generations",
        "hip_marker",
        "hop_collection",
        "hop_duration_generations",
        "hop_duration_hours",
        "measurement_definition",
        "n_samples_reference",
        "n_samples_treated",
        "screen_id",
        "solvent_percent",
        "table_s5_affected_strains",
        "table_s5_mutations",
        "temperature_c",
    ]
    assert payload["temperature_c"] == {
        "value": 30.0,
        "provenance": {
            "source_uri": "paper.md",
            "citation_key": _CITATION_KEY,
            "sha256": "a9877549eff2fe1aaf8aa403d9fea1c381284de030326f4475e869c102af0aeb",
            "method": "MinerU OCR of the publisher PDF (mirrored artifact)",
            "page": None,
            "retrieved": None,
        },
        "quote": (
            "Plates were incubated for $1 6 \\mathrm { h }$ in a robotic shaking "
            "incubator at $3 0 ^ { \\circ } C / 5 5 0$ RPM allowing for ${ \\sim } 5$ "
            "doublings."
        ),
        "note": None,
    }
    assert payload == {
        key: value.model_dump(mode="json") for key, value in m.SOURCED_VALUES.items()
    }
    assert json.loads((preprocess / "gene_set.json").read_text()) == [
        "YAL001C",
        "YAL034C-B",
        "YBR271W",
        "YLR074C",
    ]
    manifest = json.loads((preprocess / "build_manifest.json").read_text())
    assert manifest["dataset_name"] == "env_chemgen_hoepfner2014"
    assert manifest["loader_class"] == _DATASET
    assert {
        "EnvironmentResponseExperiment",
        "EngineeredCopyNumberPerturbation",
        "KanMxDeletionPerturbation",
    } <= set(manifest["closure"])
    assert dataset.experiment_class is EnvironmentResponseExperiment
    assert dataset.reference_class is EnvironmentResponseExperimentReference
    assert dataset.raw_file_names == [
        "HIP_scores.txt",
        "HOP_scores.txt",
        "Table_S1.xls",
    ]
    with pytest.raises(NotImplementedError):
        dataset.create_experiment()
    frame = {"untouched": True}
    assert dataset.preprocess_raw(frame) is frame


def test_compound_meta_reads_names_and_structures_across_the_three_sheets(
    tmp_path: Path,
) -> None:
    """Names come from the two MoA sheets and SMILES from ``All Structures``; an id cell
    may be an int, a digit string or a comma list (``244, 1818`` gives both ids the same
    SMILES); an empty id cell contributes nothing (the ``Nameless`` row), a NaN name
    leaves CMB 1234 nameless, and a NaN SMILES leaves CMB 555 structureless.
    """
    _write_table_s1(tmp_path / "Table_S1.xls")
    assert m._load_compound_meta(str(tmp_path / "Table_S1.xls")) == {
        "3": {"common_name": "Amitriptyline", "smiles": _AMITRIPTYLINE},
        "409": {"common_name": "Boromycin", "smiles": _BOROMYCIN},
        "888": {"common_name": "Unreleased", "smiles": None},
        "1234": {"common_name": None, "smiles": None},
        "1235": {"common_name": "Novel compound", "smiles": None},
        "777": {"common_name": None, "smiles": _AMITRIPTYLINE},
        "244": {"common_name": None, "smiles": "CC(=O)O"},
        "1818": {"common_name": None, "smiles": "CC(=O)O"},
        "555": {"common_name": None, "smiles": None},
    }


def test_sgd_gene_universe_is_the_fasta_headers_first_token(tmp_path: Path) -> None:
    data_root = tmp_path / "data_root"
    _write_genomes_tier(data_root)
    assert m._load_sgd_genes(str(data_root)) == {
        "YAL001C",
        "YAL034C-B",
        "YBR271W",
        "YLR074C",
        "YCL074W",
        "YNCA0001W",
    }


def _digests(raw: Path) -> dict[str, dict[str, str]]:
    return {
        name: {**spec, "sha256": hashlib.sha256((raw / name).read_bytes()).hexdigest()}
        for name, spec in m._DRYAD_FILES.items()
    }


def test_download_links_the_mirror_only_when_its_bytes_match_the_pinned_sha256(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With a mirror at ``$DATA_ROOT/torchcell-raw/<key>/``, a mirror file whose digest
    is not the pinned Dryad digest raises ``RawSha256MismatchError`` with both digests
    and links nothing; with the pins repointed at the fixture's digests each absent raw
    file is symlinked, a raw file already present (``Table_S1.xls``, a plain copy) is
    left as it is, and the build runs to its thirteen records under the real build-time
    check.
    """
    monkeypatch.setattr(m, "verify_raw_files", verify_raw_files)
    data_root = tmp_path / "data_root"
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    _write_genomes_tier(data_root)
    mirror = data_root / "torchcell-raw" / _CITATION_KEY
    _write_raw(mirror)
    digest = hashlib.sha256((mirror / "HIP_scores.txt").read_bytes()).hexdigest()
    with pytest.raises(RawSha256MismatchError) as err:
        m.EnvChemgenHoepfner2014Dataset(root=str(tmp_path / "a"), genome=_StubGenome())
    assert str(err.value) == (
        f"sha256 mismatch for {mirror / 'HIP_scores.txt'}: expected "
        f"{m._DRYAD_FILES['HIP_scores.txt']['sha256']}, observed {digest}"
    )
    assert list((tmp_path / "a" / "raw").iterdir()) == []
    monkeypatch.setattr(m, "_DRYAD_FILES", _digests(mirror))
    raw = tmp_path / "b" / "raw"
    raw.mkdir(parents=True)
    (raw / "Table_S1.xls").write_bytes((mirror / "Table_S1.xls").read_bytes())
    dataset = m.EnvChemgenHoepfner2014Dataset(
        root=str(tmp_path / "b"), genome=_StubGenome()
    )
    for name in ("HIP_scores.txt", "HOP_scores.txt"):
        assert (raw / name).is_symlink() and os.readlink(raw / name) == str(
            mirror / name
        )
    assert not (raw / "Table_S1.xls").is_symlink()
    assert len(dataset) == 13


class _FakeResponse:
    """The two members of a streamed ``requests.Response`` that ``_fetch_from_dryad`` uses."""

    def __init__(self, body: bytes) -> None:
        self.body = body
        self.closed = False

    def iter_content(self, chunk_size: int) -> list[bytes]:
        return [
            self.body[i : i + chunk_size] for i in range(0, len(self.body), chunk_size)
        ]

    def close(self) -> None:
        self.closed = True


def test_download_fetches_a_file_the_mirror_lacks_and_verifies_the_pin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the mirror has no ``HOP_scores.txt``, ``download()`` runs the recorded Dryad
    retrieval: ``_dryad_get`` (stubbed here, no network) streams the body, which is
    written to a ``.partial`` sibling and hashed chunk by chunk; a body whose digest is
    not the pin raises ``RawSha256MismatchError`` naming the URL and both digests and
    leaves no HOP file in ``raw/``; the right body is renamed into place as a plain file
    beside the two symlinks.
    """
    data_root = tmp_path / "data_root"
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    _write_genomes_tier(data_root)
    mirror = data_root / "torchcell-raw" / _CITATION_KEY
    _write_raw(mirror)
    pins = _digests(mirror)
    monkeypatch.setattr(m, "_DRYAD_FILES", pins)
    hop_bytes = (mirror / "HOP_scores.txt").read_bytes()
    (mirror / "HOP_scores.txt").unlink()
    responses: list[_FakeResponse] = []
    calls: list[str] = []

    def fake_dryad_get(session: Any, url: str) -> _FakeResponse:
        calls.append(url)
        responses.append(_FakeResponse(b"wrong body" if len(calls) == 1 else hop_bytes))
        return responses[-1]

    monkeypatch.setattr(m, "_dryad_get", fake_dryad_get)
    wrong = hashlib.sha256(b"wrong body").hexdigest()
    with pytest.raises(RawSha256MismatchError) as err:
        m.EnvChemgenHoepfner2014Dataset(root=str(tmp_path / "a"), genome=_StubGenome())
    assert str(err.value) == (
        "sha256 mismatch for https://datadryad.org/downloads/file_stream/4834609: "
        f"expected {pins['HOP_scores.txt']['sha256']}, observed {wrong}"
    )
    assert sorted(p.name for p in (tmp_path / "a" / "raw").iterdir()) == [
        "HIP_scores.txt"
    ]
    dataset = m.EnvChemgenHoepfner2014Dataset(
        root=str(tmp_path / "b"), genome=_StubGenome()
    )
    assert calls == ["https://datadryad.org/downloads/file_stream/4834609"] * 2
    assert [r.closed for r in responses] == [True, True]
    hop = tmp_path / "b" / "raw" / "HOP_scores.txt"
    assert not hop.is_symlink() and hop.read_bytes() == hop_bytes
    assert (tmp_path / "b" / "raw" / "HIP_scores.txt").is_symlink()
    assert len(dataset) == 13


def test_deposit_raw_mirror_records_the_dryad_retrieval_per_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A source file whose digest is not the pin is refused with
    ``RawSha256MismatchError`` and both digests BEFORE anything is copied, so no mirror
    directory exists afterwards; with the pins repointed the manifest lists the three
    files with the ``_dryad_get`` retriever and the Dryad DOI as the SI source.
    """
    source = tmp_path / "source"
    _write_raw(source)
    data_root = tmp_path / "data_root"
    digest = hashlib.sha256((source / "HIP_scores.txt").read_bytes()).hexdigest()
    with pytest.raises(RawSha256MismatchError) as err:
        m.deposit_raw_mirror(
            source_dir=source, retrieved_at="2026-09-27", data_root=str(data_root)
        )
    assert str(err.value) == (
        f"sha256 mismatch for {source / 'HIP_scores.txt'}: expected "
        f"{m._DRYAD_FILES['HIP_scores.txt']['sha256']}, observed {digest}"
    )
    assert not data_root.exists()
    digests = _digests(source)
    monkeypatch.setattr(m, "_DRYAD_FILES", digests)
    root = m.deposit_raw_mirror(
        source_dir=source, retrieved_at="2026-09-27", data_root=str(data_root)
    )
    assert root == data_root / "torchcell-raw" / _CITATION_KEY
    manifest = m.load_manifest(str(data_root))
    assert manifest.citation_key == _CITATION_KEY
    assert manifest.doi == _DOI
    assert manifest.si_data_sources == ["https://doi.org/10.5061/dryad.v5m8v"]
    assert manifest.si_expected == ["HIP_scores.txt", "HOP_scores.txt", "Table_S1.xls"]
    assert [(f.path, f.role, f.sha256, f.source) for f in manifest.files] == [
        (name, "raw_data", spec["sha256"], spec["url"])
        for name, spec in digests.items()
    ]
    for record in manifest.files:
        assert record.retrieval is not None
        assert record.retrieval.method.value == "direct_url"
        assert (
            record.retrieval.retriever
            == "torchcell.datasets.scerevisiae.hoepfner2014._dryad_get"
        )
        assert record.retrieval.retrieved_at == "2026-09-27"
        assert record.bytes == (root / record.path).stat().st_size


def test_a_short_matrix_line_stops_at_its_last_cell(
    dataset: m.EnvChemgenHoepfner2014Dataset,
) -> None:
    """Record 11 is the HOP row ``YAL034C-B`` whose line carries only the first kept
    cell (0.33); the dropped-compound column lies beyond the line's end and adds nothing
    to the HOP drop count (1, from YAL001C's 0.7 alone).
    """
    (perturbation,) = dataset[11]["experiment"]["genotype"]["perturbations"]
    assert (
        perturbation
        == KanMxDeletionPerturbation(
            systematic_gene_name="YAL034C-B", perturbed_gene_name="YAL034C-B"
        ).model_dump()
    )
    assert dataset[11]["experiment"]["phenotype"]["environment_response"] == 0.33
    report = json.loads((Path(dataset.root) / "dropped_records.json").read_text())
    assert report["dropped_by_assay"]["HOP"] == 1


def test_solve_anubis_returns_the_smallest_nonce_meeting_the_difficulty() -> None:
    """The Dryad frontend's proof of work, solved offline: ``difficulty`` counts leading
    zero hex nibbles of sha256(random_data + nonce). For difficulty 2 (one zero byte)
    and 3 (a zero byte plus a zero high nibble) the returned digest is that hash, meets
    the bound, and no smaller nonce does.
    """
    for difficulty, nibbles in ((2, "00"), (3, "000")):
        digest, nonce = m._solve_anubis("challenge-seed", difficulty)
        assert digest == hashlib.sha256(f"challenge-seed{nonce}".encode()).hexdigest()
        assert digest.startswith(nibbles)
        assert not any(
            hashlib.sha256(f"challenge-seed{n}".encode())
            .hexdigest()
            .startswith(nibbles)
            for n in range(nonce)
        )


def test_a_repeated_orf_row_is_stored_twice_under_one_genotype(
    dataset: m.EnvChemgenHoepfner2014Dataset,
) -> None:
    """Finding: ``_iter_records`` (hoepfner2014.py lines 1266-1269) caches the genotype
    per source ORF and never checks that an ORF row is unique, so the second HOP
    ``YAL001C`` line becomes record 12 with the same genotype and reference as record 9
    and a second (strain, condition) measurement of 0.99; nothing in the ledger reports
    the repeat.
    """
    assert dataset[12]["experiment"]["genotype"] == dataset[9]["experiment"]["genotype"]
    assert dataset[12]["reference"] == dataset[9]["reference"]
    assert (
        dataset[12]["experiment"]["environment"]
        == dataset[9]["experiment"]["environment"]
    )
    assert dataset[12]["experiment"]["phenotype"]["environment_response"] == 0.99
    assert dataset[9]["experiment"]["phenotype"]["environment_response"] == 1.0


def test_a_raw_file_off_the_pin_is_refused_at_build_time(
    off_pin_raw: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Contract (issue #518's sweep): with the three Dryad files already in ``raw/`` PyG
    skips ``download()``, so ``process()`` verifies each against ``_DRYAD_FILES`` first.
    ``HIP_scores.txt`` off its pin raises ``RawSha256MismatchError`` naming it and both
    digests before a row is read; no store is written and the file is left as found.
    """
    monkeypatch.delenv("TC_DATA_URL", raising=False)
    staged = off_pin_raw(m, ["HIP_scores.txt", "HOP_scores.txt", "Table_S1.xls"])
    raw = staged.root / "raw" / "HIP_scores.txt"
    with pytest.raises(RawSha256MismatchError) as err:
        m.EnvChemgenHoepfner2014Dataset(root=str(staged.root), genome=_StubGenome())
    assert str(err.value) == (
        f"sha256 mismatch for {raw}: expected "
        "dbc5041defea9c046da0890d5e569f97d5f7afbf50ea0885f539ea8e5980cd24, "
        f"observed {staged.observed}"
    )
    assert list((staged.root / "processed").iterdir()) == []
    assert not (staged.root / "preprocess").exists()
    assert hashlib.sha256(raw.read_bytes()).hexdigest() == staged.observed
