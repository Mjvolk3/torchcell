# tests/torchcell/datasets/scerevisiae/test_nadal_ribelles2025_synthetic.py
# [[tests.torchcell.datasets.scerevisiae.test_nadal_ribelles2025_synthetic]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/scerevisiae/test_nadal_ribelles2025_synthetic.py
"""Nadal-Ribelles 2025 pseudobulk loader built end to end on synthetic ``.Rdata`` files.

``FC_genotype.Rdata`` (object ``fcs``) and ``ptb_summary.Rdata`` (object ``ptbs``) are
written with ``rdata.write_rda``, the writer of the same pure-Python ``rdata`` package
the loader reads with, as named lists of data frames; ``README.txt`` is plain text. All
three sit in ``raw/``, so ``download()`` is not called. The files are not checked
against R itself (no R in the env); the contract tested is the loader's read of what
``rdata`` round-trips. No genome is consulted (#1045): every gene name is a key AS
RELEASED and the deleted gene is the ORF of the genotype label.

``fcs`` tables (key: names -> logfoldchanges) and what they become:

    DEG_Control_bc_YAL012W.csv    MET14 1.0, mup1 -0.5, YKL001C 9.0, 15S_RRNA 3.0,
                                  STR1 0.25: five keys, verbatim (record 0)
    DEG_NaCl_bc_YAL012W.csv       YPL999W 2.0, OLDSYM -1.0 (record 1)
    DEG_Control_bc_YBR020W-1.csv  GAL1 0.5 (replacement strain: ORF YBR020W; record 2)
    DEG_Control_bc_YMR175w-1.csv  YMR175W -3.0 (lower-case label, not in ptbs; record 3)
    DEG_Control_bc_NOTANORF.csv   genotype label not ORF-shaped: stored on the
                                  released-gene-name leaf, counted
    DEG_Control_bc_YGR055W.csv    15S_RRNA 1.0 (record 4; not in ptbs)

``ptbs``: control rows bc-YAL012W (120.4 cells, sd 1.25), WT (500, 1.0), bc-YBR020W-1
(33.6 cells, sd 0.75; one row per label, as in the release), NaCl rows bc-YAL012W (80, 2.0) and
WT (400, 1.1). ``n_cells`` = round(cell_number): 120, 34. References: control logFC 0
over the seven released control names with WT 1.0 / 500; NaCl logFC 0 over YPL999W and
OLDSYM with WT 1.1 / 400.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import rdata

from torchcell.data import RawSha256MismatchError
from torchcell.data.experiment_dataset import verify_raw_files
from torchcell.datamodels.schema import (
    Compound,
    Concentration,
    ConcentrationUnit,
    Environment,
    Genotype,
    MarkerDeletionPerturbation,
    PseudobulkExpressionExperiment,
    PseudobulkExpressionExperimentReference,
    PseudobulkExpressionPhenotype,
    Publication,
    ReferenceGenome,
    ReleasedIdentifierPerturbation,
    ReleasedKind,
    SmallMoleculePerturbation,
    Temperature,
)
from torchcell.datasets.scerevisiae import nadal_ribelles2025 as m
from torchcell.sequence.genome.registry import (
    SGD_S288C_R64,
    SGD_S288C_R64_GENBANK_ACCESSION,
)

_DATASET = "NadalRibellesPerturbSeq2025Dataset"


def _deg(names: list[str], lfcs: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"names": names, "logfoldchanges": lfcs})


_FCS = {
    "DEG_Control_bc_YAL012W.csv": _deg(
        ["MET14", "mup1", "YKL001C", "15S_RRNA", "STR1"], [1.0, -0.5, 9.0, 3.0, 0.25]
    ),
    "DEG_NaCl_bc_YAL012W.csv": _deg(["YPL999W", "OLDSYM"], [2.0, -1.0]),
    "DEG_Control_bc_YBR020W-1.csv": _deg(["GAL1"], [0.5]),
    "DEG_Control_bc_YMR175w-1.csv": _deg(["YMR175W"], [-3.0]),
    "DEG_Control_bc_NOTANORF.csv": _deg(["MET14"], [1.0]),
    "DEG_Control_bc_YGR055W.csv": _deg(["15S_RRNA"], [1.0]),
}
_PTBS = {
    "control": pd.DataFrame(
        {
            "assignment_consensus2": ["bc-YAL012W", "WT", "bc-YBR020W-1"],
            "cell_number": [120.4, 500.0, 33.6],
            "sd_lvscore_scaledFU2": [1.25, 1.0, 0.75],
        }
    ),
    "NaCl": pd.DataFrame(
        {
            "assignment_consensus2": ["bc-YAL012W", "WT"],
            "cell_number": [80.0, 400.0],
            "sd_lvscore_scaledFU2": [2.0, 1.1],
        }
    ),
}
_README = b"Zenodo 10.5281/zenodo.14062629 synthetic README\n"


def _write_files(directory: Path) -> dict[str, bytes]:
    directory.mkdir(parents=True)
    rdata.write_rda(str(directory / m.FC_NAME), {"fcs": _FCS})
    rdata.write_rda(str(directory / m.PTB_NAME), {"ptbs": _PTBS})
    (directory / m.README_NAME).write_bytes(_README)
    return {name: (directory / name).read_bytes() for name in m.RAW_FILES}


@pytest.fixture
def dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> m.NadalRibellesPerturbSeq2025Dataset:
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "data_root"))
    _write_files(tmp_path / "nadal" / "raw")
    return m.NadalRibellesPerturbSeq2025Dataset(root=str(tmp_path / "nadal"))


_YPD = m.NADAL_RIBELLES_YPD
_CONTROL = Environment(
    media=_YPD, temperature=Temperature(value=30.0), aerobicity="aerobic"
)
_NACL = Environment(
    media=_YPD,
    temperature=Temperature(value=30.0),
    perturbations=[
        SmallMoleculePerturbation(
            compound=Compound(
                name="sodium chloride",
                inchikey="FAPWRFPIFSIZLT-UHFFFAOYSA-M",
                smiles="[Na+].[Cl-]",
                pubchem_cid=5234,
                chebi_id="CHEBI:26710",
            ),
            concentration=Concentration(value=0.4, unit=ConcentrationUnit.molar),
        )
    ],
    aerobicity="aerobic",
    duration_hours=0.25,
)


def _experiment(
    orf: str,
    name: str,
    strain: str,
    environment: Environment,
    lfc: dict[str, float],
    dispersion: float | None,
    n_cells: int | None,
) -> dict[str, Any]:
    perturbation: MarkerDeletionPerturbation | ReleasedIdentifierPerturbation
    if orf.startswith("Y"):
        perturbation = MarkerDeletionPerturbation(
            systematic_gene_name=orf,
            perturbed_gene_name=name,
            marker="URA3",
            strain_id=strain,
        )
    else:
        perturbation = ReleasedIdentifierPerturbation(
            systematic_gene_name=orf,
            gene_namespace="scerevisiae_s288c_sgd_systematic",
            perturbed_gene_name=name,
            identifier_kind="gene identifier, as the release writes it",
            released_kind=ReleasedKind.deletion,
            cassette="URA3",
            strain_id=strain,
        )
    return PseudobulkExpressionExperiment(
        dataset_name=_DATASET,
        genotype=Genotype(perturbations=[perturbation]),
        environment=environment,
        phenotype=PseudobulkExpressionPhenotype(
            expression_log2_ratio=lfc,
            dispersion=dispersion,
            n_cells=n_cells,
            measurement_type="pseudobulk_scrnaseq_log2fc",
        ),
    ).model_dump()


def _reference(
    environment: Environment, genes: list[str], dispersion: float, n_cells: int
) -> dict[str, Any]:
    return PseudobulkExpressionExperimentReference(
        dataset_name=_DATASET,
        genome_reference=ReferenceGenome(
            species="Saccharomyces cerevisiae",
            strain="S288C",
            assembly_set=SGD_S288C_R64,
            assembly_accession=SGD_S288C_R64_GENBANK_ACCESSION,
        ),
        environment_reference=environment,
        phenotype_reference=PseudobulkExpressionPhenotype(
            expression_log2_ratio=dict.fromkeys(genes, 0.0),
            dispersion=dispersion,
            n_cells=n_cells,
            measurement_type="pseudobulk_scrnaseq_log2fc",
        ),
    ).model_dump()


_REF_CONTROL = _reference(
    _CONTROL,
    ["MET14", "mup1", "YKL001C", "15S_RRNA", "STR1", "GAL1", "YMR175W"],
    1.0,
    500,
)
_REF_NACL = _reference(_NACL, ["YPL999W", "OLDSYM"], 1.1, 400)


def test_five_records_keep_released_names_and_carry_single_cell_scalars(
    dataset: m.NadalRibellesPerturbSeq2025Dataset,
) -> None:
    """Records follow the ``fcs`` key order; the one label that is not ORF-shaped
    (``NOTANORF``) is record 4 on the released-gene-name leaf (#1045). Every gene name
    is a key as released.
    """
    assert len(dataset) == 6
    expected = [
        (
            _experiment(
                "YAL012W",
                "YAL012W",
                "bc_YAL012W",
                _CONTROL,
                {
                    "MET14": 1.0,
                    "mup1": -0.5,
                    "YKL001C": 9.0,
                    "15S_RRNA": 3.0,
                    "STR1": 0.25,
                },
                1.25,
                120,
            ),
            _REF_CONTROL,
        ),
        (
            _experiment(
                "YAL012W",
                "YAL012W",
                "bc_YAL012W",
                _NACL,
                {"YPL999W": 2.0, "OLDSYM": -1.0},
                2.0,
                80,
            ),
            _REF_NACL,
        ),
        (
            _experiment(
                "YBR020W", "YBR020W", "bc_YBR020W-1", _CONTROL, {"GAL1": 0.5}, 0.75, 34
            ),
            _REF_CONTROL,
        ),
        (
            _experiment(
                "YMR175W",
                "YMR175W",
                "bc_YMR175w-1",
                _CONTROL,
                {"YMR175W": -3.0},
                None,
                None,
            ),
            _REF_CONTROL,
        ),
        (
            _experiment(
                "NOTANORF",
                "NOTANORF",
                "bc_NOTANORF",
                _CONTROL,
                {"MET14": 1.0},
                None,
                None,
            ),
            _REF_CONTROL,
        ),
        (
            _experiment(
                "YGR055W",
                "YGR055W",
                "bc_YGR055W",
                _CONTROL,
                {"15S_RRNA": 1.0},
                None,
                None,
            ),
            _REF_CONTROL,
        ),
    ]
    for i, (experiment, reference) in enumerate(expected):
        assert dataset[i]["experiment"] == experiment
        assert dataset[i]["reference"] == reference
    assert (
        dataset[0]["publication"]
        == Publication(
            doi="10.1038/s41467-025-57600-4",
            doi_url="https://doi.org/10.1038/s41467-025-57600-4",
        ).model_dump()
    )
    log = json.loads(
        (Path(dataset.root) / "preprocess" / "dropped_records.json").read_text()
    )
    assert (log["source_records"], log["kept_records"], log["dropped_records"]) == (
        6,
        6,
        0,
    )
    assert log["rules"][0]["rule"] == "stored_on_released_gene_name_leaf"
    assert log["rules"][0]["n_records"] == 0
    assert log["rules"][0]["items"] == ["NOTANORF"]


def test_the_deleted_gene_is_the_label_orf_as_both_names(
    dataset: m.NadalRibellesPerturbSeq2025Dataset,
) -> None:
    """The release names the deleted gene only by its ORF label, so the ORF is both
    ``systematic_gene_name`` and ``perturbed_gene_name``; no standard name is looked up
    (the previous build stored the string ``"nan"`` for 940 of 6188 records whose ORF
    had no standard name in the genome table).
    """
    (perturbation,) = dataset[3]["experiment"]["genotype"]["perturbations"]
    assert perturbation["systematic_gene_name"] == "YMR175W"
    assert perturbation["perturbed_gene_name"] == "YMR175W"


def test_side_files_gene_set_reference_index_and_strain_table(
    dataset: m.NadalRibellesPerturbSeq2025Dataset,
) -> None:
    preprocess = Path(dataset.root) / "preprocess"
    assert json.loads((preprocess / "gene_set.json").read_text()) == [
        "NOTANORF",
        "YAL012W",
        "YBR020W",
        "YGR055W",
        "YMR175W",
    ]
    index = json.loads((preprocess / "experiment_reference_index.json").read_text())
    assert [entry["member_indices"] for entry in index] == [[0, 2, 3, 4, 5], [1]]
    assert [entry["reference"] for entry in index] == [_REF_CONTROL, _REF_NACL]
    samples = pd.read_csv(preprocess / "data.csv")
    assert samples.to_dict(orient="list") == {
        "strain_id": [
            "bc_YAL012W",
            "bc_YAL012W",
            "bc_YBR020W-1",
            "bc_YMR175w-1",
            "bc_NOTANORF",
            "bc_YGR055W",
        ],
        "condition": ["control", "nacl", "control", "control", "control", "control"],
    }
    manifest = json.loads((preprocess / "build_manifest.json").read_text())
    assert (manifest["dataset_name"], manifest["loader_class"]) == ("nadal", _DATASET)
    assert dataset.raw_file_names == [
        "FC_genotype.Rdata",
        "ptb_summary.Rdata",
        "README.txt",
    ]
    assert dataset.experiment_class is PseudobulkExpressionExperiment
    assert dataset.reference_class is PseudobulkExpressionExperimentReference
    frame = pd.DataFrame({"a": [1]})
    assert dataset.preprocess_raw(frame) is frame


def test_ptb_table_normalizes_the_hyphenated_genotype_label(
    dataset: m.NadalRibellesPerturbSeq2025Dataset,
) -> None:
    ptbs = dataset._load_ptbs("unused")
    assert sorted(ptbs) == ["NaCl", "control"]
    assert ptbs["control"].index.tolist() == ["bc_YAL012W", "WT", "bc_YBR020W-1"]
    assert m.NadalRibellesPerturbSeq2025Dataset._ptb_scalars(None, "WT") == (None, None)


def test_label_parsers() -> None:
    """A key without ``DEG_`` raises; ``-A`` suffixes are ORF names, ``-<digits>`` is a
    replacement strain, and anything else is not an ORF.
    """
    assert m._parse_fc_key("DEG_NaCl_bc_YBR020W-1.csv") == ("NaCl", "YBR020W-1")
    assert m._parse_fc_key("DEG_Control_bc_WT") == ("Control", "WT")
    with pytest.raises(
        ValueError,
        match="unexpected fcs key \\(no DEG_ prefix\\): 'X_Control_bc_A.csv'",
    ):
        m._parse_fc_key("X_Control_bc_A.csv")
    assert [
        m._deletion_systematic(label)
        for label in ("YAL037C-A", "ymr175w-1", "Q0010", "YNCA0001W", "ABC-1", "WT")
    ] == ["YAL037C-A", "YMR175W", "Q0010", "YNCA0001W", None, None]


def test_a_gene_name_listed_twice_in_one_table_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One released name on two rows of a DEG table is refused by name, never
    collapsed to the first value.
    """
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "data_root"))
    raw = tmp_path / "nadal" / "raw"
    raw.mkdir(parents=True)
    fcs = {"DEG_Control_bc_YAL012W.csv": _deg(["MET14", "MET14"], [1.0, 2.0])}
    rdata.write_rda(str(raw / m.FC_NAME), {"fcs": fcs})
    rdata.write_rda(str(raw / m.PTB_NAME), {"ptbs": _PTBS})
    (raw / m.README_NAME).write_bytes(_README)
    with pytest.raises(m.RepeatedGeneNameError, match="lists gene name 'MET14' twice"):
        m.NadalRibellesPerturbSeq2025Dataset(root=str(tmp_path / "nadal"))


def test_download_verifies_every_mirror_file_then_links_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Against the real pins the synthetic ``FC_genotype.Rdata`` is refused with
    ``RawSha256MismatchError`` and both digests, nothing linked; with the pins repointed,
    each file is symlinked into ``raw/`` and the build, under the real build-time check,
    yields the four records. A missing mirror file is refused by path.
    """
    monkeypatch.setattr(m, "verify_raw_files", verify_raw_files)
    data_root = tmp_path / "data_root"
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    mirror = data_root / m.RAW_DIR_REL
    files = _write_files(mirror)
    got = hashlib.sha256(files[m.FC_NAME]).hexdigest()
    with pytest.raises(RawSha256MismatchError) as err:
        m.NadalRibellesPerturbSeq2025Dataset(root=str(tmp_path / "a"))
    assert str(err.value) == (
        f"sha256 mismatch for {mirror / m.FC_NAME}: expected {m.FC_SHA256}, "
        f"observed {got}"
    )
    assert list((tmp_path / "a" / "raw").iterdir()) == []
    for name, data in files.items():
        monkeypatch.setitem(m.SHA256_EXPECTED, name, hashlib.sha256(data).hexdigest())
    dataset = m.NadalRibellesPerturbSeq2025Dataset(root=str(tmp_path / "b"))
    assert {
        name: os.readlink(tmp_path / "b" / "raw" / name) for name in m.RAW_FILES
    } == {name: str(mirror / name) for name in m.RAW_FILES}
    assert len(dataset) == 6
    (mirror / m.README_NAME).unlink()
    with pytest.raises(
        RuntimeError,
        match=f"required raw artifact missing from mirror: {mirror / m.README_NAME}",
    ):
        m.NadalRibellesPerturbSeq2025Dataset(root=str(tmp_path / "c"))
