# tests/torchcell/datasets/scerevisiae/test_smith2016_synthetic.py
# [[tests.torchcell.datasets.scerevisiae.test_smith2016_synthetic]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/scerevisiae/test_smith2016_synthetic.py
"""Hermetic end-to-end build of ``CrispriChemgenSmith2016Dataset`` from two hand-written xlsx.

Additional file 10 (sheet ``Fitness and Effect Data``: ``#Pool, Guide, ORF, Drug,
Concentration, A, var(A)``) and Additional file 4 (sheet ``gRNAs``: ``Guide_name,
Specificity_sequence``) are written with openpyxl under ``<root>/raw/`` so PyG never calls
``download()``; ``process()`` runs once against a duck-typed genome stub carrying
``gene_set``, ``feature_index["standard_to_ids"]`` and ``resolve_gene_name`` (everything
the loader and ``canonical_common_names`` read). The stub is the authority for every
mapping: YHR007C (ERG11) and YJR066W (TOR1) are the current ORFs.

Effect rows, in file order, and their fate:

=====  ================  ========  =========  ===========  =============  =====  ======
row    #Pool             Guide     ORF        Drug         Concentration  A      var(A)
=====  ================  ========  =========  ===========  =============  =====  ======
0      broad_tiling      ERG11_g1  YHR007C    Fluconazole  20 uM          -1.5   0.04
1      gene_tiling_20bp  ERG11_g1  YHR007C    Fluconazole  20 uM          -0.7   0.02
2      broad_tiling      TOR1_g5   YJR066W    DMSO         0.01           0.3    0.01
3      broad_tiling      ERG11_g1  YHR007C    1181-0519    5 uM           -2.0   0.1
4      broad_tiling      ERG11_g2  YHR007C    Fluconazole  20 uM          (blank) 0.03
5      broad_tiling      ERG11_g2  YHR007C    Fluconazole  20 uM          0.1    n/a
6      broad_tiling      GHOST_g1  YZZ999W    Fluconazole  20 uM          0.1    0.01
7      broad_tiling      ERG11_g9  YHR007C    Rapamycin    1 nM           0.5    0.05
8      broad_tiling      ERG11_g3  " erg11 "  Rapamycin    1 nM           0.25   0.0025
=====  ================  ========  =========  ===========  =============  =====  ======

Rows 0, 1, 2 and 8 are kept as records 0-3 (row 8 resolves the padded lowercase common
name through the stub's renamed branch). Row 3 drops on the vendor-code rule (1181-0519
is PROPRIETARY in the pinned identity table); rows 4 and 5 drop on the no-A/var(A) rule
(``to_numeric(errors="coerce")`` turns the blank and ``n/a`` into NaN); row 6 drops as
an unresolved target; row 7 drops because ERG11_g9 has no spacer in Additional file 4.
So 9 source rows, 4 kept, 5 dropped = 1 + 2 + 1 + 1.

Every record stores ``n_samples = 1`` and the released ``var(A)`` verbatim as
``UncertaintyType.variance``, so the derived SE is ``sqrt(var / 1)``: 0.2, 0.1414...,
0.1 and 0.05. The DMSO control stores the sourced 1.0 percent v/v dose, not the released
cell 0.01. Fluconazole 20 uM in two pools is one environment (the env cache keys on
``(drug, str(concentration))``), so records 0 and 1 share one reference.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import socket
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import lmdb
import openpyxl
import pytest

from torchcell.data import RawSha256MismatchError
from torchcell.datamodels.compound_identity import resolved_compound
from torchcell.datamodels.media import SC_URA
from torchcell.datamodels.schema import (
    AssayType,
    Concentration,
    ConcentrationUnit,
    CrisprConstruct,
    CrisprInterferencePerturbation,
    Environment,
    EnvironmentResponseExperiment,
    EnvironmentResponseExperimentReference,
    EnvironmentResponsePhenotype,
    Genotype,
    MeasurementType,
    Publication,
    ReferenceGenome,
    SampleUnit,
    SmallMoleculePerturbation,
    SourceType,
    Temperature,
    UncertaintyType,
)
from torchcell.datasets.scerevisiae import smith2016 as s
from torchcell.literature.manifest import (
    ROLE_SI_DATA,
    ArtifactRecord,
    Manifest,
    RetrievalMethod,
    RetrievalRecord,
)
from torchcell.sequence.genome.scerevisiae import SCerevisiaeGenome

_GENES = {"YHR007C", "YJR066W"}
_STANDARD = {"ERG11": ["YHR007C"], "TOR1": ["YJR066W"]}


class _Resolution:
    """The two attributes and the one property the loader reads from a resolution."""

    def __init__(self, status: str, systematic: str | None) -> None:
        self.status = status
        self.systematic_name = systematic

    @property
    def is_current_gene(self) -> bool:
        return self.status in ("current", "renamed")


class _StubGenome:
    """``gene_set``, ``feature_index`` and the resolver: all the genome the loader reads."""

    gene_set = _GENES
    feature_index = {"standard_to_ids": _STANDARD}

    def resolve_gene_name(self, name: str) -> _Resolution:
        upper = name.strip().upper()
        if upper in _GENES:
            return _Resolution("current", upper)
        if upper in _STANDARD:
            return _Resolution("renamed", _STANDARD[upper][0])
        return _Resolution("retired", upper)


def _genome() -> SCerevisiaeGenome:
    return cast(SCerevisiaeGenome, _StubGenome())


_EFFECT_HEADER = ["#Pool", "Guide", "ORF", "Drug", "Concentration", "A", "var(A)"]
_EFFECT_ROWS: list[list[Any]] = [
    ["broad_tiling", "ERG11_g1", "YHR007C", "Fluconazole", "20 uM", -1.5, 0.04],
    ["gene_tiling_20bp", "ERG11_g1", "YHR007C", "Fluconazole", "20 uM", -0.7, 0.02],
    ["broad_tiling", "TOR1_g5", "YJR066W", "DMSO", 0.01, 0.3, 0.01],
    ["broad_tiling", "ERG11_g1", "YHR007C", "1181-0519", "5 uM", -2.0, 0.1],
    ["broad_tiling", "ERG11_g2", "YHR007C", "Fluconazole", "20 uM", None, 0.03],
    ["broad_tiling", "ERG11_g2", "YHR007C", "Fluconazole", "20 uM", 0.1, "n/a"],
    ["broad_tiling", "GHOST_g1", "YZZ999W", "Fluconazole", "20 uM", 0.1, 0.01],
    ["broad_tiling", "ERG11_g9", "YHR007C", "Rapamycin", "1 nM", 0.5, 0.05],
    ["broad_tiling", "ERG11_g3", " erg11 ", "Rapamycin", "1 nM", 0.25, 0.0025],
]
_SPACERS = {
    "ERG11_g1": "ACGTACGTACGTACGTACGT",
    "ERG11_g2": "TTGCATTGCATTGCATTG",
    "ERG11_g3": "GGCCGGCCGGCCGGCCGGCC",
    "TOR1_g5": "AATTAATTAATTAATTAATT",
    "GHOST_g1": "CCCCAAAATTTTGGGGCCCC",
}


def _write_workbooks(raw: Path, effect_rows: list[list[Any]] = _EFFECT_ROWS) -> None:
    """Write the two SI workbooks with the header on row 0 of the named sheets."""
    raw.mkdir(parents=True, exist_ok=True)
    effect = openpyxl.Workbook()
    sheet = effect.active
    sheet.title = s.EFFECT_SHEET
    sheet.append(_EFFECT_HEADER)
    for row in effect_rows:
        sheet.append(row)
    effect.save(raw / s.EFFECT_FILENAME)
    guides = openpyxl.Workbook()
    sheet = guides.active
    sheet.title = s.GUIDE_SHEET
    sheet.append(["Guide_name", "Specificity_sequence"])
    for name, spacer in _SPACERS.items():
        sheet.append([name, spacer])
    guides.save(raw / s.GUIDE_FILENAME)


def _root(tmp_path: Path, slug: str = "crispri_chemgen_smith2016") -> Path:
    root = tmp_path / slug
    _write_workbooks(root / "raw")
    return root


@pytest.fixture
def dataset(tmp_path: Path) -> s.CrispriChemgenSmith2016Dataset:
    return s.CrispriChemgenSmith2016Dataset(root=str(_root(tmp_path)), genome=_genome())


_PUBLICATION = Publication(doi=s.DOI, doi_url=f"https://doi.org/{s.DOI}")


def _environment(drug: str, value: float, unit: ConcentrationUnit) -> Environment:
    return Environment(
        media=SC_URA,
        temperature=Temperature(value=30.0),
        perturbations=[
            SmallMoleculePerturbation(
                compound=resolved_compound(drug),
                concentration=Concentration(value=value, unit=unit),
            )
        ],
        aerobicity="aerobic",
        duration_generations=20.0,
        provenance_gaps=[s.DURATION_HOURS_GAP],
    )


_FLUCONAZOLE = _environment("Fluconazole", 20.0, ConcentrationUnit.micromolar)
_DMSO = _environment("DMSO", 1.0, ConcentrationUnit.percent_v_v)
_RAPAMYCIN = _environment("Rapamycin", 1.0, ConcentrationUnit.nanomolar)


def _experiment(
    orf: str, gene: str, spacer: str, pool: str, env: Environment, a: float, var: float
) -> EnvironmentResponseExperiment:
    return EnvironmentResponseExperiment(
        dataset_name="CrispriChemgenSmith2016Dataset",
        genotype=Genotype(
            perturbations=[
                CrisprInterferencePerturbation(
                    systematic_gene_name=orf,
                    perturbed_gene_name=gene,
                    crispr=CrisprConstruct(
                        effector="dCas9-Mxi1",
                        guide_sequence=spacer,
                        n_guides=1,
                        library_pool=pool,
                    ),
                )
            ]
        ),
        environment=env,
        phenotype=EnvironmentResponsePhenotype(
            measurement_type=MeasurementType.log2_ratio,
            assay_type=AssayType.pooled_competitive_growth_barcode,
            environment_response=a,
            environment_response_uncertainty=var,
            environment_response_uncertainty_type=UncertaintyType.variance,
            n_samples=1,
            sample_unit=SampleUnit.pooled,
            units=s.UNITS,
        ),
    )


def _reference(env: Environment) -> EnvironmentResponseExperimentReference:
    return EnvironmentResponseExperimentReference(
        dataset_name="CrispriChemgenSmith2016Dataset",
        genome_reference=ReferenceGenome(
            species="Saccharomyces cerevisiae", strain="BY4741"
        ),
        environment_reference=env,
        phenotype_reference=EnvironmentResponsePhenotype(
            measurement_type=MeasurementType.log2_ratio,
            assay_type=AssayType.pooled_competitive_growth_barcode,
            environment_response=0.0,
            units=s.UNITS,
        ),
    )


_EXPECTED = [
    _experiment(
        "YHR007C",
        "ERG11",
        _SPACERS["ERG11_g1"],
        "broad_tiling",
        _FLUCONAZOLE,
        -1.5,
        0.04,
    ),
    _experiment(
        "YHR007C",
        "ERG11",
        _SPACERS["ERG11_g1"],
        "gene_tiling_20bp",
        _FLUCONAZOLE,
        -0.7,
        0.02,
    ),
    _experiment(
        "YJR066W", "TOR1", _SPACERS["TOR1_g5"], "broad_tiling", _DMSO, 0.3, 0.01
    ),
    _experiment(
        "YHR007C",
        "ERG11",
        _SPACERS["ERG11_g3"],
        "broad_tiling",
        _RAPAMYCIN,
        0.25,
        0.0025,
    ),
]


def _interned_entries(root: Path) -> int:
    env = lmdb.open(str(root / "processed" / "interned"), readonly=True, lock=False)
    with env.begin() as txn:
        entries: int = txn.stat()["entries"]
    env.close()
    return entries


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_build_keeps_four_records_in_row_order_with_exact_records(
    dataset: s.CrispriChemgenSmith2016Dataset,
) -> None:
    """Records 0-3 are rows 0, 1, 2 and 8 of the effect table, each equal to a hand-built
    ``EnvironmentResponseExperiment``: the genome's standard name as the perturbed name
    (row 8's `` erg11 `` is stripped and resolved through the renamed branch to
    YHR007C/ERG11), the joined spacer, the row's pool, the drug environment, and the SE
    derived from ``var(A)`` with n = 1: sqrt(0.04) = 0.2, sqrt(0.02) = 0.141421...,
    sqrt(0.01) = 0.1, sqrt(0.0025) = 0.05. The DMSO record carries the canonical compound
    name ``dimethyl sulfoxide`` at the sourced 1.0 percent v/v, not the released 0.01.
    """
    assert len(dataset) == 4
    assert [dataset[i]["experiment"] for i in range(4)] == [
        e.model_dump() for e in _EXPECTED
    ]
    assert [
        dataset[i]["experiment"]["phenotype"]["environment_response_se"]
        for i in range(4)
    ] == [0.2, math.sqrt(0.02), 0.1, 0.05]
    dmso = dataset[2]["experiment"]["environment"]["perturbations"]
    assert [(p["compound"]["name"], p["concentration"]) for p in dmso] == [
        (
            "dimethyl sulfoxide",
            {"value": 1.0, "unit": ConcentrationUnit.percent_v_v, "basis": None},
        )
    ]
    assert dataset[0]["publication"] == _PUBLICATION.model_dump()
    assert dataset[0]["publication"] == {
        "pubmed_id": None,
        "pubmed_url": None,
        "doi": "10.1186/s13059-016-0900-9",
        "doi_url": "https://doi.org/10.1186/s13059-016-0900-9",
        # a journal article: the non-journal identity fields stay unset (1.7.0)
        "source_type": SourceType.journal_article,
        "title": None,
        "identifier": None,
        "identifier_url": None,
    }


def test_one_reference_per_drug_environment_and_the_reference_carries_no_n(
    dataset: s.CrispriChemgenSmith2016Dataset,
) -> None:
    """Finding: the uninduced (-ATc) reference phenotype (smith2016.py 547-552) carries
    ``environment_response = 0.0`` with ``n_samples``, ``sample_unit`` and the uncertainty
    fields all None, while every record stores ``n_samples = 1`` and ``sample_unit =
    pooled`` (636-645); the reference is built per environment, not per record, so the
    replicate design is asserted on records only.

    The same spacer in two pools (records 0 and 1) shares the Fluconazole 20 uM
    reference, so ``experiment_reference_index.json`` has three entries with member
    indices [0, 1], [2], [3], each reference equal to the hand-built one.
    """
    index = json.loads(
        (
            Path(dataset.root) / "preprocess" / "experiment_reference_index.json"
        ).read_text()
    )
    assert [entry["member_indices"] for entry in index] == [[0, 1], [2], [3]]
    assert [dataset[i]["reference"] for i in range(4)] == [
        _reference(env).model_dump()
        for env in (_FLUCONAZOLE, _FLUCONAZOLE, _DMSO, _RAPAMYCIN)
    ]
    reference_phenotype = dataset[0]["reference"]["phenotype_reference"]
    assert {
        key: reference_phenotype[key]
        for key in (
            "environment_response",
            "n_samples",
            "sample_unit",
            "environment_response_uncertainty",
            "environment_response_uncertainty_type",
            "environment_response_se",
        )
    } == {
        "environment_response": 0.0,
        "n_samples": None,
        "sample_unit": None,
        "environment_response_uncertainty": None,
        "environment_response_uncertainty_type": None,
        "environment_response_se": None,
    }
    assert (
        dataset[0]["reference"]["environment_reference"]
        == dataset[0]["experiment"]["environment"]
    )


def test_drop_log_accounts_for_every_rule_in_application_order(
    dataset: s.CrispriChemgenSmith2016Dataset,
) -> None:
    """``dropped_records.json``: 9 source rows, 4 kept, 5 dropped = 1 vendor code (row 3,
    item 1181-0519) + 2 no-A/var(A) (rows 4 and 5: a blank A and a ``n/a`` variance both
    coerce to NaN) + 1 unresolved target (row 6, item YZZ999W) + 1 guide with no spacer
    (row 7, item ERG11_g9). The log lists the vendor-code rule first, although the loop
    checks a missing A or var(A) (smith2016.py line 600) before the vendor code (line 604).
    """
    drop_log = json.loads(
        (Path(dataset.root) / "preprocess" / "dropped_records.json").read_text()
    )
    assert drop_log == {
        "dataset": "CrispriChemgenSmith2016Dataset",
        "source_records": 9,
        "kept_records": 4,
        "dropped_records": 5,
        "rules": [
            {
                "rule": "drug_is_a_vendor_catalog_code_with_no_released_structure",
                "scope": "compound",
                "description": (
                    "the Drug label is a ChemDiv / ChemBridge / TimTec catalog id and "
                    "the primary released no structure, SMILES or CAS for it, so "
                    "resolve_compound_identity returns PROPRIETARY with no InChIKey, "
                    "ChEBI id or PubChem CID and the compound entity cannot be encoded "
                    "or joined"
                ),
                "n_records": 1,
                "items": ["1181-0519"],
            },
            {
                "rule": "row_has_no_A_or_var_A",
                "scope": "row",
                "description": (
                    "the released row carries no ATc-induced fold change or no variance "
                    "for it (0 today; every row of Additional file 10 is complete)"
                ),
                "n_records": 2,
                "items": [],
            },
            {
                "rule": "target_orf_is_not_a_current_genome_gene",
                "scope": "row",
                "description": (
                    "the target ORF does not resolve to a gene of the current R64 "
                    "annotation (0 today; all 20 targets are current)"
                ),
                "n_records": 1,
                "items": ["YZZ999W"],
            },
            {
                "rule": "guide_has_no_released_spacer",
                "scope": "guide",
                "description": (
                    "the guide name is absent from Additional file 4, so no spacer "
                    "exists to carry the strain identity (0 today; all 977 screened "
                    "guides join)"
                ),
                "n_records": 1,
                "items": ["ERG11_g9"],
            },
        ],
    }
    assert drop_log == s.DropLog.model_validate(drop_log).model_dump()


def test_gene_set_build_manifest_and_interned_store(
    dataset: s.CrispriChemgenSmith2016Dataset,
) -> None:
    """``gene_set.json`` is the two resolved ORFs sorted; the build manifest names the
    root slug, the loader class and module, this host and the worktree HEAD; the
    ``interned`` store holds 6 entries: the three distinct environments and the three
    references (each above ``INTERN_MIN_BYTES``), while the four-field DOI-only
    publication stays inline.
    """
    preprocess = Path(dataset.root) / "preprocess"
    assert json.loads((preprocess / "gene_set.json").read_text()) == [
        "YHR007C",
        "YJR066W",
    ]
    assert dataset.gene_set == {"YHR007C", "YJR066W"}
    assert dataset.experiment_class is EnvironmentResponseExperiment
    assert dataset.reference_class is EnvironmentResponseExperimentReference
    manifest = json.loads((preprocess / "build_manifest.json").read_text())
    head = subprocess.run(
        ["git", "-C", str(Path(s.__file__).parent), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert manifest["manifest_schema_version"] == 1
    assert manifest["dataset_name"] == "crispri_chemgen_smith2016"
    assert manifest["loader_class"] == "CrispriChemgenSmith2016Dataset"
    assert manifest["loader_module"] == "torchcell.datasets.scerevisiae.smith2016"
    assert manifest["hostname"] == socket.gethostname()
    assert manifest["torchcell_commit"] == head
    assert {
        "EnvironmentResponseExperiment",
        "CrisprInterferencePerturbation",
        "SmallMoleculePerturbation",
    } <= set(manifest["closure"])
    assert _interned_entries(Path(dataset.root)) == 6


def test_store_reopens_without_a_genome_and_serves_the_getters(
    dataset: s.CrispriChemgenSmith2016Dataset,
) -> None:
    """With ``processed/lmdb`` present a second instance on the same root skips both
    ``download()`` and ``process()``, so it needs no genome and no raw files: its four
    records equal the first instance's, ``get([0, 3])`` returns the two named records
    as a list while ``dataset[[0, 3]]`` is PyG's two-item subset view of the same records,
    ``transform_item`` rebuilds the typed objects, and ``repr`` is ``ClassName(4)``.
    The inline hooks: ``create_experiment`` raises ``NotImplementedError`` and
    ``preprocess_raw`` returns its input unchanged.
    """
    records = [dataset[i] for i in range(4)]
    dataset.close_lmdb()
    for name in (s.EFFECT_FILENAME, s.GUIDE_FILENAME):
        (Path(dataset.root) / "raw" / name).unlink()
    reopened = s.CrispriChemgenSmith2016Dataset(root=dataset.root, genome=None)
    assert len(reopened) == 4
    assert [reopened[i] for i in range(4)] == records
    assert reopened.get([0, 3]) == [records[0], records[3]]
    subset = reopened[[0, 3]]
    assert len(subset) == 2
    assert [subset[0], subset[1]] == [records[0], records[3]]
    # PyG's list index is a shallow copy that opens its own handle on the same store;
    # py-lmdb refuses a second open of one path in a process, so close it first.
    subset.close_lmdb()
    assert reopened.transform_item(reopened[3]) == {
        "experiment": _EXPECTED[3],
        "reference": _reference(_RAPAMYCIN),
        "publication": _PUBLICATION,
    }
    assert repr(reopened) == "CrispriChemgenSmith2016Dataset(4)"
    assert reopened.gene_set == {"YHR007C", "YJR066W"}
    with pytest.raises(NotImplementedError):
        reopened.create_experiment()
    sentinel = object()
    assert reopened.preprocess_raw(sentinel) is sentinel
    reopened.close_lmdb()


def test_genome_is_required_before_the_store_is_opened(tmp_path: Path) -> None:
    """With the raw workbooks present and ``genome=None`` the resolver (line 563) raises
    the loader's message before ``_open_write_lmdb`` (592), so no ``processed/lmdb`` is
    left behind and a retry with a genome builds normally.
    """
    root = _root(tmp_path)
    with pytest.raises(
        RuntimeError,
        match=re.escape(
            "CrispriChemgenSmith2016Dataset requires a genome; "
            "inject SCerevisiaeGenome(...)"
        ),
    ):
        s.CrispriChemgenSmith2016Dataset(root=str(root), genome=None)
    assert not (root / "processed" / "lmdb").exists()
    retry = s.CrispriChemgenSmith2016Dataset(root=str(root), genome=_genome())
    assert len(retry) == 4
    retry.close_lmdb()


def _staged_workbooks(tmp_path: Path, name: str) -> tuple[Path, str, str]:
    """Write the fixture workbooks to ``<tmp>/<name>`` and return (dir, effect, guide)."""
    staging = tmp_path / name
    _write_workbooks(staging)
    return (
        staging,
        _sha256(staging / s.EFFECT_FILENAME),
        _sha256(staging / s.GUIDE_FILENAME),
    )


def _artifact(relpath: str, filename: str, digest: str, size: int) -> ArtifactRecord:
    url = f"{s._ESM_BASE}{filename}"
    return ArtifactRecord(
        path=relpath,
        role=ROLE_SI_DATA,
        bytes=size,
        sha256=digest,
        source=url,
        retrieval=RetrievalRecord(
            method=RetrievalMethod.springer_esm,
            source_url=url,
            retriever="torchcell.literature.retrieve.springer_esm",
            params={"url": url},
            sha256=digest,
            retrieved_at="2026-09-12",
        ),
    )


def test_deposit_raw_mirror_pins_hashes_and_writes_the_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``deposit_raw_mirror`` refuses a source whose sha256 is not the pinned one, naming
    both digests. With the pinned constants patched to the fixture digests it copies both
    workbooks under ``$DATA_ROOT/torchcell-raw/<key>/si/si_data/`` and writes a
    ``Manifest`` with two ``springer_esm`` artifact records (byte sizes from the copies,
    ``retrieved_at`` 2026-09-12), the DOI, the two ESM URLs, ``provenance_complete``
    True and a UTC ``created_at``. A second call with the same files is a no-op; a
    different source file under a re-pinned digest is refused because the destination
    already holds other bytes. ``manifest_sha256`` reads a record back and raises
    ``KeyError`` for an unlisted path.
    """
    data_root = tmp_path / "data_root"
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    staging, effect_sha, guide_sha = _staged_workbooks(tmp_path, "staging")
    effect_path = staging / s.EFFECT_FILENAME
    guide_path = staging / s.GUIDE_FILENAME
    with pytest.raises(RawSha256MismatchError) as err:
        s.deposit_raw_mirror(effect_path=effect_path, guide_path=guide_path)
    assert str(err.value) == (
        f"sha256 mismatch for {effect_path}: expected {s.EFFECT_SHA256}, "
        f"observed {effect_sha}"
    )
    assert not (data_root / "torchcell-raw").exists()
    # The guide workbook off its pin is caught before the effect workbook is written.
    monkeypatch.setattr(s, "EFFECT_SHA256", effect_sha)
    with pytest.raises(RawSha256MismatchError) as err:
        s.deposit_raw_mirror(effect_path=effect_path, guide_path=guide_path)
    assert str(err.value) == (
        f"sha256 mismatch for {guide_path}: expected {s.GUIDE_SHA256}, "
        f"observed {guide_sha}"
    )
    assert not (data_root / "torchcell-raw").exists()

    monkeypatch.setattr(s, "EFFECT_SHA256", effect_sha)
    monkeypatch.setattr(s, "GUIDE_SHA256", guide_sha)
    mirror = s.deposit_raw_mirror(effect_path=effect_path, guide_path=guide_path)
    assert mirror == data_root / "torchcell-raw" / s.CITATION_KEY
    assert mirror == s.raw_mirror_dir()
    assert (mirror / s.EFFECT_REL).read_bytes() == effect_path.read_bytes()
    assert (mirror / s.GUIDE_REL).read_bytes() == guide_path.read_bytes()
    manifest = s.load_manifest()
    expected = Manifest(
        citation_key=s.CITATION_KEY,
        doi=s.DOI,
        title=(
            "Quantitative CRISPR interference screens in yeast identify chemical-genetic "
            "interactions and new rules for guide RNA design"
        ),
        files=[
            _artifact(
                s.EFFECT_REL, s.EFFECT_FILENAME, effect_sha, effect_path.stat().st_size
            ),
            _artifact(
                s.GUIDE_REL, s.GUIDE_FILENAME, guide_sha, guide_path.stat().st_size
            ),
        ],
        si_data_sources=[
            f"{s._ESM_BASE}{s.EFFECT_FILENAME}",
            f"{s._ESM_BASE}{s.GUIDE_FILENAME}",
        ],
        si_expected=[
            "Additional file 10 (ATc effects A, drug effects D, no-drug ATc effects A0 "
            "for each gRNA in every tested condition)",
            "Additional file 4 (the five gRNA libraries and their Specificity_sequence "
            "spacers)",
        ],
        provenance_complete=True,
    )
    assert manifest.model_dump(exclude={"created_at"}) == expected.model_dump(
        exclude={"created_at"}
    )
    assert manifest.created_at is not None
    assert datetime.fromisoformat(manifest.created_at).tzinfo == UTC
    assert s.manifest_sha256(manifest, s.EFFECT_REL) == effect_sha
    assert s.manifest_sha256(manifest, s.GUIDE_REL) == guide_sha
    with pytest.raises(KeyError, match="nope is not in the raw-mirror manifest"):
        s.manifest_sha256(manifest, "nope")

    written = (mirror / "manifest.json").read_text()
    assert (
        s.deposit_raw_mirror(effect_path=effect_path, guide_path=guide_path) == mirror
    )
    assert (mirror / s.EFFECT_REL).read_bytes() == effect_path.read_bytes()
    assert s.load_manifest().files == Manifest.model_validate_json(written).files

    other, other_sha, _ = _staged_workbooks(tmp_path, "other")
    (other / s.EFFECT_FILENAME).write_bytes(b"a different workbook")
    other_sha = _sha256(other / s.EFFECT_FILENAME)
    monkeypatch.setattr(s, "EFFECT_SHA256", other_sha)
    with pytest.raises(
        RuntimeError,
        match=re.escape(
            f"{mirror / s.EFFECT_REL} exists with a different sha256; refusing"
        ),
    ):
        s.deposit_raw_mirror(
            effect_path=other / s.EFFECT_FILENAME, guide_path=guide_path
        )
    assert (mirror / s.EFFECT_REL).read_bytes() == effect_path.read_bytes()


def test_download_links_the_verified_mirror_and_refuses_drift_or_absence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no raw files, ``download()`` needs the mirror manifest (``FileNotFoundError``
    naming ``manifest.json`` when absent). Once the fixture workbooks are deposited, a
    root whose ``raw/`` already links the guide workbook keeps that link (``download()``
    only creates a missing one), gets the effect workbook linked beside it and builds
    the same four records. A
    mirror file whose bytes drift from the manifest is refused naming both digests; a
    removed mirror file is refused naming its path.
    """
    data_root = tmp_path / "data_root"
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    manifest_path = data_root / "torchcell-raw" / s.CITATION_KEY / "manifest.json"
    with pytest.raises(FileNotFoundError, match=re.escape(str(manifest_path))):
        s.CrispriChemgenSmith2016Dataset(root=str(tmp_path / "a"), genome=_genome())

    staging, effect_sha, guide_sha = _staged_workbooks(tmp_path, "staging")
    monkeypatch.setattr(s, "EFFECT_SHA256", effect_sha)
    monkeypatch.setattr(s, "GUIDE_SHA256", guide_sha)
    mirror = s.deposit_raw_mirror(
        effect_path=staging / s.EFFECT_FILENAME, guide_path=staging / s.GUIDE_FILENAME
    )
    root = tmp_path / "b"
    (root / "raw").mkdir(parents=True)
    os.symlink(mirror / s.GUIDE_REL, root / "raw" / s.GUIDE_FILENAME)
    built = s.CrispriChemgenSmith2016Dataset(root=str(root), genome=_genome())
    assert (root / "raw" / s.EFFECT_FILENAME).is_symlink()
    assert (root / "raw" / s.EFFECT_FILENAME).resolve() == (
        mirror / s.EFFECT_REL
    ).resolve()
    assert (root / "raw" / s.GUIDE_FILENAME).resolve() == (
        mirror / s.GUIDE_REL
    ).resolve()
    assert len(built) == 4
    assert [built[i]["experiment"] for i in range(4)] == [
        e.model_dump() for e in _EXPECTED
    ]
    built.close_lmdb()

    (mirror / s.EFFECT_REL).write_bytes(b"drifted upstream")
    drifted = _sha256(mirror / s.EFFECT_REL)
    with pytest.raises(RawSha256MismatchError) as err:
        s.CrispriChemgenSmith2016Dataset(root=str(tmp_path / "c"), genome=_genome())
    assert str(err.value) == (
        f"sha256 mismatch for {mirror / s.EFFECT_REL}: expected {effect_sha}, "
        f"observed {drifted}"
    )
    assert list((tmp_path / "c" / "raw").iterdir()) == []
    (mirror / s.EFFECT_REL).unlink()
    with pytest.raises(
        RuntimeError,
        match=re.escape(
            f"required raw artifact missing from mirror: {mirror / s.EFFECT_REL}"
        ),
    ):
        s.CrispriChemgenSmith2016Dataset(root=str(tmp_path / "d"), genome=_genome())


def test_a_raw_file_off_the_pin_is_refused_at_build_time(
    off_pin_raw: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Contract (issue #518's sweep): with both workbooks already in ``raw/`` PyG skips
    ``download()``, so ``process()`` verifies them against ``EFFECT_SHA256`` and
    ``GUIDE_SHA256`` first. The effect workbook off its pin raises
    ``RawSha256MismatchError`` naming it and both digests; no store is written.
    """
    monkeypatch.delenv("TC_DATA_URL", raising=False)
    staged = off_pin_raw(s, [s.EFFECT_FILENAME, s.GUIDE_FILENAME])
    raw = staged.root / "raw" / s.EFFECT_FILENAME
    with pytest.raises(RawSha256MismatchError) as err:
        s.CrispriChemgenSmith2016Dataset(root=str(staged.root), genome=_genome())
    assert str(err.value) == (
        f"sha256 mismatch for {raw}: expected "
        "02962e51e492b0505e8595fc1c80fab5fca8a8c8f05e05969dbff18ddff71cd0, "
        f"observed {staged.observed}"
    )
    assert list((staged.root / "processed").iterdir()) == []
    assert not (staged.root / "preprocess").exists()
    assert _sha256(raw) == staged.observed


# Phase 24: the drop-accounting guard


def test_a_drop_log_that_disagrees_with_its_rules_refuses_after_writing_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``DropLog`` is patched to report one more dropped record than the build lost.

    The rules total 5 (1 vendor code + 2 no A/var(A) + 1 unresolved + 1 no spacer, as
    in ``test_drop_log_accounts_for_every_rule_in_application_order``) while the log
    says 6: the exact mismatch message, and the tampered 6 already on disk.
    """
    real = s.DropLog

    def inflated(**kwargs: Any) -> Any:
        kwargs["dropped_records"] += 1
        return real(**kwargs)

    monkeypatch.setattr(s, "DropLog", inflated)
    root = _root(tmp_path)
    with pytest.raises(RuntimeError) as err:
        s.CrispriChemgenSmith2016Dataset(root=str(root), genome=_genome())
    assert str(err.value) == (
        "drop accounting mismatch: rules total 5, 6 records missing from the build"
    )
    log = json.loads((root / "preprocess" / "dropped_records.json").read_text())
    assert log["dropped_records"] == 6
