# tests/torchcell/datasets/scerevisiae/test_mormino2022_synthetic.py
# [[tests.torchcell.datasets.scerevisiae.test_mormino2022_synthetic]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/scerevisiae/test_mormino2022_synthetic.py
"""Hermetic end-to-end build of ``CrispriMormino2022Dataset`` from a synthetic OCR page.

The loader's data is the module literal ``TABLE_1`` (12 isolated strains); the raw files
are only the article PDF (never read by ``process()``) and ``paper.md``, which the build
audits every Table 1 row against verbatim. Both are written under ``<root>/raw/`` so PyG
never calls ``download()``: the PDF as stub bytes, ``paper.md`` as one HTML row per
``table_1_row_fragment``. ``process()`` runs once against a duck-typed genome stub
carrying ``resolve_gene_name`` and ``feature_index["standard_to_ids"]``; the stub is the
authority for every mapping (QCR8 YJL166W, TIF34 YMR146C, MSN5 YDR335W, NDC1 YML031W,
PAP1 YKR002W, CBP2 YHL038C, COX10 YPL172C, TRA1 YHR099W, UBA2 YDR390C, RPS30B YOR182C,
HSH49 YOR319W, LCB1 YMR296C).

Expected: 12 records in Table 1 order, each a categorical ``EnvironmentResponseExperiment``
whose genotype is one dCas9-Mxi1 CRISPRi perturbation (no guide, ``n_guides = 1``) plus
the two pMM4_14L cassette additions, whose environment is SC + 50 mM acetic acid + 2
ug/mL anhydrotetracycline in DMSO + pH 3.5 at 30 C, and whose call is ``+`` ->
``enhanced`` (rows #3, #8, #13, #17, #33, #35) or ``=`` -> ``no_change`` (#15, #32, #37,
#43, #46, #49), ``n_samples = 2`` biological replicates. One environment, so one CBL
reference (``no_change``, ``=``) covering all 12.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import socket
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import lmdb
import pytest

from torchcell.datamodels.compound_identity import resolved_compound
from torchcell.datamodels.media import SC
from torchcell.datamodels.schema import (
    AssayType,
    Concentration,
    ConcentrationUnit,
    CrisprConstruct,
    CrisprInterferencePerturbation,
    Environment,
    EnvironmentPhysicalPerturbation,
    EnvironmentResponseExperiment,
    EnvironmentResponseExperimentReference,
    EnvironmentResponsePhenotype,
    Genotype,
    MeasurementType,
    PhysicalFactor,
    Publication,
    ReferenceGenome,
    ResponseCategory,
    SampleUnit,
    SmallMoleculePerturbation,
    Solvent,
    Temperature,
)
from torchcell.datasets.scerevisiae import mormino2022 as mo
from torchcell.literature.manifest import (
    ROLE_PAPER_OCR,
    ROLE_PAPER_PDF,
    ArtifactRecord,
    Manifest,
    ProcessingRecord,
    RetrievalMethod,
    RetrievalRecord,
)
from torchcell.sequence.genome.scerevisiae import SCerevisiaeGenome

_ORF = {
    "QCR8": "YJL166W",
    "TIF34": "YMR146C",
    "MSN5": "YDR335W",
    "NDC1": "YML031W",
    "PAP1": "YKR002W",
    "CBP2": "YHL038C",
    "COX10": "YPL172C",
    "TRA1": "YHR099W",
    "UBA2": "YDR390C",
    "RPS30B": "YOR182C",
    "HSH49": "YOR319W",
    "LCB1": "YMR296C",
}


class _Resolution:
    """The two attributes and the one property the loader reads from a resolution."""

    def __init__(self, status: str, systematic: str | None) -> None:
        self.status = status
        self.systematic_name = systematic

    @property
    def is_current_gene(self) -> bool:
        return self.status in ("current", "renamed")


class _StubGenome:
    """A genome whose standard names are ``standard``; everything else is retired."""

    def __init__(self, standard: dict[str, str]) -> None:
        self.standard = standard
        self.gene_set = set(standard.values())
        self.feature_index = {
            "standard_to_ids": {name: [orf] for name, orf in standard.items()}
        }

    def resolve_gene_name(self, name: str) -> _Resolution:
        upper = name.strip().upper()
        if upper in self.gene_set:
            return _Resolution("current", upper)
        if upper in self.standard:
            return _Resolution("renamed", self.standard[upper])
        return _Resolution("retired", upper)


def _genome(standard: dict[str, str] = _ORF) -> SCerevisiaeGenome:
    return cast(SCerevisiaeGenome, _StubGenome(standard))


def _paper_md(rows: list[tuple[str, str, str, str]] = mo.TABLE_1) -> str:
    body = "</tr><tr>".join(mo.table_1_row_fragment(row) for row in rows)
    return f"# Table 1\n\n<table><tr>{body}</tr></table>\n"


def _write_raw(raw: Path, paper_md: str | None = None) -> None:
    raw.mkdir(parents=True, exist_ok=True)
    (raw / mo.PDF_FILENAME).write_bytes(b"%PDF-1.4 stub article bytes")
    (raw / mo.PAPER_MD).write_text(
        _paper_md() if paper_md is None else paper_md, encoding="utf-8"
    )


def _root(tmp_path: Path, slug: str = "crispri_mormino2022") -> Path:
    root = tmp_path / slug
    _write_raw(root / "raw")
    return root


@pytest.fixture
def dataset(tmp_path: Path) -> mo.CrispriMormino2022Dataset:
    return mo.CrispriMormino2022Dataset(root=str(_root(tmp_path)), genome=_genome())


_PUBLICATION = Publication(
    pubmed_id="36284296",
    pubmed_url="https://pubmed.ncbi.nlm.nih.gov/36284296/",
    doi="10.1186/s12934-022-01938-7",
    doi_url="https://doi.org/10.1186/s12934-022-01938-7",
)
_ENVIRONMENT = Environment(
    media=SC,
    temperature=Temperature(value=30.0),
    perturbations=[
        SmallMoleculePerturbation(
            compound=resolved_compound("acetic acid"),
            concentration=Concentration(value=50.0, unit=ConcentrationUnit.millimolar),
        ),
        SmallMoleculePerturbation(
            compound=resolved_compound("anhydrotetracycline"),
            concentration=Concentration(value=2.0, unit=ConcentrationUnit.ug_per_ml),
            solvent=Solvent(name="DMSO", compound=resolved_compound("DMSO")),
        ),
        EnvironmentPhysicalPerturbation(
            factor=PhysicalFactor.ph,
            magnitude=Concentration(value=3.5, unit=ConcentrationUnit.ph),
        ),
    ],
    aerobicity="aerobic",
)
_CALL = {"+": (ResponseCategory.enhanced, "+"), "=": (ResponseCategory.no_change, "=")}


def _experiment(common: str, rfp: str) -> EnvironmentResponseExperiment:
    category, label = _CALL[rfp]
    return EnvironmentResponseExperiment(
        dataset_name="CrispriMormino2022Dataset",
        genotype=Genotype(
            perturbations=[
                CrisprInterferencePerturbation(
                    systematic_gene_name=_ORF[common],
                    perturbed_gene_name=common,
                    crispr=CrisprConstruct(
                        effector="dCas9-Mxi1", guide_sequence=None, n_guides=1
                    ),
                ),
                *mo.biosensor_cassette(),
            ]
        ),
        environment=_ENVIRONMENT,
        phenotype=EnvironmentResponsePhenotype(
            measurement_type=MeasurementType.categorical,
            assay_type=AssayType.biosensor_readout,
            category=category,
            category_label=label,
            n_samples=2,
            sample_unit=SampleUnit.biological_replicate,
            units=mo.UNITS,
        ),
    )


_REFERENCE = EnvironmentResponseExperimentReference(
    dataset_name="CrispriMormino2022Dataset",
    genome_reference=ReferenceGenome(
        species="Saccharomyces cerevisiae", strain="BY4742"
    ),
    environment_reference=_ENVIRONMENT,
    phenotype_reference=EnvironmentResponsePhenotype(
        measurement_type=MeasurementType.categorical,
        assay_type=AssayType.biosensor_readout,
        category=ResponseCategory.no_change,
        category_label="=",
        n_samples=2,
        sample_unit=SampleUnit.biological_replicate,
        units=mo.UNITS,
    ),
)
_EXPECTED = [_experiment(common, rfp) for _, rfp, _, common in mo.TABLE_1]


def _interned_entries(root: Path) -> int:
    env = lmdb.open(str(root / "processed" / "interned"), readonly=True, lock=False)
    with env.begin() as txn:
        entries: int = txn.stat()["entries"]
    env.close()
    return entries


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_build_writes_the_twelve_table_1_strains_as_categorical_records(
    dataset: mo.CrispriMormino2022Dataset,
) -> None:
    """Twelve records in Table 1 order, each equal to a hand-built categorical experiment:
    ``+`` rows (#3 QCR8, #8 TIF34, #13 MSN5, #17 PAP1, #33 COX10, #35 TRA1) store
    ``enhanced`` / ``+``, ``=`` rows (#15 NDC1, #32 CBP2, #37 UBA2, #43 RPS30B, #46 HSH49,
    #49 LCB1) store ``no_change`` / ``=``; ``environment_response`` is None on every
    record, the perturbed name is the genome's standard name, and the two cassette
    additions ride in every genotype. The publication carries PMID 36284296 and the DOI.
    """
    assert len(dataset) == 12
    assert [dataset[i]["experiment"] for i in range(12)] == [
        e.model_dump() for e in _EXPECTED
    ]
    assert [
        (
            dataset[i]["experiment"]["phenotype"]["category"],
            dataset[i]["experiment"]["phenotype"]["category_label"],
            dataset[i]["experiment"]["phenotype"]["environment_response"],
        )
        for i in range(12)
    ] == [
        (ResponseCategory.enhanced, "+", None),
        (ResponseCategory.enhanced, "+", None),
        (ResponseCategory.enhanced, "+", None),
        (ResponseCategory.no_change, "=", None),
        (ResponseCategory.enhanced, "+", None),
        (ResponseCategory.no_change, "=", None),
        (ResponseCategory.enhanced, "+", None),
        (ResponseCategory.enhanced, "+", None),
        (ResponseCategory.no_change, "=", None),
        (ResponseCategory.no_change, "=", None),
        (ResponseCategory.no_change, "=", None),
        (ResponseCategory.no_change, "=", None),
    ]
    genotype = dataset[0]["experiment"]["genotype"]["perturbations"]
    assert [
        (p["systematic_gene_name"], p["perturbed_gene_name"], p["perturbation_type"])
        for p in genotype
    ] == [
        ("BM3R1-HAA1-mTurquoise2", "BM3R1-HAA1-mTurquoise2", "gene_addition"),
        ("YJL166W", "QCR8", "crispr_interference"),
        ("sfpHluorin", "sfpHluorin", "gene_addition"),
    ]
    assert dataset[0]["publication"] == _PUBLICATION.model_dump()


def test_one_cbl_reference_and_the_gene_set_carries_the_cassette_slots(
    dataset: mo.CrispriMormino2022Dataset,
) -> None:
    """Finding: the module docstring says the two cassette additions "leave ... the L4
    gene set unchanged", but that holds for the verifier's ``BACKGROUND_GENES`` only; the
    dataset's own ``gene_set.json`` is computed from every perturbation's
    ``systematic_gene_name`` (experiment_dataset.py 469-473 over the genotype written at
    mormino2022.py 642-651), so it lists ``BM3R1-HAA1-mTurquoise2`` and ``sfpHluorin``
    beside the twelve ORFs: 14 entries, sorted with uppercase before lowercase.

    One environment gives one CBL reference (BY4742, ``no_change`` / ``=``, n = 2)
    covering member indices 0-11; the build manifest names the slug, class, module, host
    and HEAD; ``interned`` holds 2 entries (the environment and the reference) and the
    publication stays inline.
    """
    preprocess = Path(dataset.root) / "preprocess"
    index = json.loads((preprocess / "experiment_reference_index.json").read_text())
    assert [entry["member_indices"] for entry in index] == [list(range(12))]
    assert index[0]["reference"] == _REFERENCE.model_dump()
    assert [dataset[i]["reference"] for i in range(12)] == [
        _REFERENCE.model_dump()
    ] * 12
    assert json.loads((preprocess / "gene_set.json").read_text()) == [
        "BM3R1-HAA1-mTurquoise2",
        "YDR335W",
        "YDR390C",
        "YHL038C",
        "YHR099W",
        "YJL166W",
        "YKR002W",
        "YML031W",
        "YMR146C",
        "YMR296C",
        "YOR182C",
        "YOR319W",
        "YPL172C",
        "sfpHluorin",
    ]
    assert dataset.gene_set == set(_ORF.values()) | mo.BACKGROUND_GENES
    assert dataset.experiment_class is EnvironmentResponseExperiment
    assert dataset.reference_class is EnvironmentResponseExperimentReference
    manifest = json.loads((preprocess / "build_manifest.json").read_text())
    head = subprocess.run(
        ["git", "-C", str(Path(mo.__file__).parent), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert manifest["manifest_schema_version"] == 1
    assert manifest["dataset_name"] == "crispri_mormino2022"
    assert manifest["loader_class"] == "CrispriMormino2022Dataset"
    assert manifest["loader_module"] == "torchcell.datasets.scerevisiae.mormino2022"
    assert manifest["hostname"] == socket.gethostname()
    assert manifest["torchcell_commit"] == head
    assert {
        "CrisprInterferencePerturbation",
        "GeneAdditionPerturbation",
        "EnvironmentPhysicalPerturbation",
    } <= set(manifest["closure"])
    assert _interned_entries(Path(dataset.root)) == 2


def test_drop_log_records_the_single_no_op_rule(
    dataset: mo.CrispriMormino2022Dataset,
) -> None:
    """``dropped_records.json`` is 12 source, 12 kept, 0 dropped with one ``none`` rule."""
    drop_log = json.loads(
        (Path(dataset.root) / "preprocess" / "dropped_records.json").read_text()
    )
    assert drop_log == {
        "dataset": "CrispriMormino2022Dataset",
        "source_records": 12,
        "kept_records": 12,
        "dropped_records": 0,
        "rules": [
            {
                "rule": "none",
                "scope": "row",
                "description": (
                    "no retention rule removes a Mormino record: acetic acid, "
                    "anhydrotetracycline and DMSO all resolve to a structure identifier, "
                    "all 12 targets resolve to current R64 genes, and every Table 1 row "
                    "is verified verbatim against the OCR table"
                ),
                "n_records": 0,
                "items": [],
            }
        ],
    }
    assert drop_log == mo.DropLog.model_validate(drop_log).model_dump()


def test_store_reopens_without_a_genome_and_serves_the_getters(
    dataset: mo.CrispriMormino2022Dataset,
) -> None:
    """With ``processed/lmdb`` present a second instance on the same root skips both
    ``download()`` and ``process()`` (no genome, raw files deleted): twelve equal records,
    ``get([0, 11])`` as a list, ``dataset[[0, 11]]`` as PyG's two-item subset,
    ``transform_item`` rebuilding the typed LCB1 record, ``repr``
    ``CrispriMormino2022Dataset(12)``, and the inline hooks.
    """
    records = [dataset[i] for i in range(12)]
    dataset.close_lmdb()
    for name in (mo.PDF_FILENAME, mo.PAPER_MD):
        (Path(dataset.root) / "raw" / name).unlink()
    reopened = mo.CrispriMormino2022Dataset(root=dataset.root, genome=None)
    assert len(reopened) == 12
    assert [reopened[i] for i in range(12)] == records
    assert reopened.get([0, 11]) == [records[0], records[11]]
    subset = reopened[[0, 11]]
    assert len(subset) == 2
    assert [subset[0], subset[1]] == [records[0], records[11]]
    # PyG's list index is a shallow copy that opens its own handle on the same store;
    # py-lmdb refuses a second open of one path in a process, so close it first.
    subset.close_lmdb()
    assert reopened.transform_item(reopened[11]) == {
        "experiment": _EXPECTED[11],
        "reference": _REFERENCE,
        "publication": _PUBLICATION,
    }
    assert repr(reopened) == "CrispriMormino2022Dataset(12)"
    assert reopened.gene_set == set(_ORF.values()) | mo.BACKGROUND_GENES
    with pytest.raises(NotImplementedError):
        reopened.create_experiment()
    sentinel = object()
    assert reopened.preprocess_raw(sentinel) is sentinel
    reopened.close_lmdb()


def test_genome_none_fails_on_the_bare_assert_not_the_resolver_message(
    tmp_path: Path, dataset: mo.CrispriMormino2022Dataset
) -> None:
    """Finding: ``process()`` asserts ``self.genome is not None`` (mormino2022.py 620)
    before the first ``_resolve`` call (637), so a build with ``genome=None`` raises a
    bare ``AssertionError`` with an empty message and the loader's own "requires a
    genome; inject SCerevisiaeGenome(...)" message (552-554) is unreachable from a build.
    Called directly on a built instance whose genome is cleared, ``_resolve`` raises it.
    No ``processed/lmdb`` is left behind by the failed build.
    """
    root = _root(tmp_path, "no_genome")
    with pytest.raises(AssertionError) as excinfo:
        mo.CrispriMormino2022Dataset(root=str(root), genome=None)
    assert str(excinfo.value) == ""
    assert not (root / "processed" / "lmdb").exists()
    dataset.genome = None
    with pytest.raises(
        RuntimeError,
        match=re.escape(
            "CrispriMormino2022Dataset requires a genome; inject SCerevisiaeGenome(...)"
        ),
    ):
        dataset._resolve("QCR8")


def test_unresolved_table_1_gene_raises_and_leaves_a_store_a_retry_serves_empty(
    tmp_path: Path,
) -> None:
    """Finding: a Table 1 gene the genome does not resolve raises the loader's message
    naming the gene and the resolver status, but the write env was already opened at
    mormino2022.py 633, so the failed build leaves an empty ``processed/lmdb`` (the
    aborted transaction wrote nothing) and no ``dropped_records.json``; a retry on the
    same root, even with a complete genome, finds the store present, skips ``process()``
    and serves 0 records. The root has to be cleared by hand before a rebuild.
    """
    root = _root(tmp_path, "partial")
    incomplete = {name: orf for name, orf in _ORF.items() if name != "LCB1"}
    with pytest.raises(
        RuntimeError,
        match=re.escape(
            "Mormino2022: gene 'LCB1' did not resolve to a current R64 gene (retired)"
        ),
    ):
        mo.CrispriMormino2022Dataset(root=str(root), genome=_genome(incomplete))
    assert (root / "processed" / "lmdb").is_dir()
    assert not (root / "preprocess" / "dropped_records.json").exists()
    assert not (root / "preprocess" / "gene_set.json").exists()
    retry = mo.CrispriMormino2022Dataset(root=str(root), genome=_genome())
    assert len(retry) == 0
    assert not (root / "preprocess" / "dropped_records.json").exists()
    retry.close_lmdb()


def test_audit_failure_raises_before_the_store_is_opened_so_a_retry_rebuilds(
    tmp_path: Path,
) -> None:
    """A ``paper.md`` missing row #49 fails the verbatim audit (mormino2022.py 619) with
    the missing fragment in the message before any store is opened, so no
    ``processed/lmdb`` exists and, once the OCR page is complete, a retry on the same
    root builds all 12 records.
    """
    root = tmp_path / "audit"
    _write_raw(root / "raw", paper_md=_paper_md(mo.TABLE_1[:-1]))
    fragment = "<td>#49</td><td>=</td><td>ND</td><td>LCB1</td>"
    assert mo.table_1_row_fragment(mo.TABLE_1[-1]) == fragment
    with pytest.raises(
        RuntimeError,
        match=re.escape(
            f"1 Table 1 rows are not present verbatim in paper.md: ['{fragment}']"
        ),
    ):
        mo.CrispriMormino2022Dataset(root=str(root), genome=_genome())
    assert not (root / "processed" / "lmdb").exists()
    (root / "raw" / mo.PAPER_MD).write_text(_paper_md(), encoding="utf-8")
    retry = mo.CrispriMormino2022Dataset(root=str(root), genome=_genome())
    assert len(retry) == 12
    assert retry[11]["experiment"] == _EXPECTED[11].model_dump()
    retry.close_lmdb()


def _staged_raw(tmp_path: Path, name: str) -> tuple[Path, str, str]:
    """Write the fixture raw files to ``<tmp>/<name>``; return (dir, pdf sha, md sha)."""
    staging = tmp_path / name
    _write_raw(staging)
    return (staging, _sha256(staging / mo.PDF_FILENAME), _sha256(staging / mo.PAPER_MD))


def _expected_manifest(
    pdf_sha: str, pdf_bytes: int, md_sha: str, md_bytes: int
) -> Manifest:
    return Manifest(
        citation_key=mo.CITATION_KEY,
        doi=mo.DOI,
        title=(
            "Identification of acetic acid sensitive strains through biosensor-based "
            "screening of a Saccharomyces cerevisiae CRISPRi library"
        ),
        library_id="6582362",
        zotero_item_key="XKUDX4UN",
        files=[
            ArtifactRecord(
                path="paper.pdf",
                role=ROLE_PAPER_PDF,
                bytes=pdf_bytes,
                sha256=pdf_sha,
                source=mo.PDF_URL,
                retrieval=RetrievalRecord(
                    method=RetrievalMethod.direct_url,
                    source_url=mo.PDF_URL,
                    retriever="torchcell.literature.retrieve.direct_url",
                    params={"url": mo.PDF_URL},
                    sha256=pdf_sha,
                    retrieved_at="2026-09-12",
                ),
            ),
            ArtifactRecord(
                path="paper.md",
                role=ROLE_PAPER_OCR,
                bytes=md_bytes,
                sha256=md_sha,
                source="mineru-ocr",
                processing=ProcessingRecord(
                    processor="torchcell.literature.ocr.run_mineru",
                    tool="mineru",
                    version="unrecorded",
                    params={
                        "note": (
                            "the OCR was run 2026-07-12, before the tool version and DPI "
                            "were recorded per artifact; the artifact is sha256-pinned "
                            "and the runner is versioned source, but the exact version "
                            "is unknown and is NOT fabricated here"
                        )
                    },
                    input_sha256=[pdf_sha],
                ),
            ),
        ],
        si_data_sources=[mo.PDF_URL],
        si_expected=[
            "none -- the article Table 1 IS the data; no supplementary data file was "
            "released for this paper"
        ],
        provenance_complete=False,
    )


def test_deposit_raw_mirror_records_pdf_retrieval_and_unrecorded_ocr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``deposit_raw_mirror`` refuses a PDF whose sha256 is not the pinned one. With the
    pins patched to the fixture digests it copies ``paper.pdf`` and ``paper.md`` to the
    mirror root and writes a manifest: a ``direct_url`` retrieval for the PDF, a
    ``mineru`` ``ProcessingRecord`` with ``version = "unrecorded"`` and the PDF digest as
    its input for the OCR, ``library_id`` 6582362, Zotero key XKUDX4UN and
    ``provenance_complete`` False. A repeat deposit is a no-op; a different OCR page under
    a re-pinned digest is refused because the destination holds other bytes.
    """
    data_root = tmp_path / "data_root"
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    staging, pdf_sha, md_sha = _staged_raw(tmp_path, "staging")
    pdf_path = staging / mo.PDF_FILENAME
    md_path = staging / mo.PAPER_MD
    with pytest.raises(
        RuntimeError,
        match=re.escape(
            f"{pdf_path} sha256 mismatch: got {pdf_sha}, expected {mo.PDF_SHA256}"
        ),
    ):
        mo.deposit_raw_mirror(pdf_path=pdf_path, paper_md_path=md_path)
    assert sorted(p.name for p in (data_root / "torchcell-raw").rglob("*")) == [
        mo.CITATION_KEY
    ]

    monkeypatch.setattr(mo, "PDF_SHA256", pdf_sha)
    monkeypatch.setattr(mo, "PAPER_MD_SHA256", md_sha)
    mirror = mo.deposit_raw_mirror(pdf_path=pdf_path, paper_md_path=md_path)
    assert (
        mirror == data_root / "torchcell-raw" / mo.CITATION_KEY == mo.raw_mirror_dir()
    )
    assert (mirror / "paper.pdf").read_bytes() == pdf_path.read_bytes()
    assert (mirror / "paper.md").read_text(encoding="utf-8") == _paper_md()
    manifest = mo.load_manifest()
    expected = _expected_manifest(
        pdf_sha, pdf_path.stat().st_size, md_sha, md_path.stat().st_size
    )
    assert manifest.model_dump(exclude={"created_at"}) == expected.model_dump(
        exclude={"created_at"}
    )
    assert manifest.created_at is not None
    assert datetime.fromisoformat(manifest.created_at).tzinfo == UTC
    assert mo.manifest_sha256(manifest, "paper.pdf") == pdf_sha
    assert mo.manifest_sha256(manifest, "paper.md") == md_sha
    with pytest.raises(
        KeyError, match="si/nope.xlsx is not in the raw-mirror manifest"
    ):
        mo.manifest_sha256(manifest, "si/nope.xlsx")

    written = (mirror / "manifest.json").read_text()
    assert mo.deposit_raw_mirror(pdf_path=pdf_path, paper_md_path=md_path) == mirror
    assert mo.load_manifest().files == Manifest.model_validate_json(written).files

    other = tmp_path / "other"
    other.mkdir()
    (other / mo.PAPER_MD).write_text("a different OCR page", encoding="utf-8")
    monkeypatch.setattr(mo, "PAPER_MD_SHA256", _sha256(other / mo.PAPER_MD))
    with pytest.raises(
        RuntimeError,
        match=re.escape(
            f"{mirror / 'paper.md'} exists with a different sha256; refusing"
        ),
    ):
        mo.deposit_raw_mirror(pdf_path=pdf_path, paper_md_path=other / mo.PAPER_MD)
    assert (mirror / "paper.md").read_text(encoding="utf-8") == _paper_md()


def test_download_links_the_verified_mirror_and_refuses_drift_or_absence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no raw files ``download()`` needs the mirror manifest (``FileNotFoundError``
    naming it). After the fixture files are deposited a root whose ``raw/`` already links
    ``paper.pdf`` keeps that link (``download()`` only creates a missing one), gets
    ``paper.md`` linked beside it and builds the twelve records; a drifted OCR
    page is refused naming the relative path and both digests, and a removed one is
    refused naming its mirror path.
    """
    data_root = tmp_path / "data_root"
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    manifest_path = data_root / "torchcell-raw" / mo.CITATION_KEY / "manifest.json"
    with pytest.raises(FileNotFoundError, match=re.escape(str(manifest_path))):
        mo.CrispriMormino2022Dataset(root=str(tmp_path / "a"), genome=_genome())

    staging, pdf_sha, md_sha = _staged_raw(tmp_path, "staging")
    monkeypatch.setattr(mo, "PDF_SHA256", pdf_sha)
    monkeypatch.setattr(mo, "PAPER_MD_SHA256", md_sha)
    mirror = mo.deposit_raw_mirror(
        pdf_path=staging / mo.PDF_FILENAME, paper_md_path=staging / mo.PAPER_MD
    )
    root = tmp_path / "b"
    (root / "raw").mkdir(parents=True)
    os.symlink(mirror / "paper.pdf", root / "raw" / mo.PDF_FILENAME)
    built = mo.CrispriMormino2022Dataset(root=str(root), genome=_genome())
    assert (root / "raw" / mo.PDF_FILENAME).is_symlink()
    assert (root / "raw" / mo.PDF_FILENAME).resolve() == (
        mirror / "paper.pdf"
    ).resolve()
    assert (root / "raw" / mo.PAPER_MD).resolve() == (mirror / "paper.md").resolve()
    assert len(built) == 12
    assert [built[i]["experiment"] for i in range(12)] == [
        e.model_dump() for e in _EXPECTED
    ]
    built.close_lmdb()

    (mirror / "paper.md").write_text("drifted upstream", encoding="utf-8")
    drifted = _sha256(mirror / "paper.md")
    with pytest.raises(
        RuntimeError,
        match=re.escape(f"paper.md sha256 mismatch: got {drifted}, expected {md_sha}"),
    ):
        mo.CrispriMormino2022Dataset(root=str(tmp_path / "c"), genome=_genome())
    (mirror / "paper.md").unlink()
    with pytest.raises(
        RuntimeError,
        match=re.escape(
            f"required raw artifact missing from mirror: {mirror / 'paper.md'}"
        ),
    ):
        mo.CrispriMormino2022Dataset(root=str(tmp_path / "d"), genome=_genome())
