# tests/torchcell/datasets/pputida/test_lim2022.py
# [[tests.torchcell.datasets.pputida.test_lim2022]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/pputida/test_lim2022.py
"""Lim 2022 putidaPRECISE321 loader: pairing, strain calls, sources, environment, gate.

The synthetic tests build tiny frames and documents in ``tmp_path`` and need no data
root. The ``data`` tests read the real ``$DATA_ROOT``: the mirrored quotes, the raw
mirror's pins, and the built dev LMDB with its ledgers (they fail when the build is
absent; build it with ``python -m torchcell.database.build_dataset_lmdb --dataset
PutidaPrecise321Lim2022Dataset``).

Expected values of the data tests are the build's measured counts: 321 compendium
samples, 180 stored, 141 dropped under seven reasons; 291 count columns paired by name
and 30 by an in-house ``SBRG_*`` name.
"""

import json
import math
import os
import os.path as osp
import pickle
import zipfile
from pathlib import Path
from typing import Any

import lmdb
import pandas as pd
import pytest

from torchcell.data.experiment_dataset import file_sha256
from torchcell.datamodels.schema import (
    AssemblyReferenceGenome,
    BacterialRNASeqExpressionExperiment,
    BacterialRNASeqExpressionExperimentReference,
    EnvironmentPhysicalPerturbation,
    Genotype,
    PhysicalFactor,
)
from torchcell.datasets.pputida import lim2022 as m
from torchcell.verification.sourced import ProvenanceGapReason, audit_sourced_value

# --------------------------------------------------------------------------- #
# Readers
# --------------------------------------------------------------------------- #
_DOC = (
    '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/'
    'main"><w:body><w:p><w:r><w:t>2 g/L (NH</w:t></w:r><w:r><w:t>4</w:t></w:r>'
    "<w:r><w:t>)</w:t></w:r></w:p><w:p><w:r><w:t>second</w:t></w:r></w:p>"
    "</w:body></w:document>"
)


def test_docx_paragraphs_join_the_runs_of_each_paragraph(tmp_path: Path) -> None:
    """A subscript run is joined as plain text; paragraphs stay separate."""
    path = tmp_path / "doc.docx"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("word/document.xml", _DOC)
    assert m.docx_paragraphs(path) == ["2 g/L (NH4)", "second"]


def test_tpm_from_log_back_transforms_log2_tpm_plus_one() -> None:
    """2**X - 1 is returned when every column sums to one million."""
    log_tpm = pd.DataFrame({"s": [math.log2(x + 1.0) for x in (2.5e5, 7.5e5, 0.0)]})
    out = m.tpm_from_log(log_tpm)
    assert out["s"].tolist() == pytest.approx([2.5e5, 7.5e5, 0.0])


def test_tpm_from_log_refuses_a_log_without_the_pseudocount() -> None:
    """log2(TPM) back-transformed as log2(TPM + 1) misses the scale and raises."""
    log_tpm = pd.DataFrame({"s": [math.log2(x) for x in (2.5e5, 7.5e5)]})
    with pytest.raises(ValueError, match="do not sum to 1000000"):
        m.tpm_from_log(log_tpm)


def _sheet(n: int, *, conditions: int, projects: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Sample_name": [f"s{i}" for i in range(n)],
            "full_name": [f"c{i % conditions}" for i in range(n)],
            "project": [f"p{(i % conditions) % projects}" for i in range(n)],
            "putidaPRECISE321": [1.0] * n,
        }
    )


def test_compendium_samples_hold_the_paper_counts() -> None:
    """321 flagged rows over 118 conditions and 21 projects pass; other counts raise."""
    good = pd.concat(
        [
            _sheet(321, conditions=118, projects=21),
            pd.DataFrame({"Sample_name": ["x"], "putidaPRECISE321": [0.0]}),
        ]
    )
    assert len(m.compendium_samples(good)) == 321
    with pytest.raises(ValueError, match="the paper states"):
        m.compendium_samples(_sheet(320, conditions=118, projects=21))


# --------------------------------------------------------------------------- #
# Count pairing
# --------------------------------------------------------------------------- #
_GENES = ["g1", "g2", "g3", "g4", "g5"]
_LENGTHS: "pd.Series[float]" = pd.Series(
    [100.0, 200.0, 300.0, 400.0, 1000.0], index=_GENES
)


def _log_tpm(counts: list[int], lengths: "pd.Series[float]" = _LENGTHS) -> list[float]:
    rate = [c / n for c, n in zip(counts, lengths.tolist(), strict=True)]
    return [math.log2(r / sum(rate) * 1e6 + 1.0) for r in rate]


def test_pair_count_columns_takes_the_one_reproducing_column() -> None:
    """s1's same-named column is a decoy, so it pairs with c_true; s2 pairs by name.

    g5's length is wrong in the table handed in, so it is excluded from the comparison;
    the free per-sample scale absorbs the normalization it shifts.
    """
    true_s1 = [10, 0, 30, 400, 7]
    true_s2 = [5, 50, 0, 40, 9]
    log_tpm = pd.DataFrame(
        {"s1": _log_tpm(true_s1), "s2": _log_tpm(true_s2)}, index=_GENES
    )
    counts = pd.DataFrame(
        {"s1": [11, 0, 30, 400, 7], "c_true": true_s1, "s2": true_s2}, index=_GENES
    )
    lengths = _LENGTHS.copy()
    lengths["g5"] = 50.0
    pairing = m.pair_count_columns(log_tpm, counts, lengths, exclude=["g5"])
    assert pairing.pairs == {"s1": "c_true", "s2": "s2"}
    assert pairing.n_same_name == 1
    assert pairing.renamed == {"s1": "c_true"}
    assert set(pairing.same_name_rejected) == {"s1"}
    assert pairing.same_name_rejected["s1"] > 1e-3
    assert pairing.max_deviation < 1e-9
    assert pairing.n_genes_compared == 4
    assert pairing.excluded_genes == ["g5"]


def test_pair_count_columns_refuses_zero_or_two_reproducing_columns() -> None:
    """No match raises; two identical matching columns raise too."""
    true_s1 = [10, 20, 30, 40, 50]
    log_tpm = pd.DataFrame({"s1": _log_tpm(true_s1)}, index=_GENES)
    none = pd.DataFrame({"a": [1, 2, 3, 4, 6]}, index=_GENES)
    with pytest.raises(m.CountPairingError, match="0 count columns"):
        m.pair_count_columns(log_tpm, none, _LENGTHS)
    two = pd.DataFrame({"a": true_s1, "b": true_s1}, index=_GENES)
    with pytest.raises(m.CountPairingError, match="2 count columns"):
        m.pair_count_columns(log_tpm, two, _LENGTHS)


def test_length_disagreements_name_the_genes_whose_lengths_differ() -> None:
    """PP_1495-like case: the release table records a segment of the feature."""
    genome = pd.Series({"a": 100.0, "b": 1096.0})
    release = pd.Series({"a": 100.0, "b": 72.0})
    assert m.length_disagreements(genome, release) == ["b"]
    with pytest.raises(ValueError, match="lengths missing"):
        m.length_disagreements(genome, pd.Series({"a": 100.0}))


# --------------------------------------------------------------------------- #
# Strain calls
# --------------------------------------------------------------------------- #
def test_every_condition_has_exactly_one_strain_call() -> None:
    """56 derivative and 62 reference conditions, disjoint, 118 in all."""
    derivative = set(m.NON_REFERENCE_CONDITIONS)
    assert len(derivative) == 56
    assert len(m.REFERENCE_CONDITIONS) == 62
    assert not derivative & m.REFERENCE_CONDITIONS
    assert m.strain_call("Multistress:Control").kind is m.StrainKind.reference
    assert m.strain_call("RelA:Del_relA_lignin") == m.StrainCall(
        kind=m.StrainKind.deletion, deleted_symbols=("relA",)
    )
    assert m.strain_call("Crc:Del_crcZ_crcY").deleted_symbols == ("crcZ", "crcY")
    assert (
        m.strain_call("Muconate:GB045_glucose").drop_reason
        is m.DropReason.engineered_strain_code
    )
    with pytest.raises(m.UnclassifiedConditionError, match="no strain call"):
        m.strain_call("New:Del_xyz")


def test_dropped_calls_carry_a_reason_and_deletions_carry_symbols() -> None:
    """A dropped call has a reason and no symbols; a deletion has symbols and no reason."""
    for call in m.NON_REFERENCE_CONDITIONS.values():
        if call.kind is m.StrainKind.dropped:
            assert call.drop_reason is not None and not call.deleted_symbols
        else:
            assert call.kind is m.StrainKind.deletion
            assert call.drop_reason is None and call.deleted_symbols
    assert set(m.DROP_REASON_TEXT) == set(m.DropReason)


# --------------------------------------------------------------------------- #
# Sources and publications
# --------------------------------------------------------------------------- #
def _row(**cells: Any) -> "pd.Series[Any]":
    base: dict[str, Any] = {
        "project": "P",
        "DOI": math.nan,
        "PMID": math.nan,
        "BioProject": math.nan,
        "GEO Series": math.nan,
        "CenterName": "GEO",
        "Note": math.nan,
    }
    base.update(cells)
    row: pd.Series[Any] = pd.Series(base)
    return row


def test_record_publication_prefers_the_source_paper() -> None:
    """In-house -> Lim 2022; a DOI -> that DOI; a URL in 'DOI' + PMID -> the PMID."""
    in_house = m.source_study(
        _row(DOI="10.1/x", Note="in-house data, carbon metabolism engineering")
    )
    assert in_house.generated_in_this_study
    assert m.record_publication(in_house) == m.LIM2022_PUBLICATION

    doi = m.source_study(_row(DOI="10.1128/AEM.03236-16", PMID=28130298.0))
    assert doi.label == "doi:10.1128/AEM.03236-16"
    assert m.record_publication(doi).doi == "10.1128/AEM.03236-16"

    url = "https://www.ncbi.nlm.nih.gov/bioproject/269721"
    pmid = m.source_study(_row(DOI=url, PMID=25711694, BioProject="PRJNA269721"))
    assert pmid.doi is None and pmid.doi_column == url
    publication = m.record_publication(pmid)
    assert publication.pubmed_id == "25711694" and publication.doi is None

    two = m.source_study(_row(PMID="32267616;31860438"))
    assert two.pmid == "32267616"

    bare = m.source_study(_row(BioProject="PRJNA455036"))
    assert bare.label == "bioproject:PRJNA455036"
    assert m.record_publication(bare) == m.LIM2022_PUBLICATION
    assert m.source_study(_row()).label == "project:P"


def test_aggregation_summary_groups_in_house_by_project() -> None:
    """In-house samples group under their project; reprocessed ones under their source."""
    frame = pd.DataFrame(
        [
            dict(_row(project="A", Note="In-house data, x"), full_name="A:1"),
            dict(_row(project="A", Note="In-house data, x"), full_name="A:2"),
            dict(_row(project="B", DOI="10.2/y"), full_name="B:1"),
        ]
    )
    summary = m.aggregation_summary(frame)
    assert (summary.n_samples, summary.n_generated_in_this_study) == (3, 2)
    assert summary.n_reprocessed == 1
    assert [(s.source, s.n_samples, s.n_conditions) for s in summary.by_source] == [
        ("Lim 2022 in-house (A)", 2, 2),
        ("doi:10.2/y", 1, 1),
    ]
    assert summary.by_source[1].doi_cells == ["10.2/y"]


def test_samples_without_a_named_publication_group_by_project() -> None:
    """One BioProject per sample still makes one study (the compendium project)."""
    frame = pd.DataFrame(
        [
            dict(_row(project="Fuel", BioProject=f"PRJNA45503{i}"), full_name="Fuel:x")
            for i in range(3)
        ]
    )
    (only,) = m.aggregation_summary(frame).by_source
    assert only.source == "no publication in the sheet (Fuel)"
    assert only.bioprojects == ["PRJNA455030", "PRJNA455031", "PRJNA455032"]


# --------------------------------------------------------------------------- #
# Environment and phenotype
# --------------------------------------------------------------------------- #
def test_aromatic_environment_is_the_stated_m9_plus_its_carbon_source() -> None:
    """2.5 g/L of one stated carbon source; the base medium carries none."""
    env = m.aromatic_environment("Aromatic:Coumarate")
    assert env.media is m.LIM2022_M9_NO_CARBON
    names = {c.compound.name for c in env.media.components}
    assert "D-glucose" not in names and "ammonium sulfate" in names
    assert [d.name for d in env.media.dropouts] == ["ammonium chloride"]
    (carbon,) = env.perturbations
    assert isinstance(carbon, EnvironmentPhysicalPerturbation)
    assert carbon.factor is PhysicalFactor.carbon_source
    assert carbon.agent is not None and carbon.agent.name == "p-coumaric acid"
    assert carbon.magnitude is not None and carbon.magnitude.value == 2.5
    assert env.temperature is None
    assert [g.field for g in env.provenance_gaps] == ["temperature"]


def test_aromatic_mixture_gaps_the_unstated_split() -> None:
    """Coumarate + ferulate: two carbon sources, each with a gapped magnitude."""
    env = m.aromatic_environment("Aromatic:Coumarate+ferulate")
    perturbations = [
        p for p in env.perturbations if isinstance(p, EnvironmentPhysicalPerturbation)
    ]
    assert len(perturbations) == 2
    agents = sorted(p.agent.name for p in perturbations if p.agent is not None)
    assert agents == ["ferulic acid", "p-coumaric acid"]
    for perturbation in perturbations:
        assert perturbation.magnitude is None
        assert [g.field for g in perturbation.provenance_gaps] == ["magnitude"]


def test_deferred_environments_are_distinct_per_condition_and_name_the_source() -> None:
    """Two conditions of one study never share a placeholder medium."""
    source = m.source_study(_row(DOI="10.1128/AEM.03236-16"))
    one = m.deferred_environment("Multistress:NaCl_7min", [source])
    two = m.deferred_environment("Multistress:Control", [source])
    assert one.media != two.media
    (component,) = one.media.components
    assert component.note is not None and "doi:10.1128/AEM.03236-16" in component.note
    (gap,) = one.provenance_gaps
    assert gap.reason is ProvenanceGapReason.not_carried_by_curation

    muconate = m.source_study(_row(project="Muconate", Note="in-house data, x"))
    env = m.deferred_environment("Muconate:KT2440_fructose", [muconate, muconate])
    assert "Bentley 2020" in (env.media.components[0].note or "")
    assert env.media.provenance == [m.MUCONATE_DEFERRAL]
    assert env.provenance_gaps[0].reason is ProvenanceGapReason.not_reported_by_primary
    with pytest.raises(ValueError, match="mixes in-house and reprocessed"):
        m.deferred_environment("Muconate:KT2440_glucose", [muconate, source])


def test_deferred_environment_names_every_source_of_a_split_condition() -> None:
    """A condition deposited as one BioProject per sample names all of them."""
    sources = [
        m.source_study(_row(project="Fuel", BioProject=b))
        for b in ("PRJNA455031", "PRJNA455030")
    ]
    env = m.deferred_environment("Fuel:Butanol", sources)
    note = env.media.components[0].note or ""
    assert "bioproject:PRJNA455030, bioproject:PRJNA455031" in note


def test_reference_phenotype_is_the_log_mean_level() -> None:
    """TPM is 2**mean(X) - 1 (the centering level); counts are the rounded mean."""
    log_tpm = pd.DataFrame({"a": [1.0, 3.0], "b": [3.0, 5.0]}, index=["g1", "g2"])
    counts = pd.DataFrame({"a": [1, 10], "b": [2, 13]}, index=["g1", "g2"])
    phenotype = m.reference_phenotype(log_tpm, counts)
    assert dict(phenotype.expression_tpm) == {"g1": 3.0, "g2": 15.0}
    assert dict(phenotype.expression_count) == {"g1": 2, "g2": 12}
    assert phenotype.measurement_type == "rnaseq_tpm"
    assert [g.field for g in phenotype.provenance_gaps] == ["n_mapped_reads"]


# --------------------------------------------------------------------------- #
# Ledgers
# --------------------------------------------------------------------------- #
def _sample(name: str, full_name: str, reason: m.DropReason | None) -> m.SampleRecord:
    source = m.source_study(_row())
    return m.SampleRecord(
        index=None if reason else 0,
        sample_name=name,
        srx=None,
        project="P",
        condition=full_name.split(":")[1],
        full_name=full_name,
        rep_name=1,
        reference_condition="c",
        source=source,
        publication=m.record_publication(source),
        count_column=name,
        strain=m.StrainKind.dropped if reason else m.StrainKind.reference,
        deleted_genes=[],
        drop_reason=reason,
    )


def test_drop_log_tallies_every_reason_and_its_conditions() -> None:
    """Stored rows count as stored; dropped rows group by reason, conditions sorted."""
    rows = [
        _sample("a", "P:x", None),
        _sample("b", "P:y", m.DropReason.plasmid_content),
        _sample("c", "P:z", m.DropReason.plasmid_content),
        _sample("d", "P:w", m.DropReason.evolved_isolate),
    ]
    log = m.drop_log(rows)
    assert (log.n_compendium, log.n_stored) == (4, 1)
    assert [(r.reason, r.n_samples, r.conditions) for r in log.rules] == [
        (m.DropReason.evolved_isolate, 1, ["P:w"]),
        (m.DropReason.plasmid_content, 2, ["P:y", "P:z"]),
    ]


# --------------------------------------------------------------------------- #
# Raw mirror
# --------------------------------------------------------------------------- #
def test_deposit_raw_mirror_is_idempotent_and_refuses_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A second deposit leaves the files alone; an edited mirror file is refused."""
    xlsx = tmp_path / "in" / "si2.xlsx"
    counts = tmp_path / "in" / "counts.csv"
    xlsx.parent.mkdir()
    xlsx.write_bytes(b"workbook")
    counts.write_bytes(b"Geneid,s\nPP_0001,1\n")
    monkeypatch.setattr(m, "SI_XLSX_SHA256", file_sha256(xlsx))
    monkeypatch.setattr(m, "COUNTS_SHA256", file_sha256(counts))
    data_root = tmp_path / "root"
    root = m.deposit_raw_mirror(
        si_xlsx_path=xlsx, counts_path=counts, data_root=str(data_root)
    )
    assert root == data_root / m.RAW_DIR_REL
    m.deposit_raw_mirror(
        si_xlsx_path=xlsx, counts_path=counts, data_root=str(data_root)
    )
    manifest = m.load_manifest(str(data_root))
    assert [f.path for f in manifest.files] == [m.SI_XLSX_REL, m.COUNTS_REL]
    retrieval = manifest.files[1].retrieval
    assert retrieval is not None and retrieval.params == {"url": m.COUNTS_URL}
    assert m.GITHUB_COMMIT in m.COUNTS_URL
    (root / m.COUNTS_REL).write_bytes(b"edited")
    with pytest.raises(RuntimeError, match="different sha256"):
        m.deposit_raw_mirror(
            si_xlsx_path=xlsx, counts_path=counts, data_root=str(data_root)
        )
    other = tmp_path / "in" / "other.csv"
    other.write_bytes(b"x")
    with pytest.raises(RuntimeError, match="sha256 mismatch"):
        m.deposit_raw_mirror(
            si_xlsx_path=xlsx, counts_path=other, data_root=str(data_root)
        )


# --------------------------------------------------------------------------- #
# Verification gate on synthetic records
# --------------------------------------------------------------------------- #
_PIN = AssemblyReferenceGenome(
    species="Pseudomonas putida",
    strain="KT2440",
    assembly_set="pputida_KT2440_ASM756v2",
    assembly_accession="GCA_000007565.2",
)


def _record(tpm: dict[str, float], counts: dict[str, int]) -> dict[str, Any]:
    env = m.aromatic_environment("Aromatic:Glucose")
    experiment = BacterialRNASeqExpressionExperiment(
        dataset_name="d",
        genotype=Genotype(perturbations=[]),
        environment=env,
        phenotype=m.expression_phenotype(tpm, counts),
    )
    reference = BacterialRNASeqExpressionExperimentReference(
        dataset_name="d",
        genome_reference=_PIN,
        environment_reference=env,
        phenotype_reference=m.expression_phenotype(tpm, counts),
    )
    return {
        "experiment": experiment.model_dump(),
        "reference": reference.model_dump(),
        "publication": m.LIM2022_PUBLICATION.model_dump(),
    }


def test_verify_records_passes_distinct_scaled_profiles() -> None:
    """Two distinct profiles on the KT2440 pin, each summing to 1e6 TPM, pass L0-L4."""
    records = [
        _record({"PP_0001": 4e5, "PP_0002": 6e5}, {"PP_0001": 4, "PP_0002": 6}),
        _record({"PP_0001": 5e5, "PP_0002": 5e5}, {"PP_0001": 5, "PP_0002": 5}),
    ]
    report = m.verify_records(
        records, expected_count=2, gene_universe={"PP_0001", "PP_0002"}
    )
    assert report.passed, report.summary()


def test_verify_records_fails_a_repeated_profile_and_an_unknown_gene() -> None:
    """A duplicated profile fails L1; a gene outside the universe fails L4."""
    record = _record({"PP_0001": 4e5, "PP_0002": 6e5}, {"PP_0001": 4, "PP_0002": 6})
    report = m.verify_records(
        [record, record], expected_count=2, gene_universe={"PP_0001"}
    )
    failed = {r.name for r in report.results if not r.passed}
    assert failed == {"sample_uniqueness", "gene_containment_kt2440"}


# --------------------------------------------------------------------------- #
# Data-gated: the mirrors and the built dev LMDB
# --------------------------------------------------------------------------- #
def _library() -> Path:
    return Path(os.environ["DATA_ROOT"]) / "torchcell-library"


@pytest.mark.data
def test_every_sourced_value_is_verbatim_in_the_mirror() -> None:
    """paper.md quotes pass the audit; docx quotes are in one rendered paragraph."""
    library = _library()
    docx = library / m.CITATION_KEY / m.SI_DOCX
    assert file_sha256(docx) == m.SI_DOCX_SHA256
    paragraphs = m.docx_paragraphs(docx)
    for value in m.SOURCED_VALUES:
        if value.provenance.source_uri == m.SI_DOCX:
            assert value.provenance.sha256 == m.SI_DOCX_SHA256
            assert any(value.quote in p for p in paragraphs), value.quote[:60]
        else:
            result = audit_sourced_value(value, library)
            assert result.passed, (result.message, value.quote[:60])


@pytest.mark.data
def test_raw_mirror_manifest_carries_the_module_pins() -> None:
    """Both mirror files hash to the module pins the manifest records."""
    manifest = m.load_manifest()
    for relpath, pin in (
        (m.SI_XLSX_REL, m.SI_XLSX_SHA256),
        (m.COUNTS_REL, m.COUNTS_SHA256),
    ):
        assert m.manifest_sha256(manifest, relpath) == pin
        assert file_sha256(m.raw_mirror_dir() / relpath) == pin
    library_manifest = json.loads(
        (_library() / m.CITATION_KEY / "manifest.json").read_text()
    )
    (si,) = [f for f in library_manifest["files"] if f["path"] == m.SI_XLSX_REL]
    assert si["sha256"] == m.SI_XLSX_SHA256


def _build_root() -> str:
    return osp.join(os.environ["DATA_ROOT"], "data/torchcell/putida_precise321_lim2022")


def _ledger(name: str) -> Any:
    with open(osp.join(_build_root(), "preprocess", name)) as handle:
        return json.load(handle)


@pytest.mark.data
def test_build_report_states_the_measured_aggregation_and_drops() -> None:
    """321 samples: 96 in-house, 225 reprocessed; 180 stored; drops by reason."""
    report = _ledger("build_report.json")
    aggregation = report["aggregation"]
    assert (
        aggregation["n_samples"],
        aggregation["n_generated_in_this_study"],
        aggregation["n_reprocessed"],
    ) == (321, 96, 225)
    drops = {r["reason"]: r["n_samples"] for r in report["drops"]["rules"]}
    assert drops == {
        "deleted_gene_unresolved": 11,
        "engineered_evolved_sugar_strain": 14,
        "engineered_strain_code": 71,
        "evolved_isolate": 4,
        "non_reference_background": 24,
        "plasmid_content": 15,
        "sequence_variant_allele": 2,
    }
    assert report["drops"]["n_stored"] == 180
    pairing = report["count_pairing"]
    assert (pairing["n_same_name"], len(pairing["renamed"])) == (291, 30)
    assert len(pairing["same_name_rejected"]) == 16
    assert pairing["excluded_genes"] == ["PP_1495"]
    assert pairing["max_deviation"] < m.PAIRING_TOLERANCE
    assert report["deletion_loci"] == {"fleQ": "PP_4373", "relA": "PP_1656"}
    genes = report["expression_genes"]
    assert genes["unique_names"] == 5564
    assert genes["status_histogram"]["current"] == 5564


@pytest.mark.data
def test_built_records_pass_the_gate_and_carry_their_sources() -> None:
    """180 records pass L0-L4; ledger indices, publications and deletions agree."""
    from torchcell.verification.runners import _gene_set_for_reference, load_records

    records = load_records(_build_root())
    universe = _gene_set_for_reference(
        records[0]["reference"]["genome_reference"], os.environ["DATA_ROOT"]
    )
    report = m.verify_records(records, expected_count=180, gene_universe=universe)
    assert report.passed, report.summary()

    ledger = [row for row in _ledger("sample_ledger.json") if row["index"] is not None]
    assert sorted(row["index"] for row in ledger) == list(range(180))
    by_index = {row["index"]: row for row in ledger}
    deletions: dict[str, int] = {}
    env = lmdb.open(
        osp.join(_build_root(), "processed", "lmdb"), readonly=True, lock=False
    )
    with env.begin() as txn:
        by_key = {int(key): pickle.loads(value) for key, value in txn.cursor()}
    env.close()
    assert sorted(by_key) == list(range(180))
    for index, record in by_key.items():
        row = by_index[index]
        assert record["publication"] == row["publication"]
        tags = [
            p["systematic_gene_name"]
            for p in record["experiment"]["genotype"]["perturbations"]
        ]
        assert tags == row["deleted_genes"]
        for tag in tags:
            deletions[tag] = deletions.get(tag, 0) + 1
        total = sum(record["experiment"]["phenotype"]["expression_tpm"].values())
        assert math.isclose(total, 1e6, rel_tol=1e-6)
    assert deletions == {"PP_1656": 6, "PP_4373": 3}
