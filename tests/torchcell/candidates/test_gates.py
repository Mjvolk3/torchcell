# tests/torchcell/candidates/test_gates.py
# [[tests.torchcell.candidates.test_gates]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/candidates/test_gates.py
"""The deterministic gates on fixture rows, a fixture genomes tier, mirrors and a store.

G2 covers the plan's four outcomes on MG1655 (readable), W3110 (resolvable, not
ingestible), a named host with no set (BL21, provisionable), an unnamed host, and a Bloom
2019-shaped mosaic. G3 reads a concatenated-archive fixture. G4-value runs on a
Thompson-shaped fixture (one released column, the right sample and a wrong one).
"""

from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict

from tests.torchcell.candidates._builders import (
    COMMIT,
    aggregation,
    evidence,
    fitness_record,
    sha,
    write_assembly_set,
    write_mirror,
    write_store,
    xlsx_bytes,
)
from torchcell.candidates import gates
from torchcell.candidates.gates import CandidateRow, G1Run, GateRun, SourceKindFinding
from torchcell.candidates.verdict import (
    G2Record,
    G4KeyRecord,
    G4ValueRecord,
    GateResult,
)
from torchcell.sequence.genome.registry import (
    ECOLI_K12_BW25113,
    ECOLI_K12_MG1655,
    ECOLI_K12_W3110,
    PETER2018_1011,
    SGD_S288C_R64,
)


class FakeRow(BaseModel):
    """Stands in for a table script's Candidate: only model_dump is read."""

    model_config = ConfigDict(extra="allow")


BACTERIA_FIELDS: dict[str, Any] = {
    "name": "Fixture 2020",
    "organism": "E. coli",
    "citation": "Fixture A. A screen. J 2020",
    "url": "https://doi.org/10.1000/fixture",
    "klass": "Transposon fitness",
    "genotypes": "100 mutants",
    "env": "1 condition",
    "phenotype": "fitness",
    "seq_basis": "K-12+transposon",
    "modality": "transposon insertion",
    "why": "a screen",
    "accession": "GSE123 (supplement)",
    "accession_confirmed": True,
    "status": "candidate",
    "schema_need": "",
}


def bacteria_row(**overrides: Any) -> CandidateRow:
    """A bacteria-table row through row_from_table."""
    return gates.row_from_table("bacteria", FakeRow(**{**BACTERIA_FIELDS, **overrides}))


def yeast_row(**overrides: Any) -> CandidateRow:
    """A yeast-table row through row_from_table (no organism, no accession flag)."""
    fields = {
        "name": "Yeast 2019",
        "citation": "Y. A cross. J 2019",
        "url": "https://doi.org/10.1000/yeast",
        "klass": "Natural variation",
        "genotypes": "segregants",
        "env": "38 conditions",
        "phenotype": "growth",
        "shape": "scalar",
        "seq_basis": "segregant-WGS",
        "why": "a cross",
        "accession": "",
        "status": "candidate",
    }
    return gates.row_from_table("yeast", FakeRow(**{**fields, **overrides}))


# --------------------------------------------------------------------------- #
# Rows
# --------------------------------------------------------------------------- #
def test_row_from_table_projects_both_tables() -> None:
    """Bacteria keep organism and the accession flag; yeast rows get neither."""
    b = bacteria_row()
    assert (b.organism, b.accession_confirmed, b.doi) == (
        "E. coli",
        True,
        "10.1000/fixture",
    )
    assert b.text.splitlines()[0] == "Fixture A. A screen. J 2020"
    y = yeast_row()
    assert (y.organism, y.accession_confirmed, y.doi) == (
        "S. cerevisiae",
        None,
        "10.1000/yeast",
    )
    assert bacteria_row(url="https://zenodo.org/records/1").doi is None


def test_find_row_exact_then_unique_prefix() -> None:
    """An exact name wins; a prefix must match one row; zero or two raise."""
    rows = [
        bacteria_row(name="CeCaFDB flux compendium"),
        bacteria_row(name="Lim 2022 putidaPRECISE321"),
        bacteria_row(name="Lim 2025 isoprenol TALE"),
    ]
    assert gates.find_row(rows, "CeCaFDB").name == "CeCaFDB flux compendium"
    assert (
        gates.find_row(rows, "Lim 2022 putidaPRECISE321").name
        == "Lim 2022 putidaPRECISE321"
    )
    with pytest.raises(LookupError, match="2 prefix matches"):
        gates.find_row(rows, "Lim")
    with pytest.raises(LookupError, match="0 prefix matches"):
        gates.find_row(rows, "Nope")


# --------------------------------------------------------------------------- #
# G1
# --------------------------------------------------------------------------- #
def finding(kind: str, row: str = "Fixture 2020") -> SourceKindFinding:
    """A finding of the given kind (an aggregation carries its record)."""
    return SourceKindFinding(
        row_name=row,
        citation_key="fixtureKey2020",
        source_kind=kind,  # type: ignore[arg-type]
        reason="read",
        evidence=(evidence(),),
        aggregation=aggregation(4) if kind == "aggregation" else None,
    )


@pytest.mark.parametrize(
    ("kind", "outcome"),
    [
        ("primary", "pass"),
        ("aggregation", "pass"),
        ("transcription", "fail"),
        ("prediction", "fail"),
    ],
)
def test_g1_reads_a_recorded_finding(kind: str, outcome: str) -> None:
    """A finding decides G1 and its quotes become the gate's evidence."""
    run = gates.evaluate_g1(bacteria_row(status="aggregation"), finding(kind))
    assert (run.result.outcome, run.source_kind) == (outcome, kind)
    assert run.result.reason == f"{kind}: read"
    assert run.result.evidence == (evidence(),)
    assert (run.aggregation is not None) == (kind == "aggregation")


def test_g1_finding_must_be_for_the_row() -> None:
    """A finding filed under another row is refused."""
    with pytest.raises(
        ValueError, match="finding for 'Other' given for 'Fixture 2020'"
    ):
        gates.evaluate_g1(bacteria_row(), finding("primary", row="Other"))


def test_finding_aggregation_record_iff_aggregation() -> None:
    """A finding validates the record against its kind."""
    with pytest.raises(ValueError, match="required for, and only for"):
        SourceKindFinding(
            row_name="r",
            citation_key="k",
            source_kind="aggregation",
            reason="x",
            evidence=(evidence(),),
            aggregation=None,
        )


@pytest.mark.parametrize(
    ("overrides", "outcome", "kind", "phrase"),
    [
        ({"status": "aggregation"}, "unmeasured", None, "agentic G1 judgment"),
        ({"klass": "Aggregation / support"}, "unmeasured", None, "agentic G1 judgment"),
        ({"accession_confirmed": False}, "unmeasured", None, "not established"),
        ({}, "pass", "primary", "no aggregation marker"),
    ],
)
def test_g1_without_a_finding(
    overrides: dict[str, Any], outcome: str, kind: str | None, phrase: str
) -> None:
    """Off row fields alone: a marked aggregation or unconfirmed accession is unmeasured."""
    run = gates.evaluate_g1(bacteria_row(**overrides), None)
    assert (run.result.outcome, run.source_kind) == (outcome, kind)
    assert phrase in run.result.reason


def test_g1_yeast_row_is_primary_without_the_bacteria_flag() -> None:
    """The yeast table has no accession flag, so only the aggregation marker matters."""
    assert gates.evaluate_g1(yeast_row(), None).source_kind == "primary"


# --------------------------------------------------------------------------- #
# G2
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("text", "seq_basis", "organism", "expected"),
    [
        ("BW25113 Keio deletions", "K-12-KO", "E. coli", ("BW25113",)),
        ("the Keio collection", "K-12-KO", "E. coli", ("BW25113",)),
        ("a K-12 screen", "K-12-KO", "E. coli", ("MG1655", "BW25113")),
        ("MG1655 and BL21(DE3)", "engineered-chassis", "E. coli", ("MG1655", "BL21")),
        ("wild type", "KT2440-KO", "P. putida", ("KT2440",)),
        ("wild type", "reference-only", "E. coli", ()),
    ],
)
def test_hosts_named(
    text: str, seq_basis: str, organism: str, expected: tuple[str, ...]
) -> None:
    """Strain tokens in the row text (seq_basis included), K-12 alone -> both K-12 sets."""
    row = bacteria_row(
        why=text, seq_basis=seq_basis, organism=organism, citation="c", genotypes="g"
    )
    assert gates.hosts_named(row) == expected


def test_hosts_named_yeast_and_unknown_organism() -> None:
    """BY47xx is S288C; 1,011 is the Peter panel; an unknown organism raises."""
    row = yeast_row(why="BY4741 x the 1,011 isolates", genotypes="g")
    assert gates.hosts_named(row) == ("S288C", "Peter 2018 isolates")
    weird = bacteria_row().model_copy(update={"organism": "B. subtilis"})
    with pytest.raises(ValueError, match="unknown organism 'B. subtilis'"):
        gates.hosts_named(weird)


@pytest.fixture
def tier(tmp_path: Path) -> Path:
    """A genomes tier holding MG1655, BW25113, W3110, S288C and the Peter panel."""
    for assembly_set in (
        ECOLI_K12_MG1655,
        ECOLI_K12_BW25113,
        ECOLI_K12_W3110,
        SGD_S288C_R64,
        PETER2018_1011,
    ):
        write_assembly_set(tmp_path, assembly_set)
    return tmp_path


@pytest.mark.parametrize(
    ("why", "outcome", "kind", "g2"),
    [
        ("MG1655 screen", "pass", None, "resolvable_readable"),
        ("W3110 screen", "blocked", "ingest", "resolvable_not_ingestible"),
        ("BL21 production", "blocked", "provision", "absent_provisionable"),
        ("REL606 lines", "blocked", "provision", "absent_provisionable"),
        ("no strain at all", "fail", None, "absent_unnamed"),
    ],
)
def test_g2_outcomes(
    tier: Path, why: str, outcome: str, kind: str | None, g2: str
) -> None:
    """Readable, not ingestible, provisionable (no set / set not in tier), unnamed."""
    row = bacteria_row(why=why, seq_basis="reference-only", citation="c", genotypes="g")
    run = gates.evaluate_g2(row, str(tier), source_kind="primary")
    assert (run.result.outcome, run.result.blocked_kind) == (outcome, kind)
    assert run.record.outcome == g2


def test_g2_reason_names_each_host(tier: Path) -> None:
    """The reason states what the tier said about every host."""
    row = bacteria_row(why="MG1655 and W3110", seq_basis="reference-only", citation="c")
    run = gates.evaluate_g2(row, str(tier), source_kind="primary")
    assert run.result.reason == (
        "assembly_set: MG1655 -> ecoli_K12_MG1655_ASM584v2 resolves and reads; "
        "W3110 -> ecoli_K12_W3110_ASM1024v1 resolves but no genome class reads it"
    )


def test_g2_unnamed_aggregation_is_unmeasured(tier: Path) -> None:
    """An aggregation with no host is resolved per source study, not refused."""
    row = bacteria_row(why="many studies", seq_basis="reference-only", citation="c")
    run = gates.evaluate_g2(row, str(tier), source_kind="aggregation")
    assert (run.result.outcome, run.record) == ("unmeasured", None)


def test_g2_bloom_mosaic(tier: Path) -> None:
    """Both parents resolve and read -> pass; one named parent -> absent_unnamed."""
    both = yeast_row(why="BY4716 x the 1,011 panel", genotypes="g")
    run = gates.evaluate_g2(both, str(tier), source_kind="primary", verify=True)
    assert (run.result.outcome, run.record.branch, run.record.outcome) == (
        "pass",
        "haplotype_mosaic",
        "resolvable_readable",
    )
    assert [h.assembly_set for h in run.record.hosts] == [SGD_S288C_R64, PETER2018_1011]
    one = yeast_row(why="BY4741 only", genotypes="g")
    run = gates.evaluate_g2(one, str(tier), source_kind="primary")
    assert (run.result.outcome, run.record.outcome) == ("fail", "absent_unnamed")


def test_probe_host_detects_tampered_bytes(tier: Path) -> None:
    """resolve() re-hashes the tier, so altered bytes raise."""
    (tier / "torchcell-genomes" / ECOLI_K12_MG1655 / "genome.fna").write_bytes(b"x")
    with pytest.raises(RuntimeError, match="sha256"):
        gates.probe_host("MG1655", str(tier))
    assert gates.probe_host("MG1655", str(tier), verify=False).readable is True


# --------------------------------------------------------------------------- #
# G3
# --------------------------------------------------------------------------- #
def test_leading_archive_of_a_concatenated_file() -> None:
    """The leading archive is the first zip; a file with no EOCD raises."""
    first = xlsx_bytes({"S1": [["a"], [1]]})
    assert gates.leading_archive(first + first) == first
    with pytest.raises(ValueError, match="no zip end-of-central-directory"):
        gates.leading_archive(b"plain text")


def test_inventory_of_a_concatenated_archive(tmp_path: Path) -> None:
    """Full-bytes sha, leading-archive sha, testzip and openpyxl row counts."""
    book = xlsx_bytes(
        {"S1A": [["gene", "fit"], ["b0001", 0.1], ["b0002", 0.2]], "S1B": [["x"]]}
    )
    concatenated = book + book
    write_mirror(
        tmp_path,
        "torchcell-raw",
        "tongKey2020",
        {"data/t.xlsx": concatenated, "paper/p.txt": b"text"},
    )
    record = gates.inventory("tongKey2020", str(tmp_path))
    assert record is not None
    assert record.trees_read == ("raw",)
    (only,) = record.files
    assert (only.path, only.sha256_disk, only.leading_archive_sha256) == (
        "data/t.xlsx",
        sha(concatenated),
        sha(book),
    )
    assert (only.is_zip, only.zip_ok, only.intact) == (True, True, True)
    assert [(s.sheet, s.rows) for s in only.sheets] == [("S1A", 3), ("S1B", 1)]
    run = gates.evaluate_g3(record)
    assert (run.result.outcome, run.result.reason) == (
        "pass",
        "1 file(s) in raw, 4 worksheet row(s) counted; all intact",
    )


def test_inventory_plain_file_has_no_zip_fields(tmp_path: Path) -> None:
    """A CSV is hashed and nothing else."""
    write_mirror(tmp_path, "torchcell-library", "k", {"si/a.csv": b"a,b\n"})
    record = gates.inventory("k", str(tmp_path))
    assert record is not None
    (only,) = record.files
    assert (
        only.tree,
        only.is_zip,
        only.zip_ok,
        only.leading_archive_sha256,
        only.sheets,
    ) == ("library", False, None, None, ())


def test_g3_outcomes(tmp_path: Path) -> None:
    """None -> unmeasured; drift -> fail; a wall -> blocked; missing script -> deposit."""
    assert gates.evaluate_g3(None).result.outcome == "unmeasured"
    write_mirror(
        tmp_path,
        "torchcell-raw",
        "drift",
        {"data/a.csv": b"x"},
        pinned={"data/a.csv": "0" * 64},
    )
    drift = gates.evaluate_g3(gates.inventory("drift", str(tmp_path)))
    assert (drift.result.outcome, drift.result.reason) == (
        "fail",
        "1 file(s) in raw, 0 worksheet row(s) counted; sha256 drift or a bad zip "
        "member: raw:data/a.csv",
    )
    write_mirror(
        tmp_path, "torchcell-raw", "wall", {"si/s1.xlsx": None}, manual=("si/s1.xlsx",)
    )
    wall = gates.evaluate_g3(gates.inventory("wall", str(tmp_path)))
    assert (wall.result.outcome, wall.result.blocked_kind) == ("blocked", "wall")
    assert wall.record.walls == ("raw:si/s1.xlsx: {'recipe': 'fetch si/s1.xlsx'}",)
    write_mirror(tmp_path, "torchcell-raw", "gone", {"data/a.csv": None})
    gone = gates.evaluate_g3(gates.inventory("gone", str(tmp_path)))
    assert (gone.result.outcome, gone.result.blocked_kind) == ("blocked", "deposit")
    write_mirror(tmp_path, "torchcell-library", "paperonly", {"paper.md": b"x"})
    empty = gates.evaluate_g3(gates.inventory("paperonly", str(tmp_path)))
    assert (empty.result.outcome, empty.record) == ("unmeasured", None)


def test_g3_bad_zip_fails(tmp_path: Path) -> None:
    """A zip whose member CRC is wrong reads as zip_ok False."""
    book = bytearray(xlsx_bytes({"S": [["a"]]}))
    # Flip a byte inside the first member's compressed data (after its 30-byte header
    # and file name), leaving the central directory intact.
    name_length = int.from_bytes(book[26:28], "little")
    book[30 + name_length + 2] ^= 0xFF
    write_mirror(tmp_path, "torchcell-raw", "bad", {"data/b.xlsx": bytes(book)})
    record = gates.inventory("bad", str(tmp_path))
    assert record is not None
    assert (record.files[0].zip_ok, record.files[0].sheets) == (False, ())
    assert gates.evaluate_g3(record).result.outcome == "fail"


def test_truncated_and_unreadable_archives(tmp_path: Path) -> None:
    """A zip header with no EOCD, and an EOCD pointing at no directory, are both bad."""
    book = xlsx_bytes({"S": [["a"]]})
    write_mirror(tmp_path, "torchcell-raw", "cut", {"data/c.xlsx": book[:100]})
    record = gates.inventory("cut", str(tmp_path))
    assert record is not None
    assert (record.files[0].is_zip, record.files[0].zip_ok) == (True, False)
    eocd = (
        b"PK\x05\x06"
        + (0).to_bytes(4, "little")
        + (1).to_bytes(2, "little") * 2
        + (46).to_bytes(4, "little")
        + (0).to_bytes(4, "little")
        + b"\x00\x00"
    )
    assert gates._zip_ok(b"junk" + eocd) is False


# --------------------------------------------------------------------------- #
# G4-key
# --------------------------------------------------------------------------- #
@pytest.fixture
def index(tmp_path: Path) -> tuple[gates.ModuleKeys, ...]:
    """Two dataset modules: the candidate's own, and a compendium that mentions it."""
    datasets = tmp_path / "torchcell" / "datasets" / "ecoli"
    datasets.mkdir(parents=True)
    (datasets / "own.py").write_text(
        'CITATION_KEY = "fixtureKey2020"\nDOI: str = "10.1000/fixture"\n'
    )
    (datasets / "compendium.py").write_text(
        'CITATION_KEY = "compendiumKey2024"\nPAPER_DOI = "10.1000/compendium"\n'
        '"""Samples of 10.1000/fixture are in here, and GSE999."""\n'
    )
    return gates.module_index(tmp_path / "torchcell" / "datasets")


def test_module_index_reads_constants_without_importing(
    index: tuple[gates.ModuleKeys, ...],
) -> None:
    """Paths, keys and own DOIs, sorted by path."""
    assert [(m.path, m.citation_key, m.own_dois) for m in index] == [
        (
            "torchcell/datasets/ecoli/compendium.py",
            "compendiumKey2024",
            ("10.1000/compendium",),
        ),
        ("torchcell/datasets/ecoli/own.py", "fixtureKey2020", ("10.1000/fixture",)),
    ]


def test_citation_key_for_doi(
    index: tuple[gates.ModuleKeys, ...], tmp_path: Path
) -> None:
    """A module's own DOI first, then a mirror manifest, else None."""
    assert (
        gates.citation_key_for_doi("10.1000/fixture", index, None) == "fixtureKey2020"
    )
    root = tmp_path / "root"
    write_mirror(
        root, "torchcell-raw", "rawKey2019", {"data/a.csv": b"x"}, doi="10.1000/RAW"
    )
    assert gates.citation_key_for_doi("10.1000/raw", index, str(root)) == "rawKey2019"
    assert gates.citation_key_for_doi("10.1000/none", index, str(root)) is None
    assert gates.citation_key_for_doi("10.1000/none", index, None) is None
    twice = (
        *index,
        index[1].model_copy(update={"path": "x.py", "citation_key": "otherKey"}),
    )
    with pytest.raises(LookupError, match="modules disagree"):
        gates.citation_key_for_doi("10.1000/fixture", twice, None)


def test_g4_key_own_module_and_mention_pass(
    index: tuple[gates.ModuleKeys, ...],
) -> None:
    """The candidate's own module is not an overlap; a mention is recorded, not failed."""
    run = gates.evaluate_g4_key(
        bacteria_row(accession="none"),
        "fixtureKey2020",
        index,
        doi_to_pmid=None,
        aggregator_pmids=None,
    )
    assert [(h.where.split("/")[-1], h.relation) for h in run.record.hits] == [
        ("compendium.py", "mentioned"),
        ("own.py", "own"),
    ]
    assert (run.result.outcome, run.result.reason) == (
        "pass",
        "no key held by another module (1 mention(s)); no PMID check; values not "
        "compared (no dev store named)",
    )


def test_g4_key_fails_on_another_modules_doi_or_accession(
    index: tuple[gates.ModuleKeys, ...],
) -> None:
    """Another module's primary DOI, or a shared accession, is already held."""
    by_doi = gates.evaluate_g4_key(
        bacteria_row(url="https://doi.org/10.1000/compendium", accession="none"),
        "fixtureKey2020",
        index,
        doi_to_pmid=None,
        aggregator_pmids=None,
    )
    assert (by_doi.result.outcome, by_doi.result.reason) == (
        "fail",
        "already held: doi 10.1000/compendium in torchcell/datasets/ecoli/compendium.py",
    )
    by_accession = gates.evaluate_g4_key(
        bacteria_row(accession="GEO GSE999"),
        "fixtureKey2020",
        index,
        doi_to_pmid=None,
        aggregator_pmids=None,
    )
    assert by_accession.record.accessions == ("GSE999",)
    assert by_accession.result.outcome == "fail"


def test_g4_key_pmid_against_aggregators(index: tuple[gates.ModuleKeys, ...]) -> None:
    """A PMID an aggregator re-serves fails; no aggregator data means unchecked."""
    row = bacteria_row(accession="none")
    hit = gates.evaluate_g4_key(
        row,
        "fixtureKey2020",
        index,
        doi_to_pmid={"10.1000/fixture": "123"},
        aggregator_pmids={"SynLethDB SL": frozenset({"123"})},
    )
    assert (hit.record.pmid, hit.record.pmid_checked, hit.result.outcome) == (
        "123",
        True,
        "fail",
    )
    unchecked = gates.evaluate_g4_key(
        row,
        "fixtureKey2020",
        index,
        doi_to_pmid={"10.1000/fixture": "123"},
        aggregator_pmids=None,
    )
    assert (unchecked.record.pmid_checked, unchecked.result.outcome) == (False, "pass")


def test_g4_result_with_values() -> None:
    """A subsumed value record fails G4 with the margin; a clean one passes."""
    key = G4KeyRecord(doi=None, pmid=None, accessions=(), hits=(), pmid_checked=False)
    matches = gates.match_by_value(
        {"lysine": {"g1": 1.0, "g2": 2.0}},
        {"s1": {"g1": 1.01, "g2": 2.0}, "s2": {"g1": 0.0, "g2": 2.0}},
    )
    value = G4ValueRecord(
        store="st", store_commit=None, records_read=4, tolerance=0.05, matches=matches
    )
    assert gates.g4_result(key, value).reason == (
        "subsumed: every released column matches a sample of st within 0.05 (max abs "
        "diff 0.01, nearest-wrong margin >= 0.99)"
    )
    clean = value.model_copy(update={"tolerance": 0.001})
    assert gates.g4_result(key, clean).reason == (
        "no key held by another module (0 mention(s)); no PMID check; values checked "
        "on 1 column(s), not subsumed"
    )


# --------------------------------------------------------------------------- #
# G4-value
# --------------------------------------------------------------------------- #
def test_match_by_value_thompson_shape() -> None:
    """Best and runner-up by max abs difference over shared keys; no shared key raises."""
    released = {"glucose": {"PP_1": 0.10, "PP_2": -1.20, "PP_3": 0.40}}
    served = {
        "set5IT001": {"PP_1": 0.12, "PP_2": -1.20, "PP_3": 0.40, "PP_9": 3.0},
        "set5IT002": {"PP_1": 0.70, "PP_2": -0.50, "PP_3": 0.40},
        "other": {"PP_8": 1.0},
    }
    (match,) = gates.match_by_value(released, served)
    assert (match.best_sample, match.runner_up_sample, match.keys_compared) == (
        "set5IT001",
        "set5IT002",
        3,
    )
    assert match.best_max_abs_diff == pytest.approx(0.02)
    assert match.runner_up_max_abs_diff == pytest.approx(0.70)
    (single,) = gates.match_by_value(released, {"only": served["set5IT001"]})
    assert (single.runner_up_sample, single.margin) == (None, None)
    with pytest.raises(LookupError, match="shares no key"):
        gates.match_by_value(released, {"other": served["other"]})


def test_read_store_values(tmp_path: Path) -> None:
    """Records by dotted path, interned refs resolved, sample filter applied."""
    store = tmp_path / "store"
    records = [
        fitness_record("s1", "PP_1", 0.5),
        fitness_record("s1", "PP_2", -0.25),
        fitness_record("s2", "PP_1", 1.5),
        {"experiment": {"$ref": "exp:0"}},
    ]
    write_store(
        store,
        records,
        interned={"exp:0": fitness_record("s3", "PP_3", 9.0)["experiment"]},
    )
    n, commit, values = gates.read_store_values(
        str(store),
        sample_path="experiment.phenotype.screen_id",
        key_path="experiment.genotype.perturbations.0.systematic_gene_name",
        value_path="experiment.phenotype.environment_response",
    )
    assert (n, commit) == (4, COMMIT)
    assert values == {
        "s1": {"PP_1": 0.5, "PP_2": -0.25},
        "s2": {"PP_1": 1.5},
        "s3": {"PP_3": 9.0},
    }
    _, _, only = gates.read_store_values(
        str(store),
        sample_path="experiment.phenotype.screen_id",
        key_path="experiment.genotype.perturbations.0.systematic_gene_name",
        value_path="experiment.phenotype.environment_response",
        samples=frozenset({"s2"}),
    )
    assert only == {"s2": {"PP_1": 1.5}}


def test_read_store_values_refuses(tmp_path: Path) -> None:
    """A dirty build, no build manifest, and no LMDB are refused."""
    dirty = tmp_path / "dirty"
    write_store(dirty, [], dirty=True)
    with pytest.raises(gates.DirtyStoreError, match="built from a dirty tree"):
        gates.read_store_values(
            str(dirty), sample_path="a", key_path="b", value_path="c"
        )
    with pytest.raises(FileNotFoundError, match="no build manifest"):
        gates.read_store_values(
            str(tmp_path / "missing"), sample_path="a", key_path="b", value_path="c"
        )
    nolmdb = tmp_path / "nolmdb"
    write_store(nolmdb, [])
    (nolmdb / "processed" / "lmdb" / "data.mdb").unlink()
    (nolmdb / "processed" / "lmdb" / "lock.mdb").unlink()
    (nolmdb / "processed" / "lmdb").rmdir()
    with pytest.raises(FileNotFoundError, match="has no records LMDB"):
        gates.read_store_values(
            str(nolmdb), sample_path="a", key_path="b", value_path="c"
        )


# --------------------------------------------------------------------------- #
# G5
# --------------------------------------------------------------------------- #
def test_phenotype_class_exists() -> None:
    """A Phenotype subclass of schema.py; a non-phenotype or absent name is not."""
    assert gates.phenotype_class_exists("FitnessPhenotype") is True
    assert gates.phenotype_class_exists("Genotype") is False
    assert gates.phenotype_class_exists("NoSuchPhenotype") is False


def test_g5_outcomes() -> None:
    """Pass; a filed gap; an unfiled gap blocks; unresolved compounds are named."""
    ok = gates.evaluate_g5("FitnessPhenotype", ["glucose"], issue=None)
    assert (ok.result.outcome, ok.record.unresolved_compounds) == ("pass", ())
    gap = gates.evaluate_g5(
        "ProteinDegradationPhenotype",
        [],
        issue=857,
        proposed_class="ProteinDegradationPhenotype",
    )
    assert (gap.result.outcome, gap.result.issue, gap.result.reason) == (
        "gap",
        857,
        "no Phenotype subclass 'ProteinDegradationPhenotype' in schema.py",
    )
    unfiled = gates.evaluate_g5(
        "FitnessPhenotype", ["notacompoundxyz", "glucose"], issue=None
    )
    assert (unfiled.result.outcome, unfiled.result.blocked_kind) == (
        "blocked",
        "unfiled_gap",
    )
    assert unfiled.result.reason == (
        "1 compound(s) unresolved: notacompoundxyz; file the issue and re-run with its number"
    )


# --------------------------------------------------------------------------- #
# Zotero, aggregators, composition
# --------------------------------------------------------------------------- #
def test_zotero_item(tmp_path: Path) -> None:
    """Present only when the literature manifest names a Zotero item."""
    write_mirror(
        tmp_path, "torchcell-library", "z", {"si/a.csv": b"x"}, zotero_item_key="ABCD"
    )
    write_mirror(tmp_path, "torchcell-library", "n", {"si/a.csv": b"x"})
    assert [gates.zotero_item(k, str(tmp_path)) for k in ("z", "n", "missing")] == [
        "present",
        "absent",
        "absent",
    ]


def test_aggregator_pmid_sets(tmp_path: Path) -> None:
    """Column 5 digits per aggregator; one aggregator missing -> None."""
    for name, rel in (
        ("SL", "data/torchcell/syn_leth_db_yeast/raw/Yeast_SL.csv"),
        ("SR", "data/torchcell/syn_rescue_db_yeast/raw/Yeast_SR.csv"),
    ):
        path = tmp_path / rel
        assert gates.aggregator_pmid_sets(str(tmp_path)) is None
        path.parent.mkdir(parents=True)
        path.write_text(
            f"h0,h1,h2,h3,h4,h5\na,b,c,d,e,1{name == 'SR'}\na,b,c,d,e,42\na,b\n"
        )
    assert gates.aggregator_pmid_sets(str(tmp_path)) == {
        "SynLethDB SL": frozenset({"42"}),
        "SynLethDB SR": frozenset({"42"}),
    }


def test_compose_verdict_applies_the_stop_rule() -> None:
    """After G2 fails, G3..G5 are unmeasured and drop whatever was computed."""
    row = bacteria_row()
    g1 = G1Run(
        result=gates.evaluate_g1(row, None).result,
        source_kind="primary",
        aggregation=None,
    )
    g2 = GateRun(
        result=GateResult(gate="G2", outcome="fail", reason="unnamed"),
        record=G2Record(branch="assembly_set", outcome="absent_unnamed", hosts=()),
    )
    g3 = gates.evaluate_g3(None)
    verdict = gates.compose_verdict(
        row=row,
        citation_key="fixtureKey2020",
        g1=g1,
        runs={"G2": g2, "G3": g3},
        zotero="absent",
        decided_at="2026-10-10",
        torchcell_commit="0" * 40,
    )
    assert [g.outcome for g in verdict.gates] == [
        "pass",
        "fail",
        "unmeasured",
        "unmeasured",
        "unmeasured",
    ]
    assert verdict.gate("G3").reason == "not evaluated: G2 stopped the run"
    assert (verdict.outcome, verdict.g3) == ("refused", None)


def test_compose_verdict_marks_skipped_gates() -> None:
    """A gate missing from runs is unmeasured 'not evaluated on this call'."""
    row = bacteria_row()
    verdict = gates.compose_verdict(
        row=row,
        citation_key="fixtureKey2020",
        g1=gates.evaluate_g1(row, None),
        runs={},
        zotero="present",
        decided_at="2026-10-10",
        torchcell_commit="0" * 40,
    )
    assert [g.reason for g in verdict.gates[1:]] == ["not evaluated on this call"] * 4
    assert verdict.outcome == "pending"
