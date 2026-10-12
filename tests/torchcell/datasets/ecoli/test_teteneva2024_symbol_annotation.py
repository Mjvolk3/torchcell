# tests/torchcell/datasets/ecoli/test_teteneva2024_symbol_annotation.py
# [[tests.torchcell.datasets.ecoli.test_teteneva2024_symbol_annotation]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/ecoli/test_teteneva2024_symbol_annotation.py

"""The downstream measurement of how Teteneva 2024's symbols land on the W3110
annotations (``experiments/database/scripts/teteneva2024_symbol_annotation.py``).

Not loader logic: the loader stores the released symbols verbatim (#910). The hermetic
tests run the measurement functions on synthetic flat files and loci; the ``--data``
tests re-derive the published counts off the deposited tier.
"""

import gzip
import importlib.util
import os
import os.path as osp
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Literal

import pytest
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqFeature import SeqFeature, SimpleLocation
from Bio.SeqRecord import SeqRecord

from torchcell.sequence.genome.bacterial import GenBankLocus
from torchcell.sequence.genome.base import GeneNameResolution, GeneNameStatus
from torchcell.sequence.genome.registry import ECOLI_K12_W3110

SCRIPT = (
    Path(__file__).resolve().parents[4]
    / "experiments/database/scripts/teteneva2024_symbol_annotation.py"
)


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "teteneva2024_symbol_annotation", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


sa = _load()

DATA_ROOT = os.environ.get("DATA_ROOT", "")
requires_tier = pytest.mark.skipif(
    not osp.isfile(
        osp.join(DATA_ROOT, "torchcell-genomes", ECOLI_K12_W3110, "manifest.json")
    ),
    reason=f"requires {ECOLI_K12_W3110} under $DATA_ROOT/torchcell-genomes",
)


def _write_gbff(path: Path, *, tagged: bool, notes: tuple[str, ...] = ()) -> None:
    """A one-locus GenBank flat file, gzipped, with or without a ``locus_tag``."""
    record = SeqRecord(
        Seq("ATGAAACCCGGGTTTAAACCCGGGTTTAAACCCGGGTTTAAACCCGGGTTTAAACCCTAA"),
        id="XX000001.1",
        name="XX000001",
        annotations={"molecule_type": "DNA", "topology": "circular"},
    )
    gene_qualifiers: dict[str, list[str]] = {"gene": ["thrL"]}
    cds_qualifiers: dict[str, list[str]] = {
        "gene": ["thrL"],
        "product": ["thr operon leader peptide"],
    }
    if tagged:
        gene_qualifiers["locus_tag"] = ["Y75_RS00005"]
        cds_qualifiers["locus_tag"] = ["Y75_RS00005"]
    if notes:
        cds_qualifiers["note"] = list(notes)
    location = SimpleLocation(0, 60, strand=1)
    record.features.append(
        SeqFeature(location, type="gene", qualifiers=gene_qualifiers)
    )
    record.features.append(SeqFeature(location, type="CDS", qualifiers=cds_qualifiers))
    with gzip.open(path, "wt") as handle:
        SeqIO.write(record, handle, "genbank")


def _serve(
    monkeypatch: pytest.MonkeyPatch, members: dict[str, Path]
) -> list[tuple[str, str]]:
    """Stub ``resolve`` in the module under test; record every ``(set, member)`` asked."""
    calls: list[tuple[str, str]] = []

    def serve(assembly_set: str, filename: str, **_: object) -> str:
        calls.append((assembly_set, filename))
        return str(members[filename])

    monkeypatch.setattr(sa, "resolve", serve)
    return calls


def test_gene_feature_counts_separate_tagged_from_untagged_features(
    tmp_path: Path,
) -> None:
    """One gene feature each way: the tagged file counts 1 of 1, the other 0 of 1."""
    tagged, untagged = tmp_path / "tagged.gbff.gz", tmp_path / "untagged.gbff.gz"
    _write_gbff(tagged, tagged=True)
    _write_gbff(untagged, tagged=False)
    assert sa._gene_feature_counts(str(tagged)) == (1, 1)
    assert sa._gene_feature_counts(str(untagged)) == (1, 0)


def test_annotation_routes_refuses_the_untagged_member_and_parses_the_tagged_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The route pair is built from whichever bytes the tier serves, GenBank first."""
    members = {
        sa.GENBANK_MEMBER: tmp_path / "gca.gbff.gz",
        sa.REFSEQ_MEMBER: tmp_path / "gcf.gbff.gz",
    }
    _write_gbff(members[sa.GENBANK_MEMBER], tagged=False)
    _write_gbff(members[sa.REFSEQ_MEMBER], tagged=True)
    calls = _serve(monkeypatch, members)
    genbank, refseq = sa.annotation_routes()
    assert calls == [
        (ECOLI_K12_W3110, sa.GENBANK_MEMBER),
        (ECOLI_K12_W3110, sa.REFSEQ_MEMBER),
    ]
    assert genbank.gene_features == 1
    assert genbank.gene_features_with_locus_tag == 0
    assert genbank.read_genbank_error == (
        f"{sa.GENBANK_MEMBER}: a gene feature at [0:60](+) has no locus_tag"
    )
    assert genbank.loci is None
    assert genbank.replicon is None
    assert refseq.gene_features_with_locus_tag == 1
    assert refseq.read_genbank_error is None
    assert refseq.loci == 1
    assert refseq.replicon == "XX000001.1"


def test_symbol_routes_count_unique_ambiguous_and_absent_symbols(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ThrL is on one feature of each member; an unknown symbol lands on none."""
    members = {
        sa.GENBANK_MEMBER: tmp_path / "gca.gbff.gz",
        sa.REFSEQ_MEMBER: tmp_path / "gcf.gbff.gz",
    }
    _write_gbff(members[sa.GENBANK_MEMBER], tagged=False)
    _write_gbff(members[sa.REFSEQ_MEMBER], tagged=True)
    _serve(monkeypatch, members)
    assert sa.symbol_index(sa.REFSEQ_MEMBER) == {"thrL": {"Y75_RS00005"}}
    assert sa.symbol_routes(("thrL", "zzzZ")) == {
        sa.GENBANK_MEMBER: {"unique": 1, "ambiguous": 0, "none": 1},
        sa.REFSEQ_MEMBER: {"unique": 1, "ambiguous": 0, "none": 1},
    }


def test_eck_jw_b_notes_reads_only_the_crosswalk_notes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The triple note is read apart; any other note on the same CDS is not a triple."""
    member = tmp_path / "gca.gbff.gz"
    _write_gbff(member, tagged=False, notes=("ECK0001:JW4367:b0001", "not a crosswalk"))
    _serve(monkeypatch, {sa.GENBANK_MEMBER: member})
    assert sa.eck_jw_b_notes() == (("ECK0001", "JW4367", "b0001"),)


# --------------------------------------------------------------------------------------
# The three resolution steps, on synthetic loci
# --------------------------------------------------------------------------------------
def _locus(
    tag: str,
    symbol: str | None,
    start: int,
    end: int,
    strand: Literal["+", "-"],
    pseudo: bool = False,
) -> GenBankLocus:
    return GenBankLocus(
        locus_tag=tag,
        symbol=symbol,
        synonyms=(),
        old_locus_tags=(),
        db_xrefs=(),
        pseudo=pseudo,
        replicon="NC_007779.1",
        start=start,
        end=end,
        strand=strand,
        segments=1,
        product_feature_type="CDS",
        product=None,
        protein_id=None,
        isoform_protein_ids=(),
    )


LOCI = {
    "Y75_RS00005": _locus("Y75_RS00005", "thrL", 1, 66, "+"),
    "Y75_RS00010": _locus("Y75_RS00010", "cspE", 100, 300, "+"),
    "Y75_RS00015": _locus("Y75_RS00015", "cspE", 400, 600, "-"),
    "Y75_RS00020": _locus("Y75_RS00020", "glaH", 700, 900, "+"),
    "Y75_RS00025": _locus("Y75_RS00025", "dcuC", 1000, 1100, "+", pseudo=True),
    "Y75_RS00030": _locus("Y75_RS00030", "dcuC", 1150, 1300, "+", pseudo=True),
    "Y75_RS00035": _locus("Y75_RS00035", "ygjG", 1400, 1500, "+"),
}


def _feature(
    kind: str,
    symbol: str | None,
    start: int,
    end: int,
    strand: str,
    triple: tuple[str, str, str] | None = None,
) -> Any:
    return sa.GenbankFeature(
        kind=kind, symbol=symbol, start=start, end=end, strand=strand, eck_jw_b=triple
    )


GENBANK = sa.GenbankMember(
    genes=(
        _feature("gene", "cspA", 400, 610, "-"),
        _feature("gene", "dcuC", 990, 1300, "+"),
        _feature("gene", "rzoD", 1600, 1700, "+"),
    ),
    cds=(
        _feature("CDS", "cspA", 400, 610, "-", ("ECK3546", "JW3524", "b3556")),
        _feature("CDS", "ygaT", 650, 900, "+", ("ECK2654", "JW2634", "b2659")),
        _feature("CDS", "ygjG", 1400, 1500, "+", ("ECK3066", "JW3045", "b3073")),
    ),
)


def _mg1655(name: str) -> GeneNameResolution:
    table = {
        "csiD": ("b2659", GeneNameStatus.RENAMED, None),
        "pat": ("b3073", GeneNameStatus.RENAMED, None),
        "Tuf": (
            "b3339",
            GeneNameStatus.RENAMED,
            "gene symbol (case-insensitive match)",
        ),
    }
    if name not in table:
        return GeneNameResolution(
            input_name=name, status=GeneNameStatus.RETIRED, systematic_name=name
        )
    tag, status, note = table[name]
    return GeneNameResolution(
        input_name=name, status=status, systematic_name=tag, note=note
    )


def test_each_step_resolves_the_symbol_it_is_for() -> None:
    out = sa.resolve_symbols(["thrL", "cspA", "csiD"], LOCI, GENBANK, _mg1655)
    assert {s: (r.step, r.locus_tag) for s, r in out.items()} == {
        "cspA": ("genbank_symbol_eck_jw_b", "Y75_RS00015"),
        "csiD": ("mg1655_symbol_eck_jw_b", "Y75_RS00020"),
        "thrL": ("refseq_symbol", "Y75_RS00005"),
    }
    assert out["cspA"].eck_jw_b == ("ECK3546", "JW3524", "b3556")
    assert out["csiD"].refseq_symbol == "glaH"


def test_a_split_pseudogene_a_missing_stop_and_a_folded_name_stay_unresolved() -> None:
    out = sa.resolve_symbols(["dcuC", "rzoD", "Tuf", "zzzZ"], LOCI, GENBANK, _mg1655)
    assert {s: r.status for s, r in out.items()} == dict.fromkeys(
        ("Tuf", "dcuC", "rzoD", "zzzZ"), "unresolved"
    )
    assert [(a.step, a.result) for a in out["dcuC"].attempts][:2] == [
        ("refseq_symbol", "several_features"),
        ("genbank_symbol_eck_jw_b", "split_pseudogene"),
    ]
    assert out["rzoD"].attempts[1].result == "no_refseq_stop_match"
    assert out["Tuf"].attempts[2].result == "mg1655_not_one_gene"


def test_two_symbols_on_one_locus_keep_the_earlier_step_s() -> None:
    out = sa.resolve_symbols(["ygjG", "pat"], LOCI, GENBANK, _mg1655)
    assert (out["ygjG"].step, out["ygjG"].locus_tag) == ("refseq_symbol", "Y75_RS00035")
    assert out["pat"].status == "unresolved"
    last = out["pat"].attempts[-1]
    assert (last.result, last.candidates) == ("locus_held_by_another_symbol", ("ygjG",))


def test_the_report_counts_steps_conflicts_and_agreement() -> None:
    out = sa.resolve_symbols(
        ["thrL", "cspA", "csiD", "dcuC", "cspE"], LOCI, GENBANK, _mg1655
    )
    report = sa.resolution_report(out, LOCI, GENBANK, _mg1655)
    assert report.resolved_by_step == {
        "refseq_symbol": 1,
        "genbank_symbol_eck_jw_b": 1,
        "mg1655_symbol_eck_jw_b": 1,
    }
    assert report.unresolved == ("cspE", "dcuC")
    assert report.step2_against_step3.step3_unresolved == ("cspA",)
    assert report.genbank_route_conflicts == ()


# --------------------------------------------------------------------------------------
# The deposited tier (--data)
# --------------------------------------------------------------------------------------
@pytest.mark.data
@requires_tier
def test_the_genbank_member_is_refused_and_the_refseq_member_parses() -> None:
    genbank, refseq = sa.annotation_routes()
    assert genbank.gene_features == sa.GENBANK_GENE_FEATURES == 4_444
    assert genbank.gene_features_with_locus_tag == 0
    assert genbank.read_genbank_error == sa.GENBANK_READ_ERROR
    assert refseq.loci == sa.REFSEQ_LOCI == 4_531
    assert refseq.replicon == "NC_007779.1"


@pytest.mark.data
@requires_tier
def test_the_measured_routes_and_resolution_reproduce() -> None:
    result = sa.measure(DATA_ROOT)
    assert result["symbol_routes"] == sa.SYMBOL_ROUTES
    resolution = result["resolution"]
    assert resolution["resolved_by_step"] == sa.RESOLVED_BY_STEP
    assert tuple(resolution["unresolved"]) == sa.UNRESOLVED
    assert resolution["step2_against_step3"]["disagree"] == []
