# tests/torchcell/datasets/ecoli/test_mutalik2020
# [[tests.torchcell.datasets.ecoli.test_mutalik2020]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/ecoli/test_mutalik2020
"""Tests for the Mutalik 2020 provenance and sourcing layer.

Two tiers. The hermetic tests cover the pins, the refusals and the description parsing
with no file access. The ``@pytest.mark.data`` tests read the real ``$DATA_ROOT``: that
every quote is verbatim in the sha256-pinned mirrors, that the deposited mirror matches
its manifest and re-deposits idempotently, that the experiment axis parses to the counts
the paper states, and that the two released MOI sources are reconciled the way the
Methods designate. The heaviest ones also stream the deposited 915 MB figshare tarball,
so they carry ``@pytest.mark.slow`` as well.
"""

from __future__ import annotations

import csv
import hashlib
import io
import os
import os.path as osp
from pathlib import Path

import pytest

from torchcell.datasets.ecoli import mutalik2020 as mut
from torchcell.sequence.genome.ecoli.k12 import (
    EcoliK12BW25113Genome,
    EcoliK12MG1655Genome,
)

LIBRARY_ROOT = osp.join(os.environ.get("DATA_ROOT", ""), "torchcell-library")
MIRROR_PRESENT = osp.isfile(
    osp.join(os.environ.get("DATA_ROOT", ""), mut.RAW_DIR_REL, "manifest.json")
)
PAPERS_PRESENT = all(
    osp.isfile(osp.join(LIBRARY_ROOT, key, "paper.md"))
    for key in (mut.CITATION_KEY, mut.WETMORE_KEY)
)
on_mirror = [
    pytest.mark.data,
    pytest.mark.skipif(
        not MIRROR_PRESENT,
        reason="requires the deposited raw mirror under $DATA_ROOT/torchcell-raw",
    ),
]
TIER_PRESENT = all(
    osp.isfile(
        osp.join(
            os.environ.get("DATA_ROOT", ""),
            "torchcell-genomes",
            assembly_set,
            "manifest.json",
        )
    )
    for assembly_set in ("ecoli_K12_MG1655_ASM584v2", "ecoli_K12_BW25113_ASM75055v1")
)


# --------------------------------------------------------------------------- #
# Pins and refusals (hermetic)
# --------------------------------------------------------------------------- #
def test_every_artifact_is_pinned_once_and_is_retrievable_by_an_existing_retriever() -> (
    None
):
    """Each raw artifact has a unique path, a 64-hex sha256 and a real retriever."""
    import torchcell.literature.retrieve as retrieve

    rels = [record.rel for record in mut.RAW_ARTIFACTS]
    assert len(rels) == len(set(rels)) == 5
    for record in mut.RAW_ARTIFACTS:
        assert len(record.sha256) == 64
        assert int(record.sha256, 16) >= 0
        assert record.bytes > 0
        module, _, function = record.retriever.rpartition(".")
        assert module == "torchcell.literature.retrieve"
        assert callable(getattr(retrieve, function))
        assert set(record.params) <= {"url", "key"}


def test_artifact_lookup_names_the_unknown_path() -> None:
    """``artifact`` resolves a pinned path and refuses anything else."""
    assert mut.artifact(mut.S1_TABLE_REL).bytes == 4126084
    with pytest.raises(KeyError, match="not a pinned raw artifact"):
        mut.artifact("data/does-not-exist.xlsx")


def test_tarball_members_cover_every_analysis_set() -> None:
    """Each analysis set contributes its fitness, t, SE, quality and metadata members."""
    sets = sorted(set(mut.ANALYSIS_SETS.values()))
    assert sets == ["Keio_ML9_set16_set19", "Keio_ML9_set28_set29", "Keio_ML9_set30"]
    for name in sets:
        for leaf in (
            "exps",
            "fit_logratios.tab",
            "fit_t.tab",
            "fit_standard_error_obs.tab",
            "fit_quality.tab",
        ):
            assert f"html/{name}/{leaf}" in mut.TARBALL_MEMBERS
    assert "g/Keio/genes.tab" in mut.TARBALL_MEMBERS
    assert all(len(digest) == 64 for digest in mut.TARBALL_MEMBERS.values())


def test_deposit_refuses_an_incomplete_source_set(tmp_path: Path) -> None:
    """Every pinned artifact needs a source; the refusal names the ones missing."""
    with pytest.raises(ValueError, match="needs a source for every artifact"):
        mut.deposit_raw_mirror(
            sources={mut.S1_TABLE_REL: tmp_path / "s009.xlsx"}, data_root=str(tmp_path)
        )


def test_deposit_refuses_bytes_that_do_not_match_the_pin(tmp_path: Path) -> None:
    """A source file whose sha256 differs from the pin raises before anything is copied."""
    sources: dict[str, str | Path] = {}
    for record in mut.RAW_ARTIFACTS:
        path = tmp_path / Path(record.rel).name
        path.write_bytes(b"not the released bytes")
        sources[record.rel] = path
    with pytest.raises(RuntimeError, match="sha256 mismatch"):
        mut.deposit_raw_mirror(sources=sources, data_root=str(tmp_path / "root"))
    assert not (tmp_path / "root").exists()


def test_manifest_sha256_names_a_path_it_does_not_carry() -> None:
    """``manifest_sha256`` is a lookup, not a search that silently returns nothing."""
    from torchcell.literature.manifest import ROLE_RAW_DATA, ArtifactRecord, Manifest

    manifest = Manifest(
        citation_key=mut.CITATION_KEY,
        doi=mut.PAPER_DOI,
        title=mut.PAPER_TITLE,
        files=[
            ArtifactRecord(
                path=mut.S1_TABLE_REL, role=ROLE_RAW_DATA, bytes=1, sha256="ab" * 32
            )
        ],
    )
    assert mut.manifest_sha256(manifest, mut.S1_TABLE_REL) == "ab" * 32
    with pytest.raises(KeyError, match="not in the raw-mirror manifest"):
        mut.manifest_sha256(manifest, mut.S13_TABLE_REL)


def test_read_tarball_member_refuses_an_unpinned_member() -> None:
    """Only a member with a recorded sha256 can be read at all."""
    with pytest.raises(KeyError):
        mut.read_tarball_member("html/Keio_ML9_set30/strain_fit.tab")


# --------------------------------------------------------------------------- #
# Description parsing (hermetic)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "description,kind",
    [
        ("Time0", "time_zero"),
        ("LB_plus_SM_buffer", "no_phage_control"),
        ("LB", "no_phage_control"),
        ("No phage control ML9a", "no_phage_control"),
        ("NoPhageControl", "no_phage_control"),
        ("LB_plus_SM_buffer with T2_phage 187.5 MOI", "phage"),
        ("T4_phage_221.875_MOI", "phage"),
        ("P2 dilution 10-1", "phage"),
    ],
)
def test_assay_kind_splits_start_control_and_challenge(
    description: str, kind: str
) -> None:
    """A released description is a start sample, a no-phage control or a challenge."""
    assert mut._assay_kind(description) == kind


@pytest.mark.parametrize(
    "description,phage,moi",
    [
        ("LB_plus_SM_buffer with T2_phage 187.5 MOI", "T2", 187.5),
        ("LB_plus_SM_buffer with N4_phage 0.0234375 MOI", "N4", 0.0234375),
        ("T5_phage_625_MOI", "T5", 625.0),
        ("186_phage_125_MOI", "186", 125.0),
        ("P2 dilution 10-2", "P2", 0.01),
    ],
)
def test_phage_and_dose_read_from_a_challenge_description(
    description: str, phage: str, moi: float
) -> None:
    """The three description shapes the release uses are all read, dose included."""
    assert mut._phage_and_moi_from_description(description) == (phage, moi)


def test_an_unreadable_challenge_description_raises() -> None:
    """A description shape nobody wrote a rule for stops the parse."""
    with pytest.raises(ValueError, match="unreadable phage challenge description"):
        mut._phage_and_moi_from_description("something else entirely")


# --------------------------------------------------------------------------- #
# Sourced values (hermetic shape, data-gated content)
# --------------------------------------------------------------------------- #
def test_every_sourced_value_is_auditable() -> None:
    """Each value carries a verbatim quote and a hashed, citation-keyed provenance."""
    values = mut.sourced_values()
    assert "uncertainty_type" in values and "n_samples" in values
    for key, sourced in values.items():
        assert sourced.quote.strip(), key
        assert sourced.provenance.citation_key in (mut.CITATION_KEY, mut.WETMORE_KEY)
        assert sourced.provenance.sha256 in (
            mut.PAPER_MD_SHA256,
            mut.WETMORE_PAPER_MD_SHA256,
        )
    assert values["uncertainty_type"].value == "standard_error"
    assert values["n_samples"].value == mut.MEDIAN_STRAINS_PER_GENE
    assert values["n_samples"].provenance.citation_key == mut.WETMORE_KEY
    assert values["measurement_type"].value == "log2_ratio"
    assert values["reference_strain"].value == mut.REFERENCE_STRAIN


@pytest.mark.data
@pytest.mark.skipif(not PAPERS_PRESENT, reason="requires the torchcell-library mirror")
def test_every_quote_is_verbatim_in_the_hashed_mirror() -> None:
    """Each quote is a substring of the exact bytes its provenance sha256 names."""
    cache: dict[str, str] = {}
    for key, sourced in mut.sourced_values().items():
        citation = sourced.provenance.citation_key
        assert citation is not None
        path = Path(LIBRARY_ROOT) / citation / sourced.provenance.source_uri
        if citation not in cache:
            payload = path.read_bytes()
            assert hashlib.sha256(payload).hexdigest() == sourced.provenance.sha256
            cache[citation] = payload.decode()
        assert sourced.quote in cache[citation], key


# --------------------------------------------------------------------------- #
# The deposited mirror (data-gated)
# --------------------------------------------------------------------------- #
class TestDepositedMirror:
    """The mirror on disk, its manifest and the axis parsed out of it."""

    pytestmark = on_mirror

    def test_manifest_records_every_artifact_with_its_retrieval(self) -> None:
        """The manifest carries all five files, each with a re-runnable retrieval."""
        manifest = mut.load_manifest()
        assert manifest.citation_key == mut.CITATION_KEY
        assert manifest.doi == mut.PAPER_DOI
        assert len(manifest.files) == len(mut.RAW_ARTIFACTS)
        for record in mut.RAW_ARTIFACTS:
            assert mut.manifest_sha256(manifest, record.rel) == record.sha256
            stored = next(f for f in manifest.files if f.path == record.rel)
            assert stored.retrieval is not None
            assert stored.retrieval.retriever == record.retriever
            assert stored.retrieval.params == record.params
            assert stored.retrieval.sha256 == record.sha256

    def test_the_bytes_on_disk_carry_the_recorded_hash(self) -> None:
        """The deposited files are the pinned bytes, sizes included."""
        root = mut.raw_mirror_dir()
        for record in mut.RAW_ARTIFACTS:
            path = root / record.rel
            assert path.stat().st_size == record.bytes, record.rel
            if record.bytes < 50_000_000:
                assert mut._sha256(path) == record.sha256, record.rel

    def test_redepositing_from_the_mirror_itself_changes_nothing(self) -> None:
        """The deposit is idempotent: matching files are left exactly as they are."""
        root = mut.raw_mirror_dir()
        small = [r for r in mut.RAW_ARTIFACTS if r.bytes < 50_000_000]
        before = {r.rel: (root / r.rel).stat().st_mtime_ns for r in small}
        mut.deposit_raw_mirror(sources={r.rel: root / r.rel for r in mut.RAW_ARTIFACTS})
        for record in small:
            assert (root / record.rel).stat().st_mtime_ns == before[record.rel]
            assert mut._sha256(root / record.rel) == record.sha256

    def test_the_experiment_axis_is_the_paper_s_own_analysis_set(self) -> None:
        """99 experiments: 21 start samples, 10 no-phage controls, 68 challenges."""
        axis = mut.read_experiment_axis()
        assert axis.counts == {
            "time_zero": 21,
            "no_phage_control": 10,
            "phage": 68,
            "phages": 14,
        }
        assert axis.phages == mut.PHAGES
        assert {a.analysis_set for a in axis.assays} == set(mut.ANALYSIS_SETS.values())
        assert all(a.moi is not None for a in axis.challenges)

    def test_the_text_says_nine_controls_and_the_released_list_holds_ten(self) -> None:
        """The one no-phage control run in plain LB, without SM buffer, is the extra."""
        axis = mut.read_experiment_axis()
        assert len(axis.controls) == 10
        assert mut.sourced_values()["experiment_counts"].value["no_phage_controls"] == 9
        plain = [a for a in axis.controls if a.description.strip() == "LB"]
        assert [a.exp_name for a in plain] == ["set16IT014"]

    def test_the_dose_comes_from_s13_and_one_assay_falls_back_to_its_description(
        self,
    ) -> None:
        """S13 covers 67 of the 68 challenges; set16IT019 states its own MOI."""
        moi_table = mut.read_moi_table()
        assert len(moi_table) == 67
        axis = mut.read_experiment_axis()
        fallback = [
            a for a in axis.challenges if a.moi_source == "experiment_description"
        ]
        assert [(a.exp_name, a.phage, a.moi) for a in fallback] == [
            ("set16IT019", "T2", 0.01875)
        ]

    @pytest.mark.parametrize(
        "exp_name,s13_moi,s1_label_moi",
        [
            ("set16IT049", 0.0171875, 17.1875),
            ("set16IT073", 484.375, 4.84375),
            ("set16IT078", 1437.5, 143.75),
            ("set30IT064", 1937.5, 19.375),
        ],
    )
    def test_s13_disagrees_with_the_s1_column_labels_by_a_power_of_ten(
        self, exp_name: str, s13_moi: float, s1_label_moi: float
    ) -> None:
        """The Methods designate S13, and S13 is not what the S1 headers say."""
        moi = mut.read_moi_table()[exp_name].moi
        assert moi == pytest.approx(s13_moi, rel=1e-9)
        assert moi != pytest.approx(s1_label_moi, rel=1e-6)

    def test_the_s13_dose_matches_the_experiment_s_own_description(self) -> None:
        """For 64 of the 67 rows the two released dose statements agree exactly."""
        table = mut.read_moi_table()
        agree: dict[str, tuple[str, float]] = {}
        disagree: dict[str, tuple[str, float]] = {}
        for exp_name, row in table.items():
            read = mut._phage_and_moi_from_description(row.description)
            bucket = agree if read[1] == pytest.approx(row.moi, rel=1e-9) else disagree
            bucket[exp_name] = read
        assert len(agree) == 64
        assert sorted(disagree) == ["set16IT051", "set19IT073", "set19IT074"]
        assert agree["set16IT008"][0] == "T2"
        assert agree["set16IT008"][1] == pytest.approx(187.5, rel=1e-9)
        assert agree["set16IT008"][1] == pytest.approx(
            table["set16IT008"].moi, rel=1e-9
        )
        # The two P2 descriptions state a DILUTION, which S13 turns into a dose.
        assert disagree["set19IT073"][0] == "P2"
        assert disagree["set19IT073"][1] == pytest.approx(0.1, rel=1e-9)
        assert table["set19IT073"].moi == pytest.approx(0.5625, rel=1e-9)


# --------------------------------------------------------------------------- #
# The deposited tarball (data-gated and slow: it streams 915 MB per member)
# --------------------------------------------------------------------------- #
class TestDepositedTarball:
    """The complete release the Data Availability statement names."""

    pytestmark = [*on_mirror, pytest.mark.slow]

    def test_a_pinned_member_reads_back_at_its_recorded_hash(self) -> None:
        """The FEBA gene table comes out of the archive at its pinned sha256."""
        payload = mut.read_tarball_member("g/Keio/genes.tab")
        assert (
            hashlib.sha256(payload).hexdigest()
            == mut.TARBALL_MEMBERS["g/Keio/genes.tab"]
        )
        rows = list(csv.DictReader(io.StringIO(payload.decode()), delimiter="\t"))
        assert len(rows) == 4610
        assert rows[0]["sysName"] == "b0001" and rows[0]["name"] == "thrL"

    def test_the_released_table_is_bw25113_biology_in_mg1655_identifiers(self) -> None:
        """The genes BW25113 deleted carry no fitness row; its point lesions do."""
        import pandas as pd

        payload = mut.read_tarball_member("html/Keio_ML9_set16_set19/fit_logratios.tab")
        frame = pd.read_csv(
            io.StringIO(payload.decode()), sep="\t", dtype={"sysName": str}
        )
        measured = set(frame["sysName"])
        assert not measured & {"b0062", "b0063", "b3903", "b3904"}
        assert {"b0344", "b4350", "b3643"} <= measured

    def test_the_uncertainty_the_release_carries_is_an_se_per_record(self) -> None:
        """Every fitness column has a t and an estimated standard error beside it."""
        import pandas as pd

        frames = {}
        for leaf in ("fit_logratios.tab", "fit_t.tab", "fit_standard_error_obs.tab"):
            payload = mut.read_tarball_member(f"html/Keio_ML9_set28_set29/{leaf}")
            frames[leaf] = pd.read_csv(
                io.StringIO(payload.decode()), sep="\t", dtype={"sysName": str}
            )
        fitness = frames["fit_logratios.tab"]
        experiments = [c for c in fitness.columns if c.startswith("set")]
        for leaf in ("fit_t.tab", "fit_standard_error_obs.tab"):
            assert [c for c in frames[leaf].columns if c.startswith("set")] == (
                experiments
            )
            assert len(frames[leaf]) == len(fitness)
        assert (frames["fit_standard_error_obs.tab"][experiments] >= 0).all().all()


# --------------------------------------------------------------------------- #
# The identifier route the records take (data-gated, slow: genomes + tarball)
# --------------------------------------------------------------------------- #
def _k12_genomes() -> tuple[EcoliK12MG1655Genome, EcoliK12BW25113Genome]:
    """The two deposited K-12 annotations, reopened read-only from their caches."""
    from torchcell.datasets.bacteria_common import bacterial_genome

    mg1655 = bacterial_genome("ecoli", "MG1655")
    bw25113 = bacterial_genome("ecoli", "BW25113")
    assert isinstance(mg1655, EcoliK12MG1655Genome)
    assert isinstance(bw25113, EcoliK12BW25113Genome)
    return mg1655, bw25113


class TestIdentifierRoute:
    """b-number to BW25113 locus tag, scored against the two naive alternatives."""

    pytestmark = [
        *on_mirror,
        pytest.mark.slow,
        pytest.mark.skipif(
            not TIER_PRESENT,
            reason="requires the two deposited E. coli K-12 assembly sets",
        ),
    ]

    def test_the_eck_join_resolves_the_genes_that_carry_a_value(self) -> None:
        """3,697 of 3,716, with the one numeric disagreement the join exists to catch."""
        mg1655, bw25113 = _k12_genomes()
        measured = mut.measured_gene_ids()
        assert len(measured) == 3716
        route = mut.eck_route(mg1655, bw25113, measured)
        assert (route.pairs, route.numeric_disagreements) == (4423, 11)
        assert (route.requested, route.mapped) == (3716, 3697)
        assert route.fraction == pytest.approx(0.994887, abs=1e-6)
        assert len(route.unmapped) == 19
        assert "b4659" in route.unmapped  # yabP, the ambiguous MG1655 pair
        assert route.disagreeing == (("b0018", "BW25113_4412"),)

    def test_the_eck_route_beats_both_naive_routes(self) -> None:
        """The audit scores all three, so the choice is measured rather than asserted."""
        mg1655, bw25113 = _k12_genomes()
        audit = mut.audit_identifiers(mg1655, bw25113, measured=mut.measured_gene_ids())
        assert audit.genes == 4610
        assert audit.b_numbers_vs_mg1655.resolved_fraction == pytest.approx(
            0.997999, abs=1e-6
        )
        assert audit.symbols_vs_bw25113.resolved_fraction == pytest.approx(
            0.96123, abs=1e-5
        )
        assert audit.eck_route.fraction > audit.symbols_vs_bw25113.resolved_fraction


# --------------------------------------------------------------------------- #
# A synthetic mirror (hermetic): every reader driven end to end on tiny fixtures
# --------------------------------------------------------------------------- #
S13_HEADER = [
    "Library/assay",
    "expName",
    "Phage",
    "expDescription",
    "Batch",
    "pfu/ml",
    "phage count for 350 ul",
    "1 OD cfu/ml",
    "Cell count at od 0.04, for 350 ul",
    "phage dilution",
    "MOI",
]


def _s13_row(
    row: int,
    library: str,
    exp_name: str,
    phage: str,
    description: str,
    pfu: float | None,
    dilution: float | str | None,
    *,
    one_od: str = "=8*10^8",
    moi: str | None = None,
) -> list[object]:
    """One S13 Table row in the sheet's own formula shapes."""
    return [
        library,
        exp_name,
        phage,
        description,
        1,
        pfu,
        f"=F{row}*0.35",
        one_od,
        "=0.04*0.35*8*10^8",
        dilution,
        moi if moi is not None else f"=G{row}*J{row}/I{row}",
    ]


def _write_s13(path: Path, rows: list[list[object]]) -> None:
    """Write a one-sheet ``MOI_used_runs`` workbook (formulas stay formula strings)."""
    import openpyxl

    workbook = openpyxl.Workbook()
    sheet = workbook.active
    assert sheet is not None
    sheet.title = "MOI_used_runs"
    sheet.append(S13_HEADER)
    for values in rows:
        sheet.append(values)
    workbook.save(path)


GOOD_S13_ROWS: list[list[object]] = [
    # 6e10 pfu/ml at a 0.1 dilution: 6e10 * 0.35 * 0.1 / (0.04 * 0.35 * 8e8) = 187.5
    _s13_row(
        2,
        "RBTnSeq-BW25113",
        "set16IT008",
        "T2",
        "LB_plus_SM_buffer with T2_phage 187.5 MOI",
        6e10,
        0.1,
    ),
    # a dilution chained off the row above: 0.1 * 0.1 -> 18.75
    _s13_row(
        3,
        "RBTnSeq-BW25113",
        "set16IT009",
        "T2",
        "LB_plus_SM_buffer with T2_phage 18.75 MOI",
        6e10,
        "=J2*0.1",
    ),
    # another library's row is skipped even though it is well formed
    _s13_row(4, "Dub-seq BW25113 ", "IT047", "T2", "IT047_T2_Phage", 6e10, 0.1),
    # 6.2e11 pfu/ml at 0.1: 1937.5
    _s13_row(
        5, "RBTnSeq-BW25113", "set30IT064", "N4", "N4_phage_1937.5_MOI", 6.2e11, 0.1
    ),
    # a control row: no dose, no dilution
    _s13_row(6, "RBTnSeq-BW25113", "set28IT004", "", "LB_plus_SM_buffer", None, None),
]

EXPS_USED = (
    "expName\texpDescription\tt0set\tgMean\tmaxFit\n"
    "set16IT001\tTime0\t22-Nov-16 Keio_ML9_set16\t557.2\t1.37\n"
    "set16IT007\tLB_plus_SM_buffer\t22-Nov-16 Keio_ML9_set16\t633.4\t3.30\n"
    "set16IT008\tLB_plus_SM_buffer with T2_phage 187.5 MOI\t22-Nov-16 Keio_ML9_set16"
    "\t659.5\t16.75\n"
    "set16IT019\tLB_plus_SM_buffer with T2_phage 0.01875 MOI\t22-Nov-16 Keio_ML9_set16"
    "\t640.2\t14.70\n"
    "set19IT073\tP2 dilution 10-1\t13-Dec-16 Keio_ML9_set19\t500.0\t9.00\n"
    "set28IT004\tLB_plus_SM_buffer\t16-Nov-18 Keio_ML9_set28\t610.0\t2.10\n"
    "set30IT064\tN4_phage_1937.5_MOI\t28-Jan-19 Keio_ML9_set30\t720.0\t12.30\n"
)

GENES_TAB = (
    "locusId\tsysName\ttype\tscaffoldId\tbegin\tend\tstrand\tname\tdesc\n"
    "14146\tb0001\t1\t7023\t190\t255\t+\tthrL\tthr operon leader peptide\n"
    "14147\tb0002\t1\t7023\t337\t2799\t+\tthrA\taspartokinase\n"
    "14148\tb0003\t1\t7023\t2801\t3733\t+\t\tno symbol released\n"
    "14163\tb0018\t1\t7023\t17489\t17665\t+\tmokC\tregulatory peptide\n"
)


def _fitness_table(genes: list[str]) -> str:
    """A two-experiment ``fit_logratios.tab`` over ``genes``."""
    lines = ["locusId\tsysName\tdesc\tsetXIT001 A\tsetXIT002 B"]
    lines += [f"1\t{g}\td\t0.5\t-1.25" for g in genes]
    return "\n".join(lines) + "\n"


SYNTHETIC_MEMBERS: dict[str, str] = {
    "g/Keio/genes.tab": GENES_TAB,
    "html/Keio_ML9_set16_set19/fit_logratios.tab": _fitness_table(["b0001", "b0002"]),
    "html/Keio_ML9_set28_set29/fit_logratios.tab": _fitness_table(["b0002", "b0003"]),
    "html/Keio_ML9_set30/fit_logratios.tab": _fitness_table(["b0003", "b0018"]),
}


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


@pytest.fixture
def synthetic_mirror(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """Build tiny stand-ins for all five artifacts, re-pin them, and deposit them.

    ``RAW_ARTIFACTS`` and ``TARBALL_MEMBERS`` are monkeypatched to the synthetic bytes'
    hashes, so the real deposit, manifest and reader code paths run unchanged. Returns
    the ``data_root`` the mirror was deposited under.
    """
    import tarfile

    src = tmp_path / "src"
    src.mkdir()
    payloads: dict[str, bytes] = {
        mut.S1_TABLE_REL: b"synthetic S1 table",
        mut.EXPS_USED_REL: EXPS_USED.encode(),
        mut.FIGSHARE_README_REL: b"synthetic README",
    }
    s13 = src / "s13.xlsx"
    _write_s13(s13, GOOD_S13_ROWS)
    payloads[mut.S13_TABLE_REL] = s13.read_bytes()

    tarball = src / "RBTnSeq.tar.gz"
    with tarfile.open(tarball, mode="w:gz") as archive:
        directory = tarfile.TarInfo("g/Keio")
        directory.type = tarfile.DIRTYPE
        archive.addfile(directory)
        for member, text in SYNTHETIC_MEMBERS.items():
            data = text.encode()
            info = tarfile.TarInfo(member)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    payloads[mut.TARBALL_REL] = tarball.read_bytes()

    sources: dict[str, str | Path] = {}
    artifacts = []
    for record in mut.RAW_ARTIFACTS:
        path = src / Path(record.rel).name
        path.write_bytes(payloads[record.rel])
        sources[record.rel] = path
        artifacts.append(
            record.model_copy(
                update={
                    "sha256": _sha(payloads[record.rel]),
                    "bytes": len(payloads[record.rel]),
                }
            )
        )
    monkeypatch.setattr(mut, "RAW_ARTIFACTS", tuple(artifacts))
    pins = {m: _sha(t.encode()) for m, t in SYNTHETIC_MEMBERS.items()}
    pins["g/Keio"] = "0" * 64  # a directory member, to reach the not-a-file refusal
    monkeypatch.setattr(mut, "TARBALL_MEMBERS", pins)

    data_root = str(tmp_path / "root")
    mut.deposit_raw_mirror(
        sources=sources, retrieved_at="2026-10-07", data_root=data_root
    )
    return data_root


def test_raw_mirror_dir_reads_data_root_from_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no explicit root, the mirror lives under ``$DATA_ROOT``."""
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    assert mut.raw_mirror_dir() == tmp_path / mut.RAW_DIR_REL


class TestSyntheticMirror:
    """The deposit, manifest and every reader, on a mirror built in ``tmp_path``."""

    def test_deposit_writes_every_file_and_a_manifest_naming_its_retrieval(
        self, synthetic_mirror: str
    ) -> None:
        """Five files land at their pinned paths; the manifest records how to re-get them."""
        root = mut.raw_mirror_dir(synthetic_mirror)
        manifest = mut.load_manifest(synthetic_mirror)
        assert manifest.citation_key == mut.CITATION_KEY
        assert manifest.doi == mut.PAPER_DOI
        assert [f.path for f in manifest.files] == [r.rel for r in mut.RAW_ARTIFACTS]
        for record in mut.RAW_ARTIFACTS:
            assert mut._sha256(root / record.rel) == record.sha256
            stored = next(f for f in manifest.files if f.path == record.rel)
            assert stored.role == "raw_data"
            assert stored.bytes == record.bytes
            assert stored.retrieval is not None
            assert stored.retrieval.method == record.method
            assert stored.retrieval.params == record.params
            assert stored.retrieval.retrieved_at == "2026-10-07"
        assert len(manifest.si_expected) == 4
        assert any("Dub-seq" in line for line in manifest.si_expected)

    def test_redeposit_is_a_no_op_and_a_drifted_file_is_refused(
        self, synthetic_mirror: str
    ) -> None:
        """Matching bytes are left alone; differing bytes are never overwritten."""
        root = mut.raw_mirror_dir(synthetic_mirror)
        sources: dict[str, str | Path] = {
            r.rel: root / r.rel for r in mut.RAW_ARTIFACTS
        }
        target = root / mut.FIGSHARE_README_REL
        stamp = target.stat().st_mtime_ns
        mut.deposit_raw_mirror(sources=sources, data_root=synthetic_mirror)
        assert target.stat().st_mtime_ns == stamp

        clean = root.parent / "clean-readme"
        clean.write_bytes(target.read_bytes())
        target.write_bytes(b"drifted")
        sources[mut.FIGSHARE_README_REL] = clean
        with pytest.raises(RuntimeError, match="exists with a different sha256"):
            mut.deposit_raw_mirror(sources=sources, data_root=synthetic_mirror)

    def test_a_tarball_member_is_read_and_verified(self, synthetic_mirror: str) -> None:
        """The pinned member comes back byte for byte."""
        payload = mut.read_tarball_member("g/Keio/genes.tab", synthetic_mirror)
        assert payload == GENES_TAB.encode()

    def test_a_moved_member_and_a_directory_member_are_refused(
        self, synthetic_mirror: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A member whose bytes moved, or one that is not a file, raises."""
        with pytest.raises(RuntimeError, match="is not a file"):
            mut.read_tarball_member("g/Keio", synthetic_mirror)
        moved = dict(mut.TARBALL_MEMBERS)
        moved["g/Keio/genes.tab"] = "f" * 64
        monkeypatch.setattr(mut, "TARBALL_MEMBERS", moved)
        with pytest.raises(RuntimeError, match="sha256 mismatch"):
            mut.read_tarball_member("g/Keio/genes.tab", synthetic_mirror)

    def test_the_gene_table_and_the_measured_genes(self, synthetic_mirror: str) -> None:
        """Blank symbols stay blank strings; measured genes are the union of the sets."""
        genes = mut.read_feba_gene_table(synthetic_mirror)
        assert [g["sysName"] for g in genes] == ["b0001", "b0002", "b0003", "b0018"]
        assert genes[2]["name"] == ""
        assert mut.measured_gene_ids(synthetic_mirror) == {
            "b0001",
            "b0002",
            "b0003",
            "b0018",
        }

    def test_the_moi_table_evaluates_the_sheet_s_own_formulas(
        self, synthetic_mirror: str
    ) -> None:
        """RB-TnSeq rows with a dose are evaluated, chains resolved, the rest skipped."""
        table = mut.read_moi_table(synthetic_mirror)
        assert sorted(table) == ["set16IT008", "set16IT009", "set30IT064"]
        assert table["set16IT008"].moi == pytest.approx(187.5, rel=1e-12)
        assert table["set16IT009"].dilution == pytest.approx(0.01, rel=1e-12)
        assert table["set16IT009"].moi == pytest.approx(18.75, rel=1e-12)
        assert table["set30IT064"].phage == "N4"
        assert table["set30IT064"].moi == pytest.approx(1937.5, rel=1e-12)
        assert table["set16IT008"].library == "RBTnSeq-BW25113"

    def test_the_experiment_axis_classifies_and_doses_every_experiment(
        self, synthetic_mirror: str
    ) -> None:
        """Starts, controls and challenges; S13 doses first, descriptions second."""
        axis = mut.read_experiment_axis(synthetic_mirror)
        assert axis.counts == {
            "time_zero": 1,
            "no_phage_control": 2,
            "phage": 4,
            "phages": 3,
        }
        assert axis.phages == ("N4", "P2", "T2")
        assert [a.exp_name for a in axis.time_zero] == ["set16IT001"]
        assert [a.exp_name for a in axis.controls] == ["set16IT007", "set28IT004"]
        by_name = {a.exp_name: a for a in axis.challenges}
        assert by_name["set16IT008"].moi_source == "s13_table"
        assert by_name["set16IT008"].moi == pytest.approx(187.5, rel=1e-12)
        assert by_name["set16IT019"].moi_source == "experiment_description"
        assert by_name["set16IT019"].moi == pytest.approx(0.01875, rel=1e-12)
        assert (by_name["set19IT073"].phage, by_name["set19IT073"].moi) == ("P2", 0.1)
        assert by_name["set30IT064"].analysis_set == "Keio_ML9_set30"
        assert by_name["set16IT008"].time_zero_set == "22-Nov-16 Keio_ML9_set16"
        assert by_name["set16IT008"].g_mean == pytest.approx(659.5)
        assert by_name["set16IT008"].max_fit == pytest.approx(16.75)

    def test_an_unreadable_experiment_name_stops_the_parse(
        self, synthetic_mirror: str
    ) -> None:
        """A row whose name has no ``setNN`` prefix is refused, not guessed."""
        path = mut.raw_mirror_dir(synthetic_mirror) / mut.EXPS_USED_REL
        path.write_text(EXPS_USED + "badname\tTime0\tx\t1.0\t1.0\n")
        with pytest.raises(ValueError, match="unreadable experiment name"):
            mut.read_experiment_axis(synthetic_mirror)


@pytest.mark.parametrize(
    "rows,message",
    [
        (
            [_s13_row(2, "RBTnSeq-BW25113", "set16IT008", "T2", "d", 6e10, "=K2+1")],
            "unreadable dilution",
        ),
        (
            [
                _s13_row(2, "RBTnSeq-BW25113", "set16IT007", "", "c", None, None),
                _s13_row(
                    3, "RBTnSeq-BW25113", "set16IT008", "T2", "d", 6e10, "=J2*0.1"
                ),
            ],
            "dilution chain starts from a blank cell",
        ),
        (
            [
                _s13_row(
                    2,
                    "RBTnSeq-BW25113",
                    "set16IT008",
                    "T2",
                    "d",
                    6e10,
                    0.1,
                    one_od="=9*10^8",
                )
            ],
            "1 OD cfu/ml is",
        ),
        (
            [
                _s13_row(
                    2,
                    "RBTnSeq-BW25113",
                    "set16IT008",
                    "T2",
                    "d",
                    6e10,
                    0.1,
                    moi="=G2*J2",
                )
            ],
            "not =G\\*J/I",
        ),
    ],
)
def test_an_s13_formula_of_any_other_shape_is_refused(
    tmp_path: Path, rows: list[list[object]], message: str
) -> None:
    """A changed sheet is detected instead of mis-evaluated."""
    path = mut.raw_mirror_dir(str(tmp_path)) / mut.S13_TABLE_REL
    path.parent.mkdir(parents=True)
    _write_s13(path, rows)
    with pytest.raises(ValueError, match=message):
        mut.read_moi_table(str(tmp_path))


# --------------------------------------------------------------------------- #
# The identifier routes on stand-in genomes (hermetic)
# --------------------------------------------------------------------------- #
def _stand_in_genomes() -> tuple[EcoliK12MG1655Genome, EcoliK12BW25113Genome]:
    """Uninitialized genome instances: the crosswalk and resolver are patched out."""
    return (
        EcoliK12MG1655Genome.__new__(EcoliK12MG1655Genome),
        EcoliK12BW25113Genome.__new__(EcoliK12BW25113Genome),
    )


def _fake_crosswalk(monkeypatch: pytest.MonkeyPatch) -> None:
    """Three one-to-one ECK pairs, one of them with disagreeing numbers."""
    from types import SimpleNamespace

    from torchcell.sequence.genome.ecoli.k12 import EckPair

    pairs = (
        EckPair(eck="ECK0001", mg1655="b0001", bw25113="BW25113_0001"),
        EckPair(eck="ECK0002", mg1655="b0002", bw25113="BW25113_0002"),
        EckPair(eck="ECK0018", mg1655="b0018", bw25113="BW25113_4412"),
    )
    crosswalk = SimpleNamespace(pairs=pairs, numeric_disagreements=(pairs[2],))
    monkeypatch.setattr(mut, "eck_crosswalk", lambda mg1655, bw25113: crosswalk)


def test_the_eck_route_counts_maps_and_flags_disagreeing_numbers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Duplicates collapse, the unmapped are named, the disagreeing pair is listed."""
    _fake_crosswalk(monkeypatch)
    mg1655, bw25113 = _stand_in_genomes()
    route = mut.eck_route(
        mg1655, bw25113, ["b0001", "b0002", "b0003", "b0018", "b0001"]
    )
    assert (route.pairs, route.numeric_disagreements) == (3, 1)
    assert (route.requested, route.mapped) == (4, 3)
    assert route.unmapped == ("b0003",)
    assert route.disagreeing == (("b0018", "BW25113_4412"),)
    assert route.fraction == pytest.approx(0.75)
    mapping = mut.eck_mapping(mg1655, bw25113)
    assert {b: pair.bw25113 for b, pair in mapping.items()} == {
        "b0001": "BW25113_0001",
        "b0002": "BW25113_0002",
        "b0018": "BW25113_4412",
    }


def test_the_audit_sends_each_route_the_right_genome_and_names(
    synthetic_mirror: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """b-numbers go to MG1655, symbols (b-number when blank) to BW25113, ECK scored."""
    import pandas as pd

    from torchcell.datasets.bacteria_common import LocusTagReconciliation
    from torchcell.sequence.genome.base import GeneNameStatus

    _fake_crosswalk(monkeypatch)
    mg1655, bw25113 = _stand_in_genomes()
    calls: list[tuple[object, list[str], str]] = []

    def fake_reconcile(
        genome: object, names: pd.Series, *, label: str
    ) -> tuple[pd.Series, LocusTagReconciliation]:
        calls.append((genome, list(names), label))
        unique = len(set(names))
        report = LocusTagReconciliation(
            label=label,
            assembly_set="ecoli_K12_MG1655_ASM584v2",
            gene_namespace="ecoli_k12_mg1655_bnumber",
            unique_names=unique,
            status_histogram={
                status: unique if status is GeneNameStatus.CURRENT else 0
                for status in GeneNameStatus
            },
            layer_histogram={"locus tag": unique},
            remapped=0,
            kept_on_collision=(),
            retired_kept=(),
            ambiguous_kept={},
            case_insensitive=(),
            outside_namespace=(),
        )
        return names.copy(), report

    monkeypatch.setattr(mut, "reconcile_locus_tags", fake_reconcile)
    audit = mut.audit_identifiers(mg1655, bw25113, synthetic_mirror)
    assert audit.genes == 4
    assert calls[0][0] is mg1655
    assert calls[0][1] == ["b0001", "b0002", "b0003", "b0018"]
    assert calls[0][2].endswith("b-numbers-vs-MG1655")
    assert calls[1][0] is bw25113
    assert calls[1][1] == ["thrL", "thrA", "b0003", "mokC"]
    assert calls[1][2].endswith("symbols-vs-BW25113")
    assert audit.b_numbers_vs_mg1655.unique_names == 4
    assert (audit.eck_route.requested, audit.eck_route.mapped) == (4, 3)

    scored = mut.audit_identifiers(
        mg1655, bw25113, synthetic_mirror, measured=["b0001", "b0018"]
    )
    assert (scored.eck_route.requested, scored.eck_route.mapped) == (2, 2)
    assert scored.eck_route.fraction == pytest.approx(1.0)
