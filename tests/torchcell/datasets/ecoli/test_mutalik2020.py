# tests/torchcell/datasets/ecoli/test_mutalik2020
# [[tests.torchcell.datasets.ecoli.test_mutalik2020]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/ecoli/test_mutalik2020
"""Tests for the Mutalik 2020 provenance, sourcing and loader layers.

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
from typing import Any, cast

import pytest

from tests.torchcell.sequence.genome._bacterial_fixtures import (
    BW25113_LOCI,
    MG1655_GAF,
    MG1655_LOCI,
    forbid_network,
    serve_tier,
    write_assembly,
)
from torchcell.data import verify_raw_files
from torchcell.datamodels.media import MEDIA_LIBRARY
from torchcell.datamodels.schema import (
    ASSEMBLY_SET_ACCESSIONS,
    BACTERIAL_ASSEMBLY_SETS,
    AssayType,
    AssemblyReferenceGenome,
    MeasurementType,
    MediaComponentRole,
    TransposonInsertionPerturbation,
    UncertaintyType,
)
from torchcell.datasets.ecoli import mutalik2020 as mut
from torchcell.sequence.genome.ecoli.k12 import (
    BW25113_ASSEMBLY,
    MG1655_ASSEMBLY,
    EcoliK12BW25113Genome,
    EcoliK12MG1655Genome,
)
from torchcell.verification.sourced import ProvenanceGapReason

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
    assert values["no_phage_control"] is mut.NO_PHAGE_CONTROL
    assert "replaced phages with simply the phage dilution buffer" in (
        mut.NO_PHAGE_CONTROL.quote
    )


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


# --------------------------------------------------------------------------- #
# The loader: media, phage dose, records (hermetic)
# --------------------------------------------------------------------------- #
EXPS_HEADER = [
    "SetName",
    "Date_pool_expt_started",
    "Drop",
    "Person",
    "Mutant Library",
    "Description",
    "gDNA plate",
    "gDNA well",
    "Index",
    "Sequenced at",
    "Media",
    "MediaStrength",
    "Growth Method",
    "Group",
    "Temperature",
    "pH",
    "Liquid v. solid",
    "Aerobic_v_Anaerobic",
    "Shaking",
    "Condition_1",
    "Concentration_1",
    "Units_1",
    "Condition_2",
    "Concentration_2",
    "Units_2",
    "Timecourse",
    "Timecourse Sample",
    "Growth Plate ID",
    "Growth Plate wells",
    "StartOD",
    "EndOD",
    "Total Generations",
]


def _exps_row(
    set_name: str,
    index: str,
    description: str,
    *,
    media: str,
    state: str,
    growth_method: str = "flask",
    group: str = "phage",
    shaking: str = "orbital",
    aerobicity: str = "Aerobic",
    temperature: str = "37",
    condition_1: str = "",
    concentration_1: str = "",
    units_1: str = "",
    condition_2: str = "",
    concentration_2: str = "",
    units_2: str = "",
    drop: str = "",
) -> str:
    """One ``exps`` row, in the released column order."""
    values = {
        "SetName": set_name,
        "Date_pool_expt_started": "22-Nov-16",
        "Drop": drop,
        "Person": "Vivek",
        "Mutant Library": "Keio_ML9a",
        "Description": description,
        "Index": index,
        "Media": media,
        "Growth Method": growth_method,
        "Group": group,
        "Temperature": temperature,
        "Liquid v. solid": state,
        "Aerobic_v_Anaerobic": aerobicity,
        "Shaking": shaking,
        "Condition_1": condition_1,
        "Concentration_1": concentration_1,
        "Units_1": units_1,
        "Condition_2": condition_2,
        "Concentration_2": concentration_2,
        "Units_2": units_2,
    }
    return "\t".join(str(values.get(column, "")) for column in EXPS_HEADER)


def _exps(rows: list[str]) -> str:
    """An ``exps`` member: the released header plus ``rows``."""
    return "\n".join(["\t".join(EXPS_HEADER), *rows]) + "\n"


def _released(
    exp_name: str = "set16IT008",
    *,
    media_label: str = "LB_plus_SM_buffer",
    state: str = "liquid",
    condition_2: str | None = None,
    concentration_2: float | None = None,
    units_2: str | None = None,
    temperature_c: float = 37.0,
    aerobicity: str = "aerobic",
) -> mut.ReleasedAssay:
    """A released metadata row, built directly (no file) for the refusal branches."""
    return mut.ReleasedAssay(
        exp_name=exp_name,
        set_name="Keio_ML9_set16_set19",
        media_label=media_label,
        state=state,
        growth_method="48 well microplate; Tecan Infinite F200",
        shaking="orbital",
        aerobicity=aerobicity,
        group="phage",
        mutant_library="Keio_ML9a",
        temperature_c=temperature_c,
        condition_2=condition_2,
        concentration_2=concentration_2,
        units_2=units_2,
        dropped_by_release=False,
    )


def _assay(
    exp_name: str = "set16IT008",
    *,
    kind: str = "phage",
    phage: str | None = "T2",
    moi: float | None = 187.5,
    moi_source: str | None = "s13_table",
) -> mut.Assay:
    """One parsed assay of the axis."""
    return mut.Assay(
        exp_name=exp_name,
        set_name="set16",
        analysis_set="Keio_ML9_set16_set19",
        description="LB_plus_SM_buffer with T2_phage 187.5 MOI",
        kind=cast(Any, kind),
        phage=phage,
        moi=moi,
        moi_source=cast(Any, moi_source),
        time_zero_set="22-Nov-16 Keio_ML9_set16",
        g_mean=659.5,
        max_fit=16.75,
    )


def test_every_phage_of_the_panel_has_an_s12_row_and_a_propagation_host() -> None:
    """The 14 names the S13 Table gives are exactly the keys of the identity tables."""
    assert set(mut.PHAGE_S12_ROWS) == set(mut.PHAGES)
    determined = {
        name: row.value
        for name, row in mut.PHAGE_S12_ROWS.items()
        if row.value is not None
    }
    assert len(determined) == 11
    assert determined["T4"] == "AF158101.6"
    assert determined["lambda cI857"] == "NC_001416.1"
    assert sorted(set(mut.PHAGE_S12_ROWS) - set(determined)) == ["CEV1", "CEV2", "LZ4"]
    assert set(mut.PHAGE_PROPAGATION_HOSTS) <= set(mut.PHAGES)
    assert mut.PHAGE_PROPAGATION_HOSTS == {"P2": "E. coli C", "N4": "E. coli W3350"}


def test_the_phage_perturbation_carries_the_dose_the_accession_and_the_host() -> None:
    """A challenge's typed edit: name, dsDNA, accession, propagation host, MOI, titer."""
    row = mut.MoiRow(
        exp_name="set16IT008",
        library="RBTnSeq-BW25113",
        phage="T2",
        description="LB_plus_SM_buffer with T2_phage 187.5 MOI",
        pfu_per_ml=6e10,
        dilution=0.1,
        moi=187.5,
    )
    perturbation = mut.phage_perturbation(
        _assay(), titer_pfu_per_ml=mut.titer_in_culture(row, "liquid")
    )
    assert perturbation.perturbation_type == "phage"
    assert perturbation.name == "T2"
    assert perturbation.genome_type == "dsDNA"
    assert perturbation.genome_accession == "MH751506.1"
    assert perturbation.host_of_propagation == "E. coli BW25113"
    assert perturbation.multiplicity_of_infection == pytest.approx(187.5)
    # 6e10 pfu/mL diluted 0.1 and mixed 350 uL into 700 uL: 3e9 pfu/mL in the culture
    assert perturbation.titer_pfu_per_ml == pytest.approx(3e9, rel=1e-12)
    assert [gap.field for gap in perturbation.provenance_gaps] == ["family"]
    assert perturbation.family is None
    assert perturbation.ncbi_taxid is None


def test_the_three_undetermined_phages_carry_no_accession() -> None:
    """A phage whose genome the sheet calls "Not determined" stores None, not a guess."""
    perturbation = mut.phage_perturbation(
        _assay(phage="CEV1", moi=0.0171875), titer_pfu_per_ml=None
    )
    assert perturbation.genome_accession is None
    assert perturbation.multiplicity_of_infection == pytest.approx(0.0171875)


def test_p2_and_n4_name_the_strain_their_stock_was_grown_on() -> None:
    """The two phages the Methods propagate on another host say so on the record."""
    assert (
        mut.phage_perturbation(
            _assay(phage="P2", moi=0.5625), titer_pfu_per_ml=None
        ).host_of_propagation
        == "E. coli C"
    )
    assert (
        mut.phage_perturbation(
            _assay(phage="N4", moi=1937.5), titer_pfu_per_ml=None
        ).host_of_propagation
        == "E. coli W3350"
    )


def test_an_unstated_moi_is_a_typed_gap_not_a_borrowed_dose() -> None:
    """A phage challenge always has a dose, so an absent MOI is a ProvenanceGap."""
    perturbation = mut.phage_perturbation(
        _assay(moi=None, moi_source=None), titer_pfu_per_ml=None
    )
    assert perturbation.multiplicity_of_infection is None
    gaps = {gap.field: gap for gap in perturbation.provenance_gaps}
    assert sorted(gaps) == ["family", "multiplicity_of_infection"]
    dose_gap = gaps["multiplicity_of_infection"]
    assert dose_gap.reason == ProvenanceGapReason.not_reported_by_primary
    assert dose_gap.looked_in is not None
    assert "set16IT008" in str(dose_gap.looked_in.page)
    assert dose_gap.note is not None and "borrowed" in dose_gap.note


def test_a_control_has_no_phage_so_it_cannot_be_given_one() -> None:
    """``phage_perturbation`` refuses a start sample or a no-phage control."""
    with pytest.raises(ValueError, match="not a phage challenge"):
        mut.phage_perturbation(
            _assay(kind="no_phage_control", phage=None, moi=None, moi_source=None),
            titer_pfu_per_ml=None,
        )


def test_the_culture_titer_is_only_computed_where_the_volume_is_stated() -> None:
    """The planktonic 350 + 350 uL is stated; the solid format's volume is not."""
    row = mut.MoiRow(
        exp_name="set30IT064",
        library="RBTnSeq-BW25113",
        phage="N4",
        description="N4_phage_1937.5_MOI",
        pfu_per_ml=6.2e11,
        dilution=0.1,
        moi=1937.5,
    )
    assert mut.titer_in_culture(row, "liquid") == pytest.approx(3.1e10, rel=1e-12)
    assert mut.titer_in_culture(row, "solid") is None
    assert mut.titer_in_culture(None, "liquid") is None


def test_the_three_media_are_lb_based_and_state_no_amount_mutalik_did_not_give() -> (
    None
):
    """Every medium joins on the ``LB`` library base and invents no formulation."""
    assert sorted(mut.MEDIA_BY_LABEL) == ["LB", "LB_agar", "LB_plus_SM_buffer"]
    for media in mut.MEDIA_BY_LABEL.values():
        assert media.base_medium == "LB"
        assert media.base_medium in MEDIA_LIBRARY
        assert media is not MEDIA_LIBRARY["LB"]
        assert media is not MEDIA_LIBRARY["LB_AGAR"]
        for component in media.components:
            if component.compound.name in (
                "tryptone",
                "yeast extract",
                "sodium chloride",
            ):
                assert component.concentration is None
    assert mut.MUTALIK2020_LB.state == "liquid"
    assert mut.MUTALIK2020_LB_SM_BUFFER.state == "liquid"
    assert mut.MUTALIK2020_LB_AGAR_KAN.state == "solid"
    buffer_names = {c.compound.name for c in mut.MUTALIK2020_LB_SM_BUFFER.components}
    assert {
        "SM buffer (Teknova)",
        "calcium chloride",
        "magnesium sulfate",
    } <= buffer_names
    kanamycin = next(
        c
        for c in mut.MUTALIK2020_LB_AGAR_KAN.components
        if c.compound.name == "kanamycin"
    )
    assert kanamycin.role == MediaComponentRole.selection_agent
    assert kanamycin.concentration is not None
    assert kanamycin.concentration.unit is not None
    assert (kanamycin.concentration.value, kanamycin.concentration.unit.value) == (
        50.0,
        "ug/mL",
    )


def test_the_planktonic_and_solid_environments_are_not_one_environment() -> None:
    """The format lands on the medium, its state, and the exposure duration."""
    liquid = mut.environment(_assay(), _released(), None)
    solid = mut.environment(
        _assay("set30IT064", phage="N4", moi=1937.5),
        _released(
            "set30IT064",
            media_label="LB_agar",
            state="solid",
            condition_2="Kan",
            concentration_2=50.0,
            units_2="ug/ml",
        ),
        None,
    )
    assert liquid.media is mut.MUTALIK2020_LB_SM_BUFFER
    assert solid.media is mut.MUTALIK2020_LB_AGAR_KAN
    assert (liquid.media.state, solid.media.state) == ("liquid", "solid")
    assert liquid.duration_hours == pytest.approx(8.0)
    assert solid.duration_hours is None
    assert [gap.field for gap in liquid.provenance_gaps] == []
    assert [gap.field for gap in solid.provenance_gaps] == ["duration_hours"]
    assert liquid.media.name != solid.media.name
    assert liquid.temperature is not None and liquid.temperature.value == 37.0
    assert liquid.aerobicity == "aerobic"


def test_a_no_phage_control_environment_carries_no_phage_at_all() -> None:
    """The controls replaced the phage with buffer, so no zero-MOI phage is invented."""
    control = mut.environment(
        _assay(
            "set16IT007", kind="no_phage_control", phage=None, moi=None, moi_source=None
        ),
        _released("set16IT007"),
        None,
    )
    assert control.perturbations == []
    assert control.media is mut.MUTALIK2020_LB_SM_BUFFER
    challenge = mut.environment(_assay(), _released(), None)
    assert [p.perturbation_type for p in challenge.perturbations] == ["phage"]


def test_the_environment_refuses_a_release_row_its_medium_contradicts() -> None:
    """The hardcoded media are checked against the released row, not trusted blindly."""
    with pytest.raises(ValueError, match="is 'liquid'"):
        mut.environment(_assay(), _released(state="solid"), None)
    with pytest.raises(ValueError, match="carries kanamycin at"):
        mut.environment(
            _assay(),
            _released(media_label="LB_agar", state="solid", condition_2=None),
            None,
        )
    with pytest.raises(ValueError, match="which .* does not carry"):
        mut.environment(
            _assay(),
            _released(condition_2="Kan", concentration_2=50.0, units_2="ug/ml"),
            None,
        )


def test_the_released_metadata_reader_types_the_format_and_the_oxygen_regime(
    tmp_path: Path,
) -> None:
    """A row is read into a ``ReleasedAssay``; an unreadable value stops the parse."""
    path = tmp_path / "exps"
    path.write_text(
        _exps(
            [
                _exps_row(
                    "Keio_ML9_set30",
                    "IT064",
                    "N4_phage_1937.5_MOI",
                    media="LB_agar",
                    state="Solid",
                    growth_method="plate",
                    shaking="",
                    condition_1="N4_phage",
                    concentration_1="1937.5",
                    units_1="MOI",
                    condition_2="Kan",
                    concentration_2="50",
                    units_2="ug/ml",
                ),
                _exps_row(
                    "Keio_ML9_set30",
                    "IT066",
                    "NoPhageControl",
                    media="LB_agar",
                    state="Solid",
                    group="nophagecontrol",
                    growth_method="plate",
                    shaking="",
                    condition_2="Kan",
                    concentration_2="50",
                    units_2="ug/ml",
                    drop="TRUE",
                ),
            ]
        )
    )
    assays = mut.read_released_assays({"Keio_ML9_set30": path})
    assert sorted(assays) == ["set30IT064", "set30IT066"]
    challenge = assays["set30IT064"]
    assert challenge.state == "solid"
    assert challenge.aerobicity == "aerobic"
    assert challenge.media_label == "LB_agar"
    assert challenge.temperature_c == 37.0
    assert (challenge.condition_2, challenge.concentration_2, challenge.units_2) == (
        "Kan",
        50.0,
        "ug/ml",
    )
    assert challenge.mutant_library == "Keio_ML9a"
    assert challenge.dropped_by_release is False
    assert assays["set30IT066"].dropped_by_release is True


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"state": "Semisolid"}, "unreadable format"),
        ({"aerobicity": "Hypoxic"}, "unreadable oxygen regime"),
        ({"temperature": " "}, "states no temperature"),
    ],
)
def test_an_unreadable_released_row_stops_the_parse(
    tmp_path: Path, kwargs: dict[str, str], message: str
) -> None:
    """A value outside the released vocabulary is refused, never defaulted."""
    path = tmp_path / "exps"
    path.write_text(
        _exps(
            [
                _exps_row(
                    "Keio_ML9_set30",
                    "IT064",
                    "N4_phage_1937.5_MOI",
                    media="LB_agar",
                    **cast(Any, {"state": "Solid", **kwargs}),
                )
            ]
        )
    )
    with pytest.raises(ValueError, match=message):
        mut.read_released_assays({"Keio_ML9_set30": path})


def test_two_sets_claiming_one_experiment_name_is_refused(tmp_path: Path) -> None:
    """A name two analysis sets both release is a collision, not an overwrite."""
    first = tmp_path / "a"
    second = tmp_path / "b"
    row = _exps_row(
        "Keio_ML9_set30", "IT064", "N4_phage_1937.5_MOI", media="LB", state="Liquid"
    )
    first.write_text(_exps([row]))
    second.write_text(_exps([row]))
    with pytest.raises(ValueError, match="released by two analysis sets"):
        mut.read_released_assays({"a": first, "b": second})


def test_fitness_columns_key_on_the_experiment_name_not_the_description() -> None:
    """The release labels a column ``"<expName> <expDescription>"``."""
    assert mut.fitness_columns(
        [
            "locusId",
            "sysName",
            "desc",
            "set16IT001 Time0",
            "set16IT008 LB_plus_SM_buffer with T2_phage 187.5 MOI",
        ]
    ) == {
        "set16IT001": "set16IT001 Time0",
        "set16IT008": "set16IT008 LB_plus_SM_buffer with T2_phage 187.5 MOI",
    }


def test_the_raw_pins_cover_every_file_the_build_reads() -> None:
    """``raw_pins`` is the two linked artifacts plus the twelve extracted members."""
    pins = mut.raw_pins()
    assert sorted(pins) == sorted([*mut.LINKED_ARTIFACTS, *mut.RAW_MEMBERS])
    assert len(mut.RAW_MEMBERS) == 12
    assert all(len(digest) == 64 for digest in pins.values())
    assert pins[mut.S13_NAME] == mut.artifact(mut.S13_TABLE_REL).sha256
    for raw_rel, member in mut.RAW_MEMBERS.items():
        assert member == f"html/{raw_rel}"
        assert pins[raw_rel] == mut.TARBALL_MEMBERS[member]


def test_the_phenotype_is_a_signed_log2_ratio_with_the_release_s_own_se() -> None:
    """The SE is an SE of the estimate, so it is stored as-is and the SE is derived."""
    phenotype = mut.build_phenotype(
        -4.0689, 0.2275, exp_name="set16IT008", state="liquid"
    )
    assert phenotype.measurement_type == MeasurementType.log2_ratio
    assert phenotype.assay_type == AssayType.pooled_competitive_growth_barcode
    assert phenotype.environment_response == pytest.approx(-4.0689)
    assert phenotype.environment_response_uncertainty == pytest.approx(0.2275)
    assert (
        phenotype.environment_response_uncertainty_type
        == UncertaintyType.standard_error
    )
    assert phenotype.environment_response_se == pytest.approx(0.2275)
    assert phenotype.screen_id == "Keio:set16IT008"
    assert phenotype.n_samples is None and phenotype.sample_unit is None
    assert sorted(gap.field for gap in phenotype.provenance_gaps) == [
        "n_samples",
        "sample_unit",
    ]
    assert "planktonic" in str(phenotype.units)
    assert "solid-agar" in str(
        mut.build_phenotype(1.0, 0.1, exp_name="set30IT064", state="solid").units
    )


def test_the_genotype_records_that_the_locus_tag_was_derived() -> None:
    """A BW25113 tag reached from an MG1655 b-number says so on the perturbation."""
    genotype = mut.build_genotype(
        mut.GeneMapping(
            b_number="b0003",
            eck="ECK0003",
            locus_tag="BW25113_4412",
            perturbed_gene_name="hokC",
            numerics_agree=False,
        )
    )
    (perturbation,) = genotype.perturbations
    assert isinstance(perturbation, TransposonInsertionPerturbation)
    assert perturbation.systematic_gene_name == "BW25113_4412"
    assert perturbation.perturbed_gene_name == "hokC"
    assert perturbation.gene_namespace == "ecoli_k12_bw25113_locus_tag"
    assert perturbation.identifier_mapping is not None
    assert perturbation.identifier_mapping.source_identifier == "b0003"
    assert perturbation.identifier_mapping.route == "eck_crosswalk"
    assert perturbation.transposon == "Tn5 transpososome"
    assert perturbation.library_pool == "Keio_ML9a"
    assert perturbation.barcode is None
    assert perturbation.insertion_position is None


def test_the_reference_is_the_typical_gene_of_the_same_experiment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Gene fitness is normalized to 0, which is what the reference phenotype holds.

    ``assembly_reference`` is patched out: it reads the deposited assembly report, so the
    real one would make this test need the genomes tier. ``_assembly_pin`` builds the same
    pin from the schema's own ``BACTERIAL_ASSEMBLY_SETS`` and ``ASSEMBLY_SET_ACCESSIONS``,
    which is what keeps the assertions below about the pin honest.
    """
    monkeypatch.setattr(mut, "assembly_reference", _assembly_pin)
    reference = mut.build_reference(
        "PhageRbTnseqMutalik2020Dataset",
        mut.environment(_assay(), _released(), None),
        exp_name="set16IT008",
        state="liquid",
    )
    assert reference.phenotype_reference.environment_response == 0.0
    assert reference.phenotype_reference.screen_id == "Keio:set16IT008"
    assert reference.genome_reference.strain == "BW25113"
    assert reference.genome_reference.assembly_set == "ecoli_K12_BW25113_ASM75055v1"
    assert reference.environment_reference.media is mut.MUTALIK2020_LB_SM_BUFFER


def _census_records(census: dict[str, tuple[str, str, int]]) -> list[dict[str, Any]]:
    """Records carrying only what ``experiment_census`` reads."""
    out: list[dict[str, Any]] = []
    for exp_name, (state, kind, count) in census.items():
        perturbations = [{"perturbation_type": "phage"}] if kind == "phage" else []
        for _ in range(count):
            out.append(
                {
                    "experiment": {
                        "phenotype": {"screen_id": f"{mut.ORG_ID}:{exp_name}"},
                        "environment": {
                            "perturbations": perturbations,
                            "media": {"state": state},
                        },
                    }
                }
            )
    return out


def test_the_censuses_pin_the_set_kind_and_format_splits() -> None:
    """The three SUPPLEMENTARY rows, over records shaped like the release's."""
    release = {
        "set16IT008": (
            "liquid",
            "phage",
            mut.EXPECTED_SET_CENSUS["Keio_ML9_set16_set19"],
        ),
        "set28IT005": (
            "liquid",
            "phage",
            mut.EXPECTED_SET_CENSUS["Keio_ML9_set28_set29"],
        ),
        "set30IT064": ("solid", "phage", mut.EXPECTED_SET_CENSUS["Keio_ML9_set30"]),
    }
    rows = {row.name: row for row in mut.experiment_census(_census_records(release))}
    assert sorted(rows) == [
        "analysis_set_census",
        "assay_format_census",
        "experiment_kind_census",
    ]
    assert rows["analysis_set_census"].passed
    assert rows["analysis_set_census"].details["observed"] == mut.EXPECTED_SET_CENSUS
    # every record above is a challenge and the liquid/solid split is by set, so the
    # kind row must FAIL (no controls) while the format row passes
    assert not rows["experiment_kind_census"].passed
    assert "is not" in rows["experiment_kind_census"].message
    assert rows["assay_format_census"].passed
    assert sum(mut.EXPECTED_KIND_CENSUS.values()) == mut.EXPECTED_RECORDS
    assert sum(mut.EXPECTED_FORMAT_CENSUS.values()) == mut.EXPECTED_RECORDS


def _release_shaped_axis() -> tuple[mut.ExperimentAxis, dict[str, mut.ReleasedAssay]]:
    """An axis with the release's shape: 58 / 9 / 11 kept experiments and 10 controls.

    Seven controls in set16/19 (six in the planktonic buffer culture, one in plain LB),
    two in set28/29 (both buffer) and one solid-agar control in set30, plus one start
    sample, which is never record-bearing.
    """
    layout = {
        "Keio_ML9_set16_set19": ("set16", 58, ["LB_plus_SM_buffer"] * 6 + ["LB"]),
        "Keio_ML9_set28_set29": ("set28", 9, ["LB_plus_SM_buffer"] * 2),
        "Keio_ML9_set30": ("set30", 11, ["LB_agar"]),
    }
    assays = [
        _assay("set16IT001", kind="time_zero", phage=None, moi=None, moi_source=None)
    ]
    released = {"set16IT001": _released("set16IT001")}
    for analysis_set, (set_name, n_kept, control_media) in layout.items():
        for index in range(n_kept):
            exp_name = f"{set_name}IT{100 + index:03d}"
            is_control = index < len(control_media)
            media_label = (
                control_media[index]
                if is_control
                else ("LB_agar" if set_name == "set30" else "LB_plus_SM_buffer")
            )
            assays.append(
                _assay(
                    exp_name,
                    kind="no_phage_control" if is_control else "phage",
                    phage=None if is_control else "T2",
                    moi=None if is_control else 187.5,
                    moi_source=None if is_control else "s13_table",
                ).model_copy(
                    update={"set_name": set_name, "analysis_set": analysis_set}
                )
            )
            released[exp_name] = _released(
                exp_name,
                media_label=media_label,
                state="solid" if media_label == "LB_agar" else "liquid",
            )
    return mut.ExperimentAxis(assays=tuple(assays)), released


def test_the_declared_controls_are_the_planktonic_buffer_controls() -> None:
    """#888: 8 buffer controls are declared; the plain-LB and solid ones are not."""
    axis, released = _release_shaped_axis()
    assert axis.counts["no_phage_control"] == 10
    kept, dropped = mut.kept_assays(axis, released)
    assert (len(kept), dropped) == (78, [])
    assert mut.records_per_experiment(kept) == {
        "Keio_ML9_set16_set19": 3667,
        "Keio_ML9_set28_set29": 3695,
        "Keio_ML9_set30": 3673,
    }
    declared = mut.declared_unperturbed_records(axis, released)
    assert declared == 6 * 3667 + 2 * 3695 == 29392
    # the two undeclared controls are what the gate reads as a medium edit
    assert mut.EXPECTED_KIND_CENSUS["no_phage_control"] - declared == 3667 + 3673
    assert mut.UNPERTURBED_CONTROL_MEDIA_LABEL == "LB_plus_SM_buffer"


def test_a_control_whose_medium_has_no_library_base_is_not_declared() -> None:
    """A dropped control contributes no record, so it is never declared."""
    axis, released = _release_shaped_axis()
    target = "set28IT100"
    released[target] = released[target].model_copy(update={"media_label": "M9"})
    kept, dropped = mut.kept_assays(axis, released)
    assert dropped == [target]
    # set28/29 now has 8 kept experiments, which its frozen census does not divide
    with pytest.raises(ValueError, match="do not divide over its 8 kept"):
        mut.declared_unperturbed_records(axis, released)


def test_records_per_experiment_refuses_a_census_the_experiments_do_not_span() -> None:
    """Every pinned analysis set must have kept experiments."""
    axis, released = _release_shaped_axis()
    kept, _ = mut.kept_assays(axis, released)
    with pytest.raises(ValueError, match="kept experiments span"):
        mut.records_per_experiment(
            [a for a in kept if a.analysis_set != "Keio_ML9_set30"]
        )


def test_an_unreadable_screen_id_stops_the_census() -> None:
    """The census reads the experiment back out of the screen id, or refuses."""
    with pytest.raises(ValueError, match="unreadable screen_id"):
        mut.experiment_census(
            [
                {
                    "experiment": {
                        "phenotype": {"screen_id": "Keio:not-an-experiment"},
                        "environment": {
                            "perturbations": [],
                            "media": {"state": "liquid"},
                        },
                    }
                }
            ]
        )


def test_stored_tags_are_loci_accepts_a_pseudogene_and_names_a_stray() -> None:
    """A pseudogene locus resolves to itself as ``non_gene_feature``, which passes."""
    from torchcell.sequence.genome.base import GeneNameResolution, GeneNameStatus

    table = {
        "BW25113_0001": GeneNameResolution(
            input_name="BW25113_0001",
            systematic_name="BW25113_0001",
            status=GeneNameStatus.CURRENT,
        ),
        "BW25113_0004": GeneNameResolution(
            input_name="BW25113_0004",
            systematic_name="BW25113_0004",
            status=GeneNameStatus.NON_GENE_FEATURE,
        ),
        "BW25113_9999": GeneNameResolution(
            input_name="BW25113_9999",
            systematic_name=None,
            status=GeneNameStatus.RETIRED,
        ),
    }
    good = mut.stored_tags_are_loci(
        ["BW25113_0001", "BW25113_0004"], lambda tag: table[tag]
    )
    assert good.passed
    assert good.details["statuses"] == {"current": 1, "non_gene_feature": 1}
    bad = mut.stored_tags_are_loci(list(table), lambda tag: table[tag])
    assert not bad.passed
    assert bad.details["elsewhere"] == ["BW25113_9999"]


# --------------------------------------------------------------------------- #
# The ECK route and the build, end to end on the synthetic K-12 pair (hermetic)
# --------------------------------------------------------------------------- #
@pytest.fixture
def k12(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[EcoliK12MG1655Genome, EcoliK12BW25113Genome]:
    """Both synthetic K-12 genomes, built from the shared fixtures; no network."""
    files = write_assembly(tmp_path / "tier", MG1655_ASSEMBLY, MG1655_LOCI, MG1655_GAF)
    files |= write_assembly(tmp_path / "tier", BW25113_ASSEMBLY, BW25113_LOCI)
    forbid_network(monkeypatch)
    serve_tier(monkeypatch, files)
    return (
        EcoliK12MG1655Genome(genome_root=str(tmp_path / "mg1655"), overwrite=True),
        EcoliK12BW25113Genome(genome_root=str(tmp_path / "bw25113"), overwrite=True),
    )


#: The genes of the synthetic release. b0001-b0004 have a one-to-one ECK pair in the
#: synthetic BW25113 annotation; b0005's ECK is on two BW25113 loci and b0006/b0007's
#: ECKs are absent, so those three are the ``no_one_to_one_eck_pair`` drops.
SYNTHETIC_GENES = ["b0001", "b0002", "b0003", "b0004", "b0005", "b0006", "b0007"]
SYNTHETIC_MAPPED = {
    "b0001": "BW25113_0001",
    "b0002": "BW25113_0002",
    "b0003": "BW25113_4412",
    "b0004": "BW25113_0004",
}


def test_the_eck_route_places_the_genes_it_can_and_names_the_rest(
    k12: tuple[EcoliK12MG1655Genome, EcoliK12BW25113Genome],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Four of seven place; the three without a one-to-one pair are named, not guessed."""
    mg1655, bw25113 = k12
    monkeypatch.setattr(mut, "MIN_ECK_ROUTE_FRACTION", 0.5)
    mapping, report = mut.map_measured_genes(
        SYNTHETIC_GENES, mg1655, bw25113, label="synthetic"
    )
    assert {b: m.locus_tag for b, m in mapping.items()} == SYNTHETIC_MAPPED
    assert (report.n_measured_genes, report.n_mapped) == (7, 4)
    assert report.mapped_fraction == pytest.approx(4 / 7)
    assert report.unmapped == ("b0005", "b0006", "b0007")
    # b0003 carries ECK0003, which BW25113 puts on BW25113_4412: the numbers disagree,
    # which is why no string surgery relates the two namespaces
    assert report.numeric_disagreements == (("b0003", "BW25113_4412", "ECK0003"),)
    # the stored common name is the BW25113 symbol when it resolves back to the locus
    assert mapping["b0003"].perturbed_gene_name == "hokC"
    assert mapping["b0001"].perturbed_gene_name == "thrL"
    assert report.n_symbol_names + report.n_tag_names == 4
    assert report.reconcile_layer_histogram["gene synonym"] == 4


def test_the_eck_route_stops_below_its_threshold(
    k12: tuple[EcoliK12MG1655Genome, EcoliK12BW25113Genome],
) -> None:
    """The release route places 0.9949; a worse one stops the build, it does not drop."""
    mg1655, bw25113 = k12
    with pytest.raises(ValueError, match=r"places 4 of 7 measured genes \(0\.5714\)"):
        mut.map_measured_genes(SYNTHETIC_GENES, mg1655, bw25113, label="threshold")


def _table(genes: list[str], columns: dict[str, float]) -> str:
    """A released per-gene table: the three key columns plus one column per experiment."""
    header = ["locusId", "sysName", "desc", *columns]
    lines = ["\t".join(header)]
    for index, gene in enumerate(genes, start=1):
        row = [str(index), gene, f"product of {gene}"]
        row += [f"{value + index / 100:.6f}" for value in columns.values()]
        lines.append("\t".join(row))
    return "\n".join(lines) + "\n"


def _quality(columns: dict[str, float], usable: set[str]) -> str:
    """A released ``fit_quality.tab``: one row per experiment with its ``u`` flag."""
    lines = ["name\tshort\tt0set\tgMean\tmaxFit\tu"]
    for label in columns:
        name = label.split(" ", 1)[0]
        lines.append(
            f"{name}\tshort\tt0\t600.0\t12.0\t{'TRUE' if name in usable else 'FALSE'}"
        )
    return "\n".join(lines) + "\n"


#: The synthetic release: three analysis sets, six record-bearing experiments and one
#: start sample, matching the seven rows of ``RELEASE_EXPS_USED`` below.
SET_16_19_COLUMNS = {
    "set16IT001 Time0": 0.0,
    "set16IT007 LB_plus_SM_buffer": 0.1,
    "set16IT008 LB_plus_SM_buffer with T2_phage 187.5 MOI": 1.0,
    "set16IT019 LB_plus_SM_buffer with T2_phage 0.01875 MOI": 2.0,
    "set19IT073 P2 dilution 10-1": -1.0,
}
SET_28_29_COLUMNS = {"set28IT004 LB_plus_SM_buffer": 0.2}
SET_30_COLUMNS = {"set30IT064 N4_phage_1937.5_MOI": 3.0}

RELEASE_EXPS_USED = (
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

_LIQUID = {
    "media": "LB_plus_SM_buffer",
    "state": "Liquid",
    "growth_method": "48 well microplate; Tecan Infinite F200",
}
RELEASE_EXPS: dict[str, str] = {
    "Keio_ML9_set16_set19": _exps(
        [
            _exps_row(
                "Keio_ML9_set16",
                "IT001",
                "Time0",
                media="LB",
                state="Liquid",
                group="Time0",
                shaking="200 rpm",
            ),
            _exps_row("Keio_ML9_set16", "IT007", "LB_plus_SM_buffer", **_LIQUID),
            _exps_row(
                "Keio_ML9_set16",
                "IT008",
                "LB_plus_SM_buffer with T2_phage 187.5 MOI",
                condition_1="T2_phage",
                concentration_1="187.5",
                units_1="MOI",
                **_LIQUID,
            ),
            _exps_row(
                "Keio_ML9_set16",
                "IT019",
                "LB_plus_SM_buffer with T2_phage 0.01875 MOI",
                condition_1="T2_phage",
                concentration_1="0.01875",
                units_1="MOI",
                **_LIQUID,
            ),
            _exps_row(
                "Keio_ML9_set19",
                "IT073",
                "P2 dilution 10-1",
                condition_1="P2_phage",
                concentration_1="0.1",
                units_1="MOI",
                **_LIQUID,
            ),
        ]
    ),
    "Keio_ML9_set28_set29": _exps(
        [_exps_row("Keio_ML9_set28", "IT004", "LB_plus_SM_buffer", **_LIQUID)]
    ),
    "Keio_ML9_set30": _exps(
        [
            _exps_row(
                "Keio_ML9_set30",
                "IT064",
                "N4_phage_1937.5_MOI",
                media="LB_agar",
                state="Solid",
                growth_method="plate",
                shaking="",
                condition_1="N4_phage",
                concentration_1="1937.5",
                units_1="MOI",
                condition_2="Kan",
                concentration_2="50",
                units_2="ug/ml",
            )
        ]
    ),
}

RELEASE_MEMBERS: dict[str, str] = {
    "g/Keio/genes.tab": GENES_TAB,
    **{
        f"html/{analysis_set}/exps": text for analysis_set, text in RELEASE_EXPS.items()
    },
    **{
        f"html/{analysis_set}/{leaf}": payload
        for analysis_set, columns in (
            ("Keio_ML9_set16_set19", SET_16_19_COLUMNS),
            ("Keio_ML9_set28_set29", SET_28_29_COLUMNS),
            ("Keio_ML9_set30", SET_30_COLUMNS),
        )
        for leaf, payload in (
            ("fit_logratios.tab", _table(SYNTHETIC_GENES, columns)),
            (
                "fit_standard_error_obs.tab",
                _table(SYNTHETIC_GENES, dict.fromkeys(columns, 0.25)),
            ),
            ("fit_quality.tab", _quality(columns, {"set16IT008"})),
        )
    },
}

RELEASE_S13_ROWS: list[list[object]] = [
    _s13_row(
        2,
        "RBTnSeq-BW25113",
        "set16IT008",
        "T2",
        "LB_plus_SM_buffer with T2_phage 187.5 MOI",
        6e10,
        0.1,
    ),
    _s13_row(3, "RBTnSeq-BW25113", "set19IT073", "P2", "P2 dilution 10-1", 3.6e8, 0.1),
    _s13_row(
        4, "RBTnSeq-BW25113", "set30IT064", "N4", "N4_phage_1937.5_MOI", 6.2e11, 0.1
    ),
]


def _assembly_pin(
    strain: str, *, background: Any = None, data_root: str | None = None
) -> AssemblyReferenceGenome:
    """``assembly_reference`` without the deposited assembly report."""
    assembly_set = BACTERIAL_ASSEMBLY_SETS[strain]
    return AssemblyReferenceGenome(
        species="Escherichia coli",
        strain=strain if background is None else background.name,
        assembly_set=cast(Any, assembly_set),
        assembly_accession=ASSEMBLY_SET_ACCESSIONS[assembly_set][0],
        background=background,
    )


@pytest.fixture
def release_mirror(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    k12: tuple[EcoliK12MG1655Genome, EcoliK12BW25113Genome],
) -> str:
    """A tmp ``DATA_ROOT`` holding a deposited synthetic release the loader can build.

    The five artifacts are stand-ins re-pinned by monkeypatch, the tarball is a real
    ``.tar.gz`` carrying the twelve members the loader extracts, and the genome tier and
    assembly pin are the synthetic K-12 pair, so the whole build runs with no
    ``$DATA_ROOT`` and no network.
    """
    import tarfile

    mg1655, bw25113 = k12
    src = tmp_path / "release"
    src.mkdir()
    s13 = src / "s13.xlsx"
    _write_s13(s13, RELEASE_S13_ROWS)
    payloads: dict[str, bytes] = {
        mut.S1_TABLE_REL: b"synthetic S1 table",
        mut.S13_TABLE_REL: s13.read_bytes(),
        mut.EXPS_USED_REL: RELEASE_EXPS_USED.encode(),
        mut.FIGSHARE_README_REL: b"synthetic README",
    }
    tarball = src / "RBTnSeq.tar.gz"
    with tarfile.open(tarball, mode="w:gz") as archive:
        for member, text in RELEASE_MEMBERS.items():
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
    monkeypatch.setattr(
        mut,
        "TARBALL_MEMBERS",
        {member: _sha(text.encode()) for member, text in RELEASE_MEMBERS.items()},
    )
    data_root = str(tmp_path / "root")
    mut.deposit_raw_mirror(sources=sources, data_root=data_root)
    monkeypatch.setenv("DATA_ROOT", data_root)
    monkeypatch.delenv("TC_DATA_URL", raising=False)
    monkeypatch.setattr(mut, "MIN_ECK_ROUTE_FRACTION", 0.5)
    monkeypatch.setattr(
        mut,
        "bacterial_genome",
        lambda host, strain, data_root=None: mg1655 if strain == "MG1655" else bw25113,
    )
    monkeypatch.setattr(mut, "assembly_reference", _assembly_pin)
    return data_root


class TestSyntheticRelease:
    """The loader over a deposited synthetic release, with no ``$DATA_ROOT``."""

    def test_download_links_the_mirror_and_extracts_the_pinned_members(
        self, tmp_path: Path, release_mirror: str
    ) -> None:
        """Two links and twelve extracted members, every one sha256-verified."""
        root = tmp_path / "linked"
        dataset = mut.PhageRbTnseqMutalik2020Dataset.__new__(
            mut.PhageRbTnseqMutalik2020Dataset
        )
        dataset.root = str(root)
        dataset.download()
        raw = root / "raw"
        for name in mut.LINKED_ARTIFACTS:
            assert (raw / name).is_file()
        for raw_rel in mut.RAW_MEMBERS:
            assert (raw / raw_rel).is_file()
        verify_raw_files(str(raw), mut.raw_pins())

    def test_download_refuses_a_mirror_missing_an_artifact(
        self, tmp_path: Path, release_mirror: str
    ) -> None:
        """A missing artifact raises before ``raw/`` exists, leaving nothing behind."""
        (mut.raw_mirror_dir(release_mirror) / mut.TARBALL_REL).unlink()
        root = tmp_path / "incomplete"
        dataset = mut.PhageRbTnseqMutalik2020Dataset.__new__(
            mut.PhageRbTnseqMutalik2020Dataset
        )
        dataset.root = str(root)
        with pytest.raises(RuntimeError, match="missing from mirror"):
            dataset.download()
        assert not (root / "raw").exists()

    def test_extract_refuses_a_member_the_archive_does_not_carry(
        self, tmp_path: Path, release_mirror: str
    ) -> None:
        """A re-packed archive is detected, never silently followed."""
        with pytest.raises(RuntimeError, match="does not carry"):
            mut.extract_tarball_members(
                {"gone": "html/Keio_ML9_set30/strain_fit.tab"},
                tmp_path / "out",
                release_mirror,
            )

    def test_extract_refuses_two_destinations_for_one_member(
        self, tmp_path: Path, release_mirror: str
    ) -> None:
        """A member map that collapses is a mistake, not a last-writer-wins extract."""
        member = "html/Keio_ML9_set30/exps"
        with pytest.raises(ValueError, match="same tarball member"):
            mut.extract_tarball_members(
                {"a": member, "b": member}, tmp_path / "out", release_mirror
            )

    def test_the_loader_builds_the_synthetic_release_end_to_end(
        self, tmp_path: Path, release_mirror: str
    ) -> None:
        """Six record-bearing experiments x four mapped genes, with the start sample out."""
        root = tmp_path / "dataset"
        dataset = mut.PhageRbTnseqMutalik2020Dataset(root=str(root))
        assert len(dataset) == 24
        assert sorted(dataset.gene_set) == sorted(SYNTHETIC_MAPPED.values())
        references = dataset.experiment_reference_index
        assert references is not None
        assert len(references) == 6
        census: dict[str, int] = {}
        for index in range(len(dataset)):
            screen = dataset[index]["experiment"]["phenotype"]["screen_id"]
            census[screen] = census.get(screen, 0) + 1
        assert census == {
            "Keio:set16IT007": 4,
            "Keio:set16IT008": 4,
            "Keio:set16IT019": 4,
            "Keio:set19IT073": 4,
            "Keio:set28IT004": 4,
            "Keio:set30IT064": 4,
        }
        assert "Keio:set16IT001" not in census
        assert (root / "preprocess" / "build_manifest.json").is_file()

    def test_each_built_record_carries_its_phage_dose_and_derived_mapping(
        self, tmp_path: Path, release_mirror: str
    ) -> None:
        """One challenge record, read back out of the LMDB."""
        root = tmp_path / "dataset"
        dataset = mut.PhageRbTnseqMutalik2020Dataset(root=str(root))
        by_screen: dict[str, Any] = {}
        for index in range(len(dataset)):
            record = dataset[index]
            screen = record["experiment"]["phenotype"]["screen_id"]
            # the first record of each screen, bound by subscript rather than
            # `setdefault` so the value stays traceable from the store to the asserts
            if screen not in by_screen:
                by_screen[screen] = record
        challenge = by_screen["Keio:set16IT008"]["experiment"]
        (phage,) = challenge["environment"]["perturbations"]
        assert phage["perturbation_type"] == "phage"
        assert phage["name"] == "T2"
        assert phage["multiplicity_of_infection"] == pytest.approx(187.5)
        assert phage["titer_pfu_per_ml"] == pytest.approx(3e9, rel=1e-9)
        assert phage["genome_accession"] == "MH751506.1"
        assert challenge["environment"]["media"]["state"] == "liquid"
        assert challenge["environment"]["duration_hours"] == pytest.approx(8.0)
        (perturbation,) = challenge["genotype"]["perturbations"]
        assert perturbation["systematic_gene_name"] in SYNTHETIC_MAPPED.values()
        assert perturbation["identifier_mapping"]["route"] == "eck_crosswalk"
        assert perturbation["identifier_mapping"]["source_identifier"].startswith("b")
        assert perturbation["library_pool"] == "Keio_ML9a"

        control = by_screen["Keio:set16IT007"]["experiment"]
        assert control["environment"]["perturbations"] == []
        solid = by_screen["Keio:set30IT064"]["experiment"]
        assert solid["environment"]["media"]["state"] == "solid"
        assert solid["environment"]["duration_hours"] is None
        (solid_phage,) = solid["environment"]["perturbations"]
        assert solid_phage["name"] == "N4"
        assert solid_phage["titer_pfu_per_ml"] is None
        assert solid_phage["host_of_propagation"] == "E. coli W3350"

    def test_the_dose_of_an_experiment_s13_omits_comes_from_its_description(
        self, tmp_path: Path, release_mirror: str
    ) -> None:
        """``set16IT019`` has no S13 row, so its own description states the MOI."""
        root = tmp_path / "dataset"
        dataset = mut.PhageRbTnseqMutalik2020Dataset(root=str(root))
        for index in range(len(dataset)):
            experiment = dataset[index]["experiment"]
            if experiment["phenotype"]["screen_id"] != "Keio:set16IT019":
                continue
            (phage,) = experiment["environment"]["perturbations"]
            assert phage["multiplicity_of_infection"] == pytest.approx(0.01875)
            # no S13 row means no released stock titer, so the culture titer is unset
            assert phage["titer_pfu_per_ml"] is None
            return
        raise AssertionError("no set16IT019 record was built")

    def test_the_build_accounts_for_every_dropped_synthetic_record(
        self, tmp_path: Path, release_mirror: str
    ) -> None:
        """Seven genes x six experiments in, 24 out, the 18 drops named by rule."""
        import json as _json

        root = tmp_path / "dataset"
        mut.PhageRbTnseqMutalik2020Dataset(root=str(root))
        drops = _json.loads((root / "preprocess" / "dropped_records.json").read_text())
        assert (
            drops["source_records"],
            drops["kept_records"],
            drops["dropped_records"],
        ) == (42, 24, 18)
        assert {
            rule["rule"]: (rule["n_records"], rule["items"]) for rule in drops["rules"]
        } == {
            mut.DROP_MEDIUM_NOT_IN_LIBRARY: (0, []),
            mut.DROP_NO_ECK_PAIR: (18, ["b0005", "b0006", "b0007"]),
        }
        ledger = _json.loads((root / "preprocess" / "assay_ledger.json").read_text())
        assert (
            ledger["used_experiments"],
            ledger["time_zero"],
            ledger["record_bearing"],
        ) == (7, 1, 6)
        assert ledger["kind_census"] == {"phage": 4, "no_phage_control": 2}
        assert ledger["format_census"] == {"liquid": 5, "solid": 1}
        assert ledger["media_census"] == {"LB_plus_SM_buffer": 5, "LB_agar": 1}
        assert ledger["phage_census"] == {"T2": 2, "P2": 1, "N4": 1}
        assert ledger["moi_source_census"] == {
            "s13_table": 3,
            "experiment_description": 1,
        }
        assert ledger["release_quality_flag"] == {"True": 1, "False": 5}
        assert ledger["dropped_for_medium"] == []
        assert ledger["released_columns"] == 7
        identifiers = _json.loads(
            (root / "preprocess" / "identifier_mapping.json").read_text()
        )
        assert (identifiers["n_measured_genes"], identifiers["n_mapped"]) == (7, 4)

    def test_an_assay_whose_medium_has_no_library_base_is_dropped_and_counted(
        self, tmp_path: Path, release_mirror: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The refusal that keeps a new released label from getting an invented recipe."""
        import json as _json

        monkeypatch.delitem(mut.MEDIA_BY_LABEL, "LB_agar")
        root = tmp_path / "dataset"
        dataset = mut.PhageRbTnseqMutalik2020Dataset(root=str(root))
        assert len(dataset) == 20
        drops = _json.loads((root / "preprocess" / "dropped_records.json").read_text())
        by_rule = {rule["rule"]: rule for rule in drops["rules"]}
        assert by_rule[mut.DROP_MEDIUM_NOT_IN_LIBRARY]["n_records"] == 7
        assert by_rule[mut.DROP_MEDIUM_NOT_IN_LIBRARY]["items"] == ["set30IT064"]
        assert drops["source_records"] == 42 and drops["kept_records"] == 20
        ledger = _json.loads((root / "preprocess" / "assay_ledger.json").read_text())
        assert ledger["dropped_for_medium"] == ["set30IT064"]
        assert ledger["record_bearing"] == 5

    def test_the_build_refuses_a_genome_of_the_wrong_strain(
        self,
        tmp_path: Path,
        release_mirror: str,
        k12: tuple[EcoliK12MG1655Genome, EcoliK12BW25113Genome],
    ) -> None:
        """``ecoli_genome`` is the BW25113 genome; MG1655 is refused by name."""
        mg1655, _ = k12
        with pytest.raises(TypeError, match="needs the BW25113 genome"):
            mut.PhageRbTnseqMutalik2020Dataset(
                root=str(tmp_path / "wrong"), ecoli_genome=mg1655
            )


# --------------------------------------------------------------------------- #
# The built dev store (data-gated)
# --------------------------------------------------------------------------- #
DEV_ROOT = osp.join(
    os.environ.get("DATA_ROOT", ""), "data/torchcell/phage_rbtnseq_mutalik2020"
)
on_dev_store = [
    pytest.mark.data,
    pytest.mark.skipif(
        not osp.isfile(osp.join(DEV_ROOT, "preprocess", "build_manifest.json")),
        reason="requires the built dev-tree LMDB of PhageRbTnseqMutalik2020Dataset",
    ),
]


def test_the_oracles_are_internally_consistent() -> None:
    """The frozen counts agree with each other, with no store mounted."""
    assert sum(mut.EXPECTED_SET_CENSUS.values()) == mut.EXPECTED_RECORDS
    assert sum(mut.EXPECTED_KIND_CENSUS.values()) == mut.EXPECTED_RECORDS
    assert sum(mut.EXPECTED_FORMAT_CENSUS.values()) == mut.EXPECTED_RECORDS
    assert mut.EXPECTED_RECORDS == 286344
    assert mut.EXPECTED_GENES == 3697
    # every solid-agar assay is in set30 and set30 holds nothing else
    assert (
        mut.EXPECTED_FORMAT_CENSUS["solid"] == mut.EXPECTED_SET_CENSUS["Keio_ML9_set30"]
    )


def test_the_verify_provenance_points_at_the_deposited_tarball() -> None:
    """The verifier's provenance names the artifact the records' values come from."""
    provenance = mut.VERIFY_PROVENANCE
    assert provenance.citation_key == mut.CITATION_KEY
    assert provenance.sha256 == mut.artifact(mut.TARBALL_REL).sha256
    assert provenance.retrieved == mut.RETRIEVED_AT


class TestDevStore:
    """The built store's own numbers, read back out of its reports and its LMDB."""

    pytestmark = on_dev_store

    def test_the_build_reports_the_frozen_record_and_gene_counts(self) -> None:
        """Every number the dendron note states, from the three preprocess reports."""
        import json as _json

        preprocess = osp.join(DEV_ROOT, "preprocess")
        drops = _json.loads(Path(preprocess, "dropped_records.json").read_text())
        assert drops["kept_records"] == mut.EXPECTED_RECORDS
        assert drops["source_records"] == 287815
        assert drops["dropped_records"] == 1471
        by_rule = {rule["rule"]: rule for rule in drops["rules"]}
        assert by_rule[mut.DROP_MEDIUM_NOT_IN_LIBRARY]["n_records"] == 0
        assert by_rule[mut.DROP_NO_ECK_PAIR]["n_records"] == 1471
        assert len(by_rule[mut.DROP_NO_ECK_PAIR]["items"]) == 19
        identifiers = _json.loads(
            Path(preprocess, "identifier_mapping.json").read_text()
        )
        assert identifiers["n_mapped"] == mut.EXPECTED_GENES
        assert identifiers["n_measured_genes"] == 3716
        assert identifiers["numeric_disagreements"] == [
            ["b0018", "BW25113_4412", "ECK0018"]
        ]
        ledger = _json.loads(Path(preprocess, "assay_ledger.json").read_text())
        assert (
            ledger["used_experiments"],
            ledger["time_zero"],
            ledger["record_bearing"],
        ) == (99, 21, 78)
        assert ledger["kind_census"] == {"phage": 68, "no_phage_control": 10}
        assert ledger["format_census"] == {"liquid": 67, "solid": 11}
        assert len(ledger["phage_census"]) == 14
        assert sum(ledger["phage_census"].values()) == 68
        assert ledger["moi_source_census"] == {
            "s13_table": 67,
            "experiment_description": 1,
        }
        assert ledger["dropped_for_medium"] == []
        # the release's own usability flag is FALSE for 68 of the 78 kept experiments;
        # the paper states its standard metrics are unsuitable under phage selection,
        # which is why the selection rule is Keio_exps_used.tab and not `u`
        assert ledger["release_quality_flag"] == {"True": 10, "False": 68}

    def test_the_declared_controls_come_from_the_tree_s_own_release_files(self) -> None:
        """#888: the raw experiment list declares 29,392 unedited control records."""
        assert mut.read_declared_unperturbed(DEV_ROOT) == 29392

    def test_the_report_passes_environment_perturbed_on_the_declared_controls(
        self,
    ) -> None:
        """#888: the gate's unedited count equals the declaration in the store's report."""
        import json as _json

        report = _json.loads(
            Path(DEV_ROOT, "preprocess", "verification_report.json").read_text()
        )
        (row,) = [r for r in report["results"] if r["name"] == "environment_perturbed"]
        assert row["passed"]
        assert row["details"]["n_missing"] == 29392
        assert row["details"]["expected_unperturbed"] == 29392

    def test_the_store_serves_phage_challenges_and_phage_free_controls(self) -> None:
        """The LMDB's own length, gene set, reference count and first record."""
        dataset = mut.PhageRbTnseqMutalik2020Dataset(root=DEV_ROOT)
        assert len(dataset) == mut.EXPECTED_RECORDS
        assert len(dataset.gene_set) == mut.EXPECTED_GENES
        references = dataset.experiment_reference_index
        assert references is not None
        assert len(references) == 78
        record = dataset[0]["experiment"]
        assert record["experiment_type"] == "bacterial_environment_response"
        assert record["phenotype"]["measurement_type"] == "log2_ratio"
        assert str(record["phenotype"]["screen_id"]).startswith(f"{mut.ORG_ID}:")
        (perturbation,) = record["genotype"]["perturbations"]
        assert perturbation["gene_namespace"] == "ecoli_k12_bw25113_locus_tag"
        assert perturbation["identifier_mapping"]["route"] == "eck_crosswalk"
