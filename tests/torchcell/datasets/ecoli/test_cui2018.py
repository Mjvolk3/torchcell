# tests/torchcell/datasets/ecoli/test_cui2018.py
# [[tests.torchcell.datasets.ecoli.test_cui2018]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/ecoli/test_cui2018.py
"""Tests for the Cui 2018 genome-wide dCas9 knockdown loader.

Everything outside the ``data``-marked block runs with NO network and NO ``$DATA_ROOT``:
the screen table is synthesized to the shape the released CSV has, the raw mirror is
deposited under ``tmp_path`` by the module's own ``deposit_raw_mirror`` with its pins
monkeypatched to the synthetic digest, and the MG1655 annotation is the real genome
class over a synthetic assembly carrying one locus per retention rule. The synthetic
table exercises every rule once: a kept single-target guide, a kept multi-target guide,
a kept pseudogene target, a guide with two positions in one gene, and one guide per
drop reason.

The ``data``-marked block pins the numbers measured on the REAL bytes (85,381 rows,
78,137 guides, 141,542 records, 4,263 targets) and audits every quote against the
pinned mirror.
"""

from __future__ import annotations

import hashlib
import json
import os
import os.path as osp
from pathlib import Path
from typing import Any

import pytest

import tests.torchcell.sequence.genome._bacterial_fixtures as fixtures
import torchcell.datasets.ecoli.cui2018 as C
from tests.torchcell.sequence.genome._bacterial_fixtures import (
    MG1655_GAF,
    MG1655_LOCI,
    SyntheticLocus,
)
from torchcell.data import ManifestPinMismatchError
from torchcell.datamodels.media import LB
from torchcell.datamodels.schema import (
    BacterialEnvironmentResponseExperiment,
    BacterialEnvironmentResponseExperimentReference,
    MeasurementType,
    SmallMoleculePerturbation,
)
from torchcell.sequence.genome.ecoli.k12 import MG1655_ASSEMBLY, EcoliK12MG1655Genome

# --------------------------------------------------------------------------- #
# The synthetic assembly: MG1655's fixture loci plus a pair sharing one synonym
# --------------------------------------------------------------------------- #
#: ``dup1`` names both of these, so a source cell spelling ``dup1`` is AMBIGUOUS in the
#: pinned assembly and the guide that targets it cannot be written.
AMBIGUOUS_LOCI = [
    SyntheticLocus(
        tag="b0008",
        parts=((100, 111),),
        strand="+",
        symbol="yabX",
        synonyms=("ECK0008", "dup1"),
        product="synthetic duplicate one",
        protein_id="AAC73118.1",
        protein="MQQ",
    ),
    SyntheticLocus(
        tag="b0009",
        parts=((115, 126),),
        strand="-",
        symbol="yabY",
        synonyms=("ECK0009", "dup1"),
        product="synthetic duplicate two",
        protein_id="AAC73119.1",
        protein="MWW",
    ),
]

# --------------------------------------------------------------------------- #
# The synthetic screen table, one guide per retention rule
# --------------------------------------------------------------------------- #
#: Spacer per rule. ``proB`` and ``pro2`` both resolve to ``b0005`` in the fixture
#: annotation, so the two collision guides are the pair that makes that rule fire.
SPACERS: dict[str, str] = {
    "kept_single": "AAAAAAAAAAAAAAAAAACG",
    "kept_multi": "AAAAAAAAAAAAAAAAAAGG",
    "kept_pseudogene": "AAAAAAAAAAAAAAAAAACA",
    "kept_two_positions_one_gene": "AAAAAAAAAAAAAAAAAAAG",
    "drop_no_gene": "AAAAAAAAAAAAAAAAAAGC",
    "drop_mixed": "AAAAAAAAAAAAAAAAAACC",
    "drop_retired": "AAAAAAAAAAAAAAAAAATT",
    "drop_collision_a": "AAAAAAAAAAAAAAAAAATA",
    "drop_collision_b": "AAAAAAAAAAAAAAAAAAAT",
    "drop_ambiguous": "AAAAAAAAAAAAAAAAAAGA",
}
#: ``(rule, targeted gene cells)``; one row per cell, in table order.
TARGETS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("kept_single", ("thrL",)),
    ("drop_no_gene", ("NA",)),
    ("kept_multi", ("thrA", "thrW")),
    ("drop_mixed", ("thrA", "NA")),
    ("drop_retired", ("zzzA",)),
    ("drop_collision_a", ("pro2",)),
    ("drop_collision_b", ("proB",)),
    ("drop_ambiguous", ("dup1",)),
    ("kept_pseudogene", ("yaaP",)),
    ("kept_two_positions_one_gene", ("thrL", "thrL")),
)
#: ``(fit18, fit75)`` per rule; distinct so a record can be traced to its guide.
RESPONSES: dict[str, tuple[float, float]] = {
    rule: (-1.0 - index, -0.5 - index / 2.0) for index, (rule, _) in enumerate(TARGETS)
}
SYNTHETIC_ROWS = sum(len(cells) for _, cells in TARGETS)
SYNTHETIC_GUIDES = len(TARGETS)
SYNTHETIC_KEPT_GUIDES = sum(1 for rule, _ in TARGETS if rule.startswith("kept_"))
SYNTHETIC_RECORDS = SYNTHETIC_KEPT_GUIDES * len(C.SCREENS)
SYNTHETIC_TARGETS = 4
SYNTHETIC_PERTURBATIONS = 5 * len(C.SCREENS)
SYNTHETIC_DROPS: dict[str, int] = {
    C.DROP_NO_GENE: 1,
    C.DROP_MIXED: 1,
    C.DROP_RETIRED: 1,
    C.DROP_COLLISION: 2,
    C.DROP_AMBIGUOUS: 1,
}


def synthetic_screen_csv(
    targets: tuple[tuple[str, tuple[str, ...]], ...] = TARGETS,
    *,
    header: tuple[str, ...] = C.SCREEN_COLUMNS,
    fit_override: dict[tuple[str, int], str] | None = None,
    ntargets_override: dict[str, int] | None = None,
) -> str:
    """The released table's shape over the synthetic loci.

    ``fit_override`` replaces one ``(spacer, row index within the guide)`` cell's
    ``fit18``, and ``ntargets_override`` replaces a guide's declared target count; both
    exist so the structural assertions can be exercised without a second builder.
    """
    lines = [",".join(header)]
    position = 1000
    for rule, cells in targets:
        spacer = SPACERS[rule]
        fit18, fit75 = RESPONSES[rule]
        declared = (ntargets_override or {}).get(spacer, len(cells))
        for index, cell in enumerate(cells):
            position += 7
            in_gene = cell != C.NO_GENE
            value = (fit_override or {}).get((spacer, index), f"{fit18}")
            lines.append(
                ",".join(
                    (
                        spacer,
                        cell,
                        "FALSE" if in_gene else C.NO_GENE,
                        str(position),
                        "+",
                        "TRUE" if in_gene else C.NO_GENE,
                        value,
                        f"{fit75}",
                        str(declared),
                        "" if len(cells) > 1 else "G" * 60,
                    )
                )
            )
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
ASSEMBLY_REPORT = """# Assembly name:  ASM584v2
# Organism name:  Escherichia coli str. K-12 substr. MG1655 (E. coli)
# Infraspecific name:  strain=K-12 substr. MG1655
# Taxid:          511145
# GenBank assembly accession: GCA_000005845.2
# RefSeq assembly accession: GCF_000005845.2
# RefSeq assembly and GenBank assemblies identical: yes
#
## Assembly-Units:
"""
ASSEMBLY_REPORT_MEMBER = "GCA_000005845.2_ASM584v2_assembly_report.txt"


@pytest.fixture
def synthetic_mg1655(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    """The real MG1655 genome class over the synthetic assembly, with its report served."""
    monkeypatch.setattr(fixtures, "SEQUENCE", fixtures.SEQUENCE * 2)
    files = fixtures.write_assembly(
        tmp_path / "tier", MG1655_ASSEMBLY, MG1655_LOCI + AMBIGUOUS_LOCI, MG1655_GAF
    )
    fixtures.forbid_network(monkeypatch)
    fixtures.serve_tier(monkeypatch, files)
    report = tmp_path / "tier" / ASSEMBLY_REPORT_MEMBER
    report.write_text(ASSEMBLY_REPORT)

    def serve_report(assembly_set: str, filename: str, **_: Any) -> str:
        if filename != ASSEMBLY_REPORT_MEMBER:
            raise FileNotFoundError(f"{assembly_set}/{filename} is not in the fixture")
        return str(report)

    import torchcell.datasets.bacteria_common as bacteria_common

    monkeypatch.setattr(bacteria_common, "resolve", serve_report)
    root = tmp_path / "mg1655"
    root.mkdir()
    return EcoliK12MG1655Genome(genome_root=str(root), overwrite=False)


def _write_screen(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, text: str) -> Path:
    """Write a synthetic screen table and pin the module to its bytes."""
    path = tmp_path / "screen.csv"
    path.write_text(text)
    monkeypatch.setattr(
        C, "SCREEN_SHA256", hashlib.sha256(path.read_bytes()).hexdigest()
    )
    monkeypatch.setattr(C, "SCREEN_BYTES", path.stat().st_size)
    return path


def _pin_synthetic_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    """Swap the frozen real-bytes oracles for the synthetic table's own counts."""
    monkeypatch.setattr(C, "EXPECTED_SOURCE_ROWS", SYNTHETIC_ROWS)
    monkeypatch.setattr(C, "EXPECTED_GUIDES", SYNTHETIC_GUIDES)
    monkeypatch.setattr(C, "EXPECTED_RECORDS", SYNTHETIC_RECORDS)
    monkeypatch.setattr(C, "EXPECTED_TARGETS", SYNTHETIC_TARGETS)
    monkeypatch.setattr(C, "EXPECTED_DROPS", SYNTHETIC_DROPS)
    monkeypatch.setattr(C.CrispriKnockdownCui2018Dataset, "MIN_RESOLVED_FRACTION", 0.5)


@pytest.fixture
def synthetic_mirror(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The raw mirror deposited under ``tmp_path`` from the synthetic screen table."""
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "dr"))
    path = _write_screen(tmp_path, monkeypatch, synthetic_screen_csv())
    return C.deposit_raw_mirror(screen_path=path)


@pytest.fixture
def built(
    synthetic_mirror: Path,
    synthetic_mg1655: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Any:
    """The loader built over the synthetic mirror and the synthetic annotation."""
    _pin_synthetic_counts(monkeypatch)
    return C.CrispriKnockdownCui2018Dataset(
        root=str(tmp_path / "build"), ecoli_genome=synthetic_mg1655
    )


def _json(dataset: Any, name: str) -> Any:
    return json.loads(Path(dataset.preprocess_dir, name).read_text())


# --------------------------------------------------------------------------- #
# The released table: header, shape and the per-guide structure
# --------------------------------------------------------------------------- #
def test_the_reader_keeps_the_releases_own_na_token_as_a_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A position in no gene is a FACT, so ``NA`` must not become a float NaN."""
    monkeypatch.setattr(C, "EXPECTED_SOURCE_ROWS", SYNTHETIC_ROWS)
    path = tmp_path / "s.csv"
    path.write_text(synthetic_screen_csv())
    frame = C.read_screen_table(str(path))
    assert list(frame.columns) == list(C.SCREEN_COLUMNS)
    assert frame["gene"].tolist().count(C.NO_GENE) == 2
    assert frame["gene"].isna().sum() == 0
    assert frame["seq"].tolist().count("") == 6


def test_a_changed_header_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A renamed column means the release moved; parsing on would mis-key the data."""
    monkeypatch.setattr(C, "EXPECTED_SOURCE_ROWS", SYNTHETIC_ROWS)
    path = tmp_path / "s.csv"
    header = ("spacer",) + C.SCREEN_COLUMNS[1:]
    path.write_text(synthetic_screen_csv(header=header))
    with pytest.raises(RuntimeError, match="screen table header is"):
        C.read_screen_table(str(path))


def test_a_changed_row_count_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The row count is a frozen oracle, so a different release stops the build."""
    monkeypatch.setattr(C, "EXPECTED_SOURCE_ROWS", SYNTHETIC_ROWS + 1)
    path = tmp_path / "s.csv"
    path.write_text(synthetic_screen_csv())
    with pytest.raises(RuntimeError, match=f"not the pinned {SYNTHETIC_ROWS + 1}"):
        C.read_screen_table(str(path))


def test_collapse_to_guides_makes_one_measurement_per_guide(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A guide's rows collapse to one measurement carrying every targeted cell."""
    monkeypatch.setattr(C, "EXPECTED_SOURCE_ROWS", SYNTHETIC_ROWS)
    monkeypatch.setattr(C, "EXPECTED_GUIDES", SYNTHETIC_GUIDES)
    path = tmp_path / "s.csv"
    path.write_text(synthetic_screen_csv())
    guides = C.collapse_to_guides(C.read_screen_table(str(path)))
    assert len(guides) == SYNTHETIC_GUIDES
    by_spacer = {guide.guide: guide for guide in guides}
    multi = by_spacer[SPACERS["kept_multi"]]
    assert multi.n_targets == 2
    assert multi.source_genes == ("thrA", "thrW")
    assert multi.responses == {
        "LC-E18": RESPONSES["kept_multi"][0],
        "LC-E75": RESPONSES["kept_multi"][1],
    }
    assert len(multi.positions) == 2


def test_a_guide_whose_fold_change_differs_across_its_rows_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The release's fold change is per guide; two values would make that a guess."""
    monkeypatch.setattr(C, "EXPECTED_SOURCE_ROWS", SYNTHETIC_ROWS)
    path = tmp_path / "s.csv"
    path.write_text(
        synthetic_screen_csv(fit_override={(SPACERS["kept_multi"], 1): "-9.5"})
    )
    with pytest.raises(RuntimeError, match="distinct fit18 values"):
        C.collapse_to_guides(C.read_screen_table(str(path)))


def test_a_guide_whose_row_count_contradicts_ntargets_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``ntargets`` is what makes "one row per perfect match" a checked fact."""
    monkeypatch.setattr(C, "EXPECTED_SOURCE_ROWS", SYNTHETIC_ROWS)
    path = tmp_path / "s.csv"
    path.write_text(synthetic_screen_csv(ntargets_override={SPACERS["kept_multi"]: 5}))
    with pytest.raises(RuntimeError, match="declares ntargets 5"):
        C.collapse_to_guides(C.read_screen_table(str(path)))


def test_a_guide_declaring_two_target_counts_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``ntargets`` is a property of the guide, so two values are not a fact."""
    monkeypatch.setattr(C, "EXPECTED_SOURCE_ROWS", SYNTHETIC_ROWS)
    path = tmp_path / "s.csv"
    text = synthetic_screen_csv()
    spacer = SPACERS["kept_multi"]
    rows = [
        line.replace(",2,", ",3,", 1)
        if line.startswith(spacer) and "thrW" in line
        else line
        for line in text.splitlines()
    ]
    path.write_text("\n".join(rows) + "\n")
    with pytest.raises(RuntimeError, match="carries ntargets"):
        C.collapse_to_guides(C.read_screen_table(str(path)))


def test_a_malformed_spacer_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The spacer is the perturbation's identity and the only target it carries."""
    monkeypatch.setattr(C, "EXPECTED_SOURCE_ROWS", SYNTHETIC_ROWS)
    path = tmp_path / "s.csv"
    path.write_text(synthetic_screen_csv().replace(SPACERS["kept_single"], "ACGTN" * 4))
    with pytest.raises(RuntimeError, match="is not a 20-nt ACGT spacer"):
        C.collapse_to_guides(C.read_screen_table(str(path)))


def test_a_guide_count_other_than_the_oracle_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """78,137 distinct guides is a frozen number, not an observation of the run."""
    monkeypatch.setattr(C, "EXPECTED_SOURCE_ROWS", SYNTHETIC_ROWS)
    monkeypatch.setattr(C, "EXPECTED_GUIDES", SYNTHETIC_GUIDES + 1)
    path = tmp_path / "s.csv"
    path.write_text(synthetic_screen_csv())
    with pytest.raises(RuntimeError, match="distinct guides, not the pinned"):
        C.collapse_to_guides(C.read_screen_table(str(path)))


# --------------------------------------------------------------------------- #
# Retention rules
# --------------------------------------------------------------------------- #
def _measurement(rule: str, cells: tuple[str, ...]) -> C.GuideMeasurement:
    return C.GuideMeasurement(
        guide=SPACERS[rule],
        n_targets=len(cells),
        source_genes=cells,
        positions=tuple(str(i) for i in range(len(cells))),
        responses={"LC-E18": -1.0, "LC-E75": -0.5},
    )


def _classify(rule: str, cells: tuple[str, ...], **kwargs: Any) -> C.RetentionRule:
    stored = kwargs.pop("stored", {"thrA": "b0002", "thrW": "b0003"})
    return C.classify_guide(
        _measurement(rule, cells),
        stored,
        retired=frozenset(kwargs.pop("retired", ())),
        collided=frozenset(kwargs.pop("collided", ())),
        ambiguous=frozenset(kwargs.pop("ambiguous", ())),
    )


def test_a_guide_with_no_match_in_a_gene_has_no_genotype_to_write() -> None:
    """There is no gene perturbation, so the measurement cannot become a record."""
    verdict = _classify("drop_no_gene", ("NA", "NA"))
    assert verdict.drop_reason == C.DROP_NO_GENE
    assert verdict.stored_tags == ()


def test_a_partly_intergenic_guide_is_dropped_rather_than_partly_asserted() -> None:
    """Naming a subset of the bound loci would assert a genotype the release lacks."""
    assert _classify("drop_mixed", ("thrA", "NA")).drop_reason == C.DROP_MIXED


def test_a_retired_symbol_is_dropped_because_the_leaf_refuses_it() -> None:
    """``systematic_gene_name`` must be a b-number of the pinned assembly."""
    verdict = _classify("drop_retired", ("zzzA",), retired=("zzzA",))
    assert verdict.drop_reason == C.DROP_RETIRED


def test_a_shared_symbol_is_dropped_and_named_as_a_collision() -> None:
    """Two source names on one locus are kept as given, so neither is a b-number."""
    verdict = _classify("drop_collision_a", ("pro2",), collided=("pro2",))
    assert verdict.drop_reason == C.DROP_COLLISION


def test_an_ambiguous_symbol_is_dropped_and_named_as_ambiguous() -> None:
    """A symbol on two loci names no single gene to perturb."""
    verdict = _classify("drop_ambiguous", ("dup1",), ambiguous=("dup1",))
    assert verdict.drop_reason == C.DROP_AMBIGUOUS


def test_the_drop_reasons_are_applied_in_their_declared_order() -> None:
    """A guide hitting several unresolvable kinds reports the first rule, once."""
    verdict = _classify(
        "drop_retired",
        ("zzzA", "pro2"),
        retired=("zzzA",),
        collided=("pro2",),
        stored={"zzzA": "zzzA", "pro2": "pro2"},
    )
    assert verdict.drop_reason == C.DROP_RETIRED


def test_a_retained_guides_targets_deduplicate_to_sorted_b_numbers() -> None:
    """Two positions in one gene are one knockdown, and the tags come out sorted."""
    assert _classify("kept_multi", ("thrW", "thrA")).stored_tags == ("b0002", "b0003")
    assert _classify("kept_two_positions_one_gene", ("thrA", "thrA")).stored_tags == (
        "b0002",
    )


def test_a_resolved_name_that_is_not_a_b_number_is_a_refusal_not_a_drop() -> None:
    """A name outside the namespace that no reconciliation status explains is a bug."""
    with pytest.raises(RuntimeError, match="which is not all b-numbers"):
        _classify("kept_single", ("thrL",), stored={"thrL": "ECK0001"})


# --------------------------------------------------------------------------- #
# The accounting arithmetic
# --------------------------------------------------------------------------- #
def _accounting(**overrides: Any) -> C.BuildAccounting:
    fields: dict[str, Any] = {
        "dataset": "CrispriKnockdownCui2018Dataset",
        "source_rows": SYNTHETIC_ROWS,
        "source_guides": SYNTHETIC_GUIDES,
        "screens": ("LC-E18", "LC-E75"),
        "candidate_records": SYNTHETIC_GUIDES * 2,
        "kept_records": SYNTHETIC_RECORDS,
        "dropped_records": SYNTHETIC_GUIDES * 2 - SYNTHETIC_RECORDS,
        "dropped_guides_by_reason": dict(SYNTHETIC_DROPS),
        "distinct_targets": SYNTHETIC_TARGETS,
        "perturbations_written": SYNTHETIC_PERTURBATIONS,
        "reconciliation": None,
    }
    fields.update(overrides)
    return C.BuildAccounting.model_construct(**fields)


def test_the_accounting_accepts_the_synthetic_arithmetic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Kept plus dropped equals guides times screens, per reason and in total."""
    monkeypatch.setattr(C, "EXPECTED_DROPS", SYNTHETIC_DROPS)
    accounting = _accounting()
    accounting.check()
    assert accounting.kept_records + accounting.dropped_records == (
        accounting.source_guides * len(accounting.screens)
    )
    assert (
        sum(accounting.dropped_guides_by_reason.values()) * len(accounting.screens)
        == accounting.dropped_records
    )


def test_the_accounting_refuses_a_kept_plus_dropped_that_misses_the_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A record that vanished between the table and the store must be visible."""
    monkeypatch.setattr(C, "EXPECTED_DROPS", SYNTHETIC_DROPS)
    with pytest.raises(RuntimeError, match="!= .* candidates"):
        _accounting(kept_records=SYNTHETIC_RECORDS - 2).check()


def test_the_accounting_refuses_candidates_that_are_not_guides_times_screens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every guide was measured in both strains, so the product is the candidate set."""
    monkeypatch.setattr(C, "EXPECTED_DROPS", SYNTHETIC_DROPS)
    with pytest.raises(RuntimeError, match="screens !="):
        _accounting(source_guides=SYNTHETIC_GUIDES + 1).check()


def test_the_accounting_refuses_per_reason_drops_that_do_not_total(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A drop reason that lost a guide would hide it from the ledger."""
    monkeypatch.setattr(C, "EXPECTED_DROPS", SYNTHETIC_DROPS)
    skewed = dict(SYNTHETIC_DROPS)
    skewed[C.DROP_NO_GENE] = 0
    with pytest.raises(RuntimeError, match="per-reason drops total"):
        _accounting(dropped_guides_by_reason=skewed).check()


def test_the_accounting_refuses_drops_that_are_not_the_pinned_ones() -> None:
    """The per-reason counts are a frozen oracle of the released bytes."""
    with pytest.raises(RuntimeError, match="are not the pinned"):
        _accounting().check()


# --------------------------------------------------------------------------- #
# Genotype, environment and phenotype
# --------------------------------------------------------------------------- #
def test_the_publication_names_both_identifiers() -> None:
    """A record's publication carries the PubMed id and the DOI with both URLs."""
    pub = C.publication()
    assert (pub.pubmed_id, pub.doi) == (C.PMID, C.DOI)
    assert pub.doi_url == f"https://doi.org/{C.DOI}"


@pytest.mark.parametrize("screen_id", ["LC-E18", "LC-E75"])
def test_each_screens_background_is_the_strain_tables_own_row(screen_id: str) -> None:
    """The genotype statement is Supplementary Table 7's row, kept verbatim."""
    background = C.strain_background(screen_id)
    assert background.name == screen_id
    assert background.reference_strain == "MG1655"
    assert background.assembly_set == "ecoli_K12_MG1655_ASM584v2"
    assert background.parents == ["MG1655"]
    assert background.alleles == []
    assert background.genotype_statement == C.STRAIN_BACKGROUNDS[screen_id][0]
    assert f">{screen_id}<" in background.genotype_statement
    assert background.provenance is not None
    assert len(background.provenance) == 2
    assert background.provenance_gaps == []


def test_the_two_backgrounds_differ_in_the_dcas9_cassette_they_name() -> None:
    """The whole difference between the screens is the dCas9 expression cassette."""
    high = C.strain_background("LC-E18").construction or ""
    low = C.strain_background("LC-E75").construction or ""
    assert "pOSIP-KH-RBS2-dCas9" in high
    assert "pOSIP-CO-RBS-library-dCas9" in low
    assert "2.6-fold lower" in low


def test_the_environment_is_lb_at_37c_with_one_nanomolar_atc() -> None:
    """One environment serves both screens; what differs is the strain, not the medium."""
    environment = C.screen_environment()
    assert environment.media is LB
    assert environment.temperature is not None
    assert environment.temperature.value == 37.0
    assert environment.duration_generations == 17.0
    assert environment.duration_hours is None
    assert environment.aerobicity == "aerobic"
    (inducer,) = environment.perturbations
    assert isinstance(inducer, SmallMoleculePerturbation)
    assert inducer.compound.name == "anhydrotetracycline"
    dose = inducer.concentration
    assert dose is not None
    assert dose.unit is not None
    assert (dose.value, dose.unit.value) == (1.0, "nM")


def test_a_measured_phenotype_is_a_signed_log2_ratio_with_no_dispersion() -> None:
    """The readout is a log2 fold change and the release carries no standard error."""
    phenotype = C.screen_phenotype("LC-E18", -3.5)
    assert phenotype.measurement_type is MeasurementType.log2_ratio
    assert phenotype.assay_type is not None
    assert phenotype.assay_type.value == "pooled_competitive_growth_barcode"
    assert phenotype.environment_response == -3.5
    assert phenotype.screen_id == "LC-E18"
    assert phenotype.sample_unit is not None
    assert (phenotype.n_samples, phenotype.sample_unit.value) == (
        3,
        "biological_replicate",
    )
    assert phenotype.environment_response_se is None
    assert phenotype.gapped_fields() == set(C._UNCERTAINTY_FIELDS)


def test_the_reference_phenotype_is_the_control_guide_at_zero() -> None:
    """Normalization to the non-targeting guide makes its own log2FC zero."""
    reference = C.screen_phenotype("LC-E75", None)
    assert reference.environment_response == 0.0
    assert reference.screen_id == "LC-E75"
    assert reference.units is not None
    assert str(C.NORMALIZATION.value) in reference.units


def test_a_knockdown_carries_the_spacer_and_names_its_source_symbol() -> None:
    """The spacer is the identity; the released symbol rides on identifier_mapping."""
    rule = _classify("kept_multi", ("thrA", "thrW"))
    perturbations = C.crispri_perturbations(
        rule, {"thrA": "b0002", "thrW": "b0003"}, {"b0002": "thrA", "b0003": "thrW"}
    )
    assert [p.systematic_gene_name for p in perturbations] == ["b0002", "b0003"]
    assert [p.perturbed_gene_name for p in perturbations] == ["thrA", "thrW"]
    for perturbation in perturbations:
        assert perturbation.perturbation_type == "bacterial_crispr_interference"
        assert perturbation.gene_namespace == "ecoli_k12_mg1655_bnumber"
        assert perturbation.expression_direction == "decreased"
        assert perturbation.crispr.effector == "dCas9"
        assert perturbation.crispr.guide_sequence == SPACERS["kept_multi"]
        assert perturbation.crispr.n_guides == 1
        assert perturbation.identifier_mapping is not None
        assert perturbation.identifier_mapping.route == "gene_symbol"
    assert [
        p.identifier_mapping.source_identifier
        for p in perturbations
        if p.identifier_mapping is not None
    ] == ["thrA", "thrW"]


def test_annotation_symbols_prefer_the_assemblys_spelling_and_fall_back_to_the_tag(
    synthetic_mg1655: Any,
) -> None:
    """One gene carries one spelling across datasets; a symbol-less locus keeps its tag."""
    symbols = C.annotation_symbols(synthetic_mg1655, ("b0001", "b0002", "b9999"))
    assert symbols == {"b0001": "thrL", "b0002": "thrA", "b9999": "b9999"}


# --------------------------------------------------------------------------- #
# The raw mirror
# --------------------------------------------------------------------------- #
def test_the_mirror_records_a_scriptable_pmc_retrieval(synthetic_mirror: Path) -> None:
    """The one deposited file names the bucket key that reproduced its digest."""
    manifest = C.load_manifest()
    (record,) = manifest.files
    assert record.path == C.SCREEN_REL
    assert record.role == "raw_data"
    assert record.sha256 == C.SCREEN_SHA256
    assert record.retrieval is not None
    assert record.retrieval.method.value == "pmc_cloud"
    assert record.retrieval.params == {"key": C.pmc_cloud_key(C.SCREEN_FILENAME)}
    assert (
        record.retrieval.retriever == "torchcell.literature.retrieve.pmc_cloud_object"
    )
    assert manifest.doi == C.DOI
    assert C.manifest_sha256(manifest, C.SCREEN_REL) == C.SCREEN_SHA256


def test_the_mirror_names_every_released_file_it_does_not_consume(
    synthetic_mirror: Path,
) -> None:
    """Five released supplementary files are not deposited, each with its reason."""
    expected = C.load_manifest().si_expected
    assert len(expected) == 7
    assert any(
        "Supplementary Data 3" in line and "NOT deposited" in line for line in expected
    )
    assert any("NO SEQUENCE-READ ACCESSION IS RELEASED" in line for line in expected)


def test_a_file_absent_from_the_manifest_is_refused(synthetic_mirror: Path) -> None:
    """Asking the manifest for an undeposited path is an error, not a None."""
    with pytest.raises(KeyError, match="is not in the raw-mirror manifest"):
        C.manifest_sha256(C.load_manifest(), "data/nothing.csv")


def test_a_differing_file_is_refused_rather_than_overwritten(
    synthetic_mirror: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A deposited artifact is never silently replaced by different bytes."""
    other = tmp_path / "other.csv"
    other.write_text(synthetic_screen_csv(targets=TARGETS[:2]))
    monkeypatch.setattr(
        C, "SCREEN_SHA256", hashlib.sha256(other.read_bytes()).hexdigest()
    )
    monkeypatch.setattr(C, "SCREEN_BYTES", other.stat().st_size)
    with pytest.raises(RuntimeError, match="exists with a different sha256"):
        C.deposit_raw_mirror(screen_path=other)


def test_a_file_whose_digest_is_not_the_pin_is_refused_before_any_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A refusal leaves no partial deposit, so the digest is checked first."""
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "dr2"))
    path = tmp_path / "s.csv"
    path.write_text(synthetic_screen_csv())
    monkeypatch.setattr(C, "SCREEN_SHA256", "ab" * 32)
    with pytest.raises(RuntimeError, match="not the pinned"):
        C.deposit_raw_mirror(screen_path=path)
    assert not (C.raw_mirror_dir() / C.SCREEN_REL).exists()


def test_a_byte_count_that_is_not_the_pin_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The deposited size is pinned beside the digest, so a truncation is caught."""
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "dr3"))
    path = _write_screen(tmp_path, monkeypatch, synthetic_screen_csv())
    monkeypatch.setattr(C, "SCREEN_BYTES", 1)
    with pytest.raises(RuntimeError, match="bytes, not the pinned 1"):
        C.deposit_raw_mirror(screen_path=path)


def test_the_download_step_links_the_mirror_and_refuses_a_moved_manifest_pin(
    synthetic_mirror: Path, tmp_path: Path
) -> None:
    """``download`` reads the manifest first, so a drifted pin stops the build."""
    pins = ((C.SCREEN_REL, C.SCREEN_FILENAME, C.SCREEN_SHA256),)
    raw = tmp_path / "linked"
    C._link_mirror_files(str(raw), pins)
    assert os.readlink(raw / C.SCREEN_FILENAME) == str(synthetic_mirror / C.SCREEN_REL)
    manifest = json.loads((synthetic_mirror / "manifest.json").read_text())
    manifest["files"][0]["sha256"] = "cd" * 32
    (synthetic_mirror / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ManifestPinMismatchError):
        C._link_mirror_files(str(tmp_path / "linked2"), pins)


def test_the_download_step_refuses_a_mirror_file_that_is_not_there(
    synthetic_mirror: Path, tmp_path: Path
) -> None:
    """A manifest entry whose bytes are gone is a refusal, not an empty raw dir."""
    (synthetic_mirror / C.SCREEN_REL).unlink()
    with pytest.raises(RuntimeError, match="required raw artifact missing"):
        C._link_mirror_files(
            str(tmp_path / "linked3"),
            ((C.SCREEN_REL, C.SCREEN_FILENAME, C.SCREEN_SHA256),),
        )


# --------------------------------------------------------------------------- #
# The end-to-end synthetic build
# --------------------------------------------------------------------------- #
def test_the_build_writes_one_record_per_retained_guide_and_screen(built: Any) -> None:
    """Six of ten guides are dropped, so four survive in each of the two screens."""
    assert len(built) == SYNTHETIC_RECORDS
    assert built.experiment_class is BacterialEnvironmentResponseExperiment
    assert built.reference_class is BacterialEnvironmentResponseExperimentReference
    assert built.raw_file_names == [C.SCREEN_FILENAME]
    accounting = _json(built, "build_accounting.json")
    assert {
        key: accounting[key]
        for key in (
            "source_rows",
            "source_guides",
            "candidate_records",
            "kept_records",
            "dropped_records",
            "distinct_targets",
            "perturbations_written",
        )
    } == {
        "source_rows": SYNTHETIC_ROWS,
        "source_guides": SYNTHETIC_GUIDES,
        "candidate_records": SYNTHETIC_GUIDES * 2,
        "kept_records": SYNTHETIC_RECORDS,
        "dropped_records": SYNTHETIC_GUIDES * 2 - SYNTHETIC_RECORDS,
        "distinct_targets": SYNTHETIC_TARGETS,
        "perturbations_written": SYNTHETIC_PERTURBATIONS,
    }
    assert accounting["dropped_guides_by_reason"] == SYNTHETIC_DROPS
    assert accounting["screens"] == ["LC-E18", "LC-E75"]


def test_the_retention_ledger_names_every_guide_and_its_verdict(built: Any) -> None:
    """One ledger row per distinct guide, kept or dropped, with its released cells."""
    import pandas as pd

    ledger = pd.read_csv(
        osp.join(built.preprocess_dir, "guide_retention.csv"), keep_default_na=False
    )
    assert len(ledger) == SYNTHETIC_GUIDES
    verdicts = dict(zip(ledger["guide"], ledger["drop_reason"], strict=True))
    assert verdicts[SPACERS["kept_single"]] == ""
    assert verdicts[SPACERS["drop_no_gene"]] == C.DROP_NO_GENE
    assert verdicts[SPACERS["drop_ambiguous"]] == C.DROP_AMBIGUOUS
    multi = ledger[ledger["guide"] == SPACERS["kept_multi"]].iloc[0]
    assert multi["source_genes"] == "thrA;thrW"
    assert multi["stored_tags"] == "b0002;b0003"
    assert float(multi["fit18"]) == RESPONSES["kept_multi"][0]


def test_every_record_pins_mg1655_with_its_own_screens_background(built: Any) -> None:
    """The two screens are two strains, readable from the reference, not just the label."""
    pins = set()
    for index in range(len(built)):
        record = built[index]
        reference = record["reference"]["genome_reference"]
        pins.add(
            (
                record["experiment"]["phenotype"]["screen_id"],
                reference["assembly_set"],
                reference["assembly_accession"],
                reference["background"]["name"],
                reference["strain"],
            )
        )
    assert pins == {
        (
            screen_id,
            "ecoli_K12_MG1655_ASM584v2",
            "GCA_000005845.2",
            screen_id,
            screen_id,
        )
        for screen_id, _ in C.SCREENS
    }


def test_the_stored_responses_are_the_released_columns_per_screen(built: Any) -> None:
    """``fit18`` lands on the LC-E18 record and ``fit75`` on the LC-E75 one."""
    seen: dict[tuple[str, str], float] = {}
    for index in range(len(built)):
        experiment = built[index]["experiment"]
        spacer = experiment["genotype"]["perturbations"][0]["crispr"]["guide_sequence"]
        seen[(experiment["phenotype"]["screen_id"], spacer)] = experiment["phenotype"][
            "environment_response"
        ]
    for rule in ("kept_single", "kept_multi", "kept_pseudogene"):
        spacer = SPACERS[rule]
        assert seen[("LC-E18", spacer)] == RESPONSES[rule][0]
        assert seen[("LC-E75", spacer)] == RESPONSES[rule][1]


def test_a_multi_target_guide_is_one_record_with_several_knockdowns(built: Any) -> None:
    """The repeated-element guides are genuine multi-locus knockdowns, not duplicates."""
    sizes: dict[str, int] = {}
    for index in range(len(built)):
        experiment = built[index]["experiment"]
        if experiment["phenotype"]["screen_id"] != "LC-E18":
            continue
        perturbations = experiment["genotype"]["perturbations"]
        spacer = perturbations[0]["crispr"]["guide_sequence"]
        sizes[spacer] = len(perturbations)
    assert sizes[SPACERS["kept_multi"]] == 2
    assert sizes[SPACERS["kept_single"]] == 1
    assert sizes[SPACERS["kept_two_positions_one_gene"]] == 1
    assert sizes[SPACERS["kept_pseudogene"]] == 1


def test_the_gene_set_is_the_resolved_b_numbers_only(built: Any) -> None:
    """A dropped guide's gene never reaches the store, pseudogene loci do."""
    assert set(built.gene_set) == {"b0001", "b0002", "b0003", "b0004"}


def test_a_record_count_other_than_the_oracle_is_refused(
    synthetic_mirror: Path,
    synthetic_mg1655: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The written count is checked against the frozen oracle before anything else."""
    _pin_synthetic_counts(monkeypatch)
    monkeypatch.setattr(C, "EXPECTED_RECORDS", SYNTHETIC_RECORDS + 2)
    with pytest.raises(RuntimeError, match="records written, not the pinned"):
        C.CrispriKnockdownCui2018Dataset(
            root=str(tmp_path / "build_bad"), ecoli_genome=synthetic_mg1655
        )


def test_a_target_count_other_than_the_oracle_is_refused(
    synthetic_mirror: Path,
    synthetic_mg1655: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The perturbed gene universe is pinned too, so an annotation move reports."""
    _pin_synthetic_counts(monkeypatch)
    monkeypatch.setattr(C, "EXPECTED_TARGETS", SYNTHETIC_TARGETS + 1)
    with pytest.raises(RuntimeError, match="distinct targets, not the pinned"):
        C.CrispriKnockdownCui2018Dataset(
            root=str(tmp_path / "build_bad2"), ecoli_genome=synthetic_mg1655
        )


def test_a_resolution_rate_below_the_stated_floor_stops_the_build(
    synthetic_mirror: Path,
    synthetic_mg1655: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Records are never dropped to pass a threshold; the build reports instead."""
    from torchcell.datasets.bacteria_common import LocusTagResolutionError

    monkeypatch.setattr(C, "EXPECTED_SOURCE_ROWS", SYNTHETIC_ROWS)
    monkeypatch.setattr(C, "EXPECTED_GUIDES", SYNTHETIC_GUIDES)
    with pytest.raises(LocusTagResolutionError, match="below 0.95"):
        C.CrispriKnockdownCui2018Dataset(
            root=str(tmp_path / "build_floor"), ecoli_genome=synthetic_mg1655
        )


def test_create_experiment_is_not_the_entry_point_for_this_loader(built: Any) -> None:
    """Records are built inline in ``process``; the hook would be a second path."""
    with pytest.raises(NotImplementedError):
        built.create_experiment()
    assert built.preprocess_raw("sentinel") == "sentinel"


# --------------------------------------------------------------------------- #
# The supplementary verification rows, over the synthetic store
# --------------------------------------------------------------------------- #
def _records(dataset: Any) -> list[dict[str, Any]]:
    return [dataset[index] for index in range(len(dataset))]


def test_one_pass_summarizes_the_store_the_supplementary_rows_read(built: Any) -> None:
    """The summary is what keeps the rows memory-bounded on a 141,542-record store."""
    summary = C.summarize_store(_records(built))
    assert summary.n_records == SYNTHETIC_RECORDS
    assert summary.targets == ("b0001", "b0002", "b0003", "b0004")
    assert summary.n_spacers == SYNTHETIC_KEPT_GUIDES
    assert summary.malformed_spacers == ()
    assert summary.per_screen == {
        "LC-E18": SYNTHETIC_KEPT_GUIDES,
        "LC-E75": SYNTHETIC_KEPT_GUIDES,
    }
    assert len(summary.pins) == len(C.SCREENS)


def test_the_stored_targets_row_accepts_genes_and_pseudogene_loci(
    built: Any, synthetic_mg1655: Any
) -> None:
    """A pseudogene resolves to itself as ``non_gene_feature``, which is a locus."""
    result = C.stored_tags_are_loci(
        C.summarize_store(_records(built)), synthetic_mg1655
    )
    assert result.passed
    assert result.details["n_targets"] == SYNTHETIC_TARGETS


def test_the_stored_targets_row_fails_on_a_tag_the_assembly_does_not_carry(
    built: Any, synthetic_mg1655: Any
) -> None:
    """A record keyed to a locus the pinned annotation lacks must be visible."""
    records = _records(built)
    records[0]["experiment"]["genotype"]["perturbations"][0]["systematic_gene_name"] = (
        "b9999"
    )
    result = C.stored_tags_are_loci(C.summarize_store(records), synthetic_mg1655)
    assert not result.passed
    assert "b9999" in result.details["not_a_locus"]


def test_the_spacer_row_accepts_the_synthetic_spacers_and_fails_a_malformed_one(
    built: Any,
) -> None:
    """The spacer carries the target, so a malformed one cannot be re-mapped."""
    records = _records(built)
    assert C.spacers_are_twenty_nt(C.summarize_store(records)).passed
    records[0]["experiment"]["genotype"]["perturbations"][0]["crispr"][
        "guide_sequence"
    ] = "ACGT"
    result = C.spacers_are_twenty_nt(C.summarize_store(records))
    assert not result.passed
    assert result.details["malformed"] == ["ACGT"]


def test_the_screen_balance_row_passes_and_fails_on_an_unbalanced_store(
    built: Any,
) -> None:
    """No drop rule fires per screen, so the two groups must be the same size."""
    records = _records(built)
    result = C.screens_are_balanced(C.summarize_store(records))
    assert result.passed
    assert result.details["per_screen"] == {
        "LC-E18": SYNTHETIC_KEPT_GUIDES,
        "LC-E75": SYNTHETIC_KEPT_GUIDES,
    }
    assert not C.screens_are_balanced(C.summarize_store(records[:-1])).passed


def test_the_screen_balance_row_fails_when_a_reference_loses_its_background(
    built: Any,
) -> None:
    """A record whose reference does not pin its screen's strain is not this dataset."""
    records = _records(built)
    for record in records:
        record["reference"]["genome_reference"]["background"] = None
    assert not C.screens_are_balanced(C.summarize_store(records)).passed


def test_verify_build_runs_the_whole_gate_over_the_synthetic_store(
    built: Any, synthetic_mg1655: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """L0-L4 plus the three supplementary rows, on the store the fixture just built.

    The verifier's own rows carry their verdicts here as they do on the real store; the
    point of running it hermetically is that the three appended rows and the report file
    are exercised without the 141,542-record pass.
    """
    monkeypatch.setattr(C, "bacterial_genome", lambda *a, **k: synthetic_mg1655)
    monkeypatch.setattr(C, "EXPECTED_RECORDS", SYNTHETIC_RECORDS)
    report = C.verify_build(built.root)
    names = {result.name for result in report.results}
    assert {
        "stored_targets_are_loci_of_the_pinned_assembly",
        "guide_spacers_are_twenty_nt_acgt",
        "both_screens_are_balanced_and_strain_pinned",
    } <= names
    assert all(
        result.passed
        for result in report.results
        if result.name
        in {
            "structural_validation",
            "count",
            "pair_uniqueness",
            "stored_targets_are_loci_of_the_pinned_assembly",
            "guide_spacers_are_twenty_nt_acgt",
            "both_screens_are_balanced_and_strain_pinned",
        }
    ), [(r.name, r.message) for r in report.results if not r.passed]
    written = json.loads(
        Path(built.root, "preprocess", "verification_report.json").read_text()
    )
    assert written["dataset_name"] == "CrispriKnockdownCui2018Dataset"


# --------------------------------------------------------------------------- #
# The real mirror and the dev-tree build (--data)
# --------------------------------------------------------------------------- #
def _real_data_root() -> str:
    return os.environ["DATA_ROOT"]


@pytest.mark.data
def test_every_sourced_value_is_a_verbatim_quote_in_the_pinned_mirror() -> None:
    """Each sourced value's quote is still a substring of the bytes it pins."""
    from torchcell.verification.sourced import SourcedValue, audit_sourced_value

    library = osp.join(_real_data_root(), "torchcell-library")
    values = [
        getattr(C, name)
        for name in dir(C)
        if isinstance(getattr(C, name), SourcedValue)
    ]
    assert len(values) == 20
    assert {v.provenance.source_uri for v in values} == {C.PAPER_MD, C.SI1_MD}
    for screen_id in C.STRAIN_BACKGROUNDS:
        values.extend(C.strain_background(screen_id).provenance or [])
    for value in values:
        result = audit_sourced_value(value, library)
        assert result.passed, (value.quote, result.message)


@pytest.mark.data
def test_the_deposited_raw_mirror_matches_its_recorded_digest() -> None:
    """The mirror is the authority, so its bytes must still hash to the pin."""
    manifest = C.load_manifest()
    (record,) = manifest.files
    path = C.raw_mirror_dir() / record.path
    assert path.exists()
    assert C._sha256(path) == C.SCREEN_SHA256
    assert path.stat().st_size == C.SCREEN_BYTES


@pytest.mark.data
def test_the_released_table_has_the_shape_the_oracles_pin() -> None:
    """85,381 rows over 78,137 guides, with the fold change constant per guide."""
    path = str(C.raw_mirror_dir() / C.SCREEN_REL)
    frame = C.read_screen_table(path)
    assert len(frame) == C.EXPECTED_SOURCE_ROWS == 85381
    guides = C.collapse_to_guides(frame)
    assert len(guides) == C.EXPECTED_GUIDES == 78137
    assert sum(guide.n_targets for guide in guides) == len(frame)
    assert C.NORMALIZATION.value not in set(frame["guide"])


@pytest.mark.data
def test_the_built_store_holds_the_pinned_records_over_the_pinned_targets() -> None:
    """The dev-tree build is 141,542 records over 4,263 MG1655 loci."""
    root = osp.join(_real_data_root(), "data/torchcell/crispri_knockdown_cui2018")
    if not osp.exists(osp.join(root, "processed", "lmdb")):
        pytest.skip("the Cui 2018 dev store is not built")
    accounting = json.loads(
        Path(root, "preprocess", "build_accounting.json").read_text()
    )
    assert accounting["kept_records"] == C.EXPECTED_RECORDS == 141542
    assert accounting["distinct_targets"] == C.EXPECTED_TARGETS == 4263
    assert accounting["dropped_records"] == 14732
    assert accounting["dropped_guides_by_reason"] == C.EXPECTED_DROPS
