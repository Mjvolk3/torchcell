# tests/torchcell/verification/test_runners.py
# [[tests.torchcell.verification.test_runners]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/verification/test_runners.py
"""The verification runners on tiny LMDB stores built under ``tmp_path``.

Every store is written the way the dataset loaders write theirs: ``<root>/processed/lmdb``
with keys ``"0"``, ``"1"``, ... and pickled ``{"experiment", "reference", "publication"}``
dicts from ``model_dump()``; an optional sibling ``processed/interned`` env holds
content-addressed sub-objects that the record points at with ``{"$ref": key}``. The data
root is ``tmp_path``; a runner's dataset registry is monkeypatched down to the datasets
built here (as ``scripts/verify_datasets.py`` narrows it), and the count oracles that are
module constants (Ohya 4718, Ohnuki 2018 1112, Ohnuki 2022 1979) are monkeypatched to
the built record count where a pass is wanted. ``_genome`` and ``_sgd_gene_set`` are
stubbed; nothing here loads the genome or reads the real data root.

Derived expectations: LMDB iterates keys in byte order, so eleven records written under
``"0"``..``"10"`` come back as 0, 1, 10, 2, ..., 9. An expression L4 with reference universe
{A, B, C} and other universe {A, B, C, D} has ``n_overlap`` 4 and one disagreement,
``{"entity": "D", "a": 0.0, "b": 1.0, "diff": 1.0}``. A gene containment of {A, B, D} in
Ohya's {A, B, C} is 2/3 = 0.667 against the 0.90 floor. ``run_all`` calls eleven family
runners in a fixed order and evaluates every one before combining with ``and``.
"""

from __future__ import annotations

import json
import pickle
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import lmdb
import pytest

from torchcell.datamodels.calmorph_labels import CALMORPH_LABELS, CALMORPH_STATISTICS
from torchcell.datamodels.media import SC, YP_GALACTOSE
from torchcell.datamodels.schema import (
    CalMorphExperiment,
    CalMorphExperimentReference,
    CalMorphPhenotype,
    Compound,
    Concentration,
    DoseBasis,
    Environment,
    EnvironmentResponseExperiment,
    EnvironmentResponseExperimentReference,
    EnvironmentResponsePhenotype,
    FitnessExperiment,
    FitnessExperimentReference,
    FitnessPhenotype,
    Genotype,
    KanMxDeletionPerturbation,
    MeasurementType,
    Media,
    MetaboliteExperiment,
    MetaboliteExperimentReference,
    MetabolitePhenotype,
    MicroarrayExpressionExperiment,
    MicroarrayExpressionExperimentReference,
    MicroarrayExpressionPhenotype,
    ProteinAbundanceExperiment,
    ProteinAbundanceExperimentReference,
    ProteinAbundancePhenotype,
    ReferenceGenome,
    RNASeqExpressionExperiment,
    RNASeqExpressionExperimentReference,
    RNASeqExpressionPhenotype,
    SampleUnit,
    SequenceVariantPerturbation,
    SmallMoleculePerturbation,
    Temperature,
    VisualScoreExperiment,
    VisualScoreExperimentReference,
    VisualScorePhenotype,
)
from torchcell.sequence.genome.registry import PETER2018_1011, SGD_S288C_R64
from torchcell.verification import runners
from torchcell.verification.report import (
    Level,
    LevelResult,
    Provenance,
    VerificationReport,
)

Record = dict[str, Any]
PROV = Provenance(source_uri="test://synthetic", citation_key="runnersTest2026")
GENES = ["YAL001C", "YBR085W", "YJR155W"]
HYDROQUINONE = "QIGBRXMKCJKVMJ-UHFFFAOYSA-N"

EXPRESSION_NAMES = [
    "structural",
    "count",
    "gene_completeness",
    "value_fidelity",
    "se_nonnegative",
    "n_replicates_ge_1",
    "reference_log2_zero",
    "deletion_downregulates",
]
MORPHOLOGY_NAMES = [
    "structural",
    "count",
    "calmorph_completeness",
    "reference_populated",
    "value_fidelity",
    "cv_nonnegative",
    "vocabulary_parity",
]
RUN_ALL_ORDER = [
    "run_expression",
    "run_morphology",
    "run_morphology_ohnuki",
    "run_ohnuki_morphology",
    "run_visual_score",
    "run_metabolite",
    "run_protein",
    "run_rnaseq",
    "run_environment_response",
    "run_fitness",
    "run_segregant_growth",
]


# --------------------------------------------------------------------------- #
# record builders (schema-valid by construction, dumped like the loaders do)
# --------------------------------------------------------------------------- #
def _genotype(genes: list[str]) -> Genotype:
    return Genotype(
        perturbations=[
            KanMxDeletionPerturbation(systematic_gene_name=g, perturbed_gene_name=g)
            for g in genes
        ]
    )


def _reference_genome() -> ReferenceGenome:
    return ReferenceGenome(species="Saccharomyces cerevisiae", strain="BY4741")


def _publication() -> dict[str, Any]:
    return {"pubmed_id": "1", "pubmed_url": "https://pubmed.ncbi.nlm.nih.gov/1/"}


def _expression_record(deleted: str, universe: list[str]) -> Record:
    """A microarray deletion record measuring ``universe`` with ``deleted`` at log2 -3."""
    log2 = {g: (-3.0 if g == deleted else 0.1) for g in universe}
    phenotype = MicroarrayExpressionPhenotype(
        expression={g: 100.0 for g in universe},
        expression_log2_ratio=log2,
        expression_log2_ratio_se={g: 0.05 for g in universe},
        expression_log2_ratio_variance={g: 0.0025 for g in universe},
        n_replicates={g: 4 for g in universe},
    )
    reference_phenotype = MicroarrayExpressionPhenotype(
        expression={g: 100.0 for g in universe},
        expression_log2_ratio={g: 0.0 for g in universe},
        expression_log2_ratio_se=None,
        expression_log2_ratio_variance=None,
        n_replicates={g: 1 for g in universe},
    )
    env = Environment(
        media=Media(name="SC", state="liquid", is_synthetic=True),
        temperature=Temperature(value=30),
    )
    experiment = MicroarrayExpressionExperiment(
        dataset_name="test",
        genotype=_genotype([deleted]),
        environment=env,
        phenotype=phenotype,
    )
    reference = MicroarrayExpressionExperimentReference(
        dataset_name="test",
        genome_reference=_reference_genome(),
        environment_reference=env.model_copy(),
        phenotype_reference=reference_phenotype,
    )
    return {
        "experiment": experiment.model_dump(),
        "reference": reference.model_dump(),
        "publication": _publication(),
    }


def _morphology_record(gene: str) -> Record:
    base = {k: 1.0 for k in CALMORPH_LABELS}
    cv = {k: 0.5 for k in CALMORPH_STATISTICS}
    env = Environment(
        media=Media(name="YEPD", state="solid", is_synthetic=False),
        temperature=Temperature(value=30),
    )
    experiment = CalMorphExperiment(
        dataset_name="test",
        genotype=_genotype([gene]),
        environment=env,
        phenotype=CalMorphPhenotype(
            calmorph=base, calmorph_coefficient_of_variation=cv
        ),
    )
    reference = CalMorphExperimentReference(
        dataset_name="test",
        genome_reference=_reference_genome(),
        environment_reference=env.model_copy(),
        phenotype_reference=CalMorphPhenotype(
            calmorph=base, calmorph_coefficient_of_variation=cv
        ),
    )
    return {
        "experiment": experiment.model_dump(),
        "reference": reference.model_dump(),
        "publication": _publication(),
    }


def _visual_record(gene: str) -> Record:
    def phenotype(score: float, n: int) -> VisualScorePhenotype:
        return VisualScorePhenotype(
            visual_score=score,
            n_replicates=n,
            score_scale_min=-5,
            score_scale_max=5,
            score_semantics="higher = more orange",
            target_product="beta-carotene",
        )

    env = Environment(
        media=Media(name="SC-URA", state="solid", is_synthetic=True),
        temperature=Temperature(value=30),
    )
    experiment = VisualScoreExperiment(
        dataset_name="test",
        genotype=_genotype([gene]),
        environment=env,
        phenotype=phenotype(2.0, 2),
    )
    reference = VisualScoreExperimentReference(
        dataset_name="test",
        genome_reference=_reference_genome(),
        environment_reference=env.model_copy(),
        phenotype_reference=phenotype(0.0, 1),
    )
    return {"experiment": experiment.model_dump(), "reference": reference.model_dump()}


def _metabolite_record(gene: str, level: float, ref_level: float) -> Record:
    def phenotype(value: float, *, ref: bool) -> MetabolitePhenotype:
        return MetabolitePhenotype(
            metabolite_level={"betaxanthin": value},
            metabolite_level_se=None if ref else {"betaxanthin": 0.1},
            n_replicates={"betaxanthin": 1 if ref else 8},
            measurement_type="cri_spa_corrected_hsv_yellowness_24h",
        )

    env = Environment(
        media=Media(name="SC", state="solid", is_synthetic=True),
        temperature=Temperature(value=30),
    )
    experiment = MetaboliteExperiment(
        dataset_name="test",
        genotype=_genotype([gene]),
        environment=env,
        phenotype=phenotype(level, ref=False),
    )
    reference = MetaboliteExperimentReference(
        dataset_name="test",
        genome_reference=_reference_genome(),
        environment_reference=env.model_copy(),
        phenotype_reference=phenotype(ref_level, ref=True),
    )
    return {"experiment": experiment.model_dump(), "reference": reference.model_dump()}


def _protein_record(gene: str, scale: float) -> Record:
    prots = ["YAL003W", "YAL005C"]

    def phenotype(value: float) -> ProteinAbundancePhenotype:
        return ProteinAbundancePhenotype(
            protein_abundance={p: value + i for i, p in enumerate(prots)},
            protein_abundance_se={p: 0.1 for p in prots},
            n_replicates={p: 3 for p in prots},
            measurement_type="swath_ms_label_free_log_signal_sva",
        )

    env = Environment(
        media=Media(name="SM", state="liquid", is_synthetic=True),
        temperature=Temperature(value=30),
    )
    experiment = ProteinAbundanceExperiment(
        dataset_name="test",
        genotype=_genotype([gene]),
        environment=env,
        phenotype=phenotype(scale),
    )
    reference = ProteinAbundanceExperimentReference(
        dataset_name="test",
        genome_reference=_reference_genome(),
        environment_reference=env.model_copy(),
        phenotype_reference=phenotype(9.0),
    )
    return {"experiment": experiment.model_dump(), "reference": reference.model_dump()}


def _rnaseq_record(strain: str, genes: list[str]) -> Record:
    env = Environment(
        media=Media(name="SC", state="liquid", is_synthetic=True),
        temperature=Temperature(value=30),
    )
    tpm = {g: 2.0 + i for i, g in enumerate(genes)}
    count = {g: 10 + i for i, g in enumerate(genes)}
    experiment = RNASeqExpressionExperiment(
        dataset_name="test",
        genotype=Genotype(
            perturbations=[
                SequenceVariantPerturbation(
                    systematic_gene_name="YAL001C",
                    perturbed_gene_name="TFC3",
                    strain_id=strain,
                )
            ]
        ),
        environment=env,
        phenotype=RNASeqExpressionPhenotype(expression_tpm=tpm, expression_count=count),
    )
    reference = RNASeqExpressionExperimentReference(
        dataset_name="test",
        genome_reference=ReferenceGenome(
            species="Saccharomyces cerevisiae", strain="S288C"
        ),
        environment_reference=env.model_copy(),
        phenotype_reference=RNASeqExpressionPhenotype(
            expression_tpm=tpm, expression_count=count
        ),
    )
    return {"experiment": experiment.model_dump(), "reference": reference.model_dump()}


def _env_record(gene: str, value: float) -> Record:
    def phenotype(v: float) -> EnvironmentResponsePhenotype:
        return EnvironmentResponsePhenotype(
            measurement_type=MeasurementType.log2_ratio,
            environment_response=v,
            units="log2(treatment/control)",
        )

    env = Environment(
        media=YP_GALACTOSE,
        temperature=Temperature(value=30),
        perturbations=[
            SmallMoleculePerturbation(
                compound=Compound(name="hydroquinone", inchikey=HYDROQUINONE),
                concentration=Concentration(basis=DoseBasis.IC30),
            )
        ],
    )
    experiment = EnvironmentResponseExperiment(
        dataset_name="test",
        genotype=_genotype([gene]),
        environment=env,
        phenotype=phenotype(value),
    )
    reference = EnvironmentResponseExperimentReference(
        dataset_name="test",
        genome_reference=_reference_genome(),
        environment_reference=env.model_copy(),
        phenotype_reference=phenotype(0.0),
    )
    return {"experiment": experiment.model_dump(), "reference": reference.model_dump()}


def _fitness_record(gene: str, fitness: float) -> Record:
    def phenotype(v: float) -> FitnessPhenotype:
        return FitnessPhenotype(
            fitness=v, n_samples=2, sample_unit=SampleUnit.biological_replicate
        )

    env = Environment(media=SC, temperature=Temperature(value=30))
    experiment = FitnessExperiment(
        dataset_name="test",
        genotype=_genotype([gene]),
        environment=env,
        phenotype=phenotype(fitness),
    )
    reference = FitnessExperimentReference(
        dataset_name="test",
        genome_reference=_reference_genome(),
        environment_reference=env.model_copy(),
        phenotype_reference=phenotype(1.0),
    )
    return {"experiment": experiment.model_dump(), "reference": reference.model_dump()}


# --------------------------------------------------------------------------- #
# store + stub helpers
# --------------------------------------------------------------------------- #
def _write_lmdb(
    abs_root: Path, records: Iterable[Record], *, interned: dict[str, Any] | None = None
) -> Path:
    """Write ``<abs_root>/processed/lmdb`` (and ``interned`` when given); return the lmdb dir."""
    lmdb_dir = abs_root / "processed" / "lmdb"
    lmdb_dir.mkdir(parents=True)
    env = lmdb.open(str(lmdb_dir), map_size=int(5e7))
    with env.begin(write=True) as txn:
        for i, record in enumerate(records):
            txn.put(f"{i}".encode(), pickle.dumps(record))
    env.close()
    if interned is not None:
        interned_dir = abs_root / "processed" / "interned"
        interned_dir.mkdir()
        ienv = lmdb.open(str(interned_dir), map_size=int(5e7))
        with ienv.begin(write=True) as itxn:
            for key, value in interned.items():
                itxn.put(key.encode(), pickle.dumps(value))
        ienv.close()
    return lmdb_dir


def _read_report(abs_root: Path) -> dict[str, Any]:
    path = abs_root / "preprocess" / "verification_report.json"
    data: dict[str, Any] = json.loads(path.read_text())
    return data


def _names(report: dict[str, Any]) -> list[str]:
    return [str(r["name"]) for r in report["results"]]


def _result(report: dict[str, Any], name: str) -> dict[str, Any]:
    matches = [r for r in report["results"] if r["name"] == name]
    assert len(matches) == 1, _names(report)
    result: dict[str, Any] = matches[0]
    return result


class _Resolution:
    def __init__(self, status: str, systematic_name: str | None) -> None:
        self.status = status
        self.systematic_name = systematic_name


class _FakeGenome:
    """Stands in for SCerevisiaeGenome: every name is a current gene of itself."""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs

    def resolve_gene_name(self, name: str) -> _Resolution:
        return _Resolution("current", name)


def _stub_sgd(monkeypatch: pytest.MonkeyPatch, genes: set[str]) -> list[str]:
    """Replace ``_sgd_gene_set`` with a stub returning ``genes``; return its call log."""
    calls: list[str] = []

    def fake(data_root: str) -> set[str]:
        calls.append(data_root)
        return set(genes)

    monkeypatch.setattr(runners, "_sgd_gene_set", fake)
    return calls


def _stub_genome(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    calls: list[str] = []

    def fake(data_root: str) -> _FakeGenome:
        calls.append(data_root)
        return _FakeGenome()

    monkeypatch.setattr(runners, "_genome", fake)
    return calls


def _spec(root_name: str, **extra: Any) -> dict[str, Any]:
    return {"root": f"data/torchcell/{root_name}", "provenance": PROV, **extra}


def _root(tmp_path: Path, root_name: str) -> Path:
    return tmp_path / "data" / "torchcell" / root_name


# --------------------------------------------------------------------------- #
# store readers + report writer
# --------------------------------------------------------------------------- #
def test_load_interned_is_empty_without_the_dir_and_the_decoded_dict_with_it(
    tmp_path: Path,
) -> None:
    plain = tmp_path / "plain"
    _write_lmdb(plain, [{"i": 0}])
    assert runners._load_interned(str(plain)) == {}

    shared = {"h1": {"media": "SC"}, "h2": [1, 2, 3]}
    with_interned = tmp_path / "interned"
    _write_lmdb(with_interned, [{"i": 0}], interned=shared)
    assert runners._load_interned(str(with_interned)) == shared


def test_load_records_resolves_ref_pointers_in_lmdb_key_order(tmp_path: Path) -> None:
    environment = {"media": {"name": "SC"}, "temperature": {"value": 30.0}}
    records = [
        {
            "experiment": {
                "i": i,
                "environment": {"$ref": "env-hash", "name": "environment"},
            },
            "reference": {"environment_reference": {"$ref": "env-hash"}},
        }
        for i in range(11)
    ]
    root = tmp_path / "store"
    _write_lmdb(root, records, interned={"env-hash": environment})

    loaded = runners.load_records(str(root))
    assert [r["experiment"]["i"] for r in loaded] == [0, 1, 10, 2, 3, 4, 5, 6, 7, 8, 9]
    assert loaded[0] == {
        "experiment": {"i": 0, "environment": environment},
        "reference": {"environment_reference": environment},
    }
    assert all(r["experiment"]["environment"] == environment for r in loaded)


def test_stream_records_yields_load_records_and_closes_on_early_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "store"
    records = [{"i": i, "env": {"$ref": "e"}} for i in range(3)]
    _write_lmdb(root, records, interned={"e": {"name": "YPD"}})

    # record every env the runner opens (interned first, then the records env)
    opened: list[Any] = []
    real_open = lmdb.open

    def recording_open(*args: Any, **kwargs: Any) -> Any:
        env = real_open(*args, **kwargs)
        opened.append(env)
        return env

    # runners calls ``lmdb.open`` through the module object, so this patch reaches it
    monkeypatch.setattr(lmdb, "open", recording_open)

    generator = runners.stream_records(str(root))
    assert next(generator) == {"i": 0, "env": {"name": "YPD"}}
    assert len(opened) == 2
    generator.close()
    # the finally clause closed the records env: a transaction on it now fails
    with pytest.raises(lmdb.Error):
        opened[-1].begin()

    eager = runners.load_records(str(root))
    assert eager == [{"i": i, "env": {"name": "YPD"}} for i in range(3)]
    assert list(runners.stream_records(str(root))) == eager


def test_write_report_creates_the_dir_and_writes_indent2_json(tmp_path: Path) -> None:
    report = VerificationReport(dataset_name="d", provenance=Provenance(source_uri="u"))
    report.add(
        LevelResult(
            level=Level.L1, name="count", passed=True, message="m", details={"n": 1}
        )
    )
    report_dir = tmp_path / "nested" / "preprocess"
    out = runners._write_report(report, str(report_dir))
    assert out == str(report_dir / "verification_report.json")
    text = Path(out).read_text()
    assert text == report.model_dump_json(indent=2)
    assert json.loads(text) == {
        "dataset_name": "d",
        "provenance": {
            "source_uri": "u",
            "citation_key": None,
            "sha256": None,
            "method": None,
            "page": None,
            "retrieved": None,
        },
        "results": [
            {
                "level": 1,
                "name": "count",
                "passed": True,
                "message": "m",
                "details": {"n": 1},
            }
        ],
    }
    # a second write overwrites in place (exist_ok)
    assert runners._write_report(report, str(report_dir)) == out


# --------------------------------------------------------------------------- #
# L4 closed forms + genome helpers
# --------------------------------------------------------------------------- #
def test_l4_gene_containment_closed_forms() -> None:
    ohya = {"YAL001C", "YBR085W", "YJR155W"}
    partial = runners._l4_gene_containment(
        ohya, "microarray_kemmeren2014", {"YAL001C", "YBR085W", "YDR001C"}
    )
    assert partial.level is Level.L4
    assert partial.name == "gene_containment_microarray_kemmeren2014"
    assert partial.passed is False
    assert partial.message == (
        "0.667 of microarray_kemmeren2014's 3 deletion genes are in Ohya (>= 0.9)"
    )
    assert partial.details["n_other"] == 3
    assert partial.details["n_in_ohya"] == 2
    assert partial.details["overlap"] == pytest.approx(2 / 3)
    assert partial.details["missing_examples"] == ["YDR001C"]

    full = runners._l4_gene_containment(ohya, "x", {"YAL001C", "YJR155W"})
    assert full.passed is True
    assert full.details == {
        "n_other": 2,
        "n_in_ohya": 2,
        "overlap": 1.0,
        "missing_examples": [],
    }

    empty = runners._l4_gene_containment(ohya, "x", set())
    assert empty.passed is False
    assert empty.message == "0.000 of x's 0 deletion genes are in Ohya (>= 0.9)"
    assert empty.details["overlap"] == 0.0

    many = {f"YZZ{i:03d}W" for i in range(25)}
    capped = runners._l4_gene_containment(ohya, "x", many)
    assert capped.details["missing_examples"] == sorted(many)[:20]


def test_l4_rnaseq_gene_containment_closed_forms() -> None:
    sgd = {"YAL001C", "YBR085W"}
    result = runners._l4_rnaseq_gene_containment(sgd, {"YAL001C", "YBR085W", "YJR155W"})
    assert result.level is Level.L4
    assert result.name == "gene_containment_sgd"
    assert result.passed is False
    assert result.message == (
        "0.667 of 3 measured genes are S288C reference genes (>= 0.9)"
    )
    assert result.details["n_measured"] == 3
    assert result.details["n_in_sgd"] == 2
    assert result.details["overlap"] == pytest.approx(2 / 3)
    assert result.details["missing_examples"] == ["YJR155W"]

    empty = runners._l4_rnaseq_gene_containment(sgd, set())
    assert empty.passed is False
    assert empty.details == {
        "n_measured": 0,
        "n_in_sgd": 0,
        "overlap": 0.0,
        "missing_examples": [],
    }


def test_sgd_gene_set_reads_fasta_headers_through_the_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    orf = tmp_path / "orf.fasta"
    orf.write_text(">YAL001C TFC3 SGDID:S000000001\nATGATG\n>YAL002W VPS8\nGGG\n")
    rna = tmp_path / "rna.fasta"
    rna.write_text(">YNCA0001W tRNA\nGGG\nnot a header > YFAKE\n>Q0010\n")
    paths = {
        "orf_coding_all_R64-4-1_20230830.fasta": orf,
        "rna_coding_R64-4-1_20230830.fasta": rna,
    }
    calls: list[tuple[str, str, str | None]] = []

    def fake_resolve(
        assembly_set: str, filename: str, *, data_root: str | None = None
    ) -> str:
        calls.append((assembly_set, filename, data_root))
        return str(paths[filename])

    monkeypatch.setattr(runners, "resolve", fake_resolve)
    assert runners._sgd_gene_set("/root") == {
        "YAL001C",
        "YAL002W",
        "YNCA0001W",
        "Q0010",
    }
    assert calls == [
        (SGD_S288C_R64, "orf_coding_all_R64-4-1_20230830.fasta", "/root"),
        (SGD_S288C_R64, "rna_coding_R64-4-1_20230830.fasta", "/root"),
    ]


def test_genome_constructs_the_s288c_genome_without_overwrite(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "torchcell.sequence.genome.scerevisiae.SCerevisiaeGenome", _FakeGenome
    )
    genome = runners._genome("/root")
    assert isinstance(genome, _FakeGenome)
    assert genome.kwargs == {
        "genome_root": "/root/data/sgd/genome",
        "go_root": "/root/data/go",
        "overwrite": False,
    }


# --------------------------------------------------------------------------- #
# family runners
# --------------------------------------------------------------------------- #
def test_run_expression_compares_gene_universes_against_the_first_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    universe = ["YAL001C", "YBR085W", "YJR155W"]
    _write_lmdb(
        _root(tmp_path, "expr_ref"),
        [_expression_record(g, universe) for g in universe[:2]],
    )
    _write_lmdb(
        _root(tmp_path, "expr_same"), [_expression_record(universe[2], universe)]
    )
    monkeypatch.setattr(
        runners,
        "EXPRESSION_DATASETS",
        {
            "expr_ref": _spec("expr_ref", expected_count=2),
            "expr_same": _spec("expr_same", expected_count=1),
        },
    )
    assert runners.run_expression(str(tmp_path)) is True

    ref = _read_report(_root(tmp_path, "expr_ref"))
    assert ref["dataset_name"] == "expr_ref"
    assert _names(ref) == EXPRESSION_NAMES  # the reference universe gets no L4 row
    same = _read_report(_root(tmp_path, "expr_same"))
    assert _names(same) == EXPRESSION_NAMES + ["gene_universe_vs_expr_ref"]
    l4 = _result(same, "gene_universe_vs_expr_ref")
    assert l4["level"] == 4
    assert l4["passed"] is True
    assert l4["message"] == "3 overlapping entities agree within 0.0"
    assert l4["details"] == {
        "tol": 0.0,
        "n_overlap": 3,
        "n_disagreements": 0,
        "worst": [],
    }
    out = capsys.readouterr().out
    assert (
        f"  -> wrote {_root(tmp_path, 'expr_ref') / 'preprocess' / 'verification_report.json'}\n"
        in out
    )
    assert out.count("PASS") == 2


def test_run_expression_fails_on_a_gene_universe_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    universe = ["YAL001C", "YBR085W", "YJR155W"]
    _write_lmdb(_root(tmp_path, "expr_ref"), [_expression_record("YAL001C", universe)])
    _write_lmdb(
        _root(tmp_path, "expr_wide"),
        [_expression_record("YBR085W", universe + ["YDR001C"])],
    )
    monkeypatch.setattr(
        runners,
        "EXPRESSION_DATASETS",
        {
            "expr_ref": _spec("expr_ref", expected_count=1),
            "expr_wide": _spec("expr_wide", expected_count=1),
        },
    )
    assert runners.run_expression(str(tmp_path)) is False
    wide = _read_report(_root(tmp_path, "expr_wide"))
    l4 = _result(wide, "gene_universe_vs_expr_ref")
    assert l4["passed"] is False
    assert l4["message"] == "1/4 overlapping entities disagree > 0.0"
    assert l4["details"] == {
        "tol": 0.0,
        "n_overlap": 4,
        "n_disagreements": 1,
        "worst": [{"entity": "YDR001C", "a": 0.0, "b": 1.0, "diff": 1.0}],
    }
    assert [r["passed"] for r in wide["results"][:-1]] == [True] * 8


def test_run_morphology_adds_containment_for_each_built_deletion_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    ohya_root = tmp_path / runners.OHYA_SOURCE_ROOT
    lmdb_dir = _write_lmdb(ohya_root, [_morphology_record(g) for g in GENES])
    # only Kemmeren is built; the two Sameith sources are skipped, not failed
    kemmeren_root = tmp_path / runners.DELETION_GENE_SOURCES["microarray_kemmeren2014"]
    _write_lmdb(kemmeren_root, [_expression_record(g, GENES) for g in GENES[:2]])
    monkeypatch.setattr(runners, "OHYA_EXPECTED_COUNT", 3)

    assert runners.run_morphology(str(tmp_path)) is True
    report = _read_report(tmp_path / runners.OHYA_REPORT_ROOT)
    assert report["dataset_name"] == "scmd_ohya2005"
    assert (
        report["provenance"]["citation_key"]
        == "ohyaHighdimensionalLargescalePhenotyping2005a"
    )
    assert _names(report) == MORPHOLOGY_NAMES + [
        "gene_containment_microarray_kemmeren2014"
    ]
    assert _result(report, "count")["details"] == {"observed": 3, "expected": 3}
    assert _result(report, "gene_containment_microarray_kemmeren2014")["details"] == {
        "n_other": 2,
        "n_in_ohya": 2,
        "overlap": 1.0,
        "missing_examples": [],
    }
    out = capsys.readouterr().out
    report_path = (
        tmp_path / runners.OHYA_REPORT_ROOT / "preprocess" / "verification_report.json"
    )
    assert out.endswith(
        f"  -> verified LMDB: {lmdb_dir}\n  -> wrote report:  {report_path}\n\n"
    )


def test_run_morphology_pins_the_4718_mutant_oracle(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_lmdb(
        tmp_path / runners.OHYA_SOURCE_ROOT, [_morphology_record(g) for g in GENES]
    )
    assert runners.run_morphology(str(tmp_path)) is False
    report = _read_report(tmp_path / runners.OHYA_REPORT_ROOT)
    assert _names(report) == MORPHOLOGY_NAMES  # no deletion source built -> no L4
    count = _result(report, "count")
    assert count["passed"] is False
    assert count["details"] == {"observed": 3, "expected": 4718}
    assert capsys.readouterr().out.startswith("scmd_ohya2005: FAIL\n")


def test_run_morphology_ohnuki_uses_sgd_containment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_lmdb(
        tmp_path / runners.OHNUKI_SOURCE_ROOT, [_morphology_record(g) for g in GENES]
    )
    sgd_calls = _stub_sgd(monkeypatch, {"YAL001C", "YBR085W"})
    monkeypatch.setattr(runners, "OHNUKI_EXPECTED_COUNT", 3)

    assert runners.run_morphology_ohnuki(str(tmp_path)) is False
    assert sgd_calls == [str(tmp_path)]
    report = _read_report(tmp_path / runners.OHNUKI_REPORT_ROOT)
    assert report["dataset_name"] == "scmd_ohnuki2018"
    assert _names(report) == MORPHOLOGY_NAMES + ["gene_containment_sgd"]
    expected = runners._l4_rnaseq_gene_containment({"YAL001C", "YBR085W"}, set(GENES))
    assert _result(report, "gene_containment_sgd") == expected.model_dump(mode="json")
    assert [r["passed"] for r in report["results"][:-1]] == [True] * 7


def test_run_ohnuki_morphology_adds_ohya_containment_only_when_ohya_is_built(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runners, "OHNUKI2022_EXPECTED_COUNT", 2)
    _write_lmdb(
        tmp_path / runners.OHNUKI2022_SOURCE_ROOT,
        [_morphology_record(g) for g in ["YAL001C", "YDR001C"]],
    )
    assert runners.run_ohnuki_morphology(str(tmp_path)) is True
    without = _read_report(tmp_path / runners.OHNUKI2022_REPORT_ROOT)
    assert without["dataset_name"] == "scmd_ohnuki2022"
    assert _names(without) == MORPHOLOGY_NAMES

    _write_lmdb(
        tmp_path / runners.OHYA_SOURCE_ROOT, [_morphology_record(g) for g in GENES]
    )
    assert runners.run_ohnuki_morphology(str(tmp_path)) is False
    with_ohya = _read_report(tmp_path / runners.OHNUKI2022_REPORT_ROOT)
    assert _names(with_ohya) == MORPHOLOGY_NAMES + ["gene_containment_scmd_ohnuki2022"]
    l4 = _result(with_ohya, "gene_containment_scmd_ohnuki2022")
    assert l4["passed"] is False
    assert l4["details"]["n_other"] == 2
    assert l4["details"]["n_in_ohya"] == 1
    assert l4["details"]["overlap"] == 0.5
    assert l4["details"]["missing_examples"] == ["YDR001C"]
    # the dataset's own name is passed as other_name, so it reads as the gene owner
    assert l4["message"] == (
        "0.500 of scmd_ohnuki2022's 2 deletion genes are in Ohya (>= 0.9)"
    )


def test_run_visual_score_count_oracle_is_the_record_count_and_labels_l4_as_ohya(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Finding: ``run_visual_score`` (and the metabolite/protein runners) pass
    ``other_name="scmd_ohya2005"`` with the DATASET's own genes, so the L4 row is named
    ``gene_containment_scmd_ohya2005`` and its message reads "of scmd_ohya2005's N
    deletion genes are in Ohya" although the N genes are the visual-score dataset's.
    The count oracle is ``len(records)``, so the L1 count row can never fail here.
    """
    monkeypatch.setattr(runners, "VISUAL_SCORE_DATASETS", {"vis": _spec("vis")})
    _write_lmdb(_root(tmp_path, "vis"), [_visual_record(g) for g in GENES])
    assert runners.run_visual_score(str(tmp_path)) is True
    without = _read_report(_root(tmp_path, "vis"))
    assert _names(without) == [
        "structural",
        "count",
        "orf_uniqueness",
        "value_fidelity",
        "reference_zero",
        "target_product_set",
    ]
    assert _result(without, "count")["details"] == {"observed": 3, "expected": 3}

    _write_lmdb(
        tmp_path / runners.OHYA_SOURCE_ROOT, [_morphology_record(g) for g in GENES[:2]]
    )
    assert runners.run_visual_score(str(tmp_path)) is False
    with_ohya = _read_report(_root(tmp_path, "vis"))
    l4 = _result(with_ohya, "gene_containment_scmd_ohya2005")
    assert l4["message"] == (
        "0.667 of scmd_ohya2005's 3 deletion genes are in Ohya (>= 0.9)"
    )
    assert l4["details"]["missing_examples"] == ["YJR155W"]


def test_run_metabolite_forwards_spec_flags_and_writes_every_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    absolute = [_metabolite_record(g, 1.5, 1.0) for g in GENES]
    centered = [_metabolite_record(g, 0.3, 0.0) for g in GENES[:2]]
    _write_lmdb(_root(tmp_path, "meta_abs"), absolute)
    _write_lmdb(_root(tmp_path, "meta_ctr"), centered)
    monkeypatch.setattr(
        runners,
        "METABOLITE_DATASETS",
        {
            "meta_abs": _spec("meta_abs", expected_count=99, reference_centered=False),
            "meta_ctr": _spec("meta_ctr"),
        },
    )
    assert runners.run_metabolite(str(tmp_path)) is False

    abs_report = _read_report(_root(tmp_path, "meta_abs"))
    assert _names(abs_report) == [
        "structural",
        "count",
        "genotype_uniqueness",
        "value_fidelity",
        "se_nonnegative",
        "reference_finite",
        "measurement_type_consistent",
    ]
    assert _result(abs_report, "count")["details"] == {"observed": 3, "expected": 99}
    assert _result(abs_report, "reference_finite")["passed"] is True

    ctr_report = _read_report(
        _root(tmp_path, "meta_ctr")
    )  # written despite meta_abs failing
    assert _names(ctr_report)[5] == "reference_zero"
    assert _result(ctr_report, "count")["details"] == {"observed": 2, "expected": 2}
    assert all(r["passed"] for r in ctr_report["results"])

    # with Ohya built, every metabolite report gains the Ohya containment row last
    _write_lmdb(tmp_path / runners.OHYA_SOURCE_ROOT, [_morphology_record("YAL001C")])
    assert runners.run_metabolite(str(tmp_path)) is False
    ctr_l4 = _result(
        _read_report(_root(tmp_path, "meta_ctr")), "gene_containment_scmd_ohya2005"
    )
    assert ctr_l4["level"] == 4
    assert ctr_l4["passed"] is False
    assert ctr_l4["message"] == (
        "0.500 of scmd_ohya2005's 2 deletion genes are in Ohya (>= 0.9)"
    )
    assert ctr_l4["details"]["missing_examples"] == ["YBR085W"]
    assert _names(_read_report(_root(tmp_path, "meta_abs")))[-1] == (
        "gene_containment_scmd_ohya2005"
    )


def test_run_protein_forwards_allow_duplicate_orfs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    duplicates = [_protein_record("YAL001C", 8.0), _protein_record("YAL001C", 8.5)]
    _write_lmdb(_root(tmp_path, "prot_dup"), duplicates)
    _write_lmdb(_root(tmp_path, "prot_strict"), duplicates)
    monkeypatch.setattr(
        runners,
        "PROTEIN_DATASETS",
        {
            "prot_dup": _spec("prot_dup", expected_count=2, allow_duplicate_orfs=True),
            "prot_strict": _spec("prot_strict", expected_count=2),
        },
    )
    assert runners.run_protein(str(tmp_path)) is False
    relaxed = _result(_read_report(_root(tmp_path, "prot_dup")), "orf_uniqueness")
    assert relaxed["passed"] is True
    assert relaxed["message"] == "1 ORFs, 1 with multiple strains (expected)"
    strict = _result(_read_report(_root(tmp_path, "prot_strict")), "orf_uniqueness")
    assert strict["passed"] is False
    assert strict["message"] == "1 ORFs appear in multiple records"
    assert strict["details"] == {"n_orfs": 1, "n_duplicated": 1}
    assert _names(_read_report(_root(tmp_path, "prot_dup"))) == [
        "structural",
        "count",
        "orf_uniqueness",
        "value_fidelity",
        "se_nonnegative",
        "reference_finite",
        "measurement_type_consistent",
    ]

    # with Ohya built, the one knocked-out ORF is fully contained: the row passes
    _write_lmdb(tmp_path / runners.OHYA_SOURCE_ROOT, [_morphology_record("YAL001C")])
    assert runners.run_protein(str(tmp_path)) is False  # prot_strict still fails
    dup_l4 = _result(
        _read_report(_root(tmp_path, "prot_dup")), "gene_containment_scmd_ohya2005"
    )
    assert dup_l4["passed"] is True
    assert dup_l4["details"] == {
        "n_other": 1,
        "n_in_ohya": 1,
        "overlap": 1.0,
        "missing_examples": [],
    }


def test_run_rnaseq_appends_sgd_containment_after_the_family_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    genes = ["YAL001C", "YBR001W", "YCR001W"]
    _write_lmdb(
        _root(tmp_path, "rna"),
        [_rnaseq_record("AAA", genes), _rnaseq_record("BBB", genes[:2])],
    )
    sgd_calls = _stub_sgd(monkeypatch, {"YAL001C", "YBR001W"})
    monkeypatch.setattr(
        runners, "RNASEQ_DATASETS", {"rna": _spec("rna", expected_count=2)}
    )

    assert runners.run_rnaseq(str(tmp_path)) is False
    assert sgd_calls == [str(tmp_path)]
    report = _read_report(_root(tmp_path, "rna"))
    assert _names(report) == [
        "structural",
        "count",
        "strain_uniqueness",
        "tpm_value_fidelity",
        "count_value_fidelity",
        "measurement_type_consistent",
        "reference_finite",
        "gene_containment_sgd",
    ]
    l4 = _result(report, "gene_containment_sgd")
    assert l4["passed"] is False
    assert l4["details"]["n_measured"] == 3
    assert l4["details"]["n_in_sgd"] == 2
    assert l4["details"]["missing_examples"] == ["YCR001W"]
    assert [r["passed"] for r in report["results"][:-1]] == [True] * 7


def test_run_environment_response_dispatches_eager_and_streaming(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records = [_env_record(g, v) for g, v in zip(GENES, [-1.2, 0.8, -0.3])]
    _write_lmdb(_root(tmp_path, "env_eager"), records)
    _write_lmdb(_root(tmp_path, "env_stream"), records)
    sgd_calls = _stub_sgd(monkeypatch, set(GENES))
    genome_calls = _stub_genome(monkeypatch)
    monkeypatch.setattr(
        runners,
        "ENVIRONMENT_RESPONSE_DATASETS",
        {
            "env_eager": _spec(
                "env_eager", expected_count=3, background_genes=frozenset()
            ),
            "env_stream": _spec("env_stream", expected_count=3, stream=True),
        },
    )
    assert runners.run_environment_response(str(tmp_path)) is True
    assert sgd_calls == [str(tmp_path)]
    assert genome_calls == [str(tmp_path)]
    expected_names = [
        "structural",
        "count",
        "pair_uniqueness",
        "value_fidelity",
        "se_nonnegative",
        "measurement_type_consistent",
        "reference_zero",
        "environment_perturbed",
        "provenance_gaps",
        "canonical_gene_names",
        "uncertainty_sanity",
        "compound_identity",
        "media_compound_identity",
        "media_membership",
        "gene_containment_sgd",
        "current_genome_genes",
    ]
    for name in ("env_eager", "env_stream"):
        report = _read_report(_root(tmp_path, name))
        assert report["dataset_name"] == name
        assert _names(report) == expected_names
        assert _result(report, "count")["details"] == {"observed": 3, "expected": 3}
        assert _result(report, "gene_containment_sgd")["details"]["overlap"] == 1.0
        assert _result(report, "canonical_gene_names")["message"] == (
            "3 systematic names, one canonical spelling each, each current in the genome"
        )
        assert all(r["passed"] for r in report["results"])


def test_run_environment_response_streaming_spec_requires_an_expected_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Finding: the eager branch defaults ``expected_count`` to ``len(records)``; the
    streaming branch indexes ``spec["expected_count"]`` and raises before opening the
    store, so the two branches accept different registry specs.
    """
    _stub_sgd(monkeypatch, set(GENES))
    _stub_genome(monkeypatch)
    monkeypatch.setattr(
        runners,
        "ENVIRONMENT_RESPONSE_DATASETS",
        {"env_stream": _spec("env_stream", stream=True)},
    )
    with pytest.raises(KeyError, match="expected_count"):
        runners.run_environment_response(str(tmp_path))


def test_run_fitness_forwards_the_rnaseq_containment_floor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_lmdb(
        _root(tmp_path, "fit"),
        [_fitness_record(g, f) for g, f in zip(GENES, [0.8, 1.05, 0.3])],
    )
    _stub_sgd(monkeypatch, {"YAL001C", "YBR085W"})
    genome_calls = _stub_genome(monkeypatch)
    monkeypatch.setattr(
        runners, "FITNESS_DATASETS", {"fit": _spec("fit", expected_count=3)}
    )

    assert runners.run_fitness(str(tmp_path)) is False
    assert genome_calls == [str(tmp_path)]
    report = _read_report(_root(tmp_path, "fit"))
    assert _names(report)[-2:] == ["gene_containment_sgd", "current_genome_genes"]
    containment = _result(report, "gene_containment_sgd")
    assert containment["passed"] is False
    assert containment["message"] == (
        "0.667 of 3 measured genes are S288C reference genes (>= 0.9)"
    )
    assert _result(report, "current_genome_genes")["details"]["missing_records"] == {
        "YJR155W": 1
    }
    assert [r["passed"] for r in report["results"][:-2]] == [True] * 12


def test_run_segregant_growth_wires_the_streaming_verifier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    abs_root = _root(tmp_path, "seg")
    _write_lmdb(abs_root, [{"i": 0}, {"i": 1}])
    (abs_root / "preprocess").mkdir()
    (abs_root / "preprocess" / "gene_set.json").write_text(
        json.dumps(["YAL001C", "YBR001W"])
    )
    _stub_sgd(monkeypatch, {"YAL001C"})
    monkeypatch.setattr(
        "torchcell.sequence.genome.scerevisiae.SCerevisiaeGenome", _FakeGenome
    )
    resolve_calls: list[tuple[str, str, str | None]] = []

    def fake_resolve(
        assembly_set: str, filename: str, *, data_root: str | None = None
    ) -> str:
        resolve_calls.append((assembly_set, filename, data_root))
        return "/fake/idx.tsv"

    captured: dict[str, Any] = {}

    def fake_verify(records: Iterable[Record], **kwargs: Any) -> VerificationReport:
        captured.update(kwargs)
        captured["records"] = list(records)
        report = VerificationReport(
            dataset_name=kwargs["dataset_name"], provenance=kwargs["provenance"]
        )
        return report.add(
            LevelResult(
                level=Level.L1, name="count", passed=True, message="m", details={}
            )
        )

    monkeypatch.setattr(runners, "resolve", fake_resolve)
    monkeypatch.setattr(runners, "verify_segregant_growth_streaming", fake_verify)
    monkeypatch.setattr(
        runners,
        "SEGREGANT_GROWTH_DATASETS",
        {
            "seg": _spec(
                "seg",
                raw_mirror="torchcell-raw/bloomTest2019",
                assembly_index="idx.tsv",
                expected_count=5,
            )
        },
    )
    assert runners.run_segregant_growth(str(tmp_path)) is True
    assert resolve_calls == [(PETER2018_1011, "idx.tsv", str(tmp_path))]
    assert captured["records"] == [{"i": 0}, {"i": 1}]
    assert captured["dataset_name"] == "seg"
    assert captured["provenance"] is PROV
    assert captured["expected_count"] == 5
    assert captured["raw_dir"] == str(abs_root / "raw")
    assert captured["raw_mirror"] == str(tmp_path / "torchcell-raw" / "bloomTest2019")
    assert captured["assembly_index_path"] == "/fake/idx.tsv"
    assert captured["sgd_genes"] == {"YAL001C"}
    assert captured["gene_set"] == {"YAL001C", "YBR001W"}
    genome = captured["genome"]
    assert isinstance(genome, _FakeGenome)
    assert genome.kwargs == {
        "genome_root": str(tmp_path / "data" / "sgd" / "genome"),
        "go_root": str(tmp_path / "data" / "go"),
        "overwrite": False,
    }
    assert _names(_read_report(abs_root)) == ["count"]


# --------------------------------------------------------------------------- #
# run_all + main
# --------------------------------------------------------------------------- #
def _patch_runners(
    monkeypatch: pytest.MonkeyPatch, failing: set[str]
) -> list[tuple[str, str]]:
    calls: list[tuple[str, str]] = []

    def make(name: str) -> Callable[[str], bool]:
        def run(data_root: str) -> bool:
            calls.append((name, data_root))
            return name not in failing

        return run

    for name in RUN_ALL_ORDER:
        monkeypatch.setattr(runners, name, make(name))
    return calls


def test_run_all_calls_every_family_in_order_without_short_circuit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _patch_runners(monkeypatch, failing={"run_morphology"})
    assert runners.run_all("/root") is False
    assert calls == [(name, "/root") for name in RUN_ALL_ORDER]

    calls = _patch_runners(monkeypatch, failing=set())
    assert runners.run_all("/root") is True
    assert [name for name, _ in calls] == RUN_ALL_ORDER

    calls = _patch_runners(monkeypatch, failing={"run_segregant_growth"})
    assert runners.run_all("/root") is False
    assert calls == [(name, "/root") for name in RUN_ALL_ORDER]


def test_main_returns_the_shell_code_and_prints_the_banner(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    dotenv_calls: list[bool] = []
    monkeypatch.setattr("dotenv.load_dotenv", lambda: dotenv_calls.append(True))
    monkeypatch.setenv("DATA_ROOT", "/data/root")
    seen: list[str] = []
    verdict = {"value": True}

    def fake_run_all(data_root: str) -> bool:
        seen.append(data_root)
        return verdict["value"]

    monkeypatch.setattr(runners, "run_all", fake_run_all)
    assert runners.main() == 0
    assert seen == ["/data/root"]
    assert dotenv_calls == [True]
    assert capsys.readouterr().out == "=" * 60 + "\nALL DATASETS PASS\n"

    verdict["value"] = False
    assert runners.main() == 1
    assert capsys.readouterr().out == "=" * 60 + "\nSOME DATASETS FAILED\n"


def test_main_requires_data_root_in_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("dotenv.load_dotenv", lambda: None)
    monkeypatch.delenv("DATA_ROOT", raising=False)
    monkeypatch.setattr(runners, "run_all", lambda data_root: True)
    with pytest.raises(KeyError, match="DATA_ROOT"):
        runners.main()


# --------------------------------------------------------------------------- #
# registry oracles (the numbers a rebuild is judged against)
# --------------------------------------------------------------------------- #
def _oracles(registry: dict[str, dict[str, Any]]) -> dict[str, int | None]:
    return {name: spec.get("expected_count") for name, spec in registry.items()}


def test_registry_count_oracles_and_flags_are_pinned() -> None:
    assert _oracles(runners.EXPRESSION_DATASETS) == {
        "dm_microarray_sameith2015": 72,
        "sm_microarray_sameith2015": 82,
        "microarray_kemmeren2014": 1484,
    }
    assert (runners.OHYA_EXPECTED_COUNT, runners.OHNUKI_EXPECTED_COUNT) == (4718, 1112)
    assert runners.OHNUKI2022_EXPECTED_COUNT == 1979
    assert _oracles(runners.VISUAL_SCORE_DATASETS) == {"carotenoid_ozaydin2013": None}
    assert _oracles(runners.METABOLITE_DATASETS) == {
        "betaxanthin_cachera2023": None,
        "amino_acid_mulleder2016": 4678,
        "amino_acid_cooper2010": 4313,
        "metabolite_zelezniak2018": 129,
        "metabolite_dasilveira2014": 127,
        "organic_acid_yoshida2012": 17,
        "isobutanol_screen_lopez2024": 4554,
        "isobutanol_validated_lopez2024": 224,
        "ffa_xue2025": 176,
    }
    assert {
        name
        for name, spec in runners.METABOLITE_DATASETS.items()
        if spec.get("reference_centered", True)
    } == {"betaxanthin_cachera2023"}
    # Issue #595: only Zelezniak releases several protocols, verified per protocol.
    assert {
        name
        for name, spec in runners.METABOLITE_DATASETS.items()
        if spec.get("protocol_measurement_types") is not None
    } == {"metabolite_zelezniak2018"}
    assert _oracles(runners.PROTEIN_DATASETS) == {
        "proteome_zelezniak2018": 97,
        "proteome_messner2023": 4699,
    }
    assert (
        runners.PROTEIN_DATASETS["proteome_messner2023"]["allow_duplicate_orfs"] is True
    )
    assert _oracles(runners.RNASEQ_DATASETS) == {
        "caudal_pantranscriptome2024": 943,
        "nadal_ribelles_perturbseq2025": 6188,
    }
    assert _oracles(runners.ENVIRONMENT_RESPONSE_DATASETS) == {
        "yeastphenome": 296777,
        "env_chemgen_vanacloig2022": 118662,
        "env_chemgen_mota2024": 1270,
        "env_chemgen_hoepfner2014": 3124319,
        "env_chemgen_wildenhain2015": 428206,
        "env_chemgen_auesukaree2009": 525,
        "env_chemgen_smith2006": 12747,
        "crispr_magic_lian2019": 266304,
        "crispri_mormino2022": 12,
        "env_chemgen_costanzo2021": 61430,
        "env_chemgen_hillenmeyer2008_het": 2712677,
        "env_chemgen_hillenmeyer2008_hom": 1063034,
        "crispri_chemgen_smith2016": 7053,
    }
    assert {
        name
        for name, spec in runners.ENVIRONMENT_RESPONSE_DATASETS.items()
        if spec.get("stream")
    } == {
        "env_chemgen_hoepfner2014",
        "env_chemgen_wildenhain2015",
        "crispr_magic_lian2019",
        "env_chemgen_hillenmeyer2008_het",
        "env_chemgen_hillenmeyer2008_hom",
    }
    assert {
        name: set(spec["background_genes"])
        for name, spec in runners.ENVIRONMENT_RESPONSE_DATASETS.items()
        if spec["background_genes"]
    } == {"crispri_mormino2022": {"BM3R1-HAA1-mTurquoise2", "sfpHluorin"}}
    assert _oracles(runners.FITNESS_DATASETS) == {
        "smf_oduibhir2014": 1312,
        "smf_baryshnikova2010": 5993,
    }
    assert _oracles(runners.SEGREGANT_GROWTH_DATASETS) == {"bloom2019": 530100}
    assert (runners.MIN_GENE_OVERLAP, runners.MIN_RNASEQ_GENE_CONTAINMENT) == (
        0.90,
        0.90,
    )
    assert runners.SGD_GENE_FASTAS == [
        "orf_coding_all_R64-4-1_20230830.fasta",
        "rna_coding_R64-4-1_20230830.fasta",
    ]


def test_every_registry_root_is_the_dev_tree_path_of_its_own_name() -> None:
    registries: list[dict[str, dict[str, Any]]] = [
        runners.EXPRESSION_DATASETS,
        runners.VISUAL_SCORE_DATASETS,
        runners.METABOLITE_DATASETS,
        runners.PROTEIN_DATASETS,
        runners.RNASEQ_DATASETS,
        runners.ENVIRONMENT_RESPONSE_DATASETS,
        runners.FITNESS_DATASETS,
        runners.SEGREGANT_GROWTH_DATASETS,
    ]
    roots = {
        name: spec["root"] for registry in registries for name, spec in registry.items()
    }
    assert roots == {name: f"data/torchcell/{name}" for name in roots}
    assert len(roots) == 33  # 3 + 1 + 9 + 2 + 2 + 13 + 2 + 1
    assert all(
        isinstance(spec["provenance"], Provenance)
        for registry in registries
        for spec in registry.values()
    )
    assert (runners.OHYA_SOURCE_ROOT, runners.OHYA_REPORT_ROOT) == (
        "data/torchcell/scmd_ohya2005",
        "data/torchcell/scmd_ohya2005",
    )
    assert runners.OHNUKI_SOURCE_ROOT == "data/torchcell/scmd_ohnuki2018"
    assert runners.OHNUKI2022_SOURCE_ROOT == "data/torchcell/scmd_ohnuki2022"
