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

The bacterial gene universes (2026.10.07) read a gzipped synthetic feature table and
protein FASTA through a stubbed ``resolve``; the per-reference selection is checked with
recording subclasses of the real genome classes
(``tests/torchcell/datasets/_genome_injection_fakes.py``), so a yeast reference gets an
``SCerevisiaeGenome`` and an assembly-pinned one its own strain's genome, none built. The
deposited-tier counts are data-gated.

Derived expectations: LMDB iterates keys in byte order, so eleven records written under
``"0"``..``"10"`` come back as 0, 1, 10, 2, ..., 9. An expression L4 with reference universe
{A, B, C} and other universe {A, B, C, D} has ``n_overlap`` 4 and one disagreement,
``{"entity": "D", "a": 0.0, "b": 1.0, "diff": 1.0}``. A gene containment of {A, B, D} in
Ohya's {A, B, C} is 2/3 = 0.667 against the 0.90 floor. ``run_all`` calls eleven family
runners in a fixed order and evaluates every one before combining with ``and``.
"""

from __future__ import annotations

import gzip
import json
import os
import pickle
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, Literal

import lmdb
import pytest

from tests.torchcell.datasets._genome_injection_fakes import (
    FakeYeastGenome,
    install_bacterial_fakes,
)
from torchcell.datamodels.calmorph_labels import CALMORPH_LABELS, CALMORPH_STATISTICS
from torchcell.datamodels.media import M9_NREL_CARRUTHERS2025, SC, YP_GALACTOSE
from torchcell.datamodels.schema import (
    AssemblyReferenceGenome,
    BacterialDeletionPerturbation,
    BacterialEnvironmentResponseExperiment,
    BacterialEnvironmentResponseExperimentReference,
    BacterialMetaboliteExperiment,
    BacterialMetaboliteExperimentReference,
    BacterialProteinAbundanceExperiment,
    BacterialProteinAbundanceExperimentReference,
    BacterialRNASeqExpressionExperiment,
    BacterialRNASeqExpressionExperimentReference,
    CalMorphExperiment,
    CalMorphExperimentReference,
    CalMorphPhenotype,
    Compound,
    Concentration,
    ConcentrationUnit,
    CultureEnvironment,
    DoseBasis,
    Environment,
    EnvironmentResponseExperiment,
    EnvironmentResponseExperimentReference,
    EnvironmentResponsePhenotype,
    FitnessExperiment,
    FitnessExperimentReference,
    FitnessPhenotype,
    Genotype,
    HeterologousPathwayPerturbation,
    KanMxDeletionPerturbation,
    MeasurementType,
    Media,
    MetaboliteExperiment,
    MetaboliteExperimentReference,
    MetabolitePhenotype,
    MicroarrayExpressionExperiment,
    MicroarrayExpressionExperimentReference,
    MicroarrayExpressionPhenotype,
    ProductTiterExperiment,
    ProductTiterExperimentReference,
    ProductTiterPhenotype,
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
    UncertaintyType,
    VisualScoreExperiment,
    VisualScoreExperimentReference,
    VisualScorePhenotype,
)
from torchcell.sequence.genome.ecoli.k12 import MG1655_ASSEMBLY, EcoliK12MG1655Genome
from torchcell.sequence.genome.ecoli.rel606 import EcoliBREL606Genome
from torchcell.sequence.genome.pputida.kt2440 import PPutidaKT2440Genome
from torchcell.sequence.genome.registry import (
    ECOLI_B_REL606,
    ECOLI_K12_BW25113,
    ECOLI_K12_MG1655,
    PETER2018_1011,
    PPUTIDA_KT2440,
    SGD_S288C_R64,
)
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome
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
    "run_product_titer",
    "run_bacterial_protein_abundance",
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
# bacterial gene universes and the per-reference selection
# --------------------------------------------------------------------------- #
FEATURE_TABLE_HEADER = (
    "# feature\tclass\tassembly\tassembly_unit\tseq_type\tchromosome\t"
    "genomic_accession\tstart\tend\tstrand\tproduct_accession\tnon-redundant_refseq\t"
    "related_accession\tname\tsymbol\tGeneID\tlocus_tag\tfeature_interval_length\t"
    "product_length\tattributes\n"
)


def _feature_row(feature: str, cls: str, accession: str, locus_tag: str) -> str:
    """One 20-column NCBI feature-table row; only the three read columns vary."""
    cells = [feature, cls, "GCA_000005845.2", "Primary Assembly", "chromosome", ""]
    cells += ["U00096.3", "1", "9", "+", accession, "", "", "", "", "", locus_tag]
    return "\t".join([*cells, "9", "", ""]) + "\n"


SYNTHETIC_TABLE = FEATURE_TABLE_HEADER + "".join(
    [
        _feature_row("gene", "protein_coding", "", "b0001"),
        _feature_row("CDS", "with_protein", "AAC73112.1", "b0001"),
        _feature_row("gene", "tRNA", "", "b0002"),
        _feature_row("tRNA", "", "", "b0002"),
        _feature_row("gene", "pseudogene", "", "b0003"),
        _feature_row("CDS", "without_protein", "", "b0003"),
    ]
)


def _serve_bacterial_members(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, table: str, proteins: str
) -> list[tuple[str, str, str | None]]:
    """Serve a gzipped feature table and protein FASTA through ``runners.resolve``."""
    paths: dict[str, Path] = {}
    for member, text in (
        ("GCA_000005845.2_ASM584v2_feature_table.txt.gz", table),
        ("GCA_000005845.2_ASM584v2_protein.faa.gz", proteins),
    ):
        paths[member] = tmp_path / member
        with gzip.open(paths[member], "wt") as handle:
            handle.write(text)
    calls: list[tuple[str, str, str | None]] = []

    def fake_resolve(
        assembly_set: str, filename: str, *, data_root: str | None = None
    ) -> str:
        calls.append((assembly_set, filename, data_root))
        return str(paths[filename])

    monkeypatch.setattr(runners, "resolve", fake_resolve)
    return calls


def test_bacterial_gene_set_is_every_gene_row_with_the_proteins_cross_checked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Protein-coding, tRNA and pseudogene rows are all loci; the one protein maps to
    b0001 through its CDS row; both members come from the MG1655 set.
    """
    calls = _serve_bacterial_members(
        monkeypatch, tmp_path, SYNTHETIC_TABLE, ">AAC73112.1 thr leader\nMK\n"
    )
    genes = runners._bacterial_gene_set(MG1655_ASSEMBLY, "/root")
    assert genes == {"b0001", "b0002", "b0003"}
    assert calls == [
        (ECOLI_K12_MG1655, "GCA_000005845.2_ASM584v2_feature_table.txt.gz", "/root"),
        (ECOLI_K12_MG1655, "GCA_000005845.2_ASM584v2_protein.faa.gz", "/root"),
    ]


@pytest.mark.parametrize(
    "table,proteins,message",
    [
        (
            SYNTHETIC_TABLE,
            ">AAC73112.1 a\nMK\n>AAC99999.1 b\nMK\n",
            "1 proteins have no CDS row in the feature table ['AAC99999.1']; 0 CDS locus "
            "tags are no gene row []",
        ),
        (
            SYNTHETIC_TABLE + _feature_row("CDS", "with_protein", "AAC5.1", "b0005"),
            ">AAC73112.1 a\nMK\n>AAC5.1 b\nMK\n",
            "0 proteins have no CDS row in the feature table []; 1 CDS locus tags are "
            "no gene row ['b0005']",
        ),
    ],
    ids=["protein-without-cds", "cds-without-gene"],
)
def test_bacterial_gene_set_refuses_members_that_disagree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    table: str,
    proteins: str,
    message: str,
) -> None:
    _serve_bacterial_members(monkeypatch, tmp_path, table, proteins)
    with pytest.raises(ValueError) as excinfo:
        runners._bacterial_gene_set(MG1655_ASSEMBLY, "/root")
    assert str(excinfo.value) == f"{ECOLI_K12_MG1655}: {message}"


def test_each_host_gene_set_reads_its_own_assembly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[tuple[str, str]] = []

    def fake(assembly: Any, data_root: str) -> set[str]:
        seen.append((assembly.assembly_set, data_root))
        return {assembly.genbank_assembly}

    monkeypatch.setattr(runners, "_bacterial_gene_set", fake)
    assert runners._ecoli_k12_gene_set("/r", "MG1655") == {"GCA_000005845.2_ASM584v2"}
    assert runners._ecoli_k12_gene_set("/r", "BW25113") == {
        "GCA_000750555.1_ASM75055v1"
    }
    assert runners._pputida_gene_set("/r") == {"GCA_000007565.2_ASM756v2"}
    assert seen == [
        (ECOLI_K12_MG1655, "/r"),
        (ECOLI_K12_BW25113, "/r"),
        (PPUTIDA_KT2440, "/r"),
    ]


def _assembly_reference(strain: Any, assembly_set: Any, accession: str) -> Record:
    species = "Pseudomonas putida" if strain == "KT2440" else "Escherichia coli"
    return AssemblyReferenceGenome(
        species=species,
        strain=strain,
        assembly_set=assembly_set,
        assembly_accession=accession,
    ).model_dump()


MG1655_REFERENCE = _assembly_reference(
    "MG1655", "ecoli_K12_MG1655_ASM584v2", "GCA_000005845.2"
)
KT2440_REFERENCE = _assembly_reference(
    "KT2440", "pputida_KT2440_ASM756v2", "GCA_000007565.2"
)
KT2440_REFERENCE_MODEL = AssemblyReferenceGenome.model_validate(KT2440_REFERENCE)


def test_reference_assembly_set_follows_the_record_s_own_reference() -> None:
    """The stored dump names the set; a reference without one must be yeast."""
    assert runners._reference_assembly_set(_reference_genome().model_dump()) == (
        SGD_S288C_R64
    )
    assert runners._reference_assembly_set(MG1655_REFERENCE) == ECOLI_K12_MG1655
    assert runners._reference_assembly_set(KT2440_REFERENCE) == PPUTIDA_KT2440
    with pytest.raises(ValueError, match="must be 'Saccharomyces cerevisiae', got"):
        runners._reference_assembly_set(
            {"species": "Escherichia coli", "strain": "K-12"}
        )
    with pytest.raises(
        ValueError, match="names assembly set 'peter2018_1011_assemblies'"
    ):
        runners._reference_assembly_set({"assembly_set": PETER2018_1011})


def test_gene_set_for_reference_selects_the_universe_by_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sgd_calls = _stub_sgd(monkeypatch, {"YAL001C"})
    bacterial: list[str] = []

    def fake(assembly: Any, data_root: str) -> set[str]:
        bacterial.append(assembly.assembly_set)
        return {"b0001"}

    monkeypatch.setattr(runners, "_bacterial_gene_set", fake)
    yeast = _reference_genome().model_dump()
    assert runners._gene_set_for_reference(yeast, "/r") == {"YAL001C"}
    assert runners._gene_set_for_reference(MG1655_REFERENCE, "/r") == {"b0001"}
    assert (sgd_calls, bacterial) == (["/r"], [ECOLI_K12_MG1655])


def test_genome_for_reference_hands_each_record_its_own_host_genome(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A yeast reference gets ``SCerevisiaeGenome``; MG1655 and KT2440 references get
    their own genomes from the default cache roots, read-only; S288C is never built for
    a bacterial reference.
    """
    log = install_bacterial_fakes(monkeypatch)
    monkeypatch.setattr(
        "torchcell.sequence.genome.scerevisiae.SCerevisiaeGenome", FakeYeastGenome
    )
    mg1655 = runners._genome_for_reference(MG1655_REFERENCE, "/root")
    kt2440 = runners._genome_for_reference(KT2440_REFERENCE, "/root")
    assert isinstance(mg1655, EcoliK12MG1655Genome)
    assert isinstance(kt2440, PPutidaKT2440Genome)
    yeast = runners._genome_for_reference(_reference_genome().model_dump(), "/root")
    assert isinstance(yeast, SCerevisiaeGenome)
    assert log == [
        (
            "FakeMG1655Genome",
            {"genome_root": "/root/data/ecoli/mg1655/genome", "overwrite": False},
        ),
        (
            "FakeKT2440Genome",
            {"genome_root": "/root/data/pputida/kt2440/genome", "overwrite": False},
        ),
        (
            "FakeYeastGenome",
            {
                "genome_root": "/root/data/sgd/genome",
                "go_root": "/root/data/go",
                "overwrite": False,
            },
        ),
    ]


REL606_REFERENCE = _assembly_reference(
    "REL606", "ecoli_B_REL606_ASM1798v1", "GCA_000017985.1"
)


def test_a_rel606_record_gets_the_rel606_universe_and_genome(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An E. coli B REL606 reference selects its own set, its ``ECB_`` locus-tag universe
    and its own genome (read-only, default root); no K-12 universe stands in for it.
    """
    seen: list[tuple[str, str]] = []

    def fake(assembly: Any, data_root: str) -> set[str]:
        seen.append((assembly.assembly_set, data_root))
        return {"ECB_00001"}

    monkeypatch.setattr(runners, "_bacterial_gene_set", fake)
    assert runners._reference_assembly_set(REL606_REFERENCE) == ECOLI_B_REL606
    assert runners._ecoli_rel606_gene_set("/r") == {"ECB_00001"}
    assert runners._gene_set_for_reference(REL606_REFERENCE, "/r") == {"ECB_00001"}
    assert seen == [(ECOLI_B_REL606, "/r"), (ECOLI_B_REL606, "/r")]
    log = install_bacterial_fakes(monkeypatch)
    genome = runners._genome_for_reference(REL606_REFERENCE, "/root")
    assert isinstance(genome, EcoliBREL606Genome)
    assert log == [
        (
            "FakeREL606Genome",
            {"genome_root": "/root/data/ecoli/rel606/genome", "overwrite": False},
        )
    ]


@pytest.mark.data
@pytest.mark.skipif(
    not os.path.isfile(
        os.path.join(
            os.environ.get("DATA_ROOT", ""),
            "torchcell-genomes",
            ECOLI_B_REL606,
            "manifest.json",
        )
    ),
    reason="requires the REL606 assembly set",
)
def test_rel606_gene_set_on_the_deposited_tier() -> None:
    """Every GenBank gene feature of GCA_000017985.1: 4,383 ``ECB_`` loci, tRNA and rRNA
    tags and pseudogenes included, with every protein's CDS on one of them.
    """
    data_root = os.environ["DATA_ROOT"]
    rel606 = runners._ecoli_rel606_gene_set(data_root)
    assert len(rel606) == 4383
    assert {"ECB_00001", "ECB_t00001", "ECB_r00001", "ECB_00042"} <= rel606
    assert runners._gene_set_for_reference(REL606_REFERENCE, data_root) == rel606


BACTERIAL_TIER = all(
    os.path.isfile(
        os.path.join(
            os.environ.get("DATA_ROOT", ""), "torchcell-genomes", s, "manifest.json"
        )
    )
    for s in (ECOLI_K12_MG1655, ECOLI_K12_BW25113, PPUTIDA_KT2440)
)


@pytest.mark.data
@pytest.mark.skipif(not BACTERIAL_TIER, reason="requires the bacterial assembly sets")
def test_bacterial_gene_sets_on_the_deposited_tier() -> None:
    """Every GenBank gene feature: 4,651 MG1655, 4,490 BW25113, 5,786 KT2440 loci."""
    data_root = os.environ["DATA_ROOT"]
    mg1655 = runners._ecoli_k12_gene_set(data_root, "MG1655")
    bw25113 = runners._ecoli_k12_gene_set(data_root, "BW25113")
    kt2440 = runners._pputida_gene_set(data_root)
    assert (len(mg1655), len(bw25113), len(kt2440)) == (4651, 4490, 5786)
    assert {"b0001", "b4403"} <= mg1655 and "BW25113_0001" in bw25113
    assert {"PP_0001", "PP_16SA", "PP_t01"} <= kt2440
    assert runners._gene_set_for_reference(MG1655_REFERENCE, data_root) == mg1655


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


def _bacterial_env_record(locus: str, value: float) -> Record:
    """One KT2440 environment-response record: a locus-tag deletion under an inhibitor."""
    phenotype = EnvironmentResponsePhenotype(
        measurement_type=MeasurementType.log2_ratio,
        environment_response=value,
        units="log2(treatment/control)",
    )
    env = Environment(
        media=M9_NREL_CARRUTHERS2025,
        temperature=Temperature(value=30),
        perturbations=[
            SmallMoleculePerturbation(
                compound=Compound(name="hydroquinone", inchikey=HYDROQUINONE),
                concentration=Concentration(basis=DoseBasis.IC30),
            )
        ],
    )
    return {
        "experiment": BacterialEnvironmentResponseExperiment(
            dataset_name="env_bacterial",
            genotype=Genotype(
                perturbations=[
                    BacterialDeletionPerturbation(
                        systematic_gene_name=locus,
                        perturbed_gene_name=locus,
                        gene_namespace=KT2440_NAMESPACE,
                    )
                ]
            ),
            environment=env,
            phenotype=phenotype,
        ).model_dump(),
        "reference": BacterialEnvironmentResponseExperimentReference(
            dataset_name="env_bacterial",
            genome_reference=KT2440_REFERENCE_MODEL,
            environment_reference=env.model_copy(),
            phenotype_reference=EnvironmentResponsePhenotype(
                measurement_type=MeasurementType.log2_ratio,
                environment_response=0.0,
                units="log2(treatment/control)",
            ),
        ).model_dump(),
    }


KT2440_LOCI = ["PP_0001", "PP_0002", "PP_0003"]


def _stub_genome_for_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[Any, str]]:
    """Record the (assembly set, data root) every resolver was built from.

    The recording genome fakes subclass the real classes and record kwargs only, so
    their ``resolve_gene_name`` is the real method over no index; what this test needs
    is a resolver, plus proof that the one each dataset got was built from ITS OWN
    reference. ``_genome_for_reference`` has its own tests for the selection itself.
    """
    seen: list[tuple[Any, str]] = []

    def fake(reference: Any, data_root: str) -> _FakeGenome:
        seen.append((reference.get("assembly_set"), data_root))
        return _FakeGenome()

    monkeypatch.setattr(runners, "_genome_for_reference", fake)
    return seen


def test_run_environment_response_selects_the_universe_and_resolver_per_host(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bacterial dataset gets the KT2440 locus universe, the KT2440 genome and a
    containment row that names KT2440; the yeast dataset beside it still gets S288C.

    Before this, every registered dataset was handed ``_sgd_gene_set`` and the S288C
    resolver, so these three ``PP_`` tags would have been off the universe and off the
    genome: containment 0.000 and three off-genome records, all for the wrong reason.
    """
    _write_lmdb(
        _root(tmp_path, "env_yeast"),
        [_env_record(g, v) for g, v in zip(GENES, [-1.2, 0.8, -0.3])],
    )
    _write_lmdb(
        _root(tmp_path, "env_bact"),
        [_bacterial_env_record(t, v) for t, v in zip(KT2440_LOCI, [-1.0, 0.5, -0.2])],
    )
    sgd_calls = _stub_sgd(monkeypatch, set(GENES))
    bacterial = _stub_bacterial_universe(monkeypatch, set(KT2440_LOCI))
    resolvers = _stub_genome_for_reference(monkeypatch)
    monkeypatch.setattr(
        runners,
        "ENVIRONMENT_RESPONSE_DATASETS",
        {
            "env_yeast": _spec("env_yeast", expected_count=3),
            "env_bact": _spec("env_bact", expected_count=3),
        },
    )
    assert runners.run_environment_response(str(tmp_path)) is True
    assert (sgd_calls, bacterial) == ([str(tmp_path)], [PPUTIDA_KT2440])
    assert resolvers == [(None, str(tmp_path)), (PPUTIDA_KT2440, str(tmp_path))]
    yeast = _result(_read_report(_root(tmp_path, "env_yeast")), "gene_containment_sgd")
    assert yeast["message"] == (
        "1.000 of 3 measured genes are S288C reference genes (>= 0.9)"
    )
    bact = _result(_read_report(_root(tmp_path, "env_bact")), "gene_containment_sgd")
    assert bact["message"] == (
        "1.000 of 3 measured genes are pputida_KT2440_ASM756v2 locus genes (>= 0.9)"
    )
    assert (
        _result(_read_report(_root(tmp_path, "env_bact")), "current_genome_genes")[
            "passed"
        ]
        is True
    )


def test_run_environment_response_reads_a_streamed_datasets_host_from_one_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The streaming branch cannot walk the records twice, so it peeks the first one."""
    _write_lmdb(
        _root(tmp_path, "env_bact"),
        [_bacterial_env_record(t, v) for t, v in zip(KT2440_LOCI, [-1.0, 0.5, -0.2])],
    )
    sgd_calls = _stub_sgd(monkeypatch, set(GENES))
    bacterial = _stub_bacterial_universe(monkeypatch, set(KT2440_LOCI))
    resolvers = _stub_genome_for_reference(monkeypatch)
    monkeypatch.setattr(
        runners,
        "ENVIRONMENT_RESPONSE_DATASETS",
        {"env_bact": _spec("env_bact", expected_count=3, stream=True)},
    )
    assert runners.run_environment_response(str(tmp_path)) is True
    assert (sgd_calls, bacterial) == ([], [PPUTIDA_KT2440])
    assert resolvers == [(PPUTIDA_KT2440, str(tmp_path))]
    assert _result(_read_report(_root(tmp_path, "env_bact")), "gene_containment_sgd")[
        "message"
    ] == ("1.000 of 3 measured genes are pputida_KT2440_ASM756v2 locus genes (>= 0.9)")


def test_first_genome_reference_refuses_an_empty_store(tmp_path: Path) -> None:
    """A store with no records names no host, and the runner says so."""
    _write_lmdb(_root(tmp_path, "env_empty"), [])
    with pytest.raises(
        ValueError, match="holds no records, so its host cannot be read"
    ):
        runners._first_genome_reference(str(_root(tmp_path, "env_empty")))


def test_host_for_dataset_refuses_two_assembly_sets_in_one_dataset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One resolver cannot serve two hosts, so a mixed-pin dataset is refused."""
    _stub_bacterial_universe(monkeypatch, {"b0001"})
    _stub_genome_for_reference(monkeypatch)
    with pytest.raises(ValueError, match="one resolver cannot serve two hosts"):
        runners._host_for_dataset(
            "mixed",
            (ECOLI_K12_BW25113, ECOLI_K12_MG1655),
            MG1655_REFERENCE,
            "/root",
            {},
        )


def test_gene_universe_label_names_the_host_it_was_built_from() -> None:
    """The containment row's claim: S288C for a yeast universe, the pins for a bacterial."""
    assert runners._gene_universe_label((SGD_S288C_R64,)) == "S288C reference"
    assert runners._gene_universe_label((PPUTIDA_KT2440,)) == (
        "pputida_KT2440_ASM756v2 locus"
    )
    assert runners._gene_universe_label((ECOLI_K12_BW25113, ECOLI_K12_MG1655)) == (
        "ecoli_K12_BW25113_ASM75055v1 / ecoli_K12_MG1655_ASM584v2 locus"
    )


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
        "metabolome_fuhrer2017": 3735,
        "metabolome_rapp2026": 1496,
        "targeted_metabolome_rapp2026": 406,
        "metabolite_intensity_rapp2026": 406,
    }
    assert {
        name
        for name, spec in runners.METABOLITE_DATASETS.items()
        if spec.get("reference_centered", True)
    } == {"betaxanthin_cachera2023"}
    # The two bacterial metabolomes: their L4 universe comes from their own assembly
    # pin (BW25113 for Fuhrer, MG1655 for Rapp), not the Ohya yeast deletion collection.
    assert {
        name
        for name, spec in runners.METABOLITE_DATASETS.items()
        if "fuhrer" in name or "rapp" in name
    } == {
        "metabolome_fuhrer2017",
        "metabolome_rapp2026",
        "targeted_metabolome_rapp2026",
        "metabolite_intensity_rapp2026",
    }
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
        "rnaseq_lamoureux2023": 241,
        "rnaseq_public_k12_lamoureux2023": 240,
        "putida_precise321_lim2022": 180,
    }
    # The three bacterial compendia release one row per LIBRARY, so their L1 is the
    # replicate-group rule; the yeast rows keep one record per (strain, condition).
    assert {
        name
        for name, spec in runners.RNASEQ_DATASETS.items()
        if spec.get("replicate_aware")
    } == {
        "rnaseq_lamoureux2023",
        "rnaseq_public_k12_lamoureux2023",
        "putida_precise321_lim2022",
    }
    assert {
        name: spec["min_containment"]
        for name, spec in runners.RNASEQ_DATASETS.items()
        if "min_containment" in spec
    } == {
        "rnaseq_lamoureux2023": 0.99,
        "rnaseq_public_k12_lamoureux2023": 0.99,
        "putida_precise321_lim2022": 1.0,
    }
    assert _oracles(runners.ENVIRONMENT_RESPONSE_DATASETS) == {
        "yeastphenome": 296777,
        "env_chemgen_vanacloig2022": 118662,
        "env_chemgen_mota2024": 1270,
        "env_chemgen_hoepfner2014": 3083827,
        "env_chemgen_wildenhain2015": 430820,
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
        "growth_auc_rapp2026": 1514,
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
    assert len(roots) == 41  # 3 + 1 + 13 + 2 + 5 + 13 + 3 + 1
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


# --------------------------------------------------------------------------- #
# Host-aware L4 and the replicate-aware RNA-seq dispatch
# --------------------------------------------------------------------------- #
BW25113_REFERENCE = _assembly_reference(
    "BW25113", "ecoli_K12_BW25113_ASM75055v1", "GCA_000750555.1"
)


def _bacterial_rnaseq_record(tpm: dict[str, float], reference: Record) -> Record:
    """One library of a bacterial compendium: no perturbation, so no strain id."""
    env = Environment(
        media=Media(name="M9", state="liquid", is_synthetic=True),
        temperature=Temperature(value=37),
    )
    count = {gene: 10 + index for index, gene in enumerate(sorted(tpm))}
    experiment = BacterialRNASeqExpressionExperiment(
        dataset_name="test",
        genotype=Genotype(perturbations=[]),
        environment=env,
        phenotype=RNASeqExpressionPhenotype(expression_tpm=tpm, expression_count=count),
    )
    ref = BacterialRNASeqExpressionExperimentReference(
        dataset_name="test",
        genome_reference=AssemblyReferenceGenome.model_validate(reference),
        environment_reference=env.model_copy(),
        phenotype_reference=RNASeqExpressionPhenotype(
            expression_tpm={gene: 1.0 for gene in tpm},
            expression_count={gene: 5 for gene in tpm},
        ),
    )
    return {"experiment": experiment.model_dump(), "reference": ref.model_dump()}


def _bacterial_metabolite_record(locus: str, level: float) -> Record:
    env = Environment(
        media=Media(name="M9", state="liquid", is_synthetic=True),
        temperature=Temperature(value=37),
    )

    def phenotype(value: float) -> MetabolitePhenotype:
        return MetabolitePhenotype(
            metabolite_level={"neg_0001": value},
            n_replicates={"neg_0001": 2},
            measurement_type="fia_tof_ms_ion_modified_z_score",
        )

    experiment = BacterialMetaboliteExperiment(
        dataset_name="test",
        genotype=Genotype(
            perturbations=[
                BacterialDeletionPerturbation(
                    systematic_gene_name=locus,
                    perturbed_gene_name="thrA",
                    gene_namespace="ecoli_k12_bw25113_locus_tag",
                )
            ]
        ),
        environment=env,
        phenotype=phenotype(level),
    )
    reference = BacterialMetaboliteExperimentReference(
        dataset_name="test",
        genome_reference=AssemblyReferenceGenome.model_validate(BW25113_REFERENCE),
        environment_reference=env.model_copy(),
        phenotype_reference=phenotype(0.5),
    )
    return {"experiment": experiment.model_dump(), "reference": reference.model_dump()}


def test_dataset_assembly_sets_is_the_union_of_the_records_own_pins() -> None:
    """A dataset pinned to two strains (Tong's Keio + sRNA library) names both, sorted."""
    mg1655 = _bacterial_rnaseq_record({"b0001": 2.0}, MG1655_REFERENCE)
    bw25113 = _bacterial_rnaseq_record({"BW25113_0001": 2.0}, BW25113_REFERENCE)
    assert runners._dataset_assembly_sets([mg1655]) == (ECOLI_K12_MG1655,)
    assert runners._dataset_assembly_sets([bw25113, mg1655]) == (
        ECOLI_K12_BW25113,
        ECOLI_K12_MG1655,
    )
    yeast = _rnaseq_record("AAA", ["YAL001C"])
    assert runners._dataset_assembly_sets([yeast]) == (SGD_S288C_R64,)


def test_dataset_gene_universe_unions_both_pins_and_reads_neither_twice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reads: list[str] = []

    def fake(assembly: Any, data_root: str) -> set[str]:
        reads.append(assembly.assembly_set)
        return {"b0001"} if assembly.assembly_set == ECOLI_K12_MG1655 else {"BW25113_1"}

    monkeypatch.setattr(runners, "_bacterial_gene_set", fake)
    records = [
        _bacterial_rnaseq_record({"b0001": 2.0}, MG1655_REFERENCE),
        _bacterial_rnaseq_record({"b0001": 2.1}, MG1655_REFERENCE),
        _bacterial_rnaseq_record({"BW25113_1": 2.0}, BW25113_REFERENCE),
    ]
    universe, assembly_sets = runners._dataset_gene_universe(records, "/root")
    assert universe == {"b0001", "BW25113_1"}
    assert assembly_sets == (ECOLI_K12_BW25113, ECOLI_K12_MG1655)
    assert reads == [ECOLI_K12_BW25113, ECOLI_K12_MG1655]


def test_l4_assembly_gene_containment_closed_forms() -> None:
    universe = {"b0001", "b0002"}
    partial = runners._l4_assembly_gene_containment(
        universe, (ECOLI_K12_MG1655,), {"b0001", "b0002", "b9999"}
    )
    assert partial.level is Level.L4
    assert partial.name == "gene_containment_assembly"
    assert partial.passed is False
    assert partial.message == (
        "0.667 of 3 measured genes are loci of ecoli_K12_MG1655_ASM584v2 (>= 0.9)"
    )
    assert partial.details == {
        "assembly_sets": [ECOLI_K12_MG1655],
        "n_measured": 3,
        "n_in_universe": 2,
        "n_universe": 2,
        "overlap": pytest.approx(2 / 3),
        "missing_examples": ["b9999"],
    }

    # a floor of 1.0 is what a dataset states when every id must be a locus of the pin
    strict = runners._l4_assembly_gene_containment(
        universe, (ECOLI_K12_MG1655,), {"b0001", "b9999"}, min_containment=1.0
    )
    assert strict.passed is False
    assert "(>= 1.0)" in strict.message

    both = runners._l4_assembly_gene_containment(
        {"b0001", "BW25113_1"},
        (ECOLI_K12_BW25113, ECOLI_K12_MG1655),
        {"b0001", "BW25113_1"},
    )
    assert both.passed is True
    assert both.details["assembly_sets"] == [ECOLI_K12_BW25113, ECOLI_K12_MG1655]

    # an empty measured set fails, as it does in the two sibling rules
    empty = runners._l4_assembly_gene_containment(universe, (ECOLI_K12_MG1655,), set())
    assert empty.passed is False
    assert empty.message == (
        "no measured genes, so containment in ecoli_K12_MG1655_ASM584v2 has nothing "
        "to check"
    )
    assert empty.details["n_measured"] == 0


def test_run_rnaseq_uses_the_replicate_rule_and_the_assembly_universe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bacterial compendium gets ``replicate_groups`` + the pin's universe, and the
    S288C gene set is never read for it.
    """
    records = [
        _bacterial_rnaseq_record({"b0001": 2.0, "b0002": 4.0}, MG1655_REFERENCE),
        _bacterial_rnaseq_record({"b0001": 2.1, "b0002": 3.9}, MG1655_REFERENCE),
    ]
    _write_lmdb(_root(tmp_path, "rna_bact"), records)
    sgd_calls = _stub_sgd(monkeypatch, {"YAL001C"})
    monkeypatch.setattr(runners, "_bacterial_gene_set", lambda a, r: {"b0001", "b0002"})
    monkeypatch.setattr(
        runners,
        "RNASEQ_DATASETS",
        {
            "rna_bact": _spec(
                "rna_bact", expected_count=2, replicate_aware=True, min_containment=1.0
            )
        },
    )

    assert runners.run_rnaseq(str(tmp_path)) is True
    assert sgd_calls == []  # no yeast dataset in the registry, so no SGD read
    report = _read_report(_root(tmp_path, "rna_bact"))
    assert _names(report) == [
        "structural",
        "count",
        "replicate_groups",
        "tpm_value_fidelity",
        "count_value_fidelity",
        "measurement_type_consistent",
        "reference_finite",
        "gene_containment_assembly",
    ]
    l4 = _result(report, "gene_containment_assembly")
    assert l4["details"]["assembly_sets"] == [ECOLI_K12_MG1655]
    assert l4["details"]["n_measured"] == 2
    assert _result(report, "replicate_groups")["details"]["n_groups"] == 1


def test_run_metabolite_swaps_the_ohya_rule_for_the_assembly_rule(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bacterial metabolome is checked against its own strain, not against Ohya.

    Ohya is built here, so the yeast rule WOULD have been added; the dataset's own pin
    is what selects the other rule.
    """
    _write_lmdb(tmp_path / runners.OHYA_SOURCE_ROOT, [_morphology_record("YAL001C")])
    _write_lmdb(
        _root(tmp_path, "met_bact"),
        [
            _bacterial_metabolite_record("BW25113_0002", 1.5),
            _bacterial_metabolite_record("BW25113_0003", -0.5),
        ],
    )
    monkeypatch.setattr(
        runners, "_bacterial_gene_set", lambda a, r: {"BW25113_0002", "BW25113_0003"}
    )
    monkeypatch.setattr(
        runners,
        "METABOLITE_DATASETS",
        {"met_bact": _spec("met_bact", expected_count=2, reference_centered=False)},
    )

    assert runners.run_metabolite(str(tmp_path)) is True
    report = _read_report(_root(tmp_path, "met_bact"))
    assert "gene_containment_scmd_ohya2005" not in _names(report)
    l4 = _result(report, "gene_containment_assembly")
    assert l4["passed"] is True
    assert l4["details"]["assembly_sets"] == [ECOLI_K12_BW25113]
    assert l4["details"]["overlap"] == 1.0


def test_run_fitness_refuses_a_dataset_pinned_to_two_hosts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One resolver cannot serve two hosts, so a two-pin dataset stops rather than
    resolving half its names against the wrong annotation.
    """
    _write_lmdb(
        _root(tmp_path, "fit_two"),
        [
            {
                **_fitness_record("YAL001C", 0.9),
                "reference": {
                    **_fitness_record("YAL001C", 0.9)["reference"],
                    "genome_reference": MG1655_REFERENCE,
                },
            },
            {
                **_fitness_record("YBR085W", 0.8),
                "reference": {
                    **_fitness_record("YBR085W", 0.8)["reference"],
                    "genome_reference": BW25113_REFERENCE,
                },
            },
        ],
    )
    monkeypatch.setattr(runners, "_bacterial_gene_set", lambda a, r: {"b0001"})
    monkeypatch.setattr(
        runners, "FITNESS_DATASETS", {"fit_two": _spec("fit_two", expected_count=2)}
    )
    with pytest.raises(ValueError, match="records name 2 assembly sets"):
        runners.run_fitness(str(tmp_path))


# --------------------------------------------------------------------------- #
# The two bioproduction runners (2026.10.07): they dispatch to each dataset's own
# entry point and add the host-aware locus containment.
# --------------------------------------------------------------------------- #
KT2440_NAMESPACE: Literal["pputida_kt2440_locus_tag"] = "pputida_kt2440_locus_tag"


def _pathway_perturbation(token: str, organism: str) -> HeterologousPathwayPerturbation:
    return HeterologousPathwayPerturbation(
        systematic_gene_name=token,
        perturbed_gene_name=token,
        source_organism=organism,
        is_heterologous=True,
        localization="plasmid",
        gene_namespace=KT2440_NAMESPACE,
        pathway_name="isoprenol via mevalonate",
    )


def _kt2440_genotype(deleted: list[str], *, native_copy: str | None = None) -> Genotype:
    """A pIY670-style pathway plus host deletions, and optionally a native extra copy."""
    perturbations: list[Any] = [
        _pathway_perturbation("MvaSEf", "Enterococcus faecalis"),
        _pathway_perturbation("MKMm", "Methanosarcina mazei"),
    ]
    if native_copy is not None:
        perturbations.append(_pathway_perturbation(native_copy, "Pseudomonas putida"))
    perturbations += [
        BacterialDeletionPerturbation(
            systematic_gene_name=gene,
            perturbed_gene_name=gene,
            gene_namespace=KT2440_NAMESPACE,
        )
        for gene in deleted
    ]
    return Genotype(perturbations=perturbations)


def _culture_environment() -> CultureEnvironment:
    return CultureEnvironment(
        media=M9_NREL_CARRUTHERS2025,
        temperature=Temperature(value=24.0),
        duration_hours=48.0,
    )


def _titer_phenotype(titer: float) -> ProductTiterPhenotype:
    return ProductTiterPhenotype(
        product=Compound(name="isoprenol"),
        titer=titer,
        titer_unit=ConcentrationUnit.ug_per_ml,
        titer_uncertainty=3.0,
        titer_uncertainty_type=UncertaintyType.sample_sd,
        titer_se=1.5,
        n_samples=4,
        sample_unit=SampleUnit.biological_replicate,
    )


def _titer_record(
    deleted: list[str], titer: float = 200.0, *, native_copy: str | None = None
) -> Record:
    return {
        "experiment": ProductTiterExperiment(
            dataset_name="titer",
            genotype=_kt2440_genotype(deleted, native_copy=native_copy),
            environment=_culture_environment(),
            phenotype=_titer_phenotype(titer),
        ).model_dump(),
        "reference": ProductTiterExperimentReference(
            dataset_name="titer",
            genome_reference=KT2440_REFERENCE_MODEL,
            environment_reference=_culture_environment(),
            phenotype_reference=_titer_phenotype(100.0),
        ).model_dump(),
        "publication": _publication(),
    }


def _bacterial_protein_record(deleted: list[str], proteins: list[str]) -> Record:
    phenotype = ProteinAbundancePhenotype(
        protein_abundance={tag: 10.0 for tag in proteins},
        n_replicates={tag: 3 for tag in proteins},
        measurement_type="dia_nn_top3_peptide_signal_mean",
    )
    return {
        "experiment": BacterialProteinAbundanceExperiment(
            dataset_name="proteome",
            genotype=_kt2440_genotype(deleted),
            environment=_culture_environment(),
            phenotype=phenotype,
        ).model_dump(),
        "reference": BacterialProteinAbundanceExperimentReference(
            dataset_name="proteome",
            genome_reference=KT2440_REFERENCE_MODEL,
            environment_reference=_culture_environment(),
            phenotype_reference=phenotype,
        ).model_dump(),
        "publication": _publication(),
    }


def _fake_family_verify(
    seen: list[tuple[str, str]], *, passing: bool = True
) -> Callable[[str, str], VerificationReport]:
    """Stand in for a loader's entry point: records the call, returns one L0 row."""

    def verify(dataset_root: str, data_root: str) -> VerificationReport:
        seen.append((dataset_root, data_root))
        return VerificationReport(dataset_name="fake", provenance=PROV).add(
            LevelResult(
                level=Level.L0, name="structural", passed=passing, message="fake gate"
            )
        )

    return verify


def _stub_bacterial_universe(
    monkeypatch: pytest.MonkeyPatch, genes: set[str]
) -> list[str]:
    """Replace ``_bacterial_gene_set`` with a stub universe; return its call log."""
    calls: list[str] = []

    def fake(assembly: Any, data_root: str) -> set[str]:
        calls.append(assembly.assembly_set)
        return set(genes)

    monkeypatch.setattr(runners, "_bacterial_gene_set", fake)
    return calls


def test_host_perturbed_gene_set_keeps_only_the_records_own_host_loci() -> None:
    """A heterologous token is not a host locus; a native extra copy is."""
    records = [
        _titer_record(["PP_5003"]),
        _titer_record(["PP_3540"], native_copy="PP_4042"),
    ]
    assert runners.host_perturbed_gene_set(records) == {"PP_5003", "PP_3540", "PP_4042"}


def test_bacterial_protein_locus_set_unions_quantified_and_perturbed_loci() -> None:
    """Both are identifiers the record claims are loci of its own pin."""
    records = [_bacterial_protein_record(["PP_0815"], ["PP_0001", "PP_0002"])]
    assert runners.bacterial_protein_locus_set(records) == {
        "PP_0815",
        "PP_0001",
        "PP_0002",
    }


def test_run_product_titer_appends_the_host_containment_to_each_loader_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The loader's own gate, then the family's L4, then the report on disk."""
    _write_lmdb(_root(tmp_path, "titer_a"), [_titer_record(["PP_5003"])])
    _write_lmdb(_root(tmp_path, "titer_b"), [_titer_record(["PP_3540"], 210.0)])
    seen: list[tuple[str, str]] = []
    monkeypatch.setattr(
        runners,
        "PRODUCT_TITER_DATASETS",
        {
            "titer_a": {
                "root": "data/torchcell/titer_a",
                "verify": _fake_family_verify(seen),
            },
            "titer_b": {
                "root": "data/torchcell/titer_b",
                "verify": _fake_family_verify(seen),
            },
        },
    )
    universe = _stub_bacterial_universe(monkeypatch, {"PP_5003", "PP_3540"})
    assert runners.run_product_titer(str(tmp_path)) is True
    assert seen == [
        (str(_root(tmp_path, "titer_a")), str(tmp_path)),
        (str(_root(tmp_path, "titer_b")), str(tmp_path)),
    ]
    assert universe == [PPUTIDA_KT2440, PPUTIDA_KT2440]
    report = _read_report(_root(tmp_path, "titer_a"))
    assert _names(report) == ["structural", "perturbed_gene_containment_assembly"]
    containment = _result(report, "perturbed_gene_containment_assembly")
    assert containment["level"] == Level.L4
    assert containment["passed"] is True
    assert containment["details"]["assembly_sets"] == [PPUTIDA_KT2440]
    assert containment["message"] == (
        "1.000 of 1 measured genes are loci of pputida_KT2440_ASM756v2 (>= 1.0)"
    )


def test_run_product_titer_refuses_a_perturbed_locus_off_the_pin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Containment is 1.0, not a floor with headroom: one stray tag fails the family."""
    _write_lmdb(_root(tmp_path, "titer_a"), [_titer_record(["PP_5003", "PP_9999"])])
    monkeypatch.setattr(
        runners,
        "PRODUCT_TITER_DATASETS",
        {
            "titer_a": {
                "root": "data/torchcell/titer_a",
                "verify": _fake_family_verify([]),
            }
        },
    )
    _stub_bacterial_universe(monkeypatch, {"PP_5003"})
    assert runners.run_product_titer(str(tmp_path)) is False
    containment = _result(
        _read_report(_root(tmp_path, "titer_a")), "perturbed_gene_containment_assembly"
    )
    assert containment["passed"] is False
    assert containment["details"]["missing_examples"] == ["PP_9999"]


def test_run_product_titer_fails_when_a_loader_gate_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failing dataset gate fails the family even with the L4 containment passing."""
    _write_lmdb(_root(tmp_path, "titer_a"), [_titer_record(["PP_5003"])])
    monkeypatch.setattr(
        runners,
        "PRODUCT_TITER_DATASETS",
        {
            "titer_a": {
                "root": "data/torchcell/titer_a",
                "verify": _fake_family_verify([], passing=False),
            }
        },
    )
    _stub_bacterial_universe(monkeypatch, {"PP_5003"})
    assert runners.run_product_titer(str(tmp_path)) is False


def test_run_bacterial_protein_abundance_names_its_own_l4_over_the_union(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The quantified proteins and the perturbed genes are checked as one set."""
    _write_lmdb(
        _root(tmp_path, "prot_bact"),
        [_bacterial_protein_record(["PP_0815"], ["PP_0001", "PP_0002"])],
    )
    monkeypatch.setattr(
        runners,
        "BACTERIAL_PROTEIN_ABUNDANCE_DATASETS",
        {
            "prot_bact": {
                "root": "data/torchcell/prot_bact",
                "verify": _fake_family_verify([]),
            }
        },
    )
    _stub_bacterial_universe(monkeypatch, {"PP_0815", "PP_0001", "PP_0002"})
    assert runners.run_bacterial_protein_abundance(str(tmp_path)) is True
    report = _read_report(_root(tmp_path, "prot_bact"))
    assert _names(report) == [
        "structural",
        "protein_and_perturbed_locus_containment_assembly",
    ]
    containment = _result(report, "protein_and_perturbed_locus_containment_assembly")
    assert containment["details"]["n_measured"] == 3
    assert containment["passed"] is True


def test_each_bioproduction_adapter_calls_its_own_loader_entry_point(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The registry's indirection is one call per dataset, with its family selected."""
    from torchcell.datasets.ecoli import caglar2017, foo2014
    from torchcell.datasets.pputida import (
        banerjee2025,
        carruthers2025,
        desiqueira2025,
        kang2026,
        lim2025,
    )

    calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def record(label: str) -> Callable[..., VerificationReport]:
        def fake(*args: Any, **kwargs: Any) -> VerificationReport:
            calls.append((label, args, kwargs))
            return VerificationReport(dataset_name=label, provenance=PROV)

        return fake

    monkeypatch.setattr(banerjee2025, "verify_build", record("banerjee"))
    monkeypatch.setattr(foo2014, "verify_build", record("foo"))
    monkeypatch.setattr(carruthers2025, "verify_build", record("carruthers"))
    monkeypatch.setattr(desiqueira2025, "verify_build", record("desiqueira"))
    monkeypatch.setattr(kang2026, "verify_build", record("kang"))
    monkeypatch.setattr(lim2025, "run_proteome_verification", record("lim"))
    monkeypatch.setattr(caglar2017, "run_verification", record("caglar"))

    assert runners._verify_foo_titer("/root", "/data").dataset_name == "foo"
    assert (
        runners._verify_banerjee_proteome("/root", "/data").dataset_name == "banerjee"
    )
    assert (
        runners._verify_carruthers_titer("/root", "/data").dataset_name == "carruthers"
    )
    assert (
        runners._verify_carruthers_proteome("/root", "/data").dataset_name
        == "carruthers"
    )
    assert (
        runners._verify_desiqueira_titer("/root", "/data").dataset_name == "desiqueira"
    )
    assert (
        runners._verify_desiqueira_proteome("/root", "/data").dataset_name
        == "desiqueira"
    )
    assert runners._verify_kang_titer("/root", "/data").dataset_name == "kang"
    assert runners._verify_lim_proteome("/root", "/data").dataset_name == "lim"
    assert runners._verify_caglar_proteome("/root", "/data").dataset_name == "caglar"
    assert calls == [
        ("foo", ("/root", "/data"), {}),
    ),
    _case(
        ("banerjee", ("/root", "/data"), {}),
        ("carruthers", ("/root", "/data"), {"family": "titer"}),
        ("carruthers", ("/root", "/data"), {"family": "proteome"}),
        ("desiqueira", ("/root", "/data"), {"family": "titer"}),
        ("desiqueira", ("/root", "/data"), {"family": "proteome"}),
        ("kang", ("/root", "/data"), {}),
        ("lim", ("/data",), {}),
        ("caglar", ("proteome", "/data"), {}),
    ]


def test_the_bioproduction_registries_name_every_landed_store() -> None:
    """The two registries are the list of landed datasets of each family."""
    assert {
        name: spec["root"] for name, spec in runners.PRODUCT_TITER_DATASETS.items()
    } == {
        "isopentenol_titer_foo2014": "data/torchcell/isopentenol_titer_foo2014",
        "isoprenol_titer_carruthers2025": (
            "data/torchcell/isoprenol_titer_carruthers2025"
        ),
        "isoprenol_titer_desiqueira2025": (
            "data/torchcell/isoprenol_titer_desiqueira2025"
        ),
        "isoprenyl_acetate_titer_kang2026": (
            "data/torchcell/isoprenyl_acetate_titer_kang2026"
        ),
    }
    assert {
        name: spec["root"]
        for name, spec in runners.BACTERIAL_PROTEIN_ABUNDANCE_DATASETS.items()
    } == {
        "proteome_banerjee2025": "data/torchcell/proteome_banerjee2025",
        "proteome_carruthers2025": "data/torchcell/proteome_carruthers2025",
        "proteome_desiqueira2025": "data/torchcell/proteome_desiqueira2025",
        "proteome_percent_desiqueira2025": (
            "data/torchcell/proteome_percent_desiqueira2025"
        ),
        "proteome_log10_percent_desiqueira2025": (
            "data/torchcell/proteome_log10_percent_desiqueira2025"
        ),
        "proteome_lim2025": "data/torchcell/proteome_lim2025",
        "proteome_caglar2017": "data/torchcell/proteome_caglar2017",
    }
