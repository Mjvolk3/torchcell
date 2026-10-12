# tests/torchcell/datasets/test_bacteria_common.py
# [[tests.torchcell.datasets.test_bacteria_common]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/test_bacteria_common.py
"""The shared bacterial loader skeleton (``torchcell.datasets.bacteria_common``).

Synthetic tests run everywhere. The reconciliation tests read the real
``EcoliK12MG1655Genome`` class over the synthetic MG1655 assembly of
``tests/torchcell/sequence/genome/_bacterial_fixtures.py`` (seven loci, b0001 to b0007),
served through a stubbed ``resolve`` with every network entry point raising, and built
into a fresh ``tmp_path`` root with ``overwrite=False``. The genome-construction and
injection tests route every genome class to a recording fake
(``_genome_injection_fakes``), so nothing is built. The assembly-report tests serve a
written report through a stubbed ``resolve``.

The tier tests (``@pytest.mark.data``, skipped unless the bacterial sets and the built
default caches exist under ``$DATA_ROOT``) reopen the default ``data.db`` caches
read-only through ``bacterial_genome`` and pin the deposited assembly reports and the
measured ECK crosswalk (4,423 one-to-one pairs, 11 numeric disagreements).

Derived expectations for the synthetic frame (nine distinct names, ``thrA`` twice): b0001
is a current gene; ``thrA``, ``thrW``, ``yaaP`` and ``proc`` resolve at the symbol layer
(``yaaP`` to the pseudogene b0004, ``proc`` only case-insensitively to proC, b0006);
``ECK0005`` and ``ECK0003`` at the synonym layer; ``PRO2`` matches the synonyms ``pro2``
(b0005) and ``Pro2`` (b0006) case-insensitively and is ambiguous; ``b0099`` is retired.
``thrW`` and ``ECK0003`` both reach b0003, so both are kept as given. Resolved = 1
current + 5 renamed + 1 pseudogene = 7 of 9.
"""

from __future__ import annotations

import gzip
import hashlib
import os
import os.path as osp
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from pydantic import ValidationError

import torchcell.datamodels.schema as schema
import torchcell.datasets.bacteria_common as bc
from tests.torchcell.datasets._genome_injection_fakes import (
    BacterialLoaderNamingYeastGenome,
    EcoliBW25113Loader,
    EcoliMG1655Loader,
    EcoliREL606Loader,
    FakeBW25113Genome,
    FakeKT2440Genome,
    FakeMG1655Genome,
    FakeREL606Genome,
    PputidaLoader,
    YeastLoader,
    install_bacterial_fakes,
)
from tests.torchcell.sequence.genome._bacterial_fixtures import (
    MG1655_GAF,
    MG1655_LOCI,
    forbid_network,
    serve_tier,
    write_assembly,
)
from torchcell.datamodels.schema import (
    BacterialAssemblySet,
    BacterialReferenceStrain,
    BacterialStrainBackground,
)
from torchcell.sequence.genome.bacterial import GenomeAnnotationMismatchError
from torchcell.sequence.genome.base import GeneNameResolution, GeneNameStatus
from torchcell.sequence.genome.ecoli.k12 import (
    BW25113_ASSEMBLY,
    MG1655_ASSEMBLY,
    EcoliK12BW25113Genome,
    EcoliK12Genome,
    EcoliK12MG1655Genome,
)
from torchcell.sequence.genome.ecoli.rel606 import EcoliBREL606Genome
from torchcell.sequence.genome.pputida.kt2440 import PPutidaKT2440Genome
from torchcell.sequence.genome.registry import (
    ECOLI_B_REL606,
    ECOLI_K12_BW25113,
    ECOLI_K12_MG1655,
    GO_RELEASE_20260805,
    PPUTIDA_KT2440,
)
from torchcell.verification.sourced import ProvenanceGap, ProvenanceGapReason

S = GeneNameStatus


# --------------------------------------------------------------------------- #
# Vocabulary: imported from the schema, one namespace per strain
# --------------------------------------------------------------------------- #
def test_the_namespace_vocabulary_is_the_schema_s_own_objects() -> None:
    """The patterns and the Literal are re-exported, not restated: same objects."""
    assert bc.BACTERIAL_LOCUS_TAG_PATTERNS is schema.BACTERIAL_LOCUS_TAG_PATTERNS
    assert bc.BACTERIAL_LOCUS_TAG_PATTERN is schema.BACTERIAL_LOCUS_TAG_PATTERN
    assert bc.BacterialGeneNamespace is schema.BacterialGeneNamespace
    assert bc.BACTERIAL_ASSEMBLY_SETS is schema.BACTERIAL_ASSEMBLY_SETS
    assert {ns: p.pattern for ns, p in bc.LOCUS_TAG_PATTERNS.items()} == (
        schema.BACTERIAL_LOCUS_TAG_PATTERNS
    )


def test_each_strain_has_its_own_namespace_host_and_genome_class() -> None:
    """Four strains, four namespaces, two hosts; each genome class reads its set."""
    assert bc.STRAIN_GENE_NAMESPACES == {
        "MG1655": "ecoli_k12_mg1655_bnumber",
        "BW25113": "ecoli_k12_bw25113_locus_tag",
        "KT2440": "pputida_kt2440_locus_tag",
        "REL606": "ecoli_b_rel606_locus_tag",
    }
    assert set(bc.STRAIN_GENE_NAMESPACES.values()) == set(
        schema.BACTERIAL_LOCUS_TAG_PATTERNS
    )
    assert bc.HOST_STRAINS == {
        "ecoli": ("MG1655", "BW25113", "REL606"),
        "pputida": ("KT2440",),
    }
    assert {s: bc.host_of_strain(s) for s in bc.STRAIN_GENE_NAMESPACES} == {
        "MG1655": "ecoli",
        "BW25113": "ecoli",
        "KT2440": "pputida",
        "REL606": "ecoli",
    }
    assert {
        strain: cls.ASSEMBLY.assembly_set
        for strain, cls in bc.BACTERIAL_GENOME_CLASSES.items()
    } == {
        "MG1655": ECOLI_K12_MG1655,
        "BW25113": ECOLI_K12_BW25113,
        "KT2440": PPUTIDA_KT2440,
        "REL606": ECOLI_B_REL606,
    }
    assert bc.BACTERIAL_GENOME_CLASSES == {
        "MG1655": EcoliK12MG1655Genome,
        "BW25113": EcoliK12BW25113Genome,
        "KT2440": PPutidaKT2440Genome,
        "REL606": EcoliBREL606Genome,
    }


def test_each_compiled_pattern_matches_its_own_strain_s_tags_only() -> None:
    """A b-number, a ``BW25113_`` tag and a ``PP_`` tag each belong to exactly one."""
    samples = {
        "ecoli_k12_mg1655_bnumber": "b0002",
        "ecoli_k12_bw25113_locus_tag": "BW25113_0002",
        "pputida_kt2440_locus_tag": "PP_16SA",
        "ecoli_b_rel606_locus_tag": "ECB_t00001",
    }
    for owner, tag in samples.items():
        assert {ns for ns, p in bc.LOCUS_TAG_PATTERNS.items() if p.match(tag)} == {
            owner
        }
    assert not any(p.match("YAL001C") for p in bc.LOCUS_TAG_PATTERNS.values())


def test_strain_of_assembly_set_inverts_the_schema_map_and_refuses_others() -> None:
    assert bc.strain_of_assembly_set(ECOLI_K12_BW25113) == "BW25113"
    assert bc.strain_of_assembly_set(PPUTIDA_KT2440) == "KT2440"
    assert bc.strain_of_assembly_set(ECOLI_B_REL606) == "REL606"
    with pytest.raises(
        KeyError, match="'sgd_S288C_R64-4-1_20230830' is not a bacterial"
    ):
        bc.strain_of_assembly_set("sgd_S288C_R64-4-1_20230830")


# --------------------------------------------------------------------------- #
# bacterial_genome: the right class on the default cache root, read-only
# --------------------------------------------------------------------------- #
def test_bacterial_genome_opens_each_strain_on_its_default_root_without_overwrite(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    log = install_bacterial_fakes(monkeypatch)
    mg = bc.bacterial_genome("ecoli", "MG1655", "/dr")
    bw = bc.bacterial_genome("ecoli", "BW25113", "/dr")
    kt = bc.bacterial_genome("pputida", "KT2440", "/dr")
    assert (type(mg), type(bw), type(kt)) == (
        FakeMG1655Genome,
        FakeBW25113Genome,
        FakeKT2440Genome,
    )
    assert isinstance(mg, EcoliK12Genome) and isinstance(bw, EcoliK12Genome)
    assert isinstance(kt, PPutidaKT2440Genome)
    assert log == [
        (
            "FakeMG1655Genome",
            {"genome_root": "/dr/data/ecoli/mg1655/genome", "overwrite": False},
        ),
        (
            "FakeBW25113Genome",
            {"genome_root": "/dr/data/ecoli/bw25113/genome", "overwrite": False},
        ),
        (
            "FakeKT2440Genome",
            {"genome_root": "/dr/data/pputida/kt2440/genome", "overwrite": False},
        ),
    ]


def test_bacterial_genome_reads_data_root_from_the_environment_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    log = install_bacterial_fakes(monkeypatch)
    monkeypatch.setenv("DATA_ROOT", "/from_env")
    assert type(bc.bacterial_genome("ecoli", "BW25113")) is FakeBW25113Genome
    assert log == [
        (
            "FakeBW25113Genome",
            {"genome_root": "/from_env/data/ecoli/bw25113/genome", "overwrite": False},
        )
    ]


def test_bacterial_genome_refuses_a_strain_of_the_other_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    log = install_bacterial_fakes(monkeypatch)
    with pytest.raises(ValueError, match="'KT2440' is not a ecoli strain"):
        bc.bacterial_genome("ecoli", "KT2440", "/dr")
    with pytest.raises(ValueError, match="'MG1655' is not a pputida strain"):
        bc.bacterial_genome("pputida", "MG1655", "/dr")
    assert log == []


def test_bacterial_genome_opens_rel606_as_an_ecoli_b_genome_not_a_k12_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REL606 is an E. coli strain whose genome is the B class, on its own root."""
    log = install_bacterial_fakes(monkeypatch)
    genome = bc.bacterial_genome("ecoli", "REL606", "/dr")
    assert type(genome) is FakeREL606Genome
    assert isinstance(genome, EcoliBREL606Genome)
    assert not isinstance(genome, EcoliK12Genome)
    assert log == [
        (
            "FakeREL606Genome",
            {"genome_root": "/dr/data/ecoli/rel606/genome", "overwrite": False},
        )
    ]
    with pytest.raises(ValueError, match="'REL606' is not a pputida strain"):
        bc.bacterial_genome("pputida", "REL606", "/dr")


# --------------------------------------------------------------------------- #
# reconcile_locus_tags on the synthetic MG1655 genome
# --------------------------------------------------------------------------- #
@pytest.fixture
def synthetic_mg1655(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> EcoliK12MG1655Genome:
    """The real MG1655 class over the synthetic assembly; network refused."""
    files = write_assembly(tmp_path / "tier", MG1655_ASSEMBLY, MG1655_LOCI, MG1655_GAF)
    forbid_network(monkeypatch)
    serve_tier(monkeypatch, files)
    root = tmp_path / "mg1655"
    root.mkdir()
    return EcoliK12MG1655Genome(genome_root=str(root), overwrite=False)


SYNTHETIC_NAMES = [
    "b0001",
    "thrA",
    "ECK0005",
    "thrW",
    "ECK0003",
    "b0099",
    "yaaP",
    "proc",
    "PRO2",
    "thrA",
]


def test_reconcile_remaps_resolved_names_and_keeps_the_rest_as_given(
    synthetic_mg1655: EcoliK12MG1655Genome,
) -> None:
    names = pd.Series(SYNTHETIC_NAMES, index=range(10, 20))
    stored, report = bc.reconcile_locus_tags(synthetic_mg1655, names, label="synthetic")
    assert stored.index.tolist() == list(range(10, 20))
    assert stored.tolist() == [
        "b0001",  # current locus tag
        "b0002",  # symbol thrA
        "b0005",  # ECK synonym
        "thrW",  # collision on b0003: kept as given
        "ECK0003",  # collision on b0003: kept as given
        "b0099",  # retired: kept as given
        "b0004",  # symbol of the pseudogene b0004
        "b0006",  # proC, case-insensitive
        "PRO2",  # ambiguous: kept as given
        "b0002",  # the repeated name maps the same way
    ]
    assert report.model_dump() == {
        "label": "synthetic",
        "assembly_set": ECOLI_K12_MG1655,
        "gene_namespace": "ecoli_k12_mg1655_bnumber",
        "unique_names": 9,
        "status_histogram": {
            S.CURRENT: 1,
            S.RENAMED: 5,
            S.NON_GENE_FEATURE: 1,
            S.RETIRED: 1,
            S.AMBIGUOUS: 1,
        },
        "layer_histogram": {
            "locus tag": 1,
            "old locus tag": 0,
            "RefSeq locus tag": 0,
            "gene symbol": 4,
            "gene synonym": 3,
            "not found": 1,
        },
        "remapped": 4,
        "kept_on_collision": ("ECK0003", "thrW"),
        "retired_kept": ("b0099",),
        "ambiguous_kept": {"PRO2": ("b0005", "b0006")},
        "case_insensitive": ("PRO2", "proc"),
        "outside_namespace": ("ECK0003", "PRO2", "thrW"),
    }
    assert (report.resolved, report.resolved_fraction) == (7, 7 / 9)


def test_reconcile_threshold_stops_the_dataset_instead_of_dropping(
    synthetic_mg1655: EcoliK12MG1655Genome,
) -> None:
    """7 of 9 = 0.778 passes a 0.75 floor and is refused at 0.8, with the histogram."""
    _, report = bc.reconcile_locus_tags(
        synthetic_mg1655, pd.Series(SYNTHETIC_NAMES), label="synthetic"
    )
    report.require_resolved(0.75)
    with pytest.raises(bc.LocusTagResolutionError) as excinfo:
        report.require_resolved(0.8)
    assert str(excinfo.value) == (
        "synthetic: 7 of 9 names (0.778) resolve to ecoli_K12_MG1655_ASM584v2 locus "
        "tags, below 0.8; statuses {'current': 1, 'renamed': 5, 'non_gene_feature': 1, "
        "'retired': 1, 'ambiguous': 1}"
    )


def test_reconcile_refuses_an_empty_series(
    synthetic_mg1655: EcoliK12MG1655Genome,
) -> None:
    with pytest.raises(ValueError, match="empty_set: no names to reconcile"):
        bc.reconcile_locus_tags(
            synthetic_mg1655, pd.Series([], dtype=object), label="empty_set"
        )


def test_resolution_layer_refuses_a_note_of_no_known_form(
    synthetic_mg1655: EcoliK12MG1655Genome,
) -> None:
    odd = GeneNameResolution(
        input_name="x", status=S.RENAMED, systematic_name="b0001", note="by telepathy"
    )
    with pytest.raises(
        ValueError, match="'x': resolution note 'by telepathy' names no"
    ):
        bc.resolution_layer(synthetic_mg1655, odd)
    pseudo = synthetic_mg1655.resolve_gene_name("b0004")
    assert (pseudo.status, bc.resolution_layer(synthetic_mg1655, pseudo)) == (
        S.NON_GENE_FEATURE,
        "locus tag",
    )


# --------------------------------------------------------------------------- #
# The assembly pin from the deposited report
# --------------------------------------------------------------------------- #
REPORT = """# Assembly name:  ASM584v2
# Organism name:  Escherichia coli str. K-12 substr. MG1655 (E. coli)
# Infraspecific name:  strain=K-12 substr. MG1655
# Taxid:          511145
# GenBank assembly accession: GCA_000005845.2
# RefSeq assembly accession: GCF_000005845.2
# RefSeq assembly and GenBank assemblies identical: yes
#
## Assembly-Units:
U00096.3\tassembled-molecule\tna\tChromosome\tU00096.3\t=\tNC_000913.3
"""
REPORT_MEMBER = "GCA_000005845.2_ASM584v2_assembly_report.txt"


def _serve_report(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, text: str
) -> list[tuple[str, str, str | None]]:
    path = tmp_path / REPORT_MEMBER
    path.write_text(text)
    calls: list[tuple[str, str, str | None]] = []

    def fake_resolve(
        assembly_set: str, filename: str, *, data_root: str | None = None
    ) -> str:
        calls.append((assembly_set, filename, data_root))
        return str(path)

    monkeypatch.setattr(bc, "resolve", fake_resolve)
    return calls


def test_read_assembly_report_parses_the_header_of_the_deposited_member(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = _serve_report(monkeypatch, tmp_path, REPORT)
    report = bc.read_assembly_report("MG1655", "/dr")
    assert calls == [(ECOLI_K12_MG1655, REPORT_MEMBER, "/dr")]
    assert report.model_dump() == {
        "assembly_set": ECOLI_K12_MG1655,
        "member": REPORT_MEMBER,
        "sha256": hashlib.sha256(REPORT.encode()).hexdigest(),
        "assembly_name": "ASM584v2",
        "organism_name": "Escherichia coli str. K-12 substr. MG1655 (E. coli)",
        "infraspecific_name": "strain=K-12 substr. MG1655",
        "taxid": 511145,
        "genbank_accession": "GCA_000005845.2",
        "refseq_accession": "GCF_000005845.2",
    }


@pytest.mark.parametrize(
    "text,error,message",
    [
        (
            REPORT.replace("GCA_000005845.2", "GCA_000005845.3"),
            GenomeAnnotationMismatchError,
            "names ('GCA_000005845.3_ASM584v2', 'GCF_000005845.2_ASM584v2'), but the "
            "MG1655 genome reads",
        ),
        (
            REPORT.replace("# Taxid:", "# Taxid: 1\n# Taxid:"),
            ValueError,
            "header key 'Taxid' is repeated",
        ),
        (
            REPORT.replace("# Taxid:", "# no colon here\n# Taxid:"),
            ValueError,
            "unexpected header line '# no colon here'",
        ),
    ],
    ids=["foreign-accession", "repeated-key", "line-without-colon"],
)
def test_read_assembly_report_refuses_a_report_it_cannot_trust(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    text: str,
    error: type[Exception],
    message: str,
) -> None:
    _serve_report(monkeypatch, tmp_path, text)
    with pytest.raises(error, match=message.replace("(", r"\(").replace(")", r"\)")):
        bc.read_assembly_report("MG1655")


def test_assembly_reference_pins_the_genbank_accession_of_the_report(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _serve_report(monkeypatch, tmp_path, REPORT)
    reference = bc.assembly_reference("MG1655")
    assert reference.model_dump() == {
        "species": "Escherichia coli",
        "strain": "MG1655",
        "ploidy": "haploid",
        "assembly_set": ECOLI_K12_MG1655,
        "assembly_accession": "GCA_000005845.2",
        "background": None,
    }


REL606_REPORT = """# Assembly name:  ASM1798v1
# Organism name:  Escherichia coli B str. REL606 (E. coli)
# Infraspecific name:  strain=REL606
# Taxid:          413997
# GenBank assembly accession: GCA_000017985.1
# RefSeq assembly accession: GCF_000017985.1
# RefSeq assembly and GenBank assemblies identical: yes
#
## Assembly-Units:
CP000819.1\tassembled-molecule\tna\tChromosome\tCP000819.1\t=\tNC_012967.1
"""


def test_assembly_reference_pins_rel606_to_its_genbank_accession(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The REL606 report is read from its own set and the record pins GCA_000017985.1."""
    calls = _serve_report(monkeypatch, tmp_path, REL606_REPORT)
    reference = bc.assembly_reference("REL606", data_root="/dr")
    assert calls == [
        (ECOLI_B_REL606, "GCA_000017985.1_ASM1798v1_assembly_report.txt", "/dr")
    ]
    assert reference.model_dump() == {
        "species": "Escherichia coli",
        "strain": "REL606",
        "ploidy": "haploid",
        "assembly_set": ECOLI_B_REL606,
        "assembly_accession": "GCA_000017985.1",
        "background": None,
    }


_SETS: dict[BacterialReferenceStrain, BacterialAssemblySet] = {
    "MG1655": "ecoli_K12_MG1655_ASM584v2",
    "BW25113": "ecoli_K12_BW25113_ASM75055v1",
    "KT2440": "pputida_KT2440_ASM756v2",
    "REL606": "ecoli_B_REL606_ASM1798v1",
}


def _background(
    name: str, strain: BacterialReferenceStrain
) -> BacterialStrainBackground:
    return BacterialStrainBackground(
        name=name,
        reference_strain=strain,
        assembly_set=_SETS[strain],
        provenance_gaps=[
            ProvenanceGap(
                field="provenance", reason=ProvenanceGapReason.not_carried_by_curation
            )
        ],
    )


def test_assembly_reference_takes_the_strain_name_from_a_background(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A background of MG1655 names the record's strain; one of BW25113 is refused."""
    _serve_report(monkeypatch, tmp_path, REPORT)
    chassis = _background("MG1655 delta-lacZ", "MG1655")
    reference = bc.assembly_reference("MG1655", background=chassis)
    assert (reference.strain, reference.background) == ("MG1655 delta-lacZ", chassis)
    keio = _background("BW25113", "BW25113")
    with pytest.raises(
        ValueError, match="'BW25113' is an edit of 'BW25113', not 'MG1655'"
    ):
        bc.assembly_reference("MG1655", background=keio)


# --------------------------------------------------------------------------- #
# The host-aware injector (the rule the three build entry points share)
# --------------------------------------------------------------------------- #
def test_injector_hands_each_loader_its_own_host_genome_built_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MG1655 and KT2440 loaders get their genomes; a second MG1655 loader shares the
    first one's; the yeast loader gets no bacterial keyword.
    """
    log = install_bacterial_fakes(monkeypatch)
    injector = bc.BacterialGenomeInjector("/dr")
    assert injector.genome_kwargs(YeastLoader) == {}
    assert log == []
    first = injector.genome_kwargs(EcoliMG1655Loader)
    second = injector.genome_kwargs(EcoliMG1655Loader)
    putida = injector.genome_kwargs(PputidaLoader)
    assert set(first) == {"ecoli_genome"} and set(putida) == {"pputida_genome"}
    assert isinstance(first["ecoli_genome"], EcoliK12MG1655Genome)
    assert isinstance(putida["pputida_genome"], PPutidaKT2440Genome)
    assert second["ecoli_genome"] is first["ecoli_genome"]
    assert [name for name, _ in log] == ["FakeMG1655Genome", "FakeKT2440Genome"]


def test_injector_hands_a_rel606_loader_the_b_genome_under_ecoli_genome(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A REL606 loader names ``ecoli_genome`` and gets the REL606 genome, never a K-12
    one; an MG1655 loader in the same build gets its own.
    """
    log = install_bacterial_fakes(monkeypatch)
    injector = bc.BacterialGenomeInjector("/dr")
    rel606 = injector.genome_kwargs(EcoliREL606Loader)
    mg1655 = injector.genome_kwargs(EcoliMG1655Loader)
    assert isinstance(rel606["ecoli_genome"], EcoliBREL606Genome)
    assert isinstance(mg1655["ecoli_genome"], EcoliK12MG1655Genome)
    assert bc.declared_reference_strain(EcoliREL606Loader) == "REL606"
    assert [name for name, _ in log] == ["FakeREL606Genome", "FakeMG1655Genome"]


class _BothHosts:
    REFERENCE_STRAIN = "MG1655"

    def __init__(
        self, root: str = "r", ecoli_genome: Any = None, pputida_genome: Any = None
    ) -> None:
        pass


class _NoStrain:
    def __init__(self, root: str = "r", ecoli_genome: Any = None) -> None:
        pass


class _WrongHost:
    REFERENCE_STRAIN = "KT2440"

    def __init__(self, root: str = "r", ecoli_genome: Any = None) -> None:
        pass


class _UnknownStrain:
    REFERENCE_STRAIN = "K-12"

    def __init__(self, root: str = "r", ecoli_genome: Any = None) -> None:
        pass


@pytest.mark.parametrize(
    "loader,error,message",
    [
        (
            BacterialLoaderNamingYeastGenome,
            TypeError,
            "BacterialLoaderNamingYeastGenome is a bacterial loader that names 'genome'",
        ),
        (
            _BothHosts,
            TypeError,
            "_BothHosts names \\['ecoli_genome', 'pputida_genome'\\]",
        ),
        (
            _NoStrain,
            TypeError,
            "_NoStrain declares a bacterial genome parameter but no REFERENCE_STRAIN",
        ),
        (_WrongHost, TypeError, "REFERENCE_STRAIN 'KT2440' is not a ecoli strain"),
        (
            _UnknownStrain,
            ValidationError,
            "Input should be 'MG1655', 'BW25113', 'KT2440' or 'REL606'",
        ),
    ],
    ids=[
        "names-yeast-genome",
        "both-hosts",
        "no-strain",
        "wrong-host",
        "unknown-strain",
    ],
)
def test_injector_refuses_a_misdeclared_loader_before_building_anything(
    monkeypatch: pytest.MonkeyPatch, loader: type, error: type[Exception], message: str
) -> None:
    log = install_bacterial_fakes(monkeypatch)
    with pytest.raises(error, match=message):
        bc.BacterialGenomeInjector("/dr").genome_kwargs(loader)
    assert log == []


def test_declared_reference_strain_reads_the_class_attribute() -> None:
    assert bc.declared_reference_strain(EcoliBW25113Loader) == "BW25113"
    assert bc.declared_reference_strain(PputidaLoader) == "KT2440"


# --------------------------------------------------------------------------- #
# The deposited tier and the built default caches (data-gated)
# --------------------------------------------------------------------------- #
DATA_ROOT = os.environ.get("DATA_ROOT", "")
TIER_PRESENT = all(
    osp.isfile(osp.join(DATA_ROOT, "torchcell-genomes", s, "manifest.json"))
    for s in (ECOLI_K12_MG1655, ECOLI_K12_BW25113, PPUTIDA_KT2440, GO_RELEASE_20260805)
) and all(
    osp.isfile(osp.join(DATA_ROOT, cls.ASSEMBLY.default_genome_root, "data.db"))
    for cls in (EcoliK12MG1655Genome, EcoliK12BW25113Genome)
)
on_tier = [
    pytest.mark.data,
    pytest.mark.skipif(
        not TIER_PRESENT,
        reason="requires the bacterial assembly sets and the built E. coli data.db "
        "caches under $DATA_ROOT",
    ),
]


@pytest.fixture(scope="module")
def tier_mg1655() -> EcoliK12Genome:
    """MG1655 reopened read-only from its default cache root."""
    return bc.bacterial_genome("ecoli", "MG1655")


@pytest.mark.parametrize(
    "strain,taxid,genbank,refseq,species",
    [
        ("MG1655", 511145, "GCA_000005845.2", "GCF_000005845.2", "Escherichia coli"),
        ("BW25113", 679895, "GCA_000750555.1", "GCF_000750555.1", "Escherichia coli"),
        ("KT2440", 160488, "GCA_000007565.2", "GCF_000007565.2", "Pseudomonas putida"),
    ],
)
@pytest.mark.data
@pytest.mark.skipif(not TIER_PRESENT, reason="requires the bacterial assembly sets")
def test_tier_assembly_reference_reads_each_deposited_report(
    strain: Any, taxid: int, genbank: str, refseq: str, species: str
) -> None:
    report = bc.read_assembly_report(strain)
    assert (report.taxid, report.genbank_accession, report.refseq_accession) == (
        taxid,
        genbank,
        refseq,
    )
    reference = bc.assembly_reference(strain)
    assert (reference.species, reference.strain, reference.assembly_accession) == (
        species,
        strain,
        genbank,
    )
    assert reference.assembly_set == schema.BACTERIAL_ASSEMBLY_SETS[strain]


class TestTierMG1655:
    """The real MG1655 genome, reopened from ``data/ecoli/mg1655/genome``."""

    pytestmark = on_tier

    def test_bacterial_genome_reopens_the_default_cache(
        self, tier_mg1655: EcoliK12Genome
    ) -> None:
        """The MG1655 class on ``data/ecoli/mg1655/genome``, never overwritten; 4,506
        genes (the 4,651 GenBank loci less 145 pseudogenes).
        """
        assert type(tier_mg1655) is EcoliK12MG1655Genome
        assert tier_mg1655.genome_root == osp.join(
            DATA_ROOT, "data/ecoli/mg1655/genome"
        )
        assert tier_mg1655.overwrite is False
        assert len(tier_mg1655.gene_set) == 4506

    def test_reconcile_on_the_real_annotation(
        self, tier_mg1655: EcoliK12Genome
    ) -> None:
        """b0001 and b0006 are current; thrB (symbol) is b0003 and ECK0004 (synonym)
        b0004; ECK0006 also reaches b0006, so b0006 and ECK0006 are kept as given;
        b9999 is retired.
        """
        names = pd.Series(["b0001", "thrB", "ECK0004", "b9999", "b0006", "ECK0006"])
        stored, report = bc.reconcile_locus_tags(tier_mg1655, names, label="real")
        assert stored.tolist() == [
            "b0001",
            "b0003",
            "b0004",
            "b9999",
            "b0006",
            "ECK0006",
        ]
        assert report.status_histogram == {
            S.CURRENT: 2,
            S.RENAMED: 3,
            S.NON_GENE_FEATURE: 0,
            S.RETIRED: 1,
            S.AMBIGUOUS: 0,
        }
        assert report.layer_histogram == {
            "locus tag": 2,
            "old locus tag": 0,
            "RefSeq locus tag": 0,
            "gene symbol": 1,
            "gene synonym": 2,
            "not found": 1,
        }
        assert (report.kept_on_collision, report.outside_namespace) == (
            ("ECK0006", "b0006"),
            ("ECK0006",),
        )

    def test_eck_crosswalk_of_the_two_default_genomes(
        self, tier_mg1655: EcoliK12Genome
    ) -> None:
        """4,423 one-to-one ECK pairs; 11 of them carry different numerics."""
        bw25113 = bc.bacterial_genome("ecoli", "BW25113")
        assert isinstance(tier_mg1655, EcoliK12MG1655Genome)
        assert isinstance(bw25113, EcoliK12BW25113Genome)
        crosswalk = bc.eck_crosswalk(tier_mg1655, bw25113)
        assert (len(crosswalk.pairs), len(crosswalk.numeric_disagreements)) == (
            4423,
            11,
        )
        assert all(not pair.numerics_agree for pair in crosswalk.numeric_disagreements)


REL606_TIER = osp.isfile(
    osp.join(DATA_ROOT, "torchcell-genomes", ECOLI_B_REL606, "manifest.json")
)
REL606_CACHE = osp.isfile(
    osp.join(DATA_ROOT, EcoliBREL606Genome.ASSEMBLY.default_genome_root, "data.db")
)


@pytest.mark.data
@pytest.mark.skipif(not REL606_TIER, reason="requires the REL606 assembly set")
def test_tier_assembly_reference_reads_the_deposited_rel606_report() -> None:
    """Taxid 413997 and the ASM1798v1 accession pair, from the deposited report."""
    report = bc.read_assembly_report("REL606")
    assert report.model_dump(exclude={"sha256"}) == {
        "assembly_set": ECOLI_B_REL606,
        "member": "GCA_000017985.1_ASM1798v1_assembly_report.txt",
        "assembly_name": "ASM1798v1",
        "organism_name": "Escherichia coli B str. REL606 (E. coli)",
        "infraspecific_name": "strain=REL606",
        "taxid": 413997,
        "genbank_accession": "GCA_000017985.1",
        "refseq_accession": "GCF_000017985.1",
    }
    assert report.sha256 == (
        "51968f440a6497669ad8ccf703c437d5a8055990d2c7e27194b9cc1ffeeda369"
    )
    assert bc.assembly_reference("REL606").assembly_accession == "GCA_000017985.1"


@pytest.mark.data
@pytest.mark.skipif(
    not (REL606_TIER and REL606_CACHE),
    reason="requires the REL606 assembly set and its built data.db cache",
)
def test_tier_bacterial_genome_reopens_the_rel606_default_cache() -> None:
    """The REL606 class on ``data/ecoli/rel606/genome``, never overwritten; 4,316 genes
    (4,383 GenBank loci less 67 pseudogenes); thrA reconciles to ECB_00002.
    """
    genome = bc.bacterial_genome("ecoli", "REL606")
    assert type(genome) is EcoliBREL606Genome
    assert genome.genome_root == osp.join(DATA_ROOT, "data/ecoli/rel606/genome")
    assert (genome.overwrite, len(genome.gene_set)) == (False, 4316)
    stored, report = bc.reconcile_locus_tags(
        genome, pd.Series(["ECB_00001", "thrA", "ECB_99999"]), label="rel606"
    )
    assert stored.tolist() == ["ECB_00001", "ECB_00002", "ECB_99999"]
    assert (report.gene_namespace, report.outside_namespace) == (
        "ecoli_b_rel606_locus_tag",
        (),
    )


# --------------------------------------------------------------------------- #
# uniprot_locus_crosswalk: the accession -> locus-tag map read from the GOA file
#
# Measured on the synthetic MG1655 GAF (``MG1655_GAF``, seven annotation rows over
# five UniProt objects): ``UP_thrL``, ``UP_thrA`` and ``UP_proB`` each reach exactly
# one locus tag, ``UP_insZ`` reaches two (its synonym column is ``insZ|ychG|b0004/b0099``
# and the reader splits on ``/``), and ``UP_hokC`` reaches none because its only
# locus-like synonym is ``b0005.1``, which the assembly's own locus-tag pattern
# ``b\d{4}`` does not fully match.
# --------------------------------------------------------------------------- #
CROSSWALK_SINGLE: dict[str, str] = {
    "UP_thrL": "b0001",
    "UP_thrA": "b0002",
    "UP_proB": "b0005",
}
CROSSWALK_MULTI: dict[str, tuple[str, ...]] = {"UP_insZ": ("b0004", "b0099")}
CROSSWALK_NO_TAG = "UP_hokC"


@pytest.fixture
def crosswalk_genome(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[EcoliK12MG1655Genome, str]:
    """The synthetic MG1655 genome on a genomes-root-shaped tier WITH a manifest.

    The crosswalk reads the GOA member's pinned digest out of that manifest, so the
    tier has to look like the real one: ``<data_root>/torchcell-genomes/<set>/``, with
    a ``manifest.json`` whose record for the GOA file carries the bytes' own sha256.
    """
    from torchcell.literature.manifest import ROLE_ANNOTATIONS, ArtifactRecord
    from torchcell.sequence.genome.registry import GenomeManifest

    data_root = tmp_path / "data_root"
    files = write_assembly(
        data_root / "torchcell-genomes", MG1655_ASSEMBLY, MG1655_LOCI, MG1655_GAF
    )
    forbid_network(monkeypatch)
    serve_tier(monkeypatch, files)
    spec = MG1655_ASSEMBLY.go_source
    goa = files[(spec.assembly_set, spec.member)]
    manifest = GenomeManifest(
        assembly_set=spec.assembly_set,
        organism=MG1655_ASSEMBLY.organism,
        strain_or_population=MG1655_ASSEMBLY.strain,
        source="synthetic",
        release="synthetic",
        files=[
            ArtifactRecord(
                path=spec.member,
                role=ROLE_ANNOTATIONS,
                bytes=goa.stat().st_size,
                sha256=hashlib.sha256(goa.read_bytes()).hexdigest(),
            )
        ],
        provenance_complete=False,
        created_at="2026-10-09T00:00:00+00:00",
    )
    (goa.parent / "manifest.json").write_text(manifest.model_dump_json(indent=2))
    root = tmp_path / "mg1655"
    root.mkdir()
    genome = EcoliK12MG1655Genome(genome_root=str(root), overwrite=False)
    return genome, str(data_root)


def test_the_crosswalk_reads_the_goa_synonym_column_as_accession_to_locus_tag(
    crosswalk_genome: tuple[EcoliK12MG1655Genome, str],
) -> None:
    """One locus per accession where the file gives one, and the rest kept apart."""
    genome, data_root = crosswalk_genome
    crosswalk = bc.uniprot_locus_crosswalk(genome, data_root)
    spec = MG1655_ASSEMBLY.go_source
    assert crosswalk.assembly_set == spec.assembly_set
    assert crosswalk.member == spec.member
    assert crosswalk.rows == len(MG1655_GAF)
    assert crosswalk.single == CROSSWALK_SINGLE
    assert crosswalk.multi == CROSSWALK_MULTI
    assert CROSSWALK_NO_TAG not in crosswalk.single
    assert CROSSWALK_NO_TAG not in crosswalk.multi
    assert crosswalk.accessions == len(CROSSWALK_SINGLE) + len(CROSSWALK_MULTI)
    assert (
        crosswalk.sha256
        == hashlib.sha256(
            Path(
                osp.join(data_root, "torchcell-genomes", spec.assembly_set, spec.member)
            ).read_bytes()
        ).hexdigest()
    )


def test_the_crosswalk_refuses_a_host_whose_go_comes_from_the_refseq_gff() -> None:
    """A set with no GOA proteome file has no mirrored statement of the relation."""

    class _Bw25113:
        ASSEMBLY = BW25113_ASSEMBLY

    with pytest.raises(ValueError, match="which has no GOA proteome file"):
        bc.uniprot_locus_crosswalk(_Bw25113())  # type: ignore[arg-type]


def test_the_crosswalk_refuses_a_gaf_row_with_the_wrong_column_count(
    crosswalk_genome: tuple[EcoliK12MG1655Genome, str],
) -> None:
    """A GAF 2.x row has a fixed width; a short row is a different file.

    The replacement bytes are re-pinned in the manifest first, because otherwise the
    tier's own sha256 gate refuses the file before the reader ever sees the short row.
    """
    from torchcell.sequence.genome.registry import GenomeManifest

    genome, data_root = crosswalk_genome
    spec = MG1655_ASSEMBLY.go_source
    directory = Path(data_root) / "torchcell-genomes" / spec.assembly_set
    path = directory / spec.member
    with gzip.open(path, "wt") as handle:
        handle.write("!gaf-version: 2.2\nUniProtKB\tUP_x\tx\n")
    manifest = GenomeManifest.model_validate_json(
        (directory / "manifest.json").read_text()
    )
    record = manifest.record(spec.member)
    repinned = manifest.model_copy(
        update={
            "files": [
                record.model_copy(
                    update={
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "bytes": path.stat().st_size,
                    }
                )
            ]
        }
    )
    (directory / "manifest.json").write_text(repinned.model_dump_json(indent=2))
    with pytest.raises(ValueError, match="a GAF 2.x row has"):
        bc.uniprot_locus_crosswalk(genome, data_root)


def test_resolve_uniprot_accessions_splits_resolved_multi_locus_and_unmapped(
    crosswalk_genome: tuple[EcoliK12MG1655Genome, str],
) -> None:
    """Three outcomes, each listed in full, and the fraction over what was asked."""
    genome, data_root = crosswalk_genome
    crosswalk = bc.uniprot_locus_crosswalk(genome, data_root)
    asked = [*CROSSWALK_SINGLE, *CROSSWALK_MULTI, CROSSWALK_NO_TAG, "UP_absent"]
    resolution = bc.resolve_uniprot_accessions(crosswalk, asked, label="t")
    assert resolution.label == "t"
    assert resolution.assembly_set == crosswalk.assembly_set
    assert resolution.requested == len(asked)
    assert resolution.resolved == CROSSWALK_SINGLE
    assert resolution.multi_locus == CROSSWALK_MULTI
    assert resolution.unmapped == ("UP_absent", CROSSWALK_NO_TAG)
    assert resolution.collisions == {}
    assert resolution.resolved_fraction == len(CROSSWALK_SINGLE) / len(asked)
    resolution.require_resolved(0.5)
    with pytest.raises(bc.LocusTagResolutionError, match="reach one"):
        resolution.require_resolved(0.9)


def test_resolve_uniprot_accessions_reports_a_locus_two_accessions_reach(
    crosswalk_genome: tuple[EcoliK12MG1655Genome, str],
) -> None:
    """A collision is reported, never repaired: which column is the gene's is unknown."""
    genome, data_root = crosswalk_genome
    crosswalk = bc.uniprot_locus_crosswalk(genome, data_root)
    collided = crosswalk.model_copy(
        update={"single": {**crosswalk.single, "UP_other": "b0001"}}
    )
    resolution = bc.resolve_uniprot_accessions(
        collided, [*collided.single], label="collide"
    )
    assert resolution.collisions == {"b0001": ("UP_other", "UP_thrL")}


def test_resolve_uniprot_accessions_deduplicates_and_sorts_what_it_was_asked(
    crosswalk_genome: tuple[EcoliK12MG1655Genome, str],
) -> None:
    """``requested`` counts distinct accessions, so a repeated column is asked once."""
    genome, data_root = crosswalk_genome
    crosswalk = bc.uniprot_locus_crosswalk(genome, data_root)
    resolution = bc.resolve_uniprot_accessions(
        crosswalk, ["UP_thrL", "UP_thrL", "UP_thrA"], label="dupes"
    )
    assert resolution.requested == 2
    assert resolution.resolved_fraction == 1.0
