# tests/torchcell/adapters/test_host_genome_vocabulary.py
# [[tests.torchcell.adapters.test_host_genome_vocabulary]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/adapters/test_host_genome_vocabulary.py
"""Every ``perturbation.host_genome`` is in the ``AssemblySet`` vocabulary or null (regression).

``perturbation.host_genome = genome.assembly_set`` is the value join the ontology
unification exists to make resolve, and ``genome.assembly_set`` is a member of the
closed ``AssemblySet`` vocabulary (or null for a typed gap). Before lane Q (2026-10-15)
``HOST_GENOME_OF_NAMESPACE`` mapped ``scerevisiae_peter2018_pangenome_orf`` to the
registry id ``peter2018_1011_assemblies``, which ``NON_ASSEMBLY_REGISTRY_SETS`` excludes
from the vocabulary by name, so no genome node could ever carry it and the join could
not resolve for the 79,016 pangenome-namespaced perturbations of the served Caudal 2024
store. It stayed latent because Caudal's conf lists no perturbation lane. Pinned here so
it cannot: every value of the map is a vocabulary member or ``None``, ``None`` exactly
where the namespace's registry set is a non-assembly set, the node method writes
``None`` for a real pangenome presence call, and the optional ``perturbation of genome``
edge is absent for it. The first three assertions fail on the pre-lane-Q map.
"""

from __future__ import annotations

import typing
from typing import Any, Final, cast

from torchcell.adapters.cell_adapter import HOST_GENOME_OF_NAMESPACE, CellAdapter
from torchcell.datamodels import schema as s
from torchcell.sequence.genome import registry
from torchcell.sequence.genome.registry import (
    SGD_S288C_R64,
    SGD_S288C_R64_GENBANK_ACCESSION,
)
from torchcell.verification.report import Provenance
from torchcell.verification.sourced import SourcedValue

_SHA = "0" * 64
PANGENOME: Final = "scerevisiae_peter2018_pangenome_orf"


def test_every_host_genome_value_is_an_assembly_set_member_or_none() -> None:
    """The map is total over ``GeneIdNamespace`` and every value is a member of the
    ``AssemblySet`` vocabulary or ``None``; the pangenome namespace is the ``None``.
    """
    namespaces = set(typing.get_args(s.GeneIdNamespace))
    assemblies = set(typing.get_args(s.AssemblySet))
    assert set(HOST_GENOME_OF_NAMESPACE) == namespaces
    outside = {
        namespace: value
        for namespace, value in HOST_GENOME_OF_NAMESPACE.items()
        if value is not None and value not in assemblies
    }
    assert outside == {}
    assert HOST_GENOME_OF_NAMESPACE[PANGENOME] is None
    assert [n for n, v in HOST_GENOME_OF_NAMESPACE.items() if v is None] == [PANGENOME]


def test_none_is_exactly_the_non_assembly_registry_sets() -> None:
    """``REGISTRY_SET_OF_NAMESPACE`` is the total bijection onto registry ids; a
    namespace projects to ``None`` exactly when its set is in
    ``NON_ASSEMBLY_REGISTRY_SETS``, and to its own set id otherwise. (The map is lane
    Q's, so it is imported here and not at module scope: the other three tests of this
    file run, and fail on their assertions, on the code before it.)
    """
    from torchcell.adapters.cell_adapter import REGISTRY_SET_OF_NAMESPACE

    namespaces = set(typing.get_args(s.GeneIdNamespace))
    assert set(REGISTRY_SET_OF_NAMESPACE) == namespaces
    assert len(set(REGISTRY_SET_OF_NAMESPACE.values())) == len(namespaces)
    registry_ids = {
        value
        for name, value in vars(registry).items()
        if name.isupper() and isinstance(value, str) and "_" in value
    }
    assert set(REGISTRY_SET_OF_NAMESPACE.values()) <= registry_ids
    for namespace, registry_set in REGISTRY_SET_OF_NAMESPACE.items():
        if registry_set in s.NON_ASSEMBLY_REGISTRY_SETS:
            assert HOST_GENOME_OF_NAMESPACE[namespace] is None
        else:
            assert HOST_GENOME_OF_NAMESPACE[namespace] == registry_set
    assert REGISTRY_SET_OF_NAMESPACE[PANGENOME] == registry.PETER2018_1011
    assert registry.PETER2018_1011 in s.NON_ASSEMBLY_REGISTRY_SETS


def _media() -> s.Media:
    return s.Media(
        name="YPD",
        state="liquid",
        is_synthetic=False,
        base_medium="YPD",
        components=[],
        provenance=[
            SourcedValue(
                value="YPD",
                quote="YPD",
                provenance=Provenance(
                    source_uri="paper.md", citation_key="test2026", sha256=_SHA
                ),
            )
        ],
    )


def _isolate_record() -> dict[str, Any]:
    """One natural-isolate record: a pangenome presence call beside an S288C-keyed
    absence call, referenced to the S288C set as Caudal 2024 is.
    """
    presence = s.NaturalGenePresencePerturbation(
        systematic_gene_name="13-EC1118_1N26_0012g",
        gene_namespace=PANGENOME,
        perturbed_gene_name="13-EC1118_1N26_0012g",
        strain_id="AAA",
        pangenome_orf_id="13-EC1118_1N26_0012g",
    )
    absence = s.NaturalGeneAbsencePerturbation(
        systematic_gene_name="YAL069W",
        gene_namespace="scerevisiae_s288c_sgd_systematic",
        perturbed_gene_name="YAL069W",
        strain_id="AAA",
    )
    experiment = s.FitnessExperiment(
        dataset_name="TestDataset",
        genotype=s.Genotype(perturbations=[presence, absence]),
        environment=s.Environment(media=_media(), temperature=s.Temperature(value=30)),
        phenotype=s.FitnessPhenotype(fitness=1.0, fitness_std=None),
    )
    reference = s.FitnessExperimentReference(
        dataset_name="TestDataset",
        genome_reference=s.ReferenceGenome(
            species="Saccharomyces cerevisiae",
            strain="AAA",
            assembly_set=SGD_S288C_R64,
            assembly_accession=SGD_S288C_R64_GENBANK_ACCESSION,
        ),
        environment_reference=experiment.environment,
        phenotype_reference=s.FitnessPhenotype(fitness=1.0, fitness_std=None),
    )
    return {"experiment": experiment, "reference": reference}


def _call(method: Any, data: dict[str, Any], name: str) -> Any:
    """Run a ``data_chunker`` method on one record; ``self`` is the class, enough for
    the static helpers (``_genome_node_id``) the edge method reaches through it.
    """
    return cast(Any, method).__wrapped__(CellAdapter, data, name)


def test_a_pangenome_presence_call_projects_no_host_genome() -> None:
    """The node method writes ``host_genome`` ``None`` for the pangenome presence call
    and the S288C set for the absence call of the same genotype; both nodes keep their
    content ids.
    """
    data = _isolate_record()
    nodes = _call(CellAdapter._perturbation_node, data, "perturbation (chunked)")
    by_name = {n.get_properties()["systematic_gene_name"]: n for n in nodes}
    assert by_name["13-EC1118_1N26_0012g"].get_properties()["host_genome"] is None
    assert by_name["YAL069W"].get_properties()["host_genome"] == SGD_S288C_R64
    assert {n.get_label() for n in nodes} == {"perturbation"}


def test_the_optional_genome_edge_is_absent_where_host_genome_is_null() -> None:
    """``perturbation of genome`` (off in every conf) joins the S288C-keyed call to the
    record's genome node and writes no edge for the pangenome call, whose identifier is
    not written in that node's set.
    """
    data = _isolate_record()
    edges = _call(
        CellAdapter._perturbation_to_genome_edges, data, "perturbation of genome"
    )
    nodes = _call(CellAdapter._perturbation_node, data, "perturbation (chunked)")
    by_name = {n.get_properties()["systematic_gene_name"]: n.get_id() for n in nodes}
    assert [e.get_source_id() for e in edges] == [by_name["YAL069W"]]
    assert edges[0].get_target_id() == CellAdapter._genome_node_id(
        data["reference"].genome_reference
    )
    assert CellAdapter._assembly_set_of(data["reference"].genome_reference) == (
        SGD_S288C_R64
    )
