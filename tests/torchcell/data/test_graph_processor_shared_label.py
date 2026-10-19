# tests/torchcell/data/test_graph_processor_shared_label.py
# [[tests.torchcell.data.test_graph_processor_shared_label]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/data/test_graph_processor_shared_label.py
"""Two phenotype classes that share a ``label_name`` are read once per record.

``MicroarrayExpressionPhenotype`` and ``PseudobulkExpressionPhenotype`` both carry the
label ``expression_log2_ratio``. A store holding both (experiment 043) passes both classes
in ``phenotype_info``; before the fix the processors appended the label once per class, so
a Kemmeren sample carried its 6,169 values twice (12,338). Expected here, by hand: one
microarray record with log2 ratios on 3 genes and the SE on the same 3 genes gives
``phenotype_types == ["expression_log2_ratio"]``, 3 values ``[1.0, -0.5, 2.0]`` in key
order, and 3 statistic values; the statistic names stay ``expression_log2_ratio_se`` then
``dispersion`` (distinct, so both are kept).
"""

from __future__ import annotations

from typing import Any

import networkx as nx
import pytest
from sortedcontainers import SortedDict
from torch_geometric.data import HeteroData

from torchcell.data.cell_data import to_cell_data
from torchcell.data.graph_processor import Perturbation
from torchcell.datamodels.schema import (
    Environment,
    Genotype,
    KanMxDeletionPerturbation,
    Media,
    MicroarrayExpressionExperiment,
    MicroarrayExpressionExperimentReference,
    MicroarrayExpressionPhenotype,
    PseudobulkExpressionPhenotype,
    ReferenceGenome,
)
from torchcell.graph.graph import GeneGraph, GeneMultiGraph
from torchcell.sequence import GeneSet
from torchcell.sequence.genome.registry import (
    SGD_S288C_R64,
    SGD_S288C_R64_GENBANK_ACCESSION,
)

GENES = GeneSet(["YAL001C", "YAL002W", "YAL003W"])
SHARED: list[Any] = [MicroarrayExpressionPhenotype, PseudobulkExpressionPhenotype]
ENVIRONMENT = Environment(media=Media(name="SC", state="liquid", is_synthetic=True))
GENOME = ReferenceGenome(
    species="Saccharomyces cerevisiae",
    strain="S288C",
    assembly_set=SGD_S288C_R64,
    assembly_accession=SGD_S288C_R64_GENBANK_ACCESSION,
)


def _phenotype() -> MicroarrayExpressionPhenotype:
    return MicroarrayExpressionPhenotype(
        expression={"YAL001C": 2.0, "YAL002W": 0.7, "YAL003W": 4.0},
        expression_log2_ratio={"YAL001C": 1.0, "YAL002W": -0.5, "YAL003W": 2.0},
        expression_log2_ratio_se={"YAL001C": 0.1, "YAL002W": 0.2, "YAL003W": 0.3},
        n_replicates={"YAL001C": 2, "YAL002W": 2, "YAL003W": 2},
    )


def _record() -> dict[str, Any]:
    return {
        "experiment": MicroarrayExpressionExperiment(
            dataset_name="toy",
            genotype=Genotype(
                perturbations=[
                    KanMxDeletionPerturbation(
                        systematic_gene_name="YAL001C", perturbed_gene_name="TFC3"
                    )
                ]
            ),
            environment=ENVIRONMENT,
            phenotype=_phenotype(),
        ),
        "experiment_reference": MicroarrayExpressionExperimentReference(
            dataset_name="toy",
            genome_reference=GENOME,
            environment_reference=ENVIRONMENT,
            phenotype_reference=_phenotype(),
        ),
    }


def _cell_graph() -> HeteroData:
    base = nx.Graph()
    base.add_nodes_from(GENES)
    multigraph = GeneMultiGraph(
        graphs=SortedDict(
            {"base": GeneGraph(name="base", graph=base, max_gene_set=GENES)}
        )
    )
    return to_cell_data(multigraph)


def test_a_shared_label_is_read_once_per_record() -> None:
    """The Perturbation processor (the 043 store's) reads the shared label once.

    The subgraph processors take scalar labels only (a dict label raises in their
    tensor build), so the dict-valued case is the Perturbation processor's.
    """
    out = Perturbation().process(_cell_graph(), SHARED, [_record()])
    gene = out["gene"]
    assert gene.phenotype_types == ["expression_log2_ratio"]
    assert gene.phenotype_values.tolist() == [1.0, -0.5, 2.0]
    assert gene.phenotype_type_indices.tolist() == [0, 0, 0]
    assert gene.phenotype_sample_indices.tolist() == [0, 0, 0]
    assert gene.phenotype_stat_types == ["expression_log2_ratio_se", "dispersion"]
    assert gene.phenotype_stat_values.tolist() == pytest.approx([0.1, 0.2, 0.3])
