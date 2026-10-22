"""The experiment's annotation to the schema DAG and the DAG encoder over it."""

from typing import Any

import pytest
import torch
from torch_geometric.data import Batch, HeteroData

from torchcell.data.experiment_ontology import (
    OntologyDagEncoder,
    build_experiment_ontology_vocabulary,
    describe_annotation,
)
from torchcell.data.graph_processor import Perturbation
from torchcell.datamodels.media import SGA_DM_SELECTION, SGA_TM_SELECTION
from torchcell.datamodels.schema import (
    Environment,
    FitnessExperiment,
    FitnessExperimentReference,
    FitnessPhenotype,
    GeneInteractionExperiment,
    GeneInteractionExperimentReference,
    GeneInteractionPhenotype,
    Genotype,
    Media,
    ReferenceGenome,
    SgaKanMxDeletionPerturbation,
    Temperature,
)

GENES = ["YAL001C", "YAL002W", "YAL003W"]
PHENOTYPES: list[Any] = [FitnessPhenotype, GeneInteractionPhenotype]


def _genotype(gene: str) -> Genotype:
    return Genotype(
        perturbations=[
            SgaKanMxDeletionPerturbation(
                systematic_gene_name=gene,
                perturbed_gene_name=gene,
                strain_id=f"{gene}_x",
            )
        ]
    )


def _genome() -> ReferenceGenome:
    return ReferenceGenome(species="Saccharomyces cerevisiae", strain="S288C")


def _fitness(
    dataset: str, media: Media, temp: float, gene: str = "YAL001C"
) -> dict[str, Any]:
    env = Environment(media=media, temperature=Temperature(value=temp))
    return {
        "experiment": FitnessExperiment(
            dataset_name=dataset,
            genotype=_genotype(gene),
            environment=env,
            phenotype=FitnessPhenotype(fitness=0.9, fitness_std=None),
        ),
        "experiment_reference": FitnessExperimentReference(
            dataset_name=dataset,
            genome_reference=_genome(),
            environment_reference=env,
            phenotype_reference=FitnessPhenotype(fitness=1.0, fitness_std=None),
        ),
    }


def _interaction(
    dataset: str, media: Media, temp: float, gene: str = "YAL002W"
) -> dict[str, Any]:
    env = Environment(media=media, temperature=Temperature(value=temp))
    return {
        "experiment": GeneInteractionExperiment(
            dataset_name=dataset,
            genotype=_genotype(gene),
            environment=env,
            phenotype=GeneInteractionPhenotype(
                gene_interaction=-0.1, gene_interaction_p_value=None
            ),
        ),
        "experiment_reference": GeneInteractionExperimentReference(
            dataset_name=dataset,
            genome_reference=_genome(),
            environment_reference=env,
            phenotype_reference=GeneInteractionPhenotype(
                gene_interaction=0.0, gene_interaction_p_value=None
            ),
        ),
    }


def _records() -> list[dict[str, Any]]:
    return [
        _interaction("TmiKuzmin2018Dataset", SGA_TM_SELECTION, 26.0),
        _interaction("TmiKuzmin2020Dataset", SGA_TM_SELECTION, 26.0),
        _interaction("DmiCostanzo2016Dataset", SGA_DM_SELECTION, 30.0),
        _fitness("TmfKuzmin2018Dataset", SGA_TM_SELECTION, 26.0),
        _fitness(
            "SmfSgdDataset", Media(name="YEPD", state="solid", is_synthetic=False), 30.0
        ),
    ]


def _vocab():
    return build_experiment_ontology_vocabulary(
        (r["experiment"], r["experiment_reference"]) for r in _records()
    )


def test_annotation_never_carries_the_dataset_name_and_collapses_the_two_screens() -> (
    None
):
    vocab = _vocab()
    assert not any("Dataset" in n.name or "dataset_name" in n.name for n in vocab.nodes)
    recs = _records()
    vec = {
        r["experiment"].dataset_name: vocab.vector(
            r["experiment"], r["experiment_reference"]
        )
        for r in recs
    }
    assert torch.equal(vec["TmiKuzmin2018Dataset"], vec["TmiKuzmin2020Dataset"])
    a = set(describe_annotation(vocab, vec["DmiCostanzo2016Dataset"]))
    b = set(describe_annotation(vocab, vec["TmiKuzmin2018Dataset"]))
    assert a ^ b == {
        "Media.dropout:uracil",
        "Temperature.value=0.26",
        "Temperature.value=0.3",
    }
    f = set(describe_annotation(vocab, vec["TmfKuzmin2018Dataset"]))
    assert {"FitnessExperiment", "FitnessPhenotype", "ref:FitnessPhenotype"} <= f
    assert {"GeneInteractionExperiment", "GeneInteractionPhenotype"} <= b and not (
        {"FitnessExperiment"} & b
    )
    # the owners of every annotated leaf are marked, up to the root
    assert {
        "Experiment",
        "Environment",
        "Media",
        "MediaComponent",
        "Temperature",
        "Phenotype",
    } <= b
    # a plain YEPD record has no components and no dropouts, and still annotates
    s = set(describe_annotation(vocab, vec["SmfSgdDataset"]))
    assert "Media.is_synthetic:false" in s and "MediaComponent" not in s


def test_vocabulary_is_a_dag_with_the_experiment_root_deepest_and_rejects_unknown_values() -> (
    None
):
    vocab = _vocab()
    depths = vocab.depths()
    root = vocab.index("Experiment")
    assert depths[root] == max(depths)
    assert vocab.nodes[root].parents == []
    # every value node has exactly one class parent
    for n in vocab.nodes:
        if n.kind != "class":
            assert (
                len(n.parents) == 1
                and vocab.nodes[vocab.index(n.parents[0])].kind == "class"
            )
    odd = _fitness(
        "X",
        Media(name="odd", state="liquid", is_synthetic=True, base_medium="UNSEEN"),
        30.0,
    )
    with pytest.raises(ValueError, match="not in the experiment ontology vocabulary"):
        vocab.vector(odd["experiment"], odd["experiment_reference"])
    # round trip
    again = type(vocab).model_validate(vocab.model_dump())
    assert again.names == vocab.names


def test_encoder_is_deterministic_on_the_annotation_and_zero_for_absent_subtrees() -> (
    None
):
    vocab = _vocab()
    torch.manual_seed(0)
    enc = OntologyDagEncoder(vocab, state_dim=6, token_dim=4)
    recs = _records()
    x = torch.stack(
        [vocab.vector(r["experiment"], r["experiment_reference"]) for r in recs]
    )
    tok = enc(x)
    assert tok.shape == (5, 4)
    assert torch.allclose(tok[0], tok[1])  # the two Kuzmin screens
    assert (
        tok[0] - tok[2]
    ).abs().max() > 1e-4  # Costanzo differs (small at init, learnable)
    assert (tok[0] - tok[3]).abs().max() > 1e-4  # fitness differs from interaction
    states = enc.states(x)
    # a subtree no record touches stays at zero: pick a class node never annotated
    never = [
        i
        for i, n in enumerate(vocab.nodes)
        if n.kind == "class"
        and (x[:, i] == 0).all()
        and all(x[:, c].sum() == 0 for c in vocab.children()[i])
    ]
    assert never, "the test needs an untouched class node"
    assert torch.all(states[:, never[0]] == 0)
    # gradient reaches every per-node weight on an active path
    tok.sum().backward()
    assert (
        enc.child_proj.grad is not None
        and enc.child_proj.grad[vocab.index("Environment")].abs().sum() > 0
    )


def test_processor_emits_the_annotation_beside_every_value_and_batches_by_rows() -> (
    None
):
    vocab = _vocab()
    cg = HeteroData()
    cg["gene"].num_nodes = len(GENES)
    cg["gene"].node_ids = list(GENES)
    names = [
        "DmiCostanzo2016Dataset",
        "SmfSgdDataset",
        "TmfKuzmin2018Dataset",
        "TmiKuzmin2018Dataset",
        "TmiKuzmin2020Dataset",
    ]
    proc = Perturbation(dataset_vocabulary=names, ontology_vocabulary=vocab)
    recs = _records()
    a = proc.process(cg, PHENOTYPES, recs[:3])
    b = proc.process(cg, PHENOTYPES, recs[3:])
    assert a["gene"].phenotype_ontology.shape == (3, len(vocab))
    assert torch.equal(a["gene"].phenotype_ontology[0], a["gene"].phenotype_ontology[1])
    batch = Batch.from_data_list([a, b])
    g = batch["gene"]
    assert g.phenotype_ontology.shape == (5, len(vocab))
    assert g.phenotype_ontology.shape[0] == g.phenotype_values.shape[0]
    assert g.phenotype_dataset_indices.shape[0] == 5
