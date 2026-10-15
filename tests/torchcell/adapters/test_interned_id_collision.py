# tests/torchcell/adapters/test_interned_id_collision.py
"""A whole genotype interned as a constant keeps its own genotype node (regression).

Before 2026-10-15 ``constant_id`` was the bare sha256 of the payload, and a whole
genotype's payload is ``json.dumps(genotype.model_dump())``, the exact bytes
``CellAdapter._genotype_node`` hashes for the genotype node id. ``FastCsvSink``
deduplicates node ids across labels and the adapter runs ``experiment (chunked)``
before ``genotype (chunked)``, so the ``interned constant`` row owned the id, the
``Genotype`` row was dropped, and every ``GenotypeMemberOf`` and
``PerturbationMemberOf`` row of such a record pointed at an ``InternedConstant`` node
(218,258 genotypes on the job 4901 partial; 14,573 on the served KG 3.0 store).

The synthetic dataset here is one fitness record whose genotype carries enough
deletions to cross the 8,192-byte interning floor, pushed through the adapter's own
node and edge methods and through the real chunk renderer and sink in the adapter's
method order. Pinned: the constant's id differs from the genotype node's id, both rows
are written under their own labels, the ``GenotypeMemberOf`` start id is the id the
``Genotype`` row carries, and the pointer still resolves to the inlined record byte
for byte. On the old ``constant_id`` the first three assertions fail.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

from torchcell.adapters.cell_adapter import CellAdapter
from torchcell.datamodels import schema as s
from torchcell.datamodels.interned_constant import (
    EXPERIMENT_POINTER_MIN_BYTES,
    POINTER_KEY,
    constant_id,
    resolve_pointers,
    verified_constant,
)
from torchcell.fast_csv import (
    EdgeRowSpec,
    FastCsvSink,
    NodeRowSpec,
    RenderedChunk,
    RowSpecs,
)
from torchcell.verification.report import Provenance
from torchcell.verification.sourced import SourcedValue

_SHA = "0" * 64
_ID_PROPS = {"id": "str", "preferred_id": "str"}

SPECS = RowSpecs(
    delim="\t",
    quote="'",
    adelim="|",
    nodes={
        "experiment": NodeRowSpec(
            label="experiment",
            pascal="Experiment",
            prop_types={"serialized_data": "str", **_ID_PROPS},
            labels="'Experiment|Entity'",
        ),
        "interned constant": NodeRowSpec(
            label="interned constant",
            pascal="InternedConstant",
            prop_types={"kind": "str", "serialized_data": "str", **_ID_PROPS},
            labels="'InternedConstant|InformationContentEntity|Entity'",
        ),
        "genotype": NodeRowSpec(
            label="genotype",
            pascal="Genotype",
            prop_types={
                "systematic_gene_names": "list",
                "perturbed_gene_names": "list",
                "perturbation_types": "list",
                **_ID_PROPS,
            },
            labels="'Genotype|Entity'",
        ),
    },
    edges={
        "genotype member of": EdgeRowSpec(
            label="genotype member of",
            pascal="GenotypeMemberOf",
            prop_types={},
            skip_id=False,
            labels="'GenotypeMemberOf'",
            experiment_endpoint="target",
        )
    },
)


def _media() -> s.Media:
    return s.Media(
        name="YPD",
        state="solid",
        is_synthetic=False,
        base_medium="YPD",
        components=[],
        provenance=[
            SourcedValue(
                value="YPD",
                quote="YPD plates",
                provenance=Provenance(
                    source_uri="paper.md", citation_key="test2026", sha256=_SHA
                ),
            )
        ],
    )


def _experiment(n_deletions: int) -> s.FitnessExperiment:
    return s.FitnessExperiment(
        dataset_name="TestDataset",
        genotype=s.Genotype(
            perturbations=[
                s.SgaKanMxDeletionPerturbation(
                    systematic_gene_name=f"YAL{i:03d}C",
                    perturbed_gene_name=f"GENE{i}",
                    strain_id=f"DMA{i}",
                )
                for i in range(n_deletions)
            ]
        ),
        environment=s.Environment(
            media=_media(), temperature=s.Temperature(value=30.0)
        ),
        phenotype=s.FitnessPhenotype(fitness=0.9, fitness_std=0.01),
    )


def _call(method: Any, data: dict[str, Any], name: str) -> Any:
    return cast(Any, method).__wrapped__(None, data, name)


def test_a_whole_genotype_constant_and_its_genotype_node_are_two_nodes(
    tmp_path: Path,
) -> None:
    experiment = _experiment(n_deletions=48)
    dump = experiment.model_dump()
    genotype_payload = json.dumps(dump["genotype"])
    assert len(genotype_payload) >= EXPERIMENT_POINTER_MIN_BYTES["genotype"]
    data = {"experiment": experiment}
    experiment_nodes = _call(CellAdapter._experiment_node, data, "experiment (chunked)")
    genotype_node = _call(CellAdapter._genotype_node, data, "genotype (chunked)")
    edge = _call(CellAdapter._genotype_to_experiment_edge, data, "genotype member of")

    experiment_node, constants = experiment_nodes[0], experiment_nodes[1:]
    by_kind = {c.get_properties()["kind"]: c for c in constants}
    assert sorted(by_kind) == ["environment", "genotype"]
    whole = by_kind["genotype"]
    genotype_id = hashlib.sha256(genotype_payload.encode("utf-8")).hexdigest()
    # The genotype node id is the bare hash of the dump, as every served edge expects.
    assert genotype_node.get_id() == genotype_id
    assert edge.get_source_id() == genotype_id
    # The regression: the constant does not take the genotype node's id.
    assert whole.get_id() != genotype_id
    assert whole.get_id() == constant_id("genotype", genotype_payload)

    # Through the real renderer and sink, in the adapter's method order (experiment
    # first, genotype second): both rows are written, each under its own label.
    bc = SimpleNamespace(_writer=SimpleNamespace(outdir=str(tmp_path)))
    sink = FastCsvSink(bc, SPECS)
    chunk = RenderedChunk.from_rows([*experiment_nodes, genotype_node], SPECS)
    assert sink.write_nodes(iter([chunk])) == 4
    assert sink.node_rows == {"Experiment": 1, "InternedConstant": 2, "Genotype": 1}
    assert sink.node_dups == 0
    assert sink.seen_nodes[edge.get_source_id()] == "Genotype"
    assert sink.seen_nodes[whole.get_id()] == "InternedConstant"
    assert (tmp_path / "Genotype-part000.csv").exists()

    # The pointer in the blob still resolves to the inlined record byte for byte.
    blob = json.loads(experiment_node.get_properties()["serialized_data"])
    assert blob["genotype"] == {POINTER_KEY: whole.get_id(), "kind": "genotype"}
    store = {
        c.get_id(): verified_constant(
            c.get_id(), kind, c.get_properties()["serialized_data"]
        )
        for kind, c in by_kind.items()
    }
    assert json.dumps(resolve_pointers(blob, store)) == json.dumps(dump)
    assert (
        experiment_node.get_id()
        == hashlib.sha256(json.dumps(dump).encode("utf-8")).hexdigest()
    )
