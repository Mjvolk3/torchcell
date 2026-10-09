# tests/torchcell/adapters/test_bacterial_graph_classes.py
# [[tests.torchcell.adapters.test_bacterial_graph_classes]]
"""The graph classes the bacterial schema needs, and the yeast output they leave alone.

``bacterial perturbation``, ``product titer phenotype``, ``protein turnover phenotype``,
``flux phenotype`` and ``promoter activity phenotype`` are NEW node classes, emitted by
NEW ``CellAdapter`` methods, so a served dataset's nodes cannot move. Three things are
pinned here:

- Each new method emits the class it declares, with the id rule every sub-object node
  uses (sha256 of the json-dumped ``model_dump``), the declared property set exactly, and
  the values projected the way the existing families project them (an enum as its value,
  a dict and the product ``Compound`` as a JSON string).
- The served edge methods already address the new nodes: ``perturbation member of``,
  ``crispr construct member of`` and ``phenotype member of`` compute the same ids, so no
  edge method is added.
- A yeast record's output is what ``origin/main`` (76933585) emits. The expected
  ``perturbation`` nodes are written out in full below, and the other methods a yeast
  dataset enables are pinned by the sha256 of their canonical output, computed by running
  main's ``CellAdapter`` on the same record (the record and the digest rule are defined
  in this file). The bacterial method emits nothing for it.

Fixture records are built from the Step 4 classes in ``torchcell.datamodels.schema``.
"""

from __future__ import annotations

import hashlib
import json
import typing
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import yaml

import torchcell
from torchcell.adapters.cell_adapter import (
    BACTERIAL_PERTURBATION_LEAVES,
    BACTERIAL_VARIANT_PERTURBATION_LEAVES,
    CellAdapter,
)
from torchcell.datamodels import schema as s
from torchcell.knowledge_graphs.kg_manifest import (
    CELL_ADAPTER_RELPATH,
    SCHEMA_CONFIG_RELPATH,
    cell_adapter_surface,
)

REPO_ROOT = Path(torchcell.__file__).resolve().parent.parent
SCHEMA = yaml.safe_load((REPO_ROOT / SCHEMA_CONFIG_RELPATH).read_text(encoding="utf-8"))
# BioCypherNode adds these two to every node's properties.
NODE_BOOKKEEPING = {"id", "preferred_id"}

MG1655: s.BacterialGeneNamespace = "ecoli_k12_mg1655_bnumber"


def _sha(model: Any) -> str:
    return hashlib.sha256(json.dumps(model.model_dump()).encode("utf-8")).hexdigest()


def _undecorated(method: Any) -> Any:
    """The chunk handler behind ``@data_chunker`` (the decorator needs an LMDB loader)."""
    return cast(Any, method).__wrapped__


def _adapter() -> CellAdapter:
    return CellAdapter.__new__(CellAdapter)


def _lb() -> s.Media:
    return s.Media(name="LB", state="liquid", is_synthetic=False)


def _construct() -> s.CrisprConstruct:
    return s.CrisprConstruct(effector="dCas9", guide_sequence="GTCAGGTACTCCGAATTCGA")


def _bacterial_leaves() -> list[Any]:
    """One perturbation of each bacterial leaf class, all against MG1655 b-numbers."""
    return [
        s.BacterialDeletionPerturbation(
            systematic_gene_name="b0002",
            perturbed_gene_name="thrA",
            gene_namespace=MG1655,
            collection="Keio collection",
        ),
        s.TransposonInsertionPerturbation(
            systematic_gene_name="b0003",
            perturbed_gene_name="thrB",
            gene_namespace=MG1655,
            barcode="ACGTACGTACGTACGTACGT",
            insertion_position=2850,
            insertion_strand="+",
        ),
        s.BacterialCrisprInterferencePerturbation(
            systematic_gene_name="b0004",
            perturbed_gene_name="thrC",
            gene_namespace=MG1655,
            crispr=_construct(),
        ),
        s.PromoterReplacementPerturbation(
            systematic_gene_name="b0005",
            perturbed_gene_name="yaaX",
            gene_namespace=MG1655,
            expression_direction="increased",
            promoter_name="Ptac",
        ),
        s.HeterologousPathwayPerturbation(
            systematic_gene_name="Efa:mvaE",
            perturbed_gene_name="mvaE",
            gene_namespace=MG1655,
            source_organism="Enterococcus faecalis",
            is_heterologous=True,
            localization="plasmid",
            pathway_name="isoprenol via mevalonate",
        ),
    ]


#: One call of each shape the #731 leaves carry, as rows of the Lim 2025 released
#: ``Fig 2B_Mutation List`` sheet, so the fixture exercises real coordinates. The
#: namespace is KT2440's, since that is the host those rows were called against.
KT2440: s.BacterialGeneNamespace = "pputida_kt2440_locus_tag"


def _call(**kw: Any) -> s.BacterialVariantCall:
    fields: dict[str, Any] = dict(
        variant_type=s.BacterialVariantType.snv,
        type_statement="SNP",
        reference_sequence="AE015451",
        position_start=3866001,
        position_end=3866001,
        sequence_change="G\u2192A",
        annotation="P293S (CCA\u2192TCA)",
        call_mode=s.VariantCallMode.clone,
        frequency_statement="1",
        frequency=1.0,
        frequency_basis=s.VariantFrequencyBasis.fraction,
        caller="breseq 0.33.1",
    )
    fields.update(kw)
    return s.BacterialVariantCall(**fields)


def _variant_leaves() -> list[Any]:
    """One perturbation of each called-variant leaf class, all against KT2440 tags."""
    intergenic = _call(
        variant_type=s.BacterialVariantType.insertion,
        type_statement="INS",
        position_start=4586057,
        position_end=4586057,
        sequence_change="+C",
        annotation="intergenic (+140/+75)",
    )
    span = _call(
        variant_type=s.BacterialVariantType.deletion,
        type_statement="DEL",
        position_start=4588139,
        position_end=4588139,
        sequence_change="\u03945,553 bp",
        annotation=None,
    )
    return [
        s.BacterialSequenceVariantPerturbation(
            systematic_gene_name="PP_3415",
            perturbed_gene_name="PP_3415",
            gene_namespace=KT2440,
            call=_call(),
        ),
        s.BacterialSiteVariantPerturbation(
            systematic_gene_name=s.BacterialSiteVariantPerturbation.site_id(intergenic),
            perturbed_gene_name="PP_4061, PP_4063",
            gene_namespace=KT2440,
            call=intergenic,
            site_kind=s.VariantSiteKind.intergenic,
            flanking_systematic_gene_names=("PP_4061", "PP_4063"),
            flanking_gene_statement="PP_4061, PP_4063",
        ),
        s.BacterialSpanDeletionPerturbation(
            systematic_gene_name="PP_4062",
            perturbed_gene_name="PP_4062",
            gene_namespace=KT2440,
            call=span,
            span_designation=s.BacterialSpanDeletionPerturbation.designation(span),
            span_systematic_gene_names=("PP_4062", "PP_4063"),
        ),
    ]


def _titer(**kw: Any) -> s.ProductTiterPhenotype:
    fields: dict[str, Any] = dict(
        product=s.Compound(name="isoprenol", inchikey="XHQZJYCNDZAGLW-UHFFFAOYSA-N"),
        titer=2.4,
        titer_unit=s.ConcentrationUnit.g_per_l,
    )
    fields.update(kw)
    return s.ProductTiterPhenotype(**fields)


def _full_titer() -> s.ProductTiterPhenotype:
    """Every optional titer field set, so every projection rule is exercised."""
    return _titer(
        titer_uncertainty=0.3,
        titer_uncertainty_type=s.UncertaintyType.sample_sd,
        n_samples=9,
        sample_unit=s.SampleUnit.biological_replicate,
        product_yield=0.12,
        product_yield_unit=s.ProductYieldUnit.g_per_g_substrate,
        productivity=0.05,
        productivity_unit=s.ProductivityUnit.g_per_l_per_h,
        quantification_method="GC-MS",
    )


def _turnover(**kw: Any) -> s.ProteinTurnoverPhenotype:
    fields: dict[str, Any] = dict(
        degradation_rate={"b0002": 0.11, "b0003": 0.07},
        n_replicates={"b0002": 3, "b0003": 2},
        measurement_type="pulse_silac_degradation_rate_per_hour",
    )
    fields.update(kw)
    return s.ProteinTurnoverPhenotype(**fields)


def _flux(**kw: Any) -> s.FluxPhenotype:
    fields: dict[str, Any] = dict(
        net_flux={"PGI": -1.2, "TPI": 0.4},
        measurement_type="c13_mfa_net_flux_mmol_per_gdcw_per_h",
    )
    fields.update(kw)
    return s.FluxPhenotype(**fields)


def _activity(**kw: Any) -> s.PromoterActivityPhenotype:
    fields: dict[str, Any] = dict(
        promoter_activity=12.98,
        n_samples=1,
        sample_unit="biological_replicate",
        promoter_name="thrA",
        promoter_gene="b0002",
        readout="plate_reader_fluorescence",
        reporter_gene="gfp",
        activity_units="GFP fluorescence in the reader's own units",
        well_id="Untreated|AZ01|A1",
    )
    fields.update(kw)
    return s.PromoterActivityPhenotype(**fields)


def _promoter_activity_record(phenotype: s.PromoterActivityPhenotype) -> dict[str, Any]:
    return {
        "experiment": s.PromoterActivityExperiment(
            dataset_name="BacterialToy",
            genotype=s.Genotype(perturbations=_bacterial_leaves()),
            environment=s.Environment(media=_lb()),
            phenotype=phenotype,
        )
    }


def _product_titer_record(phenotype: s.ProductTiterPhenotype) -> dict[str, Any]:
    return {
        "experiment": s.ProductTiterExperiment(
            dataset_name="BacterialToy",
            genotype=s.Genotype(perturbations=_bacterial_leaves()),
            # The product-titer family declares CultureEnvironment: a titer is read with
            # its vessel, and an Environment in that slot is refused.
            environment=s.CultureEnvironment(media=_lb()),
            phenotype=phenotype,
        )
    }


def _variant_record(phenotype: s.ProductTiterPhenotype) -> dict[str, Any]:
    """A record whose genotype is the three called-variant leaves and nothing else."""
    return {
        "experiment": s.ProductTiterExperiment(
            dataset_name="BacterialVariantToy",
            genotype=s.Genotype(perturbations=_variant_leaves()),
            environment=s.CultureEnvironment(media=_lb()),
            phenotype=phenotype,
        )
    }


def _yeast_record() -> dict[str, Any]:
    """The record the main goldens below were computed from (do not edit one alone)."""
    return {
        "experiment": s.FitnessExperiment(
            dataset_name="YeastGolden",
            genotype=s.Genotype(
                perturbations=[
                    s.SgaKanMxDeletionPerturbation(
                        systematic_gene_name="YAL001C",
                        perturbed_gene_name="TFC3",
                        strain_id="YAL001C_dma1",
                    ),
                    s.KanMxDeletionPerturbation(
                        systematic_gene_name="YAL002W", perturbed_gene_name="VPS8"
                    ),
                    s.CrisprInterferencePerturbation(
                        systematic_gene_name="YAL003W",
                        perturbed_gene_name="EFB1",
                        crispr=s.CrisprConstruct(
                            effector="dCas9-Mxi1",
                            guide_sequence="GTCAGGTACTCCGAATTCGA",
                            n_guides=1,
                        ),
                    ),
                ]
            ),
            environment=s.Environment(
                media=s.Media(name="YPD", state="solid", is_synthetic=False),
                temperature=s.Temperature(value=30.0),
            ),
            phenotype=s.FitnessPhenotype(fitness=0.82, fitness_std=0.05),
        )
    }


def _table() -> dict[str, str]:
    _, table = cell_adapter_surface(
        (REPO_ROOT / CELL_ADAPTER_RELPATH).read_text(encoding="utf-8")
    )
    return table


def _run(method_name: str, record: dict[str, Any]) -> list[Any]:
    """Call the chunked method a conf name maps to on one record; always a list."""
    fn = _undecorated(getattr(CellAdapter, _table()[method_name]))
    out = fn(_adapter(), record, method_name)
    return out if isinstance(out, list) else [out]


def _canonical(item: Any) -> list[Any]:
    if hasattr(item, "get_source_id"):
        return [
            "edge",
            item.get_source_id(),
            item.get_target_id(),
            item.get_label(),
            item.get_properties(),
        ]
    return [
        "node",
        item.get_id(),
        item.get_label(),
        item.get_preferred_id(),
        item.get_properties(),
    ]


def _digest(items: list[Any]) -> str:
    payload = json.dumps([_canonical(i) for i in items], sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


# ------------------------------------------------------------ yeast output unchanged
YEAST_PERTURBATION_NODES = [
    [
        "node",
        "b176a24b6b156aa2eaf78298acceda085ee2de202dfbb040b773bb8f67689e4a",
        "perturbation",
        "sga_kanmx_deletion",
        {
            "systematic_gene_name": "YAL001C",
            "perturbed_gene_name": "TFC3",
            "perturbation_type": "sga_kanmx_deletion",
            "description": "Deletion via KanMX or NatMX gene replacement",
            "strain_id": "YAL001C_dma1",
            "id": "b176a24b6b156aa2eaf78298acceda085ee2de202dfbb040b773bb8f67689e4a",
            "preferred_id": "sga_kanmx_deletion",
        },
    ],
    [
        "node",
        "9744bfd461ef780fafe2b71512319f17cdf6feec276f067f6d8070f3cfb26d70",
        "perturbation",
        "kanmx_deletion",
        {
            "systematic_gene_name": "YAL002W",
            "perturbed_gene_name": "VPS8",
            "perturbation_type": "kanmx_deletion",
            "description": "Deletion via KanMX or NatMX gene replacement",
            "strain_id": None,
            "id": "9744bfd461ef780fafe2b71512319f17cdf6feec276f067f6d8070f3cfb26d70",
            "preferred_id": "kanmx_deletion",
        },
    ],
    [
        "node",
        "03e1d657835ea58e04a81b7c4e752b3f5c0a2fca78bd45c379fa7cc9df8b4331",
        "perturbation",
        "crispr_interference",
        {
            "systematic_gene_name": "YAL003W",
            "perturbed_gene_name": "EFB1",
            "perturbation_type": "crispr_interference",
            "description": "CRISPR interference (decreased expression of a present gene)",
            "strain_id": None,
            "id": "03e1d657835ea58e04a81b7c4e752b3f5c0a2fca78bd45c379fa7cc9df8b4331",
            "preferred_id": "crispr_interference",
        },
    ],
]

# sha256 of ``_digest`` of each method's output on ``_yeast_record()``, computed with
# origin/main 76933585's CellAdapter (the branch reproduced all of them).
YEAST_GOLDEN_DIGESTS = {
    "genotype (chunked)": (
        "5f412c3d4e0cc79a1c2121d7b38996401404c8aef584b834199601e0ccdd4d73"
    ),
    "perturbation (chunked)": (
        "46eb790771da94e8f8f97b66efeab476735f62e2cf942b7361d4926ed56c22a3"
    ),
    "crispr construct (chunked)": (
        "0bc236c34fd3a054f8022092236388026ec9e772b555fbfb7ab9b32173f12727"
    ),
    "fitness phenotype (chunked)": (
        "dbade8c3c8ad945e439aa6688b306b27b64044bf7415df60a19c14d1f1c00f65"
    ),
    "genotype to experiment (chunked)": (
        "4fad9e3895e3d03a777ce4b7a82fb520665c1ed81933690c883a4466458bd40c"
    ),
    "perturbation to genotype (chunked)": (
        "15ab04c1633aac64a213c195cc0236dead004b933abdae1aaa55b059ae8cace3"
    ),
    "crispr construct to perturbation (chunked)": (
        "c61d8ab0f5b88b2678c2a25fc4daa28162e3cee9679e0b3683d2a747a4e70a3b"
    ),
    "phenotype to experiment (chunked)": (
        "cf4ab1db3a8bd8dbb33f9580e977655c8b4b63a5847f80c71faff2a241c908f6"
    ),
}


def test_the_yeast_perturbation_nodes_are_exactly_what_main_emits() -> None:
    """``perturbation (chunked)`` on a yeast record: ids, labels and properties as main."""
    nodes = _run("perturbation (chunked)", _yeast_record())
    assert [_canonical(n) for n in nodes] == YEAST_PERTURBATION_NODES


@pytest.mark.parametrize("method_name", sorted(YEAST_GOLDEN_DIGESTS))
def test_the_yeast_output_of_each_enabled_method_matches_main(method_name: str) -> None:
    """Every method a yeast CRISPR fitness dataset enables, digested, equals main's."""
    assert (
        _digest(_run(method_name, _yeast_record())) == YEAST_GOLDEN_DIGESTS[method_name]
    )


def test_a_yeast_record_emits_no_bacterial_perturbation_node() -> None:
    assert _run("bacterial perturbation (chunked)", _yeast_record()) == []


# ---------------------------------------------------------- bacterial perturbation
def test_bacterial_perturbation_nodes_carry_the_namespace_one_per_leaf() -> None:
    record = _product_titer_record(_titer())
    # Genotype sorts its perturbations by systematic_gene_name: 'Efa:mvaE' first.
    leaves = record["experiment"].genotype.perturbations
    nodes = _run("bacterial perturbation (chunked)", record)
    assert [n.get_label() for n in nodes] == ["bacterial perturbation"] * 5
    assert [n.get_id() for n in nodes] == [_sha(p) for p in leaves]
    assert [n.get_preferred_id() for n in nodes] == [
        "heterologous_pathway",
        "bacterial_deletion",
        "transposon_insertion",
        "bacterial_crispr_interference",
        "promoter_replacement",
    ]
    deletion = nodes[1].get_properties()
    assert deletion == {
        "systematic_gene_name": "b0002",
        "perturbed_gene_name": "thrA",
        "perturbation_type": "bacterial_deletion",
        "description": "Bacterial gene deletion, identified by a namespaced locus tag",
        "gene_namespace": MG1655,
        "id": _sha(leaves[1]),
        "preferred_id": "bacterial_deletion",
    }
    assert {n.get_properties()["gene_namespace"] for n in nodes} == {MG1655}
    assert "serialized_data" not in deletion


def test_the_bacterial_leaf_tuple_is_exactly_the_leaves_carrying_gene_namespace() -> (
    None
):
    """The two bacterial tuples PARTITION the genotype union's namespaced classes.

    A called-variant leaf carries ``gene_namespace`` too, so the set it belongs to is
    the union of the two tuples, and the tuples must be disjoint: a leaf served under
    both labels would be written twice (``BacterialSpanDeletionPerturbation`` is a
    subclass of ``BacterialDeletionPerturbation``, which is why the adapter excludes it
    by tuple rather than by isinstance alone).
    """
    (union,) = typing.get_args(s.Genotype.model_fields["perturbations"].annotation)
    namespaced = {
        cls for cls in typing.get_args(union) if "gene_namespace" in cls.model_fields
    }
    plain = set(BACTERIAL_PERTURBATION_LEAVES)
    variant = set(BACTERIAL_VARIANT_PERTURBATION_LEAVES)
    assert plain | variant == namespaced
    assert plain & variant == set()
    # 5 from PR #707 plus the round-2 marked-allele, degron and CRISPRa leaves
    # (#749, #792, #799)
    assert len(BACTERIAL_PERTURBATION_LEAVES) == 5 + 3
    assert len(BACTERIAL_VARIANT_PERTURBATION_LEAVES) == 3


# ------------------------------------------- bacterial sequence variant perturbation
def test_variant_nodes_carry_the_call_and_the_other_leaves_emit_none() -> None:
    """The #731 class: one node per called variant, with the call projected off .call."""
    record = _variant_record(_titer())
    leaves = record["experiment"].genotype.perturbations
    nodes = _run("bacterial sequence variant perturbation (chunked)", record)
    assert [n.get_label() for n in nodes] == [
        "bacterial sequence variant perturbation"
    ] * 3
    assert [n.get_id() for n in nodes] == [_sha(p) for p in leaves]
    # Genotype sorts by systematic_gene_name, so the site id sorts before the PP_ tags
    # and PP_3415 before PP_4062.
    assert [n.get_preferred_id() for n in nodes] == [
        "bacterial_site_variant",
        "bacterial_sequence_variant",
        "bacterial_span_deletion",
    ]
    assert nodes[0].get_properties() == {
        "systematic_gene_name": "AE015451:4586057",
        "perturbed_gene_name": "PP_4061, PP_4063",
        "perturbation_type": "bacterial_site_variant",
        "description": (
            "Called sequence variant keyed on its genomic site, because no locus tag "
            "of the pinned assembly holds it"
        ),
        "gene_namespace": KT2440,
        "reference_sequence": "AE015451",
        "position_start": 4586057,
        "position_end": 4586057,
        "variant_type": "insertion",
        "variant_frequency": 1.0,
        "call_mode": "clone",
        "id": _sha(leaves[0]),
        "preferred_id": "bacterial_site_variant",
    }
    assert nodes[1].get_properties()["systematic_gene_name"] == "PP_3415"
    assert nodes[1].get_properties()["variant_type"] == "snv"
    assert nodes[2].get_properties()["systematic_gene_name"] == "PP_4062"
    assert nodes[2].get_properties()["variant_type"] == "deletion"
    assert "serialized_data" not in nodes[0].get_properties()
    # the span deletion is NOT also written as a `bacterial perturbation`
    assert _run("bacterial perturbation (chunked)", record) == []
    assert (
        _run("bacterial sequence variant perturbation (chunked)", _yeast_record()) == []
    )


def test_variant_node_properties_are_exactly_the_declared_class_properties() -> None:
    """The emitted property set equals the schema config's, with no silent extra."""
    declared = set(SCHEMA["bacterial sequence variant perturbation"]["properties"])
    node = _run(
        "bacterial sequence variant perturbation (chunked)", _variant_record(_titer())
    )[0]
    assert set(node.get_properties()) - NODE_BOOKKEEPING == declared
    assert SCHEMA["bacterial sequence variant perturbation"]["is_a"] == "genotype"


def test_a_variant_record_edges_address_the_variant_nodes() -> None:
    """``perturbation to genotype`` computes the same ids, so no edge method is added."""
    record = _variant_record(_titer())
    node_ids = [
        n.get_id()
        for n in _run("bacterial sequence variant perturbation (chunked)", record)
    ]
    genotype_id = _run("genotype (chunked)", record)[0].get_id()
    edges = _run("perturbation to genotype (chunked)", record)
    assert [e.get_label() for e in edges] == ["perturbation member of"] * 3
    assert [e.get_source_id() for e in edges] == node_ids
    assert {e.get_target_id() for e in edges} == {genotype_id}


def test_a_frequency_the_release_wrote_as_a_range_is_a_null_column() -> None:
    """A range has no single number, so the node's frequency is null and the cell stays.

    de Siqueira 2025 writes ``61 -> 63`` in ``Variant Frequency`` for a multi-base call;
    the verbatim cell travels in the Experiment blob and the queryable column is null,
    which is what distinguishes it from a frequency of zero.
    """
    ranged = _call(frequency_statement="61 -> 63", frequency=None, frequency_basis=None)
    leaf = s.BacterialSequenceVariantPerturbation(
        systematic_gene_name="PP_3415",
        perturbed_gene_name="PP_3415",
        gene_namespace=KT2440,
        call=ranged,
    )
    node = CellAdapter._bacterial_variant_perturbation_node_from(leaf)
    assert node.get_properties()["variant_frequency"] is None
    assert leaf.call.frequency_statement == "61 -> 63"


def test_the_served_edge_methods_already_connect_the_bacterial_nodes() -> None:
    """``perturbation member of`` and ``crispr construct member of`` hit the new ids."""
    record = _product_titer_record(_titer())
    node_ids = [n.get_id() for n in _run("bacterial perturbation (chunked)", record)]
    genotype_id = _run("genotype (chunked)", record)[0].get_id()

    edges = _run("perturbation to genotype (chunked)", record)
    assert [e.get_label() for e in edges] == ["perturbation member of"] * 5
    assert [e.get_source_id() for e in edges] == node_ids
    assert {e.get_target_id() for e in edges} == {genotype_id}

    crispri = next(
        p
        for p in record["experiment"].genotype.perturbations
        if isinstance(p, s.BacterialCrisprInterferencePerturbation)
    )
    assert _sha(crispri) in node_ids
    construct_edges = _run("crispr construct to perturbation (chunked)", record)
    assert [e.get_target_id() for e in construct_edges] == [_sha(crispri)]
    assert [e.get_source_id() for e in construct_edges] == [_sha(_construct())]


# ------------------------------------------------------------------- phenotypes
def test_product_titer_phenotype_node_projects_every_field() -> None:
    phenotype = _full_titer()
    record = _product_titer_record(phenotype)
    [node] = _run("product titer phenotype (chunked)", record)
    pid = _sha(phenotype)
    assert node.get_id() == pid
    assert node.get_label() == "product titer phenotype"
    assert node.get_preferred_id() == f"phenotype_{pid}"
    props = node.get_properties()
    assert props == {
        "graph_level": "global",
        "label_name": "titer",
        "label_statistic_name": "titer_se",
        "product": json.dumps(phenotype.product.model_dump()),
        "titer": 2.4,
        "titer_unit": "g/L",
        "titer_se": pytest.approx(0.1),
        "titer_uncertainty": 0.3,
        "titer_uncertainty_type": "sample_sd",
        "n_samples": 9,
        "sample_unit": "biological_replicate",
        "product_yield": 0.12,
        "product_yield_unit": "g/g_substrate",
        "productivity": 0.05,
        "productivity_unit": "g/L/h",
        "quantification_method": "GC-MS",
        "id": pid,
        "preferred_id": f"phenotype_{pid}",
    }
    # the product's identity survives the JSON projection, InChIKey included
    assert json.loads(props["product"])["inchikey"] == "XHQZJYCNDZAGLW-UHFFFAOYSA-N"
    # the experiment edge addresses this node
    [edge] = _run("phenotype to experiment (chunked)", record)
    assert edge.get_source_id() == pid
    assert edge.get_label() == "phenotype member of"


def test_product_titer_optional_fields_project_as_none() -> None:
    props = _run("product titer phenotype (chunked)", _product_titer_record(_titer()))[
        0
    ].get_properties()
    for key in (
        "titer_se",
        "titer_uncertainty",
        "titer_uncertainty_type",
        "n_samples",
        "sample_unit",
        "product_yield",
        "product_yield_unit",
        "productivity",
        "productivity_unit",
        "quantification_method",
    ):
        assert props[key] is None, key


def _phenotype_record(phenotype: Any) -> dict[str, Any]:
    """A record carrying only the phenotype (the phenotype methods read nothing else)."""
    return {"experiment": SimpleNamespace(phenotype=phenotype)}


def test_protein_turnover_phenotype_node_projects_dicts_as_json() -> None:
    phenotype = _turnover(
        degradation_rate_se={"b0002": 0.01},
        half_life={"b0002": 6.3},
        synthesis_rate={"b0003": 0.2},
    )
    [node] = _run("protein turnover phenotype (chunked)", _phenotype_record(phenotype))
    pid = _sha(phenotype)
    assert (node.get_id(), node.get_label(), node.get_preferred_id()) == (
        pid,
        "protein turnover phenotype",
        f"phenotype_{pid}",
    )
    assert node.get_properties() == {
        "graph_level": "node",
        "label_name": "degradation_rate",
        "label_statistic_name": "degradation_rate_se",
        "degradation_rate": '{"b0002": 0.11, "b0003": 0.07}',
        "degradation_rate_se": '{"b0002": 0.01}',
        "half_life": '{"b0002": 6.3}',
        "synthesis_rate": '{"b0003": 0.2}',
        "n_replicates": '{"b0002": 3, "b0003": 2}',
        "measurement_type": "pulse_silac_degradation_rate_per_hour",
        "id": pid,
        "preferred_id": f"phenotype_{pid}",
    }
    bare = _run("protein turnover phenotype (chunked)", _phenotype_record(_turnover()))
    props = bare[0].get_properties()
    assert (
        props["degradation_rate_se"],
        props["half_life"],
        props["synthesis_rate"],
    ) == (None, None, None)


def test_flux_phenotype_node_keeps_the_sign_and_the_interval() -> None:
    phenotype = _flux(
        net_flux_lower={"PGI": -1.5},
        net_flux_upper={"PGI": -0.9},
        confidence_level=0.95,
        n_samples=2,
        sample_unit=s.SampleUnit.biological_replicate,
        target_reaction_ids={"PGI": "PGI"},
    )
    [node] = _run("flux phenotype (chunked)", _phenotype_record(phenotype))
    pid = _sha(phenotype)
    assert node.get_label() == "flux phenotype"
    assert node.get_properties() == {
        "graph_level": "metabolism",
        "label_name": "net_flux",
        "label_statistic_name": None,
        "net_flux": '{"PGI": -1.2, "TPI": 0.4}',
        "net_flux_lower": '{"PGI": -1.5}',
        "net_flux_upper": '{"PGI": -0.9}',
        "confidence_level": 0.95,
        "measurement_type": "c13_mfa_net_flux_mmol_per_gdcw_per_h",
        "n_samples": 2,
        "sample_unit": "biological_replicate",
        "target_reaction_ids": '{"PGI": "PGI"}',
        "id": pid,
        "preferred_id": f"phenotype_{pid}",
    }
    bare = _run("flux phenotype (chunked)", _phenotype_record(_flux()))[0]
    optional = (
        "net_flux_lower",
        "net_flux_upper",
        "confidence_level",
        "n_samples",
        "sample_unit",
        "target_reaction_ids",
    )
    assert [bare.get_properties()[k] for k in optional] == [None] * len(optional)


def test_promoter_activity_phenotype_node_projects_every_field_as_a_scalar() -> None:
    """One promoter per record, so nothing is a dict and nothing is JSON-encoded."""
    phenotype = _activity(
        promoter_activity_uncertainty=0.4,
        promoter_activity_uncertainty_type="sample_sd",
        n_samples=4,
    )
    [node] = _run("promoter activity phenotype (chunked)", _phenotype_record(phenotype))
    pid = _sha(phenotype)
    assert (node.get_id(), node.get_label(), node.get_preferred_id()) == (
        pid,
        "promoter activity phenotype",
        f"phenotype_{pid}",
    )
    assert node.get_properties() == {
        "graph_level": "node",
        "label_name": "promoter_activity",
        "label_statistic_name": "promoter_activity_se",
        "promoter_activity": 12.98,
        "promoter_activity_se": pytest.approx(0.2),
        "promoter_activity_uncertainty": 0.4,
        "promoter_activity_uncertainty_type": "sample_sd",
        "n_samples": 4,
        "sample_unit": "biological_replicate",
        "promoter_name": "thrA",
        "promoter_gene": "b0002",
        "readout": "plate_reader_fluorescence",
        "reporter_gene": "gfp",
        "activity_units": "GFP fluorescence in the reader's own units",
        "well_id": "Untreated|AZ01|A1",
        "id": pid,
        "preferred_id": f"phenotype_{pid}",
    }
    [edge] = _run(
        "phenotype to experiment (chunked)", _promoter_activity_record(phenotype)
    )
    assert edge.get_source_id() == pid
    assert edge.get_label() == "phenotype member of"


def test_promoter_activity_optional_fields_project_as_none() -> None:
    """A screen run once has no dispersion, and a label naming no gene has no join key."""
    props = _run(
        "promoter activity phenotype (chunked)",
        _phenotype_record(_activity(promoter_gene=None)),
    )[0].get_properties()
    optional = (
        "promoter_activity_se",
        "promoter_activity_uncertainty",
        "promoter_activity_uncertainty_type",
        "promoter_gene",
    )
    assert [props[k] for k in optional] == [None] * len(optional)
    assert props["sample_unit"] == "biological_replicate"


def _morphology_record(phenotype: s.BacterialMorphologyPhenotype) -> dict[str, Any]:
    return {
        "experiment": s.BacterialMorphologyExperiment(
            dataset_name="BacterialToy",
            genotype=s.Genotype(perturbations=_bacterial_leaves()),
            environment=s.Environment(media=_lb()),
            phenotype=phenotype,
        )
    }


def _morphology(**kw: Any) -> s.BacterialMorphologyPhenotype:
    fields: dict[str, Any] = dict(
        assay="campos2018",
        morphology={"<L>": 2.81, "%2N": 0.19},
        morphology_coefficient_of_variation={"CV_L": 0.24},
        n_samples=245,
        sample_unit=s.SampleUnit.cell,
    )
    fields.update(kw)
    return s.BacterialMorphologyPhenotype(**fields)


def test_bacterial_morphology_phenotype_node_projects_both_dicts_as_json() -> None:
    """``assay`` rides beside them: the keys inside are only readable against it."""
    phenotype = _morphology()
    [node] = _run(
        "bacterial morphology phenotype (chunked)", _phenotype_record(phenotype)
    )
    pid = _sha(phenotype)
    assert (node.get_id(), node.get_label(), node.get_preferred_id()) == (
        pid,
        "bacterial morphology phenotype",
        f"phenotype_{pid}",
    )
    assert node.get_properties() == {
        "graph_level": "global",
        "label_name": "morphology",
        "label_statistic_name": "morphology_coefficient_of_variation",
        "assay": "campos2018",
        "morphology": '{"<L>": 2.81, "%2N": 0.19}',
        "morphology_coefficient_of_variation": '{"CV_L": 0.24}',
        "n_samples": 245,
        "sample_unit": "cell",
        "id": pid,
        "preferred_id": f"phenotype_{pid}",
    }
    [edge] = _run("phenotype to experiment (chunked)", _morphology_record(phenotype))
    assert edge.get_source_id() == pid
    assert edge.get_label() == "phenotype member of"


def test_bacterial_morphology_optional_fields_project_as_none() -> None:
    """A strain whose assay determined no CV, and a release stating no cell count."""
    props = _run(
        "bacterial morphology phenotype (chunked)",
        _phenotype_record(
            _morphology(
                morphology_coefficient_of_variation=None,
                n_samples=None,
                sample_unit=None,
            )
        ),
    )[0].get_properties()
    optional = ("morphology_coefficient_of_variation", "n_samples", "sample_unit")
    assert [props[k] for k in optional] == [None] * len(optional)
    assert props["morphology"] == '{"<L>": 2.81, "%2N": 0.19}'


@pytest.mark.parametrize(
    "method_name,label,phenotypes",
    [
        (
            "product titer phenotype reference",
            "product titer phenotype",
            [_titer(), _titer(), _titer(titer=0.0)],
        ),
        (
            "protein turnover phenotype reference",
            "protein turnover phenotype",
            [_turnover(), _turnover(), _turnover(measurement_type="other_per_hour")],
        ),
        (
            "flux phenotype reference",
            "flux phenotype",
            [_flux(), _flux(), _flux(net_flux={"PGI": 1.0})],
        ),
        (
            "promoter activity phenotype reference",
            "promoter activity phenotype",
            [_activity(), _activity(), _activity(promoter_activity=9.5)],
        ),
        (
            "bacterial morphology phenotype reference",
            "bacterial morphology phenotype",
            [
                _morphology(),
                _morphology(),
                _morphology(morphology={"<L>": 3.02, "%2N": 0.19}),
            ],
        ),
    ],
)
def test_reference_phenotype_nodes_are_deduplicated_and_reach_their_edge(
    method_name: str, label: str, phenotypes: list[Any]
) -> None:
    """Three references, two equal: two nodes, each addressed by its reference edge."""
    adapter = _adapter()
    references = [
        SimpleNamespace(
            reference=SimpleNamespace(
                phenotype_reference=p, model_dump=lambda i=i: {"r": i}
            )
        )
        for i, p in enumerate(phenotypes)
    ]
    adapter.dataset = cast(Any, SimpleNamespace(experiment_reference_index=references))
    nodes = getattr(adapter, _table()[method_name])()
    assert [n.get_label() for n in nodes] == [label, label]
    assert [n.get_preferred_id() for n in nodes] == [label, label]
    assert [n.get_id() for n in nodes] == [_sha(phenotypes[0]), _sha(phenotypes[2])]
    edges = adapter._get_phenotype_to_experiment_reference_edges()
    assert {e.get_source_id() for e in edges} == {n.get_id() for n in nodes}


# ----------------------------------------------------------- schema <-> adapter
NEW_CLASSES = {
    "bacterial perturbation": ("bacterial perturbation (chunked)", None),
    "product titer phenotype": (
        "product titer phenotype (chunked)",
        s.ProductTiterPhenotype,
    ),
    "protein turnover phenotype": (
        "protein turnover phenotype (chunked)",
        s.ProteinTurnoverPhenotype,
    ),
    "flux phenotype": ("flux phenotype (chunked)", s.FluxPhenotype),
    "promoter activity phenotype": (
        "promoter activity phenotype (chunked)",
        s.PromoterActivityPhenotype,
    ),
    "bacterial morphology phenotype": (
        "bacterial morphology phenotype (chunked)",
        s.BacterialMorphologyPhenotype,
    ),
}


def test_the_new_phenotype_classes_declare_every_field_but_provenance_gaps() -> None:
    for label, (_, model) in NEW_CLASSES.items():
        if model is None:
            continue
        assert SCHEMA[label]["is_a"] == "phenotypic feature"
        assert set(SCHEMA[label]["properties"]) == set(model.model_fields) - {
            "provenance_gaps"
        }, label


def test_bacterial_perturbation_is_a_genotype_sibling_of_perturbation() -> None:
    """Not ``is_a: perturbation``: a served ``MATCH (:Perturbation)`` must not grow."""
    assert SCHEMA["bacterial perturbation"]["is_a"] == "genotype"
    assert SCHEMA["perturbation"]["is_a"] == "genotype"
    assert set(SCHEMA["bacterial perturbation"]["properties"]) == {
        "systematic_gene_name",
        "perturbed_gene_name",
        "perturbation_type",
        "description",
        "gene_namespace",
    }


def test_emitted_properties_equal_the_declared_properties_at_run_time() -> None:
    """The static coherence check, repeated on real emitted nodes."""
    record = _product_titer_record(_full_titer())
    emitted = {
        "bacterial perturbation": _run("bacterial perturbation (chunked)", record)[0],
        "product titer phenotype": _run("product titer phenotype (chunked)", record)[0],
        "protein turnover phenotype": _run(
            "protein turnover phenotype (chunked)", _phenotype_record(_turnover())
        )[0],
        "flux phenotype": _run("flux phenotype (chunked)", _phenotype_record(_flux()))[
            0
        ],
    }
    for label, node in emitted.items():
        assert node.get_label() == label
        assert set(node.get_properties()) - NODE_BOOKKEEPING == set(
            SCHEMA[label]["properties"]
        ), label


def test_the_new_methods_are_registered_and_the_served_edges_name_the_new_classes() -> (
    None
):
    table = _table()
    assert table["bacterial perturbation (chunked)"] == "_bacterial_perturbation_node"
    assert (
        table["bacterial sequence variant perturbation (chunked)"]
        == "_bacterial_variant_perturbation_node"
    )
    for name, fn in {
        "product titer phenotype (chunked)": "_product_titer_phenotype_node",
        "protein turnover phenotype (chunked)": "_protein_turnover_phenotype_node",
        "flux phenotype (chunked)": "_flux_phenotype_node",
        "product titer phenotype reference": (
            "_get_product_titer_phenotype_reference_nodes"
        ),
        "protein turnover phenotype reference": (
            "_get_protein_turnover_phenotype_reference_nodes"
        ),
        "flux phenotype reference": "_get_flux_phenotype_reference_nodes",
        "promoter activity phenotype (chunked)": ("_promoter_activity_phenotype_node"),
        "promoter activity phenotype reference": (
            "_get_promoter_activity_phenotype_reference_nodes"
        ),
        "bacterial morphology phenotype (chunked)": (
            "_bacterial_morphology_phenotype_node"
        ),
        "bacterial morphology phenotype reference": (
            "_get_bacterial_morphology_phenotype_reference_nodes"
        ),
    }.items():
        assert table[name] == fn
    assert SCHEMA["perturbation member of"]["source"] == [
        "perturbation",
        "bacterial perturbation",
        "bacterial sequence variant perturbation",
    ]
    assert SCHEMA["perturbation member of"]["input_label"] == "perturbation member of"
    assert SCHEMA["perturbation member of"]["target"] == "genotype"
    sources = SCHEMA["phenotype member of"]["source"]
    assert sources[-5:] == [
        "product titer phenotype",
        "protein turnover phenotype",
        "flux phenotype",
        "promoter activity phenotype",
        "bacterial morphology phenotype",
    ]


# --------------------------------------------------------------------------- #
# phage perturbation: the environment axis of a phage-resistance screen
# --------------------------------------------------------------------------- #
def _phage(**kw: Any) -> s.PhagePerturbation:
    fields: dict[str, Any] = dict(
        name="lambda cI857",
        family="Siphoviridae",
        genome_type="dsDNA",
        ncbi_taxid=10710,
        genome_accession="J02459.1",
        multiplicity_of_infection=484.375,
        titer_pfu_per_ml=1.24e10,
        host_of_propagation="E. coli K-12",
    )
    fields.update(kw)
    return s.PhagePerturbation(**fields)


def _phage_challenge_record(*perturbations: Any) -> dict[str, Any]:
    """A bacterial environment-response record whose environment carries ``perturbations``."""
    return {
        "experiment": s.BacterialEnvironmentResponseExperiment(
            dataset_name="BacterialToy",
            genotype=s.Genotype(
                perturbations=[
                    s.TransposonInsertionPerturbation(
                        systematic_gene_name="BW25113_0002",
                        perturbed_gene_name="thrA",
                        gene_namespace="ecoli_k12_bw25113_locus_tag",
                        barcode="ACGTACGTACGTACGTACGT",
                    )
                ]
            ),
            environment=s.Environment(media=_lb(), perturbations=list(perturbations)),
            phenotype=s.EnvironmentResponsePhenotype(
                measurement_type=s.MeasurementType.log2_ratio,
                assay_type=s.AssayType.pooled_competitive_growth_barcode,
                environment_response=-3.2,
                units="log2(treatment/control)",
            ),
        )
    }


def test_a_phage_node_projects_the_phage_and_its_dose() -> None:
    """The class exists so the MOI, the taxon and the accession are columns.

    They could not be properties of the served ``environment perturbation`` class
    without a full rebuild, and the dose is not a concentration, so it cannot ride that
    class's ``concentration_value`` / ``concentration_unit`` columns.
    """
    from torchcell.datamodels.identity import (
        environment_perturbation_identity,
        identity_sha256,
    )

    phage = _phage()
    record = _phage_challenge_record(phage)
    nodes = _run("phage perturbation (chunked)", record)
    assert [n.get_label() for n in nodes] == ["phage perturbation"]
    node = nodes[0]
    assert node.get_preferred_id() == "phage"
    # the id is the composition projection every environment-side node uses, so the
    # served `environment perturbation member of` edge addresses it unchanged
    assert node.get_id() == identity_sha256(environment_perturbation_identity(phage))
    assert node.get_properties() == {
        "id": node.get_id(),
        "preferred_id": "phage",
        "perturbation_type": "phage",
        "description": "Bacteriophage added to the culture",
        "phage_name": "lambda cI857",
        "ncbi_taxid": 10710,
        "genome_accession": "J02459.1",
        "multiplicity_of_infection": 484.375,
        "titer_pfu_per_ml": 1.24e10,
    }
    assert set(node.get_properties()) - NODE_BOOKKEEPING == set(
        SCHEMA["phage perturbation"]["properties"]
    )


def test_two_doses_of_one_phage_are_two_nodes_and_two_phages_are_two_nodes() -> None:
    """The dose is identity, which is what keeps 68 challenges from collapsing to one."""
    low = _phage(multiplicity_of_infection=0.01875)
    high = _phage(multiplicity_of_infection=1.875)
    other = _phage(name="T4", ncbi_taxid=10665, genome_accession="AF158101.6")
    ids = {
        _run("phage perturbation (chunked)", _phage_challenge_record(p))[0].get_id()
        for p in (low, high, other)
    }
    assert len(ids) == 3


def test_the_phage_method_emits_nothing_for_any_other_perturbation() -> None:
    """A small molecule or a physical factor is the served class's, not this one's."""
    record = _phage_challenge_record(
        s.SmallMoleculePerturbation(
            compound=s.Compound(name="kanamycin"),
            concentration=s.Concentration(
                value=50.0, unit=s.ConcentrationUnit.ug_per_ml
            ),
        ),
        s.EnvironmentPhysicalPerturbation(factor=s.PhysicalFactor.ph),
    )
    assert _run("phage perturbation (chunked)", record) == []
    # and the served method still emits both of them, unchanged
    served = _run("environment perturbation (chunked)", record)
    assert [n.get_label() for n in served] == ["environment perturbation"] * 2


def test_the_phage_reference_collector_deduplicates_by_content_id() -> None:
    """One node per distinct phage challenge across the reference index."""
    phage = _phage()
    reference = s.BacterialEnvironmentResponseExperimentReference(
        dataset_name="BacterialToy",
        genome_reference=s.AssemblyReferenceGenome(
            species="Escherichia coli",
            strain="BW25113",
            assembly_set="ecoli_K12_BW25113_ASM75055v1",
            assembly_accession="GCA_000750555.1",
        ),
        environment_reference=s.Environment(media=_lb(), perturbations=[phage]),
        phenotype_reference=s.EnvironmentResponsePhenotype(
            measurement_type=s.MeasurementType.log2_ratio,
            assay_type=s.AssayType.pooled_competitive_growth_barcode,
            environment_response=0.0,
            units="log2(treatment/control)",
        ),
    )
    no_phage = reference.model_copy(
        update={"environment_reference": s.Environment(media=_lb())}
    )
    adapter = _adapter()
    adapter.dataset = cast(
        Any,
        SimpleNamespace(
            experiment_reference_index=[
                SimpleNamespace(reference=reference),
                SimpleNamespace(reference=reference),
                SimpleNamespace(reference=no_phage),
            ]
        ),
    )
    nodes = CellAdapter._get_phage_perturbation_reference_nodes(adapter)
    assert [n.get_label() for n in nodes] == ["phage perturbation"]
    assert nodes[0].get_properties()["phage_name"] == "lambda cI857"


def test_the_phage_method_is_registered_under_its_conf_name() -> None:
    table = _table()
    assert table["phage perturbation (chunked)"] == "_phage_perturbation_node"
    assert table["phage perturbation reference"] == (
        "_get_phage_perturbation_reference_nodes"
    )
