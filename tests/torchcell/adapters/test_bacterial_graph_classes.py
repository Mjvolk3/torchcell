# tests/torchcell/adapters/test_bacterial_graph_classes.py
# [[tests.torchcell.adapters.test_bacterial_graph_classes]]
"""The graph classes the bacterial schema needs, and the yeast output beside them.

``product titer phenotype``, ``protein turnover phenotype``, ``flux phenotype`` and
``promoter activity phenotype`` are node classes emitted by their own ``CellAdapter``
methods. The perturbation axis has ONE class for every host since the ontology
unification (G, 2026-10-15): ``perturbation`` carries ``host_genome`` and ``sequence
variant perturbation`` (is_a perturbation) carries the call, both emitted by the one
``_perturbation_node`` with the label chosen by leaf; the former ``bacterial
perturbation`` and ``bacterial sequence variant perturbation`` classes are retired.
Three things are pinned here:

- Each new method emits the class it declares, with the id rule every sub-object node
  uses (sha256 of the json-dumped ``model_dump``), the declared property set exactly, and
  the values projected the way the existing families project them (an enum as its value,
  a dict and the product ``Compound`` as a JSON string).
- The served edge methods already address the new nodes: ``perturbation member of``,
  ``crispr construct member of`` and ``phenotype member of`` compute the same ids, so no
  edge method is added.
- A yeast record's output is pinned: the expected ``perturbation`` nodes are written out
  in full below, and the other methods a yeast dataset enables are pinned by the sha256
  of their canonical output (the record and the digest rule are defined in this file).
  The union-to-label map of the one perturbation method is a bijection over
  ``GenePerturbationType``.

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
    HOST_GENOME_OF_NAMESPACE,
    VARIANT_CALL_LEAVES,
    CellAdapter,
)
from torchcell.datamodels import schema as s
from torchcell.knowledge_graphs.kg_manifest import (
    CELL_ADAPTER_RELPATH,
    SCHEMA_CONFIG_RELPATH,
    cell_adapter_surface,
)
from torchcell.verification.sourced import ProvenanceGap, ProvenanceGapReason

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
        s.DeletionStrainPerturbation(
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
        s.CrisprInterferencePerturbation(
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
        s.LocusVariantCallPerturbation(
            systematic_gene_name="PP_3415",
            perturbed_gene_name="PP_3415",
            gene_namespace=KT2440,
            call=_call(),
        ),
        s.SiteVariantCallPerturbation(
            systematic_gene_name=s.SiteVariantCallPerturbation.site_id(intergenic),
            perturbed_gene_name="PP_4061, PP_4063",
            gene_namespace=KT2440,
            call=intergenic,
            site_kind=s.VariantSiteKind.intergenic,
            flanking_systematic_gene_names=("PP_4061", "PP_4063"),
            flanking_gene_statement="PP_4061, PP_4063",
        ),
        s.SpanDeletionCallPerturbation(
            systematic_gene_name="PP_4062",
            perturbed_gene_name="PP_4062",
            gene_namespace=KT2440,
            call=span,
            span_designation=s.SpanDeletionCallPerturbation.designation(span),
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
        titer_censoring=s.Censoring.uncensored,
    )


def _turnover(**kw: Any) -> s.ProteinTurnoverPhenotype:
    fields: dict[str, Any] = dict(
        degradation_rate={"b0002": 0.11, "b0003": 0.07},
        n_replicates={"b0002": 3, "b0003": 2},
        measurement_type="pulse_silac_degradation_rate_per_hour",
    )
    fields.update(kw)
    return s.ProteinTurnoverPhenotype(**fields)


def _synthesis(**kw: Any) -> s.ProteinSynthesisRatePhenotype:
    fields: dict[str, Any] = dict(
        synthesis_rate={"b0002": 120.0, "b0003": 0.0},
        rate_unit=s.SynthesisRateUnit.molecules_per_generation,
        generation_time_minutes=21.5,
        measurement_type="ribosome_profiling_footprint_density",
    )
    fields.update(kw)
    return s.ProteinSynthesisRatePhenotype(**fields)


# #921: two released Potel 2018 Table 2 sites, one on one protein and one the release
# lists on two genes under one accession (EF-Tu, tufA;tufB).
_GND_S86 = s.Phosphosite(
    protein_ids=["P00350"],
    protein_id_namespace=s.ProteinIdNamespace.uniprotkb_accession,
    gene_names=["gnd"],
    positions=[86],
    residue=s.PhosphoResidue.serine,
    sequence_windows=["LLMVKAGAGTDAAIDSLKPYLDKGDIIIDGG"],
    localization_probability=0.999615,
    posterior_error_probability=4.27602e-05,
    identification_score=66.401,
    delta_score=58.487,
    diagnostic_peak=False,
)
_TUF_H119 = s.Phosphosite(
    protein_ids=["P0CE47"],
    protein_id_namespace=s.ProteinIdNamespace.uniprotkb_accession,
    gene_names=["tufA", "tufB"],
    positions=[119, 119],
    residue=s.PhosphoResidue.histidine,
    sequence_windows=["LVVAATDGPMPQTREHILLGRQVGVPYIIVF"],
    localization_probability=0.5,
    posterior_error_probability=0.001,
    identification_score=80.0,
    delta_score=20.0,
    diagnostic_peak=True,
)


def _phosphosite(**kw: Any) -> s.PhosphositeIntensityPhenotype:
    fields: dict[str, Any] = dict(
        phosphosite_intensity={"P00350|S|86": 2714000.0},
        quantification={
            "P00350|S|86": s.PhosphositeQuantification.quantified,
            "P0CE47|H|119;119": s.PhosphositeQuantification.not_quantified,
        },
        sites={"P00350|S|86": _GND_S86, "P0CE47|H|119;119": _TUF_H119},
        measurement_type="fe3_imac_lc_msms_maxquant_summed_xic",
        sample_label="Intensity glucose_exp1",
        replicate=1,
        replicate_unit=s.SampleUnit.technical_replicate,
        harvest_growth_phase=s.CultureGrowthPhase.exponential,
        harvest_optical_density=0.6,
    )
    fields.update(kw)
    return s.PhosphositeIntensityPhenotype(**fields)


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
        "5abd40e7968f1001ace38a9419e24a82e6a00d4262a839e6f4d939fdbf49640e",
        "perturbation",
        "sga_kanmx_deletion",
        {
            "systematic_gene_name": "YAL001C",
            "perturbed_gene_name": "TFC3",
            "perturbation_type": "sga_kanmx_deletion",
            "description": "Deletion via KanMX or NatMX gene replacement",
            "host_genome": "sgd_S288C_R64-4-1_20230830",
            "strain_id": "YAL001C_dma1",
            "id": "5abd40e7968f1001ace38a9419e24a82e6a00d4262a839e6f4d939fdbf49640e",
            "preferred_id": "sga_kanmx_deletion",
        },
    ],
    [
        "node",
        "16eef850df53a4163fd496054cd2b6369f3462eea93fed9b8283c189e5c0cb56",
        "perturbation",
        "kanmx_deletion",
        {
            "systematic_gene_name": "YAL002W",
            "perturbed_gene_name": "VPS8",
            "perturbation_type": "kanmx_deletion",
            "description": "Deletion via KanMX or NatMX gene replacement",
            "host_genome": "sgd_S288C_R64-4-1_20230830",
            "strain_id": None,
            "id": "16eef850df53a4163fd496054cd2b6369f3462eea93fed9b8283c189e5c0cb56",
            "preferred_id": "kanmx_deletion",
        },
    ],
    [
        "node",
        "fc2d8c98d81112f9386edb636aee9fc6d558c1f2be10c4a24fa11eeeab73e211",
        "perturbation",
        "crispr_interference",
        {
            "systematic_gene_name": "YAL003W",
            "perturbed_gene_name": "EFB1",
            "perturbation_type": "crispr_interference",
            "description": "CRISPR interference (decreased expression of a present gene)",
            "host_genome": "sgd_S288C_R64-4-1_20230830",
            "strain_id": None,
            "id": "fc2d8c98d81112f9386edb636aee9fc6d558c1f2be10c4a24fa11eeeab73e211",
            "preferred_id": "crispr_interference",
        },
    ],
]

# sha256 of ``_digest`` of each method's output on ``_yeast_record()``, computed with
# origin/main 76933585's CellAdapter (the branch reproduced all of them). The two
# experiment-edge digests moved in the #753 wave and nowhere else: an experiment's id is
# the sha256 of its fully inlined record, so ``Environment``'s new
# ``dilution_rate_per_hour`` field changes it for every record, which is exactly the
# BREAKING verdict ``scripts/schema_impact_check.py`` reports for all 81 datasets. The
# six node digests are untouched, which is the evidence the change did not reach the
# genotype, perturbation, construct or phenotype nodes. They moved again, and only
# they, when ``Media`` gained ``provenance_gaps`` (#901, #902, #903): every inlined
# medium now dumps an empty gap list, so every experiment id changes, the BREAKING
# verdict the schema-impact check reports for every dataset. Under P1 (ontology-unify,
# 2026-10-15) every node and edge digest moved at once: ``gene_namespace`` and
# ``identifier_mapping`` are fields of ``GenePerturbation``, so every perturbation's
# ``model_dump`` (and therefore its sha256 id, the genotype's and the experiment's)
# changed, which is the full rebuild the P1 record names. The crispr construct and the
# fitness phenotype digests did not move: the change did not reach them. Under G (the
# graph stage of the same unification) only ``perturbation (chunked)`` moved again: the
# node gained ``host_genome``, and no id changed.
YEAST_GOLDEN_DIGESTS = {
    "genotype (chunked)": (
        "64f38fddb654ba1b1e3f14b536d9f1db3481488839ef9340eba60b89dcce49c7"
    ),
    "perturbation (chunked)": (
        "a7783bfc2db824171bd176f478e39c62b6c02dab2e9fdae626958696440bafa3"
    ),
    "crispr construct (chunked)": (
        "0bc236c34fd3a054f8022092236388026ec9e772b555fbfb7ab9b32173f12727"
    ),
    "fitness phenotype (chunked)": (
        "dbade8c3c8ad945e439aa6688b306b27b64044bf7415df60a19c14d1f1c00f65"
    ),
    "genotype to experiment (chunked)": (
        "07b1efadc41c94458057b423e0f929a55bd0976d92ee1fbd8bad8f8238903a2a"
    ),
    "perturbation to genotype (chunked)": (
        "403e492ebae018a7351792080971f0f8c800654ee6fbecdfb7bf710c983e7feb"
    ),
    "crispr construct to perturbation (chunked)": (
        "b058455ac3fe53b595301dcfeb37495982c239d6f7959fffa291021992aa0a21"
    ),
    "phenotype to experiment (chunked)": (
        "eabfefd287a4f84d72db5b467bdf4f450ac7cc760b1d928e8fe4c9db7cc25aa9"
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


# --------------------------------------------- one perturbation class, every host
def test_a_bacterial_record_emits_one_perturbation_node_per_leaf_with_its_host() -> (
    None
):
    """Every bacterial leaf is a ``perturbation`` node under the registry's set id."""
    record = _product_titer_record(_titer())
    # Genotype sorts its perturbations by systematic_gene_name: 'Efa:mvaE' first.
    leaves = record["experiment"].genotype.perturbations
    nodes = _run("perturbation (chunked)", record)
    assert [n.get_label() for n in nodes] == ["perturbation"] * 5
    assert [n.get_id() for n in nodes] == [_sha(p) for p in leaves]
    assert [n.get_preferred_id() for n in nodes] == [
        "heterologous_pathway",
        "deletion_strain",
        "transposon_insertion",
        "crispr_interference",
        "promoter_replacement",
    ]
    deletion = nodes[1].get_properties()
    assert deletion == {
        "systematic_gene_name": "b0002",
        "perturbed_gene_name": "thrA",
        "perturbation_type": "deletion_strain",
        "description": (
            "Gene deletion strain, with its collection, cassette and construction as "
            "stated"
        ),
        "host_genome": "ecoli_K12_MG1655_ASM584v2",
        "strain_id": None,
        "id": _sha(leaves[1]),
        "preferred_id": "deletion_strain",
    }
    assert {n.get_properties()["host_genome"] for n in nodes} == {
        "ecoli_K12_MG1655_ASM584v2"
    }
    assert "serialized_data" not in deletion
    assert "gene_namespace" not in deletion


def test_host_genome_is_a_bijection_from_the_namespace_onto_registry_set_ids() -> None:
    """``HOST_GENOME_OF_NAMESPACE`` maps every ``GeneIdNamespace`` member, one to one, onto
    an assembly-set id the genomes registry deposits, so ``perturbation.host_genome``
    joins ``genome.assembly_set`` by value and never on a strain string.
    """
    from torchcell.sequence.genome import registry

    namespaces = set(typing.get_args(s.GeneIdNamespace))
    assert set(HOST_GENOME_OF_NAMESPACE) == namespaces
    assert len(set(HOST_GENOME_OF_NAMESPACE.values())) == len(namespaces)
    registry_ids = {
        value
        for name, value in vars(registry).items()
        if name.isupper() and isinstance(value, str) and "_" in value
    }
    assert set(HOST_GENOME_OF_NAMESPACE.values()) <= registry_ids
    assert HOST_GENOME_OF_NAMESPACE["scerevisiae_s288c_sgd_systematic"] == (
        registry.SGD_S288C_R64
    )
    assert HOST_GENOME_OF_NAMESPACE[MG1655] == registry.ECOLI_K12_MG1655
    assert HOST_GENOME_OF_NAMESPACE[KT2440] == registry.PPUTIDA_KT2440


def test_the_union_to_label_map_is_a_bijection_over_every_leaf() -> None:
    """Every member of ``GenePerturbationType`` is served under exactly one label.

    The one ``_perturbation_node`` partitions the union by leaf: a leaf that composes a
    ``BacterialVariantCall`` (``VARIANT_CALL_LEAVES``) is a ``sequence variant
    perturbation`` and every other leaf a ``perturbation``. The rule is a pure function
    of the class, pinned here over the whole union (audit C 3.3), so no conf and no
    class tuple can leave a leaf out: the former class-keyed lanes excluded
    ``GeneAdditionPerturbation``, which is the Carruthers 2025 gap this stage closes.
    ``SpanDeletionCallPerturbation`` is a ``DeletionStrainPerturbation`` by class and
    carries the call beside it, which is why the predicate is the composed call and not
    the ``VariantCallPerturbation`` base alone.
    """
    (union,) = typing.get_args(s.Genotype.model_fields["perturbations"].annotation)
    leaves = set(typing.get_args(union))
    label_of = {
        cls: (
            "sequence variant perturbation"
            if issubclass(cls, VARIANT_CALL_LEAVES)
            else "perturbation"
        )
        for cls in leaves
    }
    assert set(label_of.values()) == {"perturbation", "sequence variant perturbation"}
    composes_a_call = {
        cls
        for cls in leaves
        if cls.model_fields.get("call") is not None
        and cls.model_fields["call"].annotation is s.BacterialVariantCall
    }
    assert {cls for cls, label in label_of.items() if label.startswith("sequence")} == (
        composes_a_call
    )
    assert composes_a_call == {
        s.LocusVariantCallPerturbation,
        s.SiteVariantCallPerturbation,
        s.SpanDeletionCallPerturbation,
    }
    assert label_of[s.GeneAdditionPerturbation] == "perturbation"
    assert label_of[s.SgaKanMxDeletionPerturbation] == "perturbation"
    assert label_of[s.DeletionStrainPerturbation] == "perturbation"
    # the schema side: every leaf carries the host the node projects
    assert all("gene_namespace" in cls.model_fields for cls in leaves)
    assert {label_of[p.__class__] for p in _bacterial_leaves()} == {"perturbation"}
    assert {label_of[p.__class__] for p in _variant_leaves()} == {
        "sequence variant perturbation"
    }


# ------------------------------------------------------ sequence variant perturbation
def test_variant_nodes_carry_the_call_under_the_child_label() -> None:
    """The #731 class: one node per called variant, with the call projected off .call."""
    record = _variant_record(_titer())
    leaves = record["experiment"].genotype.perturbations
    nodes = _run("perturbation (chunked)", record)
    assert [n.get_label() for n in nodes] == ["sequence variant perturbation"] * 3
    assert [n.get_id() for n in nodes] == [_sha(p) for p in leaves]
    # Genotype sorts by systematic_gene_name, so the site id sorts before the PP_ tags
    # and PP_3415 before PP_4062.
    assert [n.get_preferred_id() for n in nodes] == [
        "site_variant_call",
        "locus_variant_call",
        "span_deletion_call",
    ]
    assert nodes[0].get_properties() == {
        "systematic_gene_name": "AE015451:4586057",
        "perturbed_gene_name": "PP_4061, PP_4063",
        "perturbation_type": "site_variant_call",
        "description": (
            "Called sequence variant keyed on its genomic site, because no locus tag "
            "of the pinned assembly holds it"
        ),
        "host_genome": "pputida_KT2440_ASM756v2",
        "reference_sequence": "AE015451",
        "position_start": 4586057,
        "position_end": 4586057,
        "variant_type": "insertion",
        "variant_frequency": 1.0,
        "call_mode": "clone",
        "id": _sha(leaves[0]),
        "preferred_id": "site_variant_call",
    }
    assert nodes[1].get_properties()["systematic_gene_name"] == "PP_3415"
    assert nodes[1].get_properties()["variant_type"] == "snv"
    assert nodes[2].get_properties()["systematic_gene_name"] == "PP_4062"
    assert nodes[2].get_properties()["variant_type"] == "deletion"
    assert "serialized_data" not in nodes[0].get_properties()
    # the span deletion (a DeletionStrainPerturbation subclass) is written ONCE, under
    # the child label, and a yeast record yields no child node
    assert [n.get_label() for n in _run("perturbation (chunked)", _yeast_record())] == [
        "perturbation"
    ] * 3


def test_variant_node_properties_are_exactly_the_declared_class_properties() -> None:
    """The emitted property set equals the schema config's, with no silent extra."""
    declared = set(SCHEMA["sequence variant perturbation"]["properties"])
    node = _run("perturbation (chunked)", _variant_record(_titer()))[0]
    assert set(node.get_properties()) - NODE_BOOKKEEPING == declared
    assert SCHEMA["sequence variant perturbation"]["is_a"] == "perturbation"
    assert set(SCHEMA["perturbation"]["properties"]) - {"strain_id"} < declared


def test_a_variant_record_edges_address_the_variant_nodes() -> None:
    """``perturbation to genotype`` computes the same ids, so no edge method is added."""
    record = _variant_record(_titer())
    node_ids = [n.get_id() for n in _run("perturbation (chunked)", record)]
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
    leaf = s.LocusVariantCallPerturbation(
        systematic_gene_name="PP_3415",
        perturbed_gene_name="PP_3415",
        gene_namespace=KT2440,
        call=ranged,
    )
    record = {
        "experiment": s.ProductTiterExperiment(
            dataset_name="BacterialVariantToy",
            genotype=s.Genotype(perturbations=[leaf]),
            environment=s.CultureEnvironment(media=_lb()),
            phenotype=_titer(),
        )
    }
    (node,) = _run("perturbation (chunked)", record)
    assert node.get_label() == "sequence variant perturbation"
    assert node.get_properties()["variant_frequency"] is None
    assert leaf.call.frequency_statement == "61 -> 63"


def test_the_served_edge_methods_connect_the_bacterial_nodes() -> None:
    """``perturbation member of`` and ``crispr construct member of`` hit the same ids."""
    record = _product_titer_record(_titer())
    node_ids = [n.get_id() for n in _run("perturbation (chunked)", record)]
    genotype_id = _run("genotype (chunked)", record)[0].get_id()

    edges = _run("perturbation to genotype (chunked)", record)
    assert [e.get_label() for e in edges] == ["perturbation member of"] * 5
    assert [e.get_source_id() for e in edges] == node_ids
    assert {e.get_target_id() for e in edges} == {genotype_id}

    crispri = next(
        p
        for p in record["experiment"].genotype.perturbations
        if isinstance(p, s.CrisprInterferencePerturbation)
    )
    assert _sha(crispri) in node_ids
    construct_edges = _run("crispr construct to perturbation (chunked)", record)
    assert [e.get_target_id() for e in construct_edges] == [_sha(crispri)]
    assert [e.get_source_id() for e in construct_edges] == [_sha(_construct())]


def test_the_host_genome_edge_joins_each_perturbation_to_its_genome_node() -> None:
    """``perturbation of genome`` (OFF in every conf) targets the ``genome`` node id.

    The chunk carries the record's reference, so the edge needs no extra pass; the
    genome node's ``assembly_set`` is the registry id the perturbations' ``host_genome``
    carries, which is the value join the edge is the traversal form of.
    """
    reference = s.AssemblyReferenceGenome(
        species="Escherichia coli",
        strain="MG1655",
        assembly_set="ecoli_K12_MG1655_ASM584v2",
        assembly_accession="GCA_000005845.2",
    )
    record = _product_titer_record(_titer())
    record["reference"] = SimpleNamespace(genome_reference=reference)
    edges = _run("perturbation to genome (chunked)", record)
    nodes = _run("perturbation (chunked)", record)
    assert [e.get_label() for e in edges] == ["perturbation of genome"] * 5
    assert [e.get_source_id() for e in edges] == [n.get_id() for n in nodes]
    assert {e.get_target_id() for e in edges} == {_sha(reference)}
    adapter = _adapter()
    adapter.dataset = cast(
        Any,
        SimpleNamespace(
            experiment_reference_index=[
                SimpleNamespace(reference=SimpleNamespace(genome_reference=reference))
            ]
        ),
    )
    (genome,) = adapter._get_genome_nodes()
    assert genome.get_id() == _sha(reference)
    assert genome.get_properties()["assembly_set"] == "ecoli_K12_MG1655_ASM584v2"
    assert {n.get_properties()["host_genome"] for n in nodes} == {
        genome.get_properties()["assembly_set"]
    }
    assert set(genome.get_properties()) - NODE_BOOKKEEPING == set(
        SCHEMA["genome"]["properties"]
    )
    assert "perturbation to genome (chunked)" not in _enabled_everywhere()


def test_a_reference_without_an_assembly_set_projects_null_not_a_strain() -> None:
    """A yeast ``ReferenceGenome`` states no set today (P2 adds it): the node says so."""
    reference = s.ReferenceGenome(species="Saccharomyces cerevisiae", strain="BY4741")
    assert CellAdapter._assembly_set_of(reference) is None
    adapter = _adapter()
    adapter.dataset = cast(
        Any,
        SimpleNamespace(
            experiment_reference_index=[
                SimpleNamespace(reference=SimpleNamespace(genome_reference=reference))
            ]
        ),
    )
    (genome,) = adapter._get_genome_nodes()
    assert genome.get_properties()["assembly_set"] is None
    assert genome.get_properties()["strain"] == "BY4741"


def _enabled_everywhere() -> set[str]:
    """Every method name any adapter conf enables."""
    names: set[str] = set()
    for path in sorted((REPO_ROOT / "torchcell/adapters/conf").glob("*_adapter.yaml")):
        conf = yaml.safe_load(path.read_text(encoding="utf-8"))
        for key in ("node_methods", "edge_methods"):
            names |= {m["method_name"] for m in conf["cell_adapter"][key]}
    return names


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
        "titer_censoring": "uncensored",
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
        "titer_censoring",
    ):
        assert props[key] is None, key


def test_a_below_detection_titer_projects_as_a_left_censored_bound() -> None:
    """#867: the floor is the stored titer and the side travels as its value."""
    phenotype = _titer(titer=0.155, titer_censoring=s.Censoring.left)
    props = _run("product titer phenotype (chunked)", _product_titer_record(phenotype))[
        0
    ].get_properties()
    assert (props["titer"], props["titer_censoring"]) == (0.155, "left")
    assert (props["titer_se"], props["titer_uncertainty"]) == (None, None)


def _phenotype_record(phenotype: Any) -> dict[str, Any]:
    """A record carrying only the phenotype (the phenotype methods read nothing else)."""
    return {"experiment": SimpleNamespace(phenotype=phenotype)}


def test_protein_synthesis_rate_phenotype_node_projects_dicts_as_json() -> None:
    """#857: each dict is one JSON string; the unit is its value, the time a float."""
    phenotype = _synthesis(
        synthesis_rate_se={"b0002": 4.0},
        n_replicates={"b0002": 2, "b0003": 1},
        censoring={"b0003": s.Censoring.left},
    )
    [node] = _run(
        "protein synthesis rate phenotype (chunked)", _phenotype_record(phenotype)
    )
    pid = _sha(phenotype)
    assert (node.get_id(), node.get_label(), node.get_preferred_id()) == (
        pid,
        "protein synthesis rate phenotype",
        f"phenotype_{pid}",
    )
    assert node.get_properties() == {
        "graph_level": "node",
        "label_name": "synthesis_rate",
        "label_statistic_name": "synthesis_rate_se",
        "synthesis_rate": '{"b0002": 120.0, "b0003": 0.0}',
        "synthesis_rate_se": '{"b0002": 4.0}',
        "n_replicates": '{"b0002": 2, "b0003": 1}',
        "rate_unit": "molecules_per_generation",
        "generation_time_minutes": 21.5,
        "measurement_type": "ribosome_profiling_footprint_density",
        "censoring": '{"b0003": "left"}',
        "id": pid,
        "preferred_id": f"phenotype_{pid}",
    }
    bare = _run(
        "protein synthesis rate phenotype (chunked)", _phenotype_record(_synthesis())
    )[0].get_properties()
    assert (bare["synthesis_rate_se"], bare["n_replicates"], bare["censoring"]) == (
        None,
        None,
        None,
    )


def test_phosphosite_intensity_phenotype_node_projects_maps_and_sites_as_json() -> None:
    """#921: the three site-keyed maps are JSON strings, a site its model_dump; the
    sample's enums are their values, and a gapped sample field projects as None.
    """
    phenotype = _phosphosite()
    [node] = _run(
        "phosphosite intensity phenotype (chunked)", _phenotype_record(phenotype)
    )
    pid = _sha(phenotype)
    assert (node.get_id(), node.get_label(), node.get_preferred_id()) == (
        pid,
        "phosphosite intensity phenotype",
        f"phenotype_{pid}",
    )
    props = node.get_properties()
    assert json.loads(props.pop("sites")) == {
        "P00350|S|86": _GND_S86.model_dump(mode="json"),
        "P0CE47|H|119;119": _TUF_H119.model_dump(mode="json"),
    }
    assert props == {
        "graph_level": "node",
        "label_name": "phosphosite_intensity",
        "label_statistic_name": None,
        "phosphosite_intensity": '{"P00350|S|86": 2714000.0}',
        "quantification": (
            '{"P00350|S|86": "quantified", "P0CE47|H|119;119": "not_quantified"}'
        ),
        "measurement_type": "fe3_imac_lc_msms_maxquant_summed_xic",
        "sample_label": "Intensity glucose_exp1",
        "replicate": 1,
        "replicate_unit": "technical_replicate",
        "harvest_growth_phase": "exponential",
        "harvest_optical_density": 0.6,
        "id": pid,
        "preferred_id": f"phenotype_{pid}",
    }
    gap = ProvenanceGap(
        field="replicate_unit",
        reason=ProvenanceGapReason.deferred_pending_source_review,
    )
    gapped = _run(
        "phosphosite intensity phenotype (chunked)",
        _phenotype_record(_phosphosite(replicate_unit=None, provenance_gaps=[gap])),
    )[0].get_properties()
    assert gapped["replicate_unit"] is None


def test_protein_turnover_phenotype_node_projects_dicts_as_json() -> None:
    """Every dict is one JSON string, including #753's bounds and censoring map."""
    phenotype = _turnover(
        degradation_rate_se={"b0002": 0.01},
        half_life={"b0002": 6.3},
        synthesis_rate={"b0003": 0.2},
        degradation_rate_lower={"b0003": 0.06},
        degradation_rate_upper={"b0003": 0.08},
        confidence_level=0.95,
        interval_method="curve_fit_parameter_variance_t_quantile",
        censoring={"b0002": s.Censoring.right, "b0003": s.Censoring.uncensored},
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
        "degradation_rate_lower": '{"b0003": 0.06}',
        "degradation_rate_upper": '{"b0003": 0.08}',
        "confidence_level": 0.95,
        "interval_method": "curve_fit_parameter_variance_t_quantile",
        "censoring": '{"b0002": "right", "b0003": "uncensored"}',
        "id": pid,
        "preferred_id": f"phenotype_{pid}",
    }
    bare = _run("protein turnover phenotype (chunked)", _phenotype_record(_turnover()))
    props = bare[0].get_properties()
    assert (
        props["degradation_rate_se"],
        props["half_life"],
        props["synthesis_rate"],
        props["degradation_rate_lower"],
        props["degradation_rate_upper"],
        props["confidence_level"],
        props["interval_method"],
        props["censoring"],
    ) == (None, None, None, None, None, None, None, None)


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
        "exchange_flux": None,
        "exchange_flux_lower": None,
        "exchange_flux_upper": None,
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
        "exchange_flux",
        "exchange_flux_lower",
        "exchange_flux_upper",
    )
    assert [bare.get_properties()[k] for k in optional] == [None] * len(optional)


def test_flux_phenotype_node_serves_the_exchange_fluxes_verbatim() -> None:
    """#800: Li 2021's exchange table rides on the node as JSON, unconverted."""
    phenotype = _flux(
        exchange_flux={"vAco.x": 0.33},
        exchange_flux_lower={"vAco.x": 0.21},
        exchange_flux_upper={"vAco.x": 0.45},
    )
    [node] = _run("flux phenotype (chunked)", _phenotype_record(phenotype))
    properties = node.get_properties()
    assert properties["exchange_flux"] == '{"vAco.x": 0.33}'
    assert properties["exchange_flux_lower"] == '{"vAco.x": 0.21}'
    assert properties["exchange_flux_upper"] == '{"vAco.x": 0.45}'


def test_flux_reference_node_serves_a_gapped_map_as_null() -> None:
    """#800: a reference whose map the source does not release has a null net_flux."""
    from torchcell.adapters.cell_adapter import CellAdapter
    from torchcell.verification.sourced import ProvenanceGap, ProvenanceGapReason

    gap = ProvenanceGap(
        field="net_flux", reason=ProvenanceGapReason.not_reported_by_primary
    )
    gapped = s.FluxPhenotype(
        net_flux=None,
        measurement_type="c13_mfa_net_flux_mmol_per_gdcw_per_h",
        provenance_gaps=[gap],
    )
    properties = CellAdapter._flux_properties(gapped)
    assert properties["net_flux"] is None
    assert properties["net_flux_lower"] is None


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


def _fraction(**kw: Any) -> s.MrnaNumberFractionPhenotype:
    fields: dict[str, Any] = dict(
        mrna_number_fraction={"b0002": 0.25, "b0001": 0.5},
        n_libraries=2,
        measurement_type="rnaseq_mrna_number_fraction",
    )
    fields.update(kw)
    return s.MrnaNumberFractionPhenotype(**fields)


def test_mrna_number_fraction_node_projects_the_dict_as_sorted_json() -> None:
    """The per-gene dict is one JSON string; the two scalars stay typed."""
    phenotype = _fraction()
    [node] = _run(
        "mrna number fraction phenotype (chunked)", _phenotype_record(phenotype)
    )
    pid = _sha(phenotype)
    assert (node.get_id(), node.get_label(), node.get_preferred_id()) == (
        pid,
        "mrna number fraction phenotype",
        f"phenotype_{pid}",
    )
    assert node.get_properties() == {
        "id": pid,
        "preferred_id": f"phenotype_{pid}",
        "graph_level": "node",
        "label_name": "mrna_number_fraction",
        "label_statistic_name": None,
        "mrna_number_fraction": '{"b0001": 0.5, "b0002": 0.25}',
        "n_libraries": 2,
        "measurement_type": "rnaseq_mrna_number_fraction",
    }


def _expression_score(**kw: Any) -> s.MicroarrayExpressionScorePhenotype:
    fields: dict[str, Any] = dict(
        expression_score={"YBR001C": 1.5, "YAL001C": -3.0},
        expression_score_p_value={"YBR001C": 0.133614, "YAL001C": 0.0027},
        measurement_type="error_model_x_score_weighted_duplicate_average",
        reference_basis="wild-type BY4741 culture",
        array_platform="GEO GPL3588",
        n_replicates=2,
        harvest_optical_density=0.82,
    )
    fields.update(kw)
    return s.MicroarrayExpressionScorePhenotype(**fields)


def test_microarray_expression_score_node_projects_dicts_as_sorted_json() -> None:
    """#1022: each per-gene dict is one JSON string; the scalars stay typed."""
    phenotype = _expression_score()
    [node] = _run(
        "microarray expression score phenotype (chunked)", _phenotype_record(phenotype)
    )
    pid = _sha(phenotype)
    assert (node.get_id(), node.get_label(), node.get_preferred_id()) == (
        pid,
        "microarray expression score phenotype",
        f"phenotype_{pid}",
    )
    assert node.get_properties() == {
        "id": pid,
        "preferred_id": f"phenotype_{pid}",
        "graph_level": "node",
        "label_name": "expression_score",
        "label_statistic_name": "expression_score_p_value",
        "expression_score": '{"YAL001C": -3.0, "YBR001C": 1.5}',
        "expression_score_p_value": '{"YAL001C": 0.0027, "YBR001C": 0.133614}',
        "measurement_type": "error_model_x_score_weighted_duplicate_average",
        "reference_basis": "wild-type BY4741 culture",
        "array_platform": "GEO GPL3588",
        "n_replicates": 2,
        "harvest_optical_density": 0.82,
    }


def test_microarray_expression_score_reference_projects_its_absent_p_as_none() -> None:
    """A reference is not tested against itself: no p-value map, no harvest density."""
    phenotype = _expression_score(
        expression_score={"YAL001C": 0.0},
        expression_score_p_value=None,
        harvest_optical_density=None,
    )
    props = CellAdapter._microarray_expression_score_properties(phenotype)
    assert (props["expression_score_p_value"], props["harvest_optical_density"]) == (
        None,
        None,
    )
    assert props["expression_score"] == '{"YAL001C": 0.0}'


def _expression_ratio(**kw: Any) -> s.MicroarrayExpressionRatioPhenotype:
    fields: dict[str, Any] = dict(
        expression_log2_ratio={"YBR001C": 1.5, "YAL001C": -0.25},
        measurement_type="scanalyze_rat2n_normalized_log2_ratio",
        reference_basis="the same culture before the shift",
        array_platform="spotted yeast ORF array",
        provenance_gaps=[
            {"field": "n_replicates", "reason": "not_reported_by_primary"}
        ],
    )
    fields.update(kw)
    return s.MicroarrayExpressionRatioPhenotype(**fields)


def test_microarray_expression_ratio_node_projects_dicts_as_sorted_json() -> None:
    """#989: the ratio map is one sorted JSON string; a gapped count map stays None."""
    phenotype = _expression_ratio()
    [node] = _run(
        "microarray expression ratio phenotype (chunked)", _phenotype_record(phenotype)
    )
    pid = _sha(phenotype)
    assert (node.get_id(), node.get_label(), node.get_preferred_id()) == (
        pid,
        "microarray expression ratio phenotype",
        f"phenotype_{pid}",
    )
    assert node.get_properties() == {
        "id": pid,
        "preferred_id": f"phenotype_{pid}",
        "graph_level": "node",
        "label_name": "expression_log2_ratio",
        "label_statistic_name": None,
        "expression_log2_ratio": '{"YAL001C": -0.25, "YBR001C": 1.5}',
        "measurement_type": "scanalyze_rat2n_normalized_log2_ratio",
        "reference_basis": "the same culture before the shift",
        "array_platform": "spotted yeast ORF array",
        "n_replicates": None,
    }


def test_microarray_expression_ratio_projects_sourced_counts_as_json() -> None:
    """A release that states its per-gene counts carries them as a sorted JSON map."""
    phenotype = _expression_ratio(
        n_replicates={"YBR001C": 2, "YAL001C": 1}, provenance_gaps=[]
    )
    props = CellAdapter._microarray_expression_ratio_properties(phenotype)
    assert props["n_replicates"] == '{"YAL001C": 1, "YBR001C": 2}'


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
            "protein synthesis rate phenotype reference",
            "protein synthesis rate phenotype",
            [_synthesis(), _synthesis(), _synthesis(generation_time_minutes=56.3)],
        ),
        (
            "phosphosite intensity phenotype reference",
            "phosphosite intensity phenotype",
            [_phosphosite(), _phosphosite(), _phosphosite(replicate=2)],
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
        (
            "mrna number fraction phenotype reference",
            "mrna number fraction phenotype",
            [_fraction(), _fraction(), _fraction(n_libraries=1)],
        ),
        (
            "microarray expression score phenotype reference",
            "microarray expression score phenotype",
            [
                _expression_score(),
                _expression_score(),
                _expression_score(harvest_optical_density=1.1),
            ],
        ),
        (
            "microarray expression ratio phenotype reference",
            "microarray expression ratio phenotype",
            [
                _expression_ratio(),
                _expression_ratio(),
                _expression_ratio(reference_basis="the 33 C culture"),
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
NEW_CLASSES: dict[str, tuple[str, type[Any] | None]] = {
    "product titer phenotype": (
        "product titer phenotype (chunked)",
        s.ProductTiterPhenotype,
    ),
    "protein turnover phenotype": (
        "protein turnover phenotype (chunked)",
        s.ProteinTurnoverPhenotype,
    ),
    "protein synthesis rate phenotype": (
        "protein synthesis rate phenotype (chunked)",
        s.ProteinSynthesisRatePhenotype,
    ),
    "phosphosite intensity phenotype": (
        "phosphosite intensity phenotype (chunked)",
        s.PhosphositeIntensityPhenotype,
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
    "mrna number fraction phenotype": (
        "mrna number fraction phenotype (chunked)",
        s.MrnaNumberFractionPhenotype,
    ),
    "fitness contrast phenotype": (
        "fitness contrast phenotype (chunked)",
        s.FitnessContrastPhenotype,
    ),
    "microarray expression score phenotype": (
        "microarray expression score phenotype (chunked)",
        s.MicroarrayExpressionScorePhenotype,
    ),
    "microarray expression ratio phenotype": (
        "microarray expression ratio phenotype (chunked)",
        s.MicroarrayExpressionRatioPhenotype,
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


def test_the_host_keyed_classes_are_retired_and_the_child_sits_under_perturbation() -> (
    None
):
    """One perturbation class for every host; the variant child is_a perturbation."""
    assert "bacterial perturbation" not in SCHEMA
    assert "bacterial sequence variant perturbation" not in SCHEMA
    assert SCHEMA["perturbation"]["is_a"] == "genotype"
    assert set(SCHEMA["perturbation"]["properties"]) == {
        "systematic_gene_name",
        "perturbed_gene_name",
        "perturbation_type",
        "description",
        "host_genome",
        "strain_id",
    }
    assert SCHEMA["sequence variant perturbation"]["is_a"] == "perturbation"
    assert SCHEMA["perturbation member of"]["source"] == [
        "perturbation",
        "sequence variant perturbation",
    ]
    assert SCHEMA["crispr construct member of"]["target"] == "perturbation"
    assert SCHEMA["perturbation of genome"] == {
        "is_a": "is sequence variant of",
        "represented_as": "edge",
        "input_label": "perturbation of genome",
        "source": ["perturbation", "sequence variant perturbation"],
        "target": "genome",
    }
    assert "assembly_set" in SCHEMA["genome"]["properties"]


def test_emitted_properties_equal_the_declared_properties_at_run_time() -> None:
    """The static coherence check, repeated on real emitted nodes."""
    record = _product_titer_record(_full_titer())
    emitted = {
        "perturbation": _run("perturbation (chunked)", record)[0],
        "sequence variant perturbation": _run(
            "perturbation (chunked)", _variant_record(_titer())
        )[0],
        "product titer phenotype": _run("product titer phenotype (chunked)", record)[0],
        "protein turnover phenotype": _run(
            "protein turnover phenotype (chunked)", _phenotype_record(_turnover())
        )[0],
        "flux phenotype": _run("flux phenotype (chunked)", _phenotype_record(_flux()))[
            0
        ],
        "protein synthesis rate phenotype": _run(
            "protein synthesis rate phenotype (chunked)",
            _phenotype_record(_synthesis()),
        )[0],
        "phosphosite intensity phenotype": _run(
            "phosphosite intensity phenotype (chunked)",
            _phenotype_record(_phosphosite()),
        )[0],
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
    assert table["perturbation (chunked)"] == "_perturbation_node"
    assert table["perturbation to genome (chunked)"] == "_perturbation_to_genome_edges"
    assert "bacterial perturbation (chunked)" not in table
    assert "bacterial sequence variant perturbation (chunked)" not in table
    assert "phage perturbation (chunked)" not in table
    assert "phage perturbation reference" not in table
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
        "protein synthesis rate phenotype (chunked)": (
            "_protein_synthesis_rate_phenotype_node"
        ),
        "protein synthesis rate phenotype reference": (
            "_get_protein_synthesis_rate_phenotype_reference_nodes"
        ),
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
        "mrna number fraction phenotype (chunked)": (
            "_mrna_number_fraction_phenotype_node"
        ),
        "mrna number fraction phenotype reference": (
            "_get_mrna_number_fraction_phenotype_reference_nodes"
        ),
        "fitness contrast phenotype (chunked)": "_fitness_contrast_phenotype_node",
        "fitness contrast phenotype reference": (
            "_get_fitness_contrast_phenotype_reference_nodes"
        ),
        "microarray expression score phenotype (chunked)": (
            "_microarray_expression_score_phenotype_node"
        ),
        "microarray expression score phenotype reference": (
            "_get_microarray_expression_score_phenotype_reference_nodes"
        ),
        "phosphosite intensity phenotype (chunked)": (
            "_phosphosite_intensity_phenotype_node"
        ),
        "phosphosite intensity phenotype reference": (
            "_get_phosphosite_intensity_phenotype_reference_nodes"
        ),
        "microarray expression ratio phenotype (chunked)": (
            "_microarray_expression_ratio_phenotype_node"
        ),
        "microarray expression ratio phenotype reference": (
            "_get_microarray_expression_ratio_phenotype_reference_nodes"
        ),
    }.items():
        assert table[name] == fn
    assert SCHEMA["perturbation member of"]["source"] == [
        "perturbation",
        "sequence variant perturbation",
    ]
    assert SCHEMA["perturbation member of"]["input_label"] == "perturbation member of"
    assert SCHEMA["perturbation member of"]["target"] == "genotype"
    sources = SCHEMA["phenotype member of"]["source"]
    # The yeast single-cell leaf (Boocock 2025; Jackson 2020's pseudobulk UMI counts
    # merged into it, #1016), De Chiara 2022's yeast life cycle phenotype and Puddu
    # 2019's genome architecture, joined after the bacterial ones. The pooled sort-seq
    # expression leaf (#991) and the biosensor fluorescence leaf (#990) sit directly
    # after the promoter-activity class they are siblings of; #921's phosphosite
    # intensity phenotype sits beside the protein synthesis-rate leaf.
    assert sources[-14:] == [
        "product titer phenotype",
        "protein turnover phenotype",
        "protein synthesis rate phenotype",
        "phosphosite intensity phenotype",
        "flux phenotype",
        "promoter activity phenotype",
        "pooled sort seq expression phenotype",
        "biosensor fluorescence phenotype",
        "bacterial morphology phenotype",
        "mrna number fraction phenotype",
        "single cell expression phenotype",
        "life cycle phenotype",
        "genome architecture phenotype",
        "fitness contrast phenotype",
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
    nodes = _run("environment perturbation (chunked)", record)
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
        _run("environment perturbation (chunked)", _phage_challenge_record(p))[
            0
        ].get_id()
        for p in (low, high, other)
    }
    assert len(ids) == 3


def test_the_one_environment_lane_labels_a_compound_and_a_factor_as_environment() -> (
    None
):
    """A small molecule or a physical factor is `environment perturbation`, a phage not."""
    record = _phage_challenge_record(
        s.SmallMoleculePerturbation(
            compound=s.Compound(name="kanamycin"),
            concentration=s.Concentration(
                value=50.0, unit=s.ConcentrationUnit.ug_per_ml
            ),
        ),
        s.EnvironmentPhysicalPerturbation(factor=s.PhysicalFactor.ph),
        _phage(),
    )
    served = _run("environment perturbation (chunked)", record)
    assert [n.get_label() for n in served] == [
        "environment perturbation",
        "environment perturbation",
        "phage perturbation",
    ]


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
    nodes = CellAdapter._get_environment_perturbation_reference_nodes(adapter)
    assert [n.get_label() for n in nodes] == ["phage perturbation"]
    assert nodes[0].get_properties()["phage_name"] == "lambda cI857"


def test_the_environment_lane_is_registered_under_its_conf_names() -> None:
    table = _table()
    assert table["environment perturbation (chunked)"] == (
        "_environment_perturbation_node"
    )
    assert table["environment perturbation reference"] == (
        "_get_environment_perturbation_reference_nodes"
    )
