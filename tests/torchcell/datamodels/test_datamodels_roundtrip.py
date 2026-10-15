# tests/torchcell/datamodels/test_datamodels_roundtrip.py
# [[tests.torchcell.datamodels.test_datamodels_roundtrip]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datamodels/test_datamodels_roundtrip.py
"""Serialization round trip for every pydantic model in ``torchcell.datamodels``.

The classes are discovered by walking the package, never listed by hand, so a new model
joins the sweep the day it is defined. Each class gets an instance from ``EXAMPLES`` when
a hand-written dict is committed for it, else from a generic builder that fills every
required field from its annotation (first Literal value, first Enum member, one nested
model, one-element lists and dicts) with a few field-name conventions the schema
validators enforce (``systematic_gene_name`` is a systematic ORF name, ``*so_id`` a
Sequence Ontology CURIE, ``*sha256`` sixty-four hex digits). A class the builder cannot
construct must appear in ``UNCONSTRUCTIBLE`` with the reason; an unexplained failure is a
test failure, so nothing is skipped silently (Phase 1 of
[[plan.test-suite-buildout.2026.09.25]]).

Pinned per class: ``model_validate(model_dump())`` and ``model_validate_json(
model_dump_json())`` reproduce an equal object, ``model_dump()`` is deterministic across
two calls, and ``model_json_schema()`` is JSON-serializable and identical across two
calls.
"""

from __future__ import annotations

import datetime as dt
import enum
import importlib
import inspect
import json
import pkgutil
import types
import typing
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

import torchcell.datamodels as datamodels

# Modules whose import has side effects or needs optional dependencies.
SKIP_MODULES = {"torchcell.datamodels.compound_identity_curate"}

# Values the schema validators demand for a field of this name, whatever the class.
FIELD_VALUES: dict[str, Any] = {
    "systematic_gene_name": "YAL001C",
    "perturbed_gene_name": "TFC3",
    "so_id": "SO:0000704",
    "mechanism_so_id": "SO:0000159",
    "mechanism_so_name": "deletion",
    "pubmed_id": "12345678",
    "sha256": "a" * 64,
    # SingleCellExpressionPhenotype pins its feature universe by a 64-hex digest.
    "feature_universe_sha256": "a" * 64,
}

# One released Potel 2018 Table 2 row (gnd S86), shared by the two #921 examples.
_PHOSPHOSITE_EXAMPLE: dict[str, Any] = {
    "protein_ids": ["P00350"],
    "protein_id_namespace": "uniprotkb_accession",
    "gene_names": ["gnd"],
    "positions": [86],
    "residue": "S",
    "sequence_windows": ["LLMVKAGAGTDAAIDSLKPYLDKGDIIIDGG"],
    "localization_probability": 0.999615,
    "posterior_error_probability": 4.27602e-05,
    "identification_score": 66.401,
    "delta_score": 58.487,
    "diagnostic_peak": False,
}

# One call of each shape the #731 leaves carry, as released rows of the Lim 2025
# ``Fig 2B_Mutation List`` sheet. Shared by the four EXAMPLES entries below so the site
# id and the span designation agree with the call they name.
_VARIANT_CALL_EXAMPLE: dict[str, Any] = {
    "variant_type": "snv",
    "type_statement": "SNP",
    "reference_sequence": "AE015451",
    "position_start": 3866001,
    "position_end": 3866001,
    "sequence_change": "G\u2192A",
    "annotation": "P293S (CCA\u2192TCA)",
    "call_mode": "clone",
    "frequency_statement": "1",
    "frequency": 1.0,
    "frequency_basis": "fraction",
}
_INTERGENIC_CALL_EXAMPLE: dict[str, Any] = {
    "variant_type": "insertion",
    "type_statement": "INS",
    "reference_sequence": "AE015451",
    "position_start": 4586057,
    "position_end": 4586057,
    "sequence_change": "+C",
    "annotation": "intergenic (+140/+75)",
    "call_mode": "clone",
    "frequency_statement": "1",
    "frequency": 1.0,
    "frequency_basis": "fraction",
}
_SPAN_CALL_EXAMPLE: dict[str, Any] = {
    "variant_type": "deletion",
    "type_statement": "DEL",
    "reference_sequence": "AE015451",
    "position_start": 4588139,
    "position_end": 4593691,
    "sequence_change": "\u03945,553 bp",
    "call_mode": "clone",
    "frequency_statement": "1",
    "frequency": 1.0,
    "frequency_basis": "fraction",
}

# class qualname -> keyword arguments; committed when a validator wants a specific shape.
_ESSENTIALITY_QUOTE: dict[str, Any] = {
    "value": 2,
    "quote": "a quoted rule",
    "provenance": {
        "source_uri": "paper.md",
        "citation_key": "testKey2026",
        "sha256": "0" * 64,
    },
}

# Cachera 2023's Btx cassette at the XII-5 landing pad, shared by the three
# construct-placement examples so the contig the call names is the contig the placement
# pins. ``pBTX002.gb``'s real sha256, so the example reads as the record it models.
_CONSTRUCT_CONTIG_SHA256 = (
    "63f89c1eaa13f888610ed781c4de12c8ac34f4e20512b4eb2c4f32d31af096da"
)
_HOST_PLACEMENT_EXAMPLE: dict[str, Any] = {
    "assembly_set": "sgd_S288C_R64-4-1_20230830",
    "site_label": "XII-5",
    "span": None,
    "insert_span": {"start": 2003, "end": 10105, "orientation": "inline"},
    "provenance_gaps": [{"field": "span", "reason": "not_reported_by_primary"}],
}
_CONSTRUCT_CALL_EXAMPLE: dict[str, Any] = {
    "variant_type": "snv",
    "variant_statement": "K229L",
    "reference_contig_sha256": _CONSTRUCT_CONTIG_SHA256,
    "position_start": 3685,
    "position_end": 3685,
    "reference_allele": "A",
    "alternate_allele": "T",
    "amino_acid_change": "K -> L",
    "codon_number": 229,
}


EXAMPLES: dict[str, dict[str, Any]] = {
    # #924: a SourcedValue's value is Any, which the generic builder cannot fill, and
    # the evidence needs exactly one scoring rule and one call definition.
    "torchcell.datamodels.schema.EssentialityStatement": {
        "role": "scoring_rule",
        "provenance": _ESSENTIALITY_QUOTE,
    },
    "torchcell.datamodels.schema.GeneEssentialityEvidence": {
        "evidence_class": "mutant_recovery",
        "evidence_type": "construction_outcome",
        "statements": [
            {"role": "scoring_rule", "provenance": _ESSENTIALITY_QUOTE},
            {"role": "call_definition", "provenance": _ESSENTIALITY_QUOTE},
        ],
        "colonies_verified": 1,
        "colonies_tested": 8,
        "percent_verified": 12.5,
    },
    # #909: the graded leaf needs a type that releases a categorical call
    "torchcell.datamodels.schema.GradedGeneEssentialityPhenotype": {
        "essentiality_call": "ES",
        "evidence": {
            "evidence_class": "insertion_depletion",
            "evidence_type": "tnseq_hmm_state",
            "statements": [
                {"role": "scoring_rule", "provenance": _ESSENTIALITY_QUOTE},
                {"role": "call_definition", "provenance": _ESSENTIALITY_QUOTE},
            ],
            "hmm_state": "ES",
            "ta_sites": 8,
            "hmm_sites_essential": 5,
            "hmm_sites_growth_defect": 0,
            "hmm_sites_non_essential": 3,
            "hmm_sites_growth_advantage": 0,
            "mean_insertion_rate": 0.5,
            "mean_read_count": 23.0,
        },
    },
    # #913: the generic builder sets every required float to the same value but leaves
    # q_value_method None, which the leaf refuses (a q-value names its correction).
    "torchcell.datamodels.schema.FitnessContrastPhenotype": {
        "measurement_type": "rbtnseq_gene_fitness_two_sample_t_test",
        "test_name": "two-sample t test",
        "contrast_label": "Glu_v_Glu_FA",
        "t_statistic": 32.47,
        "p_value": 5.4e-06,
        "q_value": 0.027,
        "q_value_method": "positive_false_discovery_rate",
    },
    # The generic builder fills a str field with "x", which no registered morphology
    # assay is named and no assay symbol is, so the #774 phenotype needs a real one.
    "torchcell.datamodels.schema.BacterialMorphologyPhenotype": {
        "assay": "campos2018",
        "morphology": {"<L>": 2.81, "<W>": 1.08, "%2N": 0.19},
        "morphology_coefficient_of_variation": {"CV_L": 0.24, "CV_W": 0.07},
        "n_samples": 245,
        "sample_unit": "cell",
    },
    # The same reason as the morphology phenotype above: the generic builder's "x" is
    # neither a registered genome-architecture assay nor one of its symbols, and the
    # ANtot aggregate has to agree with the chromosome values beside it.
    "torchcell.datamodels.schema.GenomeArchitecturePhenotype": {
        "assay": "puddu2019_wgs_coverage",
        "genome_architecture": {"rDNA": 117.19, "chr01": 2.0, "ANtot": 0.0, "GCR": 0.0},
        "n_samples": 1,
        "sample_unit": "colony",
        "sample_id": "SD0863b",
        "sample_accession": "ERS838232",
    },
    # The generic builder fills every str field with one value, which the #987 leaf's
    # keying rule refuses; a frameshift row released without a code is keyed by its
    # Gene, verbatim.
    "torchcell.datamodels.schema.NaturalLossOfFunctionPerturbation": {
        "systematic_gene_name": "PAU8",
        "gene_namespace": "scerevisiae_s288c_sgd_systematic",
        "perturbed_gene_name": "PAU8",
        "strain_id": "AAB",
        "cause": "homozygous_frameshift",
        "call_ref": {
            "tier": "genomes",
            "key": "peter2018_1011_assemblies",
            "path": "genesMatrix_Frameshift.tab.gz",
            "sha256": "b" * 64,
        },
        "release_code": None,
        "release_name": None,
        "release_gene_name": "PAU8",
        "release_span": {
            "chromosome": "chrI",
            "start": 1807,
            "end": 2169,
            "assembly": "sgd_S288C_R64-4-1_20230830",
        },
    },
    "torchcell.datamodels.schema.BackgroundAllele": {
        "systematic_gene_name": "YOR202W",
        "gene_name": "HIS3",
        "allele_name": "his3Δ1",
        "edit": "partial_deletion",
        "functional": False,
        "zygosity": "haploid",
        "provenance_gaps": [
            {"field": "provenance", "reason": "deferred_pending_source_review"}
        ],
    },
    "torchcell.datamodels.schema.IntegratedCassette": {
        "name": "Delta::KanMX-[dLbCpf1-VP]-[Csy4]-[dSpCas9-RD1152]-[SaCas9]",
        "locus": "Delta",
        "elements": ["KanMX", "dLbCpf1-VP", "Csy4", "dSpCas9-RD1152", "SaCas9"],
        "marker": "KanMX",
        "zygosity": "haploid",
        "provenance_gaps": [
            {"field": "provenance", "reason": "deferred_pending_source_review"}
        ],
    },
    "torchcell.datamodels.schema.StrainBackground": {
        "name": "BY4741",
        "mating_type": "a",
        "ploidy": "haploid",
        "provenance_gaps": [
            {"field": "provenance", "reason": "deferred_pending_source_review"}
        ],
    },
    "torchcell.datamodels.schema.StrainConstruction": {"lab": "Lab 14", "batch": "3"},
    "torchcell.datamodels.schema.StrainReferenceGenome": {
        "species": "Saccharomyces cerevisiae",
        "strain": "BY4741",
        "assembly_set": "sgd_S288C_R64-4-1_20230830",
        "assembly_accession": "GCA_000146045.2",
        "background": {
            "name": "BY4741",
            "mating_type": "a",
            "ploidy": "haploid",
            "provenance_gaps": [
                {"field": "provenance", "reason": "deferred_pending_source_review"}
            ],
        },
    },
    # #991: the generic builder's "x" is not DNA, and the leaf's whole point is that
    # the promoter sequence IS the genotype, so the example is a real released construct
    # (Vaishnav 2022 origID HQ_28, GSE163045 MolEvol SC-Ura table, abbreviated insert).
    "torchcell.datamodels.schema.VariablePromoterReporterPerturbation": {
        "systematic_gene_name": "addgene127546:YFP",
        "gene_namespace": "scerevisiae_s288c_sgd_systematic",
        "perturbed_gene_name": "YFP",
        "source_organism": "synthetic construct",
        "is_heterologous": True,
        "localization": "episomal_replicon",
        "replicon_statement": "CEN plasmid",
        "placement": None,
        "provenance_gaps": [
            {"field": "placement", "reason": "not_released_by_primary"}
        ],
        "promoter_variant_id": "HQ_28",
        "promoter_sequence": (
            "TGCATTTTTTTCACATCAAAAGTAGGTTCTTGCAGCATTCTGATGCACGTATAGGCGCGATGG"
            "ATGGCAGCGGGAATGCGGTTAAGCGGTGGATCAAGGTTACGGCTGTT"
        ),
        "scaffold_name": "pTpA",
        "variable_region_start": 18,
        "variable_region_end": 97,
    },
    "torchcell.datamodels.schema.Concentration": {"value": 1.0, "unit": "mM"},
    "torchcell.datamodels.schema.EnvironmentResponsePhenotype": {
        "measurement_type": "log2_ratio",
        "environment_response": 0.5,
    },
    "torchcell.datamodels.schema.ExpressionModulationPerturbation": {
        "systematic_gene_name": "YAL001C",
        "gene_namespace": "scerevisiae_s288c_sgd_systematic",
        "perturbed_gene_name": "TFC3",
        "expression_direction": "increased",
        "crispr": {"effector": "dCas9-VPR"},
    },
    "torchcell.datamodels.schema.Media": {
        "name": "YPD",
        "state": "solid",
        "is_synthetic": False,
    },
    "torchcell.datamodels.schema.CalMorphPhenotype": {"calmorph": {"A101_A": 0.5}},
    # #989: the ratio-only microarray leaf requires its per-gene counts or a typed gap,
    # and the generic builder leaves the optional map None without one.
    "torchcell.datamodels.schema.MicroarrayExpressionRatioPhenotype": {
        "expression_log2_ratio": {"YAL001C": -0.25, "YBR001C": 1.5},
        "measurement_type": "scanalyze_rat2n_normalized_log2_ratio",
        "reference_basis": "the same culture before the shift",
        "array_platform": "spotted yeast ORF array",
        "provenance_gaps": [
            {"field": "n_replicates", "reason": "not_reported_by_primary"}
        ],
    },
    # De Chiara 2022: a life-cycle key is '<measurement type>@<time><h|d>' and the
    # generic builder's "x" key parses as none.
    "torchcell.datamodels.schema.LifeCyclePhenotype": {
        "life_cycle": {"sporulation_efficiency@24h": 22.0, "dyad_fraction@24h": 0.0},
        "assay": "microscopy_ascus_count",
        "cells_per_sample": 100,
        "cells_per_sample_is_minimum": True,
    },
    # #857: the generic builder picks the first SynthesisRateUnit member, a rate per
    # generation, which must state the generation time the builder leaves out.
    "torchcell.datamodels.schema.ProteinSynthesisRatePhenotype": {
        "synthesis_rate": {"b0001": 12.0},
        "rate_unit": "molecules_per_generation",
        "generation_time_minutes": 21.5,
        "measurement_type": "ribosome_profiling_footprint_density",
    },
    # #921: a site's key must be its own (proteins|residue|positions) and its window
    # must center on its residue, so the phosphosite family carries a released row
    # (Potel 2018 Table 2, gnd S86) rather than generic placeholders.
    "torchcell.datamodels.schema.Phosphosite": _PHOSPHOSITE_EXAMPLE,
    "torchcell.datamodels.schema.PhosphositeIntensityPhenotype": {
        "phosphosite_intensity": {"P00350|S|86": 2714000.0},
        "quantification": {"P00350|S|86": "quantified"},
        "sites": {"P00350|S|86": _PHOSPHOSITE_EXAMPLE},
        "measurement_type": "fe3_imac_lc_msms_maxquant_summed_xic",
        "sample_label": "Intensity glucose_exp1",
        "replicate": 1,
        "replicate_unit": "technical_replicate",
        "harvest_growth_phase": "exponential",
        "harvest_optical_density": 0.6,
    },
    "torchcell.datamodels.schema.PresenceAbsencePerturbation": {
        "gene_namespace": "scerevisiae_s288c_sgd_systematic",
        "systematic_gene_name": "YAL001C",
        "perturbed_gene_name": "TFC3",
        "state": "present",
        "mechanism_so_id": "SO:0000159",
        "mechanism_so_name": "deletion",
    },
    "torchcell.datamodels.schema.Publication": {
        "pubmed_id": "12345678",
        "pubmed_url": "https://pubmed.ncbi.nlm.nih.gov/12345678/",
    },
    # Bacterial additions: the locus-tag validator rejects the sweep's default
    # ``YAL001C``, and the assembly pin rejects a placeholder accession, so each of
    # these commits a real tag / a real deposited (set, accession) pair.
    "torchcell.datamodels.schema.ReferenceGenome": {
        "species": "Escherichia coli",
        "strain": "MG1655",
        "assembly_set": "ecoli_K12_MG1655_ASM584v2",
        "assembly_accession": "GCA_000005845.2",
    },
    "torchcell.datamodels.schema.BacterialBackgroundAllele": {
        "systematic_gene_name": "BW25113_3643",
        "gene_namespace": "ecoli_k12_bw25113_locus_tag",
        "gene_name": "rph",
        "allele_name": "rph-1",
        "edit": "sequence_variant",
        "functional": False,
        "provenance_gaps": [
            {"field": "provenance", "reason": "deferred_pending_source_review"}
        ],
    },
    "torchcell.datamodels.schema.BacterialStrainBackground": {
        "name": "BW25113",
        "reference_strain": "BW25113",
        "assembly_set": "ecoli_K12_BW25113_ASM75055v1",
        "genotype_statement": "rrnB3 lacZ4787 hsdR514 (araBAD)567 (rhaBAD)568 rph-1",
        "provenance_gaps": [
            {"field": "provenance", "reason": "deferred_pending_source_review"}
        ],
    },
    # A phage challenge always has a dose, so the builder's bare `name` is refused: the
    # MOI is stated or it is a typed gap.
    "torchcell.datamodels.schema.PhagePerturbation": {
        "name": "T4",
        "multiplicity_of_infection": 0.01875,
        "ncbi_taxid": 10665,
    },
    # The generic builder fills `source_identifier` with a placeholder, which the route
    # validator refuses (an eck_crosswalk starts from another strain's locus tag).
    "torchcell.datamodels.schema.DerivedIdentifierMapping": {
        "source_identifier": "b0002",
        "route": "eck_crosswalk",
    },
    # The called-variant family (#731): the generic builder cannot invent a call whose
    # site id agrees with its own coordinates, nor a frequency with a stated basis, so
    # each one is committed. Values are rows of the Lim 2025 released matrix.
    "torchcell.datamodels.schema.BacterialVariantCall": {
        "variant_type": "snv",
        "type_statement": "SNP",
        "reference_sequence": "AE015451",
        "position_start": 3866001,
        "position_end": 3866001,
        "sequence_change": "G\u2192A",
        "annotation": "P293S (CCA\u2192TCA)",
        "call_mode": "clone",
        "frequency_statement": "1",
        "frequency": 1.0,
        "frequency_basis": "fraction",
    },
    "torchcell.datamodels.schema.SiteVariantCallPerturbation": {
        "systematic_gene_name": "AE015451:4586057",
        "perturbed_gene_name": "PP_4061, PP_4063",
        "gene_namespace": "pputida_kt2440_locus_tag",
        "call": _INTERGENIC_CALL_EXAMPLE,
        "site_kind": "intergenic",
        "flanking_systematic_gene_names": ["PP_4061", "PP_4063"],
        "flanking_gene_statement": "PP_4061, PP_4063",
    },
    "torchcell.datamodels.schema.SpanDeletionCallPerturbation": {
        "systematic_gene_name": "PP_4061",
        "perturbed_gene_name": "PP_4061",
        "gene_namespace": "pputida_kt2440_locus_tag",
        "call": _SPAN_CALL_EXAMPLE,
        "span_designation": "AE015451:4588139 \u03945,553 bp",
        "span_systematic_gene_names": ["PP_4061", "PP_4063"],
    },
    "torchcell.datamodels.schema.PromoterReplacementPerturbation": {
        "systematic_gene_name": "b0002",
        "perturbed_gene_name": "thrA",
        "gene_namespace": "ecoli_k12_mg1655_bnumber",
        "expression_direction": "increased",
        "promoter_name": "Ptac",
    },
    # An exposure states a dose or declares a typed gap; the generic builder sets
    # neither, so the example carries the one Shiver 2016 releases.
    "torchcell.datamodels.schema.PhysicalExposurePerturbation": {
        "factor": "radiation",
        "exposure_duration_seconds": 12.0,
    },
    "torchcell.datamodels.schema.VisualScorePhenotype": {
        "visual_score": 2.0,
        "n_replicates": 3,
        "score_scale_min": 0,
        "score_scale_max": 4,
        "score_semantics": "higher is more colored",
        "target_product": "beta-carotene",
    },
    # #1016: the single-cell UMI-count phenotype's totals must agree with its two dicts
    # and its per-cell lists, and its library partition with the per-cell libraries,
    # which the generic builder cannot arrange (Jackson 2020 shape: markers stored, the
    # stage partition a typed gap).
    "torchcell.datamodels.schema.SingleCellExpressionPhenotype": {
        "umi_count": {"YAL001C": 3},
        "marker_umi_count": {"KANMX": 1},
        "total_umis": 4,
        "n_features": 1,
        "feature_universe_sha256": "a" * 64,
        "n_cells": 1,
        "cell_cycle_stage_counts": None,
        "library_cell_counts": {"YPD": 1},
        "cell_barcodes": ["AAACCTGAGAAACCAT"],
        "cell_libraries": ["YPD"],
        "cell_total_umis": [4],
        "provenance_gaps": [
            {"field": "cell_cycle_stage_counts", "reason": "not_reported_by_primary"}
        ],
    },
    # The construct store's placement models (2026-10-15). The generic builder cannot
    # satisfy their cross-field invariants: a placement's two contig fields must pin ONE
    # sha256, a variant call must name the placement's own contig, and a host placement's
    # two gap-able spans are set or carry a ProvenanceGap, never a silent None. The
    # example is Cachera's Btx cassette at XII-5: a named landing pad with no published
    # coordinate, which is the common case and the reason the gap exists.
    "torchcell.datamodels.schema.HostPlacement": _HOST_PLACEMENT_EXAMPLE,
    "torchcell.datamodels.schema.ConstructVariantCall": _CONSTRUCT_CALL_EXAMPLE,
    "torchcell.datamodels.schema.ConstructPlacement": {
        "contig_sha256": _CONSTRUCT_CONTIG_SHA256,
        "contig_ref": {
            "tier": "library",
            "key": "cacheraCRISPAHighthroughputMethod2023",
            "path": "si/plasmids/pBTX002.gb",
            "sha256": _CONSTRUCT_CONTIG_SHA256,
            "bytes": 21901,
        },
        "feature_key": {
            "label": "ARO4-K229L",
            "so_id": "SO:0000316",
            "span": {"start": 3001, "end": 4113, "orientation": "inline"},
        },
        "host_placement": _HOST_PLACEMENT_EXAMPLE,
        "variants": [_CONSTRUCT_CALL_EXAMPLE],
    },
}

# class qualname -> why no example exists. Strict: a class listed here that DOES round
# trip fails its xfail, so the list decays.
UNCONSTRUCTIBLE: dict[str, str] = {
    "torchcell.datamodels.schema.Phenotype": (
        "abstract base: label_name must be a class attribute of a concrete subclass"
    ),
    "torchcell.datamodels.schema.TwoChannelMicroarrayExpressionPhenotype": (
        "abstract parent (#1022 #989): it declares no field, so no label_name can name "
        "one of its own"
    ),
    "torchcell.datamodels.schema.Experiment": (
        "abstract base: its phenotype field is the abstract Phenotype"
    ),
    "torchcell.datamodels.schema.ExperimentReference": (
        "abstract base: its phenotype_reference field is the abstract Phenotype"
    ),
    "torchcell.datamodels.conversion.ConversionEntry": (
        "holds classes and callables; not JSON-serializable by design"
    ),
    "torchcell.datamodels.conversion.ConversionMap": (
        "holds ConversionEntry callables; not JSON-serializable by design"
    ),
}


class Unbuildable(Exception):
    """The generic builder has no value for an annotation."""


def _models() -> dict[str, type[BaseModel]]:
    found: dict[str, type[BaseModel]] = {}
    for info in pkgutil.iter_modules(datamodels.__path__, "torchcell.datamodels."):
        if info.name in SKIP_MODULES:
            continue
        module = importlib.import_module(info.name)
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if (
                issubclass(obj, BaseModel)
                and obj is not BaseModel
                and obj.__module__ == info.name
            ):
                found[f"{obj.__module__}.{obj.__qualname__}"] = obj
    return dict(sorted(found.items()))


MODELS = _models()


def _strip_optional(annotation: Any) -> Any:
    origin = typing.get_origin(annotation)
    if origin in (typing.Union, types.UnionType):
        args = [a for a in typing.get_args(annotation) if a is not type(None)]
        return args[0] if args else type(None)
    if origin is typing.Annotated:
        return _strip_optional(typing.get_args(annotation)[0])
    return annotation


def _value(annotation: Any, depth: int) -> Any:
    """A minimal value for an annotation, or raise Unbuildable."""
    if depth > 8:
        raise Unbuildable(f"nesting deeper than 8 at {annotation!r}")
    annotation = _strip_optional(annotation)
    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)
    if origin is typing.Literal:
        return args[0]
    if inspect.isclass(annotation):
        if issubclass(annotation, BaseModel):
            return _example(annotation, depth + 1)
        if issubclass(annotation, enum.Enum):
            return next(iter(annotation))
        if annotation is bool:
            return True
        if annotation is int:
            return 1
        if annotation is float:
            return 0.5
        if annotation is str:
            return "x"
        if annotation is bytes:
            return b"x"
        if annotation is Path:
            return Path("x")
        if annotation is dt.datetime:
            return dt.datetime(2026, 1, 2, 3, 4, 5)
        if annotation is dt.date:
            return dt.date(2026, 1, 2)
        if annotation in (dict, list, set, tuple, frozenset):
            return annotation()
    if origin in (list, set, frozenset, typing.Sequence, typing.Iterable):
        return origin([_value(args[0], depth + 1)]) if args else []
    if origin is tuple:
        if len(args) == 2 and args[1] is Ellipsis:
            return (_value(args[0], depth + 1),)
        return tuple(_value(a, depth + 1) for a in args)
    if origin is dict or origin is typing.Mapping:
        if len(args) == 2:
            return {_value(args[0], depth + 1): _value(args[1], depth + 1)}
        return {}
    raise Unbuildable(f"no generic value for {annotation!r}")


def _example(cls: type[BaseModel], depth: int = 0) -> BaseModel:
    """An instance of ``cls`` from EXAMPLES or the generic builder."""
    name = f"{cls.__module__}.{cls.__qualname__}"
    if name in EXAMPLES:
        return cls.model_validate(EXAMPLES[name])
    kwargs = {
        field: FIELD_VALUES[field]
        if field in FIELD_VALUES
        else _value(info.annotation, depth)
        for field, info in cls.model_fields.items()
        if info.is_required()
    }
    return cls.model_validate(kwargs)


def _constructible(name: str) -> object:
    if name in UNCONSTRUCTIBLE:
        return pytest.param(
            name, marks=pytest.mark.xfail(strict=True, reason=UNCONSTRUCTIBLE[name])
        )
    return name


@pytest.mark.parametrize("name", [_constructible(n) for n in MODELS])
def test_round_trip_through_dict_and_json(name: str) -> None:
    """validate(dump(obj)) == obj by dict and by JSON; dump and schema are deterministic."""
    cls = MODELS[name]
    obj = _example(cls)
    dumped = obj.model_dump()
    assert cls.model_validate(dumped) == obj
    assert cls.model_validate_json(obj.model_dump_json()) == obj
    assert obj.model_dump() == dumped
    schema = obj.model_json_schema()
    assert json.loads(json.dumps(schema)) == schema
    assert cls.model_json_schema() == schema


def test_sweep_covers_every_model_in_the_package() -> None:
    """At least the 109 models present on 2026.09.26 are collected (guards the walker)."""
    assert len(MODELS) >= 109
    assert all(issubclass(cls, BaseModel) for cls in MODELS.values())


def test_example_and_unconstructible_keys_name_collected_models() -> None:
    """Every committed key refers to a class the walker found (no stale entries)."""
    assert set(EXAMPLES) <= set(MODELS), sorted(set(EXAMPLES) - set(MODELS))
    assert set(UNCONSTRUCTIBLE) <= set(MODELS), sorted(
        set(UNCONSTRUCTIBLE) - set(MODELS)
    )
    assert not set(EXAMPLES) & set(UNCONSTRUCTIBLE)


def test_examples_are_needed() -> None:
    """Every EXAMPLES entry exists because the generic builder fails for that class."""
    for name in EXAMPLES:
        cls = MODELS[name]
        with pytest.raises(Exception):  # noqa: B017 - ValidationError or Unbuildable
            kwargs = {
                field: FIELD_VALUES[field]
                if field in FIELD_VALUES
                else _value(info.annotation, 0)
                for field, info in cls.model_fields.items()
                if info.is_required()
            }
            cls.model_validate(kwargs)
