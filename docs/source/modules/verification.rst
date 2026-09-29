torchcell.verification
======================

.. module:: torchcell.verification

.. currentmodule:: torchcell.verification

Record-level verification of built datasets at five levels, L0 to L4. ``levels`` holds the reusable checks and ``common`` the rules every dataset family shares; each family module (``fitness``, ``expression``, ``morphology``, ``metabolite``, ``protein`` and others) applies them to one phenotype type. ``report`` defines the pydantic verification report, ``sourced`` binds a single extracted value to its source quote and sha256, and ``runners`` runs the verifiers over built datasets.

.. contents:: Contents
    :local:

Classes
-------

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   CarrierGapCensus
   SharedRecordRules
   DerivationMethod
   Level
   LevelResult
   Provenance
   StatDerivation
   VerificationReport
   SourcedValue
   ProvenanceGap
   ProvenanceGapReason
   ProvenanceGapCensus

Functions
---------

.. autosummary::
   :nosignatures:
   :toctree: ../generated

   shared_rule_results
   sha256_file
   audit_sourced_value
   library_available
   provenance_gap_census
   provenance_gap_level_result
   l1_provenance_gaps
   l0_structural
   l1_completeness
   l1_count
   l2_cross_method
   l2_value_fidelity
   l3_convention
   l4_cross_source
   measured_gene_universe
   verify_expression_dataset
   perturbed_gene_set
   verify_morphology_dataset
   verify_visual_score_dataset
   visual_score_gene_set
   metabolite_gene_set
   verify_metabolite_dataset
   fitness_gene_set
   verify_fitness_dataset
