torchcell.datamodels
====================

.. module:: torchcell.datamodels

.. currentmodule:: torchcell.datamodels

The pydantic schema of a torchcell experiment. A record is a typed ``genotype x environment -> phenotype`` experiment: ``Genotype`` holds a list of gene perturbations (for example ``SgaKanMxDeletionPerturbation``), ``Environment`` holds the medium, an optional temperature and any environment perturbations, and each phenotype family (fitness, gene interaction, CalMorph morphology, microarray, RNA-seq and pseudobulk expression, metabolite, protein abundance, visual score, environment response and others) has its own phenotype, experiment and experiment-reference classes in ``schema``. The classes derive from the strict bases in ``pydant`` (``ModelStrict`` and ``ModelStrictArbitrary``). Dataset loaders emit these objects, and the converter modules map one experiment type onto another (for example gene essentiality onto fitness). The package namespace re-exports a subset (``torchcell.datamodels.__all__``); every class below is listed under the module that defines it.

.. contents:: Contents
    :local:

``compound_identity``
---------------------

Shared, pure, offline compound-identity resolver (UI-2).

Classes
~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   compound_identity.CompoundResolutionStatus
   compound_identity.CompoundIdentityRecord
   compound_identity.CompoundIdentityResolution

Functions
~~~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated

   compound_identity.normalize_compound_name
   compound_identity.resolve_compound_identity
   compound_identity.inchikey_from_smiles
   compound_identity.resolve_compound_identity_from_smiles
   compound_identity.resolved_compound

``compound_identity_curate``
----------------------------

Reproducible curator for the pinned compound-identity table (UI-2, serve-50).

Classes
~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   compound_identity_curate.CurationDirective
   compound_identity_curate.PubChemProperty
   compound_identity_curate.PubChemClient
   compound_identity_curate.CuratedRow

Functions
~~~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated

   compound_identity_curate.parse_line
   compound_identity_curate.read_name_list
   compound_identity_curate.read_cid_list
   compound_identity_curate.name_property_url
   compound_identity_curate.cid_property_url
   compound_identity_curate.synonyms_url
   compound_identity_curate.assemble
   compound_identity_curate.serialize
   compound_identity_curate.curate

``conversion``
--------------

Convert raw Neo4j experiment records into typed experiments stored in LMDB.

Classes
~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   conversion.ConversionEntry
   conversion.ConversionMap
   conversion.Converter

``fitness_composite_conversion``
--------------------------------

Converter that maps gene essentiality and synthetic lethality data into fitness.

Classes
~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   fitness_composite_conversion.CompositeFitnessConverter

``gene_essentiality_to_fitness_conversion``
-------------------------------------------

Convert gene-essentiality experiments into equivalent fitness experiments.

Classes
~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   gene_essentiality_to_fitness_conversion.GeneEssentialityToFitnessConverter

Functions
~~~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated

   gene_essentiality_to_fitness_conversion.gene_essentiality_to_fitness_experiment
   gene_essentiality_to_fitness_conversion.gene_essentiality_to_fitness_reference

``identity``
------------

Composition-based identity for the environment-side entities of the graph.

Functions
~~~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated

   identity.compound_identity_key
   identity.media_identity
   identity.temperature_identity
   identity.environment_perturbation_identity
   identity.environment_identity
   identity.identity_sha256

``media``
---------

Reusable, provenance-first definitions of the common growth media.

Functions
~~~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated

   media.dropout

``ontology_checks``
-------------------

Programmatic coherence checks over the torchcell ontology.

Classes
~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   ontology_checks.AdapterNodeSite
   ontology_checks.PropertyMismatch
   ontology_checks.OptionalTraversal
   ontology_checks.PhenotypeLabelMap
   ontology_checks.CompoundIdentityIssue
   ontology_checks.MediaBaseIssue
   ontology_checks.MediaDerivationIssue
   ontology_checks.EnvironmentCompoundCollision
   ontology_checks.JoinKeyCensus

Functions
~~~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated

   ontology_checks.schema_models
   ontology_checks.schema_enums
   ontology_checks.composition_targets
   ontology_checks.experiment_closure
   ontology_checks.orphan_models
   ontology_checks.lane_membership
   ontology_checks.lane_back_edges
   ontology_checks.sourced_value_fields
   ontology_checks.enum_value_collisions
   ontology_checks.graph_schema
   ontology_checks.cell_adapter_source
   ontology_checks.isolated_graph_node_classes
   ontology_checks.phenotype_label_map
   ontology_checks.adapter_node_sites
   ontology_checks.adapter_property_mismatches
   ontology_checks.adapter_optional_traversals
   ontology_checks.compound_has_identity
   ontology_checks.compound_has_identity_gap
   ontology_checks.media_library_compound_issues
   ontology_checks.media_base_issues
   ontology_checks.media_derivation_issues
   ontology_checks.environment_compound_collisions
   ontology_checks.join_key_audit

``pydant``
----------

Strict Pydantic base models used across torchcell data schemas.

Classes
~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   pydant.ModelStrict
   pydant.ModelStrictArbitrary

``schema``
----------

Pydantic data models for torchcell genotypes, environments, and phenotypes.

Classes
~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   schema.ReferenceGenome
   schema.SOTerm
   schema.GenePerturbation
   schema.PresenceAbsencePerturbation
   schema.SequencePerturbation
   schema.ExpressionRangeMultiplier
   schema.CrisprConstruct
   schema.DeletionPerturbation
   schema.KanMxDeletionPerturbation
   schema.BarcodedKanMxDeletionPerturbation
   schema.NatMxDeletionPerturbation
   schema.SgaKanMxDeletionPerturbation
   schema.SgaNatMxDeletionPerturbation
   schema.DampPerturbation
   schema.SgaDampPerturbation
   schema.TsAllelePerturbation
   schema.AllelePerturbation
   schema.SuppressorAllelePerturbation
   schema.SgaSuppressorAllelePerturbation
   schema.SgaTsAllelePerturbation
   schema.SgaAllelePerturbation
   schema.MeanDeletionPerturbation
   schema.MarkerDeletionPerturbation
   schema.CrisprDeletionPerturbation
   schema.GeneAdditionPerturbation
   schema.NaturalGeneAbsencePerturbation
   schema.NaturalGenePresencePerturbation
   schema.SequenceVariantPerturbation
   schema.CopyNumberVariantPerturbation
   schema.EngineeredCopyNumberPerturbation
   schema.ExpressionModulationPerturbation
   schema.CrisprActivationPerturbation
   schema.CrisprInterferencePerturbation
   schema.Genotype
   schema.TemperatureUnit
   schema.Temperature
   schema.ConcentrationUnit
   schema.DoseBasis
   schema.PhysicalFactor
   schema.ProvenanceGapMixin
   schema.Compound
   schema.Concentration
   schema.Solvent
   schema.MediaComponentRole
   schema.ComponentDefinition
   schema.MediaComponent
   schema.Media
   schema.EnvironmentPerturbation
   schema.SmallMoleculePerturbation
   schema.EnvironmentPhysicalPerturbation
   schema.BiologicAgentClass
   schema.BiologicPerturbation
   schema.Environment
   schema.Phenotype
   schema.UncertaintyType
   schema.SampleUnit
   schema.FitnessPhenotype
   schema.GeneEssentialityPhenotype
   schema.SyntheticLethalityPhenotype
   schema.SyntheticRescuePhenotype
   schema.GeneInteractionPhenotype
   schema.CalMorphPhenotype
   schema.Publication
   schema.ExperimentReference
   schema.Experiment
   schema.FitnessExperimentReference
   schema.FitnessExperiment
   schema.GeneInteractionExperimentReference
   schema.GeneInteractionExperiment
   schema.GeneEssentialityExperimentReference
   schema.GeneEssentialityExperiment
   schema.SyntheticLethalityExperimentReference
   schema.SyntheticLethalityExperiment
   schema.SyntheticRescueExperimentReference
   schema.SyntheticRescueExperiment
   schema.CalMorphExperimentReference
   schema.CalMorphExperiment
   schema.MicroarrayExpressionPhenotype
   schema.MicroarrayExpressionExperimentReference
   schema.MicroarrayExpressionExperiment
   schema.RNASeqExpressionPhenotype
   schema.RNASeqExpressionExperimentReference
   schema.RNASeqExpressionExperiment
   schema.PseudobulkExpressionPhenotype
   schema.PseudobulkExpressionExperimentReference
   schema.PseudobulkExpressionExperiment
   schema.VisualScorePhenotype
   schema.VisualScoreExperimentReference
   schema.VisualScoreExperiment
   schema.MetabolitePhenotype
   schema.MetaboliteExperimentReference
   schema.MetaboliteExperiment
   schema.ProteinAbundancePhenotype
   schema.ProteinAbundanceExperimentReference
   schema.ProteinAbundanceExperiment
   schema.AssayType
   schema.MeasurementType
   schema.ResponseCategory
   schema.EnvironmentResponsePhenotype
   schema.EnvironmentResponseExperimentReference
   schema.EnvironmentResponseExperiment
   schema.HaplotypeBlock
   schema.SegregantParent
   schema.SegregantGenotype
   schema.SegregantGrowthExperimentReference
   schema.SegregantGrowthExperiment

Functions
~~~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated

   schema.derive_se

``synthetic_lethality_to_fitness_conversion``
---------------------------------------------

Convert synthetic-lethality experiments into equivalent fitness experiments.

Classes
~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   synthetic_lethality_to_fitness_conversion.SyntheticLethalityToFitnessConverter

Functions
~~~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated

   synthetic_lethality_to_fitness_conversion.synthetic_lethality_to_fitness_experiment
   synthetic_lethality_to_fitness_conversion.synthetic_lethality_to_fitness_reference
