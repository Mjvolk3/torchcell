torchcell.metabolism
====================

.. module:: torchcell.metabolism

.. currentmodule:: torchcell.metabolism

Genome-scale metabolic modeling. ``yeast_GEM`` wraps the yeast-GEM model (through cobra) and derives its reaction and metabolite graphs; ``constraints`` turns a genome-scale model into constraint tensors; ``flux_layer`` is a differentiable, enzyme-constrained flux layer; ``media`` maps an ontology ``Media`` object onto exchange bounds; ``pathway`` adds a heterologous pathway to a model as a typed perturbation; and ``enzyme_kinetics`` and ``parameters`` hold kinetic parameters with their provenance. Nothing is re-exported at the package level, because ``yeast_GEM`` imports cobra and downloads the model on first use; import the submodule you need.

.. contents:: Contents
    :local:

``betaxanthin``
---------------

The betaxanthin cassette Cachera transferred into the yeast knockout collection.

Functions
~~~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated

   betaxanthin.build_betaxanthin_pathway
   betaxanthin.betaxanthin_demand_ids

``constraints``
---------------

Genome-scale model -> constraint tensors, as pure functions of a GEM.

Classes
~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   constraints.ThermoMode
   constraints.TableCoverage
   constraints.ThermoTable
   constraints.CatalyticUnits
   constraints.GemTensors

Functions
~~~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated

   constraints.load_thermo_table
   constraints.parse_catalytic_units
   constraints.independent_balance_rows
   constraints.null_space_basis
   constraints.build_gem_tensors
   constraints.compare_reaction_delta_g

``enzyme_kinetics``
-------------------

Enzyme kinetic parameters (k_cat, K_M) with provenance, from the Open Enzyme Database.

Classes
~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   enzyme_kinetics.KineticKind
   enzyme_kinetics.KineticSource
   enzyme_kinetics.KineticRetrieval
   enzyme_kinetics.OedKineticRecord
   enzyme_kinetics.ResolvedKineticParameter

Functions
~~~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated

   enzyme_kinetics.fetch_oed_records
   enzyme_kinetics.mirror_oed_slice
   enzyme_kinetics.load_mirrored_records
   enzyme_kinetics.resolve_parameter
   enzyme_kinetics.index_by_uniprot

``flux_layer``
--------------

A differentiable, enzyme-constrained, thermodynamically-feasible flux layer.

Classes
~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   flux_layer.CompartmentParameters
   flux_layer.FluxLayerConfig
   flux_layer.FluxLayer

Functions
~~~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated

   flux_layer.gene_index_map

``media``
---------

Map an ontology ``Media`` object onto genome-scale-model exchange bounds.

Classes
~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   media.UptakePolicy
   media.ExchangeBound
   media.ComponentResolution
   media.MediaBounds
   media.MediaBoundsDiff
   media.ExchangeIndex

Functions
~~~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated

   media.build_exchange_index
   media.resolve_component
   media.media_to_bounds
   media.diff_bounds

``parameters``
--------------

Kinetic and physical parameters for a GEM, database-first and provenance-tagged.

Classes
~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   parameters.ParameterProvenance
   parameters.KcatPredictor
   parameters.PredictorRegistry
   parameters.ParameterTable

Functions
~~~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated

   parameters.load_swissprot
   parameters.molecular_weight_table
   parameters.uniprot_for_genes
   parameters.resolve_kcat_table
   parameters.load_measured_concentrations
   parameters.concentration_prior

``pathway``
-----------

Add a heterologous pathway to a genome-scale model, as a typed perturbation.

Classes
~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   pathway.EvidenceTier
   pathway.MetaboliteSpec
   pathway.ReactionSpec
   pathway.HeterologousPathway

Functions
~~~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated

   pathway.apply_pathway

``yeast_GEM``
-------------

Wrapper around the yeast-GEM metabolic model and its derived graphs.

Classes
~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   yeast_GEM.YeastGEM

Functions
~~~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated

   yeast_GEM.plot_reaction_map
   yeast_GEM.plot_full_network
   yeast_GEM.plot_random_network
   yeast_GEM.plot_bipartite_network
   yeast_GEM.sanity_check_metabolic_networks
   yeast_GEM.analyze_reactions_without_genes
   yeast_GEM.test_bipartite_attributes
