torchcell.provenance
====================

.. module:: torchcell.provenance

.. currentmodule:: torchcell.provenance

Schema-dependency tracking, which decides when a built dataset must be rebuilt because the schema changed. ``schema_impact`` diffs the working-tree schema against a git ref, classifies each change as breaking or stale, and names the loaders that must rebuild (it runs as a pre-commit hook). ``build_manifest`` stores, next to each built LMDB, the contract fingerprints of the schema symbols its loader depends on, and its ``check`` reports which built datasets are stale. A fingerprint hashes a symbol's contract (field shape and validators), not its source text, so a docstring edit flags nothing. ``schema_deps`` is the shared AST analysis.

.. contents:: Contents
    :local:

Classes
-------

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   ContractSpec
   FieldSpec
   SchemaSurface
   ChangeKind
   ImpactReport
   LoaderImpact
   SymbolChange
   BuildManifest
   DatasetCheck
   StaleResult
   SymbolDrift

Functions
---------

.. autosummary::
   :nosignatures:
   :toctree: ../generated

   contract_spec
   fingerprint
   forward_closure
   load_default_surface
   load_surface
   load_surface_from_sources
   loader_closure
   loader_schema_deps
   build_impact_report
   classify_change
   diff_surfaces
   check_all
   check_manifest
   compute_manifest
   write_build_manifest
