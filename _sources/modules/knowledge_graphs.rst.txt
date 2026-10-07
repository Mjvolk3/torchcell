torchcell.knowledge_graphs
==========================

.. module:: torchcell.knowledge_graphs

.. currentmodule:: torchcell.knowledge_graphs

Builders for the BioCypher knowledge graph and the manifest that governs changes to the served graph. ``dataset_adapter_map`` pairs each dataset class with its adapter, and the ``create_*`` modules run a BioCypher build over a configured set of datasets. ``kg_manifest`` records, per served dataset, the schema fingerprints and adapter code the graph was built under, and decides whether a new dataset can be admitted by incremental import or requires a full rebuild; ``incremental_import`` turns one BioCypher output directory into a Neo4j incremental import; ``releases`` names and lists served releases. The package resolves its two listed submodules lazily, because each imports every dataset loader and adapter.

.. contents:: Contents
    :local:

``build_time_projection``
-------------------------

Project the tcdb knowledge-graph BUILD (CSV-generation) time for any subset config.

Classes
~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   build_time_projection.AdapterTiming
   build_time_projection.CalibrationTimings
   build_time_projection.SubsetConfig
   build_time_projection.DatasetSize
   build_time_projection.AdapterRate
   build_time_projection.AdapterContribution
   build_time_projection.BuildTimeProjection

Functions
~~~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated

   build_time_projection.load_timings
   build_time_projection.dataset_sizes
   build_time_projection.calibrate
   build_time_projection.project_build_time
   build_time_projection.gather_dataset_full_records

``create_kg``
-------------

Build a S. cerevisiae BioCypher knowledge graph from configured datasets.

Functions
~~~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated

   create_kg.get_num_workers

``create_scerevisiae_kg``
-------------------------

Build the S. cerevisiae BioCypher knowledge graph from torchcell datasets.

Functions
~~~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated

   create_scerevisiae_kg.get_num_workers

``create_scerevisiae_kg_small``
-------------------------------

Build a small S. cerevisiae BioCypher knowledge graph from Costanzo/Kuzmin data.

Functions
~~~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated

   create_scerevisiae_kg_small.get_num_workers

``gene_interactions_scerevisae_kg``
-----------------------------------

Build the S. cerevisiae gene-interaction BioCypher knowledge graph.

Functions
~~~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated

   gene_interactions_scerevisae_kg.get_num_workers

``head_ontology``
-----------------

The sha256-pinned local mirror of BioCypher's head ontology (Biolink 3.2.1).

Classes
~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   head_ontology.HeadOntologyError
   head_ontology.HeadOntologyMirror

Functions
~~~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated

   head_ontology.record_path
   head_ontology.load_mirror_record
   head_ontology.configured_head_ontology
   head_ontology.verify_head_ontology
   head_ontology.build_mirror_record

``incremental_import``
----------------------

Turn one BioCypher output directory into a Neo4j *incremental* import.

Classes
~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   incremental_import.CsvGroup
   incremental_import.ReferenceAnalysis
   incremental_import.IncrementalImportPlan
   incremental_import.ExistingEdgeFilter

Functions
~~~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated

   incremental_import.discover_csv_groups
   incremental_import.incremental_node_header
   incremental_import.write_incremental_headers
   incremental_import.constraints_cypher
   incremental_import.analyze_references
   incremental_import.incremental_import_call
   incremental_import.prepare_incremental_import
   incremental_import.query_existing_edges
   incremental_import.filter_existing_edges

``kg_manifest``
---------------

The served knowledge graph's build manifest, and the admission check for adding a dataset to it without rebuilding the others.

Classes
~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   kg_manifest.GraphSchemaEntry
   kg_manifest.SupersetLineage
   kg_manifest.KgDatasetEntry
   kg_manifest.KgEvent
   kg_manifest.KgBuildManifest
   kg_manifest.ServedDrift
   kg_manifest.AdapterDrift
   kg_manifest.SupersetCheck
   kg_manifest.AdmissionReport
   kg_manifest.BatchAdmissionReport

Functions
~~~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated

   kg_manifest.graph_schema_from_yaml
   kg_manifest.cell_adapter_surface
   kg_manifest.adapter_file_relpaths
   kg_manifest.dataset_adapter_files
   kg_manifest.dataset_conf_methods
   kg_manifest.loader_relpath
   kg_manifest.surface_at_ref
   kg_manifest.surface_in_worktree
   kg_manifest.value_surface_from_sources
   kg_manifest.value_surface_in_worktree
   kg_manifest.value_surface_at_ref
   kg_manifest.value_surface_drift
   kg_manifest.closure_at_ref
   kg_manifest.closure_in_worktree
   kg_manifest.package_version_from_source
   kg_manifest.package_version_at_ref
   kg_manifest.package_tag_at_ref
   kg_manifest.checkout_package_version
   kg_manifest.load_manifest
   kg_manifest.save_manifest
   kg_manifest.bootstrap_manifest
   kg_manifest.experiment_node_id
   kg_manifest.dev_experiment_ids
   kg_manifest.superset_check
   kg_manifest.live_experiment_ids
   kg_manifest.adapter_drift_against
   kg_manifest.check_admission
   kg_manifest.batch_report_from_members
   kg_manifest.check_batch_admission
   kg_manifest.record_admission
   kg_manifest.record_batch_admission
   kg_manifest.live_dataset_counts
   kg_manifest.live_neo4j_version
   kg_manifest.format_report
   kg_manifest.format_batch_report
   kg_manifest.load_report
   kg_manifest.split_dataset_args
   kg_manifest.parse_n_experiments

``release_snapshot``
--------------------

Committed snapshots of served knowledge-graph releases.

Classes
~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   release_snapshot.SnapshotDataset
   release_snapshot.SnapshotEvent
   release_snapshot.KgReleaseSnapshot

Functions
~~~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated

   release_snapshot.composite_sha256
   release_snapshot.snapshot_from_manifest
   release_snapshot.bootstrap_package_version
   release_snapshot.pair_package_tag
   release_snapshot.snapshot_paths
   release_snapshot.write_snapshot
   release_snapshot.load_snapshot
   release_snapshot.load_snapshots
   release_snapshot.load_closures

``releases``
------------

Knowledge-graph releases: what version a served store is, which datasets it holds, and whether a dataset's bytes changed between two versions.

Classes
~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   releases.ReleaseDataset
   releases.KgRelease
   releases.ServedDatabase
   releases.ReleaseDiff
   releases.DatasetDrift
   releases.ReleaseCompatibility
   releases.IncompatibleReleaseError

Functions
~~~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated

   releases.release_id
   releases.next_version
   releases.content_sha256
   releases.content_hashes_from_csv
   releases.content_hashes_from_store
   releases.read_release
   releases.write_release
   releases.release_from_manifest
   releases.stamp_manifest
   releases.list_databases
   releases.list_databases_bounded
   releases.resolve_database
   releases.datasets
   releases.diff
   releases.compatibility
   releases.compatibility_with_surface
   releases.closure_compatibility
   releases.require_paired
   releases.package_checkout
   releases.commit_index
   releases.commit_date
   releases.behind_main
   releases.package_label
   releases.status_rows
   releases.format_table

``subset``
----------

Prefilter and subsample datasets for KG builds.

Classes
~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   subset.RecordFilter

Functions
~~~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated

   subset.load_gene_sets
   subset.select_indices
   subset.subset_dataset

``supported_queries.check``
---------------------------

The supported-query drift check: does each registered query still hold on a release?

Classes
~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   supported_queries.check.QueryDrift
   supported_queries.check.QueryResult
   supported_queries.check.CheckReport
   supported_queries.check.GraphLabels

Functions
~~~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated

   supported_queries.check.pascal_label
   supported_queries.check.schema_is_a
   supported_queries.check.graph_labels
   supported_queries.check.selected_composite
   supported_queries.check.query_drifts
   supported_queries.check.run_check
   supported_queries.check.checkout_commit
   supported_queries.check.resolve_snapshot
   supported_queries.check.check_repo
   supported_queries.check.validate_query

``supported_queries.cypher_deps``
---------------------------------

What a Cypher query reads from the graph, extracted from its text.

Functions
~~~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated

   supported_queries.cypher_deps.strip_comments
   supported_queries.cypher_deps.blank_literals
   supported_queries.cypher_deps.split_blocks
   supported_queries.cypher_deps.extract_dependencies

``supported_queries.registry``
------------------------------

The supported-query registry: which shipped ``.cql`` queries the project stands behind.

Classes
~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   supported_queries.registry.QueryDependencies
   supported_queries.registry.SupportedQuery
   supported_queries.registry.QueryRegistry

Functions
~~~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated

   supported_queries.registry.registry_path

Not documented
--------------

Submodules left out of this page:

- ``conf`` (configuration files only)
