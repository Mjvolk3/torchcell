torchcell.database
==================

.. module:: torchcell.database

.. currentmodule:: torchcell.database

Commands that build and serve the torchcell Neo4j knowledge graph. The package exports :class:`~torchcell.database.build_command.BuildCommand`, a Cliff command that runs the database image build scripts; ``tcdb`` (the console script declared in ``pyproject.toml``) is the Cliff entry point in ``torchcell.database.tcdb``. Other modules create the Neo4j directory tree, combine BioCypher output directories into one import set, build a single registered dataset's LMDB for admission, and hold the client connection settings and the Neo4j Browser stylesheet.

.. contents:: Contents
    :local:

``biocypher_out_combine``
-------------------------

Combine multiple BioCypher output directories into a single Neo4j import set.

Functions
~~~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated

   biocypher_out_combine.merge_dicts
   biocypher_out_combine.check_yaml_compatibility
   biocypher_out_combine.remove_duplicates
   biocypher_out_combine.create_schema_info_csv
   biocypher_out_combine.combine_csv_files
   biocypher_out_combine.load_neo4j_config
   biocypher_out_combine.load_schema_info
   biocypher_out_combine.generate_neo4j_import_script

``browser_style``
-----------------

Neo4j Browser graph stylesheet (GRASS) for the served torchcell graph.

Classes
~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   browser_style.NodeRule
   browser_style.RelationshipRule
   browser_style.Caption
   browser_style.SeedNodeStyle
   browser_style.SeedState

Functions
~~~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated

   browser_style.schema_node_labels
   browser_style.lane_of
   browser_style.node_rules
   browser_style.render
   browser_style.seed_state
   browser_style.persisted_value
   browser_style.stylesheet_sha256
   browser_style.render_seed_js

``build_command``
-----------------

Cliff command for building the torchcell database image via shell scripts.

Classes
~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   build_command.BuildCommand

``build_dataset_lmdb``
----------------------

Build ONE registered dataset's LMDB in the dev tree, for knowledge-graph admission.

Functions
~~~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated

   build_dataset_lmdb.dataset_default_root
   build_dataset_lmdb.resolve_dataset_class
   build_dataset_lmdb.build_dataset

``connection``
--------------

Client-side Neo4j connection settings, resolved from the environment.

Classes
~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   connection.Neo4jConnectionSettings

Functions
~~~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated

   connection.neo4j_connection_settings

``tcdb``
--------

Cliff-based command-line entry point for the torchcell database.

Classes
~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   tcdb.TCDB
