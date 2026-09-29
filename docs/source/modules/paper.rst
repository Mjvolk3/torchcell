torchcell.paper
===============

.. module:: torchcell.paper

.. currentmodule:: torchcell.paper

Code behind the manuscript's generated artifacts. ``tables`` holds primitives that emit one table as both markdown and LaTeX; ``ontology_graph`` introspects the pydantic schema into a typed graph and ``ontology_svg`` lays it out as the zoomable SVG published at `/ontology/ <https://mjvolk3.github.io/torchcell/ontology/>`_; ``signal`` is a command-line tool that computes a built dataset's gzip signal and its derived shape and graph role.

.. contents:: Contents
    :local:

``ontology_graph``
------------------

Introspect the torchcell pydantic schema into a typed, renderable graph.

Classes
~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   ontology_graph.OntologyField
   ontology_graph.OntologyClass
   ontology_graph.CompositionEdge
   ontology_graph.OntologyGraph

Functions
~~~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated

   ontology_graph.build_ontology_graph

``ontology_svg``
----------------

Lay out and render :mod:`torchcell.paper.ontology_graph` as a zoomable SVG map.

Classes
~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   ontology_svg.Card
   ontology_svg.Lane
   ontology_svg.Layout

Functions
~~~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated

   ontology_svg.build_layout
   ontology_svg.render_schematic_svg
   ontology_svg.render_svg

``signal``
----------

CLI to compute a dataset's gzip 'signal' + derived shape/graph-role from its built LMDB. Use as new datasets are added, without touching the paper table.

Functions
~~~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated

   signal.resolve_lmdb

``tables``
----------

Reusable primitives for generating paper tables in BOTH markdown and LaTeX from a single in-code source of truth, plus data-derived columns such as a streaming-gzip "signal" (a Kolmogorov-complexity proxy) computed directly from a built LMDB.

Classes
~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   tables.Cell
   tables.DatasetSignalRecord
   tables.SignalCache
   tables.Column
   tables.Row
   tables.PaperTable

Functions
~~~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated

   tables.human_bytes
   tables.tex_escape
   tables.scientific
   tables.default_phenotype_bytes
   tables.instance_bytes
   tables.stream_gzip_signal
   tables.read_first_record
   tables.phenotype_descriptor
   tables.lmdb_fingerprint
   tables.read_frontmatter
