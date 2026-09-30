torchcell.graph
===============

.. module:: torchcell.graph

.. currentmodule:: torchcell.graph

Gene graphs for *S. cerevisiae*. :class:`~torchcell.graph.SCerevisiaeGraph` assembles the Gene Ontology DAG, SGD physical and genetic interaction networks, regulatory networks and STRING networks over the genes of a genome; :class:`~torchcell.graph.GeneGraph` and :class:`~torchcell.graph.GeneMultiGraph` wrap the resulting NetworkX graphs as named, typed containers that the cell datasets consume. The ``filter_*`` functions prune the GO DAG (by annotation date, by contained genes, by evidence code, and by redundant terms).

.. contents:: Contents
    :local:

Classes
-------

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   SCerevisiaeGraph
   GeneGraph
   GeneMultiGraph

Functions
---------

.. autosummary::
   :nosignatures:
   :toctree: ../generated

   filter_go_IGI
   filter_by_date
   filter_by_contained_genes
   filter_redundant_terms
   build_gene_multigraph
