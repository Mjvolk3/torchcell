torchcell.sequence
==================

.. module:: torchcell.sequence

.. currentmodule:: torchcell.sequence

Genome and sequence data structures. :class:`~torchcell.sequence.Genome` is the abstract genome interface (lazy gene-set and sequence access), :class:`~torchcell.sequence.Gene` a gene within it, and :class:`~torchcell.sequence.DnaWindowResult` and :class:`~torchcell.sequence.DnaSelectionResult` the sequence windows the embedding datasets consume. ``torchcell.sequence.genome.base`` holds the organism-agnostic ``AnnotatedGenome`` (the recorded ``data.db`` cache, GO, gene-name resolution) that each host subclasses; the *S. cerevisiae* S288C implementation lives in ``torchcell.sequence.genome.scerevisiae``, and ``torchcell.sequence.genome.registry`` resolves the sha256-pinned reference genome files.

.. contents:: Contents
    :local:

Classes
-------

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   DnaSelectionResult
   DnaWindowResult
   Gene
   GeneSet
   Genome
   ParsedGenome

Functions
---------

.. autosummary::
   :nosignatures:
   :toctree: ../generated

   calculate_window_bounds
   calculate_window_bounds_symmetric
   compute_codon_frequency
   get_chr_from_description
   mismatch_positions
   roman_to_int
