"""Genome reference implementations for sequence-based torchcell datasets.

``base`` holds the organism-agnostic ``AnnotatedGenome`` / ``AnnotatedGene`` (the
genomes-tier release, the recorded ``data.db`` cache, GO, gene-name resolution);
each organism subpackage (``scerevisiae``) subclasses them with its conventions.
``registry`` dereferences assembly-set ids to sha256-verified files.
"""
