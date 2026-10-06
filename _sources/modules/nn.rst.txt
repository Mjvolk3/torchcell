torchcell.nn
============

.. module:: torchcell.nn

.. currentmodule:: torchcell.nn

Neural-network layers shared by the models. ``hetero_nsa``, ``nsa_encoder``, ``masked_attention_block`` and ``self_attention_block`` build node-set attention over graphs with FlexAttention adjacency masks; ``masked_gin_conv`` is a GIN convolution for the lazy, masked subgraph representation; and ``stoichiometric_hypergraph_conv`` is a stoichiometry-aware hypergraph convolution over metabolic networks.

.. contents:: Contents
    :local:

``hetero_nsa``
--------------

Heterogeneous Node-Set Attention blocks and encoder over HeteroData graphs.

Classes
~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   hetero_nsa.HeteroNSA
   hetero_nsa.HeteroNSAEncoder

``masked_attention_block``
--------------------------

Masked attention blocks using FlexAttention with adjacency and edge masks.

Classes
~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   masked_attention_block.MaskedAttentionBlock
   masked_attention_block.NodeSelfAttention

``masked_gin_conv``
-------------------

Masked GIN convolution for lazy subgraph representation.

Classes
~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   masked_gin_conv.MaskedGINConv

``nsa_encoder``
---------------

Node self-attention encoder stacking masked and self-attention blocks.

Classes
~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   nsa_encoder.NSAEncoder

``self_attention_block``
------------------------

Device-adaptive self-attention block used across torchcell models.

Classes
~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   self_attention_block.SelfAttentionBlock

``stoichiometric_hypergraph_conv``
----------------------------------

Stoichiometry-aware hypergraph convolution layer for metabolic networks.

Classes
~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   stoichiometric_hypergraph_conv.StoichHypergraphConv
