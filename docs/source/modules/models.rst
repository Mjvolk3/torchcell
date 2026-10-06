torchcell.models
================

.. module:: torchcell.models

.. currentmodule:: torchcell.models

Model implementations, one module per architecture. They range from sequence language-model wrappers (Nucleotide Transformer, ESM-2, ProtT5, the fungal up/down-stream transformer) through set and graph models (DeepSet, GCN and GAT encoders, DiffPool and SAGPool variants) to the cell models used in the experiments: :class:`~torchcell.models.dcell.DCell`, the DANGO family (for example :class:`~torchcell.models.hetero_cell_bipartite_dango_gi.GeneInteractionDango` in ``hetero_cell_bipartite_dango_gi``), and the Cell Graph Transformer (:class:`~torchcell.models.cell_graph_transformer.CellGraphTransformer`, with its equivariant variant :class:`~torchcell.models.equivariant_cell_graph_transformer.CellGraphTransformer`). The package namespace re-exports a small subset (``torchcell.models.__all__``); every class below is listed under the module that defines it.

.. contents:: Contents
    :local:

``cell_graph_transformer``
--------------------------

Cell Graph Transformer with graph-regularized attention heads.

Classes
~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   cell_graph_transformer.GraphRegularizedTransformerLayer
   cell_graph_transformer.HyperSAGNN
   cell_graph_transformer.PerturbationHead
   cell_graph_transformer.CellGraphTransformer

Functions
~~~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated

   cell_graph_transformer.calculate_weight_l2_norm
   cell_graph_transformer.compute_smoothness

``cell_graph_transformer_metabolism``
-------------------------------------

CGT-Metabolism: the Cell Graph Transformer with production / metabolome readouts.

Classes
~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   cell_graph_transformer_metabolism.FluxMetaboliteHead
   cell_graph_transformer_metabolism.FluxScalarHead
   cell_graph_transformer_metabolism.ProductScalarHead
   cell_graph_transformer_metabolism.MetabolomeVectorHead
   cell_graph_transformer_metabolism.CellGraphTransformerMetabolism

Functions
~~~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated

   cell_graph_transformer_metabolism.perturbed_gene_pool

``dango``
---------

DANGO model: PPI-network pretraining, embedding integration, and HyperSAGNN.

Classes
~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   dango.DangoPreTrain
   dango.MetaEmbedding
   dango.HyperSAGNN
   dango.Dango

``dcell``
---------

DCell model: a GO-ontology-structured neural network over gene perturbations.

Classes
~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   dcell.DCell
   dcell.DCellSubsystem

``dcell_opt``
-------------

Optimized DCell model for torch.compile compatibility.

Classes
~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   dcell_opt.DCellOpt
   dcell_opt.DCellSubsystem

``deep_set``
------------

DeepSet model with permutation-invariant set aggregation over node features.

Classes
~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   deep_set.DeepSet

``diffusion_decoder``
---------------------

Diffusion-based decoder conditioning on graph embeddings via cross-attention.

Classes
~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   diffusion_decoder.SinusoidalTimeEmbedding
   diffusion_decoder.CrossAttention
   diffusion_decoder.DenoisingBlock
   diffusion_decoder.DiffusionDecoder

``equivariant_cell_graph_transformer``
--------------------------------------

Equivariant Cell Graph Transformer with graph-regularized attention heads.

Classes
~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   equivariant_cell_graph_transformer.GraphRegularizedTransformerLayer
   equivariant_cell_graph_transformer.HyperSAGNN
   equivariant_cell_graph_transformer.EquivariantPerturbationTransform
   equivariant_cell_graph_transformer.PerturbationGraphPropagation
   equivariant_cell_graph_transformer.LowRankBilinear
   equivariant_cell_graph_transformer.ObservedLabelEncoder
   equivariant_cell_graph_transformer.CrossGeneMixing
   equivariant_cell_graph_transformer.ResponseBasisHead
   equivariant_cell_graph_transformer.PerceiverMixing
   equivariant_cell_graph_transformer.PerturbationHead
   equivariant_cell_graph_transformer.GlobalHead
   equivariant_cell_graph_transformer.CrossAttnHead
   equivariant_cell_graph_transformer.PerGeneHead
   equivariant_cell_graph_transformer.PerMetaboliteHead
   equivariant_cell_graph_transformer.MaskedMultitaskLoss
   equivariant_cell_graph_transformer.CellGraphTransformer

Functions
~~~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated

   equivariant_cell_graph_transformer.calculate_weight_l2_norm
   equivariant_cell_graph_transformer.compute_smoothness

``esm2``
--------

ESM-2 protein language model wrapper producing per-residue or mean embeddings.

Classes
~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   esm2.Esm2

``fungal_up_down_transformer``
------------------------------

SpeciesLM-based transformer embedding fungal upstream/downstream DNA sequences.

Classes
~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   fungal_up_down_transformer.FungalUpDownTransformer

``gpu_edge_mask_generator``
---------------------------

GPU-based edge mask generator for zero-copy lazy subgraph representation.

Classes
~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   gpu_edge_mask_generator.GPUEdgeMaskGenerator

``hetero_cell_bipartite_dango``
-------------------------------

Bipartite hetero GNN with DANGO-style gene interaction attention.

Classes
~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   hetero_cell_bipartite_dango.GeneInteractionAttention
   hetero_cell_bipartite_dango.GeneInteractionPredictor
   hetero_cell_bipartite_dango.AttentionalGraphAggregation
   hetero_cell_bipartite_dango.PreProcessor
   hetero_cell_bipartite_dango.AttentionConvWrapper
   hetero_cell_bipartite_dango.HeteroCellBipartite

Functions
~~~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated

   hetero_cell_bipartite_dango.get_norm_layer

``hetero_cell_bipartite_dango_diff_gi``
---------------------------------------

Gene interaction model variant with a diffusion-based prediction head.

Classes
~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   hetero_cell_bipartite_dango_diff_gi.LinearDecoder
   hetero_cell_bipartite_dango_diff_gi.GeneInteractionDiff

``hetero_cell_bipartite_dango_gi``
----------------------------------

Heterogeneous bipartite cell-graph model for DANGO gene-interaction prediction.

Classes
~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   hetero_cell_bipartite_dango_gi.SelfAttentionGraphAggregation
   hetero_cell_bipartite_dango_gi.PairwiseGraphAggregation
   hetero_cell_bipartite_dango_gi.AggregationNormNotImplementedError
   hetero_cell_bipartite_dango_gi.HeteroConvAggregator
   hetero_cell_bipartite_dango_gi.AttentionalGraphAggregation
   hetero_cell_bipartite_dango_gi.DangoLikeHyperSAGNN
   hetero_cell_bipartite_dango_gi.GeneInteractionPredictor
   hetero_cell_bipartite_dango_gi.PreProcessor
   hetero_cell_bipartite_dango_gi.AttentionConvWrapper
   hetero_cell_bipartite_dango_gi.GeneInteractionDango

Functions
~~~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated

   hetero_cell_bipartite_dango_gi.get_activation
   hetero_cell_bipartite_dango_gi.require_no_aggregation_norm
   hetero_cell_bipartite_dango_gi.get_norm_layer
   hetero_cell_bipartite_dango_gi.create_conv_layer
   hetero_cell_bipartite_dango_gi.calculate_weight_l2_norm
   hetero_cell_bipartite_dango_gi.calculate_rolling_correlation

``hetero_cell_bipartite_dango_gi_lazy``
---------------------------------------

Lazy bipartite Dango gene-interaction model using masked, zero-copy message passing.

Classes
~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   hetero_cell_bipartite_dango_gi_lazy.SelfAttentionGraphAggregation
   hetero_cell_bipartite_dango_gi_lazy.PairwiseGraphAggregation
   hetero_cell_bipartite_dango_gi_lazy.HeteroConvAggregator
   hetero_cell_bipartite_dango_gi_lazy.AttentionalGraphAggregation
   hetero_cell_bipartite_dango_gi_lazy.DangoLikeHyperSAGNN
   hetero_cell_bipartite_dango_gi_lazy.GeneInteractionPredictor
   hetero_cell_bipartite_dango_gi_lazy.PreProcessor
   hetero_cell_bipartite_dango_gi_lazy.AttentionConvWrapper
   hetero_cell_bipartite_dango_gi_lazy.GeneInteractionDango

Functions
~~~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated

   hetero_cell_bipartite_dango_gi_lazy.get_norm_layer
   hetero_cell_bipartite_dango_gi_lazy.create_conv_layer
   hetero_cell_bipartite_dango_gi_lazy.calculate_weight_l2_norm
   hetero_cell_bipartite_dango_gi_lazy.calculate_rolling_correlation

``hetero_cell_nsa_retry``
-------------------------

Heterogeneous cell model using Node-Set Attention blocks over gene/reaction/metabolite graphs.

Classes
~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   hetero_cell_nsa_retry.AttentionalGraphAggregation
   hetero_cell_nsa_retry.PreProcessor
   hetero_cell_nsa_retry.HeteroCellNSA

Functions
~~~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated

   hetero_cell_nsa_retry.get_norm_layer

``linear``
----------

Simple set-aggregation linear model over scatter-pooled node features.

Classes
~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   linear.SimpleLinearModel

``llm``
-------

Abstract base classes for nucleotide and peptide language models.

Classes
~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   llm.NucleotideModel
   llm.PeptideModel
   llm.pretrained_LLM

``mlp``
-------

Configurable multilayer perceptron with optional normalization and activations.

Classes
~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   mlp.Mlp

``nucleotide_transformer``
--------------------------

Nucleotide Transformer wrapper for embedding DNA sequences.

Classes
~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   nucleotide_transformer.NucleotideTransformer

``protT5``
----------

ProtT5 protein language model wrapper for embedding amino-acid sequences.

Classes
~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   protT5.ProtT5

``self_attention_deep_set``
---------------------------

Self-attention Deep Sets model for permutation-invariant set encoding.

Classes
~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   self_attention_deep_set.SelfAttention
   self_attention_deep_set.SelfAttentionDeepSet
