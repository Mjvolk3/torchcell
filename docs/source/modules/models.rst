torchcell.models
================

.. module:: torchcell.models

.. currentmodule:: torchcell.models

Model implementations, one module per architecture. They range from sequence language-model wrappers (Nucleotide Transformer, ESM-2, ProtT5, the fungal up/down-stream transformer) through set and graph models (DeepSet, GCN and GAT encoders, DiffPool and SAGPool variants) to the cell models used in the experiments: :class:`~torchcell.models.dcell.DCell`, the DANGO family (for example :class:`~torchcell.models.hetero_cell_bipartite_dango_gi.GeneInteractionDango` in ``hetero_cell_bipartite_dango_gi``), and the Cell Graph Transformer (:class:`~torchcell.models.cell_graph_transformer.CellGraphTransformer`, with its equivariant variant :class:`~torchcell.models.equivariant_cell_graph_transformer.CellGraphTransformer`). The package namespace re-exports a small subset (``torchcell.models.__all__``); every class below is listed under the module that defines it.

.. contents:: Contents
    :local:

``cell_diffpool_dense``
-----------------------

Dense DiffPool models over cell graphs for hierarchical graph pooling.

Classes
~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   cell_diffpool_dense.DenseDiffPool
   cell_diffpool_dense.DenseCellDiffPool

Functions
~~~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated

   cell_diffpool_dense.load_sample_data_batch

``cell_diffpool_sparse``
------------------------

Sparse DiffPool cell model with GAT-based pooling over multiple graphs.

Classes
~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   cell_diffpool_sparse.SingleDiffPool
   cell_diffpool_sparse.CellDiffPool

Functions
~~~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated

   cell_diffpool_sparse.from_dense_batch
   cell_diffpool_sparse.load_sample_data_batch

``cell_gin_diffpool_dense``
---------------------------

Dense GIN-based DiffPool cell model over dense adjacency matrices.

Classes
~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   cell_gin_diffpool_dense.DenseDiffPool
   cell_gin_diffpool_dense.DenseCellDiffPool

Functions
~~~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated

   cell_gin_diffpool_dense.load_sample_data_batch

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

``cell_latent_perturbation``
----------------------------

Latent-perturbation cell model: hetero GNN pooling over gene and metabolism graphs.

Classes
~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   cell_latent_perturbation.ProjectedGATConv
   cell_latent_perturbation.PredictionHead
   cell_latent_perturbation.HeteroGnnPool
   cell_latent_perturbation.SetTransformer
   cell_latent_perturbation.SetNet
   cell_latent_perturbation.MetabolismProcessor
   cell_latent_perturbation.CellLatentPerturbation

Functions
~~~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated

   cell_latent_perturbation.load_sample_data_batch
   cell_latent_perturbation.plot_correlations

``cell_latent_perturbation_tform``
----------------------------------

Cell latent perturbation model with set-transformer and metabolism encoders.

Classes
~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   cell_latent_perturbation_tform.ProjectedGATConv
   cell_latent_perturbation_tform.PredictionHead
   cell_latent_perturbation_tform.HeteroGnnPool
   cell_latent_perturbation_tform.SetTransformer
   cell_latent_perturbation_tform.SetNet
   cell_latent_perturbation_tform.MetabolismProcessor
   cell_latent_perturbation_tform.CellLatentPerturbation

Functions
~~~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated

   cell_latent_perturbation_tform.load_sample_data_batch
   cell_latent_perturbation_tform.plot_correlations

``cell_latent_perturbation_unified``
------------------------------------

Unified cell latent perturbation model over gene, reaction, and metabolism graphs.

Classes
~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   cell_latent_perturbation_unified.WholeIntactProcessor
   cell_latent_perturbation_unified.PerturbedProcessor
   cell_latent_perturbation_unified.BaseGenePreprocessor
   cell_latent_perturbation_unified.ReactionGeneProcessor
   cell_latent_perturbation_unified.ProjectedGATConv
   cell_latent_perturbation_unified.PredictionHead
   cell_latent_perturbation_unified.HeteroGnn
   cell_latent_perturbation_unified.SetNet
   cell_latent_perturbation_unified.MetabolismProcessor
   cell_latent_perturbation_unified.CellLatentPerturbation

Functions
~~~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated

   cell_latent_perturbation_unified.load_sample_data_batch
   cell_latent_perturbation_unified.plot_correlations

``cell_sagpool``
----------------

Self-attention graph pooling (SAGPool) models for cell graphs.

Classes
~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   cell_sagpool.SingleSAGPool
   cell_sagpool.CellSAGPool

Functions
~~~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated

   cell_sagpool.load_sample_data_batch
   cell_sagpool.analyze_node_selections

``cell_sagpool_inception``
--------------------------

Inception-style multi-graph SAGPooling model for cell graph regression.

Classes
~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   cell_sagpool_inception.MLP
   cell_sagpool_inception.SingleSAGPool
   cell_sagpool_inception.CellSAGPool

Functions
~~~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated

   cell_sagpool_inception.load_sample_data_batch
   cell_sagpool_inception.analyze_node_selections

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

``dense_gat_conv``
------------------

Dense (adjacency-matrix) implementation of the graph attention layer.

Classes
~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   dense_gat_conv.DenseGATConv

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

``early_cell_diffpool_dense``
-----------------------------

Dense DiffPool cell model over multiple per-graph GAT stacks.

Classes
~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   early_cell_diffpool_dense.EarlyDenseCellDiffPool

Functions
~~~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated

   early_cell_diffpool_dense.load_sample_data_batch

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

``gat_diffpool``
----------------

GAT-then-DiffPool model that pools each input graph into a graph embedding.

Classes
~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   gat_diffpool.GatDiffPool

Functions
~~~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated

   gat_diffpool.load_sample_data_batch

``gat_diffpool_alt``
--------------------

Multi-graph GATv2 encoder with hierarchical DiffPool clustering.

Classes
~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   gat_diffpool_alt.GatDiffPool

Functions
~~~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated

   gat_diffpool_alt.load_sample_data_batch

``gat_diffpool_inception``
--------------------------

GAT-with-DiffPool inception model over multiple graphs for graph-level prediction.

Classes
~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   gat_diffpool_inception.GatDiffPool

Functions
~~~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated

   gat_diffpool_inception.load_sample_data_batch

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

``graph_attention``
-------------------

Graph attention network combining a DeepSet encoder with GATv2 layers.

Classes
~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   graph_attention.GraphAttention

``graph_convolution``
---------------------

DeepSet node encoder followed by GCN message passing and set aggregation.

Classes
~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   graph_convolution.GraphConvolution

``hetero_cell``
---------------

Heterogeneous GNN over gene, reaction, and metabolite graphs for fitness.

Classes
~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   hetero_cell.AttentionalGraphAggregation
   hetero_cell.PreProcessor
   hetero_cell.AttentionConvWrapper
   hetero_cell.HeteroCell

Functions
~~~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated

   hetero_cell.get_norm_layer

``hetero_cell_bipartite``
-------------------------

Bipartite hetero GNN over gene, reaction, and metabolite graphs.

Classes
~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   hetero_cell_bipartite.AttentionalGraphAggregation
   hetero_cell_bipartite.PreProcessor
   hetero_cell_bipartite.AttentionConvWrapper
   hetero_cell_bipartite.HeteroCellBipartite

Functions
~~~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated

   hetero_cell_bipartite.get_norm_layer

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

``hetero_cell_flex``
--------------------

Flexible heterogeneous cell model over gene, reaction, and metabolite graphs.

Classes
~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   hetero_cell_flex.GraphMaskedAttention
   hetero_cell_flex.Combiner
   hetero_cell_flex.AttentionalGraphAggregation
   hetero_cell_flex.PreProcessor
   hetero_cell_flex.AttentionConvWrapper
   hetero_cell_flex.HeteroCell

Functions
~~~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated

   hetero_cell_flex.get_norm_layer
   hetero_cell_flex.load_sample_data_batch
   hetero_cell_flex.plot_correlations

``hetero_cell_isab_split``
--------------------------

Heterogeneous cell model with ISAB set-transformer aggregation splits.

Classes
~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   hetero_cell_isab_split.SortedSetTransformerAggregation
   hetero_cell_isab_split.PreProcessor
   hetero_cell_isab_split.AttentionConvWrapper
   hetero_cell_isab_split.HeteroCell

Functions
~~~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated

   hetero_cell_isab_split.get_norm_layer
   hetero_cell_isab_split.load_sample_data_batch
   hetero_cell_isab_split.plot_correlations
   hetero_cell_isab_split.plot_embeddings

``hetero_cell_nsa``
-------------------

Heterogeneous cell model built from Node-Set Attention (NSA) blocks.

Classes
~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   hetero_cell_nsa.AttentionalGraphAggregation
   hetero_cell_nsa.PreProcessor
   hetero_cell_nsa.HeteroCellNSA

Functions
~~~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated

   hetero_cell_nsa.get_norm_layer

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

``hetero_cell_pma``
-------------------

Heterogeneous cell graph model with PMA pooling for fitness/interaction.

Classes
~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   hetero_cell_pma.SimplePMA
   hetero_cell_pma.PreProcessor
   hetero_cell_pma.AttentionConvWrapper
   hetero_cell_pma.HeteroCell

Functions
~~~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated

   hetero_cell_pma.get_norm_layer
   hetero_cell_pma.load_sample_data_batch
   hetero_cell_pma.plot_correlations
   hetero_cell_pma.plot_embeddings

``hetero_gnn_pool``
-------------------

Heterogeneous GNN with graph pooling and a configurable prediction head.

Classes
~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   hetero_gnn_pool.ProjectedGATConv
   hetero_gnn_pool.PredictionHead
   hetero_gnn_pool.HeteroGnnPool

Functions
~~~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated

   hetero_gnn_pool.load_sample_data_batch

``isomorphic_cell``
-------------------

Isomorphic cell model combining gene GNN and metabolism hypergraph branches.

Classes
~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   isomorphic_cell.PreProcessor
   isomorphic_cell.Combiner
   isomorphic_cell.ProjectedGATConv
   isomorphic_cell.PredictionHead
   isomorphic_cell.HeteroGnn
   isomorphic_cell.GeneContextProcessor
   isomorphic_cell.MetaboliteProcessor
   isomorphic_cell.ReactionMapper
   isomorphic_cell.GeneMapper
   isomorphic_cell.MetabolismProcessor
   isomorphic_cell.IsomorphicCell

Functions
~~~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated

   isomorphic_cell.initialize_model
   isomorphic_cell.load_sample_data_batch
   isomorphic_cell.plot_correlations

``isomorphic_cell_attentional``
-------------------------------

Attentional isomorphic cell model over gene, reaction, and metabolite graphs.

Classes
~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   isomorphic_cell_attentional.AttentionalGraphAggregation
   isomorphic_cell_attentional.PreProcessor
   isomorphic_cell_attentional.Combiner
   isomorphic_cell_attentional.ProjectedGATConv
   isomorphic_cell_attentional.PredictionHead
   isomorphic_cell_attentional.HeteroGnn
   isomorphic_cell_attentional.GeneContextProcessor
   isomorphic_cell_attentional.ReactionMapper
   isomorphic_cell_attentional.GeneMapper
   isomorphic_cell_attentional.MetabolismProcessor
   isomorphic_cell_attentional.IsomorphicCell

Functions
~~~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated

   isomorphic_cell_attentional.get_norm_layer
   isomorphic_cell_attentional.load_sample_data_batch
   isomorphic_cell_attentional.plot_correlations

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

``nsa_hetero_cell``
-------------------

Heterogeneous Node-Set Attention model over gene/metabolite/reaction graphs.

Classes
~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   nsa_hetero_cell.BlockContainer
   nsa_hetero_cell.AttentionBlock
   nsa_hetero_cell.MAB
   nsa_hetero_cell.SAB
   nsa_hetero_cell.StoichiometricMAB
   nsa_hetero_cell.CellGraphHeteroNSA
   nsa_hetero_cell.CellGraphNSAModel

Functions
~~~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated

   nsa_hetero_cell.load_sample_data_batch
   nsa_hetero_cell.group

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

``self_attention_sag``
----------------------

Self-attention pooling model combining multi-head attention with SAGPooling GCNs.

Classes
~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   self_attention_sag.SelfAttention
   self_attention_sag.SelfAttentionSAG

Not documented
--------------

Submodules left out of this page:

- ``dcell_DEPRECATED`` (scratch, demo or deprecated)
- ``species_aware_lm`` (import fails: ModuleNotFoundError)
