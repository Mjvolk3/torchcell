# torchcell/data/graph_processor
# [[torchcell.data.graph_processor]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/data/graph_processor
# Test file: tests/torchcell/data/test_graph_processor.py


from abc import ABC, abstractmethod
from typing import Any, Dict

import torch
from torch_geometric.utils._subgraph import bipartite_subgraph, subgraph
from torch_scatter import scatter

from torchcell.data.hetero_data import HeteroData
from torchcell.datamodels import ExperimentReferenceType, ExperimentType, PhenotypeType


class GraphProcessor(ABC):
    @abstractmethod
    def process(
        self,
        cell_graph: HeteroData,
        phenotype_info: list[PhenotypeType],
        data: (
            dict[str, ExperimentType | ExperimentReferenceType]
            | list[dict[str, ExperimentType | ExperimentReferenceType]]
        ),
    ) -> HeteroData:
        pass


class SubgraphRepresentation(GraphProcessor):
    def __init__(self) -> None:
        super().__init__()
        self.device: torch.device = None
        self.masks: Dict[str, Dict[str, torch.Tensor]] = {}
    
    def _initialize_masks(self, cell_graph: HeteroData) -> None:
        self.masks = {
            "gene": {
                "kept": torch.zeros(
                    cell_graph["gene"].num_nodes, dtype=torch.bool, device=self.device
                ),
                "perturbed": torch.zeros(
                    cell_graph["gene"].num_nodes, dtype=torch.bool, device=self.device
                ),
            },
            "reaction": {
                "kept": torch.zeros(
                    cell_graph["reaction"].num_nodes,
                    dtype=torch.bool,
                    device=self.device,
                ),
                "removed": torch.zeros(
                    cell_graph["reaction"].num_nodes,
                    dtype=torch.bool,
                    device=self.device,
                ),
            },
            "metabolite": {
                "kept": torch.ones(
                    cell_graph["metabolite"].num_nodes,
                    dtype=torch.bool,
                    device=self.device,
                ),
                "removed": torch.zeros(
                    cell_graph["metabolite"].num_nodes,
                    dtype=torch.bool,
                    device=self.device,
                ),
            },
        }

    def process(
        self,
        cell_graph: HeteroData,
        phenotype_info: list[Any],
        data: list[Dict[str, Any]],
    ) -> HeteroData:
        # Set the device based on gene features
        self.device = cell_graph["gene"].x.device
        self._initialize_masks(cell_graph)
        integrated_subgraph = HeteroData()

        # Process gene-level information
        gene_info = self._process_gene_info(cell_graph, data)
        self._add_gene_data(integrated_subgraph, cell_graph, gene_info)
        self._process_gene_interactions(integrated_subgraph, cell_graph, gene_info)

        # Process reaction-level (GPR) information
        reaction_info = self._process_reaction_info(
            cell_graph, gene_info, integrated_subgraph
        )
        self._add_reaction_data(integrated_subgraph, reaction_info, cell_graph)

        # Process metabolic network via the bipartite representation only
        self._process_metabolic_network(integrated_subgraph, cell_graph, reaction_info)

        # Add phenotype data to the gene nodes
        self._add_phenotype_data(integrated_subgraph, phenotype_info, data)

        # Attach the masks to the output graph
        integrated_subgraph["gene"].pert_mask = self.masks["gene"]["perturbed"]
        integrated_subgraph["reaction"].pert_mask = self.masks["reaction"]["removed"]
        integrated_subgraph["metabolite"].pert_mask = self.masks["metabolite"][
            "removed"
        ]

        return integrated_subgraph

    def _process_gene_info(
        self, cell_graph: HeteroData, data: list[Dict[str, Any]]
    ) -> Dict[str, Any]:
        perturbed_names = {
            p.systematic_gene_name
            for item in data
            for p in item["experiment"].genotype.perturbations
        }
        node_ids = cell_graph["gene"].node_ids
        keep_idx = [i for i, name in enumerate(node_ids) if name not in perturbed_names]
        remove_idx = [i for i, name in enumerate(node_ids) if name in perturbed_names]

        self.masks["gene"]["kept"][keep_idx] = True
        self.masks["gene"]["perturbed"][remove_idx] = True

        return {
            "perturbed_names": perturbed_names,
            "keep_subset": torch.tensor(keep_idx, dtype=torch.long, device=self.device),
            "remove_subset": torch.tensor(
                remove_idx, dtype=torch.long, device=self.device
            ),
            "keep_node_ids": [node_ids[i] for i in keep_idx],
        }

    def _add_gene_data(
        self,
        integrated_subgraph: HeteroData,
        cell_graph: HeteroData,
        gene_info: Dict[str, Any],
    ) -> None:
        integrated_subgraph["gene"].node_ids = gene_info["keep_node_ids"]
        integrated_subgraph["gene"].num_nodes = len(gene_info["keep_node_ids"])
        integrated_subgraph["gene"].ids_pert = list(gene_info["perturbed_names"])
        integrated_subgraph["gene"].cell_graph_idx_pert = gene_info["remove_subset"]

        x_full = cell_graph["gene"].x
        integrated_subgraph["gene"].x = x_full[gene_info["keep_subset"]]
        integrated_subgraph["gene"].x_pert = x_full[gene_info["remove_subset"]]

    def _process_gene_interactions(
        self,
        integrated_subgraph: HeteroData,
        cell_graph: HeteroData,
        gene_info: Dict[str, Any],
    ) -> None:
        edge_types = [
            ("gene", "physical_interaction", "gene"),
            ("gene", "regulatory_interaction", "gene"),
        ]
        for et in cell_graph.edge_types:
            if et in edge_types:
                orig_edge_index = cell_graph[et].edge_index
                edge_index, _, edge_mask = subgraph(
                    subset=gene_info["keep_subset"],
                    edge_index=orig_edge_index,
                    relabel_nodes=True,
                    num_nodes=cell_graph["gene"].num_nodes,
                    return_edge_mask=True,
                )
                integrated_subgraph[et].edge_index = edge_index
                integrated_subgraph[et].num_edges = edge_index.size(1)
                integrated_subgraph[et].pert_mask = ~edge_mask

    def _process_reaction_info(
        self,
        cell_graph: HeteroData,
        gene_info: Dict[str, Any],
        integrated_subgraph: HeteroData,
    ) -> Dict[str, Any]:
        max_reaction_idx = cell_graph["reaction"].num_nodes

        # Create Growth subsystem indicator first (before any conditional returns)
        w_growth = torch.zeros(max_reaction_idx, dtype=torch.float, device=self.device)

        # Check if subsystem attribute exists
        if hasattr(cell_graph["reaction"], "subsystem"):
            # Check data type and structure
            subsystems = cell_graph["reaction"].subsystem

            # Populate the w_growth tensor
            for i in range(max_reaction_idx):
                if isinstance(subsystems, list):
                    if i < len(subsystems) and subsystems[i] == "Growth":
                        w_growth[i] = 1.0
                elif torch.is_tensor(subsystems):
                    if subsystems[i] == "Growth":
                        w_growth[i] = 1.0
                elif hasattr(subsystems, "__getitem__"):
                    if subsystems[i] == "Growth":
                        w_growth[i] = 1.0

        # If no GPR relationship exists, assume all reactions are valid.
        if ("gene", "gpr", "reaction") not in cell_graph.edge_types:
            valid_reactions = torch.arange(max_reaction_idx, device=self.device)
            self.masks["reaction"]["kept"].fill_(True)
            self.masks["reaction"]["removed"].fill_(False)

            gene_map = torch.full(
                (cell_graph["gene"].num_nodes,),
                -1,
                dtype=torch.long,
                device=self.device,
            )
            gene_map[gene_info["keep_subset"]] = torch.arange(
                len(gene_info["keep_subset"]), device=self.device
            )
            reaction_map = torch.arange(max_reaction_idx, device=self.device)

            # Include w_growth in the return dictionary
            return {
                "valid_reactions": valid_reactions,
                "gene_map": gene_map.tolist(),
                "reaction_map": reaction_map.tolist(),
                "w_growth": w_growth,
            }

        # Process gene–reaction (GPR) edges
        gpr_edge_index = cell_graph["gene", "gpr", "reaction"].hyperedge_index.to(
            self.device
        )
        gene_mask = torch.zeros(
            cell_graph["gene"].num_nodes, dtype=torch.bool, device=self.device
        )
        gene_mask[gene_info["keep_subset"]] = True

        gene_indices = gpr_edge_index[0]
        reaction_indices = gpr_edge_index[1]
        max_gene_idx = cell_graph["gene"].num_nodes

        valid_mask = (gene_indices < max_gene_idx) & (
            reaction_indices < max_reaction_idx
        )
        gene_indices = gene_indices[valid_mask]
        reaction_indices = reaction_indices[valid_mask]

        has_genes = torch.zeros(max_reaction_idx, dtype=torch.bool, device=self.device)
        has_genes[reaction_indices.unique()] = True

        reaction_gene_sum = scatter(
            gene_mask[gene_indices].float(),
            reaction_indices,
            dim=0,
            dim_size=max_reaction_idx,
            reduce="sum",
        )
        total_gene_count = scatter(
            torch.ones_like(gene_indices, dtype=torch.float),
            reaction_indices,
            dim=0,
            dim_size=max_reaction_idx,
            reduce="sum",
        )
        valid_with_genes_mask = (reaction_gene_sum == total_gene_count) & has_genes
        valid_without_genes_mask = ~has_genes
        valid_mask_combined = valid_with_genes_mask | valid_without_genes_mask
        valid_reactions = torch.nonzero(valid_mask_combined).squeeze(-1)

        self.masks["reaction"]["kept"] = valid_mask_combined
        self.masks["reaction"]["removed"] = ~valid_mask_combined

        edge_mask = torch.isin(
            reaction_indices, torch.where(valid_with_genes_mask)[0]
        ) & torch.isin(gene_indices, gene_info["keep_subset"])
        new_gpr_edge_index = gpr_edge_index[:, edge_mask].clone()

        gene_map = torch.full(
            (cell_graph["gene"].num_nodes,), -1, dtype=torch.long, device=self.device
        )
        gene_map[gene_info["keep_subset"]] = torch.arange(
            len(gene_info["keep_subset"]), device=self.device
        )
        reaction_map = torch.full(
            (max_reaction_idx,), -1, dtype=torch.long, device=self.device
        )
        reaction_map[valid_reactions] = torch.arange(
            len(valid_reactions), device=self.device
        )

        new_gpr_edge_index[0] = gene_map[new_gpr_edge_index[0]]
        new_gpr_edge_index[1] = reaction_map[new_gpr_edge_index[1]]

        integrated_subgraph["gene", "gpr", "reaction"].hyperedge_index = (
            new_gpr_edge_index
        )
        integrated_subgraph["gene", "gpr", "reaction"].num_edges = (
            new_gpr_edge_index.size(1)
        )
        integrated_subgraph["gene", "gpr", "reaction"].pert_mask = ~edge_mask

        # Include w_growth in the return
        return {
            "valid_reactions": valid_reactions,
            "gene_map": gene_map.tolist(),
            "reaction_map": reaction_map.tolist(),
            "w_growth": w_growth,
        }

    def _add_reaction_data(
        self,
        integrated_subgraph: HeteroData,
        reaction_info: Dict[str, Any],
        cell_graph: HeteroData,
    ) -> None:
        if not reaction_info:
            return
        valid_reactions = reaction_info["valid_reactions"]
        integrated_subgraph["reaction"].num_nodes = len(valid_reactions)
        integrated_subgraph["reaction"].node_ids = valid_reactions.tolist()

        # Subset w_growth to valid reactions if it exists in cell_graph
        if hasattr(cell_graph["reaction"], "w_growth"):
            w_growth = cell_graph["reaction"].w_growth
            integrated_subgraph["reaction"].w_growth = w_growth[valid_reactions]

    def _process_metabolic_network(
        self,
        integrated_subgraph: HeteroData,
        cell_graph: HeteroData,
        reaction_info: Dict[str, Any],
    ) -> None:
        # Only process if bipartite representation exists
        if not reaction_info or "reaction" not in cell_graph.node_types:
            return
        if ("reaction", "rmr", "metabolite") in cell_graph.edge_types:
            self._process_metabolism_bipartite(
                integrated_subgraph, cell_graph, reaction_info
            )

    def _process_metabolism_bipartite(
        self,
        integrated_subgraph: HeteroData,
        cell_graph: HeteroData,
        reaction_info: Dict[str, Any],
    ) -> None:
        valid_reactions = reaction_info["valid_reactions"]
        rmr_edges = cell_graph["reaction", "rmr", "metabolite"]
        hyperedge_index = rmr_edges.hyperedge_index.to(self.device)
        stoichiometry = rmr_edges.stoichiometry.to(self.device)
        metabolite_subset = torch.arange(
            cell_graph["metabolite"].num_nodes, device=self.device
        )

        # The stoichiometry values already have sign information
        # No need to check edge_type
        final_edge_index, final_edge_attr = bipartite_subgraph(
            (valid_reactions, metabolite_subset),
            hyperedge_index,
            edge_attr=stoichiometry,
            relabel_nodes=True,
            size=(cell_graph["reaction"].num_nodes, cell_graph["metabolite"].num_nodes),
        )
        edge_type = ("reaction", "rmr", "metabolite")
        integrated_subgraph[edge_type].hyperedge_index = final_edge_index
        integrated_subgraph[edge_type].stoichiometry = final_edge_attr
        integrated_subgraph[edge_type].num_edges = final_edge_index.size(1)

        integrated_subgraph["metabolite"].node_ids = cell_graph["metabolite"].node_ids
        integrated_subgraph["metabolite"].num_nodes = cell_graph["metabolite"].num_nodes
        self.masks["metabolite"]["kept"].fill_(True)
        self.masks["metabolite"]["removed"].fill_(False)

    def _add_phenotype_data(
        self,
        integrated_subgraph: HeteroData,
        phenotype_info: list[Any],
        data: list[Dict[str, Any]],
    ) -> None:
        phenotype_fields = []
        for phenotype in phenotype_info:
            phenotype_fields.extend(
                [
                    phenotype.model_fields["label_name"].default,
                    phenotype.model_fields["label_statistic_name"].default,
                ]
            )
        for field in phenotype_fields:
            field_values = []
            for item in data:
                value = getattr(item["experiment"].phenotype, field, None)
                if value is not None:
                    field_values.append(value)
            integrated_subgraph["gene"][field] = torch.tensor(
                field_values if field_values else [float("nan")],
                dtype=torch.float,
                device=self.device,
            )


class Unperturbed(GraphProcessor):
    """
    Processes graph data by preserving the original graph structure and storing perturbation
    and phenotype data alongside it for later processing.

    This processor:
    1. Keeps the original graph structure intact
    2. Stores perturbation information separately
    3. Records phenotype data without modifying the graph
    4. Can be used as a base for applying perturbations later in the pipeline

    Attributes remain unchanged:
        - Node features: X ∈ ℝ^(N×d) stays as X ∈ ℝ^(N×d)
        - Edge structure: E ∈ ℤ^(2×|E|) stays as E ∈ ℤ^(2×|E|)
    """

    def process(
        self,
        cell_graph: HeteroData,
        phenotype_info: list[PhenotypeType],
        data: list[dict[str, ExperimentType | ExperimentReferenceType]],
    ) -> HeteroData:
        if not data:
            raise ValueError("Data list is empty")

        processed_graph = HeteroData()

        # Copy graph structure and features
        processed_graph["gene"].x = cell_graph["gene"].x
        processed_graph["gene"].node_ids = cell_graph["gene"].node_ids
        processed_graph["gene"].num_nodes = cell_graph["gene"].num_nodes

        # Store perturbation information
        perturbed_genes = set()
        for item in data:
            if "experiment" not in item or "experiment_reference" not in item:
                raise ValueError(
                    "Each item in data must contain both 'experiment' and "
                    "'experiment_reference' keys"
                )
            perturbed_genes.update(
                pert.systematic_gene_name
                for pert in item["experiment"].genotype.perturbations
            )

        processed_graph["gene"].perturbed_genes = list(perturbed_genes)
        processed_graph["gene"].perturbation_indices = torch.tensor(
            [cell_graph["gene"].node_ids.index(nid) for nid in perturbed_genes],
            dtype=torch.long,
        )

        # Add phenotype fields
        phenotype_fields = []
        for phenotype in phenotype_info:
            phenotype_fields.append(phenotype.model_fields["label_name"].default)
            phenotype_fields.append(
                phenotype.model_fields["label_statistic_name"].default
            )

        # Add experiment data
        for field in phenotype_fields:
            field_values = []
            for item in data:
                value = getattr(item["experiment"].phenotype, field, None)
                if value is not None:
                    field_values.append(value)
            if field_values:
                processed_graph["gene"][field] = torch.tensor(field_values)
            else:
                processed_graph["gene"][field] = torch.tensor([float("nan")])

        # Copy edge information
        for edge_type in cell_graph.edge_types:
            if edge_type[1] in ["physical_interaction", "regulatory_interaction"]:
                processed_graph[edge_type].edge_index = cell_graph[edge_type].edge_index
                processed_graph[edge_type].num_edges = cell_graph[edge_type].num_edges

        # Handle metabolite data
        if "metabolite" in cell_graph.node_types:
            processed_graph["metabolite"].num_nodes = cell_graph["metabolite"].num_nodes
            processed_graph["metabolite"].node_ids = cell_graph["metabolite"].node_ids

            edge_type = ("metabolite", "reactions", "metabolite")
            if any(e_type == edge_type for e_type in cell_graph.edge_types):
                # Copy hypergraph structure
                processed_graph[edge_type].hyperedge_index = cell_graph[
                    edge_type
                ].hyperedge_index
                processed_graph[edge_type].stoichiometry = cell_graph[
                    edge_type
                ].stoichiometry
                processed_graph[edge_type].num_edges = cell_graph[edge_type].num_edges

                # Only create reaction_to_genes_indices mapping
                node_id_to_idx = {
                    nid: idx for idx, nid in enumerate(cell_graph["gene"].node_ids)
                }
                reaction_to_genes_indices = {}

                for reaction_idx, genes in cell_graph[
                    edge_type
                ].reaction_to_genes.items():
                    gene_indices = []
                    for gene in genes:
                        gene_idx = node_id_to_idx.get(gene, -1)
                        gene_indices.append(gene_idx)
                    reaction_to_genes_indices[reaction_idx] = gene_indices

                processed_graph[edge_type].reaction_to_genes_indices = (
                    reaction_to_genes_indices
                )

        return processed_graph


class Perturbation(GraphProcessor):
    """
    Processes graph data by storing only perturbation-specific information without duplicating
    the base graph structure. This allows sharing a single base graph across instances while
    only tracking what changes between instances (perturbations and associated measurements).

    This processor:
    1. Stores perturbation information and measurements
    2. Does not duplicate the base graph structure
    3. Intended to be used with a shared base graph stored at the dataset level

    Key differences from Identity processor:
    - Does not store complete graph structure
    - Only tracks instance-specific perturbation data
    - Reduces memory usage by avoiding graph duplication
    """

    def process(
        self,
        cell_graph: HeteroData,
        phenotype_info: list[PhenotypeType],
        data: list[dict[str, ExperimentType | ExperimentReferenceType]],
    ) -> HeteroData:
        if not data:
            raise ValueError("Data list is empty")

        # Create a minimal HeteroData object to store perturbation data
        processed_data = HeteroData()

        # Store perturbation information
        perturbed_genes = set()
        for item in data:
            if "experiment" not in item or "experiment_reference" not in item:
                raise ValueError(
                    "Each item in data must contain both 'experiment' and "
                    "'experiment_reference' keys"
                )
            perturbed_genes.update(
                pert.systematic_gene_name
                for pert in item["experiment"].genotype.perturbations
            )

        # Store perturbation indices
        processed_data["gene"].perturbed_genes = list(perturbed_genes)
        processed_data["gene"].perturbation_indices = torch.tensor(
            [cell_graph["gene"].node_ids.index(nid) for nid in perturbed_genes],
            dtype=torch.long,
        )

        # Add phenotype fields
        phenotype_fields = []
        for phenotype in phenotype_info:
            phenotype_fields.append(phenotype.model_fields["label_name"].default)
            phenotype_fields.append(
                phenotype.model_fields["label_statistic_name"].default
            )

        # Add experiment data
        for field in phenotype_fields:
            field_values = []
            for item in data:
                value = getattr(item["experiment"].phenotype, field, None)
                if value is not None:
                    field_values.append(value)
            if field_values:
                processed_data["gene"][field] = torch.tensor(field_values)
            else:
                processed_data["gene"][field] = torch.tensor([float("nan")])

        return processed_data
