# torchcell/data/graph_processor
# [[torchcell.data.graph_processor]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/data/graph_processor
# Test file: tests/torchcell/data/test_graph_processor.py


from abc import ABC, abstractmethod
from typing import Any, Dict
from torchcell.datamodels import PhenotypeType, ExperimentReferenceType, ExperimentType
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
        # Initialize masks dictionary
        self.masks = {
            "gene": {
                "kept": torch.zeros(
                    cell_graph["gene"].num_nodes, dtype=torch.bool, device=self.device
                ),
                "perturbed": torch.zeros(
                    cell_graph["gene"].num_nodes, dtype=torch.bool, device=self.device
                ),
            }
        }

        # Only create masks for reaction and metabolite if they exist in the cell graph
        if "reaction" in cell_graph.node_types and hasattr(
            cell_graph["reaction"], "num_nodes"
        ):
            self.masks["reaction"] = {
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
            }

        if "metabolite" in cell_graph.node_types and hasattr(
            cell_graph["metabolite"], "num_nodes"
        ):
            self.masks["metabolite"] = {
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

        # Process reaction-level (GPR) information only if reaction nodes exist
        if "reaction" in cell_graph.node_types and hasattr(
            cell_graph["reaction"], "num_nodes"
        ):
            reaction_info = self._process_reaction_info(
                cell_graph, gene_info, integrated_subgraph
            )
            self._add_reaction_data(integrated_subgraph, reaction_info, cell_graph)

            # Process metabolic network only if metabolite nodes also exist
            if "metabolite" in cell_graph.node_types and hasattr(
                cell_graph["metabolite"], "num_nodes"
            ):
                self._process_metabolic_network(
                    integrated_subgraph, cell_graph, reaction_info
                )
        else:
            reaction_info = None

        # Add phenotype data to the gene nodes
        self._add_phenotype_data(integrated_subgraph, phenotype_info, data)

        # Attach the masks to the output graph
        integrated_subgraph["gene"].pert_mask = self.masks["gene"]["perturbed"]

        # Only add reaction and metabolite masks if they exist
        if "reaction" in self.masks:
            integrated_subgraph["reaction"].pert_mask = self.masks["reaction"][
                "removed"
            ]

        if "metabolite" in self.masks:
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
        integrated_subgraph["gene"].perturbation_indices = gene_info["remove_subset"]

        x_full = cell_graph["gene"].x
        integrated_subgraph["gene"].x = x_full[gene_info["keep_subset"]]
        integrated_subgraph["gene"].x_pert = x_full[gene_info["remove_subset"]]

    def _process_gene_interactions(
        self,
        integrated_subgraph: HeteroData,
        cell_graph: HeteroData,
        gene_info: Dict[str, Any],
    ) -> None:
        # Process all gene-to-gene edge types
        for et in cell_graph.edge_types:
            if et[0] == "gene" and et[2] == "gene":
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

    # TODO Add option for including non-coo version.
    # def _add_phenotype_data(
    #     self,
    #     integrated_subgraph: HeteroData,
    #     phenotype_info: list[PhenotypeType],
    #     data: list[Dict[str, ExperimentType | ExperimentReferenceType]],
    # ) -> None:
    #     phenotype_fields = []
    #     for phenotype in phenotype_info:
    #         phenotype_fields.extend(
    #             [
    #                 phenotype.model_fields["label_name"].default,
    #                 phenotype.model_fields["label_statistic_name"].default,
    #             ]
    #         )
    #     for field in phenotype_fields:
    #         field_values = []
    #         for item in data:
    #             value = getattr(item["experiment"].phenotype, field, None)
    #             if value is not None:
    #                 field_values.append(value)
    #         integrated_subgraph["gene"][field] = torch.tensor(
    #             field_values if field_values else [float("nan")],
    #             dtype=torch.float,
    #             device=self.device,
    #         )

    def _add_phenotype_data(
        self,
        integrated_subgraph: HeteroData,
        phenotype_info: list[PhenotypeType],
        data: list[Dict[str, ExperimentType | ExperimentReferenceType]],
    ) -> None:
        """
        Add phenotype data to the graph in COO format.
        Optimized version that ensures all tensors are on the same device.
        """
        # Ensure device is set
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize storage for phenotype values and metadata
        all_values = []
        all_type_indices = []
        all_sample_indices = []  # Which sample each value belongs to
        phenotype_types = []

        # Initialize storage for statistic values
        all_stat_values = []
        all_stat_type_indices = []
        all_stat_sample_indices = []  # Which sample each stat value belongs to
        stat_types = []

        # Extract phenotype type information
        for phenotype_class in phenotype_info:
            label_name = phenotype_class.model_fields["label_name"].default
            stat_name = phenotype_class.model_fields["label_statistic_name"].default

            phenotype_types.append(label_name)
            if stat_name:
                stat_types.append(stat_name)

        # Process each experimental data point
        for item_idx, item in enumerate(data):
            # Get phenotype object from experiment
            phenotype = item["experiment"].phenotype

            # Process each phenotype type
            for type_idx, field_name in enumerate(phenotype_types):
                value = getattr(phenotype, field_name, None)
                if value is not None:
                    # Convert single values to lists for consistent handling
                    values = [value] if not isinstance(value, (list, tuple)) else value

                    # Add all values with their type indices and sample indices
                    all_values.extend(values)
                    all_type_indices.extend([type_idx] * len(values))
                    all_sample_indices.extend([item_idx] * len(values))

            # Process statistics in the same way
            for stat_type_idx, stat_field_name in enumerate(stat_types):
                stat_value = getattr(phenotype, stat_field_name, None)
                if stat_value is not None:
                    # Convert single values to lists for consistent handling
                    stat_values = (
                        [stat_value]
                        if not isinstance(stat_value, (list, tuple))
                        else stat_value
                    )

                    # Add all statistic values with their type indices and sample indices
                    all_stat_values.extend(stat_values)
                    all_stat_type_indices.extend([stat_type_idx] * len(stat_values))
                    all_stat_sample_indices.extend([item_idx] * len(stat_values))

        # Store phenotype data in the graph - create all tensors directly on the target device
        if all_values:
            integrated_subgraph["gene"]["phenotype_values"] = torch.tensor(
                all_values, dtype=torch.float, device=self.device
            )
            integrated_subgraph["gene"]["phenotype_type_indices"] = torch.tensor(
                all_type_indices, dtype=torch.long, device=self.device
            )
            integrated_subgraph["gene"]["phenotype_sample_indices"] = torch.tensor(
                all_sample_indices, dtype=torch.long, device=self.device
            )
        else:
            # Handle empty case with placeholder values - create directly on target device
            integrated_subgraph["gene"]["phenotype_values"] = torch.tensor(
                [float("nan")], dtype=torch.float, device=self.device
            )
            integrated_subgraph["gene"]["phenotype_type_indices"] = torch.tensor(
                [0], dtype=torch.long, device=self.device
            )
            integrated_subgraph["gene"]["phenotype_sample_indices"] = torch.tensor(
                [0], dtype=torch.long, device=self.device
            )

        # Store phenotype type names (non-tensor data)
        integrated_subgraph["gene"]["phenotype_types"] = phenotype_types

        # Store statistic data in the graph - create directly on target device
        if all_stat_values:
            integrated_subgraph["gene"]["phenotype_stat_values"] = torch.tensor(
                all_stat_values, dtype=torch.float, device=self.device
            )
            integrated_subgraph["gene"]["phenotype_stat_type_indices"] = torch.tensor(
                all_stat_type_indices, dtype=torch.long, device=self.device
            )
            integrated_subgraph["gene"]["phenotype_stat_sample_indices"] = torch.tensor(
                all_stat_sample_indices, dtype=torch.long, device=self.device
            )
            integrated_subgraph["gene"]["phenotype_stat_types"] = stat_types


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
    the base graph structure, using COO format for phenotypes.
    """

    def __init__(self) -> None:
        super().__init__()
        self.device = None

    def process(
        self,
        cell_graph: HeteroData,
        phenotype_info: list[PhenotypeType],
        data: list[Dict[str, ExperimentType | ExperimentReferenceType]],
    ) -> HeteroData:
        if not data:
            raise ValueError("Data list is empty")

        # Set device based on cell_graph
        if hasattr(cell_graph["gene"], "x") and hasattr(cell_graph["gene"].x, "device"):
            self.device = cell_graph["gene"].x.device
        else:
            self.device = torch.device("cpu")

        # Create a minimal HeteroData object to store perturbation data
        processed_graph = HeteroData()

        # Add required node count to avoid PyG warnings
        processed_graph["gene"].num_nodes = cell_graph["gene"].num_nodes

        # Process perturbed genes
        perturbed_names = {
            p.systematic_gene_name
            for item in data
            for p in item["experiment"].genotype.perturbations
        }
        node_ids = cell_graph["gene"].node_ids
        perturbed_indices = [
            i for i, name in enumerate(node_ids) if name in perturbed_names
        ]

        # Create perturbation mask: True for perturbed genes
        pert_mask = torch.zeros(
            cell_graph["gene"].num_nodes, dtype=torch.bool, device=self.device
        )
        pert_mask[perturbed_indices] = True

        # Create regular mask: EXPLICITLY as logical NOT of pert_mask
        mask = ~pert_mask

        # Store perturbation information
        processed_graph["gene"].perturbed_genes = list(perturbed_names)
        processed_graph["gene"].perturbation_indices = torch.tensor(
            perturbed_indices, dtype=torch.long, device=self.device
        )

        # Store masks
        processed_graph["gene"].pert_mask = pert_mask
        processed_graph["gene"].mask = mask

        # Verify mask sums are correct
        assert pert_mask.sum() == len(
            perturbed_indices
        ), "pert_mask sum doesn't match perturbed indices"
        assert mask.sum() == cell_graph["gene"].num_nodes - len(
            perturbed_indices
        ), "mask sum isn't complement of pert_mask"

        # Add phenotype data in COO format
        self._add_phenotype_data(processed_graph, phenotype_info, data)

        return processed_graph

    def _add_phenotype_data(
        self,
        integrated_subgraph: HeteroData,
        phenotype_info: list[PhenotypeType],
        data: list[Dict[str, ExperimentType | ExperimentReferenceType]],
    ) -> None:
        # Rest of the phenotype data processing remains the same...
        # Initializing storage, processing phenotypes, etc.

        # Initialize storage for phenotype values and metadata
        all_values = []
        all_type_indices = []
        all_sample_indices = []
        phenotype_types = []

        # Initialize storage for statistic values
        all_stat_values = []
        all_stat_type_indices = []
        all_stat_sample_indices = []
        stat_types = []

        # Extract phenotype type information
        for phenotype_class in phenotype_info:
            label_name = phenotype_class.model_fields["label_name"].default
            stat_name = phenotype_class.model_fields["label_statistic_name"].default

            phenotype_types.append(label_name)
            if stat_name:
                stat_types.append(stat_name)

        # Process each experimental data point
        for item_idx, item in enumerate(data):
            # Get phenotype object from experiment
            phenotype = item["experiment"].phenotype

            # Process each phenotype type
            for type_idx, field_name in enumerate(phenotype_types):
                value = getattr(phenotype, field_name, None)
                if value is not None:
                    # Convert single values to lists for consistent handling
                    values = [value] if not isinstance(value, (list, tuple)) else value

                    # Add all values with their type indices and sample indices
                    all_values.extend(values)
                    all_type_indices.extend([type_idx] * len(values))
                    all_sample_indices.extend([item_idx] * len(values))

            # Process statistics in the same way
            for stat_type_idx, stat_field_name in enumerate(stat_types):
                stat_value = getattr(phenotype, stat_field_name, None)
                if stat_value is not None:
                    # Convert single values to lists for consistent handling
                    stat_values = (
                        [stat_value]
                        if not isinstance(stat_value, (list, tuple))
                        else stat_value
                    )

                    # Add all statistic values with their type indices and sample indices
                    all_stat_values.extend(stat_values)
                    all_stat_type_indices.extend([stat_type_idx] * len(stat_values))
                    all_stat_sample_indices.extend([item_idx] * len(stat_values))

        # Store phenotype data in the graph
        if all_values:
            integrated_subgraph["gene"]["phenotype_values"] = torch.tensor(
                all_values, dtype=torch.float, device=self.device
            )
            integrated_subgraph["gene"]["phenotype_type_indices"] = torch.tensor(
                all_type_indices, dtype=torch.long, device=self.device
            )
            integrated_subgraph["gene"]["phenotype_sample_indices"] = torch.tensor(
                all_sample_indices, dtype=torch.long, device=self.device
            )
            integrated_subgraph["gene"]["phenotype_types"] = phenotype_types
        else:
            # Handle empty case with placeholder values
            integrated_subgraph["gene"]["phenotype_values"] = torch.tensor(
                [float("nan")], dtype=torch.float, device=self.device
            )
            integrated_subgraph["gene"]["phenotype_type_indices"] = torch.tensor(
                [0], dtype=torch.long, device=self.device
            )
            integrated_subgraph["gene"]["phenotype_sample_indices"] = torch.tensor(
                [0], dtype=torch.long, device=self.device
            )
            integrated_subgraph["gene"]["phenotype_types"] = phenotype_types

        # Store statistic data in the graph
        if all_stat_values:
            integrated_subgraph["gene"]["phenotype_stat_values"] = torch.tensor(
                all_stat_values, dtype=torch.float, device=self.device
            )
            integrated_subgraph["gene"]["phenotype_stat_type_indices"] = torch.tensor(
                all_stat_type_indices, dtype=torch.long, device=self.device
            )
            integrated_subgraph["gene"]["phenotype_stat_sample_indices"] = torch.tensor(
                all_stat_sample_indices, dtype=torch.long, device=self.device
            )
            integrated_subgraph["gene"]["phenotype_stat_types"] = stat_types


class DCellGraphProcessor(GraphProcessor):
    """
    Graph processor for DCell model that applies perturbations directly to the gene ontology graph.
    This processor updates the gene ontology node states based on perturbations rather than
    deleting nodes from the graph before conversion.
    """

    def __init__(self) -> None:
        super().__init__()
        self.device = None

    def process(
        self,
        cell_graph: HeteroData,
        phenotype_info: list[PhenotypeType],
        data: list[Dict[str, ExperimentType | ExperimentReferenceType]],
    ) -> HeteroData:
        if not data:
            raise ValueError("Data list is empty")

        # Set device based on cell_graph
        if hasattr(cell_graph["gene"], "x") and hasattr(cell_graph["gene"].x, "device"):
            self.device = cell_graph["gene"].x.device
        else:
            self.device = torch.device("cpu")

        # Create a new HeteroData object for processing
        processed_graph = HeteroData()

        # Add gene node information
        processed_graph["gene"].num_nodes = cell_graph["gene"].num_nodes
        processed_graph["gene"].node_ids = cell_graph["gene"].node_ids

        if hasattr(cell_graph["gene"], "x"):
            processed_graph["gene"].x = cell_graph["gene"].x

        # Process perturbed genes
        perturbed_names = {
            p.systematic_gene_name
            for item in data
            for p in item["experiment"].genotype.perturbations
        }
        node_ids = cell_graph["gene"].node_ids
        perturbed_indices = [
            i for i, name in enumerate(node_ids) if name in perturbed_names
        ]

        # Create perturbation mask: True for perturbed genes
        pert_mask = torch.zeros(
            cell_graph["gene"].num_nodes, dtype=torch.bool, device=self.device
        )
        pert_mask[perturbed_indices] = True

        # Store perturbation information
        processed_graph["gene"].perturbed_genes = list(perturbed_names)
        processed_graph["gene"].perturbation_indices = torch.tensor(
            perturbed_indices, dtype=torch.long, device=self.device
        )
        processed_graph["gene"].pert_mask = pert_mask

        # Group perturbation indices by experiment for batch processing
        batch_indices = []
        batch_mapping = {}  # Maps experiment index to list of perturbation indices

        for i, item in enumerate(data):
            perturbed_genes = [
                p.systematic_gene_name
                for p in item["experiment"].genotype.perturbations
            ]
            gene_indices = [
                node_ids.index(gene) for gene in perturbed_genes if gene in node_ids
            ]

            batch_indices.extend([i] * len(gene_indices))
            batch_mapping[i] = gene_indices

        # Add batch indices for perturbations - needed for processing by experiment
        if batch_indices:
            processed_graph["gene"].perturbation_indices_batch = torch.tensor(
                batch_indices, dtype=torch.long, device=self.device
            )

        # Process gene ontology if it exists in cell_graph
        if "gene_ontology" in cell_graph.node_types:
            # Copy ALL gene ontology information from cell_graph
            processed_graph["gene_ontology"].num_nodes = cell_graph[
                "gene_ontology"
            ].num_nodes
            processed_graph["gene_ontology"].node_ids = cell_graph[
                "gene_ontology"
            ].node_ids

            # Copy the go_gene_strata_state and apply perturbations
            if hasattr(cell_graph["gene_ontology"], "go_gene_strata_state"):
                self._apply_go_gene_perturbations(
                    processed_graph, cell_graph, perturbed_indices
                )

        # Add phenotype data in COO format
        self._add_phenotype_data(processed_graph, phenotype_info, data)

        return processed_graph

    def _apply_go_gene_perturbations(
        self,
        processed_graph: HeteroData,
        cell_graph: HeteroData,
        perturbed_indices: list,
    ) -> None:
        """
        Copy go_gene_strata_state from cell_graph and flip perturbation bits.
        """
        # Copy the base state tensor from cell_graph
        base_state = (
            cell_graph["gene_ontology"].go_gene_strata_state.clone().to(self.device)
        )

        # Convert perturbed indices to set for O(1) lookup
        perturbed_set = set(perturbed_indices)

        # Flip state bits for perturbed genes
        # Column 1 contains gene_idx, column 3 contains state
        gene_indices = base_state[:, 1].long()
        mask = torch.tensor(
            [idx.item() in perturbed_set for idx in gene_indices],
            dtype=torch.bool,
            device=self.device,
        )
        base_state[mask, 3] = 0.0

        # Store the modified state tensor
        processed_graph["gene_ontology"].go_gene_strata_state = base_state

    def _add_phenotype_data(
        self,
        integrated_subgraph: HeteroData,
        phenotype_info: list[PhenotypeType],
        data: list[Dict[str, ExperimentType | ExperimentReferenceType]],
    ) -> None:
        """
        Add phenotype data to the graph in COO format.
        Optimized version that ensures all tensors are on the same device.
        """
        # Ensure device is set
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize storage for phenotype values and metadata
        all_values = []
        all_type_indices = []
        all_sample_indices = []  # Which sample each value belongs to
        phenotype_types = []

        # Initialize storage for statistic values
        all_stat_values = []
        all_stat_type_indices = []
        all_stat_sample_indices = []  # Which sample each stat value belongs to
        stat_types = []

        # Extract phenotype type information
        for phenotype_class in phenotype_info:
            label_name = phenotype_class.model_fields["label_name"].default
            stat_name = phenotype_class.model_fields["label_statistic_name"].default

            phenotype_types.append(label_name)
            if stat_name:
                stat_types.append(stat_name)

        # Process each experimental data point
        for item_idx, item in enumerate(data):
            # Get phenotype object from experiment
            phenotype = item["experiment"].phenotype

            # Process each phenotype type
            for type_idx, field_name in enumerate(phenotype_types):
                value = getattr(phenotype, field_name, None)
                if value is not None:
                    # Convert single values to lists for consistent handling
                    values = [value] if not isinstance(value, (list, tuple)) else value

                    # Add all values with their type indices and sample indices
                    all_values.extend(values)
                    all_type_indices.extend([type_idx] * len(values))
                    all_sample_indices.extend([item_idx] * len(values))

            # Process statistics in the same way
            for stat_type_idx, stat_field_name in enumerate(stat_types):
                stat_value = getattr(phenotype, stat_field_name, None)
                if stat_value is not None:
                    # Convert single values to lists for consistent handling
                    stat_values = (
                        [stat_value]
                        if not isinstance(stat_value, (list, tuple))
                        else stat_value
                    )

                    # Add all statistic values with their type indices and sample indices
                    all_stat_values.extend(stat_values)
                    all_stat_type_indices.extend([stat_type_idx] * len(stat_values))
                    all_stat_sample_indices.extend([item_idx] * len(stat_values))

        # Store phenotype data in the graph - create all tensors directly on the target device
        if all_values:
            integrated_subgraph["gene"]["phenotype_values"] = torch.tensor(
                all_values, dtype=torch.float, device=self.device
            )
            integrated_subgraph["gene"]["phenotype_type_indices"] = torch.tensor(
                all_type_indices, dtype=torch.long, device=self.device
            )
            integrated_subgraph["gene"]["phenotype_sample_indices"] = torch.tensor(
                all_sample_indices, dtype=torch.long, device=self.device
            )
        else:
            # Handle empty case with placeholder values - create directly on target device
            integrated_subgraph["gene"]["phenotype_values"] = torch.tensor(
                [float("nan")], dtype=torch.float, device=self.device
            )
            integrated_subgraph["gene"]["phenotype_type_indices"] = torch.tensor(
                [0], dtype=torch.long, device=self.device
            )
            integrated_subgraph["gene"]["phenotype_sample_indices"] = torch.tensor(
                [0], dtype=torch.long, device=self.device
            )

        # Store phenotype type names (non-tensor data)
        integrated_subgraph["gene"]["phenotype_types"] = phenotype_types

        # Store statistic data in the graph - create directly on target device
        if all_stat_values:
            integrated_subgraph["gene"]["phenotype_stat_values"] = torch.tensor(
                all_stat_values, dtype=torch.float, device=self.device
            )
            integrated_subgraph["gene"]["phenotype_stat_type_indices"] = torch.tensor(
                all_stat_type_indices, dtype=torch.long, device=self.device
            )
            integrated_subgraph["gene"]["phenotype_stat_sample_indices"] = torch.tensor(
                all_stat_sample_indices, dtype=torch.long, device=self.device
            )
            integrated_subgraph["gene"]["phenotype_stat_types"] = stat_types
