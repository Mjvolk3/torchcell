import io
import os
import os.path as osp
import wandb
import matplotlib.pyplot as plt
from matplotlib import rc_params_from_file
import numpy as np
from PIL import Image
from typing import Optional


class GraphRecoveryVisualization:
    """Visualization class for graph regularization and edge recovery metrics."""

    def __init__(self, base_dir: str, mplstyle_path: Optional[str] = None) -> None:
        self.base_dir = base_dir

        # Load color palette from mplstyle
        if mplstyle_path is None:
            # Default to torchcell.mplstyle in the package
            mplstyle_path = osp.join(
                osp.dirname(osp.dirname(__file__)), "torchcell.mplstyle"
            )
        self.colors = self._load_colors_from_mplstyle(mplstyle_path)

    def _load_colors_from_mplstyle(self, mplstyle_path: str) -> list[str]:
        """
        Parse color palette from mplstyle file using matplotlib's rc_params_from_file.

        Args:
            mplstyle_path: Path to .mplstyle file

        Returns:
            List of hex color strings with '#' prefix
        """
        try:
            # Use matplotlib's official parser for style files
            rc_params = rc_params_from_file(mplstyle_path, use_default_template=False)
            prop_cycle = rc_params.get('axes.prop_cycle')
            if prop_cycle is not None:
                colors = prop_cycle.by_key().get('color', [])
                # Ensure '#' prefix on all colors
                return ['#' + c if not c.startswith('#') else c for c in colors]
        except Exception as e:
            print(f"Warning: Could not load colors from {mplstyle_path}: {e}")

        # Fallback to full torchcell.mplstyle palette (22 colors)
        return ['#000000','#CC8250','#7191A9','#6B8D3A','#B73C39','#34699D',
                '#3D796E','#4A9C60','#E6A65D','#A05B2C','#3978B5','#D86E2F',
                '#775A9F','#EBB386','#8D5694','#52B2A8','#35978A','#AB4B4B',
                '#6D666F','#4E7C7F','#905353','#C17132']

    def save_and_log_figure(
        self, fig: plt.Figure, name: str, timestamp_str: Optional[str] = None
    ) -> None:
        """
        Log figure to wandb only (no disk save).

        Args:
            fig: Matplotlib figure
            name: Wandb panel key (e.g., "val_edge_recovery_summary/recall")
            timestamp_str: Unused, kept for API compatibility
        """
        # Guard: Check wandb is initialized
        if wandb.run is None:
            print(f"Warning: wandb.run is None, skipping log for {name}")
            plt.close(fig)
            return

        # Allow all ranks to log plots (DDP rank guard removed)

        # Log to wandb only
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=300)
        buf.seek(0)
        wandb_image = wandb.Image(Image.open(buf))
        wandb.log({name: wandb_image}, commit=True)
        buf.close()

    def plot_graph_info_summary(
        self,
        graph_info: dict[str, dict[str, float]],
        save_path: Optional[str] = None,
    ) -> None:
        """
        Create a comprehensive summary visualization of graph statistics.

        Args:
            graph_info: Dict mapping graph_name -> {num_edges, num_nodes, avg_degree, reg_layer, reg_head}
            save_path: Optional path to save the figure to disk
        """
        if not graph_info:
            print("No graph info to plot")
            return

        # Sort graph names alphabetically
        graph_names = sorted(graph_info.keys())
        n_graphs = len(graph_names)

        # Create figure with subplots
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 2, hspace=0.4, wspace=0.3, height_ratios=[1, 0.1, 1])

        # Subplot 1: Number of edges
        ax1 = fig.add_subplot(gs[0, 0])
        edges = [graph_info[name].get("num_edges", 0) for name in graph_names]
        bars1 = ax1.bar(range(n_graphs), edges, color=self.colors[2], alpha=0.8)

        # Add headspace and bold labels
        max_edge = max(edges) if edges else 1
        ax1.set_ylim(0, max_edge * 1.15)

        for bar, val in zip(bars1, edges):
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height(),
                f"{int(val):,}",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )
        ax1.set_xlabel("Graph Type", fontsize=11)
        ax1.set_ylabel("Number of Edges", fontsize=11)
        ax1.set_title("Edge Count by Graph Type", fontsize=12, fontweight="bold")
        ax1.set_xticks(range(n_graphs))
        ax1.set_xticklabels(graph_names, rotation=45, ha="right", fontsize=9)
        ax1.grid(axis="y", alpha=0.3)

        # Subplot 2: Average degree
        ax2 = fig.add_subplot(gs[0, 1])
        degrees = [graph_info[name].get("avg_degree", 0) for name in graph_names]
        bars2 = ax2.bar(range(n_graphs), degrees, color=self.colors[1], alpha=0.8)

        # Add headspace and bold labels
        max_degree = max(degrees) if degrees else 1
        ax2.set_ylim(0, max_degree * 1.15)

        for bar, val in zip(bars2, degrees):
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height(),
                f"{val:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )
        ax2.set_xlabel("Graph Type", fontsize=11)
        ax2.set_ylabel("Average Degree", fontsize=11)
        ax2.set_title("Graph Density (Avg Degree)", fontsize=12, fontweight="bold")
        ax2.set_xticks(range(n_graphs))
        ax2.set_xticklabels(graph_names, rotation=45, ha="right", fontsize=9)
        ax2.grid(axis="y", alpha=0.3)

        # Subplot 3: Regularization mapping table
        ax3 = fig.add_subplot(gs[2, :])
        ax3.axis("tight")
        ax3.axis("off")

        # Prepare table data
        table_data = []
        for name in graph_names:
            info = graph_info[name]

            # Handle reg_layer: can be int, list[int], or missing
            if "reg_layer" in info:
                reg_layer_val = info.get("reg_layer")
                if isinstance(reg_layer_val, list):
                    reg_layer_str = str(reg_layer_val)
                else:
                    reg_layer_str = str(int(reg_layer_val))
            else:
                reg_layer_str = "N/A"

            # Handle reg_head: can be int or missing
            if "reg_head" in info:
                reg_head_str = str(int(info.get("reg_head")))
            else:
                reg_head_str = "N/A"

            row = [
                name,
                f"{int(info.get('num_nodes', 0)):,}",
                f"{int(info.get('num_edges', 0)):,}",
                f"{info.get('avg_degree', 0):.2f}",
                reg_layer_str,
                reg_head_str,
            ]
            table_data.append(row)

        # Create table
        table = ax3.table(
            cellText=table_data,
            colLabels=[
                "Graph",
                "Nodes",
                "Edges",
                "Avg Degree",
                "Reg Layer",
                "Reg Head",
            ],
            cellLoc="center",
            loc="center",
            bbox=[0, 0, 1, 1],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)

        # Style header row
        for i in range(6):
            table[(0, i)].set_facecolor("#4A4A4A")
            table[(0, i)].set_text_props(weight="bold", color="white")

        # Alternate row colors (grayscale)
        for i in range(1, len(table_data) + 1):
            for j in range(6):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor("#F0F0F0")
                else:
                    table[(i, j)].set_facecolor("#FFFFFF")

        fig.suptitle(
            "Graph Statistics Summary", fontsize=14, fontweight="bold", y=0.98
        )

        # Save to disk if path provided
        if save_path:
            fig.savefig(save_path, format="png", bbox_inches="tight", dpi=300)
            print(f"Graph info summary saved to: {save_path}")

        # Log to wandb under graph_regularization_info panel
        self.save_and_log_figure(fig, "graph_regularization_info/summary", None)
        plt.close()

    def plot_edge_recovery_recall(
        self,
        recall_metrics: dict[str, float],
        num_epochs: int,
        timestamp_str: Optional[str] = None,
        stage: str = "val",
    ) -> None:
        """
        Plot recall@degree for each graph as a bar chart.

        Args:
            recall_metrics: Dict mapping metric_key (e.g., "physical_L0_H1") -> recall@degree value
            num_epochs: Current epoch number
            timestamp_str: Unused, kept for API compatibility
            stage: Stage name (e.g., "val")
        """
        if not recall_metrics:
            print("No recall metrics to plot")
            return

        # Parse metric keys to extract graph names and layers
        # Format: "graph_name_L{layer}_H{head}"
        parsed_metrics = []
        for metric_key, recall_value in recall_metrics.items():
            # Extract graph name (everything before _L)
            parts = metric_key.split("_L")
            if len(parts) < 2:
                # Fallback for keys without layer info
                graph_name = metric_key
                layer_idx = 0
            else:
                graph_name = parts[0]
                # Extract layer number
                layer_part = parts[1].split("_H")[0]
                try:
                    layer_idx = int(layer_part)
                except ValueError:
                    layer_idx = 0

            # Create display label
            if "_L" in metric_key:
                label = f"{graph_name} (L{layer_idx})"
            else:
                label = graph_name

            parsed_metrics.append((graph_name, layer_idx, label, recall_value))

        # Sort by graph name, then layer
        parsed_metrics.sort(key=lambda x: (x[0], x[1]))

        # Extract sorted labels and values
        labels = [x[2] for x in parsed_metrics]
        recall_values = [x[3] for x in parsed_metrics]
        graph_names = [x[0] for x in parsed_metrics]

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))

        # Assign colors based on graph name
        unique_graphs = []
        graph_to_color_idx = {}
        for graph_name in graph_names:
            if graph_name not in graph_to_color_idx:
                graph_to_color_idx[graph_name] = len(unique_graphs)
                unique_graphs.append(graph_name)

        bar_colors = [self.colors[graph_to_color_idx[g] % len(self.colors)] for g in graph_names]

        # Create bar chart with colors by graph name
        bars = ax.bar(
            range(len(labels)),
            recall_values,
            color=bar_colors,
            alpha=0.8
        )

        # Use natural bounds for recall [0, 1]
        ax.set_ylim(0, 1.0)

        # Add bold value labels on top of bars
        for bar, val in zip(bars, recall_values):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

        # Formatting
        ax.set_xlabel("Graph + Layer", fontsize=12)
        ax.set_ylabel("Recall@Degree", fontsize=12)
        ax.set_title(
            f"Edge Recovery: Recall@Degree\nEpoch {num_epochs}",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()

        # Save and log to summary panel
        key = f"{stage}_edge_recovery_summary/recall"
        self.save_and_log_figure(fig, key, timestamp_str)
        plt.close()

    def plot_edge_recovery_precision(
        self,
        precision_metrics: dict[str, dict[int, float]],
        k_values: list[int],
        num_epochs: int,
        timestamp_str: Optional[str] = None,
        stage: str = "val",
    ) -> None:
        """
        Plot precision@k for each graph as line plots.

        Args:
            precision_metrics: Dict mapping metric_key (e.g., "physical_L0_H1") -> {k -> precision value}
            k_values: List of k values (e.g., [8, 32, 128, 320])
            num_epochs: Current epoch number
            timestamp_str: Unused, kept for API compatibility
            stage: Stage name (e.g., "val")
        """
        if not precision_metrics:
            print("No precision metrics to plot")
            return

        # Parse metric keys to extract graph names and layers
        # Format: "graph_name_L{layer}_H{head}"
        parsed_metrics = {}
        for metric_key in precision_metrics.keys():
            # Extract graph name (everything before _L)
            parts = metric_key.split("_L")
            if len(parts) < 2:
                # Fallback for keys without layer info
                graph_name = metric_key
                layer_idx = 0
            else:
                graph_name = parts[0]
                # Extract layer number
                layer_part = parts[1].split("_H")[0]
                try:
                    layer_idx = int(layer_part)
                except ValueError:
                    layer_idx = 0

            if graph_name not in parsed_metrics:
                parsed_metrics[graph_name] = []
            parsed_metrics[graph_name].append((metric_key, layer_idx))

        # Sort graph names alphabetically
        graph_names = sorted(parsed_metrics.keys())

        # Collect all unique layer indices and create sequential mapping
        all_layer_indices = set()
        for entries in parsed_metrics.values():
            for _, layer_idx in entries:
                all_layer_indices.add(layer_idx)
        sorted_layers = sorted(all_layer_indices)
        layer_to_style_idx = {
            layer: idx for idx, layer in enumerate(sorted_layers)
        }

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))

        # Line styles for different layers (sequential mapping)
        line_styles = ['-', '--', '-.', ':']  # solid, dashed, dash-dot, dotted

        # Assign one color per graph name
        colors = {
            name: self.colors[i % len(self.colors)]
            for i, name in enumerate(graph_names)
        }

        # Plot lines grouped by graph name
        data_lines = []  # For graph legend
        for graph_name in graph_names:
            entries = parsed_metrics[graph_name]
            metric_entries = sorted(entries, key=lambda x: x[1])  # Sort by layer
            color = colors[graph_name]

            for metric_key, layer_idx in metric_entries:
                prec_dict = precision_metrics[metric_key]
                # Get precision values for this metric at each k
                prec_values = [prec_dict.get(k, 0.0) for k in k_values]

                # Map actual layer index to sequential style index
                style_idx = layer_to_style_idx[layer_idx]
                linestyle = line_styles[style_idx % len(line_styles)]

                # Plot line with markers (no label yet)
                line = ax.plot(
                    k_values,
                    prec_values,
                    marker="o",
                    markersize=8,
                    linewidth=2,
                    linestyle=linestyle,
                    color=color,
                    alpha=0.8,
                )[0]

                # Store first line per graph for legend
                if not any(dl[1] == graph_name for dl in data_lines):
                    data_lines.append((line, graph_name))

        # Create two separate legends
        # Legend 1: Graph names (colors)
        graph_legend_handles = [line for line, _ in data_lines]
        graph_legend_labels = [name for _, name in data_lines]
        legend1 = ax.legend(
            graph_legend_handles,
            graph_legend_labels,
            loc="upper left",
            fontsize=9,
            framealpha=0.9,
            title="Graph Type",
            title_fontsize=10
        )

        # Legend 2: Layer indices (line styles)
        from matplotlib.lines import Line2D
        layer_legend_handles = []
        layer_legend_labels = []
        for layer_idx in sorted_layers:
            style_idx = layer_to_style_idx[layer_idx]
            linestyle = line_styles[style_idx % len(line_styles)]
            handle = Line2D(
                [0], [0],
                color='gray',
                linewidth=2,
                linestyle=linestyle,
                marker='o',
                markersize=6
            )
            layer_legend_handles.append(handle)
            layer_legend_labels.append(f"Layer {layer_idx}")

        ax.legend(
            layer_legend_handles,
            layer_legend_labels,
            loc="upper right",
            fontsize=9,
            framealpha=0.9,
            title="Reg Layer",
            title_fontsize=10
        )

        # Add first legend back to the plot (second legend is added automatically)
        ax.add_artist(legend1)

        # Formatting
        ax.set_xlabel("k (Top-k Attention)", fontsize=12)
        ax.set_ylabel("Precision@k", fontsize=12)
        ax.set_title(
            f"Edge Recovery: Precision@k\nEpoch {num_epochs}",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_xscale("log")
        ax.set_xticks(k_values)
        ax.set_xticklabels([str(k) for k in k_values])
        ax.set_ylim(0, 1.0)
        ax.grid(True, alpha=0.3, which="both")

        plt.tight_layout()

        # Save and log to summary panel
        key = f"{stage}_edge_recovery_summary/precision"
        self.save_and_log_figure(fig, key, timestamp_str)
        plt.close()

    def plot_edge_recovery_per_graph(
        self,
        recall_metrics: dict[str, float],
        precision_metrics: dict[str, dict[int, float]],
        k_values: list[int],
        num_epochs: int,
        timestamp_str: Optional[str] = None,
        stage: str = "val",
    ) -> None:
        """
        Create individual edge recovery plots for each graph showing both recall and precision@k.

        Args:
            recall_metrics: Dict mapping graph_name -> recall@degree value
            precision_metrics: Dict mapping graph_name -> {k -> precision value}
            k_values: List of k values (e.g., [8, 32, 128, 320])
            num_epochs: Current epoch number
            timestamp_str: Unused, kept for API compatibility
            stage: Stage name (e.g., "val")
        """
        if not recall_metrics or not precision_metrics:
            print("No edge recovery metrics to plot")
            return

        # Sort graphs alphabetically
        graph_names = sorted(recall_metrics.keys())

        # Create a plot for each graph
        for graph_name in graph_names:
            if graph_name not in precision_metrics:
                continue

            recall_val = recall_metrics[graph_name]
            prec_dict = precision_metrics[graph_name]
            prec_values = [prec_dict.get(k, 0.0) for k in k_values]

            # Create figure
            fig, ax = plt.subplots(figsize=(10, 6))

            # Plot precision@k as bars
            x_pos = np.arange(len(k_values))
            bars = ax.bar(
                x_pos,
                prec_values,
                alpha=0.8,
                color=self.colors[2],  # #7191A9
                label="Precision@k"
            )

            # Add headspace and bold value labels on bars
            max_prec = max(prec_values) if prec_values else 1.0
            ax.set_ylim(0, min(max(max_prec, recall_val) * 1.15, 1.0))

            for bar, val in zip(bars, prec_values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    bar.get_height(),
                    f"{val:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    fontweight="bold",
                )

            # Add recall@degree as horizontal line
            ax.axhline(
                y=recall_val,
                color=self.colors[4],  # #B73C39 (red-ish)
                linestyle="--",
                linewidth=2,
                label=f"Recall@degree = {recall_val:.3f}",
            )

            # Formatting
            ax.set_xlabel("k (Top-k Attention)", fontsize=12)
            ax.set_ylabel("Score", fontsize=12)
            ax.set_title(
                f"Edge Recovery: {graph_name}\nEpoch {num_epochs}",
                fontsize=13,
                fontweight="bold",
            )
            ax.set_xticks(x_pos)
            ax.set_xticklabels([str(k) for k in k_values])
            ax.grid(axis="y", alpha=0.3)
            ax.legend(loc="upper right", fontsize=10)

            plt.tight_layout()

            # Save and log
            safe_graph_name = graph_name.replace("/", "_")
            key = f"{stage}_edge_recovery/per_graph/{safe_graph_name}"
            self.save_and_log_figure(fig, key, timestamp_str)
            plt.close()

    def plot_edge_mass_alignment(
        self,
        edge_mass_metrics: dict[str, float],
        num_epochs: int,
        timestamp_str: Optional[str] = None,
        stage: str = "val",
    ) -> None:
        """
        Plot edge-mass alignment for each graph as a bar chart.

        Edge-mass alignment = fraction of total attention mass on known graph edges.

        Args:
            edge_mass_metrics: Dict mapping metric_key (e.g., "physical_L0_H1") -> edge_mass_fraction
            num_epochs: Current epoch number
            timestamp_str: Unused, kept for API compatibility
            stage: Stage name (e.g., "val")
        """
        if not edge_mass_metrics:
            print("No edge-mass metrics to plot")
            return

        # Parse metric keys to extract graph names and create labels
        # Format: "graph_name_L{layer}_H{head}"
        parsed_metrics = []
        for metric_key, edge_mass_value in edge_mass_metrics.items():
            # Extract graph name (everything before _L)
            parts = metric_key.split("_L")
            if len(parts) < 2:
                # Fallback for keys without layer info
                graph_name = metric_key
                label = metric_key
            else:
                graph_name = parts[0]
                # Extract layer and head for label
                layer_head = parts[1]  # e.g., "0_H1"
                label = f"{graph_name} (L{layer_head.replace('_H', ' H')})"

            parsed_metrics.append((graph_name, label, edge_mass_value))

        # Sort by graph name, then label
        parsed_metrics.sort(key=lambda x: (x[0], x[1]))

        # Extract sorted data
        graph_names = [x[0] for x in parsed_metrics]
        labels = [x[1] for x in parsed_metrics]
        edge_mass_values = [x[2] for x in parsed_metrics]

        # Create figure
        fig, ax = plt.subplots(figsize=(14, 6))

        # Assign colors based on graph name
        unique_graphs = []
        graph_to_color_idx = {}
        for graph_name in graph_names:
            if graph_name not in graph_to_color_idx:
                graph_to_color_idx[graph_name] = len(unique_graphs)
                unique_graphs.append(graph_name)

        bar_colors = [self.colors[graph_to_color_idx[g] % len(self.colors)] for g in graph_names]

        # Create bar chart with colors by graph name
        bars = ax.bar(
            range(len(labels)),
            edge_mass_values,
            color=bar_colors,
            alpha=0.8
        )

        # Use natural bounds [0, 1]
        ax.set_ylim(0, 1.0)

        # Add bold value labels on top of bars
        for bar, val in zip(bars, edge_mass_values):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

        # Add reference line at random baseline (varies by graph density)
        ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Random baseline (approx.)')

        # Formatting
        ax.set_xlabel("Graph + Layer + Head", fontsize=12)
        ax.set_ylabel("Edge-Mass Fraction", fontsize=12)
        ax.set_title(
            f"Edge-Mass Alignment: Fraction of Attention on Known Edges\nEpoch {num_epochs}",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(loc="upper right")

        plt.tight_layout()

        # Save and log to summary panel
        key = f"{stage}_edge_recovery_summary/edge_mass"
        self.save_and_log_figure(fig, key, timestamp_str)
        plt.close()
