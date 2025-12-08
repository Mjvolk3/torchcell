# torchcell/viz/transformer_diagnostics
# [[torchcell.viz.transformer_diagnostics]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/viz/transformer_diagnostics
# Test file: tests/torchcell/viz/test_transformer_diagnostics.py

import io
import os
import os.path as osp
import wandb
import matplotlib.pyplot as plt
from matplotlib import rc_params_from_file
import numpy as np
from PIL import Image
from typing import Optional, Dict, List


class TransformerDiagnostics:
    """Visualization class for transformer-specific diagnostic metrics."""

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
            name: Wandb panel key (e.g., "val_transformer_diagnostics/summary")
            timestamp_str: Unused, kept for API compatibility
        """
        # Log to wandb only
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=300)
        buf.seek(0)
        wandb_image = wandb.Image(Image.open(buf))
        wandb.log({name: wandb_image}, commit=True)
        buf.close()

    def plot_attention_diagnostics(
        self,
        attention_stats: Dict[int, Dict[str, float]],
        residual_ratios: Optional[Dict[int, float]] = None,
        qk_logit_stats: Optional[Dict[int, Dict[str, float]]] = None,
        gradient_norms: Optional[Dict[int, float]] = None,
        num_epochs: int = 0,
        stage: str = "val",
    ) -> None:
        """
        Create aggregated diagnostic plot for transformer attention metrics.

        Args:
            attention_stats: Dict mapping layer_idx -> {
                "entropy": float, "effective_rank": float,
                "top5": float, "top10": float, "top50": float,
                "max_row_weight": float, "col_entropy": float, "max_col_sum": float
            }
            residual_ratios: Optional dict mapping layer_idx -> residual_update_ratio
            qk_logit_stats: Optional dict mapping layer_idx -> {
                "logit_mean": float, "logit_std": float, "saturation_ratio": float
            }
            gradient_norms: Optional dict mapping layer_idx -> gradient_norm (not currently plotted)
            num_epochs: Current epoch number
            stage: Stage name (e.g., "val")
        """
        if not attention_stats:
            print("No attention stats to plot")
            return

        # Sort layers
        layer_indices = sorted(attention_stats.keys())

        # Extract existing metrics
        entropies = [attention_stats[l]["entropy"] for l in layer_indices]
        effective_ranks = [attention_stats[l]["effective_rank"] for l in layer_indices]
        top5s = [attention_stats[l]["top5"] for l in layer_indices]
        top10s = [attention_stats[l]["top10"] for l in layer_indices]
        top50s = [attention_stats[l]["top50"] for l in layer_indices]

        # Extract NEW metrics
        max_row_weights = [attention_stats[l].get("max_row_weight", 0.0) for l in layer_indices]
        col_entropies = [attention_stats[l].get("col_entropy", 0.0) for l in layer_indices]
        max_col_sums = [attention_stats[l].get("max_col_sum", 0.0) for l in layer_indices]

        # Create grid layout: 3 rows × 2 columns - always 6 subplots
        fig, axes = plt.subplots(3, 2, figsize=(20, 15))
        axes = axes.flatten()  # Flatten to 1D array for easier indexing
        ax1, ax2, ax3, ax4, ax5, ax6 = axes

        # Subplot 1: Attention Entropy
        ax1.plot(
            layer_indices,
            entropies,
            marker='o',
            linewidth=2,
            markersize=8,
            color=self.colors[2],  # Steel blue
            label="Attention Entropy"
        )
        for layer, entropy in zip(layer_indices, entropies):
            ax1.text(
                layer,
                entropy,
                f"{entropy:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )
        ax1.set_xlabel("Transformer Layer", fontsize=12)
        ax1.set_ylabel("Entropy (nats)", fontsize=12)
        ax1.set_title("Attention Entropy per Layer", fontsize=14, fontweight="bold")
        ax1.set_xticks(layer_indices)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc="best")

        # Subplot 2: Effective Rank
        ax2.plot(
            layer_indices,
            effective_ranks,
            marker='s',
            linewidth=2,
            markersize=8,
            color=self.colors[4],  # Warm red
            label="Effective Rank"
        )
        for layer, eff_rank in zip(layer_indices, effective_ranks):
            ax2.text(
                layer,
                eff_rank,
                f"{eff_rank:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )
        ax2.set_xlabel("Transformer Layer", fontsize=12)
        ax2.set_ylabel("Effective Rank (# effective positions)", fontsize=12)
        ax2.set_title("Attention Concentration (Effective Rank)", fontsize=14, fontweight="bold")
        ax2.set_xticks(layer_indices)
        ax2.set_yscale('log')  # Log scale for better visibility
        ax2.grid(True, alpha=0.3, which="both")
        ax2.legend(loc="best")

        # Subplot 3: Top-K Concentration
        ax3.plot(
            layer_indices,
            top5s,
            marker='o',
            linewidth=2,
            markersize=8,
            color=self.colors[0],  # Black
            label="Top-5 Mass",
            alpha=0.8
        )
        ax3.plot(
            layer_indices,
            top10s,
            marker='s',
            linewidth=2,
            markersize=8,
            color=self.colors[1],  # Warm orange
            label="Top-10 Mass",
            alpha=0.8
        )
        ax3.plot(
            layer_indices,
            top50s,
            marker='^',
            linewidth=2,
            markersize=8,
            color=self.colors[3],  # Olive green
            label="Top-50 Mass",
            alpha=0.8
        )
        ax3.set_xlabel("Transformer Layer", fontsize=12)
        ax3.set_ylabel("Fraction of Attention Mass", fontsize=12)
        ax3.set_title("Top-K Attention Concentration", fontsize=14, fontweight="bold")
        ax3.set_xticks(layer_indices)
        ax3.set_ylim(0, 1.0)  # Natural bounds [0, 1]
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc="best")

        # Subplot 4: Max Row Weight (One-Hot Detection)
        ax4.plot(
            layer_indices,
            max_row_weights,
            marker='d',
            linewidth=2,
            markersize=8,
            color=self.colors[5],  # Deep blue
            label="Max Row Weight"
        )
        for layer, weight in zip(layer_indices, max_row_weights):
            ax4.text(
                layer,
                weight,
                f"{weight:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )
        ax4.set_xlabel("Transformer Layer", fontsize=12)
        ax4.set_ylabel("Max Attention Weight", fontsize=12)
        ax4.set_title("Max Row Weight (One-Hot Detection)", fontsize=14, fontweight="bold")
        ax4.set_xticks(layer_indices)
        ax4.set_ylim(0, 1.0)
        ax4.grid(True, alpha=0.3)
        ax4.legend(loc="best")

        # Subplot 5: Column Concentration (Sink Collapse Detection)
        ax5_twin = ax5.twinx()

        # Plot column entropy on left axis
        line1 = ax5.plot(
            layer_indices,
            col_entropies,
            marker='o',
            linewidth=2,
            markersize=8,
            color=self.colors[6],  # Teal
            label="Column Entropy"
        )

        # Plot max column sum on right axis (different scale)
        line2 = ax5_twin.plot(
            layer_indices,
            max_col_sums,
            marker='s',
            linewidth=2,
            markersize=8,
            color=self.colors[7],  # Green
            label="Max Column Sum"
        )

        # Formatting
        ax5.set_xlabel("Transformer Layer", fontsize=12)
        ax5.set_ylabel("Column Entropy (nats)", fontsize=12, color=self.colors[6])
        ax5_twin.set_ylabel("Max Column Sum", fontsize=12, color=self.colors[7])
        ax5.set_title("Column Concentration (Sink Collapse Detection)", fontsize=14, fontweight="bold")
        ax5.set_xticks(layer_indices)
        ax5.grid(True, alpha=0.3)

        # Combined legend
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax5.legend(lines, labels, loc="best")

        # Subplot 6: Residual Update Ratio (Layer Health)
        if residual_ratios:
            ratios = [residual_ratios.get(l, 0.0) for l in layer_indices]
        else:
            ratios = [0.0] * len(layer_indices)

        ax6.plot(
            layer_indices,
            ratios,
            marker='d',
            linewidth=2,
            markersize=8,
            color=self.colors[10],  # Purple
            label="Residual Update Ratio"
        )

        # Add reference lines for healthy range
        ax6.axhline(y=0.1, color=self.colors[3], linestyle='--', alpha=0.5, linewidth=1.5, label="Healthy Lower Bound")
        ax6.axhline(y=1.0, color=self.colors[4], linestyle='--', alpha=0.5, linewidth=1.5, label="Healthy Upper Bound")

        # Add value labels
        for layer, ratio in zip(layer_indices, ratios):
            ax6.text(
                layer,
                ratio,
                f"{ratio:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )

        ax6.set_xlabel("Transformer Layer", fontsize=12)
        ax6.set_ylabel("Update Ratio (||Δx|| / ||x||)", fontsize=12)
        ax6.set_title("Residual Update Ratio (Layer Health)", fontsize=14, fontweight="bold")
        ax6.set_xticks(layer_indices)
        ax6.set_yscale('log')  # Log scale to see both small and large values
        ax6.grid(True, alpha=0.3, which="both")
        ax6.legend(loc="best")

        # Overall title
        fig.suptitle(
            f"Transformer Attention Diagnostics - Epoch {num_epochs}",
            fontsize=16,
            fontweight="bold",
            y=0.995
        )

        plt.tight_layout()

        # Save and log
        key = f"{stage}_transformer_diagnostics/summary"
        self.save_and_log_figure(fig, key, None)
        plt.close()
