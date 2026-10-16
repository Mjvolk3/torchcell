"""General-purpose utility helpers for torchcell."""

# torchcell/utils/__init__.py
from .file_lock import FileLockHelper
from .paths import asset_images_dir, experiment_results_dir, experiment_root, repo_root
from .utils import (
    ACS_BOND_PT,
    ACS_FONT_PT,
    ACS_LINE_PT,
    MAX_HEIGHT_MM,
    PANEL_WIDTHS_MM,
    PAPER_RC,
    PLOT_PALETTE,
    PLOT_PALETTE_FILL,
    PLOT_PALETTE_NAMES,
    REPRESENTATION_DISPLAY_NAMES,
    acs_structure,
    acs_structure_box,
    acs_structure_svg,
    apply_paper_style,
    display_label,
    format_scientific_notation,
    mm_to_in,
    panel_label,
    savefig_true_size_svg,
)

__all__ = [
    "ACS_BOND_PT",
    "ACS_FONT_PT",
    "ACS_LINE_PT",
    "acs_structure",
    "acs_structure_box",
    "acs_structure_svg",
    "FileLockHelper",
    "repo_root",
    "experiment_root",
    "experiment_results_dir",
    "asset_images_dir",
    "format_scientific_notation",
    "savefig_true_size_svg",
    "mm_to_in",
    "PANEL_WIDTHS_MM",
    "MAX_HEIGHT_MM",
    "PLOT_PALETTE",
    "PLOT_PALETTE_NAMES",
    "PLOT_PALETTE_FILL",
    "REPRESENTATION_DISPLAY_NAMES",
    "PAPER_RC",
    "apply_paper_style",
    "display_label",
    "panel_label",
]
