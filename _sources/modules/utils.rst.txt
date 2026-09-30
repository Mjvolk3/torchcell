torchcell.utils
===============

.. module:: torchcell.utils

.. currentmodule:: torchcell.utils

Shared helpers. ``paths`` resolves output directories relative to the current checkout (so a script run from a git worktree writes into that worktree); ``utils`` holds the repository-wide figure standards (the ordered plot palette, Nature panel widths in millimeters, :func:`~torchcell.utils.savefig_true_size_svg` and :func:`~torchcell.utils.apply_paper_style`); and ``file_lock`` provides :class:`~torchcell.utils.FileLockHelper` for locked JSON reads and writes.

.. contents:: Contents
    :local:

Classes
-------

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   FileLockHelper

Functions
---------

.. autosummary::
   :nosignatures:
   :toctree: ../generated

   repo_root
   experiment_root
   experiment_results_dir
   asset_images_dir
   format_scientific_notation
   savefig_true_size_svg
   mm_to_in
   apply_paper_style
   display_label
   panel_label

Constants
---------

.. autosummary::
   :nosignatures:
   :toctree: ../generated

   PANEL_WIDTHS_MM
   MAX_HEIGHT_MM
   PLOT_PALETTE
   PLOT_PALETTE_NAMES
   PLOT_PALETTE_FILL
   REPRESENTATION_DISPLAY_NAMES
   PAPER_RC
