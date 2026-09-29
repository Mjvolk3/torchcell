torchcell.sga
=============

.. module:: torchcell.sga

.. currentmodule:: torchcell.sga

A colony-fitness pipeline adapted from SGAtools for single CRISPR knockouts dispensed onto plates by an ECHO acoustic liquid handler. It reads the gitter colony-size file and the ECHO picklist (``io``), corrects positional plate artifacts (``normalize``), and scores each knockout as fitness relative to the on-plate BY4741 wild type (``score``). ``image`` and ``cellpose_seg`` measure colony sizes from plate photographs, and ``viz`` draws plate heatmaps and histograms. The typical call sequence is :func:`~torchcell.sga.read_gitter_dat`, :func:`~torchcell.sga.read_echo_picklist`, :func:`~torchcell.sga.merge_layout`, :func:`~torchcell.sga.normalize_plate`, then :func:`~torchcell.sga.score_plate`.

.. contents:: Contents
    :local:

Classes
-------

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   NormalizationConfig
   ScoreReport
   StrainScore
   CellposeSegConfig
   PlateSegResult

Functions
---------

.. autosummary::
   :nosignatures:
   :toctree: ../generated

   read_gitter_dat
   read_echo_picklist
   merge_layout
   well_to_rowcol
   normalize_plate
   score_plate
   score_table
   quantify_plate_image
   quantify_plate_image_cellpose
   load_cellpose_model
   resolve_orientation
   volume_assay_metrics
   volume_position_confound
   recommend_volume
   zfactor
