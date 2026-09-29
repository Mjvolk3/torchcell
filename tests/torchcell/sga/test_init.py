# tests/torchcell/sga/test_init.py
# [[tests.torchcell.sga.test_init]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/sga/test_init.py
"""The ``torchcell.sga`` package surface: ``__all__`` and where each name comes from."""

from __future__ import annotations

import torchcell.sga as sga
from torchcell.sga import (
    assay,
    cellpose_seg,
    image,
    io,
    models,
    normalize,
    register,
    score,
)

EXPECTED_ALL = [
    "read_gitter_dat",
    "read_echo_picklist",
    "merge_layout",
    "well_to_rowcol",
    "NormalizationConfig",
    "ScoreReport",
    "StrainScore",
    "normalize_plate",
    "score_plate",
    "score_table",
    "quantify_plate_image",
    "quantify_plate_image_cellpose",
    "load_cellpose_model",
    "CellposeSegConfig",
    "PlateSegResult",
    "resolve_orientation",
    "volume_assay_metrics",
    "volume_position_confound",
    "recommend_volume",
    "zfactor",
]


def test_all_is_the_documented_surface() -> None:
    """Twenty names, in the documented order, each bound to the package."""
    assert sga.__all__ == EXPECTED_ALL
    assert all(hasattr(sga, name) for name in EXPECTED_ALL)


def test_reexports_are_the_module_objects() -> None:
    """Each public name is the very object defined in its submodule (no wrappers)."""
    assert sga.read_gitter_dat is io.read_gitter_dat
    assert sga.read_echo_picklist is io.read_echo_picklist
    assert sga.merge_layout is io.merge_layout
    assert sga.well_to_rowcol is io.well_to_rowcol
    assert sga.NormalizationConfig is models.NormalizationConfig
    assert sga.ScoreReport is models.ScoreReport
    assert sga.StrainScore is models.StrainScore
    assert sga.normalize_plate is normalize.normalize_plate
    assert sga.score_plate is score.score_plate
    assert sga.score_table is score.score_table
    assert sga.quantify_plate_image is image.quantify_plate_image
    assert (
        sga.quantify_plate_image_cellpose is cellpose_seg.quantify_plate_image_cellpose
    )
    assert sga.load_cellpose_model is cellpose_seg.load_cellpose_model
    assert sga.CellposeSegConfig is cellpose_seg.CellposeSegConfig
    assert sga.PlateSegResult is cellpose_seg.PlateSegResult
    assert sga.resolve_orientation is register.resolve_orientation
    assert sga.volume_assay_metrics is assay.volume_assay_metrics
    assert sga.volume_position_confound is assay.volume_position_confound
    assert sga.recommend_volume is assay.recommend_volume
    assert sga.zfactor is assay.zfactor


def test_viz_and_shape_helpers_are_not_exported() -> None:
    """``viz`` and ``assay.shape_by_volume`` are reachable by submodule only."""
    assert "shape_by_volume" not in sga.__all__
    assert not hasattr(sga, "plate_heatmap")
