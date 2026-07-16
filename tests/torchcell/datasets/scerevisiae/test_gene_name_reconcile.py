"""Tests for the shared retain-all gene-name reconciliation helper."""

import os
import os.path as osp

import pandas as pd
import pytest
from dotenv import load_dotenv

from torchcell.datasets.scerevisiae.gene_name_reconcile import (
    reconcile_systematic_names,
)
from torchcell.sequence.genome.scerevisiae import SCerevisiaeGenome

load_dotenv()

DATA_ROOT = os.getenv("DATA_ROOT")

if DATA_ROOT is None:
    pytest.skip("requires DATA_ROOT data (absent in CI)", allow_module_level=True)


@pytest.fixture(scope="module")
def genome():
    assert DATA_ROOT is not None
    return SCerevisiaeGenome(
        genome_root=osp.join(DATA_ROOT, "data/sgd/genome"),
        go_root=osp.join(DATA_ROOT, "data/go"),
        overwrite=False,
    )


def test_reconcile_retains_all_and_remaps(genome):
    # YGR272C is a clean SGD rename to YGR271C-A (no competing strain here) -> remapped.
    # YAR037W is retired -> retained verbatim. A current gene passes through unchanged.
    names = pd.Series(["YAL002W", "YGR272C", "YAR037W"])
    out = reconcile_systematic_names(genome, names, label="test")
    assert len(out) == len(names)  # nothing dropped
    assert out.tolist() == ["YAL002W", "YGR271C-A", "YAR037W"]


def test_reconcile_collision_keeps_both_originals(genome):
    # YDL038C is an SGD alias of the gene YDL039C, which is ALSO present as its own strain.
    # Remapping YDL038C->YDL039C would collide, so BOTH keep their original names.
    names = pd.Series(["YDL038C", "YDL039C"])
    out = reconcile_systematic_names(genome, names, label="test")
    assert out.tolist() == ["YDL038C", "YDL039C"]
    assert out.nunique() == 2  # stay distinct records


def test_reconcile_non_gene_feature_remapped(genome):
    # YER108C is an alias of the blocked_reading_frame pseudogene YER109C (a valid locus).
    out = reconcile_systematic_names(genome, pd.Series(["YER108C"]), label="test")
    assert out.tolist() == ["YER109C"]
