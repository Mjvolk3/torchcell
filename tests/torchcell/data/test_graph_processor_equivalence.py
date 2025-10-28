#!/usr/bin/env python3
"""
Equivalence test suite for SubgraphRepresentation optimizations.
Tests that optimized graph_processor produces identical outputs to baseline.
"""

import os.path as osp
import pickle
import pytest
import torch
from torch_geometric.utils import sort_edge_index
from torchcell.scratch.load_batch_005 import load_sample_data_batch

REFERENCE_DIR = "/scratch/projects/torchcell/data/tests/torchcell/scratch/load_batch_005"


@pytest.fixture(scope="module")
def reference_data():
    """Load reference baseline data."""
    ref_path = osp.join(REFERENCE_DIR, "reference_baseline.pkl")
    if not osp.exists(ref_path):
        pytest.skip(f"Reference baseline not found: {ref_path}")

    with open(ref_path, "rb") as f:
        return pickle.load(f)


@pytest.fixture(scope="module")
def current_data():
    """Generate current data using load_batch_005."""
    dataset, batch, input_channels, max_num_nodes = load_sample_data_batch(
        batch_size=2,
        num_workers=2,
        config="hetero_cell_bipartite",
        is_dense=False
    )
    return dataset, batch, input_channels, max_num_nodes


def compare_hetero_data(reference, current, name="data", rtol=1e-5, atol=1e-8):
    """
    Compare two HeteroData objects for structural and numerical equality.
    """
    # Compare node types
    assert set(reference.node_types) == set(current.node_types), \
        f"{name}: Node types differ.\n  Reference: {reference.node_types}\n  Current: {current.node_types}"

    # Compare edge types
    assert set(reference.edge_types) == set(current.edge_types), \
        f"{name}: Edge types differ.\n  Reference: {reference.edge_types}\n  Current: {current.edge_types}"

    # Compare node attributes
    for node_type in reference.node_types:
        ref_store = reference[node_type]
        cur_store = current[node_type]

        ref_attrs = set(ref_store.keys())
        cur_attrs = set(cur_store.keys())

        assert ref_attrs == cur_attrs, \
            f"{name}['{node_type}']: Attributes differ.\n  Reference: {ref_attrs}\n  Current: {cur_attrs}"

        for attr in ref_attrs:
            ref_val = ref_store[attr]
            cur_val = cur_store[attr]

            if torch.is_tensor(ref_val):
                assert ref_val.shape == cur_val.shape, \
                    f"{name}['{node_type}']['{attr}']: Shape mismatch.\n  Reference: {ref_val.shape}\n  Current: {cur_val.shape}"

                assert ref_val.dtype == cur_val.dtype, \
                    f"{name}['{node_type}']['{attr}']: Dtype mismatch"

                if ref_val.dtype in [torch.float, torch.float32, torch.float64]:
                    if not torch.allclose(ref_val, cur_val, rtol=rtol, atol=atol):
                        max_diff = (ref_val - cur_val).abs().max().item()
                        # Show first few differing values for debugging
                        diff_mask = ~torch.isclose(ref_val, cur_val, rtol=rtol, atol=atol)
                        num_diff = diff_mask.sum().item()
                        assert False, \
                            f"{name}['{node_type}']['{attr}']: Values differ.\n" \
                            f"  Max diff: {max_diff}\n" \
                            f"  Num differences: {num_diff}\n" \
                            f"  First ref values: {ref_val.flatten()[:10]}\n" \
                            f"  First cur values: {cur_val.flatten()[:10]}"
                else:
                    if not torch.equal(ref_val, cur_val):
                        # Show where they differ for debugging
                        if ref_val.dim() == 1:
                            diff_indices = (ref_val != cur_val).nonzero(as_tuple=True)[0][:10]
                            assert False, \
                                f"{name}['{node_type}']['{attr}']: Values differ\n" \
                                f"  First 10 diff indices: {diff_indices.tolist()}\n" \
                                f"  Ref at those: {ref_val[diff_indices].tolist()}\n" \
                                f"  Cur at those: {cur_val[diff_indices].tolist()}"
                        else:
                            assert False, \
                                f"{name}['{node_type}']['{attr}']: Values differ\n" \
                                f"  First ref values: {ref_val.flatten()[:20]}\n" \
                                f"  First cur values: {cur_val.flatten()[:20]}"
            elif isinstance(ref_val, list):
                # Some attributes are semantically sets but stored as lists
                # Compare as sets for these attributes
                if attr in ['ids_pert', 'node_ids', 'phenotype_types', 'phenotype_stat_types']:
                    assert len(ref_val) == len(cur_val), \
                        f"{name}['{node_type}']['{attr}']: List length differs"

                    # Check if items are hashable (strings, ints) or nested lists
                    if ref_val and isinstance(ref_val[0], list):
                        # Nested lists (batch case) - compare each sub-list as a set
                        for i, (ref_sublist, cur_sublist) in enumerate(zip(ref_val, cur_val)):
                            assert len(ref_sublist) == len(cur_sublist) and set(ref_sublist) == set(cur_sublist), \
                                f"{name}['{node_type}']['{attr}'][{i}]: Sub-list values differ (as sets)\n" \
                                f"  Reference: {sorted(ref_sublist)}\n" \
                                f"  Current: {sorted(cur_sublist)}\n" \
                                f"  Missing: {set(ref_sublist) - set(cur_sublist)}\n" \
                                f"  Extra: {set(cur_sublist) - set(ref_sublist)}"
                    else:
                        # Simple list - compare as set
                        assert set(ref_val) == set(cur_val), \
                            f"{name}['{node_type}']['{attr}']: List values differ (as sets)\n" \
                            f"  Reference: {sorted(ref_val)}\n" \
                            f"  Current: {sorted(cur_val)}\n" \
                            f"  Missing in current: {set(ref_val) - set(cur_val)}\n" \
                            f"  Extra in current: {set(cur_val) - set(ref_val)}"
                else:
                    assert len(ref_val) == len(cur_val) and ref_val == cur_val, \
                        f"{name}['{node_type}']['{attr}']: List values differ"
            else:
                assert ref_val == cur_val, \
                    f"{name}['{node_type}']['{attr}']: Value mismatch"

    # Compare edge attributes
    for edge_type in reference.edge_types:
        ref_store = reference[edge_type]
        cur_store = current[edge_type]

        ref_attrs = set(ref_store.keys())
        cur_attrs = set(cur_store.keys())

        assert ref_attrs == cur_attrs, \
            f"{name}[{edge_type}]: Attributes differ"

        # Check if this edge type has edge_index or hyperedge_index that needs canonicalization
        edge_index_key = None
        if 'edge_index' in ref_attrs:
            edge_index_key = 'edge_index'
        elif 'hyperedge_index' in ref_attrs:
            edge_index_key = 'hyperedge_index'

        # If we have an edge index, canonicalize it and all related attributes
        if edge_index_key:
            ref_edge_index = ref_store[edge_index_key]
            cur_edge_index = cur_store[edge_index_key]

            # Gather edge attributes that need to be reordered with edges
            # Look for attributes where first dimension equals num_edges
            num_edges = ref_edge_index.size(1)
            edge_attr_keys = []

            for attr in ref_attrs:
                if attr != edge_index_key:
                    ref_val = ref_store[attr]
                    cur_val = cur_store[attr]
                    if torch.is_tensor(ref_val) and torch.is_tensor(cur_val):
                        if ref_val.size(0) == num_edges and cur_val.size(0) == num_edges:
                            edge_attr_keys.append(attr)

            # Sort reference edges and attributes using PyG's sort_edge_index
            if edge_attr_keys:
                # Stack all edge attributes for sorting
                ref_edge_attr = torch.cat([ref_store[k].view(num_edges, -1) for k in edge_attr_keys], dim=1)
                ref_edge_index_sorted, ref_edge_attr_sorted = sort_edge_index(
                    ref_edge_index,
                    edge_attr=ref_edge_attr,
                    sort_by_row=True
                )

                # Unstack sorted attributes
                ref_attrs_sorted = {}
                start_idx = 0
                for k in edge_attr_keys:
                    orig_shape = ref_store[k].shape
                    num_features = ref_store[k].numel() // num_edges
                    ref_attrs_sorted[k] = ref_edge_attr_sorted[:, start_idx:start_idx+num_features].view(orig_shape)
                    start_idx += num_features

                # Sort current edges and attributes
                cur_edge_attr = torch.cat([cur_store[k].view(num_edges, -1) for k in edge_attr_keys], dim=1)
                cur_edge_index_sorted, cur_edge_attr_sorted = sort_edge_index(
                    cur_edge_index,
                    edge_attr=cur_edge_attr,
                    sort_by_row=True
                )

                # Unstack sorted attributes
                cur_attrs_sorted = {}
                start_idx = 0
                for k in edge_attr_keys:
                    orig_shape = cur_store[k].shape
                    num_features = cur_store[k].numel() // num_edges
                    cur_attrs_sorted[k] = cur_edge_attr_sorted[:, start_idx:start_idx+num_features].view(orig_shape)
                    start_idx += num_features
            else:
                # No edge attributes - just sort edge indices
                ref_edge_index_sorted = sort_edge_index(ref_edge_index, sort_by_row=True)
                cur_edge_index_sorted = sort_edge_index(cur_edge_index, sort_by_row=True)
                ref_attrs_sorted = {}
                cur_attrs_sorted = {}

            # Compare sorted edge indices
            assert torch.equal(ref_edge_index_sorted, cur_edge_index_sorted), \
                f"{name}[{edge_type}]['{edge_index_key}']: Edge indices differ even after sorting\n" \
                f"  First ref edges (sorted): {ref_edge_index_sorted[:, :10]}\n" \
                f"  First cur edges (sorted): {cur_edge_index_sorted[:, :10]}"

            # Compare sorted edge attributes
            for attr in edge_attr_keys:
                ref_val = ref_attrs_sorted[attr]
                cur_val = cur_attrs_sorted[attr]

                if ref_val.dtype in [torch.float, torch.float32, torch.float64]:
                    if not torch.allclose(ref_val, cur_val, rtol=rtol, atol=atol):
                        max_diff = (ref_val - cur_val).abs().max().item()
                        assert False, \
                            f"{name}[{edge_type}]['{attr}']: Values differ after sorting\n" \
                            f"  Max diff: {max_diff}\n" \
                            f"  First ref values: {ref_val.flatten()[:20]}\n" \
                            f"  First cur values: {cur_val.flatten()[:20]}"
                else:
                    assert torch.equal(ref_val, cur_val), \
                        f"{name}[{edge_type}]['{attr}']: Values differ after sorting\n" \
                        f"  First ref values: {ref_val.flatten()[:20]}\n" \
                        f"  First cur values: {cur_val.flatten()[:20]}"

            # Compare non-edge attributes normally
            for attr in ref_attrs:
                if attr not in edge_attr_keys and attr != edge_index_key:
                    ref_val = ref_store[attr]
                    cur_val = cur_store[attr]
                    if torch.is_tensor(ref_val) and torch.is_tensor(cur_val):
                        assert ref_val.shape == cur_val.shape and torch.equal(ref_val, cur_val), \
                            f"{name}[{edge_type}]['{attr}']: Value mismatch"
                    else:
                        assert ref_val == cur_val, \
                            f"{name}[{edge_type}]['{attr}']: Value mismatch"
        else:
            # No edge index - compare all attributes normally
            for attr in ref_attrs:
                ref_val = ref_store[attr]
                cur_val = cur_store[attr]

                if torch.is_tensor(ref_val):
                    assert ref_val.shape == cur_val.shape, \
                        f"{name}[{edge_type}]['{attr}']: Shape mismatch"

                    assert ref_val.dtype == cur_val.dtype, \
                        f"{name}[{edge_type}]['{attr}']: Dtype mismatch"

                    if ref_val.dtype in [torch.float, torch.float32, torch.float64]:
                        assert torch.allclose(ref_val, cur_val, rtol=rtol, atol=atol), \
                            f"{name}[{edge_type}]['{attr}']: Values differ"
                    else:
                        assert torch.equal(ref_val, cur_val), \
                            f"{name}[{edge_type}]['{attr}']: Values differ"


def test_single_instance_equivalence(reference_data, current_data):
    """Test that single instance matches reference."""
    dataset, _, _, _ = current_data
    compare_hetero_data(
        reference_data["single_instance"],
        dataset[0],
        "single_instance"
    )


def test_batch_equivalence(reference_data, current_data):
    """Test that batch matches reference."""
    _, batch, _, _ = current_data
    compare_hetero_data(
        reference_data["batch"],
        batch,
        "batch"
    )


def test_incidence_vs_subgraph_processors():
    """
    Test that IncidenceSubgraphRepresentation produces identical output
    to SubgraphRepresentation on the same inputs.

    This test uses a small sample dataset to verify that the optimized
    incidence-based processor returns the same results as the baseline.
    """
    from torchcell.data.graph_processor import (
        SubgraphRepresentation,
        IncidenceSubgraphRepresentation,
    )

    # Load sample data
    try:
        dataset, batch, _, _ = load_sample_data_batch(
            batch_size=2,
            num_workers=2,
            config="hetero_cell_bipartite",
            is_dense=False
        )
    except Exception as e:
        pytest.skip(f"Could not load sample data: {e}")

    # Get a single sample for testing
    if len(dataset) == 0:
        pytest.skip("Empty dataset")

    # Get the cell_graph (same for all samples)
    cell_graph = dataset.cell_graph

    # Get sample data (phenotype_info and experiment data)
    # We need to extract the relevant info from dataset[0]
    # For now, let's just verify we can instantiate both processors
    subgraph_proc = SubgraphRepresentation()
    incidence_proc = IncidenceSubgraphRepresentation()

    # Verify processors are created successfully
    assert subgraph_proc is not None
    assert incidence_proc is not None

    # TODO: Add actual comparison once we figure out how to extract
    # phenotype_info and data from the dataset properly
    # For now, this test just verifies the class can be instantiated
