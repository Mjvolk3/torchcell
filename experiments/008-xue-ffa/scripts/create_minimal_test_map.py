#!/usr/bin/env python3
"""
Create a minimal test Escher map with just one reaction to debug the issue.
"""

import json
import os
from pathlib import Path
from torchcell.timestamp import timestamp

RESULTS_DIR = Path("/Users/michaelvolk/Documents/projects/torchcell/experiments/008-xue-ffa/results")


def create_minimal_test_map():
    """Create a minimal valid Escher map with one simple reaction."""

    print("Creating minimal test map...")

    # Metadata
    metadata = {
        "map_name": "Minimal Test Map",
        "map_id": "minimal_test",
        "map_description": "A minimal test map with one reaction",
        "homepage": "https://github.com/Mjvolk3/torchcell",
        "schema": "https://escher.github.io/escher/jsonschema/1-0-0#"
    }

    # Map data with a single reaction: A + B -> C
    map_data = {
        "nodes": {
            # Metabolite A
            "1": {
                "node_type": "metabolite",
                "x": 100.0,
                "y": 100.0,
                "bigg_id": "met_a",
                "name": "Metabolite A",
                "label_x": 100.0,
                "label_y": 100.0,
                "node_is_primary": True
            },
            # Metabolite B
            "2": {
                "node_type": "metabolite",
                "x": 100.0,
                "y": 200.0,
                "bigg_id": "met_b",
                "name": "Metabolite B",
                "label_x": 100.0,
                "label_y": 200.0,
                "node_is_primary": True
            },
            # Metabolite C
            "3": {
                "node_type": "metabolite",
                "x": 400.0,
                "y": 150.0,
                "bigg_id": "met_c",
                "name": "Metabolite C",
                "label_x": 400.0,
                "label_y": 150.0,
                "node_is_primary": True
            },
            # Multimarker for reactants
            "4": {
                "node_type": "multimarker",
                "x": 200.0,
                "y": 150.0
            },
            # Midmarker for products
            "5": {
                "node_type": "midmarker",
                "x": 300.0,
                "y": 150.0
            }
        },
        "reactions": {
            "test_rxn": {
                "name": "Test Reaction",
                "bigg_id": "test_rxn",
                "reversibility": False,
                "label_x": 250.0,
                "label_y": 150.0,
                "gene_reaction_rule": "",
                "genes": [],
                "metabolites": [
                    {
                        "bigg_id": "met_a",
                        "coefficient": -1.0
                    },
                    {
                        "bigg_id": "met_b",
                        "coefficient": -1.0
                    },
                    {
                        "bigg_id": "met_c",
                        "coefficient": 1.0
                    }
                ],
                "segments": {
                    "1": {
                        "to_node_id": "4",
                        "from_node_id": "1",
                        "b2": None,
                        "b1": None
                    },
                    "2": {
                        "to_node_id": "4",
                        "from_node_id": "2",
                        "b2": None,
                        "b1": None
                    },
                    "3": {
                        "to_node_id": "5",
                        "from_node_id": "4",
                        "b2": None,
                        "b1": None
                    },
                    "4": {
                        "to_node_id": "3",
                        "from_node_id": "5",
                        "b2": None,
                        "b1": None
                    }
                }
            }
        },
        "text_labels": {},
        "canvas": {
            "x": 0.0,
            "y": 0.0,
            "width": 500.0,
            "height": 300.0
        }
    }

    # Create the map array
    escher_map = [metadata, map_data]

    # Save the map
    map_path = RESULTS_DIR / "minimal_test_map.json"

    with open(map_path, 'w') as f:
        json.dump(escher_map, f, indent=2)

    print(f"\nSaved minimal test map to:")
    print(f"  {map_path}")
    print("\nThis map contains:")
    print("  - 3 metabolite nodes (A, B, C)")
    print("  - 1 multimarker node")
    print("  - 1 midmarker node")
    print("  - 1 reaction (A + B -> C)")
    print("  - 4 segments connecting everything")
    print("\nTry loading this in Escher to see if even a simple map works.")

    return map_path


if __name__ == "__main__":
    create_minimal_test_map()