# torchcell/ontology/tc_ontology.py
# [[torchcell.ontology.tc_ontology]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/ontology/tc_ontology.py
# Test file: torchcell/ontology/test_tc_ontology.py
"""Build and inspect the torchcell BioCypher ontology schema."""

from pathlib import Path

import yaml

from biocypher import BioCypher  # type: ignore


def print_ontology_structure(
    schema_config_path: str = "biocypher/config/torchcell_schema_config.yaml",
    full: bool = False,
    to_disk: str | None = None,
) -> None:
    """Print the BioCypher ontology structure from schema config.

    Args:
        schema_config_path: Path to the schema config YAML file relative to project root
        full: If True, show complete ontology; if False, show only schema-relevant parts
        to_disk: Optional path to save GraphML file for external visualization tools

    Example:
        >>> from torchcell.ontology import print_ontology_structure
        >>> # Print focused view of your schema
        >>> print_ontology_structure()
        >>> # Print complete ontology
        >>> print_ontology_structure(full=True)
        >>> # Export for visualization in Cytoscape/Gephi
        >>> print_ontology_structure(to_disk="path/to/output")
    """
    bc = BioCypher(offline=True, schema_config_path=schema_config_path)

    bc.show_ontology_structure(full=full, to_disk=to_disk)


def print_ontology_summary(
    schema_config_path: str = "biocypher/config/torchcell_schema_config.yaml",
) -> None:
    """Print ontology summary with validation checks.

    Displays the ontology structure while checking for duplicates and unmapped
    input labels. Useful for debugging schema configurations.

    Args:
        schema_config_path: Path to the schema config YAML file relative to project root

    Example:
        >>> from torchcell.ontology import print_ontology_summary
        >>> print_ontology_summary()
    """
    bc = BioCypher(offline=True, schema_config_path=schema_config_path)
    bc.summary()


def print_schema_mappings(
    schema_config_path: str = "biocypher/config/torchcell_schema_config.yaml",
    compact: bool = True,
) -> None:
    """Print a simple visualization of schema entities and their Biolink mappings.

    Shows how torchcell schema entities map to Biolink ontology concepts via
    'is_a' relationships, without fetching the full ontology.

    Args:
        schema_config_path: Path to the schema config YAML file relative to project root
        compact: If True, use compact format; if False, use expanded tree format

    Example:
        >>> from torchcell.ontology import print_schema_mappings
        >>> print_schema_mappings()
        >>> print_schema_mappings(compact=False)  # Expanded format
    """
    # Load the YAML config
    config_path = Path(schema_config_path)
    if not config_path.exists():
        print(f"Error: Schema config not found at {schema_config_path}")
        return

    with open(config_path) as f:
        schema = yaml.safe_load(f)

    # Separate nodes and edges
    nodes = {}
    edges = {}

    for entity_name, entity_config in schema.items():
        if not isinstance(entity_config, dict):
            continue

        represented_as = entity_config.get("represented_as", "unknown")
        is_a = entity_config.get("is_a", None)

        if represented_as == "node":
            nodes[entity_name] = is_a
        elif represented_as == "edge":
            edges[entity_name] = is_a

    # Group nodes and edges by their Biolink parent
    # Also detect auto-mappings (when entity name matches Biolink class name)
    biolink_groups: dict[str, list[str]] = {}
    auto_mapped = []
    no_mapping = []

    for node_name, biolink_parent in nodes.items():
        if biolink_parent:
            if biolink_parent not in biolink_groups:
                biolink_groups[biolink_parent] = []
            biolink_groups[biolink_parent].append(node_name)
        else:
            # Check if entity name matches a likely Biolink class (auto-mapping)
            # Common Biolink classes that torchcell might use
            likely_biolink_classes = [
                "dataset",
                "genome",
                "genotype",
                "publication",
                "gene",
                "protein",
                "disease",
                "phenotype",
            ]
            if node_name.lower() in likely_biolink_classes:
                auto_mapped.append(node_name)
            else:
                no_mapping.append(node_name)

    biolink_edge_groups: dict[str, list[str]] = {}
    no_edge_mapping = []
    for edge_name, biolink_parent in edges.items():
        if biolink_parent:
            if biolink_parent not in biolink_edge_groups:
                biolink_edge_groups[biolink_parent] = []
            biolink_edge_groups[biolink_parent].append(edge_name)
        else:
            no_edge_mapping.append(edge_name)

    if compact:
        # Compact format
        print("\n" + "═" * 80)
        print("TORCHCELL SCHEMA → BIOLINK MAPPINGS")
        print("═" * 80)

        print("\n📦 NODES (16 total)")
        for biolink_parent in sorted(biolink_groups.keys()):
            entities = ", ".join(sorted(biolink_groups[biolink_parent]))
            print(f"  {biolink_parent:25} → {entities}")

        if auto_mapped:
            entities = ", ".join(sorted(auto_mapped))
            print(f"  {'✓ auto-mapped':25} → {entities}")

        if no_mapping:
            entities = ", ".join(sorted(no_mapping))
            print(f"  {'⚠️  unmapped':25} → {entities}")

        print("\n🔗 EDGES (11 total)")
        for biolink_parent in sorted(biolink_edge_groups.keys()):
            edge_names = sorted(biolink_edge_groups[biolink_parent])
            edges_str = ", ".join(edge_names)
            print(f"  {biolink_parent:25} → {edges_str}")

        if no_edge_mapping:
            edges_str = ", ".join(sorted(no_edge_mapping))
            print(f"  {'⚠️  unmapped':25} → {edges_str}")

        # Enhanced summary
        total_biolink_concepts = len(biolink_groups) + len(biolink_edge_groups)
        mapped_nodes = sum(len(v) for v in biolink_groups.values())
        mapped_edges = sum(len(v) for v in biolink_edge_groups.values())
        total_nodes_covered = mapped_nodes + len(auto_mapped)

        print("\n" + "─" * 80)
        print("📊 SUMMARY")
        print(
            f"  Nodes:    {mapped_nodes}/{len(nodes)} explicit + {len(auto_mapped)} auto-mapped = {total_nodes_covered}/{len(nodes)} total"
        )
        print(
            f"  Edges:    {mapped_edges}/{len(edges)} mapped to {len(biolink_edge_groups)} Biolink concepts"
        )
        print(f"  Total:    {total_biolink_concepts} unique Biolink concepts used")

        # Show unmapped entities if any
        if no_mapping or no_edge_mapping:
            print(
                f"  ⚠️  Warning: {len(no_mapping)} unmapped nodes, {len(no_edge_mapping)} unmapped edges"
            )
        elif auto_mapped:
            print(f"  ✓ {len(auto_mapped)} nodes auto-mapped by name matching")

        # List all Biolink concepts used (one per line)
        all_concepts = sorted(
            list(biolink_groups.keys()) + list(biolink_edge_groups.keys())
        )
        print(f"\n  Biolink concepts ({len(all_concepts)}):")
        for concept in all_concepts:
            print(f"    • {concept}")
        print("═" * 80 + "\n")

    else:
        # Expanded tree format (original)
        print("\n" + "=" * 80)
        print("TORCHCELL SCHEMA → BIOLINK ONTOLOGY MAPPINGS")
        print("=" * 80)

        print("\n📦 NODES")
        print("-" * 80)

        for biolink_parent in sorted(biolink_groups.keys()):
            print(f"\n  🔗 is_a: {biolink_parent}")
            for node_name in sorted(biolink_groups[biolink_parent]):
                print(f"      └─ {node_name}")

        if auto_mapped:
            print("\n  ✓ Auto-mapped (name matches Biolink class):")
            for node_name in sorted(auto_mapped):
                print(f"      └─ {node_name}")

        if no_mapping:
            print("\n  ⚠️  No Biolink mapping:")
            for node_name in sorted(no_mapping):
                print(f"      └─ {node_name}")

        print("\n\n🔗 EDGES (RELATIONSHIPS)")
        print("-" * 80)

        for biolink_parent in sorted(biolink_edge_groups.keys()):
            print(f"\n  🔗 is_a: {biolink_parent}")
            for edge_name in sorted(biolink_edge_groups[biolink_parent]):
                edge_config = schema[edge_name]
                source = edge_config.get("source", "?")
                target = edge_config.get("target", "?")
                print(f"      └─ {edge_name}: {source} → {target}")

        if no_edge_mapping:
            print("\n  ⚠️  No Biolink mapping:")
            for edge_name in sorted(no_edge_mapping):
                edge_config = schema[edge_name]
                source = edge_config.get("source", "?")
                target = edge_config.get("target", "?")
                print(f"      └─ {edge_name}: {source} → {target}")

        # Enhanced summary for expanded format
        total_biolink_concepts = len(biolink_groups) + len(biolink_edge_groups)
        mapped_nodes = sum(len(v) for v in biolink_groups.values())
        mapped_edges = sum(len(v) for v in biolink_edge_groups.values())
        total_nodes_covered = mapped_nodes + len(auto_mapped)

        print("\n" + "=" * 80)
        print("📊 SUMMARY")
        print(
            f"  Nodes:    {mapped_nodes}/{len(nodes)} explicit + {len(auto_mapped)} auto-mapped = {total_nodes_covered}/{len(nodes)} total"
        )
        print(
            f"  Edges:    {mapped_edges}/{len(edges)} mapped to {len(biolink_edge_groups)} Biolink concepts"
        )
        print(f"  Total:    {total_biolink_concepts} unique Biolink concepts used")

        if no_mapping or no_edge_mapping:
            print(
                f"  ⚠️  Warning: {len(no_mapping)} unmapped nodes, {len(no_edge_mapping)} unmapped edges"
            )
        elif auto_mapped:
            print(f"  ✓ {len(auto_mapped)} nodes auto-mapped by name matching")

        # List all Biolink concepts used (one per line)
        all_concepts = sorted(
            list(biolink_groups.keys()) + list(biolink_edge_groups.keys())
        )
        print(f"\n  Biolink concepts ({len(all_concepts)}):")
        for concept in all_concepts:
            print(f"    • {concept}")
        print("=" * 80 + "\n")


def main() -> None:
    """CLI entry point for tc-onto command."""
    import argparse

    parser = argparse.ArgumentParser(
        description="View torchcell schema → Biolink ontology mappings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--expand",
        "-e",
        action="store_true",
        help="Show expanded tree format instead of compact format",
    )
    parser.add_argument(
        "--config",
        "-c",
        default="biocypher/config/torchcell_schema_config.yaml",
        help="Path to schema config YAML (default: biocypher/config/torchcell_schema_config.yaml)",
    )

    args = parser.parse_args()

    print_schema_mappings(schema_config_path=args.config, compact=not args.expand)


if __name__ == "__main__":
    main()
