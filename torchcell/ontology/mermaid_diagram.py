# torchcell/ontology/mermaid_diagram
# [[torchcell.ontology.mermaid_diagram]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/ontology/mermaid_diagram
# Test file: tests/torchcell/ontology/test_mermaid_diagram.py


"""Mermaid diagram generator for BioCypher schema.

Generates visual Mermaid diagrams from BioCypher schema YAML files.
Supports both horizontal (RL) and vertical (BT) orientations.
Ensures deterministic output for clean git diffs.
"""

import re
from pathlib import Path
from typing import Any

import yaml


class MermaidDiagramGenerator:
    """Generate Mermaid diagrams from BioCypher schema configuration."""

    def __init__(self, schema_config_path: str):
        """Load and parse BioCypher schema YAML.

        Args:
            schema_config_path: Path to torchcell_schema_config.yaml
        """
        self.schema_config_path = Path(schema_config_path)
        with open(self.schema_config_path) as f:
            self.schema = yaml.safe_load(f)

        # Parse schema into structured data
        self.nodes = self._extract_nodes()
        self.edges = self._extract_edges()
        self.biolink_classes = self._extract_biolink_classes()
        self.auto_mapped_nodes = self._get_auto_mapped_nodes()

    def _extract_nodes(self) -> dict[str, dict[str, Any]]:
        """Extract all nodes from schema.

        Returns:
            Dict mapping node names to their configurations
        """
        nodes = {}
        for name, config in self.schema.items():
            if config and config.get("represented_as") == "node":
                nodes[name] = config
        return nodes

    def _extract_edges(self) -> dict[str, dict[str, Any]]:
        """Extract all edges from schema.

        Returns:
            Dict mapping edge names to their configurations
        """
        edges = {}
        for name, config in self.schema.items():
            if config and config.get("represented_as") == "edge":
                edges[name] = config
        return edges

    def _extract_biolink_classes(self) -> set[str]:
        """Extract Biolink parent classes from nodes.

        Returns:
            Set of Biolink class names (e.g., 'activity', 'phenotypic feature')
        """
        classes = set()
        for node_config in self.nodes.values():
            if "is_a" in node_config:
                classes.add(node_config["is_a"])
        return classes

    def _get_auto_mapped_nodes(self) -> set[str]:
        """Get nodes that don't have is_a (direct Biolink class usage).

        Returns:
            Set of node names that are used directly from Biolink
        """
        auto_mapped = set()
        for name, config in self.nodes.items():
            if "is_a" not in config:
                auto_mapped.add(name)
        return auto_mapped

    def _format_node_id(self, name: str) -> str:
        """Convert node name to valid Mermaid ID.

        Args:
            name: Node name (e.g., 'fitness phenotype')

        Returns:
            CamelCase ID (e.g., 'FitnessPhenotype')
        """
        # Remove special characters and convert to CamelCase
        words = re.sub(r"[^\w\s]", "", name).split()
        return "".join(word.capitalize() for word in words)

    def _format_node_label(self, name: str) -> str:
        """Format node label for display.

        Args:
            name: Node name

        Returns:
            Formatted label
        """
        return name

    def generate_diagram(self, orientation: str = "LR") -> str:
        """Generate Mermaid diagram string.

        Args:
            orientation: 'RL' (right-to-left/horizontal) or 'BT' (bottom-to-top/vertical)

        Returns:
            Mermaid diagram as string
        """
        lines = [f"graph {orientation}"]

        # Collect nodes by type
        biolink_class_nodes = []
        auto_mapped_nodes_list = []
        inherited_entity_nodes = []

        # Add Biolink Class nodes (parent entity types that are inherited from)
        for concept in sorted(self.biolink_classes):
            node_id = self._format_node_id(concept)
            label = self._format_node_label(concept)
            biolink_class_nodes.append(f'    {node_id}["{label}"]')

        # Separate nodes into auto-mapped (direct Biolink) and inherited (torchcell-specific)
        for name in sorted(self.nodes.keys()):
            node_id = self._format_node_id(name)
            label = self._format_node_label(name)
            node_def = f'    {node_id}["{label}"]'

            if name in self.auto_mapped_nodes:
                auto_mapped_nodes_list.append(node_def)
            else:
                inherited_entity_nodes.append(node_def)

        # Add section comments and nodes
        if biolink_class_nodes:
            lines.append("")
            lines.append("    %% Biolink Classes (Parent Entity Types)")
            lines.extend(biolink_class_nodes)

        if auto_mapped_nodes_list:
            lines.append("")
            lines.append("    %% Direct Biolink Usage (No Inheritance)")
            lines.extend(auto_mapped_nodes_list)

        if inherited_entity_nodes:
            lines.append("")
            lines.append("    %% Torchcell Entities (Inherited from Biolink)")
            lines.extend(inherited_entity_nodes)

        # Add relationships
        class_inheritance = []
        data_relationships = []

        # Class inheritance: Biolink Class -> Torchcell Entity (solid arrows)
        for name, config in sorted(self.nodes.items()):
            if "is_a" in config:
                parent_id = self._format_node_id(config["is_a"])
                child_id = self._format_node_id(name)
                class_inheritance.append(f"    {parent_id} -->|is_a| {child_id}")

        # Data relationships: Entity --[relationship]--> Entity (dotted arrows)
        # Annotate with Biolink predicate inheritance in edge label
        for edge_name, edge_config in sorted(self.edges.items()):
            sources = edge_config.get("source", [])
            targets = edge_config.get("target", [])

            # Normalize to lists
            if isinstance(sources, str):
                sources = [sources]
            if isinstance(targets, str):
                targets = [targets]

            # Get Biolink predicate this relationship inherits from
            biolink_predicate = edge_config.get("is_a", "")

            # Create edge label with Biolink inheritance (use <br/> for line breaks in Mermaid)
            if biolink_predicate:
                edge_label = f"{edge_name}<br/>(is_a: {biolink_predicate})"
            else:
                edge_label = edge_name

            # Create edges from each source to each target
            for source in sources:
                source_id = self._format_node_id(source)
                for target in targets:
                    target_id = self._format_node_id(target)
                    # Use dotted line for data relationships
                    data_relationships.append(
                        f'    {source_id} -.->|"{edge_label}"| {target_id}'
                    )

        # Add relationships to diagram
        if class_inheritance:
            lines.append("")
            lines.append("    %% Class Inheritance")
            lines.extend(sorted(class_inheritance))

        if data_relationships:
            lines.append("")
            lines.append("    %% Data Relationships")
            lines.extend(sorted(data_relationships))

        # Add legend
        lines.append("")
        lines.append("    %% Legend")
        lines.append("    subgraph Legend")
        lines.append('        L1["Biolink Class"]')
        lines.append('        L2["Direct Biolink Usage"]')
        lines.append('        L3["Inherited Torchcell Entity"]')
        lines.append('        L4["→ solid = inheritance"]')
        lines.append('        L5["-.-> dotted = data relationship"]')
        lines.append("    end")

        # Add styling
        lines.append("")
        lines.append("    %% Styling")
        lines.append(
            "    classDef biolinkClassStyle fill:#e1f5ff,stroke:#0288d1,stroke-width:2px"
        )
        lines.append(
            "    classDef autoMappedStyle fill:#c8e6c9,stroke:#388e3c,stroke-width:2px"
        )
        lines.append(
            "    classDef torchcellEntityStyle fill:#fff3e0,stroke:#f57c00,stroke-width:2px"
        )

        # Apply styles
        if self.biolink_classes:
            biolink_class_ids = ",".join(
                self._format_node_id(c) for c in sorted(self.biolink_classes)
            )
            lines.append(f"    class {biolink_class_ids},L1 biolinkClassStyle")

        if self.auto_mapped_nodes:
            auto_mapped_ids = ",".join(
                self._format_node_id(n) for n in sorted(self.auto_mapped_nodes)
            )
            lines.append(f"    class {auto_mapped_ids},L2 autoMappedStyle")

        # Inherited entities (those with is_a)
        inherited_nodes = [
            n for n in self.nodes.keys() if n not in self.auto_mapped_nodes
        ]
        if inherited_nodes:
            inherited_ids = ",".join(
                self._format_node_id(n) for n in sorted(inherited_nodes)
            )
            lines.append(f"    class {inherited_ids},L3 torchcellEntityStyle")

        return "\n".join(lines)

    def _extract_frontmatter(self, content: str) -> tuple[str, str]:
        """Extract Dendron frontmatter from file content.

        Args:
            content: File content

        Returns:
            Tuple of (frontmatter, body) where frontmatter includes delimiters
        """
        # Match YAML frontmatter between --- delimiters
        match = re.match(r"^(---\n.*?\n---\n)(.*)", content, re.DOTALL)
        if match:
            return match.group(1), match.group(2)
        return "", content

    def _extract_mermaid_content(self, body: str) -> str:
        """Extract just the Mermaid diagram content from body.

        Args:
            body: File content without frontmatter

        Returns:
            Mermaid diagram content (everything after the first code fence)
        """
        # Find first mermaid code block
        match = re.search(r"```mermaid\n(.*?)```", body, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""

    def has_changed(self, new_content: str, file_path: str) -> bool:
        """Check if diagram content differs from existing file.

        Only compares the Mermaid diagram content, ignoring frontmatter.

        Args:
            new_content: New Mermaid diagram content (without code fences)
            file_path: Path to existing file

        Returns:
            True if content differs or file doesn't exist
        """
        path = Path(file_path)
        if not path.exists():
            return True

        with open(path) as f:
            existing_content = f.read()

        # Extract existing mermaid content
        _, body = self._extract_frontmatter(existing_content)
        existing_mermaid = self._extract_mermaid_content(body)

        return new_content.strip() != existing_mermaid.strip()

    def write_diagram(self, output_path: str, orientation: str) -> bool:
        """Write diagram to file, preserving Dendron frontmatter.

        Args:
            output_path: Output file path
            orientation: 'RL' or 'BT'

        Returns:
            True if file was updated, False if no changes
        """
        # Generate new diagram
        new_mermaid = self.generate_diagram(orientation)

        # Check if changed
        if not self.has_changed(new_mermaid, output_path):
            print(f"✓ No changes: {output_path}")
            return False

        path = Path(output_path)

        # Preserve existing frontmatter if file exists
        frontmatter = ""
        if path.exists():
            with open(path) as f:
                existing_content = f.read()
            frontmatter, _ = self._extract_frontmatter(existing_content)

        # If no frontmatter exists, create minimal one
        if not frontmatter:
            # Extract filename without extension for title
            title = path.stem.replace(".", " ").title()
            frontmatter = (
                "---\n"
                f"id: {path.stem.replace('.', '')}\n"
                f"title: {title}\n"
                "desc: 'BioCypher Schema Diagram'\n"
                "---\n\n"
            )

        # Construct full content
        full_content = f"{frontmatter}```mermaid\n{new_mermaid}\n```\n"

        # Write file
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(full_content)

        print(f"✓ Updated: {output_path}")
        return True


def main() -> None:
    """Generate both horizontal and vertical diagrams."""
    # Determine project root (assuming script is in torchcell/ontology/)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent

    schema_path = project_root / "biocypher" / "config" / "torchcell_schema_config.yaml"
    notes_dir = project_root / "notes"

    # Create generator
    generator = MermaidDiagramGenerator(str(schema_path))

    # Generate both orientations
    horizontal_path = notes_dir / "torchcell.ontology.mermaid_diagram.horizontal.md"
    vertical_path = notes_dir / "torchcell.ontology.mermaid_diagram.vertical.md"

    horizontal_updated = generator.write_diagram(str(horizontal_path), "RL")
    vertical_updated = generator.write_diagram(str(vertical_path), "BT")

    # Summary
    if horizontal_updated or vertical_updated:
        print("\n✓ Diagram generation complete")
    else:
        print("\n✓ No changes in ontology since last update")


if __name__ == "__main__":
    main()
