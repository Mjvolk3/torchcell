import json
import os
from collections import defaultdict
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import pydot
from adjustText import adjust_text
from networkx.drawing.nx_agraph import graphviz_layout
from networkx.drawing.nx_pydot import graphviz_layout


def analyze_structure(data, parent_key="") -> dict[str, Any]:
    type_structure = defaultdict(set)

    if isinstance(data, dict):
        for key, value in data.items():
            full_key = f"{parent_key}.{key}" if parent_key else key
            type_structure.update(analyze_structure(value, full_key))
    elif isinstance(data, list):
        for i, value in enumerate(data):
            if isinstance(value, (list, dict)):
                type_structure.update(analyze_structure(value, f"{parent_key}[{i}]"))
            else:
                type_structure[f"{parent_key}[{i}]"].add(type(value).__name__)
    else:
        type_structure[parent_key].add(type(data).__name__)

    return type_structure


def build_graph(data, graph=None):
    if graph is None:
        graph = nx.DiGraph()

    if isinstance(data, dict):
        for key, value in data.items():
            key_parts = key.split(".")
            for i in range(len(key_parts)):
                # Add node to graph if it doesn't exist already
                if key_parts[i] not in graph.nodes():
                    graph.add_node(key_parts[i], label=key_parts[i])

                # Add edges between consecutive nodes in the path
                if i > 0:  # Skip root node because it has no parent
                    graph.add_edge(key_parts[i - 1], key_parts[i])

                # Add edges between sibling nodes
                if i < len(key_parts) - 1:  # Skip leaf nodes
                    sibling_keys = [
                        sibling_key
                        for sibling_key in data.keys()
                        if sibling_key.startswith(".".join(key_parts[: i + 1]) + ".")
                        and sibling_key != key  # Exclude the current key
                    ]
                    for sibling_key in sibling_keys:
                        sibling_node = sibling_key.split(".")[i + 1]
                        if sibling_node not in graph.nodes():
                            graph.add_node(sibling_node, label=sibling_node)
                        graph.add_edge(key_parts[i], sibling_node)

            # The last node in the path (representing the type) is a leaf node
            leaf_node = str(value)  # Convert list to string using str function
            if leaf_node not in graph.nodes():
                graph.add_node(leaf_node, label=leaf_node)
            graph.add_edge(key_parts[-1], leaf_node)

    return graph


def calculate_depths(graph):
    if not graph.nodes:  # if the graph is empty
        return {}

    depths = {node: 0 for node in nx.nodes(graph)}  # Initialize depths to 0

    # Find the root of the graph (node with minimal in-degree)
    root = min(graph.nodes, key=graph.in_degree)

    # Perform breadth-first search starting from root
    queue = [(root, 0)]
    while queue:
        node, depth = queue.pop(0)
        depths[node] = depth
        queue.extend((child, depth + 1) for child in graph.successors(node))

    return depths


def draw_and_save_graph(graph, output_path, layout="spring"):
    plt.figure(figsize=(40, 30))  # Adjust as necessary

    if layout == "spring":
        pos = nx.spring_layout(graph, seed=42, center=[0.5, 0.5], k=1.0)  # k=1.5
    elif layout == "circular":
        pos = nx.nx_agraph.graphviz_layout(graph, prog="twopi")
    elif layout == "tree":
        pos = nx.nx_agraph.graphviz_layout(graph, prog="dot")

    depths = calculate_depths(graph)

    max_depth = max(depths.values()) if depths else 1  # Avoid division by zero
    node_list = list(graph.nodes())
    node_colors = [plt.cm.Greys(1.0 - depths[node] / max_depth) for node in node_list]

    # Overwrite leaf node colors with red
    leaf_nodes = [node for node, degree in graph.out_degree() if degree == 0]
    for node in leaf_nodes:
        node_colors[node_list.index(node)] = "red"

    nx.draw_networkx_nodes(
        graph, pos, node_color=node_colors, node_size=200
    )  # Adjust as necessary
    nx.draw_networkx_edges(graph, pos, edge_color="gray", alpha=0.6, arrows=True)

    node_labels = nx.get_node_attributes(graph, "label")
    texts = []
    for node, (x, y) in pos.items():
        texts.append(plt.text(x, y, node_labels[node], fontsize=16))

    if texts:  # Check if the list is not empty
        adjust_text(texts)

    plt.savefig(f"{output_path}.png", format="png", dpi=300)
    plt.savefig(f"{output_path}.pdf", format="pdf", dpi=300)
    plt.close()


def get_gene_name(file_path):
    return file_path.split("/")[-1].split(".")[0]


def process_src_to_image(path):
    split_path = path.split("/")
    index_of_src = split_path.index("src")
    post_src_path = split_path[index_of_src:]  # taking all parts after 'src'
    processed_path = "/".join(post_src_path)
    return processed_path


def analyze_json(file_path, create_images=False, layout="spring"):
    with open(file_path) as f:
        data = json.load(f)

    gene_name = get_gene_name(file_path)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = process_src_to_image(base_dir)

    for key in data.keys():
        type_structure = analyze_structure(data[key])
        reduced_type_structure = {
            f"{key}.{k}": list(types) for k, types in type_structure.items()
        }

        # Dynamically construct the subdirectories for JSON files and images
        subdirs = key.split("/")  # Assuming 'key' is a path like "validation/locus"

        json_dir = os.path.join("data", "sgd", "validation", *subdirs)
        os.makedirs(json_dir, exist_ok=True)

        with open(os.path.join(json_dir, f"{gene_name}.json"), "w") as f:
            json.dump(reduced_type_structure, f, indent=4)

        if create_images:
            graph = build_graph(reduced_type_structure)  # Remove 'key' here

            image_dir = os.path.join("notes", "assets", "images", image_path, *subdirs)
            os.makedirs(image_dir, exist_ok=True)

            output_path = os.path.join(image_dir, f"{gene_name}_{key}_type_graph")
            draw_and_save_graph(graph, output_path, layout)


def main():
    import os

    file_path = "data/sgd/genes/YPR201W.json"
    analyze_json(file_path, create_images=True, layout="tree")


if __name__ == "__main__":
    main()
