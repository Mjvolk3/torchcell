# torchcell/datasets/pronto_ontology.py
# [[torchcell.datasets.pronto_ontology]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/pronto_ontology.py
# Test file: torchcell/datasets/test_pronto_ontology.py

from collections import Counter

import matplotlib.pyplot as plt
import networkx as nx

# # Draw the graph
# pos = nx.spring_layout(G)
# labels = nx.get_node_attributes(G, 'label')
# nx.draw(G, pos, labels=labels, with_labels=True, node_size=3000, node_color='skyblue', font_size=10, font_weight='bold')
# plt.show()
import pronto
from nxontology.imports import multidigraph_to_digraph, pronto_to_multidigraph
from nxontology.viz import create_similarity_graphviz
from pronto import Definition, Ontology

# # https://obofoundry.org/ontology/mco.html
# url = "https://raw.githubusercontent.com/microbial-conditions-ontology/microbial-conditions-ontology/master/mco.owl"
# env_pronto = Ontology(handle=url)
# print(env_pronto.__dict__)
# print(50 * "=")
# print(len(env_pronto.terms()))
# [print(i) for i in list(env_pronto.terms())]
# print(50 * "=")
# [print(i) for i in list(env_pronto.relationships())]
# print(len(env_pronto.relationships()))

# # if __name__ == "__main__":
# #     Plotting the graph
# #     env_multidigraph = pronto_to_multidigraph(env_pronto)
# #     pos = nx.spring_layout(env_multidigraph)
# #     plt.figure(figsize=(10, 10))
# #     nx.draw(
# #         env_multidigraph, pos, with_labels=True, node_size=1000, node_color="skyblue"
# #     )
# #     plt.title("env_multidigraph")
# #     plt.show()
# #     pass

###############


# # Create a directed graph
# G = nx.DiGraph()

# # Add nodes for each term in the ontology
# for term in ontology:
#     G.add_node(term.id, label=term.name)

# # Add edges for each relationship in the ontology
# for term in ontology:
#     for rel, parent in term.relationships.items():
#         G.add_edge(term.id, parent.id, label=rel.id)


if __name__ == "__main__":
    # Create a new ontology
    ontology = pronto.Ontology()

    # Add terms to the ontology
    term1 = ontology.create_term("GO:0000001")
    term1.name = "mitochondrion inheritance"
    term1.definition = Definition(
        "The distribution of mitochondria, including the mitochondrial genome, into daughter cells after mitosis or meiosis, mediated by interactions between mitochondria and the cytoskeleton.",
        [],
    )

    term2 = ontology.create_term("GO:0000002")
    term2.name = "mitochondrial genome maintenance"
    term2.definition = Definition(
        "The maintenance of the structure and integrity of the mitochondrial genome; includes replication and segregation of the mitochondrial chromosome.",
        [],
    )

    # Use the predefined "is_a" relationship
    relationship = ontology.get_relationship("is_a")
    term2.relationships[relationship] = {term1}

    print(ontology)

    # Plotting the graph
    onto_multidigraph = pronto_to_multidigraph(ontology)
    pos = nx.spring_layout(onto_multidigraph)
    plt.figure(figsize=(10, 10))
    nx.draw(
        onto_multidigraph, pos, with_labels=True, node_size=1000, node_color="skyblue"
    )
    plt.title("onto_multidigraph")
    plt.show()
