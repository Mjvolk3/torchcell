# torchcell/multidigraph/metabolism.py
# [[torchcell.multidigraph.metabolism]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/multidigraph/metabolism.py
# Test file: torchcell/multidigraph/test_metabolism.py

import networkx as nx
import pandas as pd

# 1. Read the Excel file
file_path = "data/sgd/genome/yeast-GEM.xlsx"
data = pd.read_excel(file_path)

# 2. Parse the dataframe
G = nx.DiGraph()  # Initialize a directed graph

for index, row in data.iterrows():
    equation = row["EQUATION"]
    if "<=>" in equation:  # bidirectional
        reactants, products = equation.split("<=>")
    elif "=>" in equation:  # one-directional
        reactants, products = equation.split("=>")
    else:
        continue

    # Attributes for the reaction node
    attributes = {
        "EC": row["EC-NUMBER"],
        "Gene Association": row["GENE ASSOCIATION"],
        "MIRIAM": row["MIRIAM"],
        "Subsystem": row["SUBSYSTEM"],
    }

    reaction_node = row["ID"]  # Node name based on the "ID" column
    G.add_node(reaction_node, **attributes)  # Add reaction node with attributes

    # Connect reactants to the reaction node
    for reactant in reactants.split("+"):
        G.add_edge(
            reactant.strip(), reaction_node, stoichiometric_coefficient=1
        )  # Assuming a default stoichiometric coefficient of 1

    # Connect the reaction node to products
    for product in products.split("+"):
        G.add_edge(
            reaction_node, product.strip(), stoichiometric_coefficient=1
        )  # Assuming a default stoichiometric coefficient of 1

print()
# If you want to visualize the graph
# import matplotlib.pyplot as plt
# nx.draw(G, with_labels=True)
# plt.show()
