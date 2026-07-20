"""Check whether a GO term is obsolete (deprecated) in the loaded GO ontology DAG."""

from goatools.obo_parser import GODag

# Initialize the GO ontology DAG (Directed Acyclic Graph)
go_dag = GODag("data/go/go.obo")

# GO Term to be checked
go_term = "GO:0000185"

# Check if the term exists in the ontology
if go_term in go_dag:
    # Check if it is obsolete
    if go_dag[go_term].is_obsolete:
        print(f"{go_term} is deprecated.")
    else:
        print(f"{go_term} is not deprecated.")
else:
    print(f"{go_term} does not exist in the loaded ontology.")

print()
