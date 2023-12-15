# torchcell/datasets/ontology.py
# [[torchcell.datasets.ontology]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/ontology.py
# Test file: torchcell/datasets/test_ontology.py
import json

from owlready2 import (
    DataProperty,
    FunctionalProperty,
    ObjectProperty,
    Thing,
    get_ontology,
)

# Create a new ontology
# currently only have rdf
onto = get_ontology(
    "https://raw.githubusercontent.com/Mjvolk3/torchcell/main/torchcell.rdf"
)
from neo4j import GraphDatabase


# Define the top-level Experiment class
class Experiment(Thing):
    namespace = onto


# Define Genotype, Phenotype, and Environment as subclasses of Experiment
class Genotype(Experiment):
    namespace = onto


class Phenotype(Experiment):
    namespace = onto


class Environment(Experiment):
    namespace = onto


# Define properties for Genotype


class ReferenceGenome(Genotype):
    namespace = onto
    domain = [Genotype]
    range = [str]


class SysGeneName(Genotype):
    namespace = onto
    domain = [Genotype]
    range = [str]


class Perturbation(Genotype):
    namespace = onto
    domain = [Genotype]
    range = [str]


class GeneDeletion(Perturbation):
    namespace = onto
    domain = [Perturbation]
    range = [str]


class SysGeneNameFull(Genotype):
    namespace = onto
    domain = [Genotype]
    range = [str]


# Define properties for Phenotype
class Smf(Phenotype):
    namespace = onto
    domain = [Phenotype]
    range = [float]


# Define properties for Phenotype
class SmfStd(Phenotype):
    namespace = onto
    domain = [Phenotype]
    range = [float]


class Dmf(Phenotype):
    namespace = onto
    domain = [Phenotype]
    range = [float]


class DmfStd(Phenotype):
    namespace = onto
    domain = [Phenotype]
    range = [float]


class GeneticInteractionScore(Phenotype):
    namespace = onto
    domain = [Phenotype]
    range = [float]


class GeneticInteractionPValue(Phenotype):
    namespace = onto
    domain = [Phenotype]
    range = [float]


# Define properties for Environment
class Media(Environment):
    namespace = onto


class Chemical(Thing):
    namespace = onto


class YeastExtract(Chemical):
    namespace = onto


class Peptone(Chemical):
    namespace = onto


class Dextrose(Chemical):
    namespace = onto


class ComposedOf(ObjectProperty):
    namespace = onto
    domain = [Media]
    range = [Chemical]


class Ypd(Media):
    namespace = onto
    composed_of = ComposedOf()


class Temperature(Environment):
    namespace = onto
    domain = [Environment]
    range = [int]


def create_unique_constraint_if_not_exists(driver):
    with driver.session() as session:
        # Get existing constraints
        constraints = session.run("SHOW CONSTRAINTS").data()

        # Check if the specific constraint exists
        if not any(
            "n10s_unique_uri" in constraint.get("name", "")
            for constraint in constraints
        ):
            # Create the constraint if it does not exist
            session.run(
                "CREATE CONSTRAINT n10s_unique_uri FOR (r:Resource) REQUIRE r.uri IS UNIQUE"
            )


def owl_import_ex():
    # Connection details
    uri = "neo4j://localhost:7687"  # Adjust as needed
    username = "neo4j"
    password = "torchcell"

    # Connect to Neo4j
    driver = GraphDatabase.driver(uri, auth=(username, password))

    # Create the unique constraint if it does not exist
    create_unique_constraint_if_not_exists(driver)

    with driver.session() as session:
        # Execute the n10s graph configuration initialization
        session.run("CALL n10s.graphconfig.init();")

        # Execute the ontology import
        session.run(
            "CALL n10s.onto.import.fetch("
            "'https://raw.githubusercontent.com/Mjvolk3/torchcell/main/torchcell.rdf', "
            "'RDF/XML');"
        )
        print("Ontology import completed successfully.")

    # Close the driver connection
    driver.close()


def main():
    # Create instances of YeastExtract, Peptone, and Dextrose
    yeast_extract = YeastExtract()
    peptone = Peptone()
    dextrose = Dextrose()

    # Create an instance of Ypd and link its components
    ypd = Ypd()
    ypd.composed_of = [yeast_extract, peptone, dextrose]

    onto.save(file="torchcell.rdf", format="rdfxml")


if __name__ == "__main__":
    main()
    owl_import_ex()
