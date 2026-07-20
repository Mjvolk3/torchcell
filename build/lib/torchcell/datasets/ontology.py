# torchcell/datasets/ontology.py
# [[torchcell.datasets.ontology]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/ontology.py
# Test file: torchcell/datasets/test_ontology.py
"""Owlready2 ontology of experiments and Neo4j/n10s ontology import helpers."""

from neo4j import Driver, GraphDatabase
from owlready2 import ObjectProperty, Thing, get_ontology

# Create a new ontology
# currently only have rdf
onto = get_ontology(
    "https://raw.githubusercontent.com/Mjvolk3/torchcell/main/torchcell.rdf"
)


# Define the top-level Experiment class
class Experiment(Thing):  # type: ignore[misc]  # owlready2 base is untyped (Any)
    """Top-level ontology class representing an experiment."""

    namespace = onto


# Define Genotype, Phenotype, and Environment as subclasses of Experiment
class Genotype(Experiment):
    """Genotype component of an experiment."""

    namespace = onto


class Phenotype(Experiment):
    """Phenotype component of an experiment."""

    namespace = onto


class Environment(Experiment):
    """Environment component of an experiment."""

    namespace = onto


# Define properties for Genotype


class ReferenceGenome(Genotype):
    """Reference genome property of a genotype."""

    namespace = onto
    domain = [Genotype]
    range = [str]


class SysGeneName(Genotype):
    """Systematic gene name property of a genotype."""

    namespace = onto
    domain = [Genotype]
    range = [str]


class Perturbation(Genotype):
    """Genetic perturbation applied to a genotype."""

    namespace = onto
    domain = [Genotype]
    range = [str]


class GeneDeletion(Perturbation):
    """Gene-deletion perturbation."""

    namespace = onto
    domain = [Perturbation]
    range = [str]


class SysGeneNameFull(Genotype):
    """Full systematic gene name property of a genotype."""

    namespace = onto
    domain = [Genotype]
    range = [str]


# Define properties for Phenotype
class Smf(Phenotype):
    """Single-mutant fitness phenotype."""

    namespace = onto
    domain = [Phenotype]
    range = [float]


# Define properties for Phenotype
class SmfStd(Phenotype):
    """Standard deviation of single-mutant fitness."""

    namespace = onto
    domain = [Phenotype]
    range = [float]


class Dmf(Phenotype):
    """Double-mutant fitness phenotype."""

    namespace = onto
    domain = [Phenotype]
    range = [float]


class DmfStd(Phenotype):
    """Standard deviation of double-mutant fitness."""

    namespace = onto
    domain = [Phenotype]
    range = [float]


class GeneticInteractionScore(Phenotype):
    """Genetic interaction score phenotype."""

    namespace = onto
    domain = [Phenotype]
    range = [float]


class GeneticInteractionPValue(Phenotype):
    """P-value associated with a genetic interaction score."""

    namespace = onto
    domain = [Phenotype]
    range = [float]


# Define properties for Environment
class Media(Environment):
    """Growth media environment."""

    namespace = onto


class Chemical(Thing):  # type: ignore[misc]  # owlready2 base is untyped (Any)
    """Chemical that media can be composed of."""

    namespace = onto


class YeastExtract(Chemical):
    """Yeast extract chemical component."""

    namespace = onto


class Peptone(Chemical):
    """Peptone chemical component."""

    namespace = onto


class Dextrose(Chemical):
    """Dextrose chemical component."""

    namespace = onto


class ComposedOf(ObjectProperty):  # type: ignore[misc]  # owlready2 base is untyped (Any)
    """Object property relating media to their constituent chemicals."""

    namespace = onto
    domain = [Media]
    range = [Chemical]


class YEPD(Media):
    """YEPD growth media composed of standard chemical components."""

    namespace = onto
    composed_of = ComposedOf()


class Temperature(Environment):
    """Temperature property of an environment."""

    namespace = onto
    domain = [Environment]
    range = [int]


def create_unique_constraint_if_not_exists(driver: Driver) -> None:
    """Create the n10s unique-URI constraint on :Resource if it is missing.

    Args:
        driver: An open Neo4j driver used to open a session.
    """
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


def owl_import_ex() -> None:
    """Import the torchcell RDF ontology into Neo4j via the n10s procedures."""
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


def main() -> None:
    """Build YEPD media instances and save the ontology to torchcell.rdf."""
    # Create instances of YeastExtract, Peptone, and Dextrose
    yeast_extract = YeastExtract()
    peptone = Peptone()
    dextrose = Dextrose()

    # Create an instance of YEPD and link its components
    yepd = YEPD()
    yepd.composed_of = [yeast_extract, peptone, dextrose]  # type: ignore[assignment]  # owlready2 relation assignment

    onto.save(file="torchcell.rdf", format="rdfxml")


if __name__ == "__main__":
    main()
    owl_import_ex()
