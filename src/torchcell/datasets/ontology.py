# src/torchcell/datasets/ontology.py
# [[src.torchcell.datasets.ontology]]
# https://github.com/Mjvolk3/torchcell/tree/main/src/torchcell/datasets/ontology.py
# Test file: src/torchcell/datasets/test_ontology.py
from owlready2 import DataProperty, FunctionalProperty, Thing, get_ontology

# Create a new ontology
onto = get_ontology("http://example.org/onto.owl")


# Define the top-level Experiment class
class Experiment(Thing):
    namespace = onto


# Define Genotype and Phenotype as subclasses of Experiment
class Genotype(Experiment):
    namespace = onto


class Phenotype(Experiment):
    namespace = onto


# Define subclasses of Genotype and Phenotype
class Allele(Genotype):
    namespace = onto


class Observation(Phenotype):
    namespace = onto


class Environment(Phenotype):
    namespace = onto


# Define Functional Data Properties for each of the subclasses
class intervention(FunctionalProperty, DataProperty):
    namespace = onto
    domain = [Allele]
    range = [str]


class id_full(FunctionalProperty, DataProperty):
    namespace = onto
    domain = [Allele]
    range = [str]


class smf(FunctionalProperty, DataProperty):
    namespace = onto
    domain = [Observation]
    range = [float]


class smf_std(FunctionalProperty, DataProperty):
    namespace = onto
    domain = [Observation]
    range = [float]


class media(FunctionalProperty, DataProperty):
    namespace = onto
    domain = [Environment]
    range = [str]


class temperature(FunctionalProperty, DataProperty):
    namespace = onto
    domain = [Environment]
    range = [int]


# Function to create and link instances
def main():
    # Create an instance of Experiment
    experiment1 = Experiment("experiment1")

    # Create instances of Genotype and Phenotype subclasses and assign properties
    allele1 = Allele("allele1")
    allele1.intervention = "deletion"
    allele1.id_full = "YDL171C_dma736"

    observation1 = Observation("observation1")
    observation1.smf = 0.9777
    observation1.smf_std = 0.0679

    environment1 = Environment("environment1")
    environment1.media = "YPD"
    environment1.temperature = 30

    # Link the Allele, Observation, and Environment instances to the Experiment instance
    # This assumes that the Experiment class has object properties to reference these instances
    experiment1.hasGenotype = [allele1]
    experiment1.hasPhenotype = [observation1, environment1]

    # Save the ontology to a file
    onto.save(file="my_ontology.owl", format="rdfxml")


if __name__ == "__main__":
    main()
