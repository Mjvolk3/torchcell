# torchcell/neo4j_fitness_query
# [[torchcell.neo4j_fitness_query]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/neo4j_fitness_query
# Test file: tests/torchcell/test_neo4j_fitness_query.py


import json
from neo4j import GraphDatabase
from torchcell.datamodels import FitnessExperiment
from tqdm import tqdm


def fetch_data_instance(uri, username, password):
    driver = GraphDatabase.driver(uri, auth=(username, password))

    with driver.session() as session:
        result = session.run(
            """
            MATCH (e:Experiment)<-[gm:GenotypeMemberOf]-(g:Genotype)<-[pm:PerturbationMemberOf]-(p:Perturbation {perturbation_type: 'deletion'})
            WITH e, COLLECT(p) AS perturbations
            WHERE ALL(p in perturbations WHERE p.perturbation_type = 'deletion')
            RETURN e
            """
            # """
            # # MATCH (e:Experiment)<-[gm:GenotypeMemberOf]-(g:Genotype)<-[pm:PerturbationMemberOf]-(p:Perturbation {perturbation_type: 'deletion'})
            # # WITH e, COLLECT(p) AS perturbations
            # # WHERE ANY(p IN perturbations WHERE p.perturbation_type = 'deletion')
            # # RETURN e
            # # """
        )

        for record in result:
            yield record["e"]["serialized_data"]

    driver.close()


if __name__ == "__main__":
    uri = "neo4j://localhost:7687"
    username = "neo4j"
    password = "neo4j"

    for serialized_data in tqdm(fetch_data_instance(uri, username, password)):
        # data = FitnessExperiment.model_validate(json.loads(serialized_data))
        pass
    print("Query successful!")
