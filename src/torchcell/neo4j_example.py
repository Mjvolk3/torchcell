import json

from neo4j import GraphDatabase


def make_fake_data_1():
    some_serialized_data = {"key1": "value1", "key2": 123}  # Example serialized data
    fake_data = {
        "genotype": [
            {"id": "YPL264C", "intervention": "deletion", "id_full": "YPL264C_sn3676"},
            {"id": "YNL051W", "intervention": "deletion", "id_full": "YNL051W_dma3963"},
        ],
        "phenotype": {
            "dmf": 0.9457,
            "dmf_std": 0.0292,
            "genetic_interaction_score": 0.0206,
            "genetic_interaction_p_value": 0.2841,
            "special_fitness": some_serialized_data,  # Serialized data added
        },
        "environment": {"media": "Ypd", "temperature": 30},
    }
    return fake_data


def make_fake_data_2():
    some_serialized_data = {"key1": "value1", "key2": 123}  # Example serialized data
    fake_data = {
        "genotype": [
            {"id": "YFL041W", "intervention": "deletion", "id_full": "YFL041W_sn2437"},
            {"id": "YDR097C", "intervention": "deletion", "id_full": "YDR097C_dma3963"},
        ],
        "phenotype": {
            "dmf": 1.0265,
            "dmf_std": 0.0034,
            "genetic_interaction_score": -0.0045,
            "genetic_interaction_p_value": 0.2112,
            "special_fitness": some_serialized_data,  # Serialized data added
        },
        "environment": {"media": "YPD", "temperature": 30},
    }
    return fake_data


def upload_data_instance(session, data_instance_name, data_instance):
    # Create the main data instance node with dynamic name
    session.run(f"CREATE (di:DataInstance {{name: '{data_instance_name}'}})")

    # Base URI of your ontology
    base_ontology_uri = (
        "https://raw.githubusercontent.com/Mjvolk3/torchcell/main/torchcell.rdf"
    )

    for genotype in data_instance["genotype"]:
        if genotype.get("intervention") == "deletion":
            # Link the Deletion node to the ontology Resource node
            session.run(
                f"""
                MATCH (di:DataInstance {{name: '{data_instance_name}'}})
                MERGE (deletion:Deletion {{id: $id, id_full: $id_full, intervention: $intervention}})
                MERGE (di)-[:HAS_GENOTYPE]->(deletion)
                WITH deletion
                MATCH (ont:Resource {{uri: '{base_ontology_uri}#GeneDeletion'}})
                MERGE (deletion)-[:IS_A]->(ont)
                """,
                {
                    "id": genotype["id"],
                    "id_full": genotype["id_full"],
                    "intervention": genotype["intervention"],
                },
            )


# def upload_data_instance(session, data_instance_name, data_instance):
#     # Create the main data instance node with dynamic name
#     session.run(f"CREATE (di:DataInstance {{name: '{data_instance_name}'}})")

#     # Base URI of your ontology
#     base_ontology_uri = "http://example.org/onto.owl"

#     # Create Genotype nodes and link them to the data instance and ontology
#     for genotype in data_instance["genotype"]:
#         session.run(
#             f"""
#             MATCH (di:DataInstance {{name: '{data_instance_name}'}})
#             MERGE (g:Genotype {{id: $id, intervention: $intervention, id_full: $id_full}})
#             MERGE (di)-[:HAS_GENOTYPE]->(g)
#             MERGE (ont:Resource {{uri: '{base_ontology_uri}#Genotype'}})
#             MERGE (g)-[:IS_A]->(ont)
#             """,
#             genotype,
#         )

# # Create Phenotype nodes and link them to the data instance and ontology
# phenotype = data_instance["phenotype"]
# serialized_data = json.dumps(phenotype["special_fitness"])
# session.run(
#     f"""
#     MATCH (di:DataInstance {{name: '{data_instance_name}'}})
#     MERGE (p:Phenotype {{dmf: $dmf, dmf_std: $dmf_std,
#         genetic_interaction_score: $genetic_interaction_score,
#         genetic_interaction_p_value: $genetic_interaction_p_value,
#         special_fitness: $special_fitness}})
#     MERGE (di)-[:HAS_PHENOTYPE]->(p)
#     MERGE (ont:Resource {{uri: '{base_ontology_uri}#Phenotype'}})
#     MERGE (p)-[:IS_A]->(ont)
#     """,
#     {
#         "dmf": phenotype["dmf"],
#         "dmf_std": phenotype["dmf_std"],
#         "genetic_interaction_score": phenotype["genetic_interaction_score"],
#         "genetic_interaction_p_value": phenotype["genetic_interaction_p_value"],
#         "special_fitness": serialized_data,
#     },
# )

# # Create Environment node and link it to the data instance and ontology
# environment = data_instance["environment"]
# session.run(
#     f"""
#     MATCH (di:DataInstance {{name: '{data_instance_name}'}})
#     MERGE (e:Environment {{media: $media, temperature: $temperature}})
#     MERGE (di)-[:HAS_ENVIRONMENT]->(e)
#     MERGE (ont:Resource {{uri: '{base_ontology_uri}#Environment'}})
#     MERGE (e)-[:IS_A]->(ont)
#     MERGE (m:Media {{name: $media}})
#     MERGE (e)-[:HAS_MEDIA]->(m)
#     MERGE (t:Temperature {{value: $temperature}})
#     MERGE (e)-[:HAS_TEMPERATURE]->(t)
#     """,
#     environment,
# )


def add_data():
    data_0 = make_fake_data_1()
    data_1 = make_fake_data_2()

    uri = "neo4j://localhost:7687"
    username = "neo4j"
    password = "torchcell"

    driver = GraphDatabase.driver(uri, auth=(username, password))
    with driver.session() as session:
        upload_data_instance(session, "dmf_dataset_0", data_0)
        upload_data_instance(session, "dmf_dataset_1", data_1)
    driver.close()


def fetch_data_instance(uri, username, password):
    driver = GraphDatabase.driver(uri, auth=(username, password))
    data_instance = {}

    with driver.session() as session:
        result = session.run(
            "MATCH (di:DataInstance)-[:HAS_GENOTYPE]->(g:Genotype), "
            "(di)-[:HAS_PHENOTYPE]->(p:Phenotype), "
            "(di)-[:HAS_ENVIRONMENT]->(e:Environment) "
            "RETURN di, collect(g) as genotypes, p, e"
        )

        for record in result:
            # Assuming we're interested in the first record only for this example
            di_node = record["di"]
            genotype_nodes = record["genotypes"]
            phenotype_node = record["p"]
            environment_node = record["e"]

            # Reconstruct the data instance
            data_instance["genotype"] = [
                genotype._properties for genotype in genotype_nodes
            ]
            data_instance["phenotype"] = phenotype_node._properties
            data_instance["environment"] = environment_node._properties

    driver.close()
    return data_instance


if __name__ == "__main__":
    uri = "neo4j://localhost:7687"
    username = "neo4j"
    password = "torchcell"
    add_data()
    # # dmf_dataset_0 = fetch_data_instance(uri, username, password)
    # # print(dmf_dataset_0)
