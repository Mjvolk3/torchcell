# torchcell/neo4j_fitness_query
# [[torchcell.neo4j_fitness_query]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/neo4j_fitness_query
# Test file: tests/torchcell/test_neo4j_fitness_query.py

import attrs
import lmdb
from neo4j import GraphDatabase
import os
from tqdm import tqdm
from attrs import define, field
import os.path as osp


@define
class Neo4jQueryDatabase:
    uri: str
    username: str
    password: str
    root_dir: str
    query: str
    raw_dir: str = field(init=False, default=None)
    env: str = field(init=False, default=None)

    def __attrs_post_init__(self):
        self.raw_dir = osp.join(self.root_dir, "raw", "lmdb")

        if not osp.exists(self.raw_dir):
            os.makedirs(self.raw_dir)
        # Ensure the root_dir exists
        os.makedirs(self.raw_dir, exist_ok=True)
        # Initialize LMDB environment
        self.env = lmdb.open(self.raw_dir, map_size=int(1e6))  # Adjust map_size as needed

        # Process the data
        self.process_data()

    def fetch_data(self):
        driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
        with driver.session() as session:
            result = session.run(self.query)
            for record in result:
                print(record)
                yield record

        driver.close()

    def write_to_lmdb(self, key: bytes, value: bytes):
        with self.env.begin(write=True) as txn:
            txn.put(key, value)

    def process_data(self):
        for i, record in tqdm(enumerate(self.fetch_data())):
            # Assuming each record can be uniquely identified and serialized
            # Adjust the key and value encoding as per your data structure
            key = f"record_{i}".encode()  # Example key
            value = str(record).encode()  # Example value serialization

            self.write_to_lmdb(key, value)
        print(f"Finished on {i}")
        self.env.close()


# Example usage
if __name__ == "__main__":
    neo4j_db = Neo4jQueryDatabase(
        uri="bolt://localhost:7687",
        username="neo4j",
        password="torchcell",  # Replace with your actual password
        root_dir="data/torchcell/dmf-2022_02_12",
        query="""
            MATCH (n)
            RETURN n LIMIT 10
        """,
    )

## Get experiments and references
# MATCH (e:Experiment)-[:GenotypeMemberOf]->(g:Genotype)-[:PerturbationMemberOf]->(p:Perturbation {perturbation_type: 'deletion'})
# WITH e, COLLECT(p) AS perturbations
# WHERE ANY(p IN perturbations WHERE p.perturbation_type = 'deletion')
# MATCH (e)-[:Reference]->(ref:ExperimentReference)
# RETURN e, ref
