from neo4j import GraphDatabase
import time

# Neo4j Bolt URI and credentials
uri = "bolt://gilahyper.zapto.org:7687"
username = "neo4j"  # Replace with actual username
password = "torchcell"  # Replace with actual password
database = "torchcell"  # Specify the database name


# Function to measure connection time
def measure_connection_time(uri, username, password):
    start_time = time.time()
    driver = GraphDatabase.driver(uri, auth=(username, password))
    connection_time = time.time() - start_time
    driver.close()
    return connection_time


# Function to measure query response time
def measure_query_time(uri, username, password, database):
    driver = GraphDatabase.driver(uri, auth=(username, password))
    session = driver.session(database=database)
    start_time = time.time()
    result = session.run("RETURN 1")
    query_time = time.time() - start_time
    session.close()
    driver.close()
    return query_time


# Measure connection time
connection_time = measure_connection_time(uri, username, password)
print(f"Connection time: {connection_time:.6f} seconds")

# Measure query response time
query_time = measure_query_time(uri, username, password, database)
print(f"Query response time: {query_time:.6f} seconds")
