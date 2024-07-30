import time
import neo4j_utils as nu

def measure_connection_time(connection_args: dict) -> float:
    """
    Measure the time taken to establish a connection to the database.
    """
    start_time = time.time()
    driver = nu.Driver(
        db_uri=f"bolt://{connection_args['host']}:{connection_args['port']}",
        user=connection_args['user'],
        password=connection_args['password'],
    )
    driver.db_connect()
    connection_time = time.time() - start_time
    driver.close()
    return connection_time

def measure_query_time(connection_args: dict) -> float:
    """
    Measure the time taken to execute a simple query.
    """
    driver = nu.Driver(
        db_uri=f"bolt://{connection_args['host']}:{connection_args['port']}",
        user=connection_args['user'],
        password=connection_args['password'],
    )
    driver.db_connect()
    start_time = time.time()
    result = driver.query("RETURN 1", db_name=connection_args['db_name'])
    query_time = time.time() - start_time
    driver.close()
    return query_time

# Example usage:
if __name__ == "__main__":
    connection_args = {
        "host": "gilahyper.zapto.org",
        "port": 7687,
        "user": "neo4j",  # Replace with actual username
        "password": "torchcell",  # Replace with actual password
        "db_name": "torchcell"  # Specify the actual database name here
    }

    try:
        connection_time = measure_connection_time(connection_args)
        print(f"Connection time: {connection_time:.6f} seconds")

        query_time = measure_query_time(connection_args)
        print(f"Query response time: {query_time:.6f} seconds")
    except Exception as e:
        print(f"An error occurred: {e}")
