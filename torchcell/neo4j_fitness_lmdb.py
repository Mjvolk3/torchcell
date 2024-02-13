import lmdb
import os.path as osp
import json
from tqdm import tqdm

def iterate_lmdb_database(lmdb_dir: str):
    """
    Iterates over all key-value pairs in an LMDB database and interprets the values as JSON.

    Parameters:
    lmdb_dir (str): The directory where the LMDB database is stored.
    """
    # Ensure the LMDB directory exists
    if not osp.exists(lmdb_dir):
        print(f"LMDB directory {lmdb_dir} does not exist.")
        return

    # Open the LMDB environment in read mode
    env = lmdb.open(lmdb_dir, readonly=True)
    with env.begin() as txn:
        # Create a cursor to iterate over the key-value pairs
        cursor = txn.cursor()
        for key, value in tqdm(cursor):
            # Deserialize the value from JSON to a Python dict
            value_as_dict = json.loads(value.decode())
            # Here, you can process each key-value pair as needed
            # For demonstration, we'll just print the key and the deserialized value
            print(f"Key: {key.decode()}, Value: {value_as_dict}")

    # Close the LMDB environment
    env.close()

# Example usage
if __name__ == "__main__":
    lmdb_dir = "data/torchcell/dmf-2022_02_12/raw/data.lmdb"  # Adjust the path to your LMDB directory
    iterate_lmdb_database(lmdb_dir)
