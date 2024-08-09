import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
import pandas as pd
# Enable the automatic conversion between R and Pandas dataframes
pandas2ri.activate()


def read_rds_to_dataframe(rds_file_path: str) -> pd.DataFrame:
    # Load the RDS file
    readRDS = ro.r["readRDS"]
    rds_data = readRDS(rds_file_path)

    # Convert the R object to a pandas DataFrame
    df = pandas2ri.rpy2py_dataframe(rds_data)

    return df


# Path to your RDS file
rds_file_path = "/Users/michaelvolk/Documents/projects/Gene_Graph/data/Mechanistic_Aware/expressionOnly.RDS"

# Read the RDS file into a pandas DataFrame
df = read_rds_to_dataframe(rds_file_path)

# Display the DataFrame
print(df.head())
