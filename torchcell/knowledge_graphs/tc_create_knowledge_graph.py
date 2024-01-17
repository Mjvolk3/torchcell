from biocypher import BioCypher, Resource
from torchcell.adapters import CostanzoSmfAdapter
from torchcell.datasets.scerevisiae import SmfCostanzo2016Dataset

bc = BioCypher()

RUN_OPTIONAL_STEPS = False

dataset = SmfCostanzo2016Dataset()
adapter = CostanzoSmfAdapter(dataset=dataset)

if RUN_OPTIONAL_STEPS:
    bc.add(adapter.get_nodes())
    # bc.add(adapter.get_edges())
    dfs = bc.to_df()
    for name, df in dfs.items():
        print(name)
        print(df.head())

bc.write_nodes(adapter.get_nodes())
bc.write_edges(adapter.get_edges())

# TODO preferred_id is not reflected in the output

# Write admin import statement and schema information (for biochatter)
bc.write_import_call()
bc.write_schema_info(as_node=True)

# Print summary
bc.summary()

