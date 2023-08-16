###

## Pandas view of data
import pandas as pd

# Specify column names
column_names = [
    "seqid",
    "source",
    "type",
    "start",
    "end",
    "score",
    "strand",
    "phase",
    "attributes",
]

# Read the file
gff_data = pd.read_csv(
    "data/ncbi/s_cerevisiae/ncbi_dataset/data/GCF_000146045.2/genomic.gff",
    sep="\t",
    comment="#",
    names=column_names,
)

# Display the first few rows of the dataframe
print(gff_data.head())

# from Bio import SeqIO

# # specify the path to your .fna file
# path_to_fna = "data/ncbi/s_cerevisiae/ncbi_dataset/data/GCF_000146045.2/GCF_000146045.2_R64_genomic.fna"

# gene_list = []
# for record in SeqIO.parse(path_to_fna, "fasta"):
#     description = record.description.split(' ')
#     for i, desc in enumerate(description):
#         if desc == 'gene':
#             gene_list.append(description[i+1])

# # remove duplicates, if any
# gene_list = list(set(gene_list))

# print(gene_list)

#####3
import gffutils

# specify the path to your .fna file
path_to_gff = "data/ncbi/s_cerevisiae/ncbi_dataset/data/GCF_000146045.2/genomic.gff"

db = gffutils.create_db(
    path_to_gff,
    dbfn="data.db",
    force=True,
    keep_order=True,
    merge_strategy="merge",
    sort_attribute_values=True,
)

genes = []
for gene_feature in db.features_of_type("gene"):
    genes.append(
        gene_feature["Name"][0]
    )  # adjust this depending on the actual structure of your .gff file

####
