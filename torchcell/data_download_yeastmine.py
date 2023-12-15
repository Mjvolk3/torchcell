from intermine.webservice import Service
import os
from Bio import Entrez
import pickle
from Bio import Entrez
from Bio import SeqIO


def main1():
    gene = "YOR202W"
    service = Service("https://yeastmine.yeastgenome.org/yeastmine/service")
    template = service.get_template("GeneTarget_GeneFactor")
    rows = template.rows(
        A={"op": "LOOKUP", "value": gene, "extra_value": "S. cerevisiae"}
    )
    for row in rows:
        row
        pass


def main2():
    filename = "yeast_genes.pkl"
    if not os.path.isfile(filename):
        # File doesn't exist, download the gene data
        Entrez.email = "michaeljvolk7@gmail.com"  # Always tell NCBI who you are
        handle = Entrez.esearch(
            db="nucleotide",
            term="S. cerevisiae[Orgn] AND gene[All Fields]",
            retmax=10000,
        )
        record = Entrez.read(handle)
        handle.close()

        # ID list of genes
        idlist = record["IdList"]

        # Fetch details for each gene and save to a file
        gene_data = []
        for gene_id in idlist:
            handle = Entrez.efetch(
                db="nucleotide", id=gene_id, rettype="gb", retmode="text"
            )
            record = SeqIO.read(handle, "genbank")
            gene_data.append(record)
            handle.close()

        with open(filename, "wb") as f:
            pickle.dump(gene_data, f)

    # File exists or download completed, read the data from the file
    with open(filename, "rb") as f:
        gene_data = pickle.load(f)

    for record in gene_data:
        print(record)


if __name__ == "__main__":
    # main1()
    main2()
