"""Download S. cerevisiae gene records from YeastMine and NCBI Entrez."""

import os
import pickle

from Bio import Entrez, SeqIO
from intermine.webservice import Service


def main1() -> None:
    """Query YeastMine for the gene/factor template of a single gene."""
    gene = "YOR202W"
    service = Service("https://yeastmine.yeastgenome.org/yeastmine/service")
    template = service.get_template("GeneTarget_GeneFactor")
    rows = template.rows(
        A={"op": "LOOKUP", "value": gene, "extra_value": "S. cerevisiae"}
    )
    for row in rows:
        row
        pass


def main2() -> None:
    """Fetch and cache all S. cerevisiae nucleotide gene records via Entrez."""
    filename = "yeast_genes.pkl"
    if not os.path.isfile(filename):
        # File doesn't exist, download the gene data
        # Biopython ships py.typed but types Entrez.email as None and leaves
        # esearch/efetch/read/SeqIO.read untyped; suppress the resulting noise.
        Entrez.email = "michaeljvolk7@gmail.com"  # type: ignore[assignment]  # Biopython stub types email as None
        handle = Entrez.esearch(  # type: ignore[no-untyped-call]  # Biopython untyped
            db="nucleotide",
            term="S. cerevisiae[Orgn] AND gene[All Fields]",
            retmax=10000,
        )
        record = Entrez.read(handle)  # type: ignore[no-untyped-call]  # Biopython untyped
        handle.close()

        # ID list of genes
        idlist = record["IdList"]

        # Fetch details for each gene and save to a file
        gene_data = []
        for gene_id in idlist:
            handle = Entrez.efetch(  # type: ignore[no-untyped-call]  # Biopython untyped
                db="nucleotide", id=gene_id, rettype="gb", retmode="text"
            )
            record = SeqIO.read(handle, "genbank")  # type: ignore[no-untyped-call]  # Biopython untyped
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
