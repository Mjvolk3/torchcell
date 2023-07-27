#############
from intermine.webservice import Service


# def ym_kb_plus(gene="rad54"):
#     service = Service("https://yeastmine.yeastgenome.org/yeastmine/service")
#     template = service.get_template("Gene_Flanking_Sequence")

#     # You can edit the constraint values below
#     # B    Gene
#     # C    Gene.flankingRegions.direction
#     # A    Gene.flankingRegions.distance
#     # D    Gene.flankingRegions.includeGene

#     rows = template.rows(
#         B={"op": "LOOKUP", "value": gene, "extra_value": "S. cerevisiae"},
#         C={"op": "=", "value": "both"},
#         A={"op": "=", "value": "1.0kb"},
#         D={"op": "=", "value": "true"},
#     )
#     for row in rows:
#         print(
#             row["secondaryIdentifier"],
#             row["symbol"],
#             row["length"],
#             row["flankingRegions.direction"],
#             row["flankingRegions.sequence.length"],
#             row["flankingRegions.sequence.residues"],
#         )
#         print()

# from Bio import Entrez, SeqIO


# def download_genome(accession_number, filename):
#     Entrez.email = "michaeljvolk7@gmail.com"  # Always tell NCBI who you are
#     # First, we need to use esearch to find the IDs of the sequences we're interested in
#     handle = Entrez.esearch(db="nucleotide", term=accession_number, retmax=10000)
#     record = Entrez.read(handle)
#     id_list = record["IdList"]
#     handle.close()
#     # Then, we can fetch the sequences
#     handle = Entrez.efetch(db="nucleotide", id=id_list, rettype="gb", retmode="text")
#     # Save to a file
#     with open(filename, "w") as out_handle:
#         out_handle.write(handle.read())
#     handle.close()


# def load_genome(filename):
#     with open(filename, "r") as handle:
#         return list(SeqIO.parse(handle, "genbank"))


# def extract_genes(genomes, upstream_length, downstream_length):
#     for genome in genomes:
#         for feature in genome.features:
#             if feature.type == "CDS":
#                 location = feature.location
#                 start = location.start
#                 end = location.end
#                 extended_start = max(0, start - upstream_length)
#                 extended_end = min(len(genome.seq), end + downstream_length)
#                 # Extract the gene sequence
#                 gene_seq = genome.seq[extended_start:extended_end]
#                 # Yield the gene sequence and its location
#                 yield gene_seq, location


# def main():
#     genome_file = "yeast_genome.gb"
#     download_genome("GCF_000146045.2", genome_file)
#     yeast_genomes = load_genome(genome_file)
#     upstream_length = 100  # Adjust as needed
#     downstream_length = 100  # Adjust as needed
#     genes = []
#     for gene_seq, location in extract_genes(
#         yeast_genomes, upstream_length, downstream_length
#     ):
#         genes.append(str(gene_seq))  # Convert Seq to str
#     print(genes)  # Prints the sequences of genes

from Bio import Entrez, SeqIO


def download_gene(gene_name, filename):
    Entrez.email = "michaeljvolk7@gmail.com"  # Always tell NCBI who you are
    handle = Entrez.esearch(
        db="nucleotide", term=f"{gene_name} Saccharomyces cerevisiae[Orgn]", retmax=1
    )
    record = Entrez.read(handle)
    id_list = record["IdList"]
    handle.close()

    if len(id_list) > 0:  # if a gene is found
        handle = Entrez.efetch(
            db="nucleotide", id=id_list[0], rettype="gb", retmode="text"
        )
        with open(filename, "w") as out_handle:
            out_handle.write(handle.read())
        handle.close()
    else:
        print(f"No gene found for {gene_name}")


def main():
    gene_file = "gene.gb"
    download_gene("YOR202W", gene_file)


if __name__ == "__main__":
    # ym_kb_plus()
    # other()
    # another()
    main()
