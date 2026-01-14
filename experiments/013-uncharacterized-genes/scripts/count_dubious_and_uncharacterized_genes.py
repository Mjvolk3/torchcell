# experiments/013-uncharacterized-genes/scripts/count_dubious_and_uncharacterized_genes
# [[experiments.013-uncharacterized-genes.scripts.count_dubious_and_uncharacterized_genes]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/013-uncharacterized-genes/scripts/count_dubious_and_uncharacterized_genes
# Test file: experiments/013-uncharacterized-genes/scripts/test_count_dubious_and_uncharacterized_genes.py


import json
import os
import os.path as osp
from typing import Optional

from dotenv import load_dotenv
from pydantic import Field

from torchcell.datamodels.pydant import ModelStrict  # noqa: F401
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome

load_dotenv()
DATA_ROOT = os.getenv("DATA_ROOT")
EXPERIMENT_ROOT = os.getenv("EXPERIMENT_ROOT")

if DATA_ROOT is None:
    raise ValueError("DATA_ROOT environment variable not set")
if EXPERIMENT_ROOT is None:
    raise ValueError("EXPERIMENT_ROOT environment variable not set")


class UncharacterizedGeneData(ModelStrict):
    """Comprehensive data model for uncharacterized genes."""

    id: str = Field(..., description="Systematic gene ID (e.g., YDR210W)")
    name: Optional[list[str]] = Field(
        None, description="Gene name(s) from Name attribute"
    )
    alias: Optional[list[str]] = Field(None, description="Gene aliases")
    orf_classification: Optional[list[str]] = Field(
        None, description="ORF classification (e.g., Uncharacterized)"
    )
    chromosome: int = Field(..., description="Chromosome number (0 for MT)")
    start: int = Field(..., description="Start position on chromosome")
    end: int = Field(..., description="End position on chromosome")
    strand: str = Field(..., description="Strand orientation (+ or -)")
    dna_sequence: str = Field(..., description="DNA sequence of the ORF")
    protein_sequence: Optional[str] = Field(
        None, description="Protein sequence if available"
    )
    cds_sequence: Optional[str] = Field(
        None, description="CDS sequence if available"
    )
    ontology_term: Optional[list[str]] = Field(
        None, description="Ontology terms including GO terms"
    )
    go_terms: Optional[list[str]] = Field(
        None, description="GO terms specifically"
    )
    note: Optional[list[str]] = Field(None, description="Notes about the gene")
    display: Optional[list[str]] = Field(None, description="Display attribute")
    dbxref: Optional[list[str]] = Field(None, description="Database cross-references")


def collect_genes_by_classification(
    classifications: list[str],
) -> dict[str, UncharacterizedGeneData]:
    """
    Collect genes matching specified ORF classifications.

    Args:
        classifications: List of classifications to filter for
            (e.g., ["Uncharacterized", "Dubious"])

    Returns:
        Dictionary mapping gene IDs to UncharacterizedGeneData objects
    """
    # Type assertions for environment variables
    assert DATA_ROOT is not None, "DATA_ROOT must be set"
    assert EXPERIMENT_ROOT is not None, "EXPERIMENT_ROOT must be set"

    # Initialize genome
    genome = SCerevisiaeGenome(
        genome_root=osp.join(DATA_ROOT, "data/sgd/genome"),
        go_root=osp.join(DATA_ROOT, "data/go"),
        overwrite=False,
    )

    genes_by_classification = {}

    # Iterate through all genes
    for gene_id in genome.gene_set:
        gene = genome[gene_id]

        if gene is None:
            continue

        # Filter for genes matching any of the specified classifications
        if gene.orf_classification and any(
            any(target in classification for target in classifications)
            for classification in gene.orf_classification
        ):
            # Extract protein sequence
            protein_seq = None
            if gene.protein is not None:
                protein_seq = str(gene.protein.seq)

            # Extract CDS sequence
            cds_seq = None
            if gene.cds is not None:
                cds_seq = str(gene.cds.seq)

            # Extract GO terms as a list if available
            go_terms = None
            if gene.go is not None:
                go_terms = list(gene.go)

            # Create gene data model
            gene_data = UncharacterizedGeneData(
                id=gene.id,
                name=gene.name,
                alias=gene.alias,
                orf_classification=gene.orf_classification,
                chromosome=gene.chromosome,
                start=gene.start,
                end=gene.end,
                strand=gene.strand,
                dna_sequence=gene.seq,
                protein_sequence=protein_seq,
                cds_sequence=cds_seq,
                ontology_term=gene.ontology_term,
                go_terms=go_terms,
                note=gene.note,
                display=gene.display,
                dbxref=gene.dbxref,
            )

            genes_by_classification[gene_id] = gene_data

    return genes_by_classification


def main():
    """Main function to collect and save uncharacterized and dubious genes."""
    # Type assertions for environment variables
    assert DATA_ROOT is not None, "DATA_ROOT must be set"
    assert EXPERIMENT_ROOT is not None, "EXPERIMENT_ROOT must be set"

    # Collect uncharacterized genes
    print("Collecting uncharacterized genes...")
    uncharacterized_genes = collect_genes_by_classification(["Uncharacterized"])
    print(f"Found {len(uncharacterized_genes)} uncharacterized genes")

    # Collect dubious genes
    print("\nCollecting dubious genes...")
    dubious_genes = collect_genes_by_classification(["Dubious"])
    print(f"Found {len(dubious_genes)} dubious genes")

    # Analyze set relationships
    unchar_set = set(uncharacterized_genes.keys())
    dubious_set = set(dubious_genes.keys())

    # Calculate intersection and exclusive sets
    intersection = unchar_set & dubious_set
    only_unchar = unchar_set - dubious_set
    only_dubious = dubious_set - unchar_set
    union = unchar_set | dubious_set

    # Print set analysis
    print("\n" + "=" * 60)
    print("SET ANALYSIS")
    print("=" * 60)
    print(f"Uncharacterized only:        {len(only_unchar):4d}")
    print(f"Dubious only:                {len(only_dubious):4d}")
    print(f"Intersection (both):         {len(intersection):4d}")
    print(f"Union (total unique):        {len(union):4d}")
    print("=" * 60)

    # Print intersection genes if any
    if intersection:
        print(f"\nGenes with BOTH classifications ({len(intersection)}):")
        for gene_id in sorted(list(intersection)[:10]):
            gene = uncharacterized_genes[gene_id]
            print(f"  {gene_id}: {gene.orf_classification}")
        if len(intersection) > 10:
            print(f"  ... and {len(intersection) - 10} more")

    # Create combined dictionary (union of all genes)
    all_genes = {**uncharacterized_genes, **dubious_genes}

    # Print first few examples of each type
    print("\nFirst 5 uncharacterized genes:")
    for i, (gene_id, gene_data) in enumerate(
        list(uncharacterized_genes.items())[:5]
    ):
        print(f"\n{i+1}. {gene_id}:")
        print(f"   Name: {gene_data.name}")
        print(f"   Classification: {gene_data.orf_classification}")
        location_str = (
            f"chr{gene_data.chromosome}:"
            f"{gene_data.start}-{gene_data.end}"
        )
        print(f"   Location: {location_str}")
        print(f"   DNA length: {len(gene_data.dna_sequence)} bp")
        if gene_data.protein_sequence:
            print(
                f"   Protein length: {len(gene_data.protein_sequence)} aa"
            )
        print(f"   GO terms: {gene_data.go_terms}")

    print("\nFirst 5 dubious genes:")
    for i, (gene_id, gene_data) in enumerate(list(dubious_genes.items())[:5]):
        print(f"\n{i+1}. {gene_id}:")
        print(f"   Name: {gene_data.name}")
        print(f"   Classification: {gene_data.orf_classification}")
        location_str = (
            f"chr{gene_data.chromosome}:"
            f"{gene_data.start}-{gene_data.end}"
        )
        print(f"   Location: {location_str}")
        print(f"   DNA length: {len(gene_data.dna_sequence)} bp")
        if gene_data.protein_sequence:
            print(
                f"   Protein length: {len(gene_data.protein_sequence)} aa"
            )
        print(f"   GO terms: {gene_data.go_terms}")

    # Save to JSON files
    output_dir = osp.join(
        EXPERIMENT_ROOT, "013-uncharacterized-genes", "results"
    )
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("SAVING DATA FILES")
    print("=" * 60)

    # Save uncharacterized genes (all that have this classification)
    output_file_uncharacterized = osp.join(
        output_dir, "uncharacterized_genes.json"
    )
    json_data_uncharacterized = {
        gene_id: gene_data.model_dump()
        for gene_id, gene_data in uncharacterized_genes.items()
    }
    with open(output_file_uncharacterized, "w") as f:
        json.dump(json_data_uncharacterized, f, indent=2)
    print(f"Uncharacterized (all):   {output_file_uncharacterized}")

    # Save dubious genes (all that have this classification)
    output_file_dubious = osp.join(output_dir, "dubious_genes.json")
    json_data_dubious = {
        gene_id: gene_data.model_dump()
        for gene_id, gene_data in dubious_genes.items()
    }
    with open(output_file_dubious, "w") as f:
        json.dump(json_data_dubious, f, indent=2)
    print(f"Dubious (all):           {output_file_dubious}")

    # Save only uncharacterized (exclusive)
    output_file_only_unchar = osp.join(
        output_dir, "only_uncharacterized_genes.json"
    )
    json_data_only_unchar = {
        gene_id: uncharacterized_genes[gene_id].model_dump()
        for gene_id in only_unchar
    }
    with open(output_file_only_unchar, "w") as f:
        json.dump(json_data_only_unchar, f, indent=2)
    print(f"Only uncharacterized:    {output_file_only_unchar}")

    # Save only dubious (exclusive)
    output_file_only_dubious = osp.join(
        output_dir, "only_dubious_genes.json"
    )
    json_data_only_dubious = {
        gene_id: dubious_genes[gene_id].model_dump()
        for gene_id in only_dubious
    }
    with open(output_file_only_dubious, "w") as f:
        json.dump(json_data_only_dubious, f, indent=2)
    print(f"Only dubious:            {output_file_only_dubious}")

    # Save intersection (genes with both classifications)
    if intersection:
        output_file_intersection = osp.join(
            output_dir, "intersection_genes.json"
        )
        json_data_intersection = {
            gene_id: uncharacterized_genes[gene_id].model_dump()
            for gene_id in intersection
        }
        with open(output_file_intersection, "w") as f:
            json.dump(json_data_intersection, f, indent=2)
        print(f"Intersection (both):     {output_file_intersection}")

    # Save union (all unique genes)
    output_file_union = osp.join(output_dir, "union_all_genes.json")
    json_data_union = {
        gene_id: gene_data.model_dump()
        for gene_id, gene_data in all_genes.items()
    }
    with open(output_file_union, "w") as f:
        json.dump(json_data_union, f, indent=2)
    print(f"Union (all unique):      {output_file_union}")
    print("=" * 60)

    return {
        "uncharacterized_all": uncharacterized_genes,
        "dubious_all": dubious_genes,
        "only_uncharacterized": {
            gene_id: uncharacterized_genes[gene_id] for gene_id in only_unchar
        },
        "only_dubious": {
            gene_id: dubious_genes[gene_id] for gene_id in only_dubious
        },
        "intersection": {
            gene_id: uncharacterized_genes[gene_id] for gene_id in intersection
        },
        "union": all_genes,
        "set_stats": {
            "uncharacterized_count": len(unchar_set),
            "dubious_count": len(dubious_set),
            "only_uncharacterized_count": len(only_unchar),
            "only_dubious_count": len(only_dubious),
            "intersection_count": len(intersection),
            "union_count": len(union),
        },
    }


if __name__ == "__main__":
    results = main()
