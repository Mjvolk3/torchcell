# experiments/034-showcase-datasets/scripts/genomes_page.py
# [[experiments.034-showcase-datasets.scripts.genomes_page]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/034-showcase-datasets/scripts/genomes_page
"""Write the Genomes page of the website from what the data endpoint serves.

Reads every assembly set the public ``tc-data`` endpoint lists (its manifest: organism,
strain, source, release, files with size and sha256), loads each genome that has a
loader class to count its genes, and writes

    experiments/034-showcase-datasets/website/genomes/index.mdx

which the website branch copies to ``website/docs/genomes/index.mdx``. The page opens
with the reconstructed genomes, then lists the reference genomes by species, then
says how to download a genome through the Python client and over HTTP.

The reconstructed *Issatchenkia orientalis* SD108 genome is unpublished: the page
names it and links its browser and release files, which sit behind a login, and
carries none of its statistics.

    python experiments/034-showcase-datasets/scripts/genomes_page.py
"""

from __future__ import annotations

import json
from pathlib import Path

from dotenv import load_dotenv

from torchcell.datasets.client import DatasetClient
from torchcell.sequence.genome.standard import load_genome, standard_genomes

OUT = Path("experiments/034-showcase-datasets/website/genomes/index.mdx")
HOST = "https://torchcell-database.ncsa.illinois.edu"
# Sets the endpoint serves that are not one reference genome, with what they are.
OTHER = {
    "peter2018_1011_assemblies": "A panel of 1,011 natural isolate assemblies and the pangenome matrices (Peter 2018), not one reference.",
    "hale2024_322134S_variants": "A table of variants of the 3S parent strain against S288C (Hale 2024), not an assembly.",
    "go_release_2024-01-17": "Gene Ontology release 2024-01-17 (`go.obo`), read by the S288C genome class.",
    "go_release_2026-08-05": "Gene Ontology release 2026-08-05 (`go.obo`), read by the bacterial and CEN.PK genome classes.",
}
# Benchmark tasks written against a genome, by assembly set.
TASKS = {
    "ecoli_K12_BW25113_ASM75055v1": "growth rate, cell shape and cell cycle, metabolome",
    "ecoli_K12_MG1655_ASM584v2": "genetic interactions",
}


def megabytes(n: int) -> str:
    """Bytes as megabytes with one decimal."""
    return f"{n / 1e6:,.1f}"


def main() -> None:
    """Write the page."""
    load_dotenv(".env")
    client = DatasetClient()
    loaders = {g.assembly_set: g for g in standard_genomes()}
    manifests = {name: client.genome_manifest(name) for name in client.genomes()}
    reference: dict[str, list[list[str]]] = {}
    other_rows = []
    for name, manifest in sorted(manifests.items()):
        size = megabytes(sum(f.bytes for f in manifest.files))
        if name in OTHER:
            other_rows.append(
                [f"`{name}`", OTHER[name], str(len(manifest.files)), size]
            )
            continue
        loader = loaders[name]
        genome = load_genome(name)
        reference.setdefault(manifest.organism, []).append(
            [
                manifest.strain_or_population,
                f"`{name}`",
                f"{manifest.source}, {manifest.release}",
                f"{len(genome.gene_set):,}",
                str(len(manifest.files)),
                size,
                f"`{loader.class_name}`",
                TASKS.get(name, ""),
            ]
        )

    def table(head: list[str], rows: list[list[str]]) -> list[str]:
        lines = [
            "| " + " | ".join(head) + " |",
            "| " + " | ".join(":--" for _ in head) + " |",
        ]
        return lines + ["| " + " | ".join(row) + " |" for row in rows] + [""]

    page = [
        "---",
        'title: "Genomes"',
        'sidebar_label: "Genomes"',
        'description: "The genomes the datasets are written against, the reconstructed genomes, and how to download them."',
        "---",
        "",
        "Every record in TorchCell names the genome assembly its gene identifiers are written against. The genomes are stored as files with a manifest that pins each file by sha256, and they are served separately from the knowledge graph.",
        "",
        "## Reconstructed genomes",
        "",
        "Genomes assembled and annotated by the project, as opposed to reference genomes taken from a public archive.",
        "",
        ":::info[*Issatchenkia orientalis* (*Pichia kudriavzevii*) SD108, reference genome v1]",
        "",
        "A reference genome and gene annotation of strain SD108, reconstructed from existing SD108 sequencing data. It is unpublished, so the browser and the files are behind a login; ask the project for the credentials.",
        "",
        f"- **Genome browser:** [{HOST}/sd108/]({HOST}/sd108/) (browse chromosomes, genes and tracks; BLAST against the genome).",
        f"- **Release files:** `{HOST}/sd108/release/v1/` holds the genome FASTA and its index, the annotation, the second haplotype and checksum files.",
        "- **In TorchCell:** assembly set `pkudriavzevii_SD108_v1`, read by `PKudriavzeviiSD108Genome`. It is not on the public download endpoint.",
        "",
        "```bash",
        "# Download the genome and check it; curl asks for the password.",
        f"curl -u sd108 -O {HOST}/sd108/release/v1/SD108_genome_v1.fasta",
        f"curl -u sd108 -O {HOST}/sd108/release/v1/SHA256SUMS",
        "sha256sum --check --ignore-missing SHA256SUMS",
        "```",
        "",
        ":::",
        "",
        "## Reference genomes by species",
        "",
        "Each row is one assembly set on the public endpoint. The gene count is the number of genes the loader class reads from the annotation.",
        "",
    ]
    head = [
        "Strain",
        "Assembly set",
        "Source and release",
        "Genes",
        "Files",
        "Size (MB)",
        "Loader class",
        "Benchmark tasks",
    ]
    for organism in sorted(reference):
        page += [f"### *{organism}*", ""]
        page += table(head, reference[organism])
    page += [
        "### Other pinned reference files",
        "",
        "Served from the same endpoint and pinned the same way, but not a single reference genome.",
        "",
    ]
    page += table(["Set", "What it is", "Files", "Size (MB)"], other_rows)
    page += [
        "## Download a genome",
        "",
        "### In Python",
        "",
        'Install with `pip install "torchcell>=1.8.8"`. `load_genome` fetches the files of an assembly set when they are absent and returns the genome object. Each file is taken from its original source when the bytes there still match the recorded sha256, and from the TorchCell mirror otherwise.',
        "",
        "```python",
        "import os",
        "",
        "from torchcell.sequence.genome.standard import load_genome, standard_genomes",
        "",
        'os.environ.setdefault("DATA_ROOT", ".")  # files go to ./torchcell-genomes/',
        "",
        "for row in standard_genomes():  # every genome that loads by its assembly set",
        "    print(row.assembly_set, row.organism, row.strain)",
        "",
        'genome = load_genome("ecoli_K12_MG1655_ASM584v2")',
        'gene = genome["b0002"]',
        "gene.seq          # DNA of the gene",
        "gene.cds.seq      # coding sequence",
        "gene.protein.seq  # protein",
        "```",
        "",
        "To get the genome a dataset is written against, without naming it: `genome_for(dataset)` from the same module.",
        "",
        "To download the files without building the object:",
        "",
        "```python",
        "from torchcell.sequence.genome.standard import fetch_assembly_set",
        "",
        'fetch_assembly_set("ecoli_K12_MG1655_ASM584v2")  # into $DATA_ROOT/torchcell-genomes/',
        "```",
        "",
        "### Over HTTP",
        "",
        f"The public endpoint is `{client.url}`. It needs no key.",
        "",
    ]
    page += table(
        ["Request", "Returns"],
        [
            ["`GET /genomes`", "The names of the assembly sets."],
            [
                "`GET /genomes/<assembly_set>/manifest`",
                "Organism, strain, source, release, and every file with its size, sha256 and original URL.",
            ],
            [
                "`GET /genomes/<assembly_set>/files`",
                "The files of the set with their download paths.",
            ],
        ],
    )
    page += [
        "```bash",
        f"curl {client.url}/genomes",
        f"curl {client.url}/genomes/ecoli_K12_MG1655_ASM584v2/manifest",
        "```",
        "",
        "Check every downloaded file against the sha256 in the manifest. The Python client does that for you (`DatasetClient().download_genome_file`).",
        "",
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(page), encoding="utf-8")
    print(
        f"wrote {OUT}: {sum(len(v) for v in reference.values())} reference genomes, {len(other_rows)} other sets"
    )
    print(json.dumps({k: len(v) for k, v in reference.items()}))


if __name__ == "__main__":
    main()
