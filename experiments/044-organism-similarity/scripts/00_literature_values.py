# experiments/044-organism-similarity/scripts/00_literature_values.py
# [[experiments.044-organism-similarity.scripts.00_literature_values]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/044-organism-similarity/scripts/00_literature_values.py

"""The within-human reference line of Fig 1: a literature value, not computed here.

Source: The 1000 Genomes Project Consortium 2015, "A global reference for human genetic
variation", Nature 526:68-74, DOI 10.1038/nature15393 (PMID 26432245, PMC4750478). It
is in neither the torchcell mirror nor ``/bulk/zotero-library`` (checked 2026-10-19: no
key matches and no ``paper.md`` there is this paper; only citing papers mention the
DOI), so its open-access full text is fetched from Europe PMC, sha256 pinned, and the
two sentences used are located VERBATIM in it (an assertion, so a changed text fails):

- "We find that a typical genome differs from the reference human genome at 4.1 million
  to 5.0 million sites"
- "a typical genome contained 149–182 sites with protein truncating variants, 10,000 to
  12,000 sites with peptide-sequence-altering variants"

The plotted line converts the second count into a protein identity against the
reference: ``1 - sites / residues`` with ``residues`` = total length of this study's
per-gene human proteome (``proteomes/hsap_grch38.faa``, 02_proteome_prep.py). The
denominator is ours, the numerator is the paper's, and the counts include heterozygous
sites at full weight, so the line is an upper bound on per-haplotype divergence. The
nucleotide analog (4.1 to 5.0 million sites over the GRCh38 ungapped length from the
NCBI assembly report) is stored for the note. Output ``results/literature_values.json``.

Run from the repo root with the torchcell env.
"""

from __future__ import annotations

import json
import os
import os.path as osp
import re
import subprocess
import sys
import urllib.request

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))

from orgsim_manifest import (  # noqa: E402
    DATASETS_BIN,
    PROTEOME_DIR,
    RESULTS_DIR,
    WORK_DIR,
    sha256_file,
)

URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC4750478/fullTextXML"
LIT_DIR = osp.join(WORK_DIR, "literature")
XML = osp.join(LIT_DIR, "PMC4750478.xml")
XML_SHA256 = "4c635f9f419d8c02af3a667d797033cf288a9f15b42461ead408f49e94af836b"
QUOTE_NT = (
    "We find that a typical genome differs from the reference human genome at "
    "4.1 million to 5.0 million sites"
)
QUOTE_AA = (
    "a typical genome contained 149–182 sites with protein truncating variants, "
    "10,000 to 12,000 sites with peptide-sequence-altering variants"
)
GRCH38 = "GCF_000001405.40"


def main() -> None:
    os.makedirs(LIT_DIR, exist_ok=True)
    if not osp.isfile(XML):
        urllib.request.urlretrieve(URL, XML)
    assert sha256_file(XML) == XML_SHA256, "Europe PMC full text changed; re-pin"
    text = re.sub(r"<[^>]+>", "", open(XML, encoding="utf-8").read())
    text = re.sub(r"\s+", " ", text)
    assert QUOTE_NT in text and QUOTE_AA in text
    residues = 0
    with open(osp.join(PROTEOME_DIR, "hsap_grch38.faa")) as fh:
        for line in fh:
            if not line.startswith(">"):
                residues += len(line.strip())
    report = json.loads(
        subprocess.run(
            [DATASETS_BIN, "summary", "genome", "accession", GRCH38, "--as-json-lines"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.splitlines()[0]
    )
    ungapped = int(report["assembly_stats"]["total_ungapped_length"])
    out = {
        "source": "1000 Genomes Project Consortium 2015, Nature 526:68-74",
        "doi": "10.1038/nature15393",
        "pmcid": "PMC4750478",
        "label": "literature value, not computed here",
        "full_text_url": URL,
        "full_text_sha256": XML_SHA256,
        "quote_nucleotide": QUOTE_NT,
        "quote_protein": QUOTE_AA,
        "peptide_altering_sites": [10_000, 12_000],
        "human_per_gene_proteome_residues": residues,
        "protein_identity_vs_reference": [1 - 12_000 / residues, 1 - 10_000 / residues],
        "nucleotide_sites": [4_100_000, 5_000_000],
        "grch38_ungapped_length": ungapped,
        "nucleotide_identity_vs_reference": [
            1 - 5_000_000 / ungapped,
            1 - 4_100_000 / ungapped,
        ],
        "denominators": "residues: this study's per-gene human proteome; "
        "ungapped length: NCBI assembly report of GCF_000001405.40",
    }
    with open(osp.join(RESULTS_DIR, "literature_values.json"), "w") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
