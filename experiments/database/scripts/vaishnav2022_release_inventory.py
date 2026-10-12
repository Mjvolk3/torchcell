# experiments/database/scripts/vaishnav2022_release_inventory.py
# [[experiments.database.scripts.vaishnav2022_release_inventory]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/database/scripts/vaishnav2022_release_inventory.py
"""Every non-verifier count the Vaishnav 2022 (#991) and Zhang 2020 (#990) notes cite.

Reads the sha256-pinned raw mirror (and, with ``--cenpk``, a locally retrieved CEN.PK
assembly) and writes the measured numbers to
``experiments/database/results/vaishnav2022_release_inventory.json``. Nothing here is
asserted from the paper: every figure is counted off the bytes the loader consumes, so
the dataset note's numbers have a script behind them.

What it measures:

- the GEO MolEvol SC-Ura table: rows, distinct ``seqID`` groups, distinct sequences, how
  many rows a group is split into, the groups with no ``ExpressionLevel``, and the
  estimate range;
- the constant pTpA scaffold flanks, which is what makes the variable-region span
  readable rather than assumed;
- the Zenodo defined-medium test set: its join to the GEO table, the agreement of its
  ``meanEL`` with the GEO value, and the identity ``sdEL == sdScale * relativeSD``, which
  is why no uncertainty is stored;
- the deferred large files, by the byte counts their hosts publish, so the later
  admission names its own size;
- with ``--cenpk <dir>``, the CEN.PK113-7D assembly Zhang 2020 is blocked on: its strain,
  gene count and locus-tag namespace, measured on the GenBank flat file.

Usage::

    PYTHONPATH=$WT python experiments/database/scripts/vaishnav2022_release_inventory.py
    PYTHONPATH=$WT python experiments/database/scripts/vaishnav2022_release_inventory.py \
        --cenpk /scratch/.../cenpk --write
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import os.path as osp
import re
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv

from torchcell.datasets.scerevisiae import vaishnav2022 as v

RESULTS = Path(__file__).resolve().parents[1] / "results"
OUT = RESULTS / "vaishnav2022_release_inventory.json"

#: The large files NOT loaded, with the byte count their host publishes (measured by a
#: HEAD against the GEO ftp listing and the Zenodo record API on 2026-10-12).
DEFERRED_FILES: dict[str, dict[str, Any]] = {
    "GSE163045_average_promoter_ELs_per_seq_N80_SD-Ura_Y8205_ALL.shuffled.txt.gz": {
        "host": "GEO GSE163045",
        "bytes": 668401864,
        "why": "the Y8205 random training library; a size decision for the owner",
    },
    "defined_media_training_data_SC_Ura.txt": {
        "host": "Zenodo 4436477",
        "bytes": 2473729542,
        "md5": "bd94100c15d7575e265c002cf6e84712",
        "why": "20,616,659 random sequences in defined medium; a size decision",
    },
    "complex_media_training_data_Glu.txt": {
        "host": "Zenodo 4436477",
        "bytes": 3592193655,
        "md5": "4983ec724b503f06e5f466a6c3aaea45",
        "why": "30,722,376 random sequences in complex medium; a size decision",
    },
    "Random_testdata_complex_media.txt": {
        "host": "Zenodo 4436477",
        "bytes": 427196,
        "md5": "fafd8196d8e939ee288bdfee0450199c",
        "why": (
            "the complex-medium (YPD) arm's test slice, whose full measured release is "
            "the 3.59 GB file above; loading the slice alone would make that medium's "
            "coverage an artifact of the size decision"
        ),
    },
}


def molevol_inventory(path: Path) -> dict[str, Any]:
    """The GEO table's shape, counted off its bytes."""
    table = pd.read_csv(path, sep="\t")
    groups = table.groupby("seqID")
    sizes = groups.size()
    measured = table.dropna(subset=["ExpressionLevel"])
    flank5 = table.seq110.str[: len(v.SCAFFOLD_5PRIME)].value_counts()
    flank3 = table.seq110.str[-len(v.SCAFFOLD_3PRIME) :].value_counts()
    prefixes = (
        table.origID.str.split(".").str[0].value_counts().head(10).to_dict()
    )
    return {
        "columns": list(table.columns),
        "n_rows": int(len(table)),
        "n_seqid_groups": int(table.seqID.nunique()),
        "n_distinct_sequences": int(table.seq110.nunique()),
        "n_rows_whose_origid_is_its_seqid": int((table.origID == table.seqID).sum()),
        "rows_per_group": {int(k): int(n) for k, n in sizes.value_counts().items()},
        "max_sequences_per_group": int(groups.seq110.nunique().max()),
        "max_values_per_group": int(groups.ExpressionLevel.nunique().max()),
        "n_rows_without_a_value": int(table.ExpressionLevel.isna().sum()),
        "n_groups_without_a_value": int(
            table.seqID.nunique() - measured.seqID.nunique()
        ),
        "n_records": int(measured.seqID.nunique()),
        "expression_level_min": float(table.ExpressionLevel.min()),
        "expression_level_max": float(table.ExpressionLevel.max()),
        "sequence_lengths": {
            int(k): int(n) for k, n in table.seq110.str.len().value_counts().items()
        },
        "scaffold_5prime": {str(flank5.index[0]): int(flank5.iloc[0])},
        "scaffold_3prime": {str(flank3.index[0]): int(flank3.iloc[0])},
        "top_origid_prefixes": {str(k): int(n) for k, n in prefixes.items()},
    }


def zenodo_inventory(testdata: Path, molevol: Path) -> dict[str, Any]:
    """The Zenodo test set's join to the GEO table, and what its three SD columns are."""
    frame = pd.read_csv(testdata)
    table = pd.read_csv(molevol, sep="\t")
    joined = frame[["origID", "expt", "proTotal", "meanEL", "sdEL"]].merge(
        table[["origID", "seqID", "ExpressionLevel"]], on="origID", how="left"
    )
    scaled = (frame.sdEL - frame.sdScale * frame.relativeSD).abs().max()
    return {
        "n_rows": int(len(frame)),
        "n_columns": int(len(frame.columns)),
        "expt_values": {str(k): int(n) for k, n in frame.expt.value_counts().items()},
        "n_rows_joining_the_geo_table": int(joined.seqID.notna().sum()),
        "n_distinct_seqid_groups": int(joined.seqID.nunique()),
        "max_abs_value_disagreement": float(
            (joined.meanEL - joined.ExpressionLevel).abs().max()
        ),
        "read_count_min": int(frame.proTotal.min()),
        "read_count_max": int(frame.proTotal.max()),
        "n_rows_under_100_reads": int((frame.proTotal < 100).sum()),
        "sdel_equals_sdscale_times_relativesd_max_abs_delta": float(scaled),
    }


def cenpk_inventory(directory: Path) -> dict[str, Any]:
    """The CEN.PK113-7D assembly #990 is blocked on, measured on its GenBank flat file.

    ``directory`` holds the NCBI objects as retrieved (the flat file and the directory's
    ``md5checksums.txt``). The md5 is checked against the listing, so the measurement is
    of the bytes NCBI publishes, not of a local copy.
    """
    name = "GCA_002571405.2_ASM257140v2_genomic.gbff.gz"
    import hashlib

    flat = directory / name
    digest = hashlib.md5(flat.read_bytes(), usedforsecurity=False).hexdigest()
    published = {
        line.split()[1].lstrip("./"): line.split()[0]
        for line in (directory / "md5checksums.txt").read_text().splitlines()
        if line.strip()
    }
    organism = strain = None
    n_gene = 0
    namespaces: dict[str, int] = {}
    with gzip.open(flat, "rt") as handle:
        for line in handle:
            if organism is None and line.startswith("  ORGANISM"):
                organism = line.strip()
            if strain is None and "/strain=" in line:
                strain = line.strip()
            if re.match(r"^     gene ", line):
                n_gene += 1
            match = re.search(r'/locus_tag="([A-Za-z]+)', line)
            if match:
                namespaces[match.group(1)] = namespaces.get(match.group(1), 0) + 1
    return {
        "assembly": "GCA_002571405.2 (ASM257140v2)",
        "source_url": (
            "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCA/002/571/405/"
            "GCA_002571405.2_ASM257140v2/"
        ),
        "flat_file": name,
        "md5_matches_the_published_listing": digest == published.get(name),
        "md5": digest,
        "organism_line": organism,
        "strain_line": strain,
        "n_gene_features": n_gene,
        "locus_tag_namespaces": namespaces,
    }


def main(argv: list[str] | None = None) -> int:
    """Measure the release and print the inventory; ``--write`` saves the JSON."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cenpk", help="directory holding the retrieved CEN.PK objects")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)
    load_dotenv()
    mirror = v.raw_mirror_dir(os.environ["DATA_ROOT"])
    inventory: dict[str, Any] = {
        "citation_key": v.CITATION_KEY,
        "raw_mirror": str(mirror),
        "molevol": molevol_inventory(mirror / v.MOLEVOL_REL),
        "zenodo_defined_test_set": zenodo_inventory(
            mirror / v.TESTDATA_REL, mirror / v.MOLEVOL_REL
        ),
        "deferred_files": DEFERRED_FILES,
    }
    if args.cenpk is not None:
        inventory["zhang2020_cenpk_assembly"] = cenpk_inventory(Path(args.cenpk))
    text = json.dumps(inventory, indent=2, sort_keys=False)
    print(text)
    if args.write:
        RESULTS.mkdir(parents=True, exist_ok=True)
        OUT.write_text(text + "\n")
        print(f"wrote {osp.relpath(OUT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
