# experiments/036-dataset-fixes-before-kg-build/scripts/cachera2023_environment_readout.py
# [[experiments.036-dataset-fixes-before-kg-build.scripts.cachera2023_environment_readout]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/036-dataset-fixes-before-kg-build/scripts/cachera2023_environment_readout
"""Issue #509 before/after: Cachera 2023 medium, temperature and measurement type.

Builds ``BetaxanthinCachera2023Dataset`` into a scratch root (the dev tree's sha256-pinned
``raw/GA1_2_4_6.csv`` copied in, the shared genome read with ``overwrite=False``), then
reads that store and the dev store ``$DATA_ROOT/data/torchcell/betaxanthin_cachera2023``
side by side and tallies, per store: record count, distinct medium (name, state,
is_synthetic, base_medium, component names), distinct temperature, distinct environment
provenance gaps and distinct ``measurement_type``. It also checks that genotype and
phenotype levels are identical record for record, so the fix is shown to touch only the
environment and the measurement-type string. Last, it counts growth-temperature
mentions in the mirror OCR and in the text of every document in the SI archive
(``si/gkad656_supplemental_files.zip``), the evidence behind the loader's typed
temperature gap.

Writes ``experiments/036-dataset-fixes-before-kg-build/results/cachera2023_environment_readout.json``.

Usage (from the repo root)::

    python experiments/036-dataset-fixes-before-kg-build/scripts/cachera2023_environment_readout.py \
        --scratch-root <dir>
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import os.path as osp
import pickle
import re
import shutil
import zipfile
from collections import Counter
from typing import Any

import lmdb
from dotenv import load_dotenv
from pypdf import PdfReader

from torchcell.data.experiment_dataset import resolve_interned
from torchcell.datasets.scerevisiae.cachera2023 import (
    CITATION_KEY,
    PAPER_MD_SHA256,
    BetaxanthinCachera2023Dataset,
)
from torchcell.sequence.genome.scerevisiae import SCerevisiaeGenome

SLUG = "data/torchcell/betaxanthin_cachera2023"
RESULTS = osp.join(
    osp.dirname(osp.dirname(osp.abspath(__file__))),
    "results",
    "cachera2023_environment_readout.json",
)


def read_records(processed_dir: str) -> list[dict[str, Any]]:
    """Every record of a built store, interned ``$ref`` pointers resolved."""
    interned: dict[str, Any] = {}
    interned_dir = osp.join(processed_dir, "interned")
    if osp.isdir(interned_dir):
        env = lmdb.open(interned_dir, readonly=True, lock=False)
        with env.begin() as txn:
            for key, value in txn.cursor():
                interned[key.decode()] = pickle.loads(value)
        env.close()
    env = lmdb.open(osp.join(processed_dir, "lmdb"), readonly=True, lock=False)
    records: list[dict[str, Any]] = []
    with env.begin() as txn:
        n = txn.stat()["entries"]
        for idx in range(n):
            raw = txn.get(f"{idx}".encode())
            assert raw is not None, f"missing key {idx} in {processed_dir}"
            records.append(resolve_interned(pickle.loads(raw), interned))
    env.close()
    return records


def tally(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Distinct environment and readout values across experiments and references."""
    media: Counter[str] = Counter()
    temperature: Counter[str] = Counter()
    gaps: Counter[str] = Counter()
    mtype: Counter[str] = Counter()
    for rec in records:
        for env in (
            rec["experiment"]["environment"],
            rec["reference"]["environment_reference"],
        ):
            m = env["media"]
            media[
                json.dumps(
                    {
                        "name": m["name"],
                        "state": m["state"],
                        "is_synthetic": m["is_synthetic"],
                        "base_medium": m.get("base_medium"),
                        "components": [
                            [
                                c["compound"]["name"],
                                str(c["concentration"]["value"]),
                                str(c["concentration"]["unit"]),
                            ]
                            for c in m.get("components", [])
                        ],
                    }
                )
            ] += 1
            t = env["temperature"]
            temperature["None" if t is None else f"{t['value']} {t['unit']}"] += 1
            gaps[
                json.dumps(
                    [
                        [g["field"], str(g["reason"])]
                        for g in env.get("provenance_gaps", [])
                    ]
                )
            ] += 1
        mtype[rec["experiment"]["phenotype"]["measurement_type"]] += 1
        mtype[rec["reference"]["phenotype_reference"]["measurement_type"]] += 1
    return {
        "records": len(records),
        "media (experiment + reference environments)": {k: v for k, v in media.items()},
        "temperature": dict(temperature),
        "environment_provenance_gaps": dict(gaps),
        "measurement_type (experiment + reference phenotypes)": dict(mtype),
    }


def unchanged_outside_environment(
    old: list[dict[str, Any]], new: list[dict[str, Any]]
) -> dict[str, int]:
    """Count records whose genotype or phenotype levels differ between the stores."""
    assert len(old) == len(new), (len(old), len(new))
    genotype_diff = 0
    level_diff = 0
    for a, b in zip(old, new, strict=True):
        if a["experiment"]["genotype"] != b["experiment"]["genotype"]:
            genotype_diff += 1
        pa, pb = a["experiment"]["phenotype"], b["experiment"]["phenotype"]
        if (pa["metabolite_level"], pa["n_replicates"]) != (
            pb["metabolite_level"],
            pb["n_replicates"],
        ):
            level_diff += 1
    return {
        "records_compared": len(old),
        "genotype_differs": genotype_diff,
        "metabolite_level_or_n_differs": level_diff,
    }


#: A growth temperature in this OCR or SI text: a degree sign, a LaTeX ``\\circ``, the
#: word "temperature", or a number followed by "C" (``30C``, ``30 C``).
_TEMPERATURE = re.compile(
    r"\u00b0|\u00ba|\\circ|temperature|\b\d{2}\s?C\b", re.IGNORECASE
)


def _xml_text(data: bytes) -> str:
    """Plain text of an Office XML part (tags removed)."""
    return re.sub(r"<[^>]+>", " ", data.decode("utf-8"))


def temperature_mentions(library_dir: str) -> dict[str, Any]:
    """Temperature-shaped matches in ``paper.md`` and in each SI document's text."""
    paper = osp.join(library_dir, "paper.md")
    data = open(paper, "rb").read()
    assert hashlib.sha256(data).hexdigest() == PAPER_MD_SHA256
    paper_hits = [
        [n, m.group(0)]
        for n, line in enumerate(data.decode("utf-8").split("\n"), 1)
        for m in _TEMPERATURE.finditer(line)
    ]
    si_zip = osp.join(library_dir, "si", "gkad656_supplemental_files.zip")
    si_hits: dict[str, list[str]] = {}
    with zipfile.ZipFile(si_zip) as outer:
        for name in outer.namelist():
            blob = outer.read(name)
            if name.endswith(".pdf"):
                text = "\n".join(
                    page.extract_text() for page in PdfReader(io.BytesIO(blob)).pages
                )
            elif name.endswith((".docx", ".pptx")):
                with zipfile.ZipFile(io.BytesIO(blob)) as office:
                    text = " ".join(
                        _xml_text(office.read(part))
                        for part in office.namelist()
                        if part.startswith(("word/document", "ppt/slides/slide"))
                        and part.endswith(".xml")
                    )
            else:
                continue
            si_hits[name] = [
                text[max(0, m.start() - 60) : m.end() + 20].replace("\n", " ")
                for m in _TEMPERATURE.finditer(text)
            ]
    return {
        "paper_md_sha256": PAPER_MD_SHA256,
        "paper_md_matches": paper_hits,
        "si_zip_sha256": hashlib.sha256(open(si_zip, "rb").read()).hexdigest(),
        "si_matches_by_document": si_hits,
    }


def main() -> None:
    """Build the scratch store, compare it with the dev store, write the JSON."""
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scratch-root", required=True)
    args = parser.parse_args()
    data_root = os.environ["DATA_ROOT"]
    dev_root = osp.join(data_root, SLUG)
    scratch_root = osp.join(args.scratch_root, "betaxanthin_cachera2023")
    os.makedirs(osp.join(scratch_root, "raw"), exist_ok=True)
    dest = osp.join(scratch_root, "raw", "GA1_2_4_6.csv")
    if not osp.exists(dest):
        shutil.copy(osp.join(dev_root, "raw", "GA1_2_4_6.csv"), dest)
    genome = SCerevisiaeGenome(
        genome_root=osp.join(data_root, "data/sgd/genome"),
        go_root=osp.join(data_root, "data/go"),
        overwrite=False,
    )
    BetaxanthinCachera2023Dataset(root=scratch_root, genome=genome)

    dev = read_records(osp.join(dev_root, "processed"))
    scratch = read_records(osp.join(scratch_root, "processed"))
    out = {
        "issue": 509,
        "dev_store": f"$DATA_ROOT/{SLUG}/processed",
        "scratch_store": osp.join(scratch_root, "processed"),
        "before_dev": tally(dev),
        "after_scratch": tally(scratch),
        "outside_environment": unchanged_outside_environment(dev, scratch),
        "temperature_search": temperature_mentions(
            osp.join(data_root, "torchcell-library", CITATION_KEY)
        ),
    }
    os.makedirs(osp.dirname(RESULTS), exist_ok=True)
    with open(RESULTS, "w") as fh:
        json.dump(out, fh, indent=2)
        fh.write("\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
