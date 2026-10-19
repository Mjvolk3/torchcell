# experiments/044-organism-similarity/scripts/01_organisms.py
# [[experiments.044-organism-similarity.scripts.01_organisms]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/044-organism-similarity/scripts/01_organisms.py

"""Resolve, retrieve and pin one annotated proteome per plan-table organism.

Reads ``conf/organisms.yaml`` (34 organisms, plan rows 1 and 3-35) and writes the
pydantic manifest ``results/organisms.json``:

- ``source: tier``: the protein FASTA is read through
  ``torchcell.sequence.genome.registry.resolve`` (sha256 verified), never re-downloaded;
  for the bacterial sets and our P. kudriavzevii SD108 deposit (``pkudriavzevii_SD108_v1``,
  the release repository at commit 8ca6a106) the GFF of the same set is resolved too, so
  02_proteome_prep.py can map proteins to genes.
- ``source: ncbi``: ``datasets summary genome taxon "<taxon_query>" --reference`` is
  recorded beside the chosen accession (the yaml says why they differ when they do), then
  ``datasets download genome accession <acc> --include protein,gff3`` into
  ``$DATA_ROOT/data/organism_similarity/ncbi/<key>/``; the extracted ``protein.faa`` and
  ``genomic.gff`` are hashed. An assembly whose package carries no protein FASTA is
  recorded as absent with that reason; nothing is substituted.
- ``source: absent``: kept with its reason (cladogram only).

Run from the repo root with the torchcell env (pydantic + the registry); the ``datasets``
binary is called from the orgsim env by absolute path.
"""

from __future__ import annotations

import datetime as dt
import gzip
import json
import os
import os.path as osp
import subprocess
import sys
import zipfile

import yaml

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))

from orgsim_manifest import (  # noqa: E402
    CONF_PATH,
    MANIFEST_PATH,
    ORGSIM_BIN,
    RESULTS_DIR,
    WORK_DIR,
    FileRecord,
    OrganismManifest,
    OrganismRecord,
    ProteomeSource,
    sha256_file,
    write_manifest,
)

from torchcell.sequence.genome.registry import (  # noqa: E402
    load_genome_manifest,
    resolve,
)

DATASETS = osp.join(ORGSIM_BIN, "datasets")
NCBI_DIR = osp.join(WORK_DIR, "ncbi")
API = "https://api.ncbi.nlm.nih.gov/datasets/v2/genome/accession"

#: The GFF beside each tier protein FASTA (same release, so protein ids match).
TIER_GFF = {
    "ecoli_K12_MG1655_ASM584v2": "GCA_000005845.2_ASM584v2_genomic.gff.gz",
    "pputida_KT2440_ASM756v2": "GCA_000007565.2_ASM756v2_genomic.gff.gz",
    "pkudriavzevii_SD108_v1": "SD108_genome_v1.gff3",
}


def now() -> str:
    """UTC timestamp."""
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


def run(cmd: list[str]) -> str:
    """Run a command, return stdout; a failure raises with stderr."""
    out = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if out.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} failed:\n{out.stderr}")
    return out.stdout


def file_record(path: str) -> FileRecord:
    """sha256 + size of a file."""
    return FileRecord(path=path, sha256=sha256_file(path), bytes=os.path.getsize(path))


def count_fasta(path: str) -> int:
    """Number of records in a (possibly gzipped) FASTA."""
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as fh:
        return sum(1 for line in fh if line.startswith(">"))


def summary_lines(args: list[str]) -> list[dict]:
    """``datasets summary genome ... --as-json-lines`` parsed."""
    out = run([DATASETS, "summary", "genome", *args, "--as-json-lines"])
    return [json.loads(line) for line in out.splitlines() if line.strip()]


def strain_of(report: dict) -> str | None:
    """Strain or isolate name of an assembly report."""
    names = report["organism"].get("infraspecific_names") or {}
    return names.get("strain") or names.get("isolate")


def ncbi_record(entry: dict, datasets_version: str) -> OrganismRecord:
    """Download and pin one NCBI assembly."""
    key = entry["key"]
    acc = entry["accession"]
    ref = summary_lines(["taxon", entry["taxon_query"], "--reference"])
    reference_accession = ref[0]["accession"] if ref else None
    rep = summary_lines(["accession", acc])
    assert len(rep) == 1, f"{key}: {acc} returned {len(rep)} reports"
    report = rep[0]
    ann = report.get("annotation_info") or {}
    out_dir = osp.join(NCBI_DIR, key)
    os.makedirs(out_dir, exist_ok=True)
    zip_path = osp.join(out_dir, f"{acc}.zip")
    cmd = [
        DATASETS,
        "download",
        "genome",
        "accession",
        acc,
        "--include",
        "protein,gff3",
        "--filename",
        zip_path,
    ]
    retrieval_command = " ".join(
        [
            "datasets",
            *cmd[1:-1],
            f"$DATA_ROOT/data/organism_similarity/ncbi/{key}/{acc}.zip",
        ]
    )
    if not osp.isfile(zip_path):
        print(f"  downloading {acc} ...", flush=True)
        run(cmd)
    retrieved_at = dt.datetime.fromtimestamp(
        os.path.getmtime(zip_path), dt.UTC
    ).isoformat(timespec="seconds")
    extract_dir = osp.join(out_dir, "extracted")
    with zipfile.ZipFile(zip_path) as zf:
        members = zf.namelist()
        zf.extractall(extract_dir)
    protein = osp.join(extract_dir, "ncbi_dataset", "data", acc, "protein.faa")
    gff = osp.join(extract_dir, "ncbi_dataset", "data", acc, "genomic.gff")
    common = dict(
        key=key,
        row=entry["row"],
        display=entry["display"],
        organism_name=report["organism"]["organism_name"],
        taxid=int(report["organism"]["tax_id"]),
        strain=strain_of(report),
        preferred_strain=str(entry["preferred_strain"]),
        groups=entry["groups"],
        justification=entry["justification"],
        reference_accession=reference_accession,
        accession_choice=entry["accession_choice"],
    )
    if not osp.isfile(protein):
        return OrganismRecord(
            **common,
            proteome=None,
            absence_reason=(
                f"{acc} package has no protein.faa (members: {sorted(members)}); "
                "no annotated proteome, so the organism stays on the cladogram only."
            ),
        )
    assert osp.isfile(gff), f"{key}: {acc} has protein.faa but no genomic.gff"
    src = ProteomeSource(
        source="ncbi",
        accession=acc,
        assembly_name=report["assembly_info"]["assembly_name"],
        annotation_provider=ann.get("provider"),
        source_url=f"{API}/{acc}/download?include_annotation_type=PROT_FASTA,GENOME_GFF",
        retrieval_method="ncbi_datasets_cli",
        retrieval_command=retrieval_command,
        datasets_version=datasets_version,
        retrieved_at=retrieved_at,
        protein_fasta=file_record(protein),
        gff=file_record(gff),
        n_proteins_raw=count_fasta(protein),
    )
    return OrganismRecord(**common, proteome=src)


def tier_record(entry: dict) -> OrganismRecord:
    """Pin a tier-sourced proteome through the registry."""
    tset = entry["tier_set"]
    gm = load_genome_manifest(tset)
    protein = resolve(tset, entry["tier_protein"])
    rec = gm.record(entry["tier_protein"])
    gff = None
    if tset in TIER_GFF:
        gff = file_record(resolve(tset, TIER_GFF[tset]))
    src = ProteomeSource(
        source="tier",
        accession=entry["tier_protein"].split("_ASM")[0]
        if entry["tier_protein"].startswith("GC")
        else None,
        tier_set=tset,
        source_url=rec.retrieval.source_url if rec.retrieval else gm.source_url,
        retrieval_method="genomes_tier_resolve",
        retrieval_command=(
            f"torchcell.sequence.genome.registry.resolve({tset!r}, "
            f"{entry['tier_protein']!r})"
        ),
        retrieved_at=now(),
        protein_fasta=file_record(protein),
        gff=gff,
        n_proteins_raw=count_fasta(protein),
    )
    assert src.protein_fasta.sha256 == rec.sha256
    return OrganismRecord(
        key=entry["key"],
        row=entry["row"],
        display=entry["display"],
        organism_name=gm.organism,
        taxid=int(entry["taxid"]),
        strain=gm.strain_or_population,
        preferred_strain=str(entry["preferred_strain"]),
        groups=entry["groups"],
        justification=entry["justification"],
        accession_choice=f"genomes tier set {tset}",
        proteome=src,
    )


def absent_record(entry: dict) -> OrganismRecord:
    """An organism with no annotated proteome."""
    ref = summary_lines(["taxon", entry["taxon_query"], "--reference"])
    return OrganismRecord(
        key=entry["key"],
        row=entry["row"],
        display=entry["display"],
        organism_name=entry["taxon_query"],
        taxid=int(entry["taxid"]),
        strain=None,
        preferred_strain=str(entry["preferred_strain"]),
        groups=entry["groups"],
        justification=entry["justification"],
        reference_accession=ref[0]["accession"] if ref else None,
        proteome=None,
        absence_reason=entry["absence_reason"],
    )


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(CONF_PATH) as fh:
        conf = yaml.safe_load(fh)
    entries = conf["organisms"]
    assert len(entries) == 34, len(entries)
    datasets_version = run([DATASETS, "--version"]).strip().split()[-1]
    records: list[OrganismRecord] = []
    for entry in entries:
        print(f"[{entry['row']:>2}] {entry['key']} ({entry['source']})", flush=True)
        if entry["source"] == "ncbi":
            rec = ncbi_record(entry, datasets_version)
        elif entry["source"] == "tier":
            rec = tier_record(entry)
        elif entry["source"] == "absent":
            rec = absent_record(entry)
        else:
            raise ValueError(entry["source"])
        n = rec.proteome.n_proteins_raw if rec.proteome else None
        print(
            f"     taxid {rec.taxid} | {rec.organism_name} | proteins {n}", flush=True
        )
        records.append(rec)
    manifest = OrganismManifest(
        created_at=now(), datasets_version=datasets_version, organisms=records
    )
    write_manifest(manifest)
    print(
        f"wrote {MANIFEST_PATH}: {len(manifest.present())} proteomes, "
        f"{len(records) - len(manifest.present())} absent"
    )


if __name__ == "__main__":
    main()
