# scripts/provision_construct_sets.py
# [[scripts.provision_construct_sets]]
# https://github.com/Mjvolk3/torchcell/tree/main/scripts/provision_construct_sets.py

r"""Deposit a construct set's index over sequence the library tier already holds.

The construct-store counterpart of ``scripts/provision_yeast_genomes.py``. It deposits
no bytes: the sequence is already in a tier (for Cachera, the 17 GenBank maps under
``torchcell-library/cacheraCRISPAHighthroughputMethod2023/si/plasmids/``, each pinned in
that key's manifest), and what is written is the index at
``$DATA_ROOT/torchcell-constructs/<construct_set>/manifest.json``.

First set, ``cachera2023_cri_spa_plasmids``:

- **Per-file retrieval, read from our own pinned record.** Each of the 17 files gets a
  ``RetrievalRecord`` whose method, source URL, recipe and date come from
  ``si/gkad656_supplemental_files.provenance.json``, itself listed and sha256-pinned in
  the key's manifest and pinned again in each record's params. Nothing is typed into
  this file that is not read from a hashed artifact: the OUP archive is CloudFront
  signed, so its retrieval is ``manual_browser``, and the honest anchor is the container
  the recipe produced, not a URL this script could pretend to fetch.
- **The chain is verified, not asserted.** Step (b) re-extracts
  ``plasmid sequences.zip`` from the pinned container in memory and checks all 17
  members against the sha256s the key manifest pins. That is what earns
  ``provenance_complete=True``: a rebuild reads the deposited container and gets these
  exact bytes back. Measured 2026-10-15: 17 of 17 reproduce.
- **Every map is parsed, and a refusal is recorded by name.** 6 of 17 index as contigs;
  11 of 17 are ``BlockedConstruct`` rows, each carrying the parser's verbatim refusal:
  one origin-wrapping location per map (an ``AmpR`` CDS or the ``pUC\\ori``), which
  Biopython rewrites into a whole-plasmid envelope. ``pBTX002.gb`` (the paper's pBTX2)
  is among them, so the 8,104 bp XII-5 insert is NOT indexable until a span holds parts.
  Flattening it instead would have stored the whole 12,096 bp plasmid as the origin.
- **Each contig's name is quoted from the paper**, never from the LOCUS line, which for
  two of these files names a different replicon than the file does.

Steps, each halting the run before anything is written:

  (a) read: the key manifest, the pinned retrieval record, and the paper's sha256;
  (b) verify: re-extract the container and match all 17 member sha256s;
  (c) parse: every map through the hardened parser, splitting contigs from blocked maps;
  (d) manifest: one ``ConstructManifest``, printed; ``--dry-run`` stops here;
  (e) deposit: ``deposit_construct_set`` (which resolves and re-hashes every contig and
      refuses an existing index), then ``verify_construct_set``.

Usage::

    PYTHONPATH=$WT python scripts/provision_construct_sets.py --dry-run
    PYTHONPATH=$WT python scripts/provision_construct_sets.py
    PYTHONPATH=$WT python scripts/provision_construct_sets.py \
        --set cachera2023_cri_spa_plasmids
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import os.path as osp
import sys
import zipfile

from pydantic import BaseModel, ConfigDict, Field

from torchcell.artifacts.ref import ArtifactRef
from torchcell.artifacts.tiers import key_dir, load_manifest
from torchcell.literature.manifest import (
    ROLE_SI_DATA,
    ArtifactRecord,
    RetrievalMethod,
    RetrievalRecord,
)
from torchcell.sequence.construct.registry import (
    CACHERA2023_CRI_SPA_PLASMIDS,
    BlockedConstruct,
    BlockedReason,
    ConstructContig,
    ConstructKind,
    ConstructManifest,
    SequenceStatus,
    deposit_construct_set,
    parse_contig_features,
    utc_now,
    verify_construct_set,
)
from torchcell.sequence.plasmid import GenBankLocationError
from torchcell.verification.report import Provenance
from torchcell.verification.sourced import SourcedValue

CACHERA_KEY = "cacheraCRISPAHighthroughputMethod2023"
CACHERA_CONTAINER = "si/gkad656_supplemental_files.zip"
CACHERA_RETRIEVAL_RECORD = "si/gkad656_supplemental_files.provenance.json"
CACHERA_INNER_ZIP = "plasmid sequences.zip"
PLASMID_PREFIX = "si/plasmids/"
PAPER_OCR = "paper.md"


class ConstructSpec(BaseModel):
    """One construct set: where its bytes already live and how each contig is named."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    construct_set: str
    citation_key: str
    source: str
    sequence_status: SequenceStatus
    tier: str = Field(description="the artifact tier holding the bytes")
    path_prefix: str = Field(description="the directory within the key holding the maps")
    container: str = Field(description="the archive the maps were extracted from")
    retrieval_record: str = Field(description="our pinned retrieval record, within the key")
    inner_zip: str = Field(description="the member of the container holding the maps")
    kind_by_path: dict[str, ConstructKind]
    #: Path -> the verbatim sentence of the paper that names that construct. The
    #: SOURCE's name for the replicon lives here, not in the LOCUS line.
    name_quote_by_path: dict[str, str]
    notes: str


CACHERA_SPEC = ConstructSpec(
    construct_set=CACHERA2023_CRI_SPA_PLASMIDS,
    citation_key=CACHERA_KEY,
    source="Cachera et al. 2023 supplementary (NAR gkad656)",
    sequence_status=SequenceStatus.design,
    tier="library",
    path_prefix=PLASMID_PREFIX,
    container=CACHERA_CONTAINER,
    retrieval_record=CACHERA_RETRIEVAL_RECORD,
    inner_zip=CACHERA_INNER_ZIP,
    kind_by_path={
        "pBTX001(ante113).gb": ConstructKind.integration_vector,
        "pBTX002.gb": ConstructKind.integration_vector,
        "pCCM023.gb": ConstructKind.plasmid,
        "pHO-ADE2(pHO24).gbk": ConstructKind.plasmid,
        "pHO22 (gRNA(ChrXV_dwADE2)-NatMX).gbk": ConstructKind.plasmid,
        "pHO25 (SNR52p-gRNA(dw XII-5)-SUP4t_natMX).gbk": ConstructKind.plasmid,
        "pHO29 - pCfB3050-hphNT1 (SNR52p-gRNA(XII-5)-SUP4t_hygR).gbk": ConstructKind.plasmid,
        "pHO8 (pCfB257(X-3-LoxP-KlLEU2)-Cas9(Hs)).gbk": ConstructKind.plasmid,
        "pbtx013.gb": ConstructKind.plasmid,
        "pbtx014.gb": ConstructKind.plasmid,
        "pbtx016.gb": ConstructKind.plasmid,
        "pbtx017.gb": ConstructKind.plasmid,
        "pbtx019.gb": ConstructKind.plasmid,
        "pbtx020.gb": ConstructKind.plasmid,
        "pbtx021.gb": ConstructKind.plasmid,
        "pbtx022.gb": ConstructKind.plasmid,
        "pbtx023.gb": ConstructKind.plasmid,
    },
    name_quote_by_path={
        "pBTX001(ante113).gb": (
            "the integrative plasmid pBTX1 harboring the betaxanthin-synthesizing "
            "genes CYP76AD1 and DOD"
        ),
        "pHO-ADE2(pHO24).gbk": "the NTC selectable ADE2 CRI-SPA vector (pHO24)",
        "pHO22 (gRNA(ChrXV_dwADE2)-NatMX).gbk": (
            "transformation with sgRNA plasmid pHO22"
        ),
        "pHO25 (SNR52p-gRNA(dw XII-5)-SUP4t_natMX).gbk": (
            "co-transforation with sgRNA plasmid pHO25"
        ),
        "pHO29 - pCfB3050-hphNT1 (SNR52p-gRNA(XII-5)-SUP4t_hygR).gbk": (
            "a hphNT1 selectable CRI-SPA plasmid (pHO29)"
        ),
        "pHO8 (pCfB257(X-3-LoxP-KlLEU2)-Cas9(Hs)).gbk": (
            "The gene-targeting substrate was liberated from pHO8 by digestion with "
            "NotI"
        ),
    },
    notes=(
        "Index over the 17 GenBank maps the paper released, which sit in the library "
        "tier under this citation key. The bytes are the DESIGN maps the authors "
        "deposited, not a resequencing of the built plasmids, hence sequence_status "
        "design. 6 of 17 index as contigs; the other 11 each carry one "
        "origin-wrapping location (an AmpR CDS, or the pUC ori of pBTX002.gb) which a "
        "single-range span cannot hold, so they are listed as blocked with the "
        "parser's refusal rather than stored as a whole-plasmid envelope. The paper's "
        "pBTX2 is among the blocked maps, so the XII-5 insert is not indexable yet."
    ),
)

SPECS: dict[str, ConstructSpec] = {CACHERA2023_CRI_SPA_PLASMIDS: CACHERA_SPEC}

#: The name a construct's source gives it, derived from the quote by the loader stating
#: it. A quote is a verbatim sentence of the paper; the NAME is the token in it that is
#: this construct's, which the spec states per path (the key of this map).
NAME_BY_PATH: dict[str, str] = {
    "pBTX001(ante113).gb": "pBTX1",
    "pHO-ADE2(pHO24).gbk": "pHO24",
    "pHO22 (gRNA(ChrXV_dwADE2)-NatMX).gbk": "pHO22",
    "pHO25 (SNR52p-gRNA(dw XII-5)-SUP4t_natMX).gbk": "pHO25",
    "pHO29 - pCfB3050-hphNT1 (SNR52p-gRNA(XII-5)-SUP4t_hygR).gbk": "pHO29",
    "pHO8 (pCfB257(X-3-LoxP-KlLEU2)-Cas9(Hs)).gbk": "pHO8",
}


def _data_root() -> str:
    from dotenv import load_dotenv

    load_dotenv()
    return os.environ["DATA_ROOT"]


def _verify_container_chain(
    spec: ConstructSpec, pinned: dict[str, str], data_root: str
) -> str:
    """Re-extract the maps from the pinned container; raise unless all 17 reproduce.

    Returns the inner archive's sha256, which the retrieval records pin so the chain is
    traceable one step at a time rather than only end to end.
    """
    container = key_dir(spec.tier, spec.citation_key, data_root) / spec.container  # type: ignore[arg-type]
    with zipfile.ZipFile(container) as outer:
        inner_bytes = outer.read(spec.inner_zip)
    inner_sha = hashlib.sha256(inner_bytes).hexdigest()
    extracted: dict[str, str] = {}
    with zipfile.ZipFile(io.BytesIO(inner_bytes)) as inner:
        for info in inner.infolist():
            base = osp.basename(info.filename)
            if info.filename.endswith("/") or "__MACOSX" in info.filename:
                continue
            if base.startswith("."):
                continue
            extracted[base] = hashlib.sha256(inner.read(info)).hexdigest()
    missing = sorted(set(pinned) - set(extracted))
    mismatched = sorted(
        base for base, sha in pinned.items() if extracted.get(base) not in (None, sha)
    )
    if missing or mismatched:
        raise RuntimeError(
            f"{spec.construct_set}: the recorded recipe does not reproduce the maps "
            f"from {spec.container} (missing {missing}, sha256 mismatch {mismatched}); "
            "the index is not deposited and provenance_complete cannot be earned"
        )
    print(
        f"(b) chain verified: all {len(pinned)} maps reproduce from {spec.container} "
        f"-> {spec.inner_zip} (sha256 {inner_sha[:12]})"
    )
    return inner_sha


def _retrieval_for(
    record: dict[str, object],
    *,
    container_sha: str,
    inner_sha: str,
    recipe_sha: str,
    recipe_path: str,
    file_sha: str,
) -> RetrievalRecord:
    """One file's retrieval record, every field read from the pinned recipe artifact."""
    return RetrievalRecord(
        method=RetrievalMethod(str(record["retrieval_method"])),
        source_url=str(record["source_url"]),
        retriever="manual",
        params={
            "retrieval_command": str(record["retrieval_command"]),
            "direct_url": str(record["direct_url"]),
            "container": CACHERA_CONTAINER,
            "container_sha256": container_sha,
            "inner_archive": CACHERA_INNER_ZIP,
            "inner_archive_sha256": inner_sha,
            "recipe_artifact": recipe_path,
            "recipe_artifact_sha256": recipe_sha,
        },
        sha256=file_sha,
        retrieved_at=str(record["retrieved_at"]),
    )


def build_manifest(spec: ConstructSpec, data_root: str) -> ConstructManifest:
    """Steps (a) to (d): read, verify the chain, parse every map, build the index."""
    key_manifest = load_manifest(spec.tier, spec.citation_key, data_root)  # type: ignore[arg-type]
    maps = {
        record.path: record
        for record in key_manifest.files
        if record.path.startswith(spec.path_prefix)
    }
    if sorted(osp.basename(path) for path in maps) != sorted(spec.kind_by_path):
        raise RuntimeError(
            f"{spec.construct_set}: the key manifest lists "
            f"{sorted(osp.basename(path) for path in maps)}, the spec names "
            f"{sorted(spec.kind_by_path)}"
        )
    container_sha = key_manifest.files[
        [record.path for record in key_manifest.files].index(spec.container)
    ].sha256
    recipe_record = key_manifest.files[
        [record.path for record in key_manifest.files].index(spec.retrieval_record)
    ]
    paper_record = key_manifest.files[
        [record.path for record in key_manifest.files].index(PAPER_OCR)
    ]
    print(f"(a) {len(maps)} maps listed in {spec.tier}/{spec.citation_key}")
    inner_sha = _verify_container_chain(
        spec,
        {osp.basename(path): record.sha256 for path, record in maps.items()},
        data_root,
    )
    key_root = key_dir(spec.tier, spec.citation_key, data_root)  # type: ignore[arg-type]
    recipe = json.loads((key_root / spec.retrieval_record).read_text(encoding="utf-8"))
    paper_text = (key_root / PAPER_OCR).read_text(encoding="utf-8")

    files: list[ArtifactRecord] = []
    contigs: list[ConstructContig] = []
    blocked: list[BlockedConstruct] = []
    for path, record in sorted(maps.items()):
        base = osp.basename(path)
        files.append(
            ArtifactRecord(
                path=path,
                role=ROLE_SI_DATA,
                bytes=record.bytes,
                sha256=record.sha256,
                retrieval=_retrieval_for(
                    recipe,
                    container_sha=container_sha,
                    inner_sha=inner_sha,
                    recipe_sha=recipe_record.sha256,
                    recipe_path=spec.retrieval_record,
                    file_sha=record.sha256,
                ),
            )
        )
        ref = ArtifactRef(
            tier=spec.tier,  # type: ignore[arg-type]
            key=spec.citation_key,
            path=path,
            sha256=record.sha256,
            bytes=record.bytes,
        )
        try:
            component, features = parse_contig_features(
                key_root / path, spec.citation_key
            )
        except GenBankLocationError as refusal:
            blocked.append(
                BlockedConstruct(
                    path=path,
                    reason=BlockedReason.origin_wrapping_location,
                    detail=str(refusal),
                )
            )
            continue
        quote = spec.name_quote_by_path.get(base)
        if quote is None:
            raise RuntimeError(
                f"{spec.construct_set}: {base} parsed but the spec quotes no name for "
                "it; a construct's name is quoted from the paper, never taken from the "
                f"LOCUS line ({component.identity!r}) or the filename"
            )
        if quote not in paper_text:
            raise RuntimeError(
                f"{spec.construct_set}: the quote for {base} is not a verbatim "
                f"substring of {PAPER_OCR}: {quote!r}"
            )
        contigs.append(
            ConstructContig(
                sha256=record.sha256,
                ref=ref,
                record_name=component.identity,
                topology=component.topology,  # type: ignore[arg-type]
                length=component.length,
                kind=spec.kind_by_path[base],
                features=features,
                source_name=SourcedValue(
                    value=NAME_BY_PATH[base],
                    provenance=Provenance(
                        source_uri=PAPER_OCR,
                        citation_key=spec.citation_key,
                        sha256=paper_record.sha256,
                        method="mineru-ocr",
                    ),
                    quote=quote,
                    note="the name the paper gives this construct; the GenBank LOCUS "
                    f"line of this file reads {component.identity!r}",
                ),
            )
        )
    print(
        f"(c) parsed: {len(contigs)} contigs, {len(blocked)} blocked "
        f"({', '.join(osp.basename(entry.path) for entry in blocked)})"
    )
    return ConstructManifest(
        construct_set=spec.construct_set,
        source=spec.source,
        sequence_status=spec.sequence_status,
        citation_key=spec.citation_key,
        files=files,
        contigs=contigs,
        blocked=blocked,
        provenance_complete=True,
        created_at=utc_now(),
        notes=spec.notes,
    )


def main(argv: list[str] | None = None) -> int:
    """Deposit one construct set's index; ``--dry-run`` prints it and writes nothing."""
    parser = argparse.ArgumentParser(prog="provision_construct_sets")
    parser.add_argument("--set", dest="sets", action="append", choices=sorted(SPECS))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--data-root", default=None)
    args = parser.parse_args(argv)
    data_root = args.data_root or _data_root()
    for name in args.sets or sorted(SPECS):
        spec = SPECS[name]
        print(f"=== {name} ===")
        manifest = build_manifest(spec, data_root)
        print(
            f"(d) index: {len(manifest.files)} files, {len(manifest.contigs)} contigs, "
            f"{len(manifest.blocked)} blocked, provenance_complete="
            f"{manifest.provenance_complete}"
        )
        for contig in manifest.contigs:
            name_value = (
                None if contig.source_name is None else contig.source_name.value
            )
            print(
                f"    {contig.handle}  {name_value!r:10} LOCUS {contig.record_name!r} "
                f"{contig.length} bp  {len(contig.features)} features"
            )
        if args.dry_run:
            print("(e) --dry-run: nothing written")
            continue
        path = deposit_construct_set(manifest, data_root)
        measured = verify_construct_set(name, data_root)
        print(f"(e) deposited {path}; verified {len(measured)} contigs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
