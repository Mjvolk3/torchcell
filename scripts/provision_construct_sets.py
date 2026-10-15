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
- **Every map is parsed, and a refusal is recorded by name.** Since 2026-10-15 a span
  holds the parts of a compound location, so **17 of 17 index and 0 are blocked**. The
  one offending feature of each of the 11 formerly blocked maps is an interval across
  the origin of a circular replicon (``pBTX002.gb``'s ``pUC\\ori`` at ``11559..128``,
  and an ``AmpR`` CDS at ``complement(4892..858)`` in the other ten), now held as the
  two parts it wrapped into: 666 bp of origin and 861 bp of beta-lactamase, not the
  whole 12,096 bp or 4,894 bp plasmid an envelope would have stored. A map the parser
  still refuses becomes a ``BlockedConstruct`` carrying the typed reason the parser gave
  (``GenBankLocationError.refusal``), not one hardcoded here.
- **Each contig's name is quoted from the source**, never from the LOCUS line, which for
  two of these files names a different replicon than the file does. Eight names come
  from ``paper.md``; the nine ``pBTX0xx`` gene-targeting vectors the paper body never
  names come from SUPPLEMENTARY TABLE S2 in ``Supplementary Material.docx``, a member of
  the same pinned container, extracted and re-hashed in memory on every run.

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
import re
import sys
import zipfile
from pathlib import Path

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
CACHERA_SI_DOCX = "Supplementary Material.docx"
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
    path_prefix: str = Field(
        description="the directory within the key holding the maps"
    )
    container: str = Field(description="the archive the maps were extracted from")
    retrieval_record: str = Field(
        description="our pinned retrieval record, within the key"
    )
    inner_zip: str = Field(description="the member of the container holding the maps")
    kind_by_path: dict[str, ConstructKind]
    #: Path -> the verbatim sentence of the paper that names that construct. The
    #: SOURCE's name for the replicon lives here, not in the LOCUS line.
    name_quote_by_path: dict[str, str]
    #: Path -> the verbatim row of the supplementary plasmid table that names that
    #: construct, for the constructs the paper body never names. The table is a member
    #: of the pinned container, extracted and re-hashed on every run, so the quote is
    #: checked against hashed bytes exactly as the paper quotes are.
    si_name_quote_by_path: dict[str, str] = Field(default_factory=dict)
    si_member: str | None = Field(
        default=None, description="the container member holding that table"
    )
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
    si_member=CACHERA_SI_DOCX,
    si_name_quote_by_path={
        "pbtx013.gb": "pBTX013\nGene::Up KanMX Gene:Dwn Gene=GLR1",
        "pbtx014.gb": "pBTX014\nGene::Up KanMX Gene:Dwn Gene=HEM25",
        "pbtx016.gb": "pBTX016\nGene::Up KanMX Gene:Dwn Gene= PRO2",
        "pbtx017.gb": "pBTX017\nGene::Up KanMX Gene:Dwn Gene= VMA16",
        "pbtx019.gb": "pBTX019\nGene::Up KanMX Gene:Dwn Gene=GET4",
        "pbtx020.gb": "pBTX020\nGene::Up KanMX Gene:Dwn Gene=PRY1",
        "pbtx021.gb": "pBTX021\nGene::Up KanMX Gene:Dwn Gene=STV1",
        "pbtx022.gb": "pBTX022\nGene::Up KanMX Gene:Dwn Gene=YCR101C",
        "pbtx023.gb": "pBTX023\nGene::Up KanMX Gene:Dwn Gene= YLR271W",
    },
    name_quote_by_path={
        "pBTX001(ante113).gb": (
            "the integrative plasmid pBTX1 harboring the betaxanthin-synthesizing "
            "genes CYP76AD1 and DOD"
        ),
        "pBTX002.gb": (
            "The BY-Btx strain was made by transforming the NotI excised insert of "
            "pBTX2"
        ),
        "pCCM023.gb": (
            "inserted into the USER cloning site of pCCM023, a modified version "
            "plasmid pCfB2909 lacking XII5-up and XII5-down homology sequences"
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
        "design. All 17 index as contigs. 11 of them carry one feature whose interval "
        "crosses the origin of the circular replicon (the pUC ori of pBTX002.gb at "
        "11559..128; an AmpR CDS at complement(4892..858) in the other ten), held as "
        "the two parts it wrapped into with composition origin_spanning, which is a "
        "different claim from a two-part join and is checked against the contig's own "
        "length and topology. 570 of 570 features regenerate the location string their "
        "file wrote. Of the 17 names, 8 are quoted from paper.md and 9 from "
        "SUPPLEMENTARY TABLE S2 of Supplementary Material.docx inside the pinned "
        "container."
    ),
)

SPECS: dict[str, ConstructSpec] = {CACHERA2023_CRI_SPA_PLASMIDS: CACHERA_SPEC}

#: The name a construct's source gives it, derived from the quote by the loader stating
#: it. A quote is a verbatim sentence of the paper; the NAME is the token in it that is
#: this construct's, which the spec states per path (the key of this map).
NAME_BY_PATH: dict[str, str] = {
    "pBTX001(ante113).gb": "pBTX1",
    "pBTX002.gb": "pBTX2",
    "pCCM023.gb": "pCCM023",
    "pbtx013.gb": "pBTX013",
    "pbtx014.gb": "pBTX014",
    "pbtx016.gb": "pBTX016",
    "pbtx017.gb": "pBTX017",
    "pbtx019.gb": "pBTX019",
    "pbtx020.gb": "pBTX020",
    "pbtx021.gb": "pBTX021",
    "pbtx022.gb": "pBTX022",
    "pbtx023.gb": "pBTX023",
    "pHO-ADE2(pHO24).gbk": "pHO24",
    "pHO22 (gRNA(ChrXV_dwADE2)-NatMX).gbk": "pHO22",
    "pHO25 (SNR52p-gRNA(dw XII-5)-SUP4t_natMX).gbk": "pHO25",
    "pHO29 - pCfB3050-hphNT1 (SNR52p-gRNA(XII-5)-SUP4t_hygR).gbk": "pHO29",
    "pHO8 (pCfB257(X-3-LoxP-KlLEU2)-Cas9(Hs)).gbk": "pHO8",
}


#: How the supplementary plasmid table's text is read out of the .docx: Word's
#: ``word/document.xml``, one newline per paragraph (a table CELL is a paragraph), every
#: tag removed and nothing else changed. Recorded as the ``method`` of every quote taken
#: from it, because a verbatim claim against an extracted text is only checkable with
#: the extraction named. Non-breaking spaces are NOT normalized: the quote carries what
#: the file carries.
SI_TEXT_METHOD = "docx-document-xml-paragraph-text"


def _container_member_text(container: Path, member: str) -> tuple[str, str]:
    """A .docx member of the pinned container as (sha256, text), extracted in memory.

    The container is already sha256-pinned in the key manifest and re-read here, so a
    quote from the supplementary table rests on hashed bytes exactly as a quote from
    ``paper.md`` does. The member's own sha256 goes into the quote's provenance, so a
    revision of the SI is detected rather than silently followed.
    """
    with zipfile.ZipFile(container) as outer:
        payload = outer.read(member)
    with zipfile.ZipFile(io.BytesIO(payload)) as document:
        xml = document.read("word/document.xml").decode("utf-8")
    paragraphs = re.sub(r"<w:p[ >]", "\n<w:p ", xml)
    return hashlib.sha256(payload).hexdigest(), re.sub(r"<[^>]+>", "", paragraphs)


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
    si_sha, si_text = (
        _container_member_text(key_root / spec.container, spec.si_member)
        if spec.si_member is not None
        else ("", "")
    )
    if spec.si_member is not None:
        print(
            f"(a) {spec.si_member} extracted from the pinned container "
            f"(sha256 {si_sha[:12]}, {len(si_text)} characters of text)"
        )

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
                BlockedConstruct(path=path, reason=refusal.refusal, detail=str(refusal))
            )
            continue
        quote = spec.name_quote_by_path.get(base)
        si_quote = spec.si_name_quote_by_path.get(base)
        if quote is None and si_quote is None:
            raise RuntimeError(
                f"{spec.construct_set}: {base} parsed but the spec quotes no name for "
                "it; a construct's name is quoted from the paper or its supplementary "
                f"plasmid table, never taken from the LOCUS line "
                f"({component.identity!r}) or the filename"
            )
        if quote is not None and si_quote is not None:
            raise RuntimeError(
                f"{spec.construct_set}: {base} has both a paper quote and a "
                "supplementary-table quote; one artifact names a construct"
            )
        if quote is not None:
            if quote not in paper_text:
                raise RuntimeError(
                    f"{spec.construct_set}: the quote for {base} is not a verbatim "
                    f"substring of {PAPER_OCR}: {quote!r}"
                )
            name_provenance = Provenance(
                source_uri=PAPER_OCR,
                citation_key=spec.citation_key,
                sha256=paper_record.sha256,
                method="mineru-ocr",
            )
        else:
            assert si_quote is not None  # noqa: S101  # narrowed by the branches above
            quote = si_quote
            if quote not in si_text:
                raise RuntimeError(
                    f"{spec.construct_set}: the quote for {base} is not a verbatim "
                    f"substring of {spec.si_member}'s text: {quote!r}"
                )
            name_provenance = Provenance(
                source_uri=f"{spec.container}!{spec.si_member}",
                citation_key=spec.citation_key,
                sha256=si_sha,
                method=SI_TEXT_METHOD,
                page="SUPPLEMENTARY TABLE S2: Plasmids Used in this Study",
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
                    provenance=name_provenance,
                    quote=quote,
                    note="the name the source gives this construct; the GenBank LOCUS "
                    f"line of this file reads {component.identity!r}",
                ),
            )
        )
    print(
        f"(c) parsed: {len(contigs)} contigs, {len(blocked)} blocked "
        f"({', '.join(osp.basename(entry.path) for entry in blocked) or 'none'})"
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
