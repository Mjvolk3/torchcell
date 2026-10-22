# experiments/043-expression-proteome-morphology/scripts/fetch_addgene_127546.py
# [[experiments.043-expression-proteome-morphology.scripts.fetch_addgene_127546]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/043-expression-proteome-morphology/scripts/fetch_addgene_127546.py

r"""Bring Addgene 127546 (``yeast_DualReporter``) into the constructs tier.

The plasmid the Vaishnav 2022 loader names and does not hold: "cloned into the promoter
of a YFP gene within a CEN plasmid (Addgene: 127546)". It is the GPRA reporter de Boer
2020 deposited, which Vaishnav 2022 reuses, so its bytes are what the 110 bp released
``seq110`` sits inside and what any promoter window longer than 110 bp must read.

WHAT ADDGENE SERVES A SCRIPT, MEASURED 2026-10-21
-------------------------------------------------
``/api/get-sequence-file-collection/<id>/`` advertises a ``genbankUrl`` and a
``snapgeneUrl`` for each of the two deposited full sequences (251803, Addgene-verified;
244230, depositor). **Both 404.** Measured: every sibling viewer artifact in the SAME
CDN directory serves 200 (``-features.json``, ``-sequence-lines.json``, ``-map.svg``,
``-orfs.json``), and the ``.gbk``/``.dna`` return a 9-byte ``Not Found`` under a plain
client, a browser user-agent, a page Referer, and a cookie jar from the sequences page
alike; plasmid 12259's ``.gbk`` 404s the same way, so it is not specific to 127546. The
GenBank file is therefore recorded with a ``manual_browser`` recipe and is NOT the
artifact this script deposits.

What IS deposited, with a retrieval record each, is exactly what Addgene served:

- ``addgene/127546/plasmid.page.txt`` and ``sequences.page.txt`` (the STABLE text of the
  pages naming the plasmid,
  its 8,218 bp vector size, its five gene inserts and its two full sequences),
- ``addgene/127546/seq-<id>/collection.json`` (the API's own answer, which carries the
  ``genbankUrl`` this script then shows to 404),
- ``addgene/127546/seq-<id>/features.json`` (the feature table: 16 features with 1-based
  inclusive ranges and strands),
- ``addgene/127546/seq-<id>/sequence-lines.json`` (the viewer's per-line SVG, which is
  where the BASES are),
- ``addgene/127546/seq-<id>/map.svg``.

THE SEQUENCE IS RECONSTRUCTED, AND THE RECONSTRUCTION IS PROVED
---------------------------------------------------------------
``sequence-lines.json`` holds the sequence as SVG glyphs, so the bases are read out of
the render by coordinate, never by guess, and the read is checked four ways before
anything is written:

1. **Two complementary rows per line.** Each rendered line draws the top strand and the
   bottom strand as two rows of exactly ``end - begin + 1`` lowercase glyphs; the lower
   ``y`` is the top strand and the other row must be its base-for-base complement. A
   line with any other shape halts the run. (This is why a naive glyph scan is wrong:
   lines also carry UPPERCASE overlay glyphs for primers and translated ORFs.)
2. **Contiguity.** Line ``n`` must begin exactly where the sequence so far ends, so the
   137 lines tile 0..8217 with no gap and no overlap.
3. **The depositor's own length.** The plasmid page states "Total vector size (bp) 8218";
   the reconstruction is 8,218 bp.
4. **Every annotated CDS is a clean ORF.** URA3 (804 bp), yEVenus (717), mCherry (711),
   NrsR (573) and AmpR (861) each start ``ATG``, end on a stop codon and carry zero
   internal stops in frame, and their N-termini read ``MSKATY``, ``MSKGEE``, ``MVSKGE``,
   ``MGTTLD`` and ``MSIQHF``. Five independent frames agreeing is what makes a
   glyph-read sequence trustworthy.

The derived GenBank is therefore filed as a ``reconstruction``, with a
``ProcessingRecord`` naming this script and the sha256 of both inputs, and the construct
index's ``provenance_complete`` is **False**, because the contig's own bytes carry
processing provenance rather than a retrieval chain. ``sequence_status`` is ``design``:
Addgene serves the depositor's and its own verified map, not a resequencing of the tube
we would receive.

Usage::

    PYTHONPATH=$WT python experiments/043-expression-proteome-morphology/scripts/\
fetch_addgene_127546.py --dry-run
    PYTHONPATH=$WT python experiments/043-expression-proteome-morphology/scripts/\
fetch_addgene_127546.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import os.path as osp
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from torchcell.artifacts.ref import ArtifactRef
from torchcell.artifacts.tiers import key_dir, load_manifest, manifest_path
from torchcell.literature.manifest import (
    ROLE_RAW_DATA,
    ArtifactRecord,
    Manifest,
    ProcessingRecord,
    RetrievalMethod,
    RetrievalRecord,
    sha256_file,
)
from torchcell.sequence.construct.registry import (
    ConstructContig,
    ConstructKind,
    ConstructManifest,
    SequenceStatus,
    deposit_construct_set,
    parse_contig_features,
    utc_now,
    verify_construct_set,
)
from torchcell.verification.report import Provenance
from torchcell.verification.sourced import SourcedValue

CITATION_KEY = "vaishnavEvolutionEvolvabilityEngineering2022"
CONSTRUCT_SET = "vaishnav2022_addgene_127546"
PLASMID_ID = "127546"
TIER = "raw"
PREFIX = f"addgene/{PLASMID_ID}"
#: The two deposited full sequences, Addgene-verified first.
SEQUENCE_OBJECTS: tuple[str, ...] = ("251803", "244230")
#: The one the construct index's contig is built from: Addgene's own verified full map.
CANONICAL_SEQUENCE_OBJECT = "251803"
USER_AGENT = "Mozilla/5.0 (torchcell provenance fetch; contact via repository)"
#: The depositor's name for the plasmid and the vector size it states, both quoted from
#: the plasmid page's stable text, both checked as verbatim substrings of it.
NAME_QUOTE = (
    "A dual reporter vector modified from Sharon et al (Nature Biotech, 2012) "
    "containing Ura3 and Nat1 selectable markers, a constant background RFP "
    "(TEF2::mCherry), and a variable YFP (pTpA+MCS::yEVenus)."
)
SOURCE_NAME = "yeast_DualReporter"
STATED_LENGTH = 8218
STATED_LENGTH_QUOTE = "Total vector size (bp)"

#: The scaffold flanks the Vaishnav loader sources from the paper, repeated here so this
#: script's report is a check against the loader rather than a restatement of it.
SCAFFOLD_5PRIME = "TGCATTTTTTTCACATC"
SCAFFOLD_3PRIME = "GGTTACGGCTGTT"
#: The reporter whose start codon a promoter window ends at: Addgene gene/insert 4,
#: "Gene/Insert name yEVenus / Alt name YFP / Insert Size (bp) 717 / Promoter pTpA+MCS".
#: SnapGene's common-feature autoannotation labels the same 717 bp CDS ``yeGFP``, which
#: is a label, not the depositor's name for the gene.
REPORTER_FEATURE_LABEL = "yeGFP"
REPORTER_INSERT_BP = 717

RULER = re.compile(r'data-begin-bp="(\d+)" data-end-bp="(\d+)"')
GLYPH = re.compile(r'<text[^>]*?x="(-?\d+)"[^>]*?y="(-?\d+)"[^>]*?>([acgtn])</text>')
TAG = re.compile(r"<[^>]+>")
COMPLEMENT_LOWER = str.maketrans("acgtn", "tgcan")
COMPLEMENT = str.maketrans("ACGTN", "TGCAN")

_BASES = "TCAG"
_AMINO = "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"
CODON_TABLE: dict[str, str] = {
    b1 + b2 + b3: _AMINO[i * 16 + j * 4 + k]
    for i, b1 in enumerate(_BASES)
    for j, b2 in enumerate(_BASES)
    for k, b3 in enumerate(_BASES)
}

#: The GenBank feature type each Addgene feature name is emitted as. Grounded, not
#: guessed: the five CDS entries are the five features this script PROVES are clean ORFs,
#: and the rest take the type their name states. A name this table does not hold halts
#: the run rather than falling to ``misc_feature`` silently.
GENBANK_TYPE_BY_FEATURE: dict[str, str] = {
    "URA3": "CDS",
    "yeGFP": "CDS",
    "mCherry": "CDS",
    "NrsR": "CDS",
    "AmpR": "CDS",
    "natMX6": "gene",
    "URA3 promoter": "promoter",
    "TEF promoter": "promoter",
    "AmpR promoter": "promoter",
    "SP6 promoter": "promoter",
    "T7 promoter": "promoter",
    "ADH1 terminator": "terminator",
    "TEF terminator": "terminator",
    "ori": "rep_origin",
    "CEN/ARS": "rep_origin",
}
#: The features whose reading frame is proved before anything is written.
PROVEN_ORFS: tuple[tuple[str, int, str], ...] = (
    ("URA3", 804, "MSKATY"),
    ("yeGFP", 717, "MSKGEE"),
    ("mCherry", 711, "MVSKGE"),
    ("NrsR", 573, "MGTTLD"),
    ("AmpR", 861, "MSIQHF"),
)

#: An Addgene catalog page is NOT byte-stable, so its sha256 is not an anchor. MEASURED
#: on two fetches seconds apart: a fresh Django ``csrfmiddlewaretoken`` and ``hx-headers``
#: CSRF token, a per-request ``applicationTime`` in the telemetry script, rotating
#: Cloudflare email-obfuscation hashes, and -- the only one that survives tag stripping --
#: a different five-plasmid "Commonly requested with" list. What IS stable is the page's
#: own text with the scripts and that one block removed, so THAT is the artifact deposited
#: and pinned, with the HTML recorded as the fetch it came from rather than as a file
#: whose hash a rebuild could reproduce. The stability is PROVED in the run: each page is
#: fetched twice and the two extractions must hash identically or the run halts.
PAGE_TEXT_SUFFIX = ".page.txt"
ROTATING_BLOCK_HEADING = "Commonly requested with:"
#: The heading the rotating block runs up to; everything between the two is dropped.
ROTATING_BLOCK_END = "How was your experience today?"
SCRIPT = re.compile(r"<(script|style)\b.*?</\1>", re.S | re.I)

#: The recipe for the two files the CDN refuses a script. Recorded, not worked around.
MANUAL_GBK_RECIPE = (
    "Open https://www.addgene.org/127546/sequences/ in a logged-out desktop browser, "
    "expand 'Addgene-Verified Full Sequences (1)' (sequence object 251803) or "
    "'Depositor Full Sequences (1)' (244230), and use that section's GenBank / SnapGene "
    "download button. The URLs the page's own API advertises "
    "(/api/get-sequence-file-collection/<id>/ -> genbankUrl, snapgeneUrl) return HTTP "
    "404 to curl, to curl with a browser user-agent, to curl with a Referer from the "
    "sequences page and to curl with that page's cookie jar, while every sibling viewer "
    "artifact in the same directory returns 200; plasmid 12259 behaves identically, so "
    "the refusal is Addgene's, not this plasmid's. Deposit the downloaded .gbk beside "
    "these files and re-run with --genbank <path> to replace the reconstruction."
)


class AddgeneSpec(BaseModel):
    """One Addgene plasmid to bring into the constructs tier."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    plasmid_id: str
    construct_set: str
    citation_key: str
    tier: str
    prefix: str
    sequence_objects: tuple[str, ...]
    canonical_sequence_object: str
    source_name: str
    name_quote: str
    stated_length: int
    kind: ConstructKind
    sequence_status: SequenceStatus


SPEC = AddgeneSpec(
    plasmid_id=PLASMID_ID,
    construct_set=CONSTRUCT_SET,
    citation_key=CITATION_KEY,
    tier=TIER,
    prefix=PREFIX,
    sequence_objects=SEQUENCE_OBJECTS,
    canonical_sequence_object=CANONICAL_SEQUENCE_OBJECT,
    source_name=SOURCE_NAME,
    name_quote=NAME_QUOTE,
    stated_length=STATED_LENGTH,
    # The deposited bytes are a reconstruction from the viewer artifacts, because the
    # CDN refuses the depositor's GenBank file to a script. Never filed as a plasmid map
    # the source released.
    kind=ConstructKind.reconstruction,
    sequence_status=SequenceStatus.design,
)


class FetchedFile(BaseModel):
    """One retrieved artifact: where it went, what it hashes to, how it was fetched."""

    model_config = ConfigDict(extra="forbid")

    rel_path: str
    url: str
    command: str
    sha256: str
    bytes: int
    retrieved_at: str
    http_status: int
    #: For a page whose HTML is not byte-stable: the sha256 of each HTML body fetched,
    #: recorded as data. Empty for an artifact that IS the retrieved bytes.
    html_fetch_sha256: list[str] = Field(default_factory=list)


class AddgeneFeature(BaseModel):
    """One row of Addgene's feature table: name, 1-based inclusive range, strand."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    begin: int
    end: int
    reverse: bool

    @property
    def bp(self) -> int:
        """Length of the feature's interval."""
        return self.end - self.begin + 1


class OrfProof(BaseModel):
    """The evidence that one annotated CDS is a clean ORF in the reconstruction."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    feature: str
    span: str
    strand: str
    bp: int
    n_terminus: str
    internal_stops: int


class FlankHit(BaseModel):
    """One located stretch of the plasmid: 1-based inclusive, with its bases."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    start: int
    end: int
    seq: str

    @property
    def bp(self) -> int:
        """Length of the stretch."""
        return self.end - self.start + 1


class ScaffoldSite(BaseModel):
    """Where the loader's pTpA flanks and the site between them sit in the contig."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    five_prime_flank: FlankHit
    between_flanks: FlankHit
    three_prime_flank: FlankHit


def _data_root() -> str:
    from dotenv import load_dotenv

    load_dotenv()
    return os.environ["DATA_ROOT"]


def curl(url: str, destination: Path | None) -> tuple[int, bytes, str]:
    """Fetch one URL with curl; return (status, body, the exact command run).

    curl rather than a python client so the recorded ``retrieval_command`` is the thing
    that ran, verbatim, and a rebuild re-runs it without this script.
    """
    command = (
        f'curl -sS -L -A "{USER_AGENT}" -w "%{{http_code}}" '
        f'-o {"-" if destination is None else str(destination)} "{url}"'
    )
    target = "-" if destination is None else str(destination)
    result = subprocess.run(  # noqa: S603  # fixed argv, no shell
        [
            "curl",
            "-sS",
            "-L",
            "-A",
            USER_AGENT,
            "-w",
            "%{http_code}",
            "-o",
            target,
            url,
        ],
        capture_output=True,
        check=True,
    )
    if destination is None:
        # with -o -, the body goes to stdout and the status is appended to it
        payload = result.stdout
        status = int(payload[-3:])
        return status, payload[:-3], command
    status = int(result.stdout[-3:])
    return status, destination.read_bytes(), command


def fetch(url: str, root: Path, rel_path: str) -> FetchedFile:
    """Fetch ``url`` into ``root/rel_path``; a non-200 halts the run."""
    destination = root / rel_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    status, payload, command = curl(url, destination)
    if status != 200:
        raise RuntimeError(f"{url} -> HTTP {status}; nothing is deposited")
    return FetchedFile(
        rel_path=rel_path,
        url=url,
        command=command,
        sha256=hashlib.sha256(payload).hexdigest(),
        bytes=len(payload),
        retrieved_at=datetime.now(UTC).isoformat(),
        http_status=status,
    )


def probe(url: str) -> int:
    """The HTTP status the CDN gives this URL, recorded whatever it is."""
    status, _, _ = curl(url, None)
    return status


def page_text(html: str) -> str:
    """The page's stable text: scripts out, tags to newlines, the rotating block cut.

    The three removals are the three measured sources of per-request variation, and
    nothing else is changed: no whitespace normalization, no case folding, so a quote
    taken against this text is verbatim against what the page served.
    """
    text = TAG.sub("\n", SCRIPT.sub("\n", html))
    start = text.find(ROTATING_BLOCK_HEADING)
    if start == -1:
        return text
    end = text.find(ROTATING_BLOCK_END, start)
    if end == -1:
        raise RuntimeError(
            f"the page carries {ROTATING_BLOCK_HEADING!r} but not "
            f"{ROTATING_BLOCK_END!r} after it, so the rotating block's extent is not "
            "established and the extraction would not be stable"
        )
    return text[:start] + text[end:]


def fetch_stable_page(url: str, root: Path, rel_path: str) -> tuple[FetchedFile, str]:
    """Fetch a catalog page TWICE and deposit its text only if both extractions agree.

    Returns the record for the deposited ``.page.txt`` and the text itself. The two HTML
    bodies are NOT deposited: their sha256 is a property of the request, not of the page,
    so pinning one would be a hash a rebuild cannot reproduce. Their hashes are recorded
    as data in the text's ``ProcessingRecord``.
    """
    first = root / f"{rel_path}.fetch1.html"
    second = root / f"{rel_path}.fetch2.html"
    first.parent.mkdir(parents=True, exist_ok=True)
    hashes: list[str] = []
    texts: list[str] = []
    commands: list[str] = []
    for destination in (first, second):
        status, payload, command = curl(url, destination)
        if status != 200:
            raise RuntimeError(f"{url} -> HTTP {status}; nothing is deposited")
        hashes.append(hashlib.sha256(payload).hexdigest())
        texts.append(page_text(payload.decode("utf-8")))
        commands.append(command)
    if texts[0] != texts[1]:
        raise RuntimeError(
            f"{url}: two fetches seconds apart give different page text, so no stable "
            "artifact can be pinned for it; the extraction in page_text() no longer "
            "covers everything the page varies"
        )
    first.unlink()
    second.unlink()
    target = root / f"{rel_path}{PAGE_TEXT_SUFFIX}"
    target.write_text(texts[0], encoding="utf-8")
    payload_bytes = target.read_bytes()
    return (
        FetchedFile(
            rel_path=f"{rel_path}{PAGE_TEXT_SUFFIX}",
            url=url,
            command=(
                f"{commands[0]}  # run twice; page_text() applied; the two extractions "
                "must hash identically"
            ),
            sha256=hashlib.sha256(payload_bytes).hexdigest(),
            bytes=len(payload_bytes),
            retrieved_at=datetime.now(UTC).isoformat(),
            http_status=200,
            html_fetch_sha256=hashes,
        ),
        texts[0],
    )


def reconstruct_top_strand(lines_json: bytes) -> str:
    """The top strand of one rendered sequence, read by glyph coordinate and checked.

    Per rendered line: the two rows of exactly ``end - begin + 1`` lowercase base glyphs
    are the two strands, the lower ``y`` is the top one, and the other row must be its
    base-for-base complement. Lines must tile the molecule with no gap. Any other shape
    raises, because a render this code has not seen is not a sequence to guess at.
    """
    pieces: dict[int, str] = {}
    for entry in json.loads(lines_json):
        content = entry["content"]
        match = RULER.search(content)
        if match is None:
            raise RuntimeError(f"rendered line {entry['data_line']} carries no ruler")
        begin, end = (int(value) for value in match.groups())
        width = end - begin + 1
        rows: dict[int, dict[int, str]] = {}
        for x, y, char in GLYPH.findall(content):
            rows.setdefault(int(y), {})[int(x)] = char
        full = sorted(y for y, row in rows.items() if len(row) == width)
        if len(full) != 2:
            raise RuntimeError(
                f"line {begin}..{end}: {len(full)} rows of {width} base glyphs, want 2 "
                f"(rows held {sorted((y, len(r)) for y, r in rows.items())})"
            )
        top = "".join(rows[full[0]][x] for x in sorted(rows[full[0]]))
        bottom = "".join(rows[full[1]][x] for x in sorted(rows[full[1]]))
        if bottom != top.translate(COMPLEMENT_LOWER):
            raise RuntimeError(
                f"line {begin}..{end}: the second strand row is not the complement of "
                "the first, so which row is the top strand is not established"
            )
        pieces[begin] = top.upper()
    sequence = ""
    for begin in sorted(pieces):
        if begin != len(sequence):
            raise RuntimeError(
                f"rendered lines do not tile: a line begins at {begin} with "
                f"{len(sequence)} bases read"
            )
        sequence += pieces[begin]
    if set(sequence) - set("ACGT"):
        raise RuntimeError(f"reconstruction carries {sorted(set(sequence))}")
    return sequence


def revcomp(sequence: str) -> str:
    """Reverse complement."""
    return sequence.translate(COMPLEMENT)[::-1]


def translate(sequence: str) -> str:
    """Standard-code translation of a sequence whose length is a multiple of three."""
    return "".join(
        CODON_TABLE[sequence[i : i + 3]] for i in range(0, len(sequence) - 2, 3)
    )


def features_of(features_json: bytes) -> list[AddgeneFeature]:
    """Addgene's feature table: name with its markup stripped, 1-based inclusive range."""
    payload = json.loads(features_json)
    return [
        AddgeneFeature(
            name=TAG.sub("", str(feature["name"])),
            begin=int(feature["rangeBegin"]),
            end=int(feature["rangeEnd"]),
            reverse=feature["direction"] == "reverse-directional",
        )
        for feature in payload["features"]
    ]


def prove_orfs(sequence: str, features: list[AddgeneFeature]) -> list[OrfProof]:
    """Every feature of ``PROVEN_ORFS`` must be a clean ORF of the stated length.

    Five frames agreeing is the evidence that a sequence read out of an SVG render is the
    molecule and not a rendering artifact, so a failure here halts the run.
    """
    by_name = {feature.name: feature for feature in features}
    proofs: list[OrfProof] = []
    for name, length, n_terminus in PROVEN_ORFS:
        feature = by_name.get(name)
        if feature is None:
            raise RuntimeError(f"the feature table carries no {name!r}")
        begin, end = feature.begin, feature.end
        block = sequence[begin - 1 : end]
        if feature.reverse:
            block = revcomp(block)
        protein = translate(block)
        failures = []
        if len(block) != length:
            failures.append(f"{len(block)} bp, the depositor states {length}")
        if block[:3] != "ATG":
            failures.append(f"starts {block[:3]!r}, not 'ATG'")
        if protein[-1] != "*":
            failures.append(f"does not end on a stop codon ({protein[-3:]!r})")
        if protein[:-1].count("*") != 0:
            failures.append(f"{protein[:-1].count('*')} internal stop codons")
        if not protein.startswith(n_terminus):
            failures.append(f"N-terminus {protein[:6]!r}, expected {n_terminus!r}")
        if failures:
            raise RuntimeError(f"{name} is not the ORF it is annotated as: {failures}")
        proofs.append(
            OrfProof(
                feature=name,
                span=f"{begin}..{end}",
                strand="-" if feature.reverse else "+",
                bp=len(block),
                n_terminus=protein[:6],
                internal_stops=0,
            )
        )
    return proofs


def locate_scaffold(sequence: str) -> ScaffoldSite:
    """Where the loader's pTpA flanks and the reporter start codon sit in the plasmid.

    This is the check step 3 of the task asks for, run against the bytes: each flank must
    occur exactly once on the forward strand, the 5' one must precede the 3' one, and the
    reporter's start codon must lie downstream of both.
    """
    found: dict[str, list[int]] = {}
    for label, flank in (
        ("five_prime", SCAFFOLD_5PRIME),
        ("three_prime", SCAFFOLD_3PRIME),
    ):
        doubled = sequence + sequence
        forward = [
            m.start() + 1
            for m in re.finditer(flank, doubled)
            if m.start() < len(sequence)
        ]
        reverse = [
            m.start() + 1
            for m in re.finditer(revcomp(flank), doubled)
            if m.start() < len(sequence)
        ]
        if len(forward) != 1 or reverse:
            raise RuntimeError(
                f"the {label} pTpA flank occurs {len(forward)} times forward and "
                f"{len(reverse)} times reverse; a unique forward hit is what makes the "
                "cloning site a position rather than a guess"
            )
        found[label] = forward
    five_start = found["five_prime"][0]
    five_end = five_start + len(SCAFFOLD_5PRIME) - 1
    three_start = found["three_prime"][0]
    three_end = three_start + len(SCAFFOLD_3PRIME) - 1
    if not five_end < three_start:
        raise RuntimeError("the 3' flank does not follow the 5' flank")
    return ScaffoldSite(
        five_prime_flank=FlankHit(start=five_start, end=five_end, seq=SCAFFOLD_5PRIME),
        between_flanks=FlankHit(
            start=five_end + 1,
            end=three_start - 1,
            seq=sequence[five_end : three_start - 1],
        ),
        three_prime_flank=FlankHit(
            start=three_start, end=three_end, seq=SCAFFOLD_3PRIME
        ),
    )


def write_genbank(
    path: Path, sequence: str, features: list[AddgeneFeature], locus: str
) -> None:
    """Write the reconstruction as a circular GenBank map Biopython round-trips."""
    from Bio.Seq import Seq
    from Bio.SeqFeature import FeatureLocation, SeqFeature
    from Bio.SeqRecord import SeqRecord

    record = SeqRecord(
        Seq(sequence),
        id=locus,
        name=locus,
        description=(
            "Addgene 127546 yeast_DualReporter, reconstructed from Addgene's served "
            "viewer artifacts (see this key's manifest.json)"
        ),
        annotations={"molecule_type": "DNA", "topology": "circular"},
    )
    for feature in features:
        genbank_type = GENBANK_TYPE_BY_FEATURE.get(feature.name)
        if genbank_type is None:
            raise RuntimeError(
                f"no GenBank type is recorded for the Addgene feature {feature.name!r}; "
                "add it to GENBANK_TYPE_BY_FEATURE rather than letting it fall to a "
                "default"
            )
        record.features.append(
            SeqFeature(  # type: ignore[no-untyped-call]  # Bio.SeqFeature is untyped
                FeatureLocation(  # type: ignore[no-untyped-call]  # likewise
                    feature.begin - 1, feature.end, strand=-1 if feature.reverse else 1
                ),
                type=genbank_type,
                qualifiers={"label": [feature.name]},
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    from Bio import SeqIO

    with path.open("w", encoding="utf-8") as handle:
        SeqIO.write(record, handle, "genbank")


def retrieval_for(fetched: FetchedFile) -> RetrievalRecord:
    """One fetched file's retrieval record, every field from the fetch that ran."""
    return RetrievalRecord(
        method=RetrievalMethod.direct_url,
        source_url=fetched.url,
        retriever="manual",
        params={
            "retrieval_command": fetched.command,
            "http_status": fetched.http_status,
            "fetched_by": (
                "experiments/043-expression-proteome-morphology/scripts/"
                "fetch_addgene_127546.py"
            ),
        },
        sha256=fetched.sha256,
        retrieved_at=fetched.retrieved_at,
    )


def extend_raw_manifest(
    records: list[ArtifactRecord], data_root: str, dry_run: bool
) -> Path:
    """Add these records to the raw key's manifest, additively; refuse a listed path."""
    manifest = load_manifest(TIER, CITATION_KEY, data_root)  # type: ignore[arg-type]
    listed = {record.path for record in manifest.files}
    clashes = sorted(record.path for record in records if record.path in listed)
    if clashes:
        raise FileExistsError(
            f"{CITATION_KEY}: {clashes} already in the raw manifest; a listed file is "
            "never re-recorded"
        )
    extended = Manifest.model_validate(
        manifest.model_copy(update={"files": [*manifest.files, *records]}).model_dump()
    )
    path = manifest_path(TIER, CITATION_KEY, data_root)  # type: ignore[arg-type]
    if not dry_run:
        path.write_text(extended.model_dump_json(indent=2), encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    """Fetch, prove, deposit; ``--dry-run`` fetches and proves but writes no manifest."""
    parser = argparse.ArgumentParser(prog="fetch_addgene_127546")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--data-root", default=None)
    args = parser.parse_args(argv)
    data_root = args.data_root or _data_root()
    root = key_dir(TIER, CITATION_KEY, data_root)  # type: ignore[arg-type]

    print(f"=== Addgene {PLASMID_ID} -> {CONSTRUCT_SET} ===")
    fetched: list[FetchedFile] = []
    plasmid_page, text = fetch_stable_page(
        f"https://www.addgene.org/{PLASMID_ID}/", root, f"{PREFIX}/plasmid"
    )
    sequences_page, _ = fetch_stable_page(
        f"https://www.addgene.org/{PLASMID_ID}/sequences/", root, f"{PREFIX}/sequences"
    )
    fetched.extend([plasmid_page, sequences_page])
    if SPEC.name_quote not in text:
        raise RuntimeError(
            "the depositor's description is not a verbatim substring of the plasmid "
            "page's text; a construct's name is never taken from a filename or a title"
        )
    if STATED_LENGTH_QUOTE not in text or str(SPEC.stated_length) not in text:
        raise RuntimeError(
            "the plasmid page does not state the vector size checked here"
        )
    print(
        f"(a) {SOURCE_NAME}: page text stable across two fetches "
        f"(sha256 {plasmid_page.sha256[:12]}), depositor description quoted, "
        f"{STATED_LENGTH} bp stated"
    )

    refusals: dict[str, dict[str, object]] = {}
    sequences: dict[str, str] = {}
    feature_tables: dict[str, list[AddgeneFeature]] = {}
    for sequence_object in SPEC.sequence_objects:
        api = f"https://www.addgene.org/api/get-sequence-file-collection/{sequence_object}/"
        fetched.append(
            fetch(api, root, f"{PREFIX}/seq-{sequence_object}/collection.json")
        )
        collection = json.loads(
            (root / f"{PREFIX}/seq-{sequence_object}/collection.json").read_bytes()
        )
        for key in ("genbankUrl", "snapgeneUrl"):
            status = probe(str(collection[key]))
            refusals[f"{sequence_object}:{key}"] = {
                "url": collection[key],
                "http_status": status,
            }
            print(f"(b) {sequence_object} {key} -> HTTP {status}")
        base = str(collection["mapSvgUrl"]).removesuffix("-map.svg")
        for asset, suffix in (
            ("features.json", "-features.json"),
            ("sequence-lines.json", "-sequence-lines.json"),
            ("map.svg", "-map.svg"),
        ):
            fetched.append(
                fetch(base + suffix, root, f"{PREFIX}/seq-{sequence_object}/{asset}")
            )
        lines = (
            root / f"{PREFIX}/seq-{sequence_object}/sequence-lines.json"
        ).read_bytes()
        table = (root / f"{PREFIX}/seq-{sequence_object}/features.json").read_bytes()
        sequence = reconstruct_top_strand(lines)
        if len(sequence) != SPEC.stated_length:
            raise RuntimeError(
                f"{sequence_object}: reconstructed {len(sequence)} bp, the depositor "
                f"states {SPEC.stated_length}"
            )
        feature_tables[sequence_object] = features_of(table)
        proofs = prove_orfs(sequence, feature_tables[sequence_object])
        sequences[sequence_object] = sequence
        print(
            f"(c) {sequence_object}: {len(sequence)} bp reconstructed, "
            f"{len(proofs)} ORFs proved, {len(feature_tables[sequence_object])} features"
        )

    rotations = {
        obj: (seq in (sequences[SPEC.canonical_sequence_object] * 2))
        for obj, seq in sequences.items()
    }
    print(f"(c) each full sequence is a rotation of the canonical one: {rotations}")

    canonical = sequences[SPEC.canonical_sequence_object]
    scaffold = locate_scaffold(canonical)
    reporter = next(
        feature
        for feature in feature_tables[SPEC.canonical_sequence_object]
        if feature.name == REPORTER_FEATURE_LABEL
    )
    print(
        f"(c) pTpA site: 5' flank {scaffold.five_prime_flank.start}.."
        f"{scaffold.five_prime_flank.end}, "
        f"{scaffold.between_flanks.bp} bp "
        f"({scaffold.between_flanks.seq}) between, 3' flank "
        f"{scaffold.three_prime_flank.start}..{scaffold.three_prime_flank.end}; "
        f"reporter {REPORTER_FEATURE_LABEL} {reporter.begin}..{reporter.end}"
    )

    genbank_rel = (
        f"{PREFIX}/derived/addgene-{PLASMID_ID}-{SPEC.canonical_sequence_object}.gb"
    )
    genbank_path = root / genbank_rel
    write_genbank(
        genbank_path,
        canonical,
        feature_tables[SPEC.canonical_sequence_object],
        f"addgene{PLASMID_ID}_{SPEC.canonical_sequence_object}",
    )
    component, construct_features = parse_contig_features(genbank_path, CITATION_KEY)
    if component.sequence != canonical or component.topology != "circular":
        raise RuntimeError("the written GenBank does not round-trip the reconstruction")
    print(
        f"(d) derived {genbank_rel}: {component.length} bp circular, "
        f"{len(construct_features)} features re-parsed"
    )

    inputs = {
        f.rel_path: f.sha256
        for f in fetched
        if f.rel_path.startswith(f"{PREFIX}/seq-{SPEC.canonical_sequence_object}/")
    }
    records = [
        ArtifactRecord(
            path=f.rel_path,
            role=ROLE_RAW_DATA,
            bytes=f.bytes,
            sha256=f.sha256,
            source=f.url,
            # A catalog page's bytes vary per request (CSRF token, telemetry stamp,
            # rotating recommendations), so its deposited text carries PROCESSING
            # provenance: the hash a rebuild reproduces is the extraction's, and the
            # two HTML bodies it came from are recorded by hash as data.
            processing=(
                ProcessingRecord(
                    processor=(
                        "experiments.043-expression-proteome-morphology.scripts."
                        "fetch_addgene_127546.page_text"
                    ),
                    tool="torchcell-addgene-page-text",
                    version="1",
                    params={
                        "retrieval_command": f.command,
                        "source_url": f.url,
                        "removed": [
                            "<script> and <style> bodies",
                            "every tag, replaced by a newline",
                            f"the rotating block from {ROTATING_BLOCK_HEADING!r} to "
                            f"{ROTATING_BLOCK_END!r}",
                        ],
                        "stability_proof": (
                            "the page was fetched twice in one run and the two "
                            "extractions hashed identically"
                        ),
                        "retrieved_at": f.retrieved_at,
                    },
                    input_sha256=f.html_fetch_sha256,
                )
                if f.html_fetch_sha256
                else None
            ),
            retrieval=None if f.html_fetch_sha256 else retrieval_for(f),
        )
        for f in fetched
    ]
    genbank_sha = sha256_file(genbank_path)
    records.append(
        ArtifactRecord(
            path=genbank_rel,
            role=ROLE_RAW_DATA,
            bytes=genbank_path.stat().st_size,
            sha256=genbank_sha,
            source="reconstructed from the Addgene viewer artifacts listed above",
            processing=ProcessingRecord(
                processor=(
                    "experiments.043-expression-proteome-morphology.scripts."
                    "fetch_addgene_127546.reconstruct_top_strand"
                ),
                tool="torchcell-addgene-viewer-reconstruction",
                version="1",
                params={
                    "sequence_object": SPEC.canonical_sequence_object,
                    "stated_length_bp": SPEC.stated_length,
                    "proved_orfs": [name for name, _, _ in PROVEN_ORFS],
                    "manual_recipe_for_the_depositor_genbank": MANUAL_GBK_RECIPE,
                    "refused_download_urls": refusals,
                },
                input_sha256=sorted(inputs.values()),
            ),
        )
    )
    raw_manifest = extend_raw_manifest(records, data_root, args.dry_run)
    print(
        f"(d) raw manifest {'(dry-run, unwritten)' if args.dry_run else 'extended'}: {raw_manifest}"
    )

    contig = ConstructContig(
        sha256=genbank_sha,
        ref=ArtifactRef(
            tier=TIER,  # type: ignore[arg-type]
            key=CITATION_KEY,
            path=genbank_rel,
            sha256=genbank_sha,
            bytes=genbank_path.stat().st_size,
        ),
        record_name=component.identity,
        topology="circular",
        length=component.length,
        kind=SPEC.kind,
        features=construct_features,
        source_name=SourcedValue(
            value=SPEC.source_name,
            provenance=Provenance(
                source_uri=f"{PREFIX}/plasmid{PAGE_TEXT_SUFFIX}",
                citation_key=CITATION_KEY,
                sha256=next(
                    f.sha256
                    for f in fetched
                    if f.rel_path.endswith(f"plasmid{PAGE_TEXT_SUFFIX}")
                ),
                method="addgene-plasmid-page-text (scripts and the rotating block removed)",
                page=f"https://www.addgene.org/{PLASMID_ID}/",
            ),
            quote=SPEC.name_quote,
            note=(
                "Addgene's own name for the plasmid, from the page that also states its "
                "8,218 bp vector size and its five gene inserts; the quote is the "
                "depositor's Purpose field, which names the variable reporter yEVenus "
                "(alt name YFP) and the constant reporter mCherry (alt name RFP)"
            ),
        ),
        source_statement=SPEC.name_quote,
    )
    manifest = ConstructManifest(
        construct_set=SPEC.construct_set,
        source=f"Addgene plasmid {PLASMID_ID} (yeast_DualReporter), deposited by the "
        "Regev lab with de Boer et al. 2020",
        sequence_status=SPEC.sequence_status,
        citation_key=CITATION_KEY,
        files=records,
        contigs=[contig],
        blocked=[],
        # The contig's own bytes are a reconstruction carrying ProcessingRecord rather
        # than a retrieval chain, so completeness is not earned and is not claimed.
        provenance_complete=False,
        created_at=utc_now(),
        notes=(
            "Addgene 127546, the GPRA dual-reporter plasmid the Vaishnav 2022 loader "
            "names and did not hold. Addgene's API advertises a GenBank and a SnapGene "
            "file for each of its two deposited full sequences (251803 Addgene-verified, "
            "244230 depositor) and the CDN returns HTTP 404 for all four, while every "
            "sibling viewer artifact in the same directory returns 200; the contig is "
            "therefore reconstructed from the served per-line SVG by glyph coordinate "
            "and proved four ways before deposit: 137 rendered lines tile 0..8217 with "
            "two complementary strand rows each, the length equals the 8,218 bp the "
            "plasmid page states, and URA3, yEVenus (labelled yeGFP by SnapGene's "
            "autoannotation), mCherry, NrsR and AmpR are each a clean ORF of the stated "
            "length with the expected N-terminus and no internal stop. 244230 is a "
            "rotation of 251803, not a second molecule. The pTpA cloning site the "
            "Vaishnav loader sources from the paper is present once on the forward "
            f"strand: 17 bp flank at {scaffold.five_prime_flank.start}.."
            f"{scaffold.five_prime_flank.end}, "
            f"{scaffold.between_flanks.bp} bp "
            f"({scaffold.between_flanks.seq}) in place of the 80 bp library "
            f"insert, 13 bp flank at {scaffold.three_prime_flank.start}.."
            f"{scaffold.three_prime_flank.end}, and the reporter start codon at "
            f"{reporter.begin}. sequence_status is design: Addgene serves maps, not "
            "a resequencing of the tube. The manual recipe for the refused GenBank is "
            "in the derived file's ProcessingRecord."
        ),
    )
    print(
        f"(e) index: {len(manifest.files)} files, {len(manifest.contigs)} contig, "
        f"provenance_complete={manifest.provenance_complete}"
    )
    report = {
        "plasmid": f"addgene:{PLASMID_ID}",
        "source_name": SPEC.source_name,
        "construct_set": SPEC.construct_set,
        "reconstructed_length_bp": len(canonical),
        "stated_length_bp": SPEC.stated_length,
        "sequence_objects": {
            obj: {"length_bp": len(seq), "rotation_of_canonical": rotations[obj]}
            for obj, seq in sequences.items()
        },
        "refused_downloads": refusals,
        "manual_recipe": MANUAL_GBK_RECIPE,
        "orf_proofs": [
            proof.model_dump()
            for proof in prove_orfs(
                canonical, feature_tables[SPEC.canonical_sequence_object]
            )
        ],
        "scaffold": scaffold.model_dump(),
        "reporter_feature": {
            "label": REPORTER_FEATURE_LABEL,
            "depositor_gene_name": "yEVenus",
            "depositor_alt_name": "YFP",
            "begin": reporter.begin,
            "end": reporter.end,
            "bp": reporter.bp,
            "stated_insert_bp": REPORTER_INSERT_BP,
        },
        "derived_genbank": {"path": genbank_rel, "sha256": genbank_sha},
        "files": [f.model_dump() for f in fetched],
    }
    results = Path(osp.dirname(osp.dirname(osp.abspath(__file__)))) / "results"
    results.mkdir(parents=True, exist_ok=True)
    report_path = results / "addgene_127546_fetch.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"(e) report {report_path}")
    if args.dry_run:
        print("(e) --dry-run: no construct index deposited")
        return 0
    path = deposit_construct_set(manifest, data_root)
    measured = verify_construct_set(SPEC.construct_set, data_root)
    print(f"(e) deposited {path}; verified {len(measured)} contig(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
