# experiments/036-dataset-fixes-before-kg-build/scripts/cachera2023_construct_span_parts.py
# [[experiments.036-dataset-fixes-before-kg-build.scripts.cachera2023_construct_span_parts]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/036-dataset-fixes-before-kg-build/scripts/cachera2023_construct_span_parts
r"""Measure what a span of parts buys the Cachera construct set, and what it does not.

Lane T of the Path U ontology unification. ``ConstructSpan`` used to be one interval, so
``parse_genbank_component`` refused what it could not hold and 11 of the 17 mirrored
Cachera maps were ``BlockedConstruct`` rows. This script re-measures the set now that a
span holds ordered parts with an INSDC-grounded composition, and answers the one
question the unblocking was supposed to settle:

1. **Coverage.** How many maps index, how many are blocked and why, how many features
   are held, and the span-composition histogram over all of them.
2. **The round trip.** For every feature of every map, the location string the source
   wrote, read out of the raw bytes, against the string the stored parts and strand
   regenerate. Both writers are checked: the compact ``11559..128`` form the maps carry,
   and the canonical INSDC ``join(11559..12096,1..128)``, which is re-parsed through
   Biopython and must come back as the same parts. A span that cannot regenerate its own
   source location is not a faithful coordinate.
3. **The origin-spanning features by name.** The one feature per formerly blocked map,
   with the parts it wrapped into and the length those parts give, against the
   whole-plasmid length the envelope used to give.
4. **Cachera's contig-side insert span.** pBTX2 is now indexed, so the three intervals
   it admits are measured: its two annotated homology arms, and the NotI fragment the
   paper's own sentence names ("the NotI excised insert of pBTX2"), found in the bytes.
   The same three are measured on pBTX1, which is the contig the four
   ``GeneAdditionPerturbation`` records actually point at. The decision follows from
   which of them agree.
5. **The placement census.** ``construct_placement_census`` over the four Btx-cassette
   perturbations the loader builds, so the report-class row's numbers are stated rather
   than assumed.

Writes ``experiments/036-dataset-fixes-before-kg-build/results/cachera2023_construct_span_parts.json``.

Usage (from the repo root)::

    python experiments/036-dataset-fixes-before-kg-build/scripts/cachera2023_construct_span_parts.py
"""

from __future__ import annotations

import io
import json
import os
import os.path as osp
import re
import warnings
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

from Bio import SeqIO
from dotenv import load_dotenv

from torchcell.datamodels.schema import SpanComposition
from torchcell.sequence.construct.registry import (
    CACHERA2023_CRI_SPA_PLASMIDS,
    ConstructContig,
    load_construct_manifest,
    parse_contig_features,
)
from torchcell.verification.construct_placement import construct_placement_census

#: The paper's name for the contig the four screened-strain records point at, and for
#: the BY-Btx vector that is newly indexed.
CD_CONTIG = "pBTX1"
BY_CONTIG = "pBTX2"

#: The homology-arm feature labels of each vector, as each file writes them.
ARMS: dict[str, tuple[str, str]] = {
    CD_CONTIG: ("XII-5 up B", "XII-5 dw A"),
    BY_CONTIG: ("XII-5\\UP", "XII-5\\Down"),
}

#: NotI recognition site and the offset of its top-strand cut within it: ``GC^GGCCGC``
#: cuts after the second base, so a site beginning at 1-based ``p`` is cut between
#: ``p + 1`` and ``p + 2``.
NOTI_SITE = "GCGGCCGC"
NOTI_CUT_OFFSET = 2

#: The paper's own statement of the cassette's size, verbatim, and the sentence that
#: names how pBTX2's insert was liberated. Both are substrings of ``paper.md``.
PAPER_CASSETTE_SIZE = "the considerable size, $9 . 3 \\mathrm { k b }$ , of the Btx"
PAPER_NOTI_INSERT = (
    "The BY-Btx strain was made by transforming the NotI excised insert of pBTX2"
)

RESULTS = osp.join(
    osp.dirname(osp.dirname(osp.abspath(__file__))),
    "results",
    "cachera2023_construct_span_parts.json",
)


def _source_locations(path: Path) -> list[tuple[str, str]]:
    """``(feature type, location string)`` for every non-``source`` feature, in file order.

    Read from the raw bytes rather than from Biopython, because Biopython is what
    rewrites the compact origin-wrapping form; the round trip is only a round trip
    against what the file says.
    """
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    inside = False
    found: list[tuple[str, str]] = []
    for index, line in enumerate(lines):
        if line.startswith("FEATURES"):
            inside = True
            continue
        if line.startswith("ORIGIN"):
            break
        if not inside:
            continue
        match = re.match(r"^ {5}(\S+)\s+(\S.*)$", line)
        if match is None:
            continue
        feature_type, location = match.group(1), match.group(2).strip()
        follower = index + 1
        while (
            follower < len(lines)
            and lines[follower].startswith(" " * 21)
            and not lines[follower].strip().startswith("/")
        ):
            location += lines[follower].strip()
            follower += 1
        if feature_type == "source":
            continue
        found.append((feature_type, location))
    return found


def _biopython_parts(text: str) -> list[tuple[int, int]] | None:
    """The 0-based half-open parts Biopython reads from a one-feature GenBank record."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        record = SeqIO.read(io.StringIO(text), "genbank")  # type: ignore[no-untyped-call]  # Bio.SeqIO is untyped
    feature = record.features[-1]
    if feature.location is None:
        return None
    return [(int(part.start), int(part.end)) for part in feature.location.parts]


def _canonical_reparses(span: Any, replicon_length: int, feature_type: str) -> bool:
    """Whether the canonical INSDC form re-parses into the parts the span holds.

    A minimal circular record is written with only this feature, so the check is of the
    location grammar alone. Biopython hands a reverse-strand compound location's parts
    back in 5'->3' order, which is the reverse of the written order the span stores, so
    the expectation is reversed to match.
    """
    sequence = "a" * replicon_length
    body = "".join(
        "{:>9} {}\n".format(
            start + 1,
            " ".join(
                sequence[block : block + 10] for block in range(start, start + 60, 10)
            ).strip(),
        )
        for start in range(0, replicon_length, 60)
    )
    text = (
        f"LOCUS       pRT{replicon_length:>24} bp    DNA     circular SYN 01-JAN-2026\n"
        "DEFINITION  round trip.\n"
        "FEATURES             Location/Qualifiers\n"
        f"     source          1..{replicon_length}\n"
        f"     {feature_type:<15} {span.insdc_location()}\n"
        f'{" " * 21}/label="rt"\n'
        f"ORIGIN\n{body}//\n"
    )
    got = _biopython_parts(text)
    expected = [(part.start - 1, part.end) for part in span.parts]
    if span.orientation == "reverse_complement":
        expected = list(reversed(expected))
    return got == expected


def _noti_fragment(sequence: str) -> dict[str, Any]:
    """The NotI sites in a circular sequence and the fragment between the two cuts.

    1-based, end-inclusive. The search runs over the sequence doubled by the site length
    minus one so a site across the origin is found; a replicon with exactly two sites
    has exactly two fragments, and the one reported is the interval between the cuts in
    ascending order, which for these vectors is the one carrying the cassette.
    """
    upper = sequence.upper()
    length = len(upper)
    sites = sorted(
        {
            (match.start() % length) + 1
            for match in re.finditer(
                re.escape(NOTI_SITE), upper + upper[: len(NOTI_SITE) - 1]
            )
        }
    )
    out: dict[str, Any] = {"site": NOTI_SITE, "site_starts": sites}
    if len(sites) != 2:
        out["fragment"] = None
        return out
    first, second = (site + NOTI_CUT_OFFSET for site in sites)
    out["fragment"] = {"start": first, "end": second - 1, "length": second - first}
    return out


def _arm_candidates(contig: ConstructContig, name: str) -> dict[str, Any]:
    """The intervals the two annotated homology arms of one vector admit."""
    up_label, down_label = ARMS[name]
    arms: dict[str, dict[str, int]] = {}
    for feature in contig.features:
        if feature.key.label in (up_label, down_label) and (
            feature.key.label not in arms
        ):
            arms[feature.key.label] = {
                "start": feature.key.span.start,
                "end": feature.key.span.end,
                "length": feature.key.span.length,
            }
    if sorted(arms) != sorted((up_label, down_label)):
        return {"arms": arms, "arm_to_arm": None, "between_arms": None}
    up, down = arms[up_label], arms[down_label]
    return {
        "arms": arms,
        "arm_to_arm": {
            "start": up["start"],
            "end": down["end"],
            "length": down["end"] - up["start"] + 1,
        },
        "between_arms": {
            "start": up["end"] + 1,
            "end": down["start"] - 1,
            "length": down["start"] - up["end"] - 1,
        },
    }


def _library_path(data_root: str, contig: ConstructContig) -> Path:
    """The mirrored artifact's absolute path in the library tier."""
    return Path(data_root) / "torchcell-library" / contig.ref.key / contig.ref.path


def _vector_report(
    data_root: str, contig: ConstructContig, name: str
) -> dict[str, Any]:
    """One Btx vector's arms, NotI sites and the intervals they admit."""
    path = _library_path(data_root, contig)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        record = SeqIO.read(path, "genbank")  # type: ignore[no-untyped-call]  # Bio.SeqIO is untyped
    report: dict[str, Any] = {
        "source_name": name,
        "handle": contig.handle,
        "length": contig.length,
        "topology": contig.topology,
        "topology_so": contig.topology_term.model_dump(),
        "n_features": len(contig.features),
        "noti": _noti_fragment(str(record.seq)),
    }
    report.update(_arm_candidates(contig, name))
    return report


def _census(perturbations: list[Any]) -> dict[str, Any]:
    """The report-class placement census over a list of perturbation models."""
    return construct_placement_census(
        [perturbation.model_dump() for perturbation in perturbations]
    ).model_dump()


def _si_table_text(data_root: str, citation_key: str) -> str:
    """SUPPLEMENTARY TABLE S2's text, read out of the pinned container in memory."""
    container = (
        Path(data_root)
        / "torchcell-library"
        / citation_key
        / "si/gkad656_supplemental_files.zip"
    )
    with zipfile.ZipFile(container) as outer:
        payload = outer.read("Supplementary Material.docx")
    with zipfile.ZipFile(io.BytesIO(payload)) as document:
        xml = document.read("word/document.xml").decode("utf-8")
    return re.sub(r"<[^>]+>", "", re.sub(r"<w:p[ >]", "\n<w:p ", xml))


def main() -> None:
    """Measure the set, the round trip, the insert-span candidates and the census."""
    load_dotenv()
    data_root = os.environ["DATA_ROOT"]
    manifest = load_construct_manifest(CACHERA2023_CRI_SPA_PLASMIDS)
    key = manifest.citation_key or ""

    compositions: Counter[str] = Counter()
    mismatches: list[dict[str, str]] = []
    wraps: list[dict[str, Any]] = []
    canonical_failures: list[str] = []
    n_features = 0
    for contig in manifest.contigs:
        path = _library_path(data_root, contig)
        _, projected = parse_contig_features(path, key)
        written = _source_locations(path)
        if len(written) != len(projected):
            raise RuntimeError(
                f"{contig.ref.path}: {len(written)} locations in the bytes, "
                f"{len(projected)} projected features"
            )
        for (feature_type, location), feature in zip(written, projected, strict=True):
            n_features += 1
            span = feature.key.span
            compositions[str(span.composition)] += 1
            regenerated = span.insdc_location(compact_wrap=True)
            if regenerated != location:
                mismatches.append(
                    {
                        "path": contig.ref.path,
                        "feature_type": feature_type,
                        "source": location,
                        "regenerated": regenerated,
                    }
                )
            if span.composition is SpanComposition.simple:
                continue
            if not _canonical_reparses(span, contig.length, feature_type):
                canonical_failures.append(f"{contig.ref.path}:{feature.key.token}")
            wraps.append(
                {
                    "path": contig.ref.path,
                    "source_name": None
                    if contig.source_name is None
                    else contig.source_name.value,
                    "contig_length": contig.length,
                    "feature_type": feature_type,
                    "label": feature.key.label,
                    "so_id": feature.key.so_id,
                    "composition": str(span.composition),
                    "source_location": location,
                    "canonical_insdc": span.insdc_location(),
                    "parts": [[part.start, part.end] for part in span.parts],
                    "feature_length": span.length,
                    "envelope_length": contig.length,
                    "token": feature.key.token,
                }
            )

    by_name = {
        contig.source_name.value: contig
        for contig in manifest.contigs
        if contig.source_name is not None
    }
    vectors = {
        name: _vector_report(data_root, by_name[name], name)
        for name in (CD_CONTIG, BY_CONTIG)
    }
    paper = (Path(data_root) / "torchcell-library" / key / "paper.md").read_text(
        encoding="utf-8"
    )

    from torchcell.datasets.scerevisiae.cachera2023 import _betaxanthin_cassette

    si_text = _si_table_text(data_root, key)
    out: dict[str, Any] = {
        "construct_set": manifest.construct_set,
        "coverage": {
            "n_files": len(manifest.files),
            "n_contigs": len(manifest.contigs),
            "n_blocked": len(manifest.blocked),
            "blocked": [
                {"path": entry.path, "reason": str(entry.reason)}
                for entry in manifest.blocked
            ],
            "n_features": n_features,
            "span_compositions": dict(sorted(compositions.items())),
            "n_named_from_paper": sum(
                1
                for contig in manifest.contigs
                if contig.source_name is not None
                and contig.source_name.provenance.source_uri == "paper.md"
            ),
            "n_named_from_si_table": sum(
                1
                for contig in manifest.contigs
                if contig.source_name is not None
                and contig.source_name.provenance.source_uri != "paper.md"
            ),
            "n_name_gapped": sum(
                1 for contig in manifest.contigs if contig.source_name is None
            ),
            "provenance_complete": manifest.provenance_complete,
        },
        "round_trip": {
            "n_features_checked": n_features,
            "n_compact_form_mismatches": len(mismatches),
            "mismatches": mismatches,
            "n_canonical_reparse_failures": len(canonical_failures),
            "canonical_reparse_failures": canonical_failures,
        },
        "compound_features": wraps,
        "vectors": vectors,
        "paper_quotes": {
            "cassette_size": PAPER_CASSETTE_SIZE,
            "cassette_size_is_verbatim": PAPER_CASSETTE_SIZE in paper,
            "noti_insert": PAPER_NOTI_INSERT,
            "noti_insert_is_verbatim": PAPER_NOTI_INSERT in paper,
            "si_table_s2_present": "SUPPLEMENTARY TABLE S2" in si_text,
        },
        "placement_census": _census(_betaxanthin_cassette()),
    }

    os.makedirs(osp.dirname(RESULTS), exist_ok=True)
    with open(RESULTS, "w", encoding="utf-8") as handle:
        json.dump(out, handle, indent=2, sort_keys=True)
    print(json.dumps(out, indent=2, sort_keys=True))
    print(f"wrote {RESULTS}")


if __name__ == "__main__":
    main()
