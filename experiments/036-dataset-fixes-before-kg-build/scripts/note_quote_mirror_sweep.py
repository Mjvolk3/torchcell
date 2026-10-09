# experiments/036-dataset-fixes-before-kg-build/scripts/note_quote_mirror_sweep.py
# [[experiments.036-dataset-fixes-before-kg-build.scripts.note_quote_mirror_sweep]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/036-dataset-fixes-before-kg-build/scripts/note_quote_mirror_sweep
"""Find every dendron-note quote that no longer matches the pinned mirror bytes (#846).

A loader pins each sourced value to a verbatim substring of a sha256-pinned mirror
artifact (``torchcell.verification.sourced.SourcedValue``), and the module's dendron note
repeats that same quote as prose evidence. The ``markdownlint-cli2 --fix`` pre-commit
hook used to rewrite inline text, so a note's copy could drift from the mirror while the
loader's copy stayed correct: #846 found a bare DOI turned into an autolink and the
spaces deleted from ``ΔPP_ 2428``. The hash pin on the SOURCE cannot catch that, because
the source is never what changed.

This sweep re-reads both sides and reports every note rendition that the mirror does not
contain:

1. **The loader side.** Every ``SourcedValue`` reachable from a module under
   ``torchcell/datasets/`` is resolved against both mirror roots
   (``$DATA_ROOT/torchcell-library/<key>/<uri>`` for the OCR'd paper and SI,
   ``$DATA_ROOT/torchcell-raw/<key>/<uri>`` for a released data file), the artifact is
   re-hashed against the pinned ``sha256``, and the quote is looked up in the bytes.
   This is ``audit_sourced_value`` applied over the whole loader tree.
2. **The note side.** Every ``notes/**/*.md`` file is searched for each quote, whitespace
   and blockquote markers normalized away (a note wraps a quote across lines and
   prefixes it with ``>``, which is layout, not text). A note that holds the quote
   exactly is VERBATIM.
3. **The verdict, from the mirror.** When a note holds a NEAR rendition instead, that
   rendition is itself looked up in the same mirror bytes. Present means the note quotes
   a different, genuine passage. Absent means the note's bytes exist in no source, which
   is the falsification #846 is about, and the mirror window the rendition drifted from
   is reported alongside it so the fix is a restore, not a rewrite.

Writes
``experiments/036-dataset-fixes-before-kg-build/results/note_quote_mirror_sweep.json``.
The whole-tree run takes about 12 minutes (1,695 quotes against 1,546 notes), so run it
in the background and poll.

Usage (from the repo root)::

    python experiments/036-dataset-fixes-before-kg-build/scripts/note_quote_mirror_sweep.py
    python .../note_quote_mirror_sweep.py --module torchcell.datasets.pputida.menasalvas2025
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import importlib
import json
import os
import pkgutil
import re
from collections.abc import Iterator
from enum import StrEnum
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field

import torchcell.datasets
from torchcell.verification.sourced import SourcedValue

REPO = Path(__file__).resolve().parents[3]
NOTES_DIR = REPO / "notes"
RESULTS = REPO / "experiments/036-dataset-fixes-before-kg-build/results"
OUT = RESULTS / "note_quote_mirror_sweep.json"

# Notes the markdownlint hook never touches, so a finding in one of them is not the
# autofixer's doing (.pre-commit-config.yaml's `exclude`).
HOOK_EXCLUDED = re.compile(r"^notes/assets/verification/")

# A blockquote marker and the wrapping it imposes are layout; the quoted text is what is
# compared. Nothing else about a line is normalized away.
_BLOCKQUOTE = re.compile(r"^[ \t]*(?:>[ \t]*)+", re.MULTILINE)
_WHITESPACE = re.compile(r"\s+")

# A quote is located in a note by its rare long words before any alignment is attempted:
# a whole-file alignment would be quadratic in the note's length.
_ANCHOR_MIN_LEN = 6
_ANCHOR_COUNT = 6
_MAX_WINDOWS = 8
# Punctuation an anchor word may be wrapped in on one side of the comparison only.
_ANCHOR_TRIM = "\"'`,.;:()[]{}<>!?*"
# Character-level similarity above which a note passage counts as a rendition OF this
# quote rather than unrelated prose. An inline autofix moves a handful of characters, so
# a real falsification scores far above this; 0.85 is low enough to also catch a note
# that dropped a word while transcribing.
_NEAR_RATIO = 0.85
# A rendition is a RENDITION of the quote, rather than the note's own prose about it,
# only when it also differs by few characters. Without this a note that introduces the
# quote ("Supplementary Table 4's own banner says what it is: ...") aligns at 0.9 and the
# alignment's framing words are reported as drift: measured on the first whole-tree run,
# 192 of 233 reported renditions were that shape. 20 is well above the 8 characters the
# #846 autofixes moved and well below the 30-plus a framing clause adds.
_MAX_EDIT_CHARS = 20
_MIN_FALSIFIED_RATIO = 0.95
# The delimiters a note wraps a quote in, trimmed from both ends before comparing: they
# are the note's punctuation, not the source's bytes.
_QUOTE_TRIM = "\"'`,.;:() "
# MinerU writes a formula as LaTeX, so these characters on the MIRROR side of a
# difference mean the note resolved the markup rather than that the bytes drifted.
_MATH_MARKUP = frozenset("$\\^{}")
_CURLY_TO_ASCII = {
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u2013": "-",
    "\u2014": "-",
    "\u2212": "-",
}


class MirrorStatus(StrEnum):
    """What the loader's own pin looks like when re-read from the mirror."""

    pinned_and_present = "pinned_and_present"
    sha256_drift = "sha256_drift"
    quote_absent_from_artifact = "quote_absent_from_artifact"
    artifact_missing = "artifact_missing"


class NoteStatus(StrEnum):
    """How a note renders a quote, judged against the mirror bytes."""

    verbatim = "verbatim"  # the note holds the quote exactly (modulo wrapping)
    falsified = "falsified"  # a near rendition that appears in NO mirror artifact
    variant_in_mirror = "variant_in_mirror"  # a near rendition the mirror does contain
    paraphrase = "paraphrase"  # the note discusses the quote in its own words
    not_quoted = "not_quoted"  # no note repeats this quote


class DriftSignature(StrEnum):
    r"""WHAT changed between the mirror's bytes and the note's rendition.

    The point of the split is attribution. Only the first three shapes are things an
    inline markdownlint autofix does, so they are the ones #846 is about and the ones a
    restore commit must close. The rest are human transcription choices made while
    quoting MinerU output, which read better in a note than the mirror's bytes do and are
    an owner call, not an automatic rewrite.
    """

    markdownlint_autolink = "markdownlint_autolink"  # MD034 wrapped a bare URL in `<>`
    markdownlint_emphasis_space = (
        "markdownlint_emphasis_space"  # MD037/MD038 deleted a space beside `_`/`*`
    )
    markdownlint_list_marker = "markdownlint_list_marker"  # MD004 rewrote a marker
    ocr_math_markup = "ocr_math_markup"  # the note resolved MinerU LaTeX (`$1 . 5 \%$`)
    typography_fold = "typography_fold"  # curly quotes or an en dash folded to ASCII
    glyph_transliteration = (
        "glyph_transliteration"  # a non-ASCII glyph spelled out (`ΔPP_` -> `dPP_`)
    )
    authored_markup = (
        "authored_markup"  # the note added `**` or a backtick for emphasis
    )
    quotation_delimiter = "quotation_delimiter"  # only `"`, `'` or a backtick moved
    other = "other"  # anything else, a dropped or added word above all


class NoteRendition(BaseModel):
    """One note passage aligned to one loader quote."""

    model_config = ConfigDict(extra="forbid")

    note: str = Field(description="repo-relative note path")
    hook_excluded: bool = Field(
        description="true when the markdownlint hook never lints this note"
    )
    status: NoteStatus
    signature: DriftSignature | None = Field(
        default=None, description="what changed; set for a falsified rendition"
    )
    edit_chars: int = Field(
        default=0, description="characters that differ from the mirror window"
    )
    ratio: float = Field(description="character-level similarity to the loader quote")
    note_text: str = Field(description="the note's rendition, normalized")
    mirror_text: str = Field(
        default="", description="the mirror window it drifted from, normalized"
    )


class QuoteCheck(BaseModel):
    """One ``SourcedValue`` quote, checked on the loader side and the note side."""

    model_config = ConfigDict(extra="forbid")

    module: str
    citation_key: str
    source_uri: str
    quote: str
    sha256_pinned: str
    sha256_actual: str = ""
    mirror_root: str = Field(
        default="", description="the mirror the artifact was found under"
    )
    mirror_status: MirrorStatus
    renditions: list[NoteRendition] = Field(default_factory=list)

    @property
    def falsified(self) -> list[NoteRendition]:
        """The renditions whose bytes exist in no mirror artifact."""
        return [r for r in self.renditions if r.status == NoteStatus.falsified]


class SweepReport(BaseModel):
    """The committed result: totals plus every finding."""

    model_config = ConfigDict(extra="forbid")

    mirror_roots: list[str]
    n_modules_scanned: int
    n_notes_scanned: int
    n_quotes_checked: int
    mirror_status_counts: dict[str, int]
    n_quotes_verbatim_in_a_note: int
    n_quotes_not_quoted_in_any_note: int
    n_falsified_renditions: int
    n_variant_renditions: int
    n_paraphrase_renditions: int
    falsified_signature_counts: dict[str, int]
    falsified: list[QuoteCheck] = Field(default_factory=list)
    unauditable: list[QuoteCheck] = Field(default_factory=list)


def normalize(text: str) -> str:
    """Strip blockquote markers and collapse every whitespace run to one space."""
    return _WHITESPACE.sub(" ", _BLOCKQUOTE.sub("", text)).strip()


def sha256_path(path: Path) -> str:
    """The artifact's hash, streamed so a large supplement does not load into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_sourced_values(
    obj: Any, depth: int = 0, seen: set[int] | None = None
) -> Iterator[SourcedValue]:
    """Every ``SourcedValue`` reachable from a module attribute.

    Loaders hold them bare, in dicts keyed by value name, in tuples of record models, and
    nested inside other pydantic models, so the walk is structural rather than a
    convention on the attribute name.
    """
    if seen is None:
        seen = set()
    if depth > 8 or id(obj) in seen:
        return
    seen.add(id(obj))
    if isinstance(obj, SourcedValue):
        yield obj
        return
    if isinstance(obj, BaseModel):
        for value in obj.__dict__.values():
            yield from iter_sourced_values(value, depth + 1, seen)
        return
    if isinstance(obj, dict):
        for value in obj.values():
            yield from iter_sourced_values(value, depth + 1, seen)
        return
    if isinstance(obj, (list, tuple, set, frozenset)):
        for value in obj:
            yield from iter_sourced_values(value, depth + 1, seen)


def collect_quotes(module_filter: str | None) -> tuple[list[QuoteCheck], int]:
    """Every unique ``(module, key, uri, quote)`` under ``torchcell/datasets/``."""
    checks: dict[tuple[str, str, str, str], QuoteCheck] = {}
    modules = 0
    for info in pkgutil.walk_packages(
        torchcell.datasets.__path__, torchcell.datasets.__name__ + "."
    ):
        if module_filter is not None and info.name != module_filter:
            continue
        module = importlib.import_module(info.name)
        modules += 1
        for name, attr in list(vars(module).items()):
            if name.startswith("__"):
                continue
            for sv in iter_sourced_values(attr):
                key = (
                    info.name,
                    str(sv.provenance.citation_key),
                    sv.provenance.source_uri,
                    sv.quote,
                )
                if key in checks:
                    continue
                checks[key] = QuoteCheck(
                    module=info.name,
                    citation_key=key[1],
                    source_uri=key[2],
                    quote=sv.quote,
                    sha256_pinned=str(sv.provenance.sha256),
                    mirror_status=MirrorStatus.artifact_missing,
                )
    return list(checks.values()), modules


class MirrorText(BaseModel):
    """A mirror artifact's bytes, hashed once and normalized once."""

    model_config = ConfigDict(extra="forbid")

    sha256: str
    normalized: str


def load_mirror(path: Path, cache: dict[Path, MirrorText]) -> MirrorText:
    """Read, hash and normalize one artifact, memoized across the quotes that cite it."""
    if path not in cache:
        raw = path.read_text(encoding="utf-8", errors="replace")
        cache[path] = MirrorText(sha256=sha256_path(path), normalized=normalize(raw))
    return cache[path]


def resolve_artifact(
    citation_key: str, source_uri: str, roots: list[Path]
) -> Path | None:
    """Locate one pinned artifact across the mirror roots.

    Two roots, not a preference order with a guess: ``torchcell-library`` holds the OCR'd
    paper and SI, ``torchcell-raw`` holds the dataset's raw files, and a loader's
    ``source_uri`` names a path inside whichever of the two released it (``si/si1.md``
    against the library, ``data/dryad/README.md`` against the raw mirror).
    """
    for root in roots:
        path = root / citation_key / source_uri
        if path.exists():
            return path
    return None


def audit_loader_side(
    checks: list[QuoteCheck], roots: list[Path], cache: dict[Path, MirrorText]
) -> None:
    """Set each check's ``mirror_status`` from the artifact on disk."""
    for check in checks:
        path = resolve_artifact(check.citation_key, check.source_uri, roots)
        if path is None:
            check.mirror_status = MirrorStatus.artifact_missing
            continue
        check.mirror_root = path.parts[-len(Path(check.source_uri).parts) - 2]
        mirror = load_mirror(path, cache)
        check.sha256_actual = mirror.sha256
        if mirror.sha256 != check.sha256_pinned:
            check.mirror_status = MirrorStatus.sha256_drift
        elif normalize(check.quote) in mirror.normalized:
            check.mirror_status = MirrorStatus.pinned_and_present
        else:
            check.mirror_status = MirrorStatus.quote_absent_from_artifact


class QuoteProbe(BaseModel):
    """A quote prepared for the note scan: normalized text and the anchors that find it."""

    model_config = ConfigDict(extra="forbid")

    index: int
    normalized: str
    anchors: list[str]


def anchor_tokens(text: str) -> set[str]:
    """The long words of a passage, stripped of the punctuation a sentence wraps them in.

    Punctuation is stripped so an anchor survives the comma or closing quote that follows
    it in one of the two texts and not the other; ``_`` is kept, because it is part of a
    locus tag (``PP_2074``).
    """
    return {
        stripped
        for word in text.split(" ")
        if len(stripped := word.strip(_ANCHOR_TRIM)) >= _ANCHOR_MIN_LEN
    }


def build_probe(index: int, quote: str) -> QuoteProbe:
    """Pick the quote's longest words as the anchors that locate it inside a note."""
    normalized = normalize(quote)
    anchors = sorted(anchor_tokens(normalized), key=lambda word: (-len(word), word))
    return QuoteProbe(
        index=index, normalized=normalized, anchors=anchors[:_ANCHOR_COUNT]
    )


def best_window(probe: QuoteProbe, haystack: str) -> tuple[float, str]:
    """The passage of ``haystack`` most similar to the quote, and its CHARACTER ratio.

    Character-level and not word-level on purpose: an inline autofix deletes a space,
    which merges two tokens and costs a word-level alignment four words
    (``ΔPP_ 2428`` -> ``ΔPP_2428`` scored 0.50 and was missed), while the same edit
    moves a character ratio by one part in fifty. Only windows around an anchor
    occurrence are aligned, so the cost is set by how often the quote's rare words
    appear rather than by the haystack's length.
    """
    span = len(probe.normalized)
    offsets: list[int] = []
    for anchor in probe.anchors:
        start = haystack.find(anchor)
        while start >= 0 and len(offsets) < _MAX_WINDOWS:
            offsets.append(start)
            start = haystack.find(anchor, start + 1)
    if not offsets:
        return 0.0, ""
    best_ratio = 0.0
    best_text = ""
    for offset in sorted(set(offsets))[:_MAX_WINDOWS]:
        window = haystack[max(0, offset - span) : offset + 2 * span]
        matcher = difflib.SequenceMatcher(
            None, probe.normalized, window, autojunk=False
        )
        blocks = [block for block in matcher.get_matching_blocks() if block.size > 0]
        if not blocks:
            continue
        passage = window[blocks[0].b : blocks[-1].b + blocks[-1].size]
        ratio = difflib.SequenceMatcher(
            None, probe.normalized, passage, autojunk=False
        ).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_text = passage
    return best_ratio, best_text


class Difference(BaseModel):
    """One non-equal opcode between the mirror window and the note's rendition."""

    model_config = ConfigDict(extra="forbid")

    op: str
    at: int = Field(description="offset into the mirror window")
    mirror: str
    note: str


def difference_pieces(mirror: str, note: str) -> list[Difference]:
    """The non-equal opcodes between the mirror window and the note's rendition."""
    matcher = difflib.SequenceMatcher(None, mirror, note, autojunk=False)
    return [
        Difference(op=op, at=i1, mirror=mirror[i1:i2], note=note[j1:j2])
        for op, i1, i2, j1, j2 in matcher.get_opcodes()
        if op != "equal"
    ]


def edit_distance(mirror: str, note: str) -> int:
    """How many characters differ, counting a replacement by its longer side."""
    return sum(
        max(len(piece.mirror), len(piece.note))
        for piece in difference_pieces(mirror, note)
    )


def fold_typography(text: str) -> str:
    """Curly quotes and the dashes folded to their ASCII equivalents."""
    return "".join(_CURLY_TO_ASCII.get(char, char) for char in text)


def inside_math_span(mirror: str, pieces: list[Difference]) -> bool:
    r"""True when every difference sits inside a ``$...$`` span of the mirror text.

    MinerU spaces out the characters of a formula (``$3 3 0 \ E .$``), and a note that
    closes those spaces up changes only spaces, which would otherwise be charged to
    MD037. The enclosing dollar signs are what tell the two apart.
    """
    if "$" not in mirror or not pieces:
        return False
    spans: list[tuple[int, int]] = []
    opens = [index for index, char in enumerate(mirror) if char == "$"]
    for start, end in zip(opens[0::2], opens[1::2], strict=False):
        spans.append((start, end))
    if not spans:
        return False
    return all(any(start < piece.at <= end for start, end in spans) for piece in pieces)


def classify_drift(mirror: str, note: str) -> DriftSignature:
    r"""Attribute the difference between the mirror's bytes and the note's rendition.

    Order matters: MinerU's LaTeX is checked before the emphasis-space shape, because
    resolving ``$1 . 5 \%$`` to ``1.5%`` also deletes spaces and would otherwise be
    charged to MD037.
    """
    pieces = difference_pieces(mirror, note)
    mirror_side = "".join(piece.mirror for piece in pieces)
    note_side = "".join(piece.note for piece in pieces)
    # MD034 inserts the two angle brackets as SEPARATE opcodes, so the autolink is
    # recognized in the note's text with the brackets charged to the difference.
    autolinked = any(
        marker in note for marker in ("<http", "<doi", "<www", "<ftp")
    ) and {"<", ">"} & set(note_side)
    if autolinked:
        return DriftSignature.markdownlint_autolink
    if _MATH_MARKUP & set(mirror_side) or inside_math_span(mirror, pieces):
        return DriftSignature.ocr_math_markup
    for piece in pieces:
        if piece.mirror.strip() == "" and piece.mirror != "" and piece.note == "":
            return DriftSignature.markdownlint_emphasis_space
    if fold_typography(mirror) == fold_typography(note):
        return DriftSignature.typography_fold
    # MD004's fix swaps ONE marker character at the head of the line, which is what makes
    # it distinguishable from a `**` an author put around a phrase mid-quote.
    if (
        len(pieces) == 1
        and pieces[0].op == "replace"
        and pieces[0].at == 0
        and len(pieces[0].mirror) == 1
        and len(pieces[0].note) == 1
        and set(mirror_side + note_side) <= set("-*+")
    ):
        return DriftSignature.markdownlint_list_marker
    if mirror_side and not mirror_side.isascii() and note_side.isascii():
        return DriftSignature.glyph_transliteration
    if not mirror_side and set(note_side) <= set("*_` "):
        return DriftSignature.authored_markup
    if set(mirror_side + note_side) <= set("\"'` "):
        return DriftSignature.quotation_delimiter
    return DriftSignature.other


def mirror_window(mirror: MirrorText, note_text: str) -> str:
    """The mirror passage a falsified rendition drifted from, for the restore."""
    ratio, window = best_window(build_probe(-1, note_text), mirror.normalized)
    return window if ratio > 0.0 else ""


def pinned_artifacts(checks: list[QuoteCheck], roots: list[Path]) -> list[Path]:
    """Every mirror artifact any loader quote pins, in a deterministic order."""
    found: dict[Path, None] = {}
    for check in checks:
        path = resolve_artifact(check.citation_key, check.source_uri, roots)
        if path is not None:
            found[path] = None
    return sorted(found)


def in_any_pinned_artifact(
    text: str, artifacts: list[Path], cache: dict[Path, MirrorText]
) -> Path | None:
    """The first pinned artifact whose bytes hold ``text``, or None.

    Checked before a near rendition is called falsified, because papers from one lab
    share Methods boilerplate: the Carruthers 2025 note's Top3 sentence aligned at 0.98
    to the Lim 2025 quote of the SAME sentence and would otherwise have been reported as
    a dropped word, when it is verbatim in the Carruthers paper.
    """
    for path in artifacts:
        if text in load_mirror(path, cache).normalized:
            return path
    return None


def scan_notes(
    checks: list[QuoteCheck], cache: dict[Path, MirrorText], roots: list[Path]
) -> int:
    """Attach every note rendition of every quote, classified against the mirror."""
    probes = [build_probe(index, check.quote) for index, check in enumerate(checks)]
    artifacts = pinned_artifacts(checks, roots)
    notes = sorted(NOTES_DIR.rglob("*.md"))
    for note in notes:
        rel = note.relative_to(REPO).as_posix()
        excluded = bool(HOOK_EXCLUDED.match(rel))
        text = normalize(note.read_text(encoding="utf-8", errors="replace"))
        present = anchor_tokens(text)
        for probe in probes:
            check = checks[probe.index]
            # The anchor set gates everything: a quote present verbatim has all of its
            # own long words in the note, so a note missing half of them cannot hold it
            # and needs no substring search. Doing the substring search first instead
            # cost 1,695 x 1,546 scans of a 20 kB note, about 12 minutes per run.
            hits = sum(1 for anchor in probe.anchors if anchor in present)
            if probe.anchors:
                if hits * 2 < len(probe.anchors):
                    continue
            elif probe.normalized not in text:
                continue
            if probe.normalized in text:
                check.renditions.append(
                    NoteRendition(
                        note=rel,
                        hook_excluded=excluded,
                        status=NoteStatus.verbatim,
                        ratio=1.0,
                        note_text=probe.normalized,
                    )
                )
                continue
            if not probe.anchors:
                continue
            ratio, window = best_window(probe, text)
            if ratio < _NEAR_RATIO or not window:
                continue
            path = resolve_artifact(check.citation_key, check.source_uri, roots)
            mirror = load_mirror(path, cache) if path is not None else None
            if mirror is not None and window in mirror.normalized:
                check.renditions.append(
                    NoteRendition(
                        note=rel,
                        hook_excluded=excluded,
                        status=NoteStatus.variant_in_mirror,
                        ratio=round(ratio, 4),
                        note_text=window,
                    )
                )
                continue
            source = "" if mirror is None else mirror_window(mirror, window)
            trimmed_note = window.strip(_QUOTE_TRIM)
            trimmed_source = source.strip(_QUOTE_TRIM)
            if trimmed_note == trimmed_source:
                # The note's own closing quote mark is the whole difference.
                check.renditions.append(
                    NoteRendition(
                        note=rel,
                        hook_excluded=excluded,
                        status=NoteStatus.variant_in_mirror,
                        ratio=round(ratio, 4),
                        note_text=window,
                        mirror_text=source,
                    )
                )
                continue
            distance = edit_distance(trimmed_source, trimmed_note)
            close = difflib.SequenceMatcher(
                None, trimmed_source, trimmed_note, autojunk=False
            ).ratio()
            if distance > _MAX_EDIT_CHARS or close < _MIN_FALSIFIED_RATIO:
                check.renditions.append(
                    NoteRendition(
                        note=rel,
                        hook_excluded=excluded,
                        status=NoteStatus.paraphrase,
                        edit_chars=distance,
                        ratio=round(ratio, 4),
                        note_text=window,
                        mirror_text=source,
                    )
                )
                continue
            # Only a near-identical candidate pays for the cross-paper check, which scans
            # every pinned artifact: running it ahead of the paraphrase bound made the
            # whole-tree sweep pay it 200 times instead of 40.
            elsewhere = in_any_pinned_artifact(window, artifacts, cache)
            if elsewhere is not None:
                check.renditions.append(
                    NoteRendition(
                        note=rel,
                        hook_excluded=excluded,
                        status=NoteStatus.variant_in_mirror,
                        ratio=round(ratio, 4),
                        note_text=window,
                        mirror_text=elsewhere.name,
                    )
                )
                continue
            check.renditions.append(
                NoteRendition(
                    note=rel,
                    hook_excluded=excluded,
                    status=NoteStatus.falsified,
                    signature=classify_drift(trimmed_source, trimmed_note),
                    edit_chars=distance,
                    ratio=round(ratio, 4),
                    note_text=window,
                    mirror_text=source,
                )
            )
    return len(notes)


def build_report(
    checks: list[QuoteCheck], n_modules: int, n_notes: int, roots: list[Path]
) -> SweepReport:
    """Tally the sweep and keep the findings that need a human."""
    status_counts: dict[str, int] = {status.value: 0 for status in MirrorStatus}
    for check in checks:
        status_counts[check.mirror_status.value] += 1
    verbatim = sum(
        1
        for check in checks
        if any(r.status == NoteStatus.verbatim for r in check.renditions)
    )
    not_quoted = sum(1 for check in checks if not check.renditions)
    falsified = [check for check in checks if check.falsified]
    n_variant = sum(
        1
        for check in checks
        for r in check.renditions
        if r.status == NoteStatus.variant_in_mirror
    )
    n_paraphrase = sum(
        1
        for check in checks
        for r in check.renditions
        if r.status == NoteStatus.paraphrase
    )
    signature_counts: dict[str, int] = {
        signature.value: 0 for signature in DriftSignature
    }
    for check in checks:
        for rendition in check.falsified:
            assert rendition.signature is not None
            signature_counts[rendition.signature.value] += 1
    return SweepReport(
        mirror_roots=[str(root) for root in roots],
        n_modules_scanned=n_modules,
        n_notes_scanned=n_notes,
        n_quotes_checked=len(checks),
        mirror_status_counts=status_counts,
        n_quotes_verbatim_in_a_note=verbatim,
        n_quotes_not_quoted_in_any_note=not_quoted,
        n_falsified_renditions=sum(len(check.falsified) for check in falsified),
        n_variant_renditions=n_variant,
        n_paraphrase_renditions=n_paraphrase,
        falsified_signature_counts=signature_counts,
        falsified=sorted(falsified, key=lambda check: (check.module, check.quote)),
        unauditable=sorted(
            (
                check
                for check in checks
                if check.mirror_status != MirrorStatus.pinned_and_present
            ),
            key=lambda check: (check.citation_key, check.source_uri),
        ),
    )


def main() -> None:
    """Run the sweep and write the JSON report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--module",
        default=None,
        help="restrict to one loader module (debugging only: the cross-paper "
        "boilerplate check reads the artifacts the scanned modules pin, so it is "
        "complete only on a whole-tree run)",
    )
    parser.add_argument("--out", default=str(OUT), help="report path")
    args = parser.parse_args()

    load_dotenv()
    data_root = Path(os.environ["DATA_ROOT"])
    roots = [data_root / "torchcell-library", data_root / "torchcell-raw"]
    for root in roots:
        if not root.is_dir():
            raise FileNotFoundError(f"mirror not mounted: {root}")

    checks, n_modules = collect_quotes(args.module)
    cache: dict[Path, MirrorText] = {}
    audit_loader_side(checks, roots, cache)
    n_notes = scan_notes(checks, cache, roots)
    report = build_report(checks, n_modules, n_notes, roots)

    RESULTS.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(
        json.dumps(report.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"{report.n_quotes_checked} quotes from {report.n_modules_scanned} modules "
        f"against {report.n_notes_scanned} notes"
    )
    print(f"mirror status: {report.mirror_status_counts}")
    print(
        f"verbatim in a note: {report.n_quotes_verbatim_in_a_note}; "
        f"not quoted: {report.n_quotes_not_quoted_in_any_note}; "
        f"variant passages: {report.n_variant_renditions}; "
        f"paraphrases: {report.n_paraphrase_renditions}"
    )
    print(f"FALSIFIED note renditions: {report.n_falsified_renditions}")
    print(f"by signature: {report.falsified_signature_counts}")
    for check in report.falsified:
        for rendition in check.falsified:
            print(
                f"  [{rendition.signature}] {rendition.note}  "
                f"({check.citation_key}/{check.source_uri}, {rendition.edit_chars} chars)"
            )
            print(f"    note   : {rendition.note_text}")
            print(f"    mirror : {rendition.mirror_text}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
