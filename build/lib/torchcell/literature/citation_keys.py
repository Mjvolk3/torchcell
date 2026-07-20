# torchcell/literature/citation_keys.py
# [[torchcell.literature.citation_keys]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/literature/citation_keys.py
# Test file: tests/torchcell/literature/test_citation_keys.py

"""Citation-key generation and normalization helpers."""

import re

from unidecode import unidecode

# Better BibTeX default skipwords (the leading articles/conjunctions/prepositions
# dropped from the title segment of a citation key).
_SKIP_WORDS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "but",
    "nor",
    "for",
    "on",
    "at",
    "to",
    "from",
    "by",
    "of",
    "in",
    "into",
    "with",
    "as",
    "is",
    "be",
}


def generate_citation_key(creators: list[str], published: str, title: str) -> str:
    """Generate a Better BibTeX-style citation key from paper metadata.

    Format: ``{firstauthorlastname}{TitleWord}{TitleWord}...{YYYY}`` matching the
    Better BibTeX default
    ``[auth:lower][Title:skipwords:nopunct:fold:condense=''][year]``.

    The key is a pure function of the metadata the Zotero web API returns
    (``creators``, ``date``, ``title``), so it reproduces identically on every
    run and host -- the property the literature mirror relies on to key each
    paper's artifact directory deterministically.

    Non-ASCII author surnames and title words are transliterated to ASCII via
    :func:`unidecode` before stripping, so ``Müller`` yields ``muller`` rather
    than the lossy ``mller``. Output is restricted to ``[A-Za-z0-9]`` and always
    begins with a letter (the ``unknown`` fallback guarantees it).

    Examples:
        ``["John Smith"], "2022", "Biodegradation of PET"`` ->
        ``smithBiodegradationPET2022``

    Args:
        creators: Author display names (e.g. ``["John Smith", "Jane Doe"]``).
        published: Publication date string containing a 4-digit year.
        title: Paper title.

    Returns:
        Citation key string (e.g. ``"costanzoGlobalGeneticInteraction2016"``).
    """
    # First-author last name: transliterate to ASCII, take the last whitespace
    # token, keep letters only, lowercase. Falls back to "unknown" (also the
    # leading-letter guard -- the key can never start with a digit or be empty).
    if creators:
        first_author = unidecode(creators[0]).strip()
        last_name = first_author.rsplit(" ", 1)[-1]
        last_name = re.sub(r"[^a-zA-Z]", "", last_name).lower() or "unknown"
    else:
        last_name = "unknown"

    # Year: first 4-digit run, else "XXXX".
    year_match = re.search(r"\d{4}", published or "")
    year = year_match.group() if year_match else "XXXX"

    # Title: transliterate, then up to 5 significant words, each with its leading
    # character upper-cased (acronym casing preserved in the remainder).
    words = re.findall(r"[a-zA-Z0-9]+", unidecode(title or ""))
    title_parts: list[str] = []
    for word in words:
        if word.lower() in _SKIP_WORDS:
            continue
        title_parts.append(word[0].upper() + word[1:])
        if len(title_parts) >= 5:
            break
    title_str = "".join(title_parts)

    return f"{last_name}{title_str}{year}"
