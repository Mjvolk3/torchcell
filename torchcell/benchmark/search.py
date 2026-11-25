# torchcell/benchmark/search.py
# [[torchcell.benchmark.search]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/benchmark/search.py
# Test file: tests/torchcell/benchmark/test_search.py

"""Search and browse over the benchmark datasets ``tc-bench`` loaded at startup.

A :class:`DatasetIndex` is built once from the public dataset records and answers two
questions: which boards match a typed query, best first (:meth:`DatasetIndex.search`),
and how the boards group by organism and task (:meth:`DatasetIndex.facets`). It needs
numpy only; the embedding model is optional and named by ``TC_BENCH_SEARCH_MODEL``.

**The card.** Each dataset is described by these fields, each with a weight ``W_f``:

========================  ======
field                     ``W_f``
========================  ======
title, slug, organism     3
task_group, split_name,   2
targets
description,              1
loader_class,
citation_key
========================  ======

**Tokens.** Text is split at camel-case boundaries, lowercased, and cut into runs of
``[a-z0-9]``. A card token that mixes letters and digits also contributes its letter
and digit runs (``campos2018`` gives ``campos2018``, ``campos`` and ``2018``), so the
year or the name alone finds it. Query tokens are the runs alone, without that
expansion, deduplicated.

**Lexical score.** For a query with tokens ``Q`` and a card ``d`` with tokens
``T_f(d)`` in field ``f``::

    m(q, t)   = 1            if t == q
              = PREFIX_FACTOR (0.8)   if t starts with q and t != q
              = 0            otherwise
    s(q, d)   = max over f, t in T_f(d) of  W_f * m(q, t)
    lex(Q, d) = sum over q in Q of s(q, d)  /  (|Q| * 3)

so ``lex`` is in [0, 1]: 1 when every query token is a whole word of a weight-3 field,
and a query token that is only the start of a word (``morph`` in ``morphology``)
counts 0.8 of a whole one. A token absent from the card adds nothing, so a card
matching two of three tokens ranks below one matching all three.

**Modes.** ``lexical`` (no model): the score is ``lex`` and a card with ``lex = 0`` is
dropped. ``hybrid`` (a model is loaded): every card is embedded once at startup and
the query once per request; with ``cos`` the cosine similarity of the two
(L2-normalized) vectors::

    score = COSINE_WEIGHT * cos + (1 - COSINE_WEIGHT) * lex      (COSINE_WEIGHT = 0.5)

and a card is dropped when ``lex = 0`` and ``cos < COSINE_FLOOR`` (0.35). The mode is
reported by ``/health`` and in every search response; an unset model is the lexical
mode, never a silent stand-in for a model that failed to load (that fails startup).

**Order.** Ties, and the whole list for an empty query (or one with no letters or
digits, which has no tokens), follow the browse order: organism, task group, title,
slug, with a missing organism or task group after the named ones. An empty query
scores every card 0.
"""

from __future__ import annotations

import bisect
import os
import re
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field

from torchcell.benchmark.bundle import BenchmarkDatasetPublic

SEARCH_MODEL_VAR = "TC_BENCH_SEARCH_MODEL"
SearchMode = Literal["lexical", "hybrid"]

FIELD_WEIGHTS: dict[str, float] = {
    "title": 3.0,
    "slug": 3.0,
    "organism": 3.0,
    "task_group": 2.0,
    "split_name": 2.0,
    "targets": 2.0,
    "description": 1.0,
    "loader_class": 1.0,
    "citation_key": 1.0,
}
MAX_FIELD_WEIGHT = max(FIELD_WEIGHTS.values())
PREFIX_FACTOR = 0.8
COSINE_WEIGHT = 0.5
COSINE_FLOOR = 0.35

_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_WORD = re.compile(r"[a-z0-9]+")
_RUN = re.compile(r"[a-z]+|[0-9]+")
_TRAILING_WORD = re.compile(r"[A-Za-z0-9]+$")
_GROUP_TITLE_STRIP = " ,;:-(/"

FloatMatrix = npt.NDArray[np.float64]


def _words(text: str) -> list[str]:
    return _WORD.findall(_CAMEL_BOUNDARY.sub(" ", text).lower())


def card_tokens(text: str) -> list[str]:
    """Tokens of card text: the words plus the letter and digit runs of mixed words.

    Args:
        text: One field of a dataset card.

    Returns:
        The tokens, in order, with repeats.
    """
    tokens: list[str] = []
    for word in _words(text):
        tokens.append(word)
        runs = _RUN.findall(word)
        if len(runs) > 1:
            tokens.extend(runs)
    return tokens


def query_tokens(query: str) -> list[str]:
    """Tokens of a query: its words, deduplicated in order of first appearance.

    Args:
        query: The text the user typed.

    Returns:
        The distinct words; empty for a query with no letters or digits.
    """
    return list(dict.fromkeys(_words(query)))


def card_fields(dataset: BenchmarkDatasetPublic) -> dict[str, str]:
    """The text of each card field of ``dataset`` (empty for an unset field).

    Args:
        dataset: A public dataset record.

    Returns:
        ``{field: text}`` over the keys of :data:`FIELD_WEIGHTS`.
    """
    return {
        "title": dataset.title,
        "slug": dataset.slug.replace("-", " "),
        "organism": dataset.organism or "",
        "task_group": dataset.task_group or "",
        "split_name": dataset.split_name or "",
        "targets": " ".join(dataset.targets),
        "description": dataset.description,
        "loader_class": dataset.loader_class,
        "citation_key": dataset.citation_key,
    }


def card_text(dataset: BenchmarkDatasetPublic) -> str:
    """The text of the whole card, the string that is embedded in hybrid mode.

    Args:
        dataset: A public dataset record.

    Returns:
        The non-empty card fields joined by newlines, in :data:`FIELD_WEIGHTS` order.
    """
    return "\n".join(text for text in card_fields(dataset).values() if text)


def browse_key(
    dataset: BenchmarkDatasetPublic,
) -> tuple[bool, str, bool, str, str, str]:
    """Sort key of the browse order: organism, task group, title, slug.

    Args:
        dataset: A public dataset record.

    Returns:
        A key that puts a missing organism or task group after the named ones.
    """
    return (
        dataset.organism is None,
        dataset.organism or "",
        dataset.task_group is None,
        dataset.task_group or "",
        dataset.title,
        dataset.slug,
    )


def group_title(titles: Sequence[str], task_group: str) -> str:
    """Display title of a task group: what its boards' titles have in common.

    The title of a lone board (or of boards that share one title) is used as is.
    Otherwise it is the boards' common prefix, cut back to a whole word and stripped
    of trailing spaces and ``, ; : - ( /``. When nothing is left, ``task_group``.

    Args:
        titles: The titles of the group's boards (at least one).
        task_group: The group's name, the title when the boards share no word.

    Returns:
        The display title.
    """
    if len(set(titles)) == 1:
        return titles[0]
    prefix = os.path.commonprefix(list(titles))
    cuts_a_word = any(
        len(title) > len(prefix) and title[len(prefix)].isalnum() for title in titles
    )
    if cuts_a_word:
        prefix = _TRAILING_WORD.sub("", prefix)
    prefix = prefix.rstrip(_GROUP_TITLE_STRIP)
    return prefix or task_group


def _unit_rows(matrix: FloatMatrix) -> FloatMatrix:
    """Each row scaled to unit L2 norm; an all-zero row stays zero (cosine 0)."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    unit: FloatMatrix = np.divide(
        matrix, norms, out=np.zeros_like(matrix), where=norms > 0
    )
    return unit


class Embedder(Protocol):
    """Turns texts into vectors, one row per text."""

    def embed(self, texts: list[str]) -> npt.NDArray[Any]:
        """Embed ``texts``.

        Args:
            texts: The texts to embed.

        Returns:
            A ``(len(texts), dim)`` array.
        """
        ...


class Model2VecEmbedder:
    """A Model2Vec static embedding model loaded from a local directory.

    ``model2vec`` is imported here, not at module import, so the lexical mode needs
    only numpy. The directory must exist: a missing one raises rather than letting
    the library read the path as a Hugging Face repository id and go to the network.
    """

    def __init__(self, model_path: Path) -> None:
        """Load the model at ``model_path``.

        Args:
            model_path: A local directory holding a Model2Vec model
                (``config.json``, ``model.safetensors``, ``tokenizer.json``).

        Raises:
            FileNotFoundError: When ``model_path`` is not a directory.
        """
        if not model_path.is_dir():
            raise FileNotFoundError(
                f"{SEARCH_MODEL_VAR}={model_path} is not a directory holding a "
                "Model2Vec model"
            )
        from model2vec import StaticModel

        self.model_path = model_path
        self._model: Any = StaticModel.from_pretrained(str(model_path))

    def embed(self, texts: list[str]) -> npt.NDArray[Any]:
        """Embed ``texts`` with the loaded model.

        Args:
            texts: The texts to embed.

        Returns:
            A ``(len(texts), dim)`` float array.
        """
        return np.asarray(self._model.encode(texts), dtype=np.float64)


class SearchHit(BaseModel):
    """One dataset in a search answer with its score (0 for an empty query)."""

    model_config = ConfigDict(frozen=True)

    dataset: BenchmarkDatasetPublic
    score: float


class SearchPage(BaseModel):
    """One page of a search, best first."""

    model_config = ConfigDict(frozen=True)

    mode: SearchMode = Field(
        description="lexical: word and prefix matches only. hybrid: plus embeddings."
    )
    total: int = Field(description="Matching datasets before paging.")
    limit: int
    offset: int
    results: list[SearchHit]


class TaskGroupFacet(BaseModel):
    """One task of an organism and how many boards it has."""

    model_config = ConfigDict(frozen=True)

    task_group: str
    title: str = Field(description="Display title; see group_title.")
    count: int


class OrganismFacet(BaseModel):
    """One organism, its board count and its task groups."""

    model_config = ConfigDict(frozen=True)

    organism: str
    count: int
    task_groups: list[TaskGroupFacet]
    n_without_task_group: int = Field(
        description="Boards of this organism with no task group; reachable by search."
    )


class DatasetFacets(BaseModel):
    """The browse tree: organisms, then their task groups."""

    model_config = ConfigDict(frozen=True)

    total: int
    organisms: list[OrganismFacet]
    n_without_organism: int = Field(
        description="Boards with no organism set; reachable by search."
    )


class DatasetIndex:
    """The searchable index of the loaded datasets, built once at startup."""

    def __init__(
        self,
        datasets: Sequence[BenchmarkDatasetPublic],
        embedder: Embedder | None = None,
    ) -> None:
        """Index ``datasets``; embed every card when ``embedder`` is given.

        Args:
            datasets: The public dataset records.
            embedder: The embedding model of the hybrid mode; ``None`` is lexical.

        Raises:
            ValueError: When the embedder returns a matrix of the wrong shape.
        """
        self._datasets = sorted(datasets, key=browse_key)
        self._embedder = embedder
        # token -> {dataset position: largest field weight the token appears in}
        postings: dict[str, dict[int, float]] = defaultdict(dict)
        for position, dataset in enumerate(self._datasets):
            for field, text in card_fields(dataset).items():
                weight = FIELD_WEIGHTS[field]
                for token in card_tokens(text):
                    held = postings[token].get(position, 0.0)
                    postings[token][position] = max(held, weight)
        self._postings = dict(postings)
        self._vocabulary = sorted(self._postings)
        self._embeddings: FloatMatrix | None = None
        if embedder is not None and self._datasets:
            matrix = np.asarray(
                embedder.embed([card_text(d) for d in self._datasets]), dtype=np.float64
            )
            if matrix.ndim != 2 or matrix.shape[0] != len(self._datasets):
                raise ValueError(
                    f"embedder returned shape {matrix.shape} for "
                    f"{len(self._datasets)} cards"
                )
            self._embeddings = _unit_rows(matrix)

    @property
    def mode(self) -> SearchMode:
        """``hybrid`` when an embedder was given, else ``lexical``."""
        return "lexical" if self._embedder is None else "hybrid"

    def __len__(self) -> int:
        """The number of indexed datasets."""
        return len(self._datasets)

    def lexical_scores(self, tokens: Sequence[str]) -> FloatMatrix:
        """``lex(Q, d)`` of every dataset, in browse order (see the module docstring).

        Args:
            tokens: The query tokens ``Q`` (:func:`query_tokens`), at least one.

        Returns:
            A vector of length ``len(self)`` with values in [0, 1].
        """
        total = np.zeros(len(self._datasets), dtype=np.float64)
        for query_token in tokens:
            best = np.zeros(len(self._datasets), dtype=np.float64)
            start = bisect.bisect_left(self._vocabulary, query_token)
            for token in self._vocabulary[start:]:
                if not token.startswith(query_token):
                    break
                strength = 1.0 if token == query_token else PREFIX_FACTOR
                for position, weight in self._postings[token].items():
                    best[position] = max(best[position], weight * strength)
            total += best
        return total / (len(tokens) * MAX_FIELD_WEIGHT)

    def cosine_scores(self, query: str) -> FloatMatrix:
        """Cosine similarity of ``query`` to every card, in browse order.

        Args:
            query: The query text (embedded as typed).

        Returns:
            A vector of length ``len(self)``.

        Raises:
            ValueError: In lexical mode, which has no embeddings.
        """
        if self._embedder is None or self._embeddings is None:
            raise ValueError("cosine scores need the hybrid mode")
        vector = np.asarray(self._embedder.embed([query]), dtype=np.float64)
        cosine: FloatMatrix = self._embeddings @ _unit_rows(vector)[0]
        return cosine

    def search(
        self,
        query: str,
        *,
        organism: str | None = None,
        task_group: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> SearchPage:
        """The datasets matching ``query`` and the filters, best first, one page.

        Args:
            query: The typed text; an empty one (no tokens) lists every dataset that
                passes the filters in browse order with score 0.
            organism: Keep only datasets with exactly this organism.
            task_group: Keep only datasets with exactly this task group.
            limit: Page size.
            offset: Results skipped before the page.

        Returns:
            The page, with ``total`` counting every match.
        """
        candidates = [
            position
            for position, dataset in enumerate(self._datasets)
            if (organism is None or dataset.organism == organism)
            and (task_group is None or dataset.task_group == task_group)
        ]
        tokens = query_tokens(query)
        if not tokens or not candidates:
            ranked = [(position, 0.0) for position in candidates]
        else:
            lexical = self.lexical_scores(tokens)
            if self._embeddings is None:
                scores = lexical
                keep = lexical > 0
            else:
                cosine = self.cosine_scores(query)
                scores = COSINE_WEIGHT * cosine + (1 - COSINE_WEIGHT) * lexical
                keep = (lexical > 0) | (cosine >= COSINE_FLOOR)
            ranked = sorted(
                (
                    (position, float(scores[position]))
                    for position in candidates
                    if keep[position]
                ),
                key=lambda pair: (-pair[1], pair[0]),
            )
        page = ranked[offset : offset + limit]
        return SearchPage(
            mode=self.mode,
            total=len(ranked),
            limit=limit,
            offset=offset,
            results=[
                SearchHit(dataset=self._datasets[position], score=score)
                for position, score in page
            ],
        )

    def facets(self) -> DatasetFacets:
        """Board counts per organism and, within each, per task group.

        Organisms are sorted by name and task groups by display title, then name.

        Returns:
            The browse tree.
        """
        by_organism: dict[str, list[BenchmarkDatasetPublic]] = defaultdict(list)
        n_without_organism = 0
        for dataset in self._datasets:
            if dataset.organism is None:
                n_without_organism += 1
            else:
                by_organism[dataset.organism].append(dataset)
        organisms = []
        for organism in sorted(by_organism):
            members = by_organism[organism]
            titles_by_group: dict[str, list[str]] = defaultdict(list)
            for dataset in members:
                if dataset.task_group is not None:
                    titles_by_group[dataset.task_group].append(dataset.title)
            task_groups = sorted(
                (
                    TaskGroupFacet(
                        task_group=name,
                        title=group_title(titles, name),
                        count=len(titles),
                    )
                    for name, titles in titles_by_group.items()
                ),
                key=lambda facet: (facet.title, facet.task_group),
            )
            grouped = sum(facet.count for facet in task_groups)
            organisms.append(
                OrganismFacet(
                    organism=organism,
                    count=len(members),
                    task_groups=task_groups,
                    n_without_task_group=len(members) - grouped,
                )
            )
        return DatasetFacets(
            total=len(self._datasets),
            organisms=organisms,
            n_without_organism=n_without_organism,
        )


def embedder_from_path(model_path: Path | None) -> Embedder | None:
    """The embedder ``TC_BENCH_SEARCH_MODEL`` names, or ``None`` when it is unset.

    Args:
        model_path: The configured model directory, or ``None``.

    Returns:
        A :class:`Model2VecEmbedder`, or ``None`` for the lexical mode.

    Raises:
        FileNotFoundError: When the path is set but is not a directory; any error of
            the library loading the model propagates, so startup fails loudly.
    """
    return None if model_path is None else Model2VecEmbedder(model_path)
