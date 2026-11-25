# torchcell/benchmark/search.py
# [[torchcell.benchmark.search]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/benchmark/search.py
# Test file: tests/torchcell/benchmark/test_search.py

"""Search and browse over the benchmark datasets ``tc-bench`` loaded at startup.

A :class:`DatasetIndex` is built once from the public dataset records and answers
three questions: which boards match a typed query, best first
(:meth:`DatasetIndex.search`); which tasks match it, each with its split boards
(:meth:`DatasetIndex.search_tasks`); and how the boards group by organism and task
(:meth:`DatasetIndex.facets`). It needs numpy only; the embedding model is optional
and named by ``TC_BENCH_SEARCH_MODEL``.

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
digits, which has no tokens), follow the browse order: organism, task group, the
board's place in its task (:func:`board_order`), title, slug, with a missing organism
or task group after the named ones. An empty query scores every card 0, so paging
``/datasets/search`` with a ``task_group`` lists a task's boards in board order.

**Tasks.** :meth:`DatasetIndex.search_tasks` ranks tasks rather than boards, so a query
that matches every split of one task answers that task once. A task is the boards
sharing a ``task_group``; a board without one is a task of its own, keyed by its slug.
A task card has no split field and no board slug, with these fields and weights:

========================  ======
field                     ``W_f``
========================  ======
title, task_group,        3
organism
targets                   2
description,              1
loader_class,
citation_key
========================  ======

``title`` is :func:`group_title` of the boards' titles, ``task_group`` the task's key
with ``-`` read as a space, ``targets`` every target of the boards in first-seen order,
``loader_class`` and ``citation_key`` their distinct values, and ``description``
:func:`task_description`: the sentences of the first board's description that every
board's description contains and that name no board's ``split_name``. Tokens, the
lexical score, the modes, the cosine floor and the weighting are those of the boards.
Ties, and an empty query, follow the task order: organism (missing last), title, key.
A task lists its boards in :func:`board_order`, at most :data:`MAX_TASK_BOARDS` of
them; ``n_boards`` is the true count, and ``/datasets/search`` with ``task_group``
pages the rest.
"""

from __future__ import annotations

import bisect
import os
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
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
TASK_FIELD_WEIGHTS: dict[str, float] = {
    "title": 3.0,
    "task_group": 3.0,
    "organism": 3.0,
    "targets": 2.0,
    "description": 1.0,
    "loader_class": 1.0,
    "citation_key": 1.0,
}
PREFIX_FACTOR = 0.8
COSINE_WEIGHT = 0.5
COSINE_FLOOR = 0.35
MAX_TASK_BOARDS = 50

_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_WORD = re.compile(r"[a-z0-9]+")
_RUN = re.compile(r"[a-z]+|[0-9]+")
_TRAILING_WORD = re.compile(r"[A-Za-z0-9_]+$")
_GROUP_TITLE_STRIP = " ,;:-(/"
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+")
_SPLIT_NAME = re.compile(r"^(?P<family>.+?)_s(?P<seed>[0-9]+)$")
_SEQID_FAMILY = re.compile(r"^seqid(?P<identity>[0-9]+)$")
_SEQID_GENE_HOLDOUT_FAMILY = re.compile(r"^seqid(?P<identity>[0-9]+)_gene_holdout$")

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
    dataset: BenchmarkDatasetPublic, board_rank: int
) -> tuple[bool, str, bool, str, int, str, str]:
    """Sort key of the browse order: organism, task group, board rank, title, slug.

    Args:
        dataset: A public dataset record.
        board_rank: The dataset's position in its task's :func:`board_order` (0 for a
            board that is a task of its own).

    Returns:
        A key that puts a missing organism or task group after the named ones.
    """
    return (
        dataset.organism is None,
        dataset.organism or "",
        dataset.task_group is None,
        dataset.task_group or "",
        board_rank,
        dataset.title,
        dataset.slug,
    )


def _is_word_char(char: str) -> bool:
    return char.isalnum() or char == "_"


def group_title(titles: Sequence[str], task_group: str) -> str:
    """Display title of a task group: what its boards' titles have in common.

    The title of a lone board (or of boards that share one title) is used as is.
    Otherwise it is the boards' common prefix, cut back to a whole word (letters,
    digits and ``_``, so a split name such as ``random_s0`` is one word) and stripped
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
        len(title) > len(prefix) and _is_word_char(title[len(prefix)])
        for title in titles
    )
    if cuts_a_word:
        prefix = _TRAILING_WORD.sub("", prefix)
    prefix = prefix.rstrip(_GROUP_TITLE_STRIP)
    return prefix or task_group


def task_description(boards: Sequence[BenchmarkDatasetPublic]) -> str:
    """The description of a task: the first board's, without its split-specific part.

    The first board's description is cut into sentences at ``.``, ``!`` or ``?``
    followed by white space. A sentence is kept when every board's description
    contains it and it names no board's ``split_name``; the kept sentences are joined
    by one space. So ``"Growth rate. One target. Split random_s0 of the task t."``
    keeps its first two sentences.

    Args:
        boards: The task's boards (at least one), first board first.

    Returns:
        The kept sentences, or ``""`` when none is kept.
    """
    sentences = [s for s in _SENTENCE_BREAK.split(boards[0].description.strip()) if s]
    split_names = [b.split_name for b in boards if b.split_name is not None]
    return " ".join(
        sentence
        for sentence in sentences
        if all(sentence in board.description for board in boards[1:])
        and not any(name in sentence for name in split_names)
    )


def split_parts(split_name: str | None) -> tuple[str | None, int]:
    """The family and seed of a split name: ``seqid30_s0`` is ``("seqid30", 0)``.

    Args:
        split_name: A board's ``split_name``.

    Returns:
        ``(family, seed)``; a name without a ``_s<seed>`` suffix is its own family
        with seed -1, and ``None`` is ``(None, -1)``.
    """
    if split_name is None:
        return None, -1
    match = _SPLIT_NAME.match(split_name)
    if match is None:
        return split_name, -1
    return match["family"], int(match["seed"])


def family_rank(family: str | None, unknown_rank: int) -> tuple[int, int, int]:
    """Sort key of a split family within a task.

    Order: no split name; ``random``; ``seqid<N>`` by ascending ``N``;
    ``gene_holdout``, then ``seqid<N>_gene_holdout`` by ascending ``N``; ``position``;
    then any other family by ``unknown_rank``.

    Args:
        family: The family of :func:`split_parts`.
        unknown_rank: The family's first-seen position among the task's boards, used
            only for a family outside the list above.

    Returns:
        A key that sorts the families in that order.
    """
    if family is None:
        return (0, 0, 0)
    if family == "random":
        return (1, 0, 0)
    seqid = _SEQID_FAMILY.match(family)
    if seqid is not None:
        return (2, int(seqid["identity"]), 0)
    if family == "gene_holdout":
        return (3, 0, 0)
    seqid_gene = _SEQID_GENE_HOLDOUT_FAMILY.match(family)
    if seqid_gene is not None:
        return (3, 1, int(seqid_gene["identity"]))
    if family == "position":
        return (4, 0, 0)
    return (5, unknown_rank, 0)


def board_order(
    boards: Sequence[BenchmarkDatasetPublic],
) -> list[BenchmarkDatasetPublic]:
    """A task's boards by split family (:func:`family_rank`), then seed, then slug.

    Args:
        boards: The task's boards in the order the bundles were loaded, which sets the
            first-seen order of families :func:`family_rank` does not name.

    Returns:
        The boards in board order.
    """
    first_seen: dict[str | None, int] = {}
    for board in boards:
        first_seen.setdefault(split_parts(board.split_name)[0], len(first_seen))

    def key(board: BenchmarkDatasetPublic) -> tuple[tuple[int, int, int], int, str]:
        family, seed = split_parts(board.split_name)
        return family_rank(family, first_seen[family]), seed, board.slug

    return sorted(boards, key=key)


class BenchmarkTask(BaseModel):
    """A task: the boards sharing a task group, or one board that has none."""

    model_config = ConfigDict(frozen=True)

    key: str = Field(description="The task group, or the slug of a board without one.")
    task_group: str | None
    title: str = Field(description="Display title; see group_title.")
    organism: str | None
    boards: list[BenchmarkDatasetPublic] = Field(description="In board_order.")


def task_key(task: BenchmarkTask) -> tuple[bool, str, str, str]:
    """Sort key of the task order: organism (missing last), title, key.

    Args:
        task: A task.

    Returns:
        The key.
    """
    return (task.organism is None, task.organism or "", task.title, task.key)


def tasks_of(datasets: Sequence[BenchmarkDatasetPublic]) -> list[BenchmarkTask]:
    """Group ``datasets`` into tasks, in task order (:func:`task_key`).

    Args:
        datasets: The public dataset records in the order the bundles were loaded.

    Returns:
        The tasks.

    Raises:
        ValueError: When the boards of one task group name different organisms, or a
            board without a task group has a slug that is also a task group's name.
    """
    members: dict[str, list[BenchmarkDatasetPublic]] = defaultdict(list)
    for dataset in datasets:
        if dataset.task_group is not None:
            members[dataset.task_group].append(dataset)
    tasks: list[BenchmarkTask] = []
    for name, boards in members.items():
        if len({board.organism for board in boards}) > 1:
            organisms = sorted({str(board.organism) for board in boards})
            raise ValueError(
                f"task group {name!r} has boards of different organisms: {organisms}"
            )
        tasks.append(
            BenchmarkTask(
                key=name,
                task_group=name,
                title=group_title([board.title for board in boards], name),
                organism=boards[0].organism,
                boards=board_order(boards),
            )
        )
    for dataset in datasets:
        if dataset.task_group is not None:
            continue
        if dataset.slug in members:
            raise ValueError(
                f"board {dataset.slug!r} has no task group but its slug names one"
            )
        tasks.append(
            BenchmarkTask(
                key=dataset.slug,
                task_group=None,
                title=dataset.title,
                organism=dataset.organism,
                boards=[dataset],
            )
        )
    return sorted(tasks, key=task_key)


def task_card_fields(task: BenchmarkTask) -> dict[str, str]:
    """The text of each task card field (see the module docstring).

    Args:
        task: A task.

    Returns:
        ``{field: text}`` over the keys of :data:`TASK_FIELD_WEIGHTS`.
    """

    def distinct(values: Sequence[str]) -> str:
        return " ".join(dict.fromkeys(values))

    return {
        "title": task.title,
        "task_group": task.key.replace("-", " "),
        "organism": task.organism or "",
        "targets": distinct([t for board in task.boards for t in board.targets]),
        "description": task_description(task.boards),
        "loader_class": distinct([board.loader_class for board in task.boards]),
        "citation_key": distinct([board.citation_key for board in task.boards]),
    }


def task_card_text(task: BenchmarkTask) -> str:
    """The text of the whole task card, the string embedded in hybrid mode.

    Args:
        task: A task.

    Returns:
        The non-empty task card fields joined by newlines, in
        :data:`TASK_FIELD_WEIGHTS` order.
    """
    return "\n".join(text for text in task_card_fields(task).values() if text)


class _Postings:
    """The lexical half of an index over one kind of card (boards or tasks)."""

    def __init__(
        self, cards: Sequence[Mapping[str, str]], weights: Mapping[str, float]
    ) -> None:
        """Index ``cards``, each ``{field: text}`` with a weight per field.

        Args:
            cards: The cards, in the order their scores are returned.
            weights: The weight of each field.
        """
        self._n_cards = len(cards)
        self._max_weight = max(weights.values())
        # token -> {card position: largest field weight the token appears in}
        postings: dict[str, dict[int, float]] = defaultdict(dict)
        for position, fields in enumerate(cards):
            for field, text in fields.items():
                weight = weights[field]
                for token in card_tokens(text):
                    held = postings[token].get(position, 0.0)
                    postings[token][position] = max(held, weight)
        self._postings = dict(postings)
        self._vocabulary = sorted(self._postings)

    def scores(self, tokens: Sequence[str]) -> FloatMatrix:
        """``lex(Q, d)`` of every card (see the module docstring).

        Args:
            tokens: The query tokens ``Q`` (:func:`query_tokens`), at least one.

        Returns:
            A vector over the cards with values in [0, 1].
        """
        total = np.zeros(self._n_cards, dtype=np.float64)
        for query_token in tokens:
            best = np.zeros(self._n_cards, dtype=np.float64)
            start = bisect.bisect_left(self._vocabulary, query_token)
            for token in self._vocabulary[start:]:
                if not token.startswith(query_token):
                    break
                strength = 1.0 if token == query_token else PREFIX_FACTOR
                for position, weight in self._postings[token].items():
                    best[position] = max(best[position], weight * strength)
            total += best
        return total / (len(tokens) * self._max_weight)


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


class TaskBoard(BaseModel):
    """One board of a task in a task search answer."""

    model_config = ConfigDict(frozen=True)

    slug: str
    split_name: str | None
    title: str
    n_train: int
    n_val: int
    n_test: int


class TaskHit(BaseModel):
    """One task in a task search answer with its score (0 for an empty query)."""

    model_config = ConfigDict(frozen=True)

    key: str = Field(description="The task group, or the slug of a board without one.")
    task_group: str | None
    title: str = Field(description="Display title; see group_title.")
    organism: str | None
    n_boards: int = Field(description="Every board of the task, listed or not.")
    score: float
    boards: list[TaskBoard] = Field(
        description=(
            f"The first {MAX_TASK_BOARDS} boards in board order (split family, then "
            "seed); page the rest with /datasets/search and task_group."
        )
    )


class TaskPage(BaseModel):
    """One page of a task search, best first."""

    model_config = ConfigDict(frozen=True)

    mode: SearchMode = Field(
        description="lexical: word and prefix matches only. hybrid: plus embeddings."
    )
    total: int = Field(description="Matching tasks before paging.")
    limit: int
    offset: int
    results: list[TaskHit]


class TaskGroupFacet(BaseModel):
    """One task of an organism and how many boards it has."""

    model_config = ConfigDict(frozen=True)

    task_group: str
    title: str = Field(description="Display title; see group_title.")
    count: int


class OrganismFacet(BaseModel):
    """One organism, its board and task counts and its task groups."""

    model_config = ConfigDict(frozen=True)

    organism: str
    count: int
    n_tasks: int = Field(
        description="Tasks of this organism: its task groups plus its boards with none."
    )
    task_groups: list[TaskGroupFacet]
    n_without_task_group: int = Field(
        description="Boards of this organism with no task group; reachable by search."
    )


class DatasetFacets(BaseModel):
    """The browse tree: organisms, then their task groups."""

    model_config = ConfigDict(frozen=True)

    total: int
    n_tasks: int = Field(description="Every task, with or without an organism.")
    organisms: list[OrganismFacet]
    n_without_organism: int = Field(
        description="Boards with no organism set; reachable by search."
    )


def _query_vector(embedder: Embedder, query: str) -> FloatMatrix:
    """The unit vector of ``query`` (all zeros when the model gives a zero vector)."""
    vector = np.asarray(embedder.embed([query]), dtype=np.float64)
    unit: FloatMatrix = _unit_rows(vector)[0]
    return unit


def _ranked(
    candidates: list[int], lexical: FloatMatrix, cosine: FloatMatrix | None
) -> list[tuple[int, float]]:
    """``(position, score)`` of the candidates that match, best first.

    Args:
        candidates: Card positions that pass the filters, in card order.
        lexical: ``lex`` of every card.
        cosine: Cosine of every card to the query (hybrid), or ``None`` (lexical).

    Returns:
        The kept candidates by descending score, ties in card order.
    """
    if cosine is None:
        scores = lexical
        keep = lexical > 0
    else:
        scores = COSINE_WEIGHT * cosine + (1 - COSINE_WEIGHT) * lexical
        keep = (lexical > 0) | (cosine >= COSINE_FLOOR)
    return sorted(
        (
            (position, float(scores[position]))
            for position in candidates
            if keep[position]
        ),
        key=lambda pair: (-pair[1], pair[0]),
    )


class DatasetIndex:
    """The searchable index of the loaded datasets and their tasks, built at startup."""

    def __init__(
        self,
        datasets: Sequence[BenchmarkDatasetPublic],
        embedder: Embedder | None = None,
    ) -> None:
        """Index ``datasets`` and their tasks; embed every card when given ``embedder``.

        Board cards and task cards are embedded in one call, boards first.

        Args:
            datasets: The public dataset records, in the order the bundles were
                loaded (it sets the first-seen order of :func:`board_order`).
            embedder: The embedding model of the hybrid mode; ``None`` is lexical.

        Raises:
            ValueError: When the embedder returns a matrix of the wrong shape, or the
                datasets do not group into tasks (:func:`tasks_of`).
        """
        self._tasks = tasks_of(datasets)
        board_rank = {
            board.slug: rank
            for task in self._tasks
            for rank, board in enumerate(task.boards)
        }
        self._datasets = sorted(
            datasets, key=lambda d: browse_key(d, board_rank[d.slug])
        )
        self._embedder = embedder
        self._board_postings = _Postings(
            [card_fields(d) for d in self._datasets], FIELD_WEIGHTS
        )
        self._task_postings = _Postings(
            [task_card_fields(t) for t in self._tasks], TASK_FIELD_WEIGHTS
        )
        self._embeddings: FloatMatrix | None = None
        self._task_embeddings: FloatMatrix | None = None
        if embedder is not None and self._datasets:
            texts = [card_text(d) for d in self._datasets]
            texts += [task_card_text(t) for t in self._tasks]
            matrix = np.asarray(embedder.embed(texts), dtype=np.float64)
            if matrix.ndim != 2 or matrix.shape[0] != len(texts):
                raise ValueError(
                    f"embedder returned shape {matrix.shape} for {len(texts)} cards"
                )
            unit = _unit_rows(matrix)
            self._embeddings = unit[: len(self._datasets)]
            self._task_embeddings = unit[len(self._datasets) :]

    @property
    def mode(self) -> SearchMode:
        """``hybrid`` when an embedder was given, else ``lexical``."""
        return "lexical" if self._embedder is None else "hybrid"

    def __len__(self) -> int:
        """The number of indexed datasets."""
        return len(self._datasets)

    @property
    def tasks(self) -> list[BenchmarkTask]:
        """The tasks, in task order (:func:`task_key`)."""
        return list(self._tasks)

    def lexical_scores(self, tokens: Sequence[str]) -> FloatMatrix:
        """``lex(Q, d)`` of every dataset, in browse order (see the module docstring).

        Args:
            tokens: The query tokens ``Q`` (:func:`query_tokens`), at least one.

        Returns:
            A vector of length ``len(self)`` with values in [0, 1].
        """
        return self._board_postings.scores(tokens)

    def task_lexical_scores(self, tokens: Sequence[str]) -> FloatMatrix:
        """``lex(Q, d)`` of every task card, in task order.

        Args:
            tokens: The query tokens ``Q`` (:func:`query_tokens`), at least one.

        Returns:
            A vector of length ``len(self.tasks)`` with values in [0, 1].
        """
        return self._task_postings.scores(tokens)

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
        cosine: FloatMatrix = self._embeddings @ _query_vector(self._embedder, query)
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
            ranked = _ranked(
                candidates,
                self.lexical_scores(tokens),
                None if self._embeddings is None else self.cosine_scores(query),
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

    def search_tasks(
        self,
        query: str,
        *,
        organism: str | None = None,
        key: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> TaskPage:
        """The tasks matching ``query`` and the filters, best first, one page.

        Args:
            query: The typed text; an empty one (no tokens) lists every task that
                passes the filters in task order with score 0.
            organism: Keep only tasks with exactly this organism.
            key: Keep only the task with exactly this key (its task group, or the
                slug of a board without one), which finds the task of a board.
            limit: Page size.
            offset: Results skipped before the page.

        Returns:
            The page, with ``total`` counting every matching task.
        """
        candidates = [
            position
            for position, task in enumerate(self._tasks)
            if (organism is None or task.organism == organism)
            and (key is None or task.key == key)
        ]
        tokens = query_tokens(query)
        if not tokens or not candidates:
            ranked = [(position, 0.0) for position in candidates]
        else:
            cosine: FloatMatrix | None = None
            if self._embedder is not None and self._task_embeddings is not None:
                cosine = self._task_embeddings @ _query_vector(self._embedder, query)
            ranked = _ranked(candidates, self.task_lexical_scores(tokens), cosine)
        return TaskPage(
            mode=self.mode,
            total=len(ranked),
            limit=limit,
            offset=offset,
            results=[
                _task_hit(self._tasks[position], score)
                for position, score in ranked[offset : offset + limit]
            ],
        )

    def facets(self) -> DatasetFacets:
        """Board and task counts per organism and, within each, per task group.

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
        tasks_by_organism: dict[str | None, int] = defaultdict(int)
        for task in self._tasks:
            tasks_by_organism[task.organism] += 1
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
                    n_tasks=tasks_by_organism[organism],
                    task_groups=task_groups,
                    n_without_task_group=len(members) - grouped,
                )
            )
        return DatasetFacets(
            total=len(self._datasets),
            n_tasks=len(self._tasks),
            organisms=organisms,
            n_without_organism=n_without_organism,
        )


def _task_hit(task: BenchmarkTask, score: float) -> TaskHit:
    """The answer row of ``task``: its first :data:`MAX_TASK_BOARDS` boards."""
    return TaskHit(
        key=task.key,
        task_group=task.task_group,
        title=task.title,
        organism=task.organism,
        n_boards=len(task.boards),
        score=score,
        boards=[
            TaskBoard(
                slug=board.slug,
                split_name=board.split_name,
                title=board.title,
                n_train=board.n_train,
                n_val=board.n_val,
                n_test=board.n_test,
            )
            for board in task.boards[:MAX_TASK_BOARDS]
        ],
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
