# tests/torchcell/benchmark/test_search.py
# [[tests.torchcell.benchmark.test_search]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/benchmark/test_search.py
"""``torchcell.benchmark.search`` on five hand-built cards, every score worked by hand.

The cards: two E. coli cell-shape boards of one task (``random_s0`` and
``seqid30_s0``), one E. coli genetic-interaction board, one yeast board with no task
group, and one board with no organism. The lexical score of the module docstring is
``sum over query tokens of (field weight * 1 or 0.8) / (3 * number of tokens)``, so
``morph`` (a prefix of ``morphology`` in the slug, weight 3) scores 2.4 / 3 = 0.8 and
``deletions`` (whole word, description only, weight 1) scores 1 / 3. The hybrid tests
use :class:`FakeEmbedder`, whose cosines are 1 or 0 by construction, so
``0.5 * cos + 0.5 * lex`` is exact too. The task tests group the same five cards into
four tasks (the two shape boards are one) and check board order, the 50-board cap and
the split-free task card.
"""

from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from tests.torchcell.benchmark._fake_embedder import FakeEmbedder
from torchcell.benchmark.bundle import BenchmarkDatasetPublic
from torchcell.benchmark.search import (
    MAX_TASK_BOARDS,
    DatasetIndex,
    Model2VecEmbedder,
    board_order,
    card_text,
    card_tokens,
    embedder_from_path,
    group_title,
    query_tokens,
    split_parts,
    task_card_text,
    task_description,
    tasks_of,
)

ECOLI = "Escherichia coli"
YEAST = "Saccharomyces cerevisiae"
SHAPE_TASK = "ecoli-morphology-campos2018"
GI_TASK = "ecoli-gene-interaction-babu2014"


def _card(
    slug: str,
    title: str,
    *,
    description: str,
    loader_class: str,
    citation_key: str,
    targets: list[str],
    organism: str | None,
    task_group: str | None,
    split_name: str | None,
    task: str = "regression",
    primary_metric: str = "pearson",
) -> BenchmarkDatasetPublic:
    return BenchmarkDatasetPublic.model_validate(
        {
            "slug": slug,
            "title": title,
            "description": description,
            "loader_class": loader_class,
            "citation_key": citation_key,
            "version": "1",
            "task": task,
            "targets": targets,
            "n_train": 8,
            "n_val": 1,
            "n_test": 1,
            "primary_metric": primary_metric,
            "splits_sha256": "0" * 64,
            "template_sha256": "1" * 64,
            "organism": organism,
            "task_group": task_group,
            "split_name": split_name,
        }
    )


SHAPE_RANDOM = _card(
    "ecoli-morphology-campos2018-random-s0",
    "E. coli cell shape (Campos 2018), random split",
    description="Cell shape of deletions.",
    loader_class="MorphologyCampos2018Dataset",
    citation_key="campos2018",
    targets=["mean_L", "mean_W"],
    organism=ECOLI,
    task_group=SHAPE_TASK,
    split_name="random_s0",
)
SHAPE_SEQID = _card(
    "ecoli-morphology-campos2018-seqid30-s0",
    "E. coli cell shape (Campos 2018), sequence identity 30 % split",
    description="Cell shape of deletions.",
    loader_class="MorphologyCampos2018Dataset",
    citation_key="campos2018",
    targets=["mean_L", "mean_W"],
    organism=ECOLI,
    task_group=SHAPE_TASK,
    split_name="seqid30_s0",
)
GI_RANDOM = _card(
    "ecoli-gene-interaction-babu2014-random-s0",
    "E. coli genetic interactions (Babu 2014), random split",
    description="Pairwise interactions.",
    loader_class="GeneInteractionBabu2014Dataset",
    citation_key="babuGlobal2014",
    targets=["gene_interaction"],
    organism=ECOLI,
    task_group=GI_TASK,
    split_name="random_s0",
)
ESSENTIAL = _card(
    "gene-essentiality-sgd",
    "Gene essentiality (SGD)",
    description="Predict whether a null mutant is inviable.",
    loader_class="GeneEssentialitySgdDataset",
    citation_key="cherrySGD1998",
    targets=["is_essential"],
    organism=YEAST,
    task_group=None,
    split_name=None,
    task="binary",
    primary_metric="auroc",
)
TOY = _card(
    "toy-fitness",
    "Toy fitness",
    description="Eight records with one target, for tests.",
    loader_class="ToyFitnessDataset",
    citation_key="toy2026",
    targets=["fitness"],
    organism=None,
    task_group=None,
    split_name=None,
)
ALL = [TOY, ESSENTIAL, SHAPE_SEQID, GI_RANDOM, SHAPE_RANDOM]
BROWSE_ORDER = [
    GI_RANDOM.slug,
    SHAPE_RANDOM.slug,
    SHAPE_SEQID.slug,
    ESSENTIAL.slug,
    TOY.slug,
]


def _hits(
    index: DatasetIndex, query: str, *, organism: str | None = None
) -> list[tuple[str, float]]:
    """``(slug, score rounded to 6 places)`` of the first page of ``query``."""
    page = index.search(query, organism=organism)
    return [(hit.dataset.slug, round(hit.score, 6)) for hit in page.results]


@pytest.fixture
def lexical() -> DatasetIndex:
    return DatasetIndex(ALL)


# ----------------------------------------------------------------------- tokens


def test_card_tokens_split_camel_case_and_mixed_runs() -> None:
    assert card_tokens("MorphologyCampos2018Dataset") == [
        "morphology",
        "campos2018",
        "campos",
        "2018",
        "dataset",
    ]
    assert card_tokens("E. coli (Babu 2014), seqid30_s0") == [
        "e",
        "coli",
        "babu",
        "2014",
        "seqid30",
        "seqid",
        "30",
        "s0",
        "s",
        "0",
    ]


def test_query_tokens_are_words_deduplicated_without_run_expansion() -> None:
    assert query_tokens("Babu babu, SEQID30!") == ["babu", "seqid30"]
    assert query_tokens("  %% -- ") == []


def test_card_text_joins_the_set_fields_in_weight_order() -> None:
    assert card_text(ESSENTIAL) == (
        "Gene essentiality (SGD)\n"
        "gene essentiality sgd\n"
        "Saccharomyces cerevisiae\n"
        "is_essential\n"
        "Predict whether a null mutant is inviable.\n"
        "GeneEssentialitySgdDataset\n"
        "cherrySGD1998"
    )


# ---------------------------------------------------------------------- lexical


def test_prefix_match_counts_four_fifths_of_a_whole_word(lexical: DatasetIndex) -> None:
    # "morph" starts "morphology" (slug, weight 3): 3 * 0.8 / 3.
    assert _hits(lexical, "morph") == [
        (SHAPE_RANDOM.slug, 0.8),
        (SHAPE_SEQID.slug, 0.8),
    ]
    assert _hits(lexical, "MORPH") == _hits(lexical, "morph")
    # "babu" is a whole word of the title (weight 3) and a run of slug "babu2014".
    assert _hits(lexical, "babu") == [(GI_RANDOM.slug, 1.0)]
    # "interact" starts "interactions" (title) and "interaction" (slug, targets).
    assert _hits(lexical, "interact") == [(GI_RANDOM.slug, 0.8)]


def test_field_weights_rank_title_over_description(lexical: DatasetIndex) -> None:
    assert _hits(lexical, "shape") == [
        (SHAPE_RANDOM.slug, 1.0),
        (SHAPE_SEQID.slug, 1.0),
    ]
    # "deletions" is in the description only (weight 1).
    assert _hits(lexical, "deletions") == [
        (SHAPE_RANDOM.slug, round(1 / 3, 6)),
        (SHAPE_SEQID.slug, round(1 / 3, 6)),
    ]
    # "escherichia" is the organism (weight 3) of all three E. coli boards.
    assert _hits(lexical, "escherichia")[0] == (GI_RANDOM.slug, 1.0)
    # "pairwise" is in the description only: weight 1, so 1/3.
    assert _hits(lexical, "pairwise") == [(GI_RANDOM.slug, round(1 / 3, 6))]


def test_score_averages_over_query_tokens(lexical: DatasetIndex) -> None:
    # babu (3) + random (3) on the GI board; random alone (3 of 6) on SHAPE_RANDOM.
    assert _hits(lexical, "babu random") == [
        (GI_RANDOM.slug, 1.0),
        (SHAPE_RANDOM.slug, 0.5),
    ]
    # "30" is a digit run of slug "seqid30" and a title word.
    assert _hits(lexical, "30") == [(SHAPE_SEQID.slug, 1.0)]


def test_scores_vector_is_the_documented_formula(lexical: DatasetIndex) -> None:
    order = {slug: position for position, slug in enumerate(BROWSE_ORDER)}
    scores = lexical.lexical_scores(["shape", "pairwise"])
    expected = np.zeros(5)
    expected[order[SHAPE_RANDOM.slug]] = 3 / 6
    expected[order[SHAPE_SEQID.slug]] = 3 / 6
    expected[order[GI_RANDOM.slug]] = 1 / 6
    np.testing.assert_allclose(scores, expected)


def test_no_match_returns_nothing_in_lexical_mode(lexical: DatasetIndex) -> None:
    page = lexical.search("zebrafish")
    assert (page.mode, page.total, page.results) == ("lexical", 0, [])


# ----------------------------------------------------------------- empty query


def test_empty_query_lists_everything_in_browse_order(lexical: DatasetIndex) -> None:
    for query in ("", "   ", "%%"):
        page = lexical.search(query, limit=20)
        assert page.total == 5
        assert [(h.dataset.slug, h.score) for h in page.results] == [
            (slug, 0.0) for slug in BROWSE_ORDER
        ]


def test_pagination_slices_the_ranked_list(lexical: DatasetIndex) -> None:
    pages = [lexical.search("", limit=2, offset=offset) for offset in (0, 2, 4, 6)]
    assert [[h.dataset.slug for h in p.results] for p in pages] == [
        BROWSE_ORDER[0:2],
        BROWSE_ORDER[2:4],
        BROWSE_ORDER[4:5],
        [],
    ]
    assert {(p.total, p.limit) for p in pages} == {(5, 2)}
    assert [p.offset for p in pages] == [0, 2, 4, 6]


def test_filters_keep_exact_organism_and_task_group(lexical: DatasetIndex) -> None:
    assert [h.dataset.slug for h in lexical.search("", organism=ECOLI).results] == (
        BROWSE_ORDER[:3]
    )
    by_task = lexical.search("", organism=ECOLI, task_group=SHAPE_TASK)
    assert [h.dataset.slug for h in by_task.results] == [
        SHAPE_RANDOM.slug,
        SHAPE_SEQID.slug,
    ]
    # A tie (both 1.0 on "random") keeps the browse order.
    assert _hits(lexical, "random", organism=ECOLI) == [
        (GI_RANDOM.slug, 1.0),
        (SHAPE_RANDOM.slug, 1.0),
    ]
    assert lexical.search("", organism="Escherichia").total == 0
    assert lexical.search("random", organism=YEAST).total == 0


# ----------------------------------------------------------------------- hybrid


def test_hybrid_adds_half_cosine_to_half_lexical() -> None:
    embedder = FakeEmbedder()
    index = DatasetIndex(ALL, embedder)
    assert index.mode == "hybrid"
    # Cards embedded once: the 5 boards in browse order, then the 4 tasks.
    assert len(embedder.calls) == 1
    assert len(embedder.calls[0]) == 9
    # "shape random": lex 1.0 / 0.5 / 0.5; cos 1 / 1 / 0.
    assert _hits(index, "shape random") == [
        (SHAPE_RANDOM.slug, 1.0),
        (SHAPE_SEQID.slug, 0.75),
        (GI_RANDOM.slug, 0.25),
    ]
    assert embedder.calls[-1] == ["shape random"]
    assert len(embedder.calls) == 2


def test_hybrid_finds_a_card_by_meaning_alone() -> None:
    index = DatasetIndex(ALL, FakeEmbedder())
    # No card holds "epistasis"; its vector is the interaction axis: cos 1, lex 0.
    page = index.search("epistasis")
    assert (page.mode, page.total) == ("hybrid", 1)
    assert _hits(index, "epistasis") == [(GI_RANDOM.slug, 0.5)]
    # Cosine 0 and no word match: dropped.
    assert index.search("zebrafish").total == 0


def test_hybrid_cosine_scores_and_zero_vectors() -> None:
    index = DatasetIndex(ALL, FakeEmbedder())
    order = {slug: position for position, slug in enumerate(BROWSE_ORDER)}
    cosine = index.cosine_scores("inviable")
    expected = np.zeros(5)
    expected[order[ESSENTIAL.slug]] = 1.0
    np.testing.assert_allclose(cosine, expected)
    # The toy card has none of the keywords: a zero vector scores cosine 0.
    np.testing.assert_allclose(index.cosine_scores("shape")[order[TOY.slug]], 0.0)


def test_cosine_scores_need_the_hybrid_mode(lexical: DatasetIndex) -> None:
    with pytest.raises(ValueError, match="hybrid mode"):
        lexical.cosine_scores("shape")


def test_embedder_of_the_wrong_shape_is_refused() -> None:
    class Short:
        def embed(self, texts: list[str]) -> npt.NDArray[np.float64]:
            return np.ones((len(texts) - 1, 3))

    with pytest.raises(ValueError, match=r"shape \(8, 3\) for 9 cards"):
        DatasetIndex(ALL, Short())


def test_empty_index_answers_empty_pages() -> None:
    index = DatasetIndex([], FakeEmbedder())
    assert (index.mode, len(index)) == ("hybrid", 0)
    assert index.search("shape").total == 0
    assert index.facets().model_dump() == {
        "total": 0,
        "n_tasks": 0,
        "organisms": [],
        "n_without_organism": 0,
    }
    assert index.search_tasks("shape").model_dump() == {
        "mode": "hybrid",
        "total": 0,
        "limit": 20,
        "offset": 0,
        "results": [],
    }


# -------------------------------------------------------------------- the model


def test_model_path_that_is_not_a_directory_fails_loudly(tmp_path: Path) -> None:
    missing = tmp_path / "potion-base-8M"
    with pytest.raises(FileNotFoundError, match="TC_BENCH_SEARCH_MODEL="):
        Model2VecEmbedder(missing)
    with pytest.raises(FileNotFoundError, match="not a directory"):
        embedder_from_path(missing)
    assert embedder_from_path(None) is None


# ----------------------------------------------------------------------- facets


def test_facets_count_organisms_and_task_groups(lexical: DatasetIndex) -> None:
    assert lexical.facets().model_dump() == {
        "total": 5,
        "n_tasks": 4,
        "organisms": [
            {
                "organism": ECOLI,
                "count": 3,
                "n_tasks": 2,
                "task_groups": [
                    {
                        "task_group": SHAPE_TASK,
                        "title": "E. coli cell shape (Campos 2018)",
                        "count": 2,
                    },
                    {
                        "task_group": GI_TASK,
                        "title": "E. coli genetic interactions (Babu 2014), random split",
                        "count": 1,
                    },
                ],
                "n_without_task_group": 0,
            },
            {
                "organism": YEAST,
                "count": 1,
                "n_tasks": 1,
                "task_groups": [],
                "n_without_task_group": 1,
            },
        ],
        "n_without_organism": 1,
    }


def test_group_title_is_the_common_prefix_cut_to_whole_words() -> None:
    assert group_title(["Only board"], "t") == "Only board"
    assert group_title(["Same", "Same"], "t") == "Same"
    assert group_title(["Foo bar", "Foo baz"], "t") == "Foo"
    assert group_title(["Growth (A 2018), random", "Growth (A 2018), seqid"], "t") == (
        "Growth (A 2018)"
    )
    assert group_title(["Foo", "Foo bar"], "t") == "Foo"
    assert group_title(["Alpha", "Beta"], "the-task") == "the-task"
    # "_" is part of a word, so a split name in a title is cut whole.
    assert group_title(["Growth, random_s1", "Growth, random_s10"], "t") == "Growth"


# ------------------------------------------------------------------------ tasks

# Task order: organism (missing last), then display title, then key. The shape task's
# title "E. coli cell shape (Campos 2018)" sorts before "E. coli genetic ...".
TASK_ORDER = [SHAPE_TASK, GI_TASK, ESSENTIAL.slug, TOY.slug]


def _task_hits(
    index: DatasetIndex, query: str, *, organism: str | None = None
) -> list[tuple[str, float]]:
    """``(key, score rounded to 6 places)`` of the first page of a task search."""
    page = index.search_tasks(query, organism=organism)
    return [(hit.key, round(hit.score, 6)) for hit in page.results]


def _split_board(
    slug: str, split_name: str | None, task_group: str
) -> BenchmarkDatasetPublic:
    return _card(
        slug,
        f"Growth, {split_name}",
        description=f"Growth of deletions. Split {split_name} of the task.",
        loader_class="GrowthDataset",
        citation_key="campos2018",
        targets=["growth"],
        organism=ECOLI,
        task_group=task_group,
        split_name=split_name,
    )


def test_a_query_matching_every_split_answers_the_task_once(
    lexical: DatasetIndex,
) -> None:
    # Two boards match "shape" (title, weight 3); one task does.
    assert _hits(lexical, "shape") == [
        (SHAPE_RANDOM.slug, 1.0),
        (SHAPE_SEQID.slug, 1.0),
    ]
    page = lexical.search_tasks("shape")
    assert (page.mode, page.total, page.limit, page.offset) == ("lexical", 1, 20, 0)
    assert page.results[0].model_dump() == {
        "key": SHAPE_TASK,
        "task_group": SHAPE_TASK,
        "title": "E. coli cell shape (Campos 2018)",
        "organism": ECOLI,
        "n_boards": 2,
        "score": 1.0,
        "boards": [
            {
                "slug": SHAPE_RANDOM.slug,
                "split_name": "random_s0",
                "title": SHAPE_RANDOM.title,
                "n_train": 8,
                "n_val": 1,
                "n_test": 1,
            },
            {
                "slug": SHAPE_SEQID.slug,
                "split_name": "seqid30_s0",
                "title": SHAPE_SEQID.title,
                "n_train": 8,
                "n_val": 1,
                "n_test": 1,
            },
        ],
    }
    # "morph" starts the task-group word "morphology" (weight 3): 0.8.
    assert _task_hits(lexical, "morph") == [(SHAPE_TASK, 0.8)]
    # Split names are not on a task card: "seqid" and "30" find boards, not tasks.
    assert _hits(lexical, "seqid") == [(SHAPE_SEQID.slug, 1.0)]
    assert _task_hits(lexical, "seqid") == []
    assert _task_hits(lexical, "30") == []


def test_task_scores_are_the_documented_formula(lexical: DatasetIndex) -> None:
    # shape (title, 3) + pairwise (absent) on the shape task: 3 / 6. pairwise is in
    # the GI description only (weight 1): 1 / 6.
    assert _task_hits(lexical, "shape pairwise") == [
        (SHAPE_TASK, 0.5),
        (GI_TASK, round(1 / 6, 6)),
    ]
    # A tie (coli, title weight 3, on both E. coli tasks) keeps the task order.
    assert _task_hits(lexical, "coli") == [(SHAPE_TASK, 1.0), (GI_TASK, 1.0)]
    # "deletions" survives in the shape task's description (both boards share it).
    assert _task_hits(lexical, "deletions") == [(SHAPE_TASK, round(1 / 3, 6))]
    np.testing.assert_allclose(
        lexical.task_lexical_scores(["shape", "pairwise"]), [3 / 6, 1 / 6, 0.0, 0.0]
    )
    assert lexical.search_tasks("zebrafish").total == 0


def test_empty_task_query_lists_tasks_in_order_and_pages(lexical: DatasetIndex) -> None:
    for query in ("", "  ", "%%"):
        page = lexical.search_tasks(query)
        assert page.total == 4
        assert [(h.key, h.score) for h in page.results] == [
            (key, 0.0) for key in TASK_ORDER
        ]
    pages = [lexical.search_tasks("", limit=3, offset=offset) for offset in (0, 3, 6)]
    assert [[h.key for h in p.results] for p in pages] == [
        TASK_ORDER[0:3],
        TASK_ORDER[3:4],
        [],
    ]
    assert {(p.total, p.limit) for p in pages} == {(4, 3)}
    assert [t.key for t in lexical.tasks] == TASK_ORDER


def test_task_search_filters_by_organism(lexical: DatasetIndex) -> None:
    assert _task_hits(lexical, "", organism=ECOLI) == [
        (SHAPE_TASK, 0.0),
        (GI_TASK, 0.0),
    ]
    assert _task_hits(lexical, "", organism=YEAST) == [(ESSENTIAL.slug, 0.0)]
    assert lexical.search_tasks("shape", organism=YEAST).total == 0
    assert lexical.search_tasks("", organism="Escherichia").total == 0
    # The key filter finds the task of a board: its task group, or its own slug.
    assert _task_hits_key(lexical, GI_TASK) == [(GI_TASK, 0.0)]
    assert _task_hits_key(lexical, ESSENTIAL.slug) == [(ESSENTIAL.slug, 0.0)]
    assert _task_hits_key(lexical, SHAPE_RANDOM.slug) == []
    assert lexical.search_tasks("shape", key=GI_TASK).total == 0


def _task_hits_key(index: DatasetIndex, key: str) -> list[tuple[str, float]]:
    return [(h.key, h.score) for h in index.search_tasks("", key=key).results]


def test_a_board_without_a_task_group_is_its_own_task(lexical: DatasetIndex) -> None:
    page = lexical.search_tasks("essentiality")
    assert [hit.model_dump() for hit in page.results] == [
        {
            "key": ESSENTIAL.slug,
            "task_group": None,
            "title": "Gene essentiality (SGD)",
            "organism": YEAST,
            "n_boards": 1,
            "score": 1.0,
            "boards": [
                {
                    "slug": ESSENTIAL.slug,
                    "split_name": None,
                    "title": "Gene essentiality (SGD)",
                    "n_train": 8,
                    "n_val": 1,
                    "n_test": 1,
                }
            ],
        }
    ]
    # The toy board has neither organism nor task group: a task with no organism.
    toy = lexical.search_tasks("toy").results
    assert [(h.key, h.task_group, h.organism, h.n_boards) for h in toy] == [
        (TOY.slug, None, None, 1)
    ]


def test_task_card_text_has_no_split_and_no_board_slug() -> None:
    shape = tasks_of(ALL)[0]
    assert task_card_text(shape) == (
        "E. coli cell shape (Campos 2018)\n"
        "ecoli morphology campos2018\n"
        "Escherichia coli\n"
        "mean_L mean_W\n"
        "Cell shape of deletions.\n"
        "MorphologyCampos2018Dataset\n"
        "campos2018"
    )


def test_task_description_drops_the_split_specific_sentences() -> None:
    def board(split_name: str, description: str) -> BenchmarkDatasetPublic:
        return SHAPE_RANDOM.model_copy(
            update={"split_name": split_name, "description": description}
        )

    random_board = board(
        "random_s0",
        "E. coli growth rate. One target. Split random_s0 of the task t: random split.",
    )
    seqid_board = board(
        "seqid30_s0",
        "E. coli growth rate. One target. Split seqid30_s0 of the task t: "
        "sequence identity 30 % split.",
    )
    # "E." and "coli growth rate." are two sentences, both shared, rejoined by a space.
    assert task_description([random_board, seqid_board]) == (
        "E. coli growth rate. One target."
    )
    # One board: the sentence naming its split is dropped.
    assert task_description([random_board]) == "E. coli growth rate. One target."
    # A sentence only some boards have is dropped, even without a split name.
    other = board("position_s0", "E. coli growth rate. Ordered by locus.")
    assert task_description([other, random_board]) == "E. coli growth rate."
    assert task_description([board("random_s0", "Split random_s0.")]) == ""


def test_board_order_is_split_family_then_seed() -> None:
    names = [
        "position_s0",
        "seqid50_s0",
        "random_s1",
        "weird_s0",
        "seqid30_gene_holdout_s0",
        "random_s0",
        "seqid30_s0",
        "gene_holdout_s0",
        "aaa_s0",
        "seqid5_gene_holdout_s0",
        "random_s10",
        "random_s2",
    ]
    boards = [_split_board(f"g-{i}", name, "g") for i, name in enumerate(names)]
    ordered = board_order(boards)
    assert [b.split_name for b in ordered] == [
        "random_s0",
        "random_s1",
        "random_s2",
        "random_s10",
        "seqid30_s0",
        "seqid50_s0",
        "gene_holdout_s0",
        "seqid5_gene_holdout_s0",
        "seqid30_gene_holdout_s0",
        "position_s0",
        # Unknown families in the order the boards were loaded.
        "weird_s0",
        "aaa_s0",
    ]
    assert split_parts("seqid30_gene_holdout_s12") == ("seqid30_gene_holdout", 12)
    assert split_parts("custom") == ("custom", -1)
    assert split_parts(None) == (None, -1)


def test_a_task_lists_at_most_fifty_boards_and_search_pages_the_rest() -> None:
    # 60 seeds of one family, loaded out of order.
    seeds = [(7 * i) % 60 for i in range(60)]
    boards = [
        _split_board(f"big-s{seed:02d}", f"random_s{seed}", "big") for seed in seeds
    ]
    index = DatasetIndex(boards)
    hit = index.search_tasks("growth").results[0]
    assert (hit.key, hit.title, hit.n_boards) == ("big", "Growth", 60)
    assert len(hit.boards) == MAX_TASK_BOARDS == 50
    assert [b.split_name for b in hit.boards] == [f"random_s{s}" for s in range(50)]
    # The board search with the task group continues where the task's list stops.
    rest = index.search("", task_group="big", limit=50, offset=50)
    assert rest.total == 60
    assert [h.dataset.split_name for h in rest.results] == [
        f"random_s{s}" for s in range(50, 60)
    ]


def test_tasks_refuse_mixed_organisms_and_a_slug_that_names_a_group() -> None:
    mixed = SHAPE_SEQID.model_copy(update={"organism": YEAST})
    with pytest.raises(ValueError, match="boards of different organisms"):
        DatasetIndex([SHAPE_RANDOM, mixed])
    clash = TOY.model_copy(update={"slug": SHAPE_TASK})
    with pytest.raises(ValueError, match="has no task group but its slug names one"):
        DatasetIndex([SHAPE_RANDOM, clash])


def test_hybrid_task_search_finds_a_task_by_meaning() -> None:
    embedder = FakeEmbedder()
    index = DatasetIndex(ALL, embedder)
    # The task cards follow the board cards in the one startup call.
    assert embedder.calls[0][5:] == [task_card_text(t) for t in index.tasks]
    # No task card holds "epistasis"; its vector is the interaction axis: cos 1, lex 0.
    assert _task_hits(index, "epistasis") == [(GI_TASK, 0.5)]
    assert embedder.calls[-1] == ["epistasis"]
    # "shape": lex 1 and cos 1 on the shape task only.
    assert _task_hits(index, "shape") == [(SHAPE_TASK, 1.0)]
    assert index.search_tasks("epistasis").mode == "hybrid"
